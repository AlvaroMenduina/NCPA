"""

                          -||  NYQUIST ERRORS  ||-

What is the effect of errors in the Nyquist-Shannon sampling criterion?

What happens to the performance when you show the model
PSF images that have a slightly different spaxel scale??

Author: Alvaro Menduina
Date: Feb 2020
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import psf
import utils
import calibration

utils.print_title(message='\nN C P A', font=None, random_font=False)

# PSF bits
N_PIX = 256                         # pixels for the Fourier arrays
pix = 30                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength
SPAX = 4.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

N_actuators = 20                    # Number of actuators in [-1, 1] line
alpha_pc = 10                       # Height [percent] at the neighbour actuator (Gaussian Model)
N_WAVES = 2                         # 2 wavelengths: 1 Nominal, 1 with Nyquist error

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
coef_strength = 0.30                # Strength of the actuator coefficients
diversity = 0.55                    # Strength of extra diversity commands
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filers = [64, 32, 16, 8]      # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
SNR = 500                           # SNR for the Readout Noise
N_loops, epochs_loop = 5, 5         # How many times to loop over the training
readout_copies = 2                  # How many copies with Readout Noise to use
N_iter = 3                          # How many iterations to run the calibration (testing)

directory = os.path.join(os.getcwd(), 'Nyquist')

import importlib
importlib.reload(calibration)

if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    ### ============================================================================================================ ###
    #                            Train the Calibration Model with 4 mas PSF images
    ### ============================================================================================================ ###

    # Find the Wavelength at which you have 4 mas, if you have 4 * (1 + eps) at 1.5
    SPAX_ERR = 0.05         # Percentage of error [10%]
    WAVE_BAD = WAVE * (1 + SPAX_ERR)
    # Double-check that is the correct wavelength
    utils.check_spaxel_scale(rho_aper=RHO_APER*(1 + SPAX_ERR), wavelength=WAVE)

    centers_multiwave = psf.actuator_centres_multiwave(N_actuators=N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                       N_waves=N_WAVES, wave0=WAVE, waveN=WAVE_BAD, wave_ref=WAVE)
    N_act = len(centers_multiwave[0][0])

    rbf_matrices = psf.actuator_matrix_multiwave(centres=centers_multiwave, alpha_pc=alpha_pc, rho_aper=RHO_APER,
                                                 rho_obsc=RHO_OBSC, N_waves=N_WAVES,  wave0=WAVE, waveN=WAVE_BAD,
                                                 wave_ref=WAVE, N_PIX=N_PIX)

    PSF_nom = psf.PointSpreadFunction(rbf_matrices[0], N_pix=N_PIX, crop_pix=pix, diversity_coef=np.zeros(N_act))


    # Create a Zernike model so we can mimic the defocus
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=5, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.0)
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))
    # Use Least Squares to find the actuator commands that mimic a Zernike defocus
    zernike_fit = calibration.Zernike_fit(PSF_zernike, PSF_nom, wavelength=WAVE, rho_aper=RHO_APER)
    defocus_zernike = np.zeros((1, zernike_matrix.shape[-1]))
    defocus_zernike[0, 1] = 1.0
    defocus_actuators = zernike_fit.fit_zernike_wave_to_actuators(defocus_zernike, plot=True, cmap='bwr')[:, 0]

    # Update the Diversity Map on the actuator model so that it matches Defocus
    diversity_defocus = diversity * defocus_actuators
    PSF_nom.define_diversity(diversity_defocus)

    ### ============================================================================================================ ###
    #                                   Generate the training sets
    ### ============================================================================================================ ###

    # Test the Saving
    _train_PSF, _train_coef, _test_PSF, _test_coef = calibration.generate_dataset_and_save(PSF_nom, N_train, N_test,
                                                                                       coef_strength, rescale,
                                                                                       dir_to_save=directory)
    # And Loading
    train_PSF, train_coef, test_PSF, test_coef = calibration.load_datasets(dir_to_load=directory,
                                                                           file_names=["train_PSF", "train_coef",
                                                                                       "test_PSF", "test_coef"])
    # Generate a training set for that nominal defocus
    # train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_nom, N_train, N_test,
    #                                                                           coef_strength, rescale)

    utils.plot_images(train_PSF)
    plt.show()

    # Train the Calibration Model on images with the nominal defocus
    calib = calibration.Calibration(PSF_model=PSF_nom)
    calib.create_cnn_model(layer_filers, kernel_size, name='CALIBR', activation='relu')
    losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    ### Sometimes the train fails (no apparent reason) probably because of random weight initialization??
    # If that happens, simply copy and paste the model definition and training bits, and try again

    RMS_evolution, residual = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                         readout_noise=True, RMS_readout=1./SNR)

    calib.plot_RMS_evolution(RMS_evolution)
    plt.show()

    ### Up until here it's been the "classic" approach of building a model and training it

    # ================================================================================================================ #
    #           Impact of Nyquist Errors
    # ================================================================================================================ #

    # Generate a set of PSF images with the wrong sampling
    # Use the fake extra wavelength for which the sampling is off
    PSF_error = psf.PointSpreadFunction(rbf_matrices[1], N_pix=N_PIX, crop_pix=pix, diversity_coef=diversity_defocus)

    test_PSF_error, test_coef_error, _PSF, _coef = calibration.generate_dataset(PSF_error, 1000, 1,
                                                                                coef_strength, rescale)

    ###  WATCHOUT
    # Calibrating here is quite tricky! Calib has a self.PSF_model the nominal model (not the one with Nyquist errors)
    # that means that after the first iteration, the PSF images will revert to having the proper modelling

    # We MUST temporarily override calib.PSF_model tem
    calib.PSF_model = PSF_error

    # In addition we need to be careful with the wavefront definition in [rad]
    # You need to rescale the coefficients

    test_PSF_error = calib.noise_effects.add_readout_noise(test_PSF_error, RMS_READ=1. / SNR)
    RMS_evolution_errors = []
    for i in range(N_iter):

        predicted_coef = calib.cnn_model.predict(test_PSF_error)        # These are the coefficients in the "nominal" model
        predicted_coef_nm = predicted_coef * WAVE
        residual_nm = test_coef_error * WAVE_BAD - predicted_coef_nm
        residual = residual_nm / WAVE
        rms_before, rms_after = calib.calculate_RMS(predicted_coef, residual, wavelength=WAVE)
        rms_pair = [rms_before, rms_after]
        RMS_evolution.append(rms_pair)

        test_PSF_error = calib.update_PSF(residual)
        test_PSF_error = calib.noise_effects.add_readout_noise(test_PSF_error, RMS_READ=1. / SNR)

    calib.PSF_model = PSF_nom

    calib.plot_RMS_evolution(RMS_evolution_errors)

    title = r'Spaxel Scale Error %.1f percent || Wavefronts [$\lambda$]' % (100*SPAX_ERR)
    utils.show_wavefronts_grid(PSF_nom, coefs=residual_errors, rho_aper=RHO_APER, title=title)

    plt.show()

    # ================================================================================================================ #
    #          See what happens as a function of Percentage Error
    # ================================================================================================================ #

    SPAX_ERR = 0.05         # Percentage of error [10%]
    WAVE_MIN = (1 - SPAX_ERR)
    WAVE_MAX = WAVE * (1 + SPAX_ERR)
    N_WAVES = 11
    # Double-check that is the correct wavelength
    utils.check_spaxel_scale(rho_aper=RHO_APER*(1 - SPAX_ERR), wavelength=WAVE)
    utils.check_spaxel_scale(rho_aper=RHO_APER*(1 + SPAX_ERR), wavelength=WAVE)

    spax_errs = np.linspace(-SPAX_ERR, SPAX_ERR, N_WAVES, endpoint=True)

    centers_multiwave = psf.actuator_centres_multiwave(N_actuators=N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                       N_waves=N_WAVES, wave0=WAVE_MIN, waveN=WAVE_MAX, wave_ref=WAVE)

    rbf_matrices = psf.actuator_matrix_multiwave(centres=centers_multiwave, alpha_pc=alpha_pc, rho_aper=RHO_APER,
                                                 rho_obsc=RHO_OBSC, N_waves=N_WAVES,  wave0=WAVE, waveN=WAVE_BAD,
                                                 wave_ref=WAVE, N_PIX=N_PIX)

    mus_nyq, std_nyq = [], []
    print("\nLooping over Spaxel Scale Errors")
    for i, spaxel_error in enumerate(spax_errs):

        print("\nSpaxel Error: %.1f percent" % (100 * spaxel_error))

        # Generate the PSF model with the wrong Spaxel Scale
        PSF_error = psf.PointSpreadFunction(rbf_matrices[i], N_pix=N_PIX, crop_pix=pix,
                                            diversity_coef=diversity_defocus)
        test_PSF_error, test_coef_error, _PSF, _coef = calibration.generate_dataset(PSF_error, 1000, 1,
                                                                                    coef_strength, rescale)

        calib.PSF_model = PSF_error
        RMS_evolution_errors, residual_errors = calib.calibrate_iterations(test_PSF_error, test_coef_error,
                                                                           wavelength=WAVE, N_iter=N_iter,
                                                                           readout_noise=True, RMS_readout=1. / SNR)
        calib.PSF_model = PSF_nom

        final_RMS = RMS_evolution_errors[-1][-1]
        avg, std = np.mean(final_RMS), np.std(final_RMS)
        mus_nyq.append(avg)
        std_nyq.append(std)
















