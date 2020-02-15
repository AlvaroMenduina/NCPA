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
import matplotlib.cm as cm

import psf
import utils
import calibration

utils.print_title(message='\nN C P A', font=None, random_font=False)

print("\n           -||  NYQUIST ERRORS  ||- ")
print("What is the effect of errors in the Nyquist-Shannon sampling criterion?")
print("What happens to the performance when you show the model")
print("PSF images that have a slightly different spaxel scale??\n")

# PSF bits
N_PIX = 256                         # pixels for the Fourier arrays
pix = 30                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength
SPAX = 4.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
print("Nominal Parameters | Spaxel Scale and Wavelength")
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

N_actuators = 20                    # Number of actuators in [-1, 1] line
alpha_pc = 10                       # Height [percent] at the neighbour actuator (Gaussian Model)
N_WAVES = 2                         # 2 wavelengths: 1 Nominal, 1 with Nyquist error

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
coef_strength = 0.30                # Strength of the actuator coefficients
diversity = 0.55                    # Strength of extra diversity commands
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filters = [64, 32, 16, 8]      # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
SNR = 750                           # SNR for the Readout Noise
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
    # utils.check_spaxel_scale(rho_aper=RHO_APER*(1 + SPAX_ERR), wavelength=WAVE)

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
    # plt.show()

    # Train the Calibration Model on images with the nominal defocus
    calib = calibration.Calibration(PSF_model=PSF_nom)
    calib.create_cnn_model(layer_filters, kernel_size, name='CALIBR', activation='relu')
    losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    ### Sometimes the train fails (no apparent reason) probably because of random weight initialization??
    # If that happens, simply copy and paste the model definition and training bits, and try again

    RMS_evolution, residual = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                         readout_noise=True, RMS_readout=1./SNR)

    calib.plot_RMS_evolution(RMS_evolution)
    # plt.show()

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

    RMS_evolution_errors, residual_errors = calib.calibrate_iterations(test_PSF_error, test_coef_error,
                                                                       wavelength=WAVE, N_iter=N_iter,
                                                                       readout_noise=True, RMS_readout=1. / SNR)
    calib.PSF_model = PSF_nom
    calib.plot_RMS_evolution(RMS_evolution_errors)
    # plt.show()

    # ================================================================================================================ #
    #          See what happens as a function of Percentage Error
    # ================================================================================================================ #

    SPAX_ERR = 0.05         # Percentage of error [10%]
    WAVE_MIN = WAVE * (1 - SPAX_ERR)
    WAVE_MAX = WAVE * (1 + SPAX_ERR)
    N_WAVES = 11
    # Double-check that is the correct wavelength
    utils.check_spaxel_scale(rho_aper=RHO_APER*(1 - SPAX_ERR), wavelength=WAVE)
    utils.check_spaxel_scale(rho_aper=RHO_APER*(1 + SPAX_ERR), wavelength=WAVE)

    spax_errs = np.linspace(-SPAX_ERR, SPAX_ERR, N_WAVES, endpoint=True)

    centers_multiwave = psf.actuator_centres_multiwave(N_actuators=N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                       N_waves=N_WAVES, wave0=WAVE_MIN, waveN=WAVE_MAX, wave_ref=WAVE)

    rbf_matrices = psf.actuator_matrix_multiwave(centres=centers_multiwave, alpha_pc=alpha_pc, rho_aper=RHO_APER,
                                                 rho_obsc=RHO_OBSC, N_waves=N_WAVES,  wave0=WAVE_MIN, waveN=WAVE_MAX,
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

    # Watch out because the Spaxel Scales go opposite to the fake Wavelengths
    spax_scales = SPAX * (1 - spax_errs)

    # Save the data as a function of SNR
    np.save(os.path.join(directory, 'spax_scales_SNR%d' % SNR), spax_scales)
    np.save(os.path.join(directory, 'mus_nyq_SNR%d' % SNR), mus_nyq)
    np.save(os.path.join(directory, 'std_nyq_SNR%d' % SNR), std_nyq)

    plt.figure()
    plt.errorbar(spax_scales, mus_nyq, std_nyq, fmt='o')
    plt.plot(spax_scales, mus_nyq, color='blue')
    plt.xlabel(r'Spaxel Scales [mas]')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.ylim([0, 50])
    plt.show()

    # ================================================================================================================ #
    #        Plot the results a function of SNR
    # ================================================================================================================ #
    # If you have ran the analysis multiple times for different values of SNR
    # there will be some files saved

    # Collect all the results as a function of SNR
    # (1) List all files in the directory
    list_files = utils.list_files_in_directory(directory)
    # (2) Find the files that contain SNR in the name
    snr_files = utils.find_substring(list_files, substring='SNR')
    # (3) Find which SNR we have used already
    list_SNR = []
    for name in snr_files:
        prefix, suffix = name.split('SNR')
        _snr, npy = suffix.split('.')
        list_SNR.append(int(_snr))
    list_SNR = np.unique(list_SNR)
    print("We have calculated results for SNR of: ", [x for x in list_SNR])
    # (4) Load the files for each case of SNR
    blues = cm.Blues(np.linspace(0.5, 1.0, len(list_SNR)))
    plt.figure()
    for i, snr in enumerate(list_SNR):
        _spax_scales = np.load(os.path.join(directory, 'spax_scales_SNR%d.npy' % snr))
        _mus_nyq = np.load(os.path.join(directory, 'mus_nyq_SNR%d.npy' % snr))
        _std_nyq = np.load(os.path.join(directory, 'std_nyq_SNR%d.npy' % snr))
        plt.errorbar(_spax_scales, _mus_nyq, _std_nyq, fmt='o', color=blues[i], label='%d' % snr)
        plt.plot(_spax_scales, _mus_nyq, color=blues[i])
    plt.legend(title='SNR')
    plt.xlabel(r'Spaxel Scales [mas]')
    plt.ylabel(r'RMS [nm] after calibration')
    plt.ylim(ymin=0)
    plt.show()

    # ================================================================================================================ #
    #           Can we train the model to be robust against Nyquist errors?
    # ================================================================================================================ #

    print("\nRobust Training")
    N_copies = N_WAVES
    # Generate multiple datasets with different diversity uncertainties
    train_PSF_robust, train_coef_robust = [], []
    test_PSF_robust, test_coef_robust = [], []
    for k in range(N_copies):

        # Generate the PSF model with the wrong Spaxel Scale
        PSF_error = psf.PointSpreadFunction(rbf_matrices[k], N_pix=N_PIX, crop_pix=pix,
                                            diversity_coef=diversity_defocus)

        _train_PSF, _train_coef, _test_PSF, _test_coef = calibration.generate_dataset(PSF_error, N_train, N_test,
                                                                                      coef_strength, rescale)
        train_PSF_robust.append(_train_PSF)
        train_coef_robust.append(_train_coef)
        test_PSF_robust.append(_test_PSF)
        test_coef_robust.append(_test_coef)
    train_PSF_robust = np.concatenate(train_PSF_robust, axis=0)
    train_coef_robust = np.concatenate(train_coef_robust, axis=0)
    test_PSF_robust = np.concatenate(test_PSF_robust, axis=0)
    test_coef_robust = np.concatenate(test_coef_robust, axis=0)

    # Robust Calibration Model
    robust_calib = calibration.Calibration(PSF_model=PSF_nom)
    robust_calib.create_cnn_model(layer_filters, kernel_size, name='ROBUST', activation='relu')
    losses = robust_calib.train_calibration_model(train_PSF_robust, train_coef_robust, test_PSF_robust, test_coef_robust,
                                                  N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                  readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    # Test the robust model
    # We go further from the +- 5 percent that the robust model was trained for to see waht happens
    SPAX_ERR_ROB = 0.10
    WAVE_MIN_ROB = WAVE * (1 - SPAX_ERR_ROB)
    WAVE_MAX_ROB = WAVE * (1 + SPAX_ERR_ROB)
    N = 21

    centers_multiwave_rob = psf.actuator_centres_multiwave(N_actuators=N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                       N_waves=N, wave0=WAVE_MIN_ROB, waveN=WAVE_MAX_ROB, wave_ref=WAVE)
    rbf_matrices_rob = psf.actuator_matrix_multiwave(centres=centers_multiwave_rob, alpha_pc=alpha_pc, rho_aper=RHO_APER,
                                                 rho_obsc=RHO_OBSC, N_waves=N,  wave0=WAVE_MIN_ROB, waveN=WAVE_MAX_ROB,
                                                 wave_ref=WAVE, N_PIX=N_PIX)

    spax_errs_rob = np.linspace(-SPAX_ERR_ROB, SPAX_ERR_ROB, N, endpoint=True)
    spax_scales_rob = SPAX * (1 - spax_errs_rob)
    mus_nyq_rob, std_nyq_rob = [], []
    print("\nLooping over Spaxel Scale Errors")
    for i, spaxel_error in enumerate(spax_errs_rob):

        print("\nSpaxel Error: %.1f percent" % (100 * spaxel_error))

        # Generate the PSF model with the wrong Spaxel Scale
        PSF_error = psf.PointSpreadFunction(rbf_matrices_rob[i], N_pix=N_PIX, crop_pix=pix,
                                            diversity_coef=diversity_defocus)
        test_PSF_error, test_coef_error, _PSF, _coef = calibration.generate_dataset(PSF_error, 500, 1,
                                                                                    coef_strength, rescale)

        robust_calib.PSF_model = PSF_error
        RMS_evolution_robust, residual_errors = robust_calib.calibrate_iterations(test_PSF_error, test_coef_error,
                                                                           wavelength=WAVE, N_iter=N_iter,
                                                                           readout_noise=True, RMS_readout=1. / SNR)
        robust_calib.PSF_model = PSF_nom

        final_RMS = RMS_evolution_robust[-1][-1]
        avg, std = np.mean(final_RMS), np.std(final_RMS)
        mus_nyq_rob.append(avg)
        std_nyq_rob.append(std)

    MIN_SPAX_ROB = SPAX * (1 - SPAX_ERR)
    MAX_SPAX_ROB = SPAX * (1 + SPAX_ERR)

    plt.figure()
    plt.errorbar(spax_scales, mus_nyq, std_nyq, fmt='o', label='Nominal')        # Nominal Model
    plt.plot(spax_scales, mus_nyq, color='blue')

    plt.errorbar(spax_scales_rob, mus_nyq_rob, std_nyq_rob, fmt='o', label='Robust')        # Robust Model
    plt.plot(spax_scales_rob, mus_nyq_rob, color='orange')

    # plt.axvline(MIN_SPAX_ROB, linestyle='--', alpha=0.7)
    # plt.axvline(MAX_SPAX_ROB, linestyle='--', alpha=0.7)
    plt.fill_between(x=[MIN_SPAX_ROB, MAX_SPAX_ROB], y1=0, y2=60, alpha=0.25, color='orange')

    plt.legend(title='Model', loc=3)
    plt.xlabel(r'Spaxel Scales [mas]')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.ylim([0, 60])
    plt.show()

    ensemb_calib = calibration.CalibrationEnsemble(PSF_model=PSF_nom)
    ensemb_calib.generate_ensemble_models(N_models=5, layer_filters=layer_filters, kernel_size=kernel_size,
                                          name='ENSEMBLE', activation='relu')
    ensemb_calib.train_ensemble_models(train_PSF, train_coef, test_PSF, test_coef,
                                           N_iter, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)
    RMS_evolution, residual = ensemb_calib.calibrate_iterations_ensemble(test_PSF, test_coef, wavelength=WAVE,
                                                                         N_iter=N_iter, readout_noise=True,
                                                                         RMS_readout=1./SNR)
    ensemb_calib.plot_RMS_evolution(RMS_evolution)
















