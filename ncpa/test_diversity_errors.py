"""

                          -||  DIVERSITY ERRORS  ||-

What happens when there are uncertainties on how much defocus we use?



Author: Alvaro Menduina
Date: Feb 2020
"""

import numpy as np
import matplotlib.pyplot as plt

import psf
import utils
import calibration
import noise

utils.print_title(message='\nN C P A', font=None, random_font=False)

### PARAMETERS ###

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

# Machine Learning bits
N_train, N_test = 5000, 1000       # Samples for the training of the models
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

# In case you need to reload a library after changing it
import importlib
importlib.reload(calibration)


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Zernike model so we can mimic the defocus
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=5, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.0)
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))

    # Calculate the Actuator Centres
    centers = psf.actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act = len(centers[0])
    psf.plot_actuators(centers, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)

    # Calculate the Actuator Model Matrix (Influence Functions)
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)

    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]

    # Create the PSF model using the Actuator Model for the wavefront
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=np.zeros(N_act))

    # Use Least Squares to find the actuator commands that mimic a Zernike defocus
    zernike_fit = calibration.Zernike_fit(PSF_zernike, PSF_actuators, wavelength=WAVE, rho_aper=RHO_APER)
    defocus_zernike = np.zeros((1, zernike_matrix.shape[-1]))
    defocus_zernike[0, 1] = 1.0
    defocus_actuators = zernike_fit.fit_zernike_wave_to_actuators(defocus_zernike, plot=True, cmap='bwr')[:, 0]

    # Update the Diversity Map on the actuator model so that it matches Defocus
    diversity_defocus = diversity * defocus_actuators
    PSF_actuators.define_diversity(diversity_defocus)

    plt.figure()
    plt.imshow(PSF_actuators.diversity_phase, cmap='RdBu')
    plt.colorbar()
    plt.title(r'Diversity Map | Defocus [rad]')
    plt.show()

    # Generate a training set for that nominal defocus
    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                              coef_strength, rescale)
    utils.plot_images(train_PSF)
    plt.show()

    # Train the Calibration Model on images with the nominal defocus
    calib = calibration.Calibration(PSF_model=PSF_actuators)
    calib.create_cnn_model(layer_filers, kernel_size, name='CALIBR', activation='relu')
    losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    RMS_evolution, residual = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                         readout_noise=True, RMS_readout=1./SNR)

    calib.plot_RMS_evolution(RMS_evolution)
    plt.show()

    # ================================================================================================================ #
    #           Impact of Defocus Uncertainties
    # ================================================================================================================ #

    # What happens when we show the model test images with a different defocus?
    # The expectation is the performance will degrade

    PV0 = 2         # The nominal Zernike Defocus has a Peak-Valley of 2 (+- 1)

    delta_divs = np.linspace(-0.10, 0.10, 15, endpoint=True)
    PV_uncertain = PV0 * (1 + delta_divs) * diversity
    mus_div, std_div = [], []
    for i, delta in enumerate(delta_divs):
        print("\nDefocus Uncertainties")
        print("%.2f" % delta)

        # Modify the Diversity to account for errors
        uncertain_diversity = (1 + delta) * diversity
        uncertain_defocus = uncertain_diversity * defocus_actuators
        PSF_actuators.define_diversity(uncertain_defocus)

        # Create a contaminated dataset for testing
        _PSF, _coef, test_PSF_uncertain, test_coef_uncertain = calibration.generate_dataset(PSF_actuators, N_train=1,
                                                                                            N_test=1000,
                                                                                            coef_strength=coef_strength,
                                                                                            rescale=rescale)

        # Let's see how that affects the performance
        RMS_evolution_uncertain, final_residual = calib.calibrate_iterations(test_PSF_uncertain, test_coef_uncertain,
                                                                             wavelength=WAVE, N_iter=N_iter,
                                                                             readout_noise=True, RMS_readout=1./SNR)
        deltaPV = delta * PV0
        title = r'$\Delta$ PV Diversity %.2f [$\lambda$] || Wavefronts [$\lambda$]' % deltaPV
        utils.show_wavefronts_grid(PSF_actuators, coefs=final_residual, rho_aper=RHO_APER, title=title)

        title = r'$\Delta$ PV Diversity %.2f [$\lambda$] || Wavefronts + $\Delta$ Diversity [$\lambda$]' % deltaPV
        utils.show_wavefronts_grid(PSF_actuators, coefs=final_residual + delta * diversity * defocus_actuators,
                                   rho_aper=RHO_APER, title=title)

        # plt.show()

        final_RMS = RMS_evolution_uncertain[-1][-1]
        avg, std = np.mean(final_RMS), np.std(final_RMS)
        mus_div.append(avg)
        std_div.append(std)


    plt.figure()
    plt.errorbar(100*delta_divs, mus_div, std_div)
    plt.xlabel(r'Diversity Variations [per cent]')
    plt.ylabel(r'RMS after calibration [nm]')

    plt.figure()
    plt.errorbar(PV_uncertain, mus_div, std_div)
    plt.xlabel(r'Diversity PV [rad]')
    plt.ylabel(r'RMS after calibration [nm]')

    plt.show()

    # Show what happens to the command residuals
    commands = np.mean(final_residual, axis=0)
    utils.plot_actuator_commands(commands=commands, centers=centers, rho_aper=RHO_APER, PIX=1024, cmap='jet',
                                 delta0=3, min_val=1)

    utils.show_wavefronts_grid(PSF_actuators, coefs=final_residual, rho_aper=RHO_APER)
    utils.show_wavefronts(PSF_actuators, coefs=final_residual, rho_aper=RHO_APER)
    plt.show()







