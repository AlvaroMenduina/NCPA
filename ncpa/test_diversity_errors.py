"""

                          -||  DIVERSITY ERRORS  ||-

What happens when there are uncertainties on how much defocus we use?
We need to train our model with some form of diversity (typically defocus)
The calibration models do not know "how much" we use, they only need it to be
consistent across the training and test sets

However, in real life, it is difficult to exactly pinpoint how much diversity is used
There can be uncertainties in the DM response, for instance

Thus, we are interested in:
(1) Quantifying how uncertainties in the diversity affect the performance of the calibration
(2) Designing ways to make the calibration models robust against such uncertainties

Author: Alvaro Menduina
Date: Feb 2020
"""

import numpy as np
import matplotlib.pyplot as plt

import psf
import utils
import calibration

# In case you need to reload a library after changing it
import importlib
importlib.reload(calibration)

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
    utils.plot_images(train_PSF[5000:])
    plt.show()

    # Train the Calibration Model on images with the nominal defocus
    calib = calibration.Calibration(PSF_model=PSF_actuators)
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

    # ================================================================================================================ #
    #           Impact of Defocus Uncertainties
    # ================================================================================================================ #

    # What happens when we show the model test images with a different defocus?
    # The expectation is the performance will degrade

    PV0 = 2         # The nominal Zernike Defocus has a Peak-Valley of 2 (+- 1)

    delta_divs = np.linspace(-0.10, 0.10, 11, endpoint=True)
    PV_uncertain = PV0 * (1 + delta_divs) * diversity
    mus_div, std_div = [], []
    for i, delta in enumerate(delta_divs):
        print("\nDefocus Uncertainties")
        print("%.2f" % delta)

        # Modify the Diversity to account for errors
        uncertain_diversity = (1 + delta) * diversity
        uncertain_defocus = uncertain_diversity * defocus_actuators
        PSF_actuators.define_diversity(uncertain_defocus)
        print(np.std(PSF_actuators.diversity_phase[PSF_actuators.pupil_mask]))

        # Create a contaminated dataset for testing
        _PSF, _coef, test_PSF_uncertain, test_coef_uncertain = calibration.generate_dataset(PSF_actuators, N_train=1,
                                                                                            N_test=500,
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
    plt.errorbar(100*delta_divs, mus_div, std_div, fmt='o')
    plt.plot(100*delta_divs, mus_div)
    plt.xlabel(r'Diversity Variations [per cent]')
    plt.ylabel(r'RMS after calibration [nm]')

    plt.figure()
    plt.errorbar(PV_uncertain, mus_div, std_div)
    plt.xlabel(r'Diversity PV [rad]')
    plt.ylabel(r'RMS after calibration [nm]')

    plt.show()

    # ================================================================================================================ #
    #           Can we train the model to be robust against variations?
    # ================================================================================================================ #

    N_copies = 5
    diversity_error = 0.10

    # Reasign the proper Defocus
    # Update the Diversity Map on the actuator model so that it matches Defocus
    diversity_defocus = diversity * defocus_actuators
    PSF_actuators.define_diversity(diversity_defocus)

    robust_calib = calibration.Calibration(PSF_model=PSF_actuators)

    # Generate multiple datasets with different diversity uncertainties
    train_PSF_rob, train_coef_rob, \
    test_PSF_rob, test_coef_rob = calibration.robust_diversity(PSF_actuators, N_train, N_test, coef_strength,
                                                               rescale=0.35, diversity_error=diversity_error,
                                                               N_copies=N_copies)

    utils.plot_images(train_PSF_rob)
    plt.show()

    robust_calib.create_cnn_model(layer_filers, kernel_size, name='ROBUST', activation='relu')
    losses = robust_calib.train_calibration_model(train_PSF_rob, train_coef_rob, test_PSF_rob, test_coef_rob,
                                                  N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                  readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    # Repeat the analysis with the robust model
    delta_divs = np.linspace(-0.10, 0.10, 11, endpoint=True)
    PV_uncertain = PV0 * (1 + delta_divs) * diversity
    mus_div_rob, std_div_rob = [], []
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
        RMS_evolution_robust, final_residual = robust_calib.calibrate_iterations(test_PSF_uncertain, test_coef_uncertain,
                                                                             wavelength=WAVE, N_iter=4,
                                                                             readout_noise=True, RMS_readout=1./SNR)
        deltaPV = delta * PV0
        title = r'$\Delta$ PV Diversity %.2f [$\lambda$] || Wavefronts [$\lambda$]' % deltaPV
        utils.show_wavefronts_grid(PSF_actuators, coefs=final_residual, rho_aper=RHO_APER, title=title)

        title = r'$\Delta$ PV Diversity %.2f [$\lambda$] || Wavefronts + $\Delta$ Diversity [$\lambda$]' % deltaPV
        utils.show_wavefronts_grid(PSF_actuators, coefs=final_residual + delta * diversity * defocus_actuators,
                                   rho_aper=RHO_APER, title=title)

        # plt.show()
        final_RMS = RMS_evolution_robust[-1][-1]
        avg, std = np.mean(final_RMS), np.std(final_RMS)
        mus_div_rob.append(avg)
        std_div_rob.append(std)

    # ## FIX
    # mus_div_rob = [x for x in mus_div_rob[::2]]
    # std_div_rob = [x for x in mus_div_rob[1::2]]

    plt.figure()
    plt.errorbar(100*delta_divs, mus_div, std_div, fmt='o', label='Nomimal')
    plt.errorbar(100*delta_divs, mus_div_rob, std_div_rob, fmt='o', label='Robust')
    plt.legend(title='Calibration Model')
    plt.ylim([0, 75])
    plt.xlabel(r'Diversity Variations [per cent]')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.show()

    # ================================================================================================================ #
    #                                           ZERNIKE Polynomials
    # ================================================================================================================ #

    # Create a Zernike model so we can mimic the defocus
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=7, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.1)
    ### Watch out, we need to oversize the Zernike radius because of the bad behaviour at the pupil edge


    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))

    # Create a random wavefront [Zernike Model]
    N_zern = zernike_matrix.shape[-1]
    defocus_zernike = np.zeros(zernike_matrix.shape[-1])
    defocus_zernike[1] = diversity
    PSF_zernike.define_diversity(defocus_zernike)

    # Nominal training set
    train_PSF_zern, train_coef_zern, \
    test_PSF_zern, test_coef_zern = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                 coef_strength, rescale)
    utils.plot_images(train_PSF_zern)
    plt.show()

    # Train the Calibration Model on images with the nominal defocus
    calib_zern = calibration.Calibration(PSF_model=PSF_zernike)
    calib_zern.create_cnn_model(layer_filers, kernel_size, name='NOM_ZERN', activation='relu')
    losses = calib_zern.train_calibration_model(train_PSF_zern, train_coef_zern, test_PSF_zern, test_coef_zern,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    RMS_evolution_zern, residual_zern = calib_zern.calibrate_iterations(test_PSF_zern, test_coef_zern, wavelength=WAVE, N_iter=4,
                                                         readout_noise=True, RMS_readout=1./SNR)

    calib_zern.plot_RMS_evolution(RMS_evolution_zern)
    plt.show()

    delta_divs = np.linspace(-0.10, 0.10, 11, endpoint=True)
    PV_uncertain = PV0 * (1 + delta_divs) * diversity
    mus_div_zern, std_div_zern = [], []
    for i, delta in enumerate(delta_divs):

        print("\nDefocus Uncertainties")
        print("%.2f" % delta)

        # Modify the Diversity to account for errors
        PSF_zernike.define_diversity((1 + delta) * defocus_zernike)
        print(np.std(PSF_zernike.diversity_phase[PSF_zernike.pupil_mask]))

        # Create a contaminated dataset for testing
        _PSF, _coef, test_PSF_uncertain, test_coef_uncertain = calibration.generate_dataset(PSF_zernike, N_train=1,
                                                                                            N_test=500,
                                                                                            coef_strength=coef_strength,
                                                                                            rescale=rescale)

        # Let's see how that affects the performance
        RMS_evolution_zernike, zernike_residual = calib_zern.calibrate_iterations(test_PSF_uncertain,
                                                                                  test_coef_uncertain,
                                                                                  wavelength=WAVE, N_iter=4,
                                                                                  readout_noise=True,
                                                                                  RMS_readout=1. / SNR)
        deltaPV = delta * PV0
        title = r'$\Delta$ PV Diversity %.2f [$\lambda$] || Wavefronts [$\lambda$]' % deltaPV
        utils.show_wavefronts_grid(PSF_zernike, coefs=zernike_residual, rho_aper=RHO_APER, title=title)

        title = r'$\Delta$ PV Diversity %.2f [$\lambda$] || Wavefronts + $\Delta$ Diversity [$\lambda$]' % deltaPV
        utils.show_wavefronts_grid(PSF_zernike, coefs=zernike_residual + delta * diversity * defocus_zernike,
                                   rho_aper=RHO_APER, title=title)

        # plt.show()

        final_RMS = RMS_evolution_zernike[-1][-1]
        avg, std = np.mean(final_RMS), np.std(final_RMS)
        mus_div_zern.append(avg)
        std_div_zern.append(std)


    plt.figure()
    plt.errorbar(100*delta_divs, mus_div_zern, std_div_zern, fmt='o', label='Nomimal')
    # plt.errorbar(100*delta_divs, mus_div_rob, std_div_rob, fmt='o', label='Robust')
    plt.legend(title='Calibration Model [Zernike]')
    # plt.ylim([0, 75])
    plt.xlabel(r'Diversity Variations [per cent]')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.show()

    # ================================================================================================================ #
    #           Can we train the model to be robust against variations?
    # ================================================================================================================ #

    N_copies = 5
    diversity_error = 0.05

    # Reasign the proper Defocus
    # Update the Diversity Map on the actuator model so that it matches Defocus
    defocus_zernike = np.zeros(zernike_matrix.shape[-1])
    defocus_zernike[1] = diversity
    PSF_zernike.define_diversity(defocus_zernike)
    print(np.std(PSF_zernike.diversity_phase[PSF_zernike.pupil_mask]))

    robust_calib_zernike = calibration.Calibration(PSF_model=PSF_zernike)

    # Generate multiple datasets with different diversity uncertainties
    train_PSF_rob_zern, train_coef_rob_zern, \
    test_PSF_rob_zern, test_coef_rob_zern = calibration.robust_diversity(PSF_zernike, N_train, N_test, coef_strength,
                                                               rescale=0.35, diversity_error=diversity_error,
                                                               N_copies=N_copies)

    utils.plot_images(train_PSF_rob_zern)
    plt.show()

    robust_calib_zernike.create_cnn_model(layer_filers, kernel_size, name='ROBUST_ZERN', activation='relu')
    losses = robust_calib_zernike.train_calibration_model(train_PSF_rob_zern, train_coef_rob_zern,
                                                          test_PSF_rob_zern, test_coef_rob_zern,
                                                  N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                  readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)



    # Repeat the analysis with the robust model
    delta_divs = np.linspace(-diversity_error, diversity_error, 11, endpoint=True)
    PV_uncertain = PV0 * (1 + delta_divs) * diversity
    mus_div_rob_zern, std_div_rob_zern = [], []
    for i, delta in enumerate(delta_divs):
        print("\nDefocus Uncertainties")
        print("%.2f" % delta)

        # Modify the Diversity to account for errors
        PSF_zernike.define_diversity((1 + delta) * defocus_zernike)
        print(np.std(PSF_zernike.diversity_phase[PSF_zernike.pupil_mask]))

        # Create a contaminated dataset for testing
        _PSF, _coef, test_PSF_uncertain, test_coef_uncertain = calibration.generate_dataset(PSF_zernike, N_train=1,
                                                                                            N_test=500,
                                                                                            coef_strength=coef_strength,
                                                                                            rescale=rescale)

        # Let's see how that affects the performance
        RMS_evolution_zernike, zernike_residual = robust_calib_zernike.calibrate_iterations(test_PSF_uncertain,
                                                                                  test_coef_uncertain,
                                                                                  wavelength=WAVE, N_iter=4,
                                                                                  readout_noise=True,
                                                                                  RMS_readout=1. / SNR)
        # deltaPV = delta * PV0
        # title = r'$\Delta$ PV Diversity %.2f [$\lambda$] || Wavefronts [$\lambda$]' % deltaPV
        # utils.show_wavefronts_grid(PSF_actuators, coefs=final_residual, rho_aper=RHO_APER, title=title)
        #
        # title = r'$\Delta$ PV Diversity %.2f [$\lambda$] || Wavefronts + $\Delta$ Diversity [$\lambda$]' % deltaPV
        # utils.show_wavefronts_grid(PSF_actuators, coefs=final_residual + delta * diversity * defocus_actuators,
        #                            rho_aper=RHO_APER, title=title)

        # plt.show()
        final_RMS = RMS_evolution_zernike[-1][-1]
        avg, std = np.mean(final_RMS), np.std(final_RMS)
        mus_div_rob_zern.append(avg)
        std_div_rob_zern.append(std)

    plt.figure()
    plt.errorbar(100*delta_divs, mus_div_zern, std_div_zern, fmt='o', label='Nomimal')
    plt.errorbar(100*delta_divs, mus_div_rob_zern, std_div_rob_zern, fmt='o', label='Robust')
    plt.legend(title='Calibration Model [Zernike]')
    # plt.ylim([0, 75])
    plt.xlabel(r'Diversity Variations [per cent]')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.show()


    # Modify the Diversity to account for errors
    delta = 0.05
    uncertain_diversity = (1 + delta) * diversity
    PSF_zernike.define_diversity((1 + delta) * defocus_zernike)
    print(np.std(PSF_zernike.diversity_phase[PSF_zernike.pupil_mask]))

    # Create a contaminated dataset for testing
    _PSF, _coef, test_PSF_uncertain, test_coef_uncertain = calibration.generate_dataset(PSF_zernike, N_train=1,
                                                                                        N_test=500,
                                                                                        coef_strength=coef_strength,
                                                                                        rescale=rescale)

    # zern_coef = np.zeros((500, N_zern))
    # zern_coef[:, 0] = np.random.uniform(-coef_strength, coef_strength, 500)
    # __PSF = calib_zern.update_PSF(zern_coef)
    # utils.plot_images(__PSF)
    # plt.show()



    # Let's see how that affects the performance
    RMS_evolution_zernike, zernike_residual = calib_zern.calibrate_iterations(test_PSF_uncertain, test_coef_uncertain,
                                                                         wavelength=WAVE, N_iter=3,
                                                                         readout_noise=True, RMS_readout=1. / SNR)

    _x = np.linspace(-coef_strength, coef_strength, 10)
    for i in range(10):
        plt.figure()
        plt.scatter(test_coef_uncertain[:,i], test_coef_uncertain[:,i] - zernike_residual[:,i], s=5)
        plt.plot(_x, _x)
        plt.xlim([-coef_strength, coef_strength])
        plt.ylim([-coef_strength, coef_strength])
    plt.show()











