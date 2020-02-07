"""

                          -||  CALIBRATION BENCHMARK  ||-

Let's benchmark every possible thing about the calibration



Author: Alvaro Menduina
Date: Feb 2020
"""

import numpy as np
import matplotlib.pyplot as plt

import psf
import utils
import calibration
import noise

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
alpha_pc = 20                       # Height [percent] at the neighbour actuator (Gaussian Model)

# Machine Learning bits
N_train, N_test = 5000, 500         # Samples for the training of the models
coef_strength = 0.30                # Strength of the actuator coefficients
diversity = 0.25                    # Strength of extra diversity commands
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

    utils.print_title(message='N C P A', font='mayhem_d')

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # ================================================================================================================ #
    #      Base Scenario - Get a model running, for reference
    # ================================================================================================================ #


    # Calculate the Actuator Centres
    centers = psf.actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act = len(centers[0])
    psf.plot_actuators(centers, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)

    # Calculate the Actuator Model Matrix (Influence Functions)
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)

    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]
    ones_diversity = np.random.uniform(-1, 1, size=N_act)
    diversity_actuators = diversity * ones_diversity

    # Create the PSF model using the Actuator Model for the wavefront
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=diversity_actuators)

    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                              coef_strength, rescale)

    calib = calibration.Calibration(PSF_model=PSF_actuators)
    calib.create_cnn_model(layer_filers, kernel_size, name='CALIBR', activation='relu')
    losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    RMS_evolution = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                               readout_noise=True, RMS_readout=1./SNR)

    calib.plot_RMS_evolution(RMS_evolution)
    plt.show()

    # ================================================================================================================ #
    #      TEST #1 - Does the strength of the Diversity Matter?
    # ================================================================================================================ #

    """
    Phase Diversity typically uses 1 wave
    """
    print("\n=================================================================================")
    print("                       TEST #1 - Does the strength of Diversity matter?")
    print("=================================================================================\n")

    ones_diversity = np.random.uniform(-1, 1, size=N_act)
    diversity_map = PSF_actuators.diversity_phase
    RMS_diversity = WAVE * 1e3 * np.std(diversity_map[PSF_actuators.pupil_mask])

    plt.figure()
    plt.imshow(diversity_map, extent=[-1, 1, -1, 1], cmap='RdBu')
    plt.colorbar()
    plt.title(r'Random Diversity Map [$\lambda$] | RMS: %.1f nm' % RMS_diversity)
    # plt.show()

    diversity_strengths = np.linspace(0.10, 1.0, 10, endpoint=True)
    RMS_divs = []
    mus_div, std_div = [], []

    print("Testing several values of diversity")
    for diversity in diversity_strengths:

        diversity_actuators = diversity * ones_diversity

        # For each diversity strength, recreate the PSF model
        PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                                crop_pix=pix, diversity_coef=diversity_actuators)

        diversity_map = PSF_actuators.diversity_phase
        RMS_diversity = WAVE * 1e3 * np.std(diversity_map[PSF_actuators.pupil_mask])
        RMS_divs.append(np.std(diversity_map[PSF_actuators.pupil_mask]))
        print("RMS of Diversity Map: %.2f nm || %.2f waves" % (RMS_diversity, RMS_diversity/WAVE/1e3))

        # Update the training set
        train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                                  coef_strength, rescale)

        calib = calibration.Calibration(PSF_model=PSF_actuators)
        calib.create_cnn_model(layer_filers, kernel_size, name='CALIBR', activation='relu')
        losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                               N_loops, epochs_loop, verbose=1, batch_size_keras=32,
                                               plot_val_loss=False,
                                               readout_noise=True, RMS_readout=[1. / SNR],
                                               readout_copies=readout_copies)

        RMS_evolution = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                   readout_noise=True, RMS_readout=1. / SNR)

        ### Check what is the final RMS after calibration
        final_RMS = RMS_evolution[-1][-1]
        avg, std = np.mean(final_RMS), np.std(final_RMS)
        mus_div.append(avg)
        std_div.append(std)

        print("\n======================================================================")
        print("RMS of Diversity Map: %.2f nm || %.2f waves" % (RMS_diversity, RMS_diversity / WAVE / 1e3))
        print("Final Performance: RMS %.2f +- %.2f nm" % (avg, std))
        print("======================================================================\n")


    plt.figure()
    plt.errorbar(RMS_divs, y=mus_div, yerr=std_div)
    plt.xlabel(r'RMS Diversity [rad]')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.show()

    # find the optimum
    i_opt = np.argmin(mus_div)
    opt_div = diversity_strengths[i_opt]

    # ================================================================================================================ #
    #      TEST #2 - Does the shape of the Diversity Matter?
    # ================================================================================================================ #


    print("\n=================================================================================")
    print("                       TEST #2 - Does the shape of the Diversity Matter?")
    print("=================================================================================\n")

    opt_div = 0.55
    N_random = 5
    RMS_divs = []
    mus_div, std_div = [], []

    print("Testing random Diversity Maps")
    for k in range(N_random):
        # Use a random Diversity
        ones_diversity = np.random.uniform(-1, 1, size=N_act)

        # Use the 'optimum' strength
        diversity_actuators = opt_div * ones_diversity

        # For each diversity strength, recreate the PSF model
        PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                                crop_pix=pix, diversity_coef=diversity_actuators)

        diversity_map = PSF_actuators.diversity_phase
        RMS_diversity = WAVE * 1e3 * np.std(diversity_map[PSF_actuators.pupil_mask])
        RMS_divs.append(np.std(diversity_map[PSF_actuators.pupil_mask]))
        print("RMS of Diversity Map: %.2f nm || %.2f waves" % (RMS_diversity, RMS_diversity/WAVE/1e3))

        # Update the training set
        train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                                  coef_strength, rescale)

        calib = calibration.Calibration(PSF_model=PSF_actuators)
        calib.create_cnn_model(layer_filers, kernel_size, name='CALIBR', activation='relu')
        losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                               N_loops, epochs_loop, verbose=1, batch_size_keras=32,
                                               plot_val_loss=False,
                                               readout_noise=True, RMS_readout=[1. / SNR],
                                               readout_copies=readout_copies)

        RMS_evolution = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                   readout_noise=True, RMS_readout=1. / SNR)

        ### Check what is the final RMS after calibration
        final_RMS = RMS_evolution[-1][-1]
        avg, std = np.mean(final_RMS), np.std(final_RMS)
        mus_div.append(avg)
        std_div.append(std)

        print("\n======================================================================")
        print("RMS of Diversity Map: %.2f nm || %.2f waves" % (RMS_diversity, RMS_diversity / WAVE / 1e3))
        print("Final Performance: RMS %.2f +- %.2f nm" % (avg, std))
        print("======================================================================\n")

    print("\n=======================================================================================")
    print("  Results for %d random Diversity Maps:" % N_random)
    for a, b, c in zip(RMS_divs, mus_div, std_div):
        print("     RMS Diversity Map: %.2f waves || RMS after calibration: %.2f +- %.2f nm" % (a, b, c))
    print("=======================================================================================\n")


    # ================================================================================================================ #
    #      TEST #3 - Is a random actuator diversity better / worse than a classic Zernike defocus?
    # ================================================================================================================ #


    print("\n=================================================================================")
    print("                       TEST #3 - Random Actuator Diversity vs Zernike Defocus")
    print("=================================================================================\n")

    # Base Actuator Model to fit the defocus
    alpha_pc = 20
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                                     N_PIX=N_PIX)
    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=np.zeros(N_act))

    # Create a random wavefront [Zernike Model]
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=5, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.0)
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    # Create a Zernike PSF model
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))

    # Use Least Squares to find the actuator commands that create a Zernike defocus
    zernike_fit = calibration.Zernike_fit(PSF_zernike, PSF_actuators, wavelength=WAVE, rho_aper=RHO_APER)
    defocus_zernike = np.zeros((1, zernike_matrix.shape[-1]))
    defocus_zernike[0, 1] = 1.0
    defocus_actuators = zernike_fit.fit_zernike_wave_to_actuators(defocus_zernike, plot=True, cmap='bwr')[:, 0]

    ### Show the Defocus Term
    plt.figure()
    plt.imshow(PSF_zernike.model_matrix[:, :, 1], cmap='jet')
    plt.colorbar()
    plt.title(r'Zernike Defocus')

    plt.show()

    diversity_strengths = np.linspace(0.1, 1.0, 20, endpoint=True)
    PV = 2 * diversity_strengths
    RMS_divs = []
    mus_div, std_div = [], []

    print("Testing several values of Actuator Defocus")
    for diversity in diversity_strengths:

        ### WATCH OUT! Now we use the Actuator equivalent of a Zernike Defocus
        diversity_defocus = diversity * defocus_actuators

        # alpha_pc = 10
        actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                         rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                                         N_PIX=N_PIX)
        actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]

        # For each diversity strength, recreate the PSF model
        PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                                crop_pix=pix, diversity_coef=diversity_defocus)

        diversity_map = PSF_actuators.diversity_phase
        RMS_diversity = WAVE * 1e3 * np.std(diversity_map[PSF_actuators.pupil_mask])
        RMS_divs.append(np.std(diversity_map[PSF_actuators.pupil_mask]))
        print("RMS of Diversity Map: %.2f nm || %.2f waves" % (RMS_diversity, RMS_diversity/WAVE/1e3))

        plt.figure()
        plt.imshow(diversity_map)
        plt.title("%.3f" % diversity)
        plt.colorbar()

        # Update the training set
        train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                                  coef_strength, rescale)

        calib = calibration.Calibration(PSF_model=PSF_actuators)
        calib.create_cnn_model(layer_filers, kernel_size, name='CALIBR', activation='relu')
        losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                               N_loops, epochs_loop, verbose=1, batch_size_keras=32,
                                               plot_val_loss=False,
                                               readout_noise=True, RMS_readout=[1. / SNR],
                                               readout_copies=readout_copies)

        RMS_evolution = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                   readout_noise=True, RMS_readout=1. / SNR)

        ### Check what is the final RMS after calibration
        final_RMS = RMS_evolution[-1][-1]
        avg, std = np.mean(final_RMS), np.std(final_RMS)
        mus_div.append(avg)
        std_div.append(std)

        print("\n======================================================================")
        print("RMS of Actuator Defocus: %.2f nm || %.2f waves" % (RMS_diversity, RMS_diversity / WAVE / 1e3))
        print("Final Performance: RMS %.2f +- %.2f nm" % (avg, std))
        print("======================================================================\n")

    plt.figure()
    plt.errorbar(RMS_divs, y=mus_div, yerr=std_div)
    plt.xlabel(r'RMS Diversity [rad]')
    plt.ylabel(r'RMS after calibration [nm]')

    plt.figure()
    plt.errorbar(PV, y=mus_div, yerr=std_div)
    plt.xlabel(r'PV Diversity [rad]')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.show()

    # ================================================================================================================ #
    #      TEST #4 - Does the Actuator Crosstalk (Alpha %) matter?
    # ================================================================================================================ #

    print("\n=================================================================================")
    print("                       TEST #4 - Does the Actuator Crosstalk (Alpha %) matter?")
    print("=================================================================================\n")

    ### WATCH OUT: we have to be very careful here because changing alpha
    # also changes the shape of the Equivalent Actuator Defocus
    # thus making the results not comparable.

    # We need to force the code to "recycle" the good Defocus map
    alpha_pc = 20
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                                     N_PIX=N_PIX)
    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]
    # Create the PSF model using the Actuator Model for the wavefront
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=0.55 * defocus_actuators)
    good_defocus = PSF_actuators.diversity_phase.copy()

    alphas = [2.5, 5.0, 10.0, 20.0, 30.0]
    mus_div, std_div = [], []
    print("Testing several values of Gaussian width ALPHA")
    for alpha_pc in alphas:


        # Calculate the Actuator Model Matrix (Influence Functions)
        actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                         rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                                         N_PIX=N_PIX)
        # plt.figure()
        # plt.imshow(actuator_matrix[:, :, 0])
        # plt.title(r"Alpha = %.2f" % alpha_pc)
        actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]

        # Create the PSF model using the Actuator Model for the wavefront
        PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                                crop_pix=pix, diversity_coef=np.zeros_like(defocus_actuators))

        c_act = 1 / (2 * np.pi) * np.random.uniform(-1, 1, size=N_act)
        phase0 = np.dot(actuator_matrix, c_act)
        p0 = min(np.min(phase0), -np.max(phase0))

        plt.figure()
        plt.imshow(phase0, extent=(-1, 1, -1, 1), cmap='bwr')
        plt.colorbar()
        plt.clim(p0, -p0)
        for c in centers[0]:
            plt.scatter(c[0], c[1], color='black', s=4)
        plt.xlim([-1.1 * RHO_APER, 1.1 * RHO_APER])
        plt.ylim([-1.1 * RHO_APER, 1.1 * RHO_APER])
        plt.title(r'%d Actuators | Wavefront [$\lambda$] | $\alpha$=%.1f' % (N_act, alpha_pc))

        ### WATCH OUT! Force the diversity to be the correct one
        PSF_actuators.diversity_phase = good_defocus.copy()

        diversity_map = PSF_actuators.diversity_phase
        RMS_diversity = WAVE * 1e3 * np.std(diversity_map[PSF_actuators.pupil_mask])
        RMS_divs.append(np.std(diversity_map[PSF_actuators.pupil_mask]))
        print("RMS of Diversity Map: %.2f nm || %.2f waves" % (RMS_diversity, RMS_diversity/WAVE/1e3))

        # Update the training set
        train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                                  coef_strength, rescale)

        calib = calibration.Calibration(PSF_model=PSF_actuators)
        calib.create_cnn_model(layer_filers, kernel_size, name='CALIBR', activation='relu')
        losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                               N_loops, epochs_loop, verbose=1, batch_size_keras=32,
                                               plot_val_loss=False,
                                               readout_noise=True, RMS_readout=[1. / SNR],
                                               readout_copies=readout_copies)

        RMS_evolution = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                   readout_noise=True, RMS_readout=1. / SNR)

        ### Check what is the final RMS after calibration
        final_RMS = RMS_evolution[-1][-1]
        avg, std = np.mean(final_RMS), np.std(final_RMS)
        mus_div.append(avg)
        std_div.append(std)

        print("\n======================================================================")
        print("Alpha Gaussian: %.2f percent" % (alpha_pc))
        print("Final Performance: RMS %.2f +- %.2f nm" % (avg, std))
        print("======================================================================\n")

    print("\n=======================================================================================")
    print("  Results for Actuator Crosstalk:")
    for a, b, c in zip(alphas, mus_div, std_div):
        print("     Alpha: %.1f percent || RMS after calibration: %.2f +- %.2f nm" % (a, b, c))
    print("=======================================================================================\n")


    # ================================================================================================================ #
    #      TEST #Y - Actuator Modelling errors?
    # ================================================================================================================ #

    # What if the spatial frequencies of the wavefront are much higher than what our actuators can correct??
    alpha_pc = 10

    ### Low-Frequency Actuator Model
    centers = psf.actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act = len(centers[0])
    psf.plot_actuators(centers, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)
    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=np.zeros(N_act))


    ### High Frequency Model
    centers_high = psf.actuator_centres(N_actuators=35, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act_high = len(centers_high[0])
    psf.plot_actuators(centers_high, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)

    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers_high, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)
    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]
    PSF_actuators_high = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                                 crop_pix=pix, diversity_coef=np.zeros(N_act_high))

    # Use Least Squares to find the actuator commands that create a Zernike defocus
    zernike_fit = calibration.Zernike_fit(PSF_zernike, PSF_actuators, wavelength=WAVE, rho_aper=RHO_APER)
    defocus_zernike = np.zeros((1, zernike_matrix.shape[-1]))
    defocus_zernike[0, 1] = 1.0
    defocus_actuators = zernike_fit.fit_zernike_wave_to_actuators(defocus_zernike, plot=True, cmap='bwr')[:, 0]

    # Update the Diversity for BOTH models
    div_foc = 0.55
    good_defocus = np.dot(PSF_actuators.model_matrix, div_foc * defocus_actuators)
    PSF_actuators.diversity_phase = good_defocus.copy()
    PSF_actuators_high.diversity_phase = good_defocus.copy()

    # Generate Random PSF images with the HIGH FREQUENCY MODEL
    train_PSF, train_coef_high, test_PSF, test_coef_high = calibration.generate_dataset(PSF_actuators_high,
                                                                                        N_train, N_test,
                                                                                        coef_strength, rescale)
    # Fit the LS coefficients from the HIGH Frequency to the Actuator Model
    model_fit = calibration.Zernike_fit(PSF_actuators_high, PSF_actuators, wavelength=WAVE, rho_aper=RHO_APER)
    train_coef = zernike_fit.fit_zernike_wave_to_actuators(train_coef_high, plot=True, cmap='bwr')
    test_coef = zernike_fit.fit_zernike_wave_to_actuators(test_coef_high, plot=True, cmap='bwr')

    calib = calibration.Calibration(PSF_model=PSF_actuators_high)
    calib.create_cnn_model(layer_filers, kernel_size, name='CALIBR', activation='relu')
    losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32,
                                           plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR],
                                           readout_copies=readout_copies)

    # Once train it is not straightforward to update the PSFs, because you need to substract the complete
    # wavefronts from different models, not just the coefficients...
    #
    # RMS_evolution = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
    #                                            readout_noise=True, RMS_readout=1. / SNR)
    #








