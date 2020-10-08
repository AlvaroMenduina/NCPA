"""




"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
N_actuators = 16                    # Number of actuators in [-1, 1] line
alpha_pc = 20                       # Height [percent] at the neighbour actuator (Gaussian Model)

# Machine Learning bits
N_train, N_test = 5000, 500       # Samples for the training of the models
coef_strength = 1.75 / (2 * np.pi)     # Strength of the actuator coefficients
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filters = [16, 8]    # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
epochs = 10                         # Training epochs
SNR = 500


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # (1) We begin by creating a Zernike PSF model with Defocus as diversity
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=5, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.1)
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))

    # Calculate the Actuator Centres
    centers = psf.actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act = len(centers[0])
    psf.plot_actuators(centers, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)

    plt.show()

    # Calculate the Actuator Model Matrix (Influence Functions)
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)

    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]

    diversity_actuators = np.zeros(N_act)
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=diversity_actuators)

    plt.show()

    # Use Least Squares to find the actuator commands that mimic the Zernike defocus
    zernike_fit = calibration.Zernike_fit(PSF_zernike, PSF_actuators, wavelength=WAVE, rho_aper=RHO_APER)
    defocus_zernike = np.zeros((1, zernike_matrix.shape[-1]))
    defocus_zernike[0, 1] = 1.0
    defocus_actuators = zernike_fit.fit_zernike_wave_to_actuators(defocus_zernike, plot=True, cmap='Reds')[:, 0]

    # Loop over the strength of defocus to see what is the optimum value
    rms_before, rms_after = [], []
    diversities, rms_div = [], []
    mean_peak_foc = []

    # Noise RMS
    N_cases = 30
    noiseSNR = np.array([500, 250, 100])
    N_noise = noiseSNR.shape[0]
    rms_after_noisy = np.zeros((N_noise, N_cases))
    rms_div_noisy = np.zeros((N_noise, N_cases))
    colors = cm.Reds(np.linspace(0.5, 0.75, N_noise))

    # beta = np.linspace(0.30, 4.0, N_cases)
    beta = np.linspace(0.25, 7.0, N_cases)
    N_foc = beta.shape[0]
    # fig, axes = plt.subplots(N_foc, 3)

    for k_noise in range(N_noise):

        for i_beta, b in enumerate(beta):

            print("\nBeta: %.2f" % b)
            # Update the Diversity Map on the actuator model so that it matches Defocus
            diversity_defocus = b * defocus_actuators / (2 * np.pi)
            PSF_actuators.define_diversity(diversity_defocus)
            # plt.figure()
            # plt.imshow(PSF_actuators.diversity_phase)
            # plt.colorbar()
            diversities.append(diversity_defocus)

            # Calculate the RMS
            div_phase = PSF_actuators.diversity_phase
            rms_foc = np.std(div_phase[PSF_actuators.pupil_mask])
            print("\nRMS Diversity: %.2f" % rms_foc)
            # rms_div.append(rms_foc)

            rms_div_noisy[k_noise, i_beta] = rms_foc

            # plt.figure()
            # plt.imshow(PSF_actuators.diversity_phase, cmap='RdBu')
            # plt.colorbar()
            # plt.title(r'Diversity Map | Defocus [rad]')

            # Generate a training set
            train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                                      coef_strength=coef_strength,
                                                                                      rescale=rescale)

            # Add some Readout Noise to the PSF images to spice things up
            SNR = noiseSNR[k_noise]
            noise_model = noise.NoiseEffects()
            train_PSF_readout = noise_model.add_readout_noise(train_PSF, RMS_READ=1. / SNR)
            test_PSF_readout = noise_model.add_readout_noise(test_PSF, RMS_READ=1. / SNR)
            #
            # # Show the Defocus phase, and some PSF examples
            # ax1 = axes[i_beta][0]
            # img1 = ax1.imshow(div_phase, cmap='RdBu', extent=[-1, 1, -1, 1])
            # img1.set_clim([-2.5, 2.5])
            # cbar1 = plt.colorbar(img1, ax=ax1)
            # ax1.set_xlim([-RHO_APER, RHO_APER])
            # ax1.set_ylim([-RHO_APER, RHO_APER])
            # ax1.set_title(r'RMS: %.2f [$\lambda$]' % (rms_foc))
            #
            # ax2 = axes[i_beta][1]
            # img2 = ax2.imshow(train_PSF_readout[0, :, :, 0], cmap='plasma')
            # # img2.set_clim([0, 1])
            # cbar2 = plt.colorbar(img2, ax=ax2)
            #
            #
            # ax3 = axes[i_beta][2]
            # img3 = ax3.imshow(train_PSF_readout[0, :, :, 1], cmap='plasma')
            # # img3.set_clim([0, 1])
            # cbar3 = plt.colorbar(img3, ax=ax3)
            #
            # for ax in axes[i_beta]:
            #     ax.xaxis.set_visible(False)
            #     ax.yaxis.set_visible(False)
            #
            # if i_beta == 0:
            #     ax2.set_title(r'In-focus PSF')
            #     ax3.set_title(r'Defocus PSF')

            calib_actu = calibration.Calibration(PSF_model=PSF_actuators)
            calib_actu.create_cnn_model(layer_filters, kernel_size, name='NOM_ACTU', activation='relu')
            losses = calib_actu.train_calibration_model(train_PSF_readout, train_coef, test_PSF_readout, test_coef,
                                                        N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                        plot_val_loss=False,
                                                        readout_noise=False, RMS_readout=[1. / SNR], readout_copies=3)

            # evaluate the RMS performance
            guess_actu = calib_actu.cnn_model.predict(test_PSF_readout)
            residual_coef = test_coef - guess_actu

            RMS0, RMS = np.zeros(N_test), np.zeros(N_test)
            for k in range(N_test):
                ini_phase = np.dot(PSF_actuators.model_matrix, test_coef[k])
                res_phase = np.dot(PSF_actuators.model_matrix, residual_coef[k])
                RMS0[k] = np.std(ini_phase[PSF_actuators.pupil_mask])
                RMS[k] = np.std(res_phase[PSF_actuators.pupil_mask])
            meanRMS0 = np.mean(RMS0)
            meanRMS = np.mean(RMS)
            print("\nMean RMS after calibration: %.3f" % meanRMS)

            rms_after_noisy[k_noise, i_beta] = meanRMS

            # # Save the RMS before and after calibration to know how we have done
            # rms_before.append(meanRMS0)
            # rms_after.append(meanRMS)

            # record the Strehl of the defocus PSF to see how much we've depressed the peak
            peaks_foc = np.max(train_PSF[:, :, :, 1], axis=(1, 2))
            mean_peak_foc.append(np.mean(peaks_foc))

    # Show how the RMS of the diversity phase relates to the RMS after calibration
    markers = ['^', 'v', 'd']
    plt.figure()
    # plt.scatter(rms_div[3:], rms_after[3:], s=10, color='black', marker='*', label='Clean')
    for k_noise in range(N_noise):
        plt.scatter(2*np.pi*rms_div_noisy[k_noise], rms_after_noisy[k_noise], s=10, marker=markers[k_noise],
                    color=colors[k_noise], label='SNR %d' % noiseSNR[k_noise])

    plt.xlim(left=0)
    plt.xlabel(r'RMS Defocus [rad]')
    plt.ylabel(r'RMS After calibration [$2\pi$ rad]')
    plt.legend(loc=1)
    plt.show()



    # # Random Diversities
    #
    # rms_before, rms_after = [], []
    # diversities, rms_div = [], []
    # div_coef = 1 / (2 * np.pi)
    # mean_peak_foc = []
    #
    # # Noise RMS
    # N_cases = 25
    # noiseSNR = np.array([500, 250, 100])
    # N_noise = noiseSNR.shape[0]
    # rms_after_noisy = np.zeros((N_noise, N_cases))
    # rms_div_noisy = np.zeros((N_noise, N_cases))
    # colors = cm.Reds(np.linspace(0.5, 0.75, N_noise))
    #
    for k_noise in range(N_noise):
        i = 0
        for alpha in np.linspace(0.15, 6.0, N_cases):
            # loop over the coefficient intensity (a proxy for the RMS diversity)
            print("\nAlpha = %.3f" % alpha)

            for k in range(1):
                # do several random instances

                diversity_actuators = alpha * div_coef * np.random.uniform(-1, 1, size=N_act)
                diversities.append(diversity_actuators)

                # Create the PSF model using the Actuator Model for the wavefront
                PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                                        crop_pix=pix, diversity_coef=diversity_actuators)

                div_phase = np.dot(PSF_actuators.model_matrix, diversity_actuators)
                # rms_div_noisy[k_noise, i] = np.std(div_phase[PSF_actuators.pupil_mask])
                rms_div.append(np.std(div_phase[PSF_actuators.pupil_mask]))

                train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                                          coef_strength=coef_strength, rescale=rescale)

                # Add some Readout Noise to the PSF images to spice things up
                SNR = noiseSNR[k_noise]
                noise_model = noise.NoiseEffects()
                train_PSF_readout = noise_model.add_readout_noise(train_PSF, RMS_READ=1./SNR)
                test_PSF_readout = noise_model.add_readout_noise(test_PSF, RMS_READ=1./SNR)

                calib_actu = calibration.Calibration(PSF_model=PSF_actuators)
                calib_actu.create_cnn_model(layer_filters, kernel_size, name='NOM_ACTU', activation='relu')
                losses = calib_actu.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                            N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                            plot_val_loss=False,
                                                            readout_noise=False, RMS_readout=[1. / SNR], readout_copies=3)

                # evaluate the RMS performance
                guess_actu = calib_actu.cnn_model.predict(test_PSF)
                residual_coef = test_coef - guess_actu

                RMS0, RMS = np.zeros(N_test), np.zeros(N_test)
                for k in range(N_test):
                    ini_phase = np.dot(PSF_actuators.model_matrix, test_coef[k])
                    res_phase = np.dot(PSF_actuators.model_matrix, residual_coef[k])
                    RMS0[k] = np.std(ini_phase[PSF_actuators.pupil_mask])
                    RMS[k] = np.std(res_phase[PSF_actuators.pupil_mask])
                meanRMS0 = np.mean(RMS0)
                meanRMS = np.mean(RMS)

                # rms_after_noisy[k_noise, i] = meanRMS

                i += 1

                # Save the RMS before and after calibration to know how we have done
                rms_before.append(meanRMS0)
                rms_after.append(meanRMS)

                # record the Strehl of the defocus PSF to see how much we've depressed the peak
                peaks_foc = np.max(train_PSF[:, :, :, 1], axis=(1, 2))
                mean_peak_foc.append(np.mean(peaks_foc))
    #
    # plt.figure()
    # plt.scatter(rms_div, mean_peak_foc, s=10)
    # plt.ylim(bottom=0)
    # # plt.show()
    #
    # # Show how the RMS of the diversity phase relates to the RMS after calibration
    # markers = ['^', 'v', 'd']
    # plt.figure()
    # plt.scatter(rms_div[3:], rms_after[3:], s=10, color='black', marker='*', label='Clean')
    # for k_noise in range(N_noise):
    #     plt.scatter(rms_div_noisy[k_noise], rms_after_noisy[k_noise], s=10, marker=markers[k_noise],
    #                 color=colors[k_noise], label='SNR %d' % noiseSNR[k_noise])
    #
    # plt.ylim(bottom=0)
    # plt.xlabel(r'RMS Diversity [$2\pi$ rad]')
    # plt.ylabel(r'RMS After calibration [$2\pi$ rad]')
    # plt.legend(loc=3)
    # plt.show()
    #
    # # find the top N diversities that give the best performance
    # N_top = 5
    # idx_sort = np.argsort(rms_after)
    # fig, axes = plt.subplots(1, N_top)
    # for i in range(N_top):
    #     # best performing
    #     ax = axes[i]
    #     div_c = diversities[idx_sort[i]]
    #     rms_c = rms_after[idx_sort[i]]
    #     div_phase = np.dot(PSF_actuators.model_matrix, div_c)
    #     img = ax.imshow(div_phase, cmap='RdBu')
    #     cbar = plt.colorbar(img, ax=ax)
    #     ax.xaxis.set_visible(False)
    #     ax.yaxis.set_visible(False)
    #     ax.set_title(r'RMS: %.3f' % rms_c)
    #     img.set_clim(-1, 1)
    #
    #     # worst performing
    #     ax = axes[1][i]
    #     div_c = diversities[idx_sort[-4 -i]]
    #     rms_c = rms_after[idx_sort[-4 -i]]
    #     div_phase = np.dot(PSF_actuators.model_matrix, div_c)
    #     img = ax.imshow(div_phase, cmap='RdBu')
    #     cbar = plt.colorbar(img, ax=ax)
    #     ax.xaxis.set_visible(False)
    #     ax.yaxis.set_visible(False)
    #     ax.set_title(r'RMS: %.3f' % rms_c)
    #     # img.set_clim(-0.5, 0.5)
    #
    # plt.show()
    #
    # def performance(diversity_coef):
    #
    #     PSF_actuators.define_diversity(diversity_coef)
    #
    #     train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
    #                                                                               coef_strength=coef_strength, rescale=rescale)
    #
    #     # peaks_nom = np.max(train_PSF[:, :, :, 0], axis=(1, 2))
    #     # peaks_foc = np.max(train_PSF[:, :, :, 1], axis=(1, 2))
    #     # # utils.plot_images(train_PSF)
    #     # # plt.show()
    #     #
    #     # plt.figure()
    #     # plt.scatter(np.arange(N_train), peaks_nom, s=2)
    #     # plt.scatter(np.arange(N_train), peaks_foc, s=2)
    #     # plt.show()
    #
    #     calib_actu = calibration.Calibration(PSF_model=PSF_actuators)
    #     calib_actu.create_cnn_model(layer_filters, kernel_size, name='NOM_ACTU', activation='relu')
    #     losses = calib_actu.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
    #                                                 N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
    #                                                 plot_val_loss=False,
    #                                                 readout_noise=False, RMS_readout=[1. / SNR], readout_copies=3)
    #
    #     # evaluate the RMS performance
    #     guess_actu = calib_actu.cnn_model.predict(test_PSF)
    #     residual_coef = test_coef - guess_actu
    #
    #     RMS = np.zeros(N_test)
    #     for k in range(N_test):
    #         res_phase = np.dot(PSF_actuators.model_matrix, residual_coef[k])
    #         RMS[k] = np.std(res_phase[PSF_actuators.pupil_mask])
    #     meanRMS = np.mean(RMS)
    #
    #     return meanRMS
    #
    # rms = performance(diversity_actuators)
    #
    # from scipy.optimize import differential_evolution, Bounds
    #
    # bounds = N_act *[(-1, 1)]
    # results = differential_evolution(func=performance, bounds=bounds, maxiter=10)