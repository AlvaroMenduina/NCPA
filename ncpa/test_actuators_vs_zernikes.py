"""
Date: August 2020

Comparison of Basis Function for Wavefront Definition in the context of ML calibration
________________________________________________________________________________________

Is there a basis that works better for calibration?
Is Zernike polynomials better-suited than Actuator Commands models?

"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns

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
N_actuators = 22                    # Number of actuators in [-1, 1] line
diversity = 0.30                    # Strength of extra diversity commands
alpha_pc = 20                       # Height [percent] at the neighbour actuator (Gaussian Model)

# Zernike model
N_levels = 12                       # How many Zernike radial orders

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
coef_strength = 1 / (2 * np.pi)     # Strength of the actuator coefficients
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filters = [128, 64, 32]    # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
epochs = 50                         # Training epochs


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # (1) We begin by creating a Zernike PSF model with Defocus as diversity
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=N_levels, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.1)
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))

    defocus_zernike = np.zeros(zernike_matrix.shape[-1])
    defocus_zernike[1] = diversity
    PSF_zernike.define_diversity(defocus_zernike)

    # (2) We create an Actuator PSF model
    # Calculate the Actuator Centres
    centers = psf.actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act = len(centers[0])
    psf.plot_actuators(centers, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)
    plt.show()

    # Calculate the Actuator Model Matrix (Influence Functions)
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)
    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]

    # Create the PSF model using the Actuator Model for the wavefront
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=np.zeros(N_act))

    # Use Least Squares to find the actuator commands that mimic the Zernike defocus
    zernike_fit = calibration.Zernike_fit(PSF_zernike, PSF_actuators, wavelength=WAVE, rho_aper=RHO_APER)
    defocus_zernike = np.zeros((1, zernike_matrix.shape[-1]))
    defocus_zernike[0, 1] = diversity
    defocus_actuators = zernike_fit.fit_zernike_wave_to_actuators(defocus_zernike, plot=True, cmap='Reds')[:, 0]

    # Update the Diversity Map on the actuator model so that it matches Defocus
    diversity_defocus = defocus_actuators
    PSF_actuators.define_diversity(diversity_defocus)

    plt.figure()
    plt.imshow(PSF_actuators.diversity_phase, cmap='RdBu')
    plt.colorbar()
    plt.title(r'Diversity Map | Defocus [rad]')
    #
    # plt.figure()
    # plt.imshow(PSF_zernike.diversity_phase, cmap='RdBu')
    # plt.colorbar()
    # plt.title(r'Diversity Map | Defocus [rad]')
    # plt.show()

    # ================================================================================================================ #
    #                    MACHINE LEARNING
    # ================================================================================================================ #

    # Create a dataset of PSF images using the Zernike model. We will use these images to train both calibration models
    train_PSF_zern, train_coef_zern, \
    test_PSF_zern, test_coef_zern = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                 coef_strength, rescale)

    peaks_nom = np.max(train_PSF_zern[:, :, :, 0], axis=(1, 2))
    peaks_foc = np.max(train_PSF_zern[:, :, :, 1], axis=(1, 2))
    utils.plot_images(train_PSF_zern)
    plt.show()

    plt.figure()
    plt.scatter(np.arange(N_train), peaks_nom, s=2)
    plt.scatter(np.arange(N_train), peaks_foc, s=2)
    plt.show()

    # Using a LS fit, calculate the Actuator commands that reproduce the Zernike phase maps
    train_coef_actu = zernike_fit.fit_zernike_wave_to_actuators(train_coef_zern, plot=True, cmap='RdBu').T
    test_coef_actu = zernike_fit.fit_zernike_wave_to_actuators(test_coef_zern, plot=True, cmap='RdBu').T

    def calculate_rms_fit(zern_coef, actu_coef):

        N_PSF = zern_coef.shape[0]
        rms0 = np.zeros(N_PSF)
        rms_fit = np.zeros(N_PSF)
        for k in range(N_PSF):
            zern_phase = np.dot(PSF_zernike.model_matrix, zern_coef[k])
            actu_phase = np.dot(PSF_actuators.model_matrix, actu_coef[k])
            residual_phase = zern_phase - actu_phase
            rms0[k] = np.std(zern_phase[PSF_zernike.pupil_mask])
            rms_fit[k] = np.std(residual_phase[PSF_zernike.pupil_mask])

        return rms0, rms_fit

    # Calculate the 'fitting' error, i.e. how badly do the actuator commands reproduce the Zernike maps
    rms0, rms_fit = calculate_rms_fit(train_coef_zern, train_coef_actu)
    rel_err = rms_fit / rms0 * 100      # percentage of RMS left over after fitting to actuators

    print("\nLS Fit error: %.1f +- %.1f" % (np.mean(rel_err), np.std(rel_err)))

    fig, ax = plt.subplots(1, 1)
    ax.hist(rel_err, bins=50, histtype='step')
    ax.set_xlim([0, np.max(rel_err)])
    ax.set_xlabel(r'Percentage of RMS after removing actuators')
    plt.show()

    # show some examples of Zernike Phase vs Actuator Phase
    N_ex = 3
    cmap = 'seismic'
    fig_ex, axes = plt.subplots(N_ex, 3, figsize=(12, 12))
    for k in range(N_ex):
        ax1, ax2, ax3 = axes[k]
        zern_phase = np.dot(PSF_zernike.model_matrix, train_coef_zern[k])
        rms_zern = np.std(zern_phase[PSF_zernike.pupil_mask])
        actu_phase = np.dot(PSF_actuators.model_matrix, train_coef_actu[k])
        residual = zern_phase - actu_phase
        rms_res = np.std(residual[PSF_zernike.pupil_mask])
        cmax_zern = max(-np.min(zern_phase), np.max(zern_phase))
        cmax_actu = max(-np.min(actu_phase), np.max(actu_phase))
        cmax = max(cmax_zern, cmax_actu)

        img1 = ax1.imshow(zern_phase, cmap=cmap, extent=[-1, 1, -1, 1])
        img1.set_clim(-cmax, cmax)
        ax1.set_xlim([-RHO_APER, RHO_APER])
        ax1.set_ylim([-RHO_APER, RHO_APER])
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        # cbar1 = plt.colorbar(img1, ax=ax1)
        # cbar1.set_label(r'[$\lambda$]', rotation=270)

        img2 = ax2.imshow(actu_phase, cmap=cmap, extent=[-1, 1, -1, 1])
        img2.set_clim(-cmax, cmax)
        ax2.set_xlim([-RHO_APER, RHO_APER])
        ax2.set_ylim([-RHO_APER, RHO_APER])
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        # cbar2 = plt.colorbar(img2, ax=ax2)
        # cbar2.set_label(r'$\lambda$')

        img3 = ax3.imshow(residual, cmap='seismic', extent=[-1, 1, -1, 1])
        img3.set_clim(-cmax, cmax)
        ax3.set_xlim([-RHO_APER, RHO_APER])
        ax3.set_ylim([-RHO_APER, RHO_APER])
        ax3.xaxis.set_visible(False)
        ax3.yaxis.set_visible(False)
        plt.colorbar(img3, ax=ax3)

        # if k == 0:
        ax1.set_title(r'Zernike $\sigma$=%.3f [rad]' % rms_zern)
        ax2.set_title(r'Actuator LS fit [rad]')
        ax3.set_title(r'Residual $\sigma$=%.3f [rad]' % rms_res)
    plt.tight_layout()
    plt.show()

    SNR = 500
    epochs = 10
    # Train the Calibration Model on images with the nominal defocus
    calib_zern = calibration.Calibration(PSF_model=PSF_zernike)
    calib_zern.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='relu')
    losses = calib_zern.train_calibration_model(train_PSF_zern, train_coef_zern, test_PSF_zern, test_coef_zern,
                                                N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                readout_noise=False, RMS_readout=[1. / SNR], readout_copies=3)

    # def calculate_rms_calibration(calibration_model, psf_model, test_images, test_coef):
    #
    #     guess_coef = calibration_model.cnn_model.predict(test_images)
    #     residual_coef = test_coef - guess_coef
    #     N_images = test_images.shape[0]
    #     RMS_before = np.zeros(N_images)
    #     RMS_after = np.zeros(N_images)
    #     for k in range(N_images):
    #         phase_before = np.dot(psf_model.model_matrix, test_coef[k])
    #         phase_after = np.dot(psf_model.model_matrix, residual_coef[k])
    #         RMS_before[k] = np.std(phase_before[psf_model.pupil_mask])
    #         RMS_after[k] = np.std(phase_after[psf_model.pupil_mask])
    #
    #     return RMS_before, RMS_after
    #
    # rms_before_zern, rms_after_zern = calculate_rms_calibration(calibration_model=calib_zern, psf_model=PSF_zernike,
    #                                                             test_images=test_PSF_zern, test_coef=test_coef_zern)
    #
    # # Evaluate performance
    # guess_coef_zern = calib_zern.cnn_model.predict(test_PSF_zern)
    # residual_coef_zern = test_coef_zern - guess_coef_zern

    calib_actu = calibration.Calibration(PSF_model=PSF_actuators)
    calib_actu.create_cnn_model(layer_filters, kernel_size, name='NOM_ACTU', activation='relu')
    losses = calib_actu.train_calibration_model(train_PSF_zern, train_coef_actu, test_PSF_zern, test_coef_actu,
                                                N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                readout_noise=False, RMS_readout=[1. / SNR], readout_copies=3)

    def fix_axes(ax):
        ax.set_xlim([-RHO_APER, RHO_APER])
        ax.set_ylim([-RHO_APER, RHO_APER])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return

    def compare_models(test_images, zern_coef, actu_coef, max_imgs=5):

        N_images = test_images.shape[0]

        guess_zern = calib_zern.cnn_model.predict(test_images)
        guess_actu = calib_actu.cnn_model.predict(test_images)

        res_zern = np.zeros_like(zern_coef)
        res_actu = np.zeros_like(actu_coef)

        # the residual before, and after calibration for each case
        RMS0 = np.zeros(N_images)
        RMS_zern = np.zeros(N_images)
        RMS_actu = np.zeros(N_images)

        choices = np.random.choice(a=N_images, size=max_imgs, replace=False)

        for k in range(N_images):

            true_phase = np.dot(PSF_zernike.model_matrix, zern_coef[k])
            RMS0[k] = np.std(true_phase[PSF_zernike.pupil_mask])

            # The Phase Map that the Zernike calibration model guesses and its residual after correction
            guess_zern_phase = np.dot(PSF_zernike.model_matrix, guess_zern[k])
            res_zern_phase = true_phase - guess_zern_phase
            RMS_zern[k] = np.std(res_zern_phase[PSF_zernike.pupil_mask])
            res_zern[k] = zern_coef[k] - guess_zern[k]

            guess_actu_phase = np.dot(PSF_actuators.model_matrix, guess_actu[k])
            res_actu_phase = true_phase - guess_actu_phase
            RMS_actu[k] = np.std(res_actu_phase[PSF_zernike.pupil_mask])
            res_actu[k] = actu_coef[k] - guess_actu[k]

            extent = [-1, 1, -1, 1]
            cmap = 'seismic'
            if k in choices:
                # show some examples
                fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

                img1 = ax1.imshow(test_images[k, :, :, 0], cmap='plasma')
                # img1.set_clim(0, np.max())
                cbar1 = plt.colorbar(img1, ax=ax1)
                ax1.xaxis.set_visible(False)
                ax1.yaxis.set_visible(False)
                ax1.set_title(r'In-focus PSF')

                cmax = max(-np.min(true_phase), np.max(true_phase))

                img2 = ax2.imshow(guess_zern_phase, extent=extent, cmap=cmap)
                img2.set_clim(-cmax, cmax)
                cbar2 = plt.colorbar(img2, ax=ax2)
                fix_axes(ax2)
                ax2.set_title(r'Zernike prediction')

                img3 = ax3.imshow(res_zern_phase, extent=extent, cmap=cmap)
                img3.set_clim(-cmax, cmax)
                cbar3 = plt.colorbar(img3, ax=ax3)
                ax3.set_title(r'Zernike residual $\sigma$ = %.3f rad' % RMS_zern[k])
                fix_axes(ax3)

                img4 = ax4.imshow(true_phase, extent=extent, cmap=cmap)
                img4.set_clim(-cmax, cmax)
                cbar4 = plt.colorbar(img4, ax=ax4)
                ax4.set_title(r'PSF wavefront $\sigma$ = %.3f rad' % RMS0[k])
                fix_axes(ax4)

                img5 = ax5.imshow(guess_actu_phase, extent=extent, cmap=cmap)
                img5.set_clim(-cmax, cmax)
                cbar5 = plt.colorbar(img5, ax=ax5)
                ax5.set_title(r'Actuator prediction')
                fix_axes(ax5)

                img6 = ax6.imshow(res_actu_phase, extent=extent, cmap=cmap)
                img6.set_clim(-cmax, cmax)
                cbar6 = plt.colorbar(img6, ax=ax6)
                ax6.set_title(r'Actuator residual $\sigma$ = %.3f rad' % RMS_actu[k])
                fix_axes(ax6)

                # plt.tight_layout()

        return RMS0, RMS_zern, RMS_actu, guess_zern, guess_actu, res_zern, res_actu

    results = compare_models(test_images=test_PSF_zern, zern_coef=test_coef_zern, actu_coef=test_coef_actu)
    RMS0, RMS_zern, RMS_actu, guess_zern, guess_actu, res_zern, res_actu = results
    plt.show()

    print("\nPerformance Comparison:")
    print("RMS before calibration: %.4f +- %.4f rad" % (np.mean(RMS0), np.std(RMS0)))
    print("Zernike Model: RMS after calibration: %.4f +- %.4f rad" % (np.mean(RMS_zern), np.std(RMS_zern)))
    print("Actuator Model: RMS after calibration: %.4f +- %.4f rad" % (np.mean(RMS_actu), np.std(RMS_actu)))

    commands = np.mean(np.abs(res_actu), axis=0)
    fig, ax = utils.plot_actuator_commands(commands=commands, centers=centers, rho_aper=RHO_APER, PIX=1024, cmap='Reds')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title(r'Mean residual actuator command [rad]')
    plt.show()

    corr_coef_zern = np.corrcoef(guess_zern.T)
    corr_coef_actu = np.corrcoef(guess_actu.T)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    img1 = ax1.imshow(corr_coef_zern, cmap='bwr')
    plt.colorbar(img1, ax=ax1, orientation='horizontal')
    img1.set_clim(-1, 1)
    ax1.set_title(r'Zernike Corr. Matrix')
    ax1.set_xlabel(r'Coefficient #')

    img2 = ax2.imshow(corr_coef_actu, cmap='bwr')
    plt.colorbar(img2, ax=ax2, orientation='horizontal')
    img2.set_clim(-1, 1)
    ax2.set_title(r'Actuator Corr. Matrix')
    ax2.set_xlabel(r'Coefficient #')
    plt.show()


    fig, ax = plt.subplots(1, 1)
    ax.hist(RMS_zern, bins=20, histtype='step')
    ax.hist(RMS_actu, bins=20, histtype='step')
    plt.show()

    grid_size = 250
    fig, ax = plt.subplots(1, 1)
    sns.kdeplot(RMS0, RMS_zern, shade=True, ax=ax, label='Zernike', gridsize=grid_size)
    sns.kdeplot(RMS0, RMS_actu, shade=False, ax=ax, label='Actuator', color='white', gridsize=grid_size)
    # ax.scatter(rms0, rms_fit, color='red', s=5)
    ax.set_xlim([0.05, 0.15])
    ax.set_ylim([0.0, 0.05])
    ax.legend(title=r'Calibration Model', facecolor='whitesmoke', loc=2)
    ax.set_xlabel(r'RMS [rad] before calibration')
    ax.set_ylabel(r'RMS [rad] after calibration')
    # ax.set_aspect('equal')
    plt.show()




    rms_before_actu, rms_after_actu = calculate_rms_calibration(calibration_model=calib_actu, psf_model=PSF_actuators,
                                                                test_images=test_PSF_zern, test_coef=test_coef_actu)

    plt.figure()
    plt.scatter(rms_before_zern, rms_after_zern, s=5)
    plt.scatter(rms_before_zern, rms_after_actu, s=5)
    plt.show()

    #
    RMS_evo_zern, final_res_zern = calib_zern.calibrate_iterations(test_images=test_PSF_zern, test_coefs=test_coef_zern,
                                                                     wavelength=WAVE, N_iter=3)
    #
    # RMS_evo_actu, final_res_actu = calib_actu.calibrate_iterations(test_images=test_PSF_zern, test_coefs=test_coef_actu,
    #                                                                  wavelength=WAVE, N_iter=3)
    #
    calib_zern.plot_RMS_evolution(RMS_evolution=RMS_evo_zern)
    # calib_actu.plot_RMS_evolution(RMS_evolution=RMS_evo_actu)
    #
    # import seaborn as sns
    # fig, ax = plt.subplots(1, 1)
    # sns.kdeplot(rms0, rms_fit, shade=True, ax=ax)
    # slope = 0.10
    # x0 = np.linspace(0, 1.1 * np.max(rms0), 10)
    # y0 = slope * x0
    # ax.plot(x0, y0, linestyle='--', color='black')
    # ax.set_xlim([0.02, 0.16])
    # ax.set_ylim([0.002, 0.016])
    # plt.show()


