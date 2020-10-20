"""
Date: August 2020

Comparison of Basis Function for Wavefront Definition in the context of ML calibration
________________________________________________________________________________________

Is there a basis that works better for calibration?
Is Zernike polynomials better-suited than Actuator Commands models?

Modified on October 2020
to include the effects of actuators -> pupil mapping errors

"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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
N_levels = 10                       # How many Zernike radial orders

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
    N_zern = zernike_matrix.shape[-1]
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
    # plt.show()

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

    # Show a random example of the phase
    rand_coef = np.random.uniform(low=-1, high=1, size=N_act)
    rand_phase = np.dot(actuator_matrix, rand_coef)
    img_max = max(-np.min(rand_phase), np.max(rand_phase))
    c = np.array(centers[0])
    plt.figure()
    img = plt.imshow(rand_phase, cmap='bwr', extent=[-1, 1, -1, 1])
    img.set_clim([-img_max, img_max])
    plt.scatter(c[:, 0], c[:, 1], color='black', s=10)
    plt.colorbar(img)
    plt.title(r'%d Actuators | Wavefront [rad]' % N_act)
    plt.xlim([-1.1*RHO_APER, 1.1*RHO_APER])
    plt.ylim([-1.1*RHO_APER, 1.1*RHO_APER])
    plt.axes().xaxis.set_visible(False)
    plt.axes().yaxis.set_visible(False)
    plt.show()

    # ================================================================================================================ #
    #                    MACHINE LEARNING
    # ================================================================================================================ #

    # Create a dataset of PSF images using the Zernike model. We will use these images to train both calibration models
    train_PSF_zern, train_coef_zern, \
    test_PSF_zern, test_coef_zern = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                 coef_strength, rescale)

    # peaks_nom = np.max(train_PSF_zern[:, :, :, 0], axis=(1, 2))
    # peaks_foc = np.max(train_PSF_zern[:, :, :, 1], axis=(1, 2))
    # utils.plot_images(train_PSF_zern)
    # plt.show()
    #
    # plt.figure()
    # plt.scatter(np.arange(N_train), peaks_nom, s=2)
    # plt.scatter(np.arange(N_train), peaks_foc, s=2)
    # plt.show()

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
    # plt.show()

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
        ax1.set_title(r'Zernike ($N_{zern}$=%d) $\sigma$=%.3f [rad]' % (zernike_matrix.shape[-1], rms_zern))
        ax2.set_title(r'Actuator ($N_{act}$=%d) LS fit [rad]' % N_act)
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

    # ================================================================================================================ #
    #                             Impact of Actuator -> Pupil Mapping errors
    # ================================================================================================================ #

    # Motivation: the mapping between actuators and their positions at the pupil plane can be unstable
    # In other words, when we push an actuator, we might think we are affecting an area of the pupil plane
    # while in reality, the projected centre for that actuator might have moved a little

    def actuator_random_shift(nominal_centres, delta_spacing, amplitude=0.10, angle=None):
        """
        Simulate mapping errors between the actuators theoretical positions
        and their actual position in the pupil

        We randomly shift the whole grid by "amplitude" a fraction of the actuator spacing
        The angle is chosen randomly, between 0 and 2 pi

        :param nominal_centres:
        :param delta_spacing:
        :param amplitude:
        :return:
        """

        N_act = len(nominal_centres)

        delta_radius = delta_spacing * amplitude
        angle = np.random.uniform(low=0.0, high=2*np.pi) if angle is None else angle

        moved_centres = []
        for k in range(N_act):

            x_nom, y_nom = nominal_centres[k]
            dx, dy = delta_radius * np.cos(angle), delta_radius * np.sin(angle)

            x_new, y_new = x_nom + dx, y_nom + dy
            moved_centres.append([x_new, y_new])

        return moved_centres, angle

    def compare_actuators(nominal_centres, moved_centres, shift, rho_aper, rho_obsc):
        """
        Plot the two sets of actuator centres [nominal] and [shifted] to see
        how they compare

        :param nominal_centres:
        :param moved_centres:
        :param shift:
        :param rho_aper:
        :param rho_obsc:
        :return:
        """

        N_act = len(nominal_centres)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        circ1 = Circle((0, 0), rho_aper, linestyle='--', fill=None)
        circ2 = Circle((0, 0), rho_obsc, linestyle='--', fill=None)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        s = 10
        for nom_c in nominal_centres:
            ax.scatter(nom_c[0], nom_c[1], s=s, color='red')
        for mov_c in moved_centres:
            ax.scatter(mov_c[0], mov_c[1], s=s, color='salmon')
        ax.set_aspect('equal')
        plt.xlim([-1.25 * rho_aper, 1.25 * rho_aper])
        plt.ylim([-1.25 * rho_aper, 1.25 * rho_aper])
        plt.title('%d actuators | Shift: %.1f percent' % (N_act, 100*shift))

        return


    shift = 0.0
    centers_rand, theta = actuator_random_shift(nominal_centres=centers[0], delta_spacing=centers[1], amplitude=shift)
    centers_rand = [centers_rand, centers[1]]
    compare_actuators(nominal_centres=centers[0], moved_centres=centers_rand[0], shift=shift,
                      rho_aper=RHO_APER, rho_obsc=RHO_OBSC)
    # psf.plot_actuators(centers, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)
    # psf.plot_actuators(centers_rand, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)
    plt.show()

    actuator_matrices_rand = psf.actuator_matrix(centres=centers_rand, alpha_pc=alpha_pc,
                                                 rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)

    # Create the PSF model using the Actuator Model for the wavefront
    PSF_actuators_rand = psf.PointSpreadFunction(matrices=actuator_matrices_rand, N_pix=N_PIX,
                                                 crop_pix=pix, diversity_coef=np.zeros(N_act))
    PSF_actuators_rand.define_diversity(diversity_defocus)

    # Compare the wavefront for the nominal actuator model, and the shifted one
    models = [PSF_actuators, PSF_actuators_rand]
    coef_rand = np.random.uniform(-0.15, 0.15, size=N_act)
    wavef_nom = np.dot(PSF_actuators.model_matrix, coef_rand)
    wavef_rand = np.dot(PSF_actuators_rand.model_matrix, coef_rand)
    cval_rand = max(-np.min(wavef_rand), np.max(wavef_rand))
    cval_nom = max(-np.min(wavef_nom), np.max(wavef_nom))
    cval = max(cval_nom, cval_rand)

    p_nom, s_nom = PSF_actuators.compute_PSF(coef_rand)
    p_rand, s_rand = PSF_actuators_rand.compute_PSF(coef_rand)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    extent = [-1, 1, -1, 1]
    img1 = ax1.imshow(wavef_nom, cmap='RdBu', extent=extent)
    img1.set_clim(-cval, cval)
    plt.colorbar(img1, ax=ax1)
    ax1.set_xlim([-1.1*RHO_APER, 1.1*RHO_APER])
    ax1.set_ylim([-1.1*RHO_APER, 1.1*RHO_APER])
    ax1.set_title(r'Nominal Actuator Model [$\lambda$]')

    img2 = ax2.imshow(wavef_rand, cmap='RdBu', extent=extent)
    img2.set_clim(-cval, cval)
    plt.colorbar(img2, ax=ax2)
    ax2.set_xlim([-1.1*RHO_APER, 1.1*RHO_APER])
    ax2.set_ylim([-1.1*RHO_APER, 1.1*RHO_APER])
    ax2.set_title(r'Shift %.1f percent,  $\theta=%.1f$ deg' % (shift * 100, theta / np.pi * 180))

    diff_rand = wavef_rand - wavef_nom
    cval_diff = max(-np.min(diff_rand), np.max(diff_rand))
    img3 = ax3.imshow(diff_rand, cmap='RdBu', extent=extent)
    img3.set_clim(-cval_diff, cval_diff)
    plt.colorbar(img3, ax=ax3)
    ax3.set_xlim([-1.1*RHO_APER, 1.1*RHO_APER])
    ax3.set_ylim([-1.1*RHO_APER, 1.1*RHO_APER])
    ax3.set_title('Wavefront difference [$\lambda$]')

    img4 = ax4.imshow(p_nom, cmap='plasma')
    # img1.set_clim(-cval, cval)
    plt.colorbar(img4, ax=ax4)
    ax4.set_title('Nominal PSF')

    img5 = ax5.imshow(p_rand, cmap='plasma')
    # img1.set_clim(-cval, cval)
    plt.colorbar(img5, ax=ax5)
    ax5.set_title('Shifted Actuators PSF')

    diff_psf = p_rand - p_nom
    cval_psf = max(-np.min(diff_psf), np.max(diff_psf))
    img6 = ax6.imshow(diff_psf, cmap='RdBu')
    img6.set_clim(-cval_psf, cval_psf)
    plt.colorbar(img6, ax=ax6)
    ax6.set_title('PSF difference')

    plt.show()

    # Test the performance on PSF images with the shifted actuator model

    # For that we need to forget about the Zernikes otherwise it will make our live very difficult

    def calculate_strehl(PSF_model, coeff):

        N_PSF = coeff.shape[0]
        strehl = np.zeros(N_PSF)
        for j in range(N_PSF):
            _p, s = PSF_model.compute_PSF(coeff[j])
            strehl[j] = s

        return strehl

    # Generate a training set based on the nominal PSF model (actuators)
    coef_strength_actu = coef_strength * 2
    train_PSF_actu, train_coef_actu, test_PSF_actu, test_coef_actu = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                                                  coef_strength_actu, rescale)

    # Train a Calibration model
    SNR = 500
    epochs = 1
    layer_filters = [32, 16, 8]
    calib_actuators = calibration.Calibration(PSF_model=PSF_actuators)
    calib_actuators.create_cnn_model(layer_filters, kernel_size, name='NOM_ACTU', activation='relu')
    losses = calib_actuators.train_calibration_model(train_PSF_actu, train_coef_actu, test_PSF_actu, test_coef_actu,
                                                N_loops=5, epochs_loop=epochs, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                readout_noise=True, RMS_readout=[1. / SNR], readout_copies=3)

    # Run the calibration on the nominal images for a few iterations
    N_iter = 4
    RMS_evo_nom, residuals_nom = calib_actuators.calibrate_iterations(test_images=test_PSF_actu, test_coefs=test_coef_actu,
                                                                      wavelength=WAVE, N_iter=N_iter,
                                                                      readout_noise=True, RMS_readout=1. / SNR)

    strehl_nom = calculate_strehl(PSF_model=PSF_actuators, coeff=residuals_nom)
    mu_nom, std_nom = np.mean(strehl_nom), np.std(strehl_nom)

    # Now we see what happens when the correction we apply is slightly off

    from numpy.fft import fftshift, fft2

    def compute_PSF_phase(PSF_model, phase, diversity=False):

        phase0 = phase
        if diversity:
            phase0 += PSF_model.diversity_phase

        pupil_function = PSF_model.pupil_mask * np.exp(1j * 2 * np.pi * phase0)
        image = (np.abs(fftshift(fft2(pupil_function)))) ** 2
        image /= PSF_model.PEAK

        image = image[PSF_model.minPix:PSF_model.maxPix, PSF_model.minPix:PSF_model.maxPix]

        return image


    def calibrate_iterations_shift(calib_model, PSF_model, PSF_model_shift, test_images, test_coef, N_iter, beta=1.0):

        coef0 = test_coef
        # we begin by calculating the wavefronts
        RMS0 = np.zeros(N_test)
        wave0 = np.zeros((N_test, N_PIX, N_PIX))
        for k in range(N_test):
            wave0[k] = np.dot(PSF_model.model_matrix, coef0[k])
            RMS0[k] = np.std(wave0[k][PSF_model.pupil_mask])

        wavefronts = np.zeros((N_iter + 1, N_test, N_PIX, N_PIX))
        wavefronts[0] = wave0

        s0 = np.mean(np.max(test_images[:, :, :, 0], axis=(1, 2)))
        strehl_evolution = [s0]
        RMS_evolution = [RMS0]

        fig, axes = plt.subplots(N_iter, 3)

        for j in range(N_iter):

            rbefore = np.std(wave0[0][PSF_model.pupil_mask])
            ax1 = axes[j][0]
            img1 = ax1.imshow(wave0[0], cmap='RdBu', extent=extent)
            plt.colorbar(img1, ax=ax1)
            ax1.set_title(r'Wavefront Before | RMS: %.3f $\lambda$' % rbefore)
            ax1.set_ylabel(r'Iteration #%d' % (j + 1))

            print("\nIteration #%d" % (j + 1))
            # (1) We begin by predicting the aberrations.
            guess = calib_model.cnn_model.predict(test_images)

            # (2) The residual wavefront is given by: the nominal WF we had, minus the "shifted" correction
            new_PSF = np.zeros((N_test, pix, pix, 2))
            new_wave = np.zeros((N_test, N_PIX, N_PIX))
            RMS_after = np.zeros(N_test)
            for k in range(N_test):
                wavef_before = wave0[k]

                # This is the KEY part. We apply a wavefront correction with the "SHIFTED" actuator model
                # so the correction will be slightly wrong, compared to what we think are doing
                bad_correction = beta * np.dot(PSF_model_shift.model_matrix, guess[k])
                bad_residual = wavef_before - bad_correction
                new_wave[k] = bad_residual
                RMS_after[k] = np.std(bad_residual[PSF_model.pupil_mask])

                if k == 0:
                    ax2 = axes[j][1]
                    img2 = ax2.imshow(bad_correction, cmap='RdBu', extent=extent)
                    plt.colorbar(img2, ax=ax2)
                    ax2.set_title(r'Correction #%d' % (j+1))
                    cval_corr = max(-np.min(bad_correction), np.max(bad_correction))
                    cval_wave = max(-np.min(wave0[0]), np.max(wave0[0]))
                    cval = max(cval_corr, cval_wave)

                    rafter = np.std(bad_residual[PSF_model.pupil_mask])
                    ax3 = axes[j][2]
                    img3 = ax3.imshow(bad_residual, cmap='RdBu', extent=extent)
                    plt.colorbar(img3, ax=ax3)
                    ax3.set_title(r'Residual Wavefront %.3f $\lambda$' % rafter)

                    img1.set_clim(-cval, cval)
                    img2.set_clim(-cval, cval)
                    img3.set_clim(-cval, cval)

                    for ax in axes[j]:
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)
                        ax.set_xlim([-RHO_APER, RHO_APER])
                        ax.set_ylim([-RHO_APER, RHO_APER])

                # Update the PSF images directly with the PHASE
                # using the residual aberrations
                new_PSF[k, :, :, 0] = compute_PSF_phase(PSF_model, phase=bad_residual)
                new_PSF[k, :, :, 1] = compute_PSF_phase(PSF_model, phase=bad_residual, diversity=True)

                #
                # # Update the WF as well
                # wave0[k] = bad_residual

            test_images = calib_actuators.noise_effects.add_readout_noise(new_PSF, RMS_READ=1 / SNR)
            s = np.mean(np.max(test_images[:, :, :, 0], axis=(1, 2)))
            strehl_evolution.append(s)
            RMS_evolution.append(RMS_after)
            wave0 = new_wave
            wavefronts[j + 1] = new_wave


        return wavefronts, RMS_evolution, strehl_evolution


    test_PSF_actu_noisy = calib_actuators.noise_effects.add_readout_noise(test_PSF_actu, RMS_READ=1/SNR)
    N_iter = 4
    wavefronts, RMS_ev, f_strehl = calibrate_iterations_shift(calib_actuators, PSF_actuators, PSF_actuators_rand,
                                                              test_PSF_actu_noisy, test_coef_actu, N_iter=N_iter)
    plt.show()
    m = [np.mean(x) for x in RMS_ev]



    # Run a loop over the Shift Errors
    N_iter = 4
    N_shift = 15
    shifts = np.linspace(0.0, 0.5, N_shift, endpoint=True)
    strehls = np.zeros((N_shift, N_iter + 1))

    import matplotlib.cm as cm

    plt.figure()
    for k in range(N_shift):

        # Create a model with the mapping error
        centers_rand, theta = actuator_random_shift(nominal_centres=centers[0], delta_spacing=centers[1],
                                                    amplitude=shifts[k])
        centers_rand = [centers_rand, centers[1]]

        actuator_matrices_rand = psf.actuator_matrix(centres=centers_rand, alpha_pc=alpha_pc,
                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)

        # Create the PSF model using the Actuator Model for the wavefront
        PSF_actuators_rand = psf.PointSpreadFunction(matrices=actuator_matrices_rand, N_pix=N_PIX,
                                                     crop_pix=pix, diversity_coef=np.zeros(N_act))
        PSF_actuators_rand.define_diversity(diversity_defocus)

        # Run the calibration using that correction model
        test_PSF_actu_noisy = calib_actuators.noise_effects.add_readout_noise(test_PSF_actu, RMS_READ=1 / SNR)
        results = calibrate_iterations_shift(calib_actuators, PSF_actuators, PSF_actuators_rand,
                                             test_PSF_actu_noisy, test_coef_actu, N_iter=N_iter, beta=1.0)
        final_strehl = results[-1]
        strehls[k] = final_strehl

    plt.close('all')


    plt.figure()
    for k in range(N_shift):
        # plot the Strehls
        plt.scatter((N_iter + 1) * [shifts[k]], strehls[k], color=colors, s=10)
    plt.show()

    colors = cm.Reds(np.linspace(0.25, 1.0, (N_iter + 1)))
    plt.figure()
    for k in range(N_iter + 1):
        plt.plot(shifts, strehls[:, k], color=colors[k], label='%d' % (k))
        plt.scatter(shifts, strehls[:, k], color=colors[k], s=15)
    plt.legend(title='Iteration')
    plt.ylabel(r'Strehl ratio [ ]')
    plt.xlabel(r'Mapping error $\epsilon / \Delta$ [ ]')
    plt.ylim(bottom=0)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xlim([0, 0.5])
    plt.grid(True)
    plt.show()

    def average_shift_error(coef_rand, shift, PSF_model, beta):

        wave_base = np.dot(PSF_model.model_matrix, coef_rand)
        N_models = 25
        angles = np.linspace(0, 2 * np.pi, N_models, endpoint=False)

        rms_res = np.zeros(N_models)
        for k in range(N_models):
            # Create a model with the mapping error
            centers_rand, theta = actuator_random_shift(nominal_centres=centers[0], delta_spacing=centers[1],
                                                        amplitude=shift,
                                                        angle=angles[k])
            centers_rand = [centers_rand, centers[1]]
            actuator_matrices_rand = psf.actuator_matrix(centres=centers_rand, alpha_pc=alpha_pc, rho_aper=RHO_APER,
                                                         rho_obsc=RHO_OBSC, N_PIX=N_PIX)
            PSF_actuators_rand = psf.PointSpreadFunction(matrices=actuator_matrices_rand, N_pix=N_PIX, crop_pix=pix,
                                                         diversity_coef=np.zeros(N_act))
            PSF_actuators_rand.define_diversity(diversity_defocus)
            wave_shift = np.dot(PSF_actuators_rand.model_matrix, c_rand)
            residual_wf = wave_base - beta * wave_shift
            rms_res[k] = np.std(residual_wf[PSF_model.pupil_mask])

        return np.mean(rms_res)


    c_rand = np.random.uniform(-1, 1, N_act)
    r = average_shift_error(coef_rand=c_rand, shift=0.10, PSF_model=PSF_actuators, beta=0.75)
    bs = np.linspace(0, 1, 10)
    rs = [average_shift_error(coef_rand=c_rand, shift=0.10, PSF_model=PSF_actuators, beta=b) for b in bs]
    plt.figure()
    plt.plot(bs, rs)
    plt.show()



    # Check what's the average impact of misalignments

    wave_base = np.dot(PSF_actuators.model_matrix, c_rand)
    N_models = 25
    shift = 0.10
    shift_models = []
    shfif_diffs = np.zeros((N_models, N_PIX, N_PIX))
    angles = np.linspace(0, 2*np.pi, N_models)
    for k in range(N_models):

        # Create a model with the mapping error
        centers_rand, theta = actuator_random_shift(nominal_centres=centers[0], delta_spacing=centers[1], amplitude=shift,
                                                    angle=angles[k])
        centers_rand = [centers_rand, centers[1]]
        actuator_matrices_rand = psf.actuator_matrix(centres=centers_rand, alpha_pc=alpha_pc, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)
        PSF_actuators_rand = psf.PointSpreadFunction(matrices=actuator_matrices_rand, N_pix=N_PIX, crop_pix=pix, diversity_coef=np.zeros(N_act))
        PSF_actuators_rand.define_diversity(diversity_defocus)
        wave_shift = np.dot(PSF_actuators_rand.model_matrix, c_rand)
        shfif_diffs[k] = wave_base - wave_shift

        shift_models.append(PSF_actuators_rand.model_matrix)

    mean_diff = np.mean(shfif_diffs, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    img1 = ax1.imshow(wave_base, cmap='RdBu')
    plt.colorbar(img1, ax=ax1)

    img2 = ax2.imshow(mean_diff, cmap='RdBu', extent=extent)
    cc = np.array(centers[0])
    ax2.scatter(cc[:, 0], cc[:, 1], s=5, color='black')
    plt.colorbar(img2, ax=ax2)
    plt.show()

    fig, axes = plt.subplots(5, 5)
    cval = max(-np.min(shfif_diffs), np.max(shfif_diffs))
    for i in range(5):
        for j in range(5):
            ax = axes[i][j]
            k = 5 * i + j
            print(k)
            img = ax.imshow(shfif_diffs[k], cmap='RdBu', extent=extent)
            # ax.scatter(cc[:, 0], cc[:, 1], s=2, color='black')
            img.set_clim(-cval, cval)
            ax.set_xlim([-RHO_APER, RHO_APER])
            ax.set_ylim([-RHO_APER, RHO_APER])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    plt.show()

    # Could you train a model to predict the misalignments???

    # Get the predicted coefficients for the nominal calibration
    train_coef_predict = calib_actuators.cnn_model.predict(train_PSF_actu)

    N_models = 50
    N = 100
    shift = 0.50
    angles = np.linspace(0, np.pi, N_models, endpoint=False)
    train_diff_PSF = []
    for k in range(N_models):

        # Create a model with the mapping error
        centers_rand, theta = actuator_random_shift(nominal_centres=centers[0], delta_spacing=centers[1], amplitude=shift,
                                                    angle=angles[k])
        centers_rand = [centers_rand, centers[1]]
        actuator_matrices_rand = psf.actuator_matrix(centres=centers_rand, alpha_pc=alpha_pc, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)
        PSF_actuators_rand = psf.PointSpreadFunction(matrices=actuator_matrices_rand, N_pix=N_PIX, crop_pix=pix, diversity_coef=np.zeros(N_act))
        PSF_actuators_rand.define_diversity(diversity_defocus)

        diff_PSF = np.zeros((N, pix, pix, 2))
        for k in range(N):

            if k % 500 == 0:
                print(k)

            wavef_before = np.dot(PSF_actuators.model_matrix, train_coef_actu[k])

            # This is the KEY part. We apply a wavefront correction with the "SHIFTED" actuator model
            # so the correction will be slightly wrong, compared to what we think are doing
            bad_correction = np.dot(PSF_actuators_rand.model_matrix, train_coef_predict[k])
            bad_residual = wavef_before - bad_correction
            new_PSF_nom = compute_PSF_phase(PSF_actuators, phase=bad_residual)
            new_PSF_foc = compute_PSF_phase(PSF_actuators, phase=bad_residual, diversity=True)

            diff_PSF[k, :, :, 0] = train_PSF_actu[k, :, :, 0] - new_PSF_nom
            diff_PSF[k, :, :, 1] = train_PSF_actu[k, :, :, 1] - new_PSF_foc
        train_diff_PSF.append(diff_PSF)

    train_diff_PSF = np.concatenate(train_diff_PSF, axis=0)
    # for k in range(N_models):
    #     plt.figure()
    #     plt.imshow(train_diff_PSF[k * N, :, :, 0] - train_diff_PSF[0, :, :, 0], cmap='RdBu')
    #     plt.colorbar()

    train_angles = np.zeros(N * N_models)
    for k in range(N_models):
        train_angles[k*N: (k + 1)*N] = angles[k]

    # Train a model to predict the angles

    # Create the PSF model using the Actuator Model for the wavefront
    PSF_useless = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=np.zeros(N_act))
    PSF_useless.N_coef = 1

    epochs = 10
    layer_filters = [32, 16, 8]
    calib_angles = calibration.Calibration(PSF_model=PSF_useless)
    calib_angles.create_cnn_model(layer_filters, kernel_size, name='ANGLES', activation='relu')
    losses = calib_angles.train_calibration_model(train_diff_PSF[:N * (N_models - 1)], train_angles[:N * (N_models - 1)],
                                                  train_diff_PSF[N * (N_models - 1):],
                                                  train_angles[N * (N_models - 1):],
                                                  N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                  readout_noise=False, RMS_readout=[1. / SNR], readout_copies=3)

    p_angles = calib_angles.cnn_model.predict(train_diff_PSF[N * (N_models - 1):])











    # Another iteration



    # res_nom = test_coef_actu - guess_actu_nom
    # print(np.mean(norm(test_coef_actu, axis=1)))
    # print(np.mean(norm(res_nom, axis=1)))

    # Generate a test set with the shifted model
    _PSF, _coef, test_PSF_actu_shift, test_coef_actu_shift = calibration.generate_dataset(PSF_actuators_rand, 0, N_test,
                                                                                          coef_strength_actu, rescale)

    guess_actu_shift = calib_actuators.cnn_model.predict(test_PSF_actu_shift)
    res_shift = test_coef_actu_shift - guess_actu_shift
    print(np.mean(norm(test_coef_actu_shift, axis=1)))
    print(np.mean(norm(res_shift, axis=1)))

    # Substitute the PSF model of the Calibration model so that it can update the PSF images correctly
    calib_actuators.PSF_model = PSF_actuators_rand
    # Calibrate the aberrations with the CNN trained on the NOMINAL model, but testing on the SHIFTED PSF images
    RMS_evolution, residuals_shift = calib_actuators.calibrate_iterations(test_images=test_PSF_actu_shift,
                                                                          test_coefs=test_coef_actu_shift,
                                                                          wavelength=WAVE, N_iter=N_iter)

    # Calculate the Strehl ratio that we would get for the nominal PSF model, for those coefficients
    strehl_shift = calculate_strehl(PSF_model=PSF_actuators, coeff=residuals_shift)
    mu_shift, std_shift = np.mean(strehl_shift), np.std(strehl_shift)






    # ================================================================================================================ #

    commands = np.mean(np.abs(res_actu), axis=0)
    fig, ax = utils.plot_actuator_commands(commands=commands, centers=centers, rho_aper=RHO_APER, PIX=1024, cmap='Reds')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title(r'Mean residual actuator command [rad]')
    plt.show()

    corr_coef_zern = np.corrcoef(res_zern.T)
    corr_coef_actu = np.corrcoef(res_actu_fit)

    res_actu_fit = zernike_fit.fit_actuator_wave_to_zernikes(res_actu)

    def actuator_correlation(corr_coef, centers, k_act, cmap='seismic'):

        cent, delta = centers
        delta0 = 4
        PIX = 1024
        x = np.linspace(-1.25 * RHO_APER, 1.25 * RHO_APER, PIX, endpoint=True)
        xx, yy = np.meshgrid(x, x)
        image = np.zeros((PIX, PIX))
        for i, (xc, yc) in enumerate(cent):
            act_mask = (xx - xc) ** 2 + (yy - yc) ** 2 <= (delta / delta0) ** 2
            image += corr_coef[k_act][i] * act_mask

        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(image, cmap=cmap)
        img.set_clim([-1, 1])
        plt.colorbar(img, ax=ax)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(r'Correlation Coefficients | Actuator #%d' % (k_act + 1))

    for k in [6, 57]:
        actuator_correlation(corr_coef_actu, centers, k_act=k)
    plt.show()


    fig, (ax1, ax2) = plt.subplots(1, 2)
    img1 = ax1.imshow(corr_coef_zern, cmap='bwr')
    plt.colorbar(img1, ax=ax1, orientation='horizontal')
    img1.set_clim(-1, 1)
    ax1.set_title(r'Zernike Corr. Matrix')
    ax1.set_xlabel(r'Coefficient #')

    img2 = ax2.imshow(corr_coef_actu - corr_coef_zern, cmap='bwr')
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

    # Compare the PSF
    eps = 1e-4
    k_act = 6
    act_cent = np.array(centers[0])
    xc, yc = act_cent[k_act]
    k_x = np.argwhere(np.abs(act_cent[:, 0] + xc) < eps)
    k_y = np.argwhere(np.abs(act_cent[:, 1] + yc) < eps)
    k_opp = np.intersect1d(k_x, k_y)[0]
    # k_opp = k_act

    print("Actuator #%d: [%.4f, %.4f]" % (k_act, xc, yc))
    print("Actuator #%d: [%.4f, %.4f]" % (k_opp, act_cent[k_opp][0], act_cent[k_opp][1]))

    alpha = 10.0
    coef = np.zeros(N_act)
    coef_opp = coef.copy()
    coef[k_act] = alpha
    # coef[k_opp] = alpha
    psf, _s = PSF_actuators.compute_PSF(coef)

    coef_opp = np.zeros(N_act)
    # coef_opp[k_act] = -alpha
    coef_opp[k_opp] = -alpha
    psf_opp, _s = PSF_actuators.compute_PSF(coef_opp)

    MAX = np.max(psf)

    cmap_phase = 'RdBu'
    cmap_psf = 'plasma'
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

    img1 = ax1.imshow(psf, cmap=cmap_psf)
    img1.set_clim([0, MAX])
    plt.colorbar(img1, ax=ax1)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title(r'PSF | Actuator #%d (+)' % (k_act + 1))

    img2 = ax2.imshow(psf_opp, cmap=cmap_psf)
    img2.set_clim([0, MAX])
    plt.colorbar(img2, ax=ax2)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_title(r'PSF | Actuator #%d (-)' % (k_opp + 1))

    diff = psf - psf_opp
    cmax = max(-np.min(diff), np.max(diff))
    img3 = ax3.imshow(diff, cmap='bwr')
    plt.colorbar(img3, ax=ax3)
    img3.set_clim([-cmax, cmax])
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set_title(r'PSF difference')

    phase = np.dot(PSF_actuators.model_matrix, coef)
    img4 = ax4.imshow(phase / alpha + 0.1 * PSF_actuators.pupil_mask, cmap=cmap_phase, extent=[-1, 1, -1, 1])
    img4.set_clim([-1, 1])
    plt.colorbar(img4, ax=ax4)
    ax4.set_xlim([-RHO_APER, RHO_APER])
    ax4.set_ylim([-RHO_APER, RHO_APER])
    ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    ax4.set_title(r'Influence Function | Actuator #%d (+)' % (k_act + 1))

    phase_opp = np.dot(PSF_actuators.model_matrix, coef_opp)
    img5 = ax5.imshow(phase_opp / alpha + 0.1 * PSF_actuators.pupil_mask, cmap=cmap_phase, extent=[-1, 1, -1, 1])
    img5.set_clim([-1, 1])
    plt.colorbar(img5, ax=ax5)
    ax5.set_xlim([-RHO_APER, RHO_APER])
    ax5.set_ylim([-RHO_APER, RHO_APER])
    ax5.xaxis.set_visible(False)
    ax5.yaxis.set_visible(False)
    ax5.set_title(r'Influence Function | Actuator #%d (-)' % (k_opp + 1))

    fig.delaxes(ax6)

    plt.show()




