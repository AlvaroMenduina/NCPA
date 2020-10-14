"""

                          -||  ANAMORPHIC ERRORS  ||-

The anamorphic magnification ratio of 2:1 for HARMONI will be subject to errors

What happens to the performance when you show the model
PSF images that have a slightly different anamorphic ratio?

Does that affect the predictions of the model?
Does it mistake the effect for some specific aberrations?

Author: Alvaro Menduina
Date: Oct 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import psf
import utils
import calibration


# PSF bits
N_PIX = 256                         # pixels for the Fourier arrays
pix = 64                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength
SPAX = 1.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
print("Nominal Parameters | Spaxel Scale and Wavelength")
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
N_levels = 8                        # Number of Zernike radial levels levels
coef_strength = 0.20                # Strength of Zernike aberrations
diversity = 1.0                     # Strength of the Defocus diversity
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filters = [64, 32, 16, 8]      # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
SNR = 500                           # SNR for the Readout Noise
N_loops, epochs_loop = 5, 5         # How many times to loop over the training
readout_copies = 2                  # How many copies with Readout Noise to use
N_iter = 3                          # How many iterations to run the calibration (testing)

directory = os.path.join(os.getcwd(), 'Anamorphic')

import importlib
importlib.reload(calibration)


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)


    ### (0) Show the impact of ellipticity variations

    # Define a nominal PSF Zernike model with no anamorphic errors. Perfectly circular pupil
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=N_levels, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC, N_PIX=N_PIX,
                                                                          radial_oversize=1.0, anamorphic_ratio=1.0)

    N_zern = zernike_matrix.shape[-1]
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))

    defocus_zernike = np.zeros(zernike_matrix.shape[-1])
    defocus_zernike[1] = diversity / (2 * np.pi)
    PSF_zernike.define_diversity(defocus_zernike)

    # Define a model with some anamorphic error
    ratio = 1.10        # a / b for the ellipse. > 1.0 elongated PSF along Y
    zernike_matrix_anam, pupil_mask_zernike_anam, flat_zernike_anam = psf.zernike_matrix(N_levels=N_levels, rho_aper=RHO_APER,
                                                                                         rho_obsc=RHO_OBSC, N_PIX=N_PIX,
                                                                                         radial_oversize=1.0,
                                                                                         anamorphic_ratio=ratio)

    zernike_matrices_anam = [zernike_matrix_anam, pupil_mask_zernike_anam, flat_zernike_anam]
    PSF_zernike_anam = psf.PointSpreadFunction(matrices=zernike_matrices_anam, N_pix=N_PIX,
                                               crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))
    PSF_zernike_anam.define_diversity(defocus_zernike)

    # Generate some PSF images with random aberrations, see how the two PSFs compare
    N_rows = 3
    N_cols = 3
    fig, axes = plt.subplots(N_rows, N_cols)
    for k in range(N_rows):

        rand_coef = np.random.uniform(low=-0.15, high=0.15, size=N_zern)
        p, s = PSF_zernike.compute_PSF(rand_coef)
        p_anam, s = PSF_zernike_anam.compute_PSF(rand_coef)
        diff = p_anam - p
        cval = max(-np.min(diff), np.max(diff))

        X = pix//2 * SPAX
        extent = [-X, X, -X, X]

        ax1 = axes[k][0]
        img1 = ax1.imshow(p, cmap='plasma', extent=extent, origin='lower')
        plt.colorbar(img1, ax=ax1)

        ax2 = axes[k][1]
        img2 = ax2.imshow(p_anam, cmap='plasma', extent=extent, origin='lower')
        plt.colorbar(img2, ax=ax2)

        ax3 = axes[k][2]
        img3 = ax3.imshow(diff, cmap='RdBu', extent=extent, origin='lower')
        img3.set_clim(-cval, cval)
        plt.colorbar(img3, ax=ax3)

        for ax in axes[k]:
            ax.set_xlabel(r'X [mas]')
            ax.set_ylabel(r'Y [mas]')

        if k == 0:
            ax1.set_title(r'Nominal')
            ax2.set_title(r'Ratio $a/b=%.2f$' % ratio)
            ax3.set_title(r'Difference')

    plt.show()

    # Now we do that for many cases and take the average difference
    N_trials = 100
    mean_diff = []
    for k in range(N_trials):
        rand_coef = np.random.uniform(low=-0.15, high=0.15, size=N_zern)
        p, s = PSF_zernike.compute_PSF(rand_coef)
        p_anam, s = PSF_zernike_anam.compute_PSF(rand_coef)
        diff = p_anam - p
        mean_diff.append(diff)

    avg_diff = np.mean(np.array(mean_diff), axis=0)

    plt.figure()
    plt.imshow(avg_diff, cmap='RdBu', extent=extent, origin='lower')
    plt.colorbar()

    plt.show()

    # Do this calculation over several cases of the anamorphic ratio variations

    ratios = [0.75, 0.9, 1.1, 1.25]
    N_ratios = len(ratios)
    fig, axes = plt.subplots(1, N_ratios)
    for k, r in enumerate(ratios):

        # define the PSF model
        zernike_matrix_anam, pupil_mask_zernike_anam, flat_zernike_anam = psf.zernike_matrix(N_levels=N_levels,
                                                                                             rho_aper=RHO_APER,
                                                                                             rho_obsc=RHO_OBSC,
                                                                                             N_PIX=N_PIX,
                                                                                             radial_oversize=1.0,
                                                                                             anamorphic_ratio=r)

        zernike_matrices_anam = [zernike_matrix_anam, pupil_mask_zernike_anam, flat_zernike_anam]
        PSF_zernike_anam = psf.PointSpreadFunction(matrices=zernike_matrices_anam, N_pix=N_PIX,
                                                   crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))
        PSF_zernike_anam.define_diversity(defocus_zernike)

        # Calculate the average change in the PSF
        N_trials = 100
        mean_diff = []
        for i in range(N_trials):
            rand_coef = np.random.uniform(low=-0.15, high=0.15, size=N_zern)
            p, s = PSF_zernike.compute_PSF(rand_coef)
            p_anam, s = PSF_zernike_anam.compute_PSF(rand_coef)
            diff = p_anam - p
            mean_diff.append(diff)

        avg_diff = np.mean(np.array(mean_diff), axis=0)

        cval = max(-np.min(avg_diff), np.max(avg_diff))
        ax = axes[k]
        img = ax.imshow(avg_diff, cmap='RdBu', extent=extent, origin='lower')
        img.set_clim(-0.075, 0.075)
        plt.colorbar(img, ax=ax, orientation='horizontal')
        ax.set_title(r'Ratio $a/b=%.2f$' % r)
        ax.set_xlabel(r'X [mas]')
        ax.set_ylabel(r'Y [mas]')

    plt.show()

    # Can we fit the PSF variations due to anamorphic magnification errors to
    # "fake" aberrations in the nominal model. In other words, is there a combination
    # of Zernike polynomials that produces a similar change?

    from scipy.optimize import least_squares

    def residuals_with_defocus(x_coef, PSF_zernike, PSF_zernike_anam):

        # Calculate the NOMINAL PSF for each model
        x = np.zeros(N_zern)
        # p_nom, s = PSF_zernike.compute_PSF(x)
        p_nom_anam, s = PSF_zernike_anam.compute_PSF(x)
        # diff_nom = p_nom_anam - p_nom

        # Calculate the NOMINAL PSF with the fake aberrations
        fake_psf_nom, s = PSF_zernike.compute_PSF(x + x_coef)
        # fake_diff_nom = fake_psf_nom - p_nom

        nom_diff = (fake_psf_nom - p_nom_anam).flatten()

        # # Now the defocus channel
        # # p_foc, s = PSF_zernike.compute_PSF(x + defocus_zernike)
        # p_foc_anam, s = PSF_zernike_anam.compute_PSF(x + defocus_zernike)
        # # diff_foc = p_foc_anam - p_foc
        #
        # fake_psf_foc, s = PSF_zernike.compute_PSF(x + x_coef + defocus_zernike)
        # # fake_diff_foc = fake_psf_foc - p_foc
        #
        # foc_diff = (fake_psf_foc - p_foc_anam).flatten()
        #
        # total = np.concatenate([nom_diff, 0.1*foc_diff])
        # print(total.shape)

        return nom_diff

    ratios = [1.05, 1.10, 1.25]
    markers = ['^', 'v', 'd', 'o']
    colors = cm.Reds(np.linspace(0.5, 1.0, 3))
    plt.figure()
    for j, r in enumerate(ratios):

        # (1) We define the anamorphic PSF model
        anamorphic_ratio = r
        zernike_matrices_anam = psf.zernike_matrix(N_levels=N_levels, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX,
                                                   radial_oversize=1.0, anamorphic_ratio=anamorphic_ratio)
        zernike_matrix_anam, pupil_mask_zernike_anam, flat_zernike_anam = zernike_matrices_anam

        zernike_matrices_anam = [zernike_matrix_anam, pupil_mask_zernike_anam, flat_zernike_anam]
        PSF_zernike_anam = psf.PointSpreadFunction(matrices=zernike_matrices_anam, N_pix=N_PIX,
                                                   crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))
        PSF_zernike_anam.define_diversity(defocus_zernike)

        x0 = np.random.uniform(low=-0.1, high=0.1, size=N_zern)
        result = least_squares(fun=residuals_with_defocus, x0=x0, args=(PSF_zernike, PSF_zernike_anam),
                               verbose=2)
        x_coef = result.x

        plt.scatter(np.arange(1, N_zern + 1), x_coef, label="%.2f" % r, s=12,
                    color=colors[j], marker=markers[j])
    plt.xlabel(r'Zernike polynomial')
    plt.ylabel(r'LS coefficient [$\lambda$]')
    plt.xticks(np.arange(1, N_zern + 1))
    plt.legend(title=r'Ratio $a/b$')
    plt.show()

    # Show which aberrations are important
    ls_aberr = [1, 2, 9, 10, 11, 21, 22, 23, 24]

    zernike_matrices_aux = psf.zernike_matrix(N_levels=N_levels, rho_aper=1.0, rho_obsc=0.0, N_PIX=N_PIX,
                                               radial_oversize=1.0, anamorphic_ratio=1.0)
    zernike_matrix_aux = zernike_matrices_aux[0]

    fig, axes = plt.subplots(3, 3)
    for k, i_ab in enumerate(ls_aberr):
        ax = axes.flatten()[k]
        ax.imshow(zernike_matrix_aux[:, :, i_ab], cmap='jet')
        ax.set_title(r'%d' % (i_ab + 1))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.show()

    def residuals(x_coef, PSF_zernike, PSF_zernike_anam):

        N_trials = 100
        mean_diff = []
        mean_fake = []
        # for k in range(1):

        rand_coef = 0.0*np.random.uniform(low=-0.15, high=0.15, size=N_zern)

        # Calculate the PSF for each model
        p, s = PSF_zernike.compute_PSF(rand_coef)
        p_anam, s = PSF_zernike_anam.compute_PSF(rand_coef)
        diff = p_anam - p
        mean_diff.append(diff)

        fake_psf, s = PSF_zernike.compute_PSF(rand_coef + x_coef)
        fake_diff = fake_psf - p
        mean_fake.append(fake_diff)

        avg_diff = np.mean(np.array(mean_diff), axis=0)
        # print(avg_diff.shape)
        avg_fake = np.mean(np.array(mean_fake), axis=0)

        # plt.figure()
        # plt.imshow(avg_diff, cmap='RdBu')
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(avg_fake, cmap='RdBu')
        # plt.colorbar()
        # plt.show()

        # the difference is what we should try to replicate with the Zernike aberrations
        residual = diff - fake_diff

        return residual.flatten()



    x0 = np.random.uniform(low=-0.01, high=0.01, size=N_zern)
    result = least_squares(fun=residuals, x0=x0, args=(PSF_zernike, PSF_zernike_anam),
                           loss='soft_l1', verbose=2)
    x_coef = result.x

    N_ls = 100
    ls_coefs = []
    for i in range(N_ls):
        # x0 = np.random.uniform(low=-0.05, high=0.05, size=N_zern)
        x0 = np.zeros(N_zern)
        rand_coef = np.random.uniform(low=-0.15, high=0.15, size=N_zern)
        result = least_squares(fun=residuals, x0=x0, args=(rand_coef, PSF_zernike, PSF_zernike_anam),
                               loss='soft_l1', verbose=2)
        x_coef = result.x
        ls_coefs.append(x_coef)

    # Show how well we match the variations with the fake aberrations
    p, s = PSF_zernike.compute_PSF(rand_coef)
    p_anam, s = PSF_zernike_anam.compute_PSF(rand_coef)
    diff = p_anam - p

    fake_psf, s = PSF_zernike.compute_PSF(rand_coef + x_coef)
    fake_diff = fake_psf - p

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

    # The Nominal PSF with some random aberrations
    img1 = ax1.imshow(p, cmap='plasma', extent=extent, origin='lower')
    plt.colorbar(img1, ax=ax1)
    ax1.set_title(r'Nominal PSF')

    # Same PSF (same aberrations) but the anamorphic error
    img2 = ax2.imshow(p_anam, cmap='plasma', extent=extent, origin='lower')
    plt.colorbar(img2, ax=ax2)
    ax2.set_title(r'Ratio $a/b=%.2f$' % anamorphic_ratio)

    img3 = ax3.imshow(diff, cmap='RdBu', extent=extent, origin='lower')
    cval = max(-np.min(diff), np.max(diff))
    img3.set_clim(-cval, cval)
    plt.colorbar(img3, ax=ax3)
    ax3.set_title(r'Difference Anamorphic - Nominal')

    # Now we do the result for the non-linear LS fit
    # Show the fake wavefront that matches
    fake_wavef = np.dot(PSF_zernike.model_matrix, x_coef)
    img4 = ax4.imshow(fake_wavef, cmap='jet', origin='lower', extent=[-1, 1, -1, 1])
    ax4.set_xlim([-RHO_APER, RHO_APER])
    ax4.set_ylim([-RHO_APER, RHO_APER])
    plt.colorbar(img4, ax=ax4)
    ax4.set_title(r'Fake wavefront')

    # Now show the PSF with the aberrations + the FAKE aberrations
    img5 = ax5.imshow(fake_psf, cmap='plasma', extent=extent, origin='lower')
    plt.colorbar(img5, ax=ax5)
    ax5.set_title(r'Nominal PSF + fake aberrations')

    # Show the difference
    img6 = ax6.imshow(fake_diff, cmap='RdBu', extent=extent, origin='lower')
    # img6.set_clim(-cval, cval)
    plt.colorbar(img6, ax=ax6)
    ax6.set_title(r'Difference Fake - Nominal')

    plt.show()


    # Show the aberrations that contributed most
    i_sort = np.argsort(np.abs(x_coef))[::-1]
    for k in range(5):

        # phase_ = np.dot(PSF_zernike.model_matrix, x_coef[k])
        plt.figure()
        plt.imshow(PSF_zernike.model_matrix[:, :, k], cmap='jet')



    phase_ = np.dot(PSF_zernike.model_matrix, result.x)

    # ================================================================================================================ #
    #                                   Impact on Machine Learning calibration
    # ================================================================================================================ #

    # REMEMBER to rescale to a sampling of 4.0 mas

    SPAX = 4.0  # mas | spaxel scale
    RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
    RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration

    pix = 16
    input_shape = (pix, pix, 2,)

    # (1) We begin by creating a Zernike PSF model with Defocus as diversity
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=N_levels, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.0,
                                                                          anamorphic_ratio=1.0)
    N_zern = zernike_matrix.shape[-1]
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))

    defocus_zernike = np.zeros(zernike_matrix.shape[-1])
    defocus_zernike[1] = diversity / (2 * np.pi)
    PSF_zernike.define_diversity(defocus_zernike)

    # C
    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                              coef_strength, rescale)

    # Train the Calibration Model on images with the nominal defocus
    epochs = 10
    calib_zern = calibration.Calibration(PSF_model=PSF_zernike)
    calib_zern.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='relu')
    losses = calib_zern.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                readout_noise=False, RMS_readout=[1. / SNR], readout_copies=0)

    # Now we test the performance on an anamorphic error
    ratio = 1.10
    zernike_matrix_anam, pupil_mask_zernike_anam, flat_zernike_anam = psf.zernike_matrix(N_levels=N_levels, rho_aper=RHO_APER,
                                                                                         rho_obsc=RHO_OBSC, N_PIX=N_PIX,
                                                                                         radial_oversize=1.0,
                                                                                         anamorphic_ratio=ratio)

    zernike_matrices_anam = [zernike_matrix_anam, pupil_mask_zernike_anam, flat_zernike_anam]
    PSF_zernike_anam = psf.PointSpreadFunction(matrices=zernike_matrices_anam, N_pix=N_PIX,
                                               crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))
    PSF_zernike_anam.define_diversity(defocus_zernike)

    _PSF, _coef, test_PSF_anam, test_coef_anam = calibration.generate_dataset(PSF_zernike_anam, 0, N_test,
                                                                              coef_strength, rescale)

    guess_coef = calib_zern.cnn_model.predict(test_PSF)
    residual_coef = test_coef - guess_coef
    print(np.mean(np.linalg.norm(residual_coef, axis=1)))

    guess_coef_anam = calib_zern.cnn_model.predict(test_PSF_anam)
    residual_coef_anam = test_coef_anam - guess_coef_anam
    print(np.mean(np.linalg.norm(residual_coef_anam, axis=1)))

    N_rows = 4
    N_cols = 8
    fig, axes = plt.subplots(N_rows, N_cols)
    deltas = []
    for k in range(N_rows * N_cols):
        coef_ = residual_coef[:, k]
        coef_anam = residual_coef_anam[:, k]
        mean_coef = np.mean(np.abs(coef_))
        mean_coef_anam = np.mean(np.abs(coef_anam))
        delta = mean_coef_anam - mean_coef
        print("%d | %.4f | %.4f | %.4f" % (k + 1, mean_coef, mean_coef_anam, delta))
        deltas.append(delta)

        ax = axes.flatten()[k]
        ax.hist(coef_, bins=25, alpha=0.5, color='lightgreen')
        ax.hist(coef_anam, bins=25, histtype='step', color='black')
        ax.set_xlim([-0.2, 0.2])
        ax.yaxis.set_visible(False)
        if k // N_cols != N_rows - 1:
            ax.xaxis.set_visible(False)
        ax.set_title('%d' % (k + 1))
        ax.set_xlabel('Residual [$\lambda$]')
    plt.show()

    mu, std = np.zeros(N_zern), np.zeros(N_zern)
    mu_anam, std_anam = np.zeros(N_zern), np.zeros(N_zern)
    for k in range(N_zern):
        coef_ = residual_coef[:, k]
        coef_anam = residual_coef_anam[:, k]
        mu[k] = np.mean(coef_)
        std[k] = np.std(coef_)
        mu_anam[k] = np.mean(coef_anam)
        std_anam[k] = np.std(coef_anam)

    delta_mu = np.abs(mu_anam - mu)
    delta_std = np.abs(std_anam - std) / 3
    c_ = delta_std / np.max(delta_std)
    colors = cm.Reds(c_)
    plt.figure()
    for i in range(N_zern):
    # plt.errorbar(np.arange(1, N_zern + 1), y=mu, yerr=std/2, fmt='o')
    #     plt.errorbar(np.arange(1, N_zern + 1) + 0.1, y=delta_mu, yerr=delta_std, fmt='o', c=colors)
        plt.errorbar(i + 1, y=delta_mu[i], yerr=delta_std[i], fmt='o', c=colors[i])
    # plt.gray()
    plt.xticks(np.arange(1, N_zern + 1))
    # plt.axhline(0, 0, linestyle='--', color='black')
    plt.xlabel(r'Zernike polynomial')
    plt.ylabel(r'Bias $|\mu_r - \mu| [\lambda]$')
    plt.ylim(bottom=0)
    plt.show()


    # How does the RMS after calibration change with the error in anamorphic scale?
    N_ratios = 10
    ratios = np.linspace(1.0, 1.5, N_ratios)
    MUS, STDS = [], []
    for k, ratio in enumerate(ratios):
        zernike_matrix_anam, pupil_mask_zernike_anam, flat_zernike_anam = psf.zernike_matrix(N_levels=N_levels, rho_aper=RHO_APER,
                                                                                             rho_obsc=RHO_OBSC, N_PIX=N_PIX,
                                                                                             radial_oversize=1.0,
                                                                                             anamorphic_ratio=ratio)

        zernike_matrices_anam = [zernike_matrix_anam, pupil_mask_zernike_anam, flat_zernike_anam]
        PSF_zernike_anam = psf.PointSpreadFunction(matrices=zernike_matrices_anam, N_pix=N_PIX,
                                                   crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))
        PSF_zernike_anam.define_diversity(defocus_zernike)

        _PSF, _coef, test_PSF_anam, test_coef_anam = calibration.generate_dataset(PSF_zernike_anam, 0, N_test,
                                                                                  coef_strength, rescale)

        guess_coef_anam = calib_zern.cnn_model.predict(test_PSF_anam)
        residual_coef_anam = test_coef_anam - guess_coef_anam
        RMS_after = np.zeros(N_test)
        for j in range(N_test):
            _wavef = np.dot(PSF_zernike.model_matrix, residual_coef_anam[j])
            RMS_after[j] = np.std(_wavef[PSF_zernike.pupil_mask])

        mean_rms = np.mean(RMS_after)
        MUS.append(mean_rms)
        std_rms = np.std(RMS_after)
        STDS.append(std_rms)

    plt.figure()
    plt.errorbar(x=ratios, y=MUS, yerr=STDS, fmt='o')
    plt.xlabel(r'Ratio [ ]')
    plt.ylabel(r'RMS after calibration [$\lambda$]')
    plt.show()


