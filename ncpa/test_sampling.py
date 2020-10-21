"""

                          -||  SPAXEL SAMPLING EFFECTS  ||-



Author: Alvaro Menduina
Date: Oct 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import norm

import psf
import utils
import calibration


# PSF bits
N_PIX = 256                         # pixels for the Fourier arrays
pix = 100                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
N_levels = 10                        # Number of Zernike radial levels levels
coef_strength = 0.175                # Strength of Zernike aberrations
diversity = 1.0                     # Strength of the Defocus diversity
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filters = [64, 32, 16]      # How many filters per layer
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

    ### (0) Define a model with a sampling of 4.0 mas Spaxels

    ### PSF sampled with 2.0 mas spaxels
    SPAX2 = 2.0  # mas | spaxel scale
    RHO_APER2 = utils.rho_spaxel_scale(spaxel_scale=SPAX2, wavelength=WAVE)
    RHO_OBSC2 = 0.30 * RHO_APER2  # ELT central obscuration

    zernike_matrix2, pupil_mask_zernike2, flat_zernike2 = psf.zernike_matrix(N_levels=N_levels, rho_aper=RHO_APER2,
                                                                          rho_obsc=RHO_OBSC2, N_PIX=N_PIX,
                                                                          radial_oversize=1.0, anamorphic_ratio=1.0)
    N_zern = zernike_matrix2.shape[-1]

    zernike_matrices2 = [zernike_matrix2, pupil_mask_zernike2, flat_zernike2]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices2, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(N_zern))

    defocus_zernike = np.zeros(N_zern)
    defocus_zernike[1] = diversity / (2 * np.pi)
    PSF_zernike.define_diversity(defocus_zernike)

    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                              coef_strength=coef_strength, rescale=rescale)

    plt.figure()
    plt.imshow(train_PSF[0, :, :, 0], cmap='plasma')
    plt.colorbar()
    plt.show()

    from scipy.ndimage import zoom
    def downsample(PSF_images, spax0, new_spax):
        print("\nDownsampling PSF images")

        N = int(new_spax // spax0)
        N_PSF = PSF_images.shape[0]
        pix = PSF_images.shape[1]
        new_pix = pix // N
        new_PSF_images = np.zeros((N_PSF, new_pix, new_pix, 2))
        for k in range(N_PSF):
            new_PSF_images[k, :, :, 0] = zoom(PSF_images[k, :, :, 0], zoom=1/N)
            new_PSF_images[k, :, :, 1] = zoom(PSF_images[k, :, :, 1], zoom=1/N)

        return new_PSF_images

    train_PSF4 = downsample(train_PSF, spax0=SPAX2, new_spax=4.0)
    spaxels = [4.0, 10.0, 20.0]

    fig, axes = plt.subplots(1, 3)
    for k in range(3):
        ax = axes[k]
        _psf = downsample(train_PSF, spax0=SPAX2, new_spax=spaxels[k])
        img = ax.imshow(_psf[-1, :, :, 0], cmap='plasma', origin='lower')
        plt.colorbar(img, ax=ax, orientation='horizontal')
        ax.set_title(r'%.1f mas spaxels' % (spaxels[k]))

    plt.show()


    train_PSF4 = downsample(train_PSF, spax0=SPAX2, new_spax=4)
    test_PSF4 = downsample(test_PSF, spax0=SPAX2, new_spax=4)

    pix4 = int(pix * SPAX2 / 4)
    input_shape4 = (pix4, pix4, 2)
    # Initialize Convolutional Neural Network model for calibration
    calib_model4 = calibration.create_cnn_model(layer_filters, kernel_size, input_shape4,
                                                N_classes=N_zern, name='4MAS', activation='relu')

    # Train the calibration model
    train_history = calib_model4.fit(x=train_PSF4, y=train_coef, validation_data=(test_PSF4, test_coef),
                                     epochs=10, batch_size=32, shuffle=True, verbose=1)

    guess_coef4 = calib_model4.predict(test_PSF4)
    residual_coef4 = test_coef - guess_coef4
    norm_before = np.mean(norm(test_coef, axis=1))
    norm_after4 = np.mean(norm(residual_coef4, axis=1))
    print("\nPerformance:")
    print("Average Norm Coefficients")
    print("Before: %.2f" % norm_before)
    print("After : %.2f" % norm_after4)

    # [10x10]
    train_PSF10 = downsample(train_PSF, spax0=SPAX2, new_spax=10)
    test_PSF10 = downsample(test_PSF, spax0=SPAX2, new_spax=10)

    pix10 = int(pix * SPAX2 / 10)
    input_shape10 = (pix10, pix10, 2)
    # Initialize Convolutional Neural Network model for calibration
    calib_model10 = calibration.create_cnn_model(layer_filters, kernel_size, input_shape10,
                                                N_classes=N_zern, name='10MAS', activation='relu')

    train_history = calib_model10.fit(x=train_PSF10, y=train_coef, validation_data=(test_PSF10, test_coef),
                                     epochs=10, batch_size=32, shuffle=True, verbose=1)

    guess_coef10 = calib_model10.predict(test_PSF10)
    residual_coef10 = test_coef - guess_coef10
    norm_before = np.mean(norm(test_coef, axis=1))
    norm_after10 = np.mean(norm(residual_coef10, axis=1))
    print("\nPerformance:")
    print("Average Norm Coefficients")
    print("Before: %.2f" % norm_before)
    print("After : %.2f" % norm_after10)

    # [20x20]
    train_PSF20 = downsample(train_PSF, spax0=SPAX2, new_spax=20)
    test_PSF20 = downsample(test_PSF, spax0=SPAX2, new_spax=20)

    pix20 = int(pix * SPAX2 / 20)
    input_shape20 = (pix20, pix20, 2)
    # Initialize Convolutional Neural Network model for calibration
    calib_model20 = calibration.create_cnn_model(layer_filters, kernel_size, input_shape20,
                                                N_classes=N_zern, name='20MAS', activation='relu')

    train_history = calib_model20.fit(x=train_PSF20, y=train_coef, validation_data=(test_PSF20, test_coef),
                                     epochs=10, batch_size=32, shuffle=True, verbose=1)

    guess_coef20 = calib_model20.predict(test_PSF20)
    residual_coef20 = test_coef - guess_coef20
    norm_before = np.mean(norm(test_coef, axis=1))
    norm_after20 = np.mean(norm(residual_coef20, axis=1))
    print("\nPerformance:")
    print("Average Norm Coefficients")
    print("Before: %.2f" % norm_before)
    print("After : %.2f" % norm_after20)

    def iterate_calibrations(calib_model, test_images, test_coef, spaxel, N_iter):

        strehl0 = np.max(test_images[:, :, :, 0], axis=(1, 2))
        # Downsample the images
        test_PSF_down = downsample(test_images, spax0=SPAX2, new_spax=spaxel)

        PSF0 = test_PSF_down
        coef0 = test_coef

        strehls = [strehl0]
        PSF_evolution = [test_images]
        residuals = [coef0]

        for j in range(N_iter):

            print("\nIteration #%d" % (j + 1))

            # predict the aberrations on the downsampled images
            guess_coef = calib_model.predict(PSF0)
            residual_coef = coef0 - guess_coef
            residuals.append(residual_coef)

            # update the nominal PSF (2 mas spaxels)
            print("Updating the PSF images")
            new_PSF = np.zeros_like(test_images)
            for k in range(N_test):
                if k % 100 == 0:
                    print(k)
                new_PSF[k, :, :, 0], s = PSF_zernike.compute_PSF(residual_coef[k])
                new_PSF[k, :, :, 1], s = PSF_zernike.compute_PSF(residual_coef[k], diversity=True)
            PSF_evolution.append(new_PSF)

            # calculate the Strehl
            new_strehl = np.max(new_PSF[:, :, :, 0], axis=(1, 2))
            strehls.append(new_strehl)

            # downsample to match the scale
            PSF0 = downsample(new_PSF, spax0=SPAX2, new_spax=spaxel)
            coef0 = residual_coef

        return PSF_evolution, residuals, strehls


    N_iter = 3
    PSF_evolution4, residuals4, strehls4 = iterate_calibrations(calib_model=calib_model4, test_images=test_PSF,
                                                                test_coef=test_coef, spaxel=4.0, N_iter=N_iter)
    PSF_evolution10, residuals10, strehls10 = iterate_calibrations(calib_model=calib_model10, test_images=test_PSF,
                                                                test_coef=test_coef, spaxel=10.0, N_iter=N_iter)
    PSF_evolution20, residuals20, strehls20 = iterate_calibrations(calib_model=calib_model20, test_images=test_PSF,
                                                                test_coef=test_coef, spaxel=20.0, N_iter=N_iter)
    print("\nStrehl ratios")
    print("    4 mas    |    10 mas    |    20 mas   ")
    for k in range(N_iter + 1):
        mu_strehl4, std_strehl4 = np.mean(strehls4[k]), np.std(strehls4[k])
        mu_strehl10, std_strehl10 = np.mean(strehls10[k]), np.std(strehls10[k])
        mu_strehl20, std_strehl20 = np.mean(strehls20[k]), np.std(strehls20[k])
        print("%.2f +- %.2f | %.2f +- %.2f | %.2f +- %.2f" % (mu_strehl4, std_strehl4,
                                                              mu_strehl10, std_strehl10,
                                                              mu_strehl20, std_strehl20))


    reds = cm.Reds(np.linspace(0.25, 1.0, (N_iter + 1)))
    blues = cm.Blues(np.linspace(0.25, 1.0, (N_iter + 1)))
    greens = cm.Greens(np.linspace(0.25, 1.0, (N_iter + 1)))
    colors = [reds, blues, greens]
    bins = np.linspace(0, 1.0, 20)

    strehls = [strehls4, strehls10, strehls20]
    fig, axes = plt.subplots(1, 3)
    for j in range(3):
        s_data = strehls[j]
        ax = axes[j]

        ax.hist(s_data[0], bins=bins, alpha=0.5, color=colors[j][0], label='0')
        for k in np.arange(1, N_iter + 1):
            ax.hist(s_data[k], bins=bins, histtype='step', color=colors[j][k], label='%d' % (k))

        ax.legend(title='Iteration', loc=2)
        ax.set_title('%d mas spaxels' % (spaxels[j]))
        ax.set_xlabel(r'Strehl ratio [ ]')
        ax.set_xlim([0, 1])

    plt.show()


    N_rows = 4
    N_cols = 8
    xmax = 0.25
    bins = np.linspace(-xmax, xmax, 25)
    fig, axes = plt.subplots(N_rows, N_cols)
    deltas = []
    for k in range(N_rows * N_cols):

        ax = axes.flatten()[k]
        ax.hist(test_coef[:, k], bins=bins, alpha=0.5, color='lightgreen', label='Initial')
        # ax.hist(residual_coef4[:, k], bins=bins, histtype='step', color='blue', label='4 mas')
        ax.hist(residuals10[-1][:, k], bins=bins, histtype='step', color='black', label='10 mas')
        # ax.hist(residuals20[-1][:, k], bins=bins, histtype='step', color='red', label='20 mas')
        ax.set_xlim([-xmax, xmax])
        ax.yaxis.set_visible(False)
        if k // N_cols != N_rows - 1:
            ax.xaxis.set_visible(False)
        ax.set_title('%d' % (k + 1))
        ax.set_xlabel('Coefficient [$\lambda$]')
        if k == 0:
            ax.legend(loc=3, facecolor='white', framealpha=1)
    plt.show()

    # Show a comparison of the PSF through the iterations
    j_img = 2
    fig, axes = plt.subplots(N_iter + 1, 3)
    PSF_evos = [PSF_evolution4, PSF_evolution10, PSF_evolution20]
    for k in range(3):
        for j in range(N_iter + 1):
            ax = axes[j][k]
            psf_ = PSF_evos[k][j][j_img, :, :, 0]
            psf_z = psf_[pix//2 - 25: pix//2 + 25, pix//2 - 25: pix//2 + 25]
            s = np.max(psf_)
            img = ax.imshow(psf_z, cmap='plasma', origin='lower')
            # img.set_clim(0, 1.0)
            plt.colorbar(img, ax=ax)
            ax.set_title(r'%.1f mas | Strehl %.2f' % (spaxels[k], s))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            ax.set_ylabel(r"Iteration #%d" % (j))
    plt.show()



    for k in range(3):

        ax0 = axes[0][k]
        psf_0 = test_PSF[j_img, :, :, 0]
        s0 = np.max(psf_0)
        img0 = ax0.imshow(psf_0, cmap='plasma', origin='lower')
        # img.set_clim(0, 1.0)
        plt.colorbar(img0, ax=ax0)
        ax0.set_title(r'%.1f mas | Strehl %.2f' % (spaxels[k], s0))

        ax1 = axes[1][k]
        psf_1 = PSF_evolution4[k + 1][j_img, :, :, 0]
        s1 = np.max(psf_1)
        img1 = ax1.imshow(psf_1, cmap='plasma', origin='lower')
        # img.set_clim(0, 1.0)
        plt.colorbar(img1, ax=ax1)
        ax1.set_title(r'%.1f mas | Strehl %.2f' % (spaxels[k], s1))

        # 2nd iteration
        ax2 = axes[2][k]
        psf_2 = PSF_evolution10[k + 1][j_img, :, :, 0]
        s2 = np.max(psf_2)
        img2 = ax2.imshow(psf_2, cmap='plasma', origin='lower')
        # img.set_clim(0, 1.0)
        plt.colorbar(img2, ax=ax2)
        ax2.set_title(r'%.1f mas | Strehl %.2f' % (spaxels[k], s2))

        # 3rd iteration
        ax3 = axes[3][k]
        psf_3 = PSF_evolution20[k + 1][j_img, :, :, 0]
        s3 = np.max(psf_3)
        img3 = ax3.imshow(psf_3, cmap='plasma', origin='lower')
        # img.set_clim(0, 1.0)
        plt.colorbar(img3, ax=ax3)
        ax3.set_title(r'%.1f mas | Strehl %.2f' % (spaxels[k], s3))

    for ax in axes.flatten():
        ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
    for k in range(4):
        ax = axes[k][0]
        ax.set_ylabel(r'Iteration #%d' % (k))


    plt.show()



    zernike_matrices_aux = psf.zernike_matrix(N_levels=N_levels, rho_aper=1.0, rho_obsc=0.0, N_PIX=N_PIX,
                                               radial_oversize=1.0, anamorphic_ratio=1.0)
    zernike_matrix_aux = zernike_matrices_aux[0]

    fig, axes = plt.subplots(N_rows, N_cols)
    for k in range(N_rows * N_cols):
        ax = axes.flatten()[k]
        ax.imshow(zernike_matrix_aux[:, :, k], cmap='jet')
        ax.set_title(r'%d' % (k + 1))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.show()

    std20 = np.std(guess_coef20, axis=0)
    i_sort = np.argsort(std20)

    for k in i_sort[:5]:
        plt.figure()
        plt.imshow(zernike_matrix_aux[:, :, k], cmap='jet')











