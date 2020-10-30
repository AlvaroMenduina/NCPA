"""



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
pix = 128                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength
SPAX = 4.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
print("Nominal Parameters | Spaxel Scale and Wavelength")
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
N_levels = 8                        # Number of Zernike radial levels levels
coef_strength = 0.22                # Strength of Zernike aberrations
diversity = 1.0                     # Strength of the Defocus diversity
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filters = [64, 32, 16, 8]      # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
SNR = 500                           # SNR for the Readout Noise
N_loops, epochs_loop = 5, 5         # How many times to loop over the training
readout_copies = 2                  # How many copies with Readout Noise to use
N_iter = 3                          # How many iterations to run the calibration (testing)

import importlib
importlib.reload(calibration)

if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    ### (0) Define a nominal PSF Zernike model with no anamorphic mag. Perfectly circular pupil
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

    ### (1) Define a PSF Zernike model with anamorphic ratio of 2:1
    anam_ratio = 0.5
    anamorphic_matrices = psf.zernike_matrix(N_levels=N_levels, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX,
                                             radial_oversize=1.0, anamorphic_ratio=anam_ratio)
    zernike_matrix_anam, pupil_mask_zernike_anam, flat_zernike_anam = anamorphic_matrices

    PSF_zernike_anam = psf.PointSpreadFunction(matrices=anamorphic_matrices, N_pix=N_PIX,
                                               crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))
    PSF_zernike_anam.define_diversity(defocus_zernike)

    p_nom, snom = PSF_zernike.compute_PSF(np.zeros(N_zern))
    p_anam, sanam = PSF_zernike_anam.compute_PSF(np.zeros(N_zern))

    cmap = 'plasma'
    extent = [-pix//2 * SPAX, pix//2 * SPAX, -pix//2 * SPAX, pix//2 * SPAX]
    extent_anam = [-pix//2 * SPAX, pix//2 * SPAX, -pix * SPAX, pix * SPAX]
    fig, (ax1, ax2) = plt.subplots(1, 2)

    img1 = ax1.imshow(p_nom, cmap=cmap, extent=extent)

    img2 = ax2.imshow(p_anam, cmap=cmap, extent=extent_anam)
    ax2.set_ylim([-pix//2 * SPAX, pix//2 * SPAX])

    plt.show()

    # ================================================================================================================ #
    #                                        FIELD OF VIEW investigation
    # ================================================================================================================ #
    # Does the number of pixels used to crop the images matter?
    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                              coef_strength, rescale)

    def crop_images(PSF_images, crop_pix):
        """
        Get some PSF datacubes and crop them
        :param PSF_images:
        :param crop_pix:
        :return:
        """

        N_PSF = PSF_images.shape[0]
        N_pix = PSF_images.shape[1]
        min_pix = N_pix//2 - crop_pix//2
        max_pix = N_pix//2 + crop_pix//2
        new_images = np.zeros((N_PSF, crop_pix, crop_pix, 2))

        for k in range(N_PSF):
            img_nom = PSF_images[k, :, :, 0]
            img_foc = PSF_images[k, :, :, 1]
            crop_nom = img_nom[min_pix:max_pix, min_pix:max_pix]
            crop_foc = img_foc[min_pix:max_pix, min_pix:max_pix]
            new_images[k, :, :, 0] = crop_nom
            new_images[k, :, :, 1] = crop_foc

        return new_images

    epochs = 4
    layer_filters = [32, 16]
    # SNR = 100
    crop_pixels = [64, 32, 16]
    N_crop = len(crop_pixels)
    N = np.arange(1, N_zern + 1)
    s = 20
    colors = cm.Reds(np.linspace(0.5, 1.0, N_crop))

    SNRs = [1e4, 1e3, 250, 100]
    fig, axes = plt.subplots(2, 2)
    for j in range(4):
        snr = SNRs[j]
        ax = axes.flatten()[j]

        stds_crop = np.zeros((N_crop, N_zern))
        for k, crop_pix in enumerate(crop_pixels):

            # modify the crop_pix attribute to get a proper input shape
            PSF_zernike.crop_pix = crop_pix
            calib_zern = calibration.Calibration(PSF_model=PSF_zernike)
            calib_zern.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='relu')

            # crop the PSF images
            train_PSF_crop = crop_images(train_PSF, crop_pix=crop_pix)
            test_PSF_crop = crop_images(test_PSF, crop_pix=crop_pix)

            # train and test the models on the crop images
            losses = calib_zern.train_calibration_model(train_PSF_crop, train_coef, test_PSF_crop, test_coef,
                                                        N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                        readout_noise=True, RMS_readout=[1. / snr], readout_copies=readout_copies)

            test_PSF_crop_noisy = calib_zern.noise_effects.add_readout_noise(test_PSF_crop, RMS_READ=1 / snr)

            guess_coef = calib_zern.cnn_model.predict(test_PSF_crop_noisy)
            residual_coef = test_coef - guess_coef
            print(np.mean(np.linalg.norm(residual_coef, axis=1)))

            # see if that had an influence on the RMS residual for each Zernike polynomial after 1 iteration
            stds = np.std(residual_coef, axis=0)
            stds_crop[k] = stds

            # Plot the RMS residual for each Zernike polynomial as a function of the crop_pix
            ax.scatter(N, stds_crop[k], s=s, color=colors[k], label='%d' % crop_pixels[k])
            ax.plot(N, stds_crop[k], color=colors[k])
        ax.legend(title=r'FoV [pix]', loc=4)
        ax.set_title(r'SNR: %d' % snr)
        ax.set_xlabel(r'Zernike polynomial')
        ax.set_ylabel(r'RMS residual coefficient [$\lambda$]')
        ax.set_xlim([1, N_zern])
        ax.set_ylim([0.02, 0.08])
    plt.show()

    # it seems like the crop_pix has little influence on the RMS of the low order Zernikes
    # but it definitely plays a role on the high order Zernikes, for which we want a wider FoV

    # ================================================================================================================ #
    #                       Influence of the ANAMORPHIC Magnification
    # ================================================================================================================ #


    # Put back the correct value
    pix = 64
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))

    defocus_zernike = np.zeros(zernike_matrix.shape[-1])
    defocus_zernike[1] = diversity / (2 * np.pi)
    PSF_zernike.define_diversity(defocus_zernike)

    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                              coef_strength, rescale)

    layer_filters = [32, 16]
    calib_zern = calibration.Calibration(PSF_model=PSF_zernike)
    calib_zern.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='relu')

    # # crop the PSF images
    # train_PSF_crop = crop_images(train_PSF, crop_pix=crop_pix)
    # test_PSF_crop = crop_images(test_PSF, crop_pix=crop_pix)

    # train and test the models on the crop images
    # this time do NOT include noise. we want to see just the effect of magnification
    epochs = 4
    SNR = 250
    losses = calib_zern.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                N_loops=3, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                plot_val_loss=False,
                                                readout_noise=True, RMS_readout=[1. / SNR],
                                                readout_copies=readout_copies)

    test_PSF_noise = calib_zern.noise_effects.add_readout_noise(test_PSF, RMS_READ=1 / SNR)
    guess_coef = calib_zern.cnn_model.predict(test_PSF_noise)
    residual_coef = test_coef - guess_coef
    print(np.mean(np.linalg.norm(residual_coef, axis=1)))
    stds = np.std(residual_coef, axis=0)

    # --------
    PSF_zernike_anam = psf.PointSpreadFunction(matrices=anamorphic_matrices, N_pix=N_PIX,
                                               crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))
    PSF_zernike_anam.define_diversity(defocus_zernike)

    # Now train a model on the anamorphic PSF images
    anam_datasets = calibration.generate_dataset(PSF_zernike_anam, N_train, N_test, coef_strength, rescale)
    train_PSF_anam, train_coef_anam, test_PSF_anam, test_coef_anam = anam_datasets

    # mask out the extra FOV?
    dpix = pix // 2
    # train_PSF_anam_m = np.zeros_like(train_PSF_anam)
    # for k in range(N_train):
    #
    #     copy_nom = train_PSF_anam[k, :, :, 0]
    #     copy_nom[:pix // 2 - dpix//2, :] *= 0
    #     copy_nom[pix // 2 + dpix//2:, :] *= 0
    #     train_PSF_anam_m[k, :, :, 0] = copy_nom
    #
    #     copy_foc = train_PSF_anam[k, :, :, 1]
    #     copy_foc[:pix // 2 - dpix//2, :] *= 0
    #     copy_foc[pix // 2 + dpix//2:, :] *= 0
    #     train_PSF_anam_m[k, :, :, 1] = copy_foc
    #
    # test_PSF_anam_m = np.zeros_like(test_PSF_anam)
    # for k in range(N_test):
    #     copy_nom = test_PSF_anam[k, :, :, 0]
    #     copy_nom[:pix // 2 - dpix // 2, :] *= 0
    #     copy_nom[pix // 2 + dpix // 2:, :] *= 0
    #     test_PSF_anam_m[k, :, :, 0] = copy_nom
    #
    #     copy_foc = test_PSF_anam[k, :, :, 1]
    #     copy_foc[:pix // 2 - dpix // 2, :] *= 0
    #     copy_foc[pix // 2 + dpix // 2:, :] *= 0
    #     test_PSF_anam_m[k, :, :, 1] = copy_foc


    # # Adjust the fov
    # dpix = pix // 2
    train_PSF_anam_p = train_PSF_anam[:, pix // 2 - dpix//2:pix // 2 + dpix//2, :]
    test_PSF_anam_p = test_PSF_anam[:, pix // 2 - dpix//2:pix // 2 + dpix//2, :]

    # layer_filters = [32, 16, 8]

    # Model Trained on N x N/2 images (constant FoV)
    calib_anam = calibration.Calibration(PSF_model=PSF_zernike_anam)
    calib_anam.create_cnn_model(layer_filters, kernel_size, name='ANAM_ZERN', activation='relu', anamorphic=True)
    losses = calib_anam.train_calibration_model(train_PSF_anam_p, train_coef_anam, test_PSF_anam_p, test_coef_anam,
                                                N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                readout_noise=True, RMS_readout=[1. / SNR], readout_copies=2)

    # Model Trained on N x N images with extra FoV in Y
    calib_anam_fov = calibration.Calibration(PSF_model=PSF_zernike_anam)
    calib_anam_fov.create_cnn_model(layer_filters, kernel_size, name='ANAM_ZERN', activation='relu', anamorphic=False)
    losses = calib_anam_fov.train_calibration_model(train_PSF_anam, train_coef_anam, test_PSF_anam, test_coef_anam,
                                                N_loops=3, epochs_loop=epochs, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    # # Generate the equivalent dataset using the same test_coef
    # test_PSF_anam = np.zeros_like(test_PSF)
    # for k in range(N_test):
    #     p_nom, s_nom = PSF_zernike_anam.compute_PSF(test_coef[k])
    #     p_foc, s_foc = PSF_zernike_anam.compute_PSF(test_coef[k], diversity=True)
    #     test_PSF_anam[k, :, :, 0] = p_nom
    #     test_PSF_anam[k, :, :, 1] = p_foc
    # test_PSF_anam_p = test_PSF_anam[:, pix // 2 - dpix // 2:pix // 2 + dpix // 2, :]

    # Model Trained on N x N/2 images (constant FoV)
    guess_coef_anam = calib_anam.cnn_model.predict(test_PSF_anam_p)
    residual_coef_anam = test_coef - guess_coef_anam
    print(np.mean(np.linalg.norm(residual_coef_anam, axis=1)))
    stds_anam = np.std(residual_coef_anam, axis=0)

    # Model Trained on N x N images with extra FoV in Y
    test_PSF_anam_noise = calib_anam_fov.noise_effects.add_readout_noise(test_PSF_anam, RMS_READ=1 / SNR)
    guess_coef_anam_fov = calib_anam_fov.cnn_model.predict(test_PSF_anam_noise)
    residual_coef_anam_fov = test_coef - guess_coef_anam_fov
    print(np.mean(np.linalg.norm(residual_coef_anam_fov, axis=1)))
    stds_anam_fov = np.std(residual_coef_anam_fov, axis=0)

    s = 20
    plt.figure()
    plt.scatter(N, stds, s=s, label='Nominal')
    plt.plot(N, stds)
    # plt.scatter(N, stds_anam, s=s, label=r'Anamorphic $N \times N/2$')
    # plt.plot(N, stds_anam)
    plt.scatter(N, stds_anam_fov, s=s, label=r'Anamorphic $N \times N$')
    plt.plot(N, stds_anam_fov)
    plt.legend(loc=4)
    plt.xlabel(r'Zernike polynomial')
    plt.ylabel(r'RMS residual coefficient [$\lambda$]')
    plt.xlim([1, N_zern])
    plt.show()

    # so it seems to affect the performance across all Zernike polynomials, not just the ones oriented in Y preferentially

    # Now the question is, what happens as a function of the iterations

    RMS_evo, res = calib_zern.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                   readout_noise=False, RMS_readout=1. / SNR)

    RMS_evo_anam, res_anam = calib_anam_fov.calibrate_iterations(test_PSF_anam, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                                 readout_noise=False, RMS_readout=1. / SNR)


    ###

    def iterate_calibrations(calib_model, PSF_model, test_images, test_coef, N_iter):

        strehl0 = np.max(test_images[:, :, :, 0], axis=(1, 2))
        test_images = calib_model.noise_effects.add_readout_noise(test_images, RMS_READ=1 / SNR)
        # Downsample the images

        PSF0 = test_images
        coef0 = test_coef

        strehls = [strehl0]
        PSF_evolution = [test_images]
        residuals = [coef0]

        for j in range(N_iter):

            print("\nIteration #%d" % (j + 1))
            guess_coef = calib_model.cnn_model.predict(PSF0)
            residual_coef = coef0 - guess_coef
            print(np.mean(np.linalg.norm(residual_coef, axis=1)))
            residuals.append(residual_coef)

            # update the PSF
            print("Updating the PSF images")
            new_PSF = np.zeros_like(test_images)
            for k in range(N_test):
                if k % 100 == 0:
                    print(k)
                new_PSF[k, :, :, 0], s = PSF_model.compute_PSF(residual_coef[k])
                new_PSF[k, :, :, 1], s = PSF_model.compute_PSF(residual_coef[k], diversity=True)
            new_PSF = calib_model.noise_effects.add_readout_noise(new_PSF, RMS_READ=1 / SNR)
            PSF_evolution.append(new_PSF)

            # calculate the Strehl
            new_strehl = np.max(new_PSF[:, :, :, 0], axis=(1, 2))
            strehls.append(new_strehl)

            PSF0 = new_PSF
            coef0 = residual_coef

        return PSF_evolution, residuals, strehls


    N_iter = 5
    PSF_evo, res, strehls = iterate_calibrations(calib_zern, PSF_zernike, test_PSF, test_coef, N_iter)
    PSF_evo_anam, res_anam, strehls_anam = iterate_calibrations(calib_anam_fov, PSF_zernike_anam, test_PSF_anam, test_coef, N_iter)

    print("Nominal: %.4f +- %.4f" % (np.mean(strehls[-1]), np.std(strehls[-1])))
    print("Anamorp: %.4f +- %.4f" % (np.mean(strehls_anam[-1]), np.std(strehls_anam[-1])))

    bins = np.linspace(0, 1.0, 50)
    plt.figure()
    plt.hist(strehls[-1], bins=bins, histtype='step')
    plt.hist(strehls_anam[-1], bins=bins, histtype='step')
    plt.show()
    ###


    N_rows = 5
    N_cols = 10
    xmax = 0.25
    bins = np.linspace(-xmax, xmax, 25)
    fig, axes = plt.subplots(N_rows, N_cols)
    deltas = []
    for k in range(N_rows * N_cols):

        ax = axes.flatten()[k]
        # ax.hist(test_coef[:, k], bins=bins, alpha=0.5, color='lightgreen', label='Initial')
        ax.hist(guess_coef[:, k], bins=bins, histtype='step', color='red', label='Nominal')
        ax.hist(guess_coef_anam[:, k], bins=bins, histtype='step', color='black', label='Anamorphic')
        ax.set_xlim([-xmax, xmax])
        ax.yaxis.set_visible(False)
        if k // N_cols != N_rows - 1:
            ax.xaxis.set_visible(False)
        ax.set_title('%d' % (k + 1))
        ax.set_xlabel('Coefficient [$\lambda$]')
        if k == 0:
            ax.legend(loc=3, facecolor='white', framealpha=1)
    plt.show()

    r = np.arange(1, N_test + 1)
    k = -2
    plt.scatter(test_coef[:, k], guess_coef[:, k], s=5)
    plt.scatter(test_coef[:, k], guess_coef_anam[:, k], s=5)
    plt.show()




