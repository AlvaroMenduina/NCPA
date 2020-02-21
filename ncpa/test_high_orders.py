"""

                          -||  HIGH ORDERS  ||-


Calibrating many Zernike polynomials with a single network can be a problem

(1) Let's see how the performance varies with the number of Zernikes we use

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
import zernike

utils.print_title(message='\nN C P A', font=None, random_font=False)

# PSF bits
N_PIX = 256                         # pixels for the Fourier arrays
pix = 32                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength

SPAX = 4.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
print("Nominal Parameters | Spaxel Scale and Wavelength")
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
coef_strength = 0.35                # Strength of the actuator coefficients
diversity = 0.55                    # Strength of extra diversity commands
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filters = [64, 32, 16, 8]     # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
SNR = 500                           # SNR for the Readout Noise
N_loops, epochs_loop = 2, 5         # How many times to loop over the training
readout_copies = 2                  # How many copies with Readout Noise to use
N_iter = 3                          # How many iterations to run the calibration (testing)

import importlib
importlib.reload(calibration)

directory = os.path.join(os.getcwd(), 'High_Orders')


def rescale_coefficients_for_strehl(PSF_model, coef_strength, rescale, min_strehl=0.25, a=0.5, b=0.75):
    # Calculate dummy datasets
    a_PSF, _coef, _PSF, __coef = calibration.generate_dataset(PSF_model, 99, 1,
                                                              a * coef_strength, a * rescale)
    b_PSF, _coef, _PSF, __coef = calibration.generate_dataset(PSF_model, 99, 1,
                                                              b * coef_strength, b * rescale)

    # Calculate peaks of training sets
    peaks_a = np.max(a_PSF[:, :, :, 0], axis=(1, 2))
    peaks_b = np.max(b_PSF[:, :, :, 0], axis=(1, 2))
    # Calculate the average minimum Strehl
    min_a = np.mean(np.sort(peaks_a)[:5])
    min_b = np.mean(np.sort(peaks_b)[:5])

    # Calculate rescaling of coefficients necessary to have a min_strehl
    ab = b - a
    dstrehl = min_b - min_a
    ds = min_strehl - min_a
    da = ab * ds / dstrehl

    return a + da


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    ### ============================================================================================================ ###
    #               How does the number of Zernike polynomials affect the performance?
    ### ============================================================================================================ ###

    # more Zernike coefficients to guess -> bigger challenge for the calibration network

    zernike_levels = np.arange(4, 16)       # How many Zernike radial levels to use
    N_levels = zernike_levels.shape[0]
    zernike_dict = zernike.triangular_numbers(N_levels=16)
    total_poly = [zernike_dict[key] - 3 for key in zernike_levels]
    # DO NOT FORGET to remove Piston and Tilts (3) from the total

    # Let's loop over the Zernike pyramid, training models to predict Zernikes
    # "up to order X", and see if at some point the performance degrades

    N_iter = 4
    RMS_zern = []
    mu_zern, std_zern = np.zeros(N_levels), np.zeros(N_levels)

    print("\nLooping over Zernike levels")
    for i, zern_level in enumerate(zernike_levels):

        print("\nZernike Radial Level: ", zern_level, i)
        zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=zern_level, rho_aper=RHO_APER,
                                                                              rho_obsc=RHO_OBSC,
                                                                              N_PIX=N_PIX, radial_oversize=1.1)
        zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
        N_zern = zernike_matrix.shape[-1]
        print("Using a total of %d Zernike polynomials" % N_zern)       # Total number of polynomials
        zernike_defocus = np.zeros(N_zern)
        zernike_defocus[1] = diversity
        # PSF model with Zernikes up to a certain order
        PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                              crop_pix=pix, diversity_coef=zernike_defocus)

        try:    # See if we already have the data saved
            train_PSF = np.load(os.path.join(directory, 'train_PSF_%d.npy' % zern_level))
            train_coef = np.load(os.path.join(directory, 'train_coef_%d.npy' % zern_level))
            test_PSF = np.load(os.path.join(directory, 'test_PSF_%d.npy' % zern_level))
            test_coef = np.load(os.path.join(directory, 'test_coef_%d.npy' % zern_level))

        except:

            # remember to rescale the coefficients. Otherwise the PSF images degrade more because of the fact that
            # there are many more Zernikes
            # we want to rescale the coefficients to ensure the minimum Strehl in the training set is sufficiently large
            new_scale = rescale_coefficients_for_strehl(PSF_zernike, coef_strength, rescale)
            train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                                      new_scale * coef_strength,
                                                                                      new_scale * rescale)

            np.save(os.path.join(directory, 'train_PSF_%d' % zern_level), train_PSF)
            np.save(os.path.join(directory, 'train_coef_%d' % zern_level), train_coef)
            np.save(os.path.join(directory, 'test_PSF_%d' % zern_level), test_PSF)
            np.save(os.path.join(directory, 'test_coef_%d' % zern_level), test_coef)

        # Train a calibration model on the PSF images
        calib = calibration.Calibration(PSF_model=PSF_zernike)
        calib.create_cnn_model(layer_filters, kernel_size, name='ZERN_%d' % zern_level, activation='relu')
        losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                               N_loops, epochs_loop, verbose=1, batch_size_keras=32,
                                               plot_val_loss=False, readout_noise=True, RMS_readout=[1. / SNR],
                                               readout_copies=readout_copies)

        RMS_evolution, _residual = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                              readout_noise=True, RMS_readout=1. / SNR)
        RMS_zern.append(RMS_evolution)

        mu_zern[i] = np.mean(RMS_evolution[-1][-1])
        std_zern[i] = np.std(RMS_evolution[-1][-1])

        calib_title = r'Zernike Polynomials: %d' % N_zern
        calib.plot_RMS_evolution(RMS_evolution, colormap=cm.Blues, title=calib_title)

    plt.figure()
    plt.errorbar(zernike_levels[:12], mu_zern[:12], yerr=std_zern[:12], fmt='o')
    plt.xlabel(r'Zernike radial order')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.ylim(bottom=0)
    plt.xlim([0, 16])
    plt.show()

    colors = cm.Reds(np.linspace(0.5, 1.0, N_iter + 1))
    plt.figure()
    k = 0
    for rms_evo, n_poly in zip(RMS_zern, total_poly):
        rms0 = rms_evo[0][0]
        mu0, std0 = np.mean(rms0), np.std(rms0)
        label = '0' if k == 0 else None
        plt.errorbar([n_poly], mu0, yerr=std0/2, fmt='o', label=label, color=colors[0])
        for i, (before, after) in enumerate(rms_evo):
            mu_after, std_after = np.mean(after), np.std(after)
            label = '%d' % (i + 1) if k == 0 else None
            plt.errorbar([n_poly], mu_after, yerr=std_after/2,  fmt='o', label=label, color=colors[i + 1], markersize=5)
        k += 1
    plt.legend(title='Iteration')
    plt.xlim([0, 125])
    plt.ylim([0, 250])
    plt.xlabel(r'Total Number of Zernike Polynomials')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.show()

    ### ============================================================================================================ ###
    #                                  AUTOENCODERS
    ### ============================================================================================================ ###

    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=14, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.1)
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    N_zern = zernike_matrix.shape[-1]
    print("Using a total of %d Zernike polynomials" % N_zern)  # Total number of polynomials
    zernike_defocus = np.zeros(N_zern)
    zernike_defocus[1] = diversity
    # PSF model with Zernikes up to a certain order
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=zernike_defocus)

    coef_ae = 0.15
    N_autoencoders = 2
    encoder_filters, decoder_filters = [256], [256]
    calib_ae = calibration.CalibrationAutoencoder(PSF_zernike, N_autoencoders, encoder_filters, decoder_filters,
                                                  kernel_size, name="AE", activation='relu', loss='binary_crossentropy',
                                                  load_directory=None)
    N_train = 25000
    N_test = 1000
    SNR = 500

    # Generate the images [noisy, clean_1, ..., clean_N_autoencoders] and coefficients
    images_ae, coeffs_ae = calib_ae.generate_datasets_autoencoder(N_train, N_test, coef_ae, rescale)

    # # Sanity check on the Strehl ratios
    # for k in range(N_autoencoders + 1):
    #     peaks_nom = np.max(images_ae[k][:,:,:,0], axis=(1, 2))
    #     peaks_foc = np.max(images_ae[k][:,:,:,1], axis=(1, 2))
    #     print(np.sort(peaks_nom))
    #     # print(np.sort(peaks_foc))

    # Save the images for later
    for i, (images, coeff) in enumerate(zip(images_ae, coeffs_ae)):

        np.save(os.path.join(directory, 'images_%d' % i), images)
        np.save(os.path.join(directory, 'coeffs_%d' % i), coeff)

    # Load the good stuff
    _images, _coefs = [], []
    for i in range(N_autoencoders + 1):
        _images.append(np.load(os.path.join(directory, 'images_%d.npy' % i)))
        _coefs.append(np.load(os.path.join(directory, 'coeffs_%d.npy' % i)))

    # Train the AUTOENCODERS
    calib_ae.train_autoencoder_models(datasets=_images, N_train=N_train, N_test=N_test,
                                      N_loops=5, epochs_per_loop=5,
                                      readout_noise=True, readout_copies=2, RMS_readout=1./SNR,
                                      save_directory=directory)

    # Clean the training images with the AUTOENCODERS to use them as training for the CALIBRATION model
    data_images = []
    for i in range(N_autoencoders):
        clean = _images[0]
        noisy = calib_ae.noise_effects.add_readout_noise(clean, RMS_READ=1./SNR)
        predicted = calib_ae.autoencoder_models[i].predict(noisy)
        data_images.append(predicted)

    # Train the CALIBRATION models
    calib_ae.create_calibration_models(layer_filters=[64, 32, 16, 8], kernel_size=kernel_size,
                                       name='CALIB', activation='relu', load_directory=None)

    # We shouldn't add Readout Noise because the Autoencoders already see the Noisy Images!!
    # and supposedly clean them too
    calib_ae.train_calibration_models(data_images, coeffs_ae[1:], N_train, N_test,
                                      N_loops=3, epochs_loop=epochs_loop, verbose=1, batch_size_keras=32,
                                      plot_val_loss=False, readout_noise=False, RMS_readout=None, readout_copies=None,
                                      save_directory=directory)


    # Extract the Test Sets
    _cut_coef = [c[N_train:] for c in coeffs_ae]
    all_coef = [np.concatenate(_cut_coef, axis=-1)]
    all_coef = all_coef + _cut_coef
    test_images = [img[N_train:] for img in _images]
    # Iterate over the calibration
    RMS_evo_ae, residual_ae = calib_ae.calibrate_iterations_autoencoder(test_images=test_images, test_coefs=all_coef,
                                                                        N_iter=3, wavelength=WAVE,
                                                                        readout_noise=True, RMS_readout=1./SNR)



    ### Old
    encoded = calib_ae.encoders[0].predict(calib_ae.noise_effects.add_readout_noise(images_ae[0], RMS_READ=1./500))

    calib_ae.validation(datasets=images_ae, N_train=N_train, N_test=N_test, RMS_readout=1./500, k_image=-1)
    plt.show()

    guess_coef = calib_ae.calibration_models[1].cnn_model.predict(data_images[1])
    resid_coef = coeffs_ae[1] - guess_coef

    rms, _res = calib_ae.calibration_models[0].calibrate_iterations(data_images[0][N_train:], coeffs_ae[0][N_train:],
                                                        wavelength=WAVE, N_iter=3)






