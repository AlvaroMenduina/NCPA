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


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    ### ============================================================================================================ ###
    #                                   Let's start with a simple Zernike model
    ### ============================================================================================================ ###

    # Create a random wavefront [Zernike Model]
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=5, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.1)
    # Remember to oversize a little bit to avoid divergence at the pupil edge
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    # Create a Zernike PSF model
    N_zern = zernike_matrix.shape[-1]
    zernike_defocus = np.zeros(N_zern)
    zernike_defocus[1] = diversity      # Zernike Defocus
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=zernike_defocus)

    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                              coef_strength, rescale)
    utils.plot_images(train_PSF)

    mu_zern, std_zern = [], []
    zernike_levels = np.arange(4, 15)       # How many Zernike radial levels to use


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

    print("\nLooping over Zernike levels")
    for zern_level in zernike_levels:

        print("\nZernike Radial Level: ", zern_level)
        zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=zern_level, rho_aper=RHO_APER,
                                                                              rho_obsc=RHO_OBSC,
                                                                              N_PIX=N_PIX, radial_oversize=1.1)
        zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
        N_zern = zernike_matrix.shape[-1]
        print("Using a total of %d Zernike polynomials" % N_zern)
        zernike_defocus = np.zeros(N_zern)
        zernike_defocus[1] = diversity
        PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                              crop_pix=pix, diversity_coef=zernike_defocus)


        try:
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

        calib = calibration.Calibration(PSF_model=PSF_zernike)
        calib.create_cnn_model(layer_filters, kernel_size, name='ZERN_%d' % zern_level, activation='relu')
        losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                               N_loops, epochs_loop, verbose=1, batch_size_keras=32,
                                               plot_val_loss=False,
                                               readout_noise=True, RMS_readout=[1. / SNR],
                                               readout_copies=readout_copies)

        RMS_evolution, _residual = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                   readout_noise=True, RMS_readout=1. / SNR)

        mu_zern.append(np.mean(RMS_evolution[-1][-1]))
        std_zern.append(np.std(RMS_evolution[-1][-1]))

        calib_title = r'Zernike Polynomials: %d' % N_zern
        calib.plot_RMS_evolution(RMS_evolution, colormap=cm.Blues, title=calib_title)



