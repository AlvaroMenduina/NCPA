"""

                               -||  CALIBRATION  ||-

Test / Tutorial on how to use Machine Learning to calibrate NCPA



Author: Alvaro Menduina
Date: Feb 2020
"""

import numpy as np
from numpy.linalg import norm
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
N_train, N_test = 10000, 1000       # Samples for the training of the models
coef_strength = 1 / (2 * np.pi)     # Strength of the actuator coefficients
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filers = [256, 128, 32, 8]    # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
epochs = 50                         # Training epochs


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Calculate the Actuator Centres
    centers = psf.actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act = len(centers[0])
    psf.plot_actuators(centers, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)

    plt.show()

    # Calculate the Actuator Model Matrix (Influence Functions)
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)

    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]
    diversity_actuators = 1 / (2 * np.pi) * np.random.uniform(-1, 1, size=N_act)

    # Create the PSF model using the Actuator Model for the wavefront
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=diversity_actuators)

    plt.show()

    # ================================================================================================================ #
    #                                         Machine Learning
    # ================================================================================================================ #

    # Generate training and test datasets (clean PSF images)
    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, 100, 100,
                                                                              coef_strength=0.25, rescale=0.35)

    # Check Defocus / Nominal ratio
    peaks_nom = np.max(train_PSF[:, :, :, 0], axis=(1, 2))
    peaks_foc = np.max(train_PSF[:, :, :, 1], axis=(1, 2))
    plt.figure()
    plt.plot(peaks_nom)
    plt.plot(peaks_foc)
    plt.show()

    # Initialize Convolutional Neural Network model for calibration
    calibration_model = calibration.create_cnn_model(layer_filers, kernel_size, input_shape,
                                                     N_classes=N_act, name='CALIBR', activation='relu')

    # Add some Readout Noise to the PSF images to spice things up
    SNR_READOUT = 750
    noise_model = noise.NoiseEffects()
    train_PSF_readout = noise_model.add_readout_noise(train_PSF, RMS_READ=1./SNR_READOUT)
    test_PSF_readout = noise_model.add_readout_noise(test_PSF, RMS_READ=1./SNR_READOUT)

    # Show some examples from the training set
    utils.plot_images(train_PSF_readout, N_images=3)
    plt.show()

    # Train the calibration model
    train_history = calibration_model.fit(x=train_PSF_readout, y=train_coef,
                                          validation_data=(test_PSF_readout, test_coef),
                                          epochs=epochs, batch_size=32, shuffle=True, verbose=1)

    # Evaluate performance
    guess_coef = calibration_model.predict(test_PSF_readout)
    residual_coef = test_coef - guess_coef
    norm_before = np.mean(norm(test_coef, axis=1))
    norm_after = np.mean(norm(residual_coef, axis=1))
    print("\nPerformance:")
    print("Average Norm Coefficients")
    print("Before: %.2f" % norm_before)
    print("After : %.2f" % norm_after)

    # Show an example of the Wavefront before and after calibration
    i_ex = 1
    wavefront_before = np.dot(PSF_actuators.model_matrix, test_coef[i_ex])
    RMS_before = WAVE * 1e3 * np.std(wavefront_before[pupil_mask])
    wavefront_after = np.dot(PSF_actuators.model_matrix, residual_coef[i_ex])
    RMS_after = WAVE * 1e3 * np.std(wavefront_after[pupil_mask])
    ph_min = min(np.min(wavefront_before), -np.max(wavefront_before))

    cmap = 'RdBu'
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = plt.subplot(1, 2, 1)
    img1 = ax1.imshow(wavefront_before, cmap=cmap, extent=[-1, 1, -1, 1])
    ax1.set_title(r'Before | RMS = %.1f nm' % RMS_before)
    ax1.set_xlim([-1.25*RHO_APER, 1.25*RHO_APER])
    ax1.set_ylim([-1.25*RHO_APER, 1.25*RHO_APER])
    img1.set_clim(ph_min, -ph_min)
    plt.colorbar(img1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 2, 2)
    img2 = ax2.imshow(wavefront_after, cmap=cmap, extent=[-1, 1, -1, 1])
    ax2.set_title(r'After | RMS = %.1f nm' % RMS_after)
    ax2.set_xlim([-1.25*RHO_APER, 1.25*RHO_APER])
    ax2.set_ylim([-1.25*RHO_APER, 1.25*RHO_APER])
    img2.set_clim(ph_min, -ph_min)
    plt.colorbar(img2, ax=ax2, orientation='horizontal')
    plt.show()

    # ================================================================================================================ #
    #      A little bit tidier and more abstract
    # ================================================================================================================ #

    # In case you need to reload a library after changing it
    import importlib
    importlib.reload(calibration)

    # Some Parameters
    SNR = 500
    N_train, N_test = 10000, 1000
    N_loops, epochs_loop = 5, 5
    readout_copies = 2
    N_iter = 3
    layer_filters, kernel_size = [64, 32, 16, 8], 3


    # Using the Calibration object from calibration.py

    diversity_actuators = 0.25 * np.random.uniform(-1, 1, size=N_act)

    # Create the PSF model using the Actuator Model for the wavefront
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=diversity_actuators)

    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                              coef_strength=0.30, rescale=0.35)

    # Check Defocus / Nominal ratio
    peaks_nom = np.max(train_PSF[:, :, :, 0], axis=(1, 2))
    peaks_foc = np.max(train_PSF[:, :, :, 1], axis=(1, 2))
    plt.figure()
    plt.plot(peaks_nom)
    plt.plot(peaks_foc)

    # Show some examples from the training set
    utils.plot_images(train_PSF, N_images=3)
    plt.show()

    calib = calibration.Calibration(PSF_model=PSF_actuators)
    calib.create_cnn_model(layer_filers, kernel_size, name='CALIBR', activation='relu')
    losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    RMS_evolution = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                               readout_noise=True, RMS_readout=1./SNR)

    calib.plot_RMS_evolution(RMS_evolution)
    plt.show()








