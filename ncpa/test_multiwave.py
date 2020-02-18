"""

                          -||  MULTIWAVE  ||-

Using multiple wavelength channels to improve the calibration

Author: Alvaro Menduina
Date: Feb 2020
"""

### Super useful code to display variables and their sizes (helps you clear the RAM)
# import sys
#
# for var, obj in locals().items():
#     print(var, sys.getsizeof(obj))

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm

import psf
import utils
import calibration

utils.print_title(message='\nN C P A', font=None, random_font=False)

print("\n           -||  MULTIWAVE  ||- ")
print("\nCan we use multiple wavelength channels to improve the calibration?\n")

# PSF bits
N_PIX = 256                         # pixels for the Fourier arrays
pix = 25                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength
SPAX = 4.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
print("Nominal Parameters | Spaxel Scale and Wavelength")
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

N_actuators = 20                    # Number of actuators in [-1, 1] line
alpha_pc = 10                       # Height [percent] at the neighbour actuator (Gaussian Model)


WAVE0 = WAVE                        # Minimum wavelength
WAVEN = 2.00                        # Maximum wavelength
N_WAVES = 5                         # How many wavelength channels to consider

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
coef_strength = 0.30                # Strength of the actuator coefficients
diversity = 0.55                    # Strength of extra diversity commands
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filers = [64, 32, 16, 8]      # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
SNR = 750                           # SNR for the Readout Noise
N_loops, epochs_loop = 5, 5         # How many times to loop over the training
readout_copies = 2                  # How many copies with Readout Noise to use
N_iter = 3                          # How many iterations to run the calibration (testing)

import importlib
importlib.reload(calibration)

if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    centers_multiwave = psf.actuator_centres_multiwave(N_actuators=N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                       N_waves=N_WAVES, wave0=WAVE0, waveN=WAVEN, wave_ref=WAVE)
    N_act = len(centers_multiwave[0][0])

    actuator_matrices = psf.actuator_matrix_multiwave(centres=centers_multiwave, alpha_pc=alpha_pc, rho_aper=RHO_APER,
                                                      rho_obsc=RHO_OBSC, N_waves=N_WAVES,  wave0=WAVE0, waveN=WAVEN,
                                                      wave_ref=WAVE, N_PIX=N_PIX)

    waves = np.linspace(WAVE0, WAVEN, N_WAVES, endpoint=True)
    waves_ratio = waves / WAVE

    for i, wave_r in enumerate(waves_ratio):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        circ1 = Circle((0, 0), RHO_APER/wave_r, linestyle='--', fill=None)
        circ2 = Circle((0, 0), RHO_OBSC/wave_r, linestyle='--', fill=None)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        for c in centers_multiwave[i][0]:
            ax.scatter(c[0], c[1], color='red', s=10)
            ax.scatter(c[0], c[1], color='black', s=10)
        ax.set_aspect('equal')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        # plt.title('%d actuators' %N_act)
        plt.title('Wavelength: %.2f microns' % waves[i])

    # Show a random wavefront map
    c_act = np.random.uniform(-1, 1, size=N_act)
    cmap = 'RdBu'
    fig, axes = plt.subplots(1, N_WAVES)
    for k in range(N_WAVES):
        ax = axes[k]
        wave_r = waves_ratio[k]
        wavefront = np.dot(actuator_matrices[k][0], c_act)
        cmin = min(np.min(wavefront), -np.max(wavefront))
        img = ax.imshow(wavefront, cmap=cmap, extent=[-1, 1, -1, 1])
        ax.set_title('Wavelength: %.2f microns' % waves[k])
        ax.set_xlim([-1.1 * RHO_APER, 1.1 * RHO_APER])
        ax.set_ylim([-1.1 * RHO_APER, 1.1 * RHO_APER])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img.set_clim(cmin, -cmin)
        plt.colorbar(img, ax=ax, orientation='horizontal')
    plt.show()

    ###

    # Temporarily use a single wavelength PSF to calculate the Zernike defocus coefficients
    _PSF = psf.PointSpreadFunction(actuator_matrices[0], N_pix=N_PIX, crop_pix=pix, diversity_coef=np.zeros(N_act))


    # Create a Zernike model so we can mimic the defocus
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=5, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.0)
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))
    # Use Least Squares to find the actuator commands that mimic a Zernike defocus
    zernike_fit = calibration.Zernike_fit(PSF_zernike, _PSF, wavelength=WAVE, rho_aper=RHO_APER)
    defocus_zernike = np.zeros((1, zernike_matrix.shape[-1]))
    defocus_zernike[0, 1] = 1.0
    defocus_actuators = zernike_fit.fit_zernike_wave_to_actuators(defocus_zernike, plot=True, cmap='bwr')[:, 0]
    diversity_defocus = diversity * defocus_actuators

    ###

    PSFs = psf.PointSpreadFunctionMultiwave(matrices=actuator_matrices, N_waves=N_WAVES, wave0=WAVE0, waveN=WAVEN,
                                            wave_ref=WAVE, N_pix=N_PIX, crop_pix=pix, diversity_coef=diversity_defocus)

    # Show the PSF as a function of wavelength
    c_act = 0.3 * np.random.uniform(-1, 1, size=N_act)
    cmap = 'hot'
    fig, axes = plt.subplots(1, N_WAVES)
    for k in range(N_WAVES):
        ax = axes[k]
        wave_r = waves_ratio[k]
        psf_image, _strehl = PSFs.compute_PSF(c_act, wave_idx=k)
        img = ax.imshow(psf_image, cmap=cmap, extent=[-1, 1, -1, 1])
        ax.set_title('Wavelength: %.2f microns' % waves[k])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.colorbar(img, ax=ax, orientation='horizontal')
    plt.show()


    ### ============================================================================================================ ###
    #                                   Generate the training sets
    ### ============================================================================================================ ###


    # Generate a training set | calibration.generate_dataset automatically knows we are Multiwavelength
    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSFs, N_train, N_test,
                                                                              coef_strength, rescale)

    directory = os.path.join(os.getcwd(), 'Multiwave')
    np.save(os.path.join(directory, 'train_PSF'), train_PSF)
    np.save(os.path.join(directory, 'train_coef'), train_coef)
    np.save(os.path.join(directory, 'test_PSF'), test_PSF)
    np.save(os.path.join(directory, 'test_coef'), test_coef)

    train_PSF = np.load(os.path.join(directory, 'train_PSF.npy'))
    train_coef = np.load(os.path.join(directory, 'train_coef.npy'))
    test_PSF = np.load(os.path.join(directory, 'test_PSF.npy'))
    test_coef = np.load(os.path.join(directory, 'test_coef.npy'))

    def show_PSF_multiwave(array, images=5, cmap='hot'):
        N_waves = array.shape[-1] // 2

        for k in range(images):
            fig, axes = plt.subplots(2, N_waves)
            for j in range(N_waves):
                for i in range(2):
                    ax = axes[i][j]
                    idx = i + 2 * j
                    img = ax.imshow(array[k, :, :, idx], cmap=cmap)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    plt.colorbar(img, ax=ax, orientation='horizontal')

    show_PSF_multiwave(train_PSF)
    plt.show()

    # Train the Calibration Model with the complete Wavelength Datacube
    calib = calibration.Calibration(PSF_model=PSFs)
    calib.create_cnn_model(layer_filers, kernel_size, name='CALIBR', activation='relu')
    losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    RMS_evolution, residual = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                         readout_noise=True, RMS_readout=1./SNR)
    MU, STD = np.mean(RMS_evolution), np.std(RMS_evolution)

    calib.plot_RMS_evolution(RMS_evolution)

    phase = np.dot(PSFs.model_matrices_flat[0], test_coef[1])
    rms = np.std(WAVE * 1e3 * phase)
    plt.figure()
    plt.imshow((PSFs.model_matrices[0][:,:,0]))
    plt.colorbar()
    plt.show()

    ### ============================================================================================================ ###
    # Does the number of channels we use matter?
    # Let's train several models with different number of wavelength channels
    mu_chans, std_chans = [], []
    wave_channels = np.arange(1, N_WAVES + 1)
    for N_channels in wave_channels:
        print("\nTraing Model with %d Wavelength Channels" % N_channels)
        _calib = calibration.Calibration(PSF_model=PSFs)
        _calib.PSF_model.N_waves = N_channels
        # Slice the datasets up to a certain channel
        sliced_train_PSF = train_PSF[:, :, :, :2*N_channels]
        sliced_test_PSF = test_PSF[:, :, :, :2*N_channels]
        _losses = _calib.train_calibration_model(sliced_train_PSF, train_coef, sliced_test_PSF, test_coef,
                                                 N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                 readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)
        _rms, _residual = _calib.calibrate_iterations(sliced_test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                         readout_noise=True, RMS_readout=1./SNR)
        final_rms = _rms[-1][-1]
        mu, std = np.mean(final_rms), np.std(final_rms)
        mu_chans.append(mu)
        std_chans.append(std)

    plt.figure()
    plt.errorbar(wave_channels, mu_chans, yerr=std_chans, fmt='o')
    plt.plot(wave_channels, mu_chans)
    plt.xlabel(r'Number of Wavelength Channels')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.ylim(bottom=0)
    plt.show()

    ### ============================================================================================================ ###
    # Are we gaining performance just from the fact that the last channel has a longer wavelength
    # and thus a better sampling of the PSF?

    # To check that we can compare the performance of a model trained ONLY with the last wavelength channel
    # and the model trained on ALL wavelength channels

    ### FIRST Channel (WAVE0)
    # See how a calibration model with 2 channels (Nominal + Defocus) works
    first_channel = calibration.Calibration(PSF_model=PSFs)
    # override the N_waves to force create_cnn_model to have the proper number of channels
    first_channel.PSF_model.N_waves = 1
    # make sure to slice the datacubes to only include the first 2 channels
    _losses = first_channel.train_calibration_model(train_PSF[:, :, :, :2], train_coef, test_PSF[:, :, :, :2], test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)
    first_rms, _residual = first_channel.calibrate_iterations(test_PSF[:, :, :, :2], test_coef, wavelength=WAVE, N_iter=N_iter,
                                                  readout_noise=True, RMS_readout=1. / SNR)
    first_rms = first_rms[-1][-1]
    first_mu, first_std = np.mean(first_rms), np.std(first_rms)

    ### LAST Channel (WAVEN)
    last_channel = calibration.Calibration(PSF_model=PSFs)
    last_channel.PSF_model.N_waves = 1
    _losses = last_channel.train_calibration_model(train_PSF[:, :, :, -2:], train_coef, test_PSF[:, :, :, -2:], test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)
    last_rms, _residual = last_channel.calibrate_iterations(test_PSF[:, :, :, -2:], test_coef, wavelength=WAVE, N_iter=N_iter,
                                                  readout_noise=True, RMS_readout=1. / SNR)
    last_rms = last_rms[-1][-1]
    last_mu, last_std = np.mean(last_rms), np.std(last_rms)

    print("\nImpact of Wavelength")
    print("Is the last channel (longest wavelength) driving the performance?")
    print("Model trained on 1st Channel only [%.2f microns] || RMS: %.2f +- %.2f nm" % (WAVE0, first_mu, first_std))
    print("Model trained on Last Channel only [%.2f microns] || RMS: %.2f +- %.2f nm" % (WAVEN, last_mu, last_std))
    print("Model trained on All %d Channels || RMS: %.2f +- %.2f nm" % (N_WAVES, MU, STD))









