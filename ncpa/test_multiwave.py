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
WAVEN = 1.75                        # Maximum wavelength
N_WAVES = 15                         # How many wavelength channels to consider

# Machine Learning bits
N_train, N_test = 10000, 500        # Samples for the training of the models
coef_strength = 0.45                # Strength of the actuator coefficients
diversity = 0.85                    # Strength of extra diversity commands
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filers = [64, 32, 16, 8]      # How many filters per layer
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
    diversity_defocus = diversity / (2 * np.pi) * defocus_actuators

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

    utils.show_PSF_multiwave(train_PSF)
    plt.show()

    # Train the Calibration Model with the complete Wavelength Datacube
    calib = calibration.Calibration(PSF_model=PSFs)
    calib.create_cnn_model(layer_filers, kernel_size, name='CALIBR', activation='relu')
    losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    RMS_evolution, residual = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                         readout_noise=True, RMS_readout=1./SNR)
    MU, STD = np.mean(RMS_evolution[-1][-1]), np.std(RMS_evolution[-1][-1])

    calib.plot_RMS_evolution(RMS_evolution)
    plt.show()

    ### ============================================================================================================ ###
    # Does the number of channels we use matter?
    # Let's train several models with different number of wavelength channels

    noiseSNR = np.array([500, 250, 125])
    N_noise = noiseSNR.shape[0]
    mu_chans_noise = np.zeros((N_noise, N_WAVES))
    std_chans_noise = np.zeros((N_noise, N_WAVES))
    for i_noise, SNR in enumerate(noiseSNR):

        # mu_chans, std_chans = [], []
        wave_channels = np.arange(1, N_WAVES + 1)
        for j_chan, N_channels in enumerate(wave_channels):
            print("\nTraing Model with %d Wavelength Channels" % N_channels)

            _PSFs = psf.PointSpreadFunctionMultiwave(matrices=actuator_matrices[:N_channels], N_waves=N_channels,
                                                     wave0=WAVE0, waveN=waves[N_channels - 1], wave_ref=WAVE,
                                                     N_pix=N_PIX, crop_pix=pix, diversity_coef=diversity_defocus)
            print(_PSFs.wavelengths)

            _calib = calibration.Calibration(PSF_model=_PSFs)
            _calib.create_cnn_model(layer_filers, kernel_size, name='%d_WAVES' % N_channels, activation='relu')
            # Slice the datasets up to a certain channel
            sliced_train_PSF = train_PSF[:, :, :, :2*N_channels]
            print(sliced_train_PSF.shape)
            sliced_test_PSF = test_PSF[:, :, :, :2*N_channels]
            _losses = _calib.train_calibration_model(sliced_train_PSF, train_coef, sliced_test_PSF, test_coef,
                                                     N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                     readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)
            _rms, _residual = _calib.calibrate_iterations(sliced_test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                             readout_noise=True, RMS_readout=1./SNR)
            final_rms = _rms[-1][-1]
            mu, std = np.mean(final_rms), np.std(final_rms)
            mu_chans_noise[i_noise, j_chan] = mu
            std_chans_noise[i_noise, j_chan] = std

            # mu_chans.append(mu)
            # std_chans.append(std)

    colors = cm.Reds(np.linspace(0.5, 0.75, N_noise))
    plt.figure()
    for i_noise, SNR in enumerate(noiseSNR[:2]):
        plt.errorbar(wave_channels, y=mu_chans_noise[i_noise], yerr=std_chans_noise[i_noise],
                     fmt='o', color=colors[i_noise], label='SNR %d' % SNR)
    plt.xlabel(r'Wavelength Channels')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.legend()
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
    first_channel.create_cnn_model(layer_filers, kernel_size, name='FIRST', activation='relu')
    # make sure to slice the datacubes to only include the first 2 channels
    _losses = first_channel.train_calibration_model(train_PSF[:, :, :, :2], train_coef, test_PSF[:, :, :, :2], test_coef,
                                                    N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)
    # to calibrate iterations we need to trick the model into thinking it's multichannel!
    # because of the difference between self.model_matrix and self.model_matrices
    first_channel.PSF_model.N_waves = N_WAVES
    first_rms, _residual = first_channel.calibrate_iterations(test_PSF[:, :, :, :2], test_coef, wavelength=WAVE, N_iter=N_iter,
                                                  readout_noise=True, RMS_readout=1. / SNR,
                                                              multiwave_slice=(0, 2))
    first_rms = first_rms[-1][-1]
    first_mu, first_std = np.mean(first_rms), np.std(first_rms)

    ### LAST Channel (WAVEN)
    last_channel = calibration.Calibration(PSF_model=PSFs)
    last_channel.PSF_model.N_waves = 1
    last_channel.create_cnn_model(layer_filers, kernel_size, name='LAST', activation='relu')
    _losses = last_channel.train_calibration_model(train_PSF[:, :, :, -2:], train_coef, test_PSF[:, :, :, -2:], test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)
    last_channel.PSF_model.N_waves = N_WAVES
    last_rms, _residual = last_channel.calibrate_iterations(test_PSF[:, :, :, -2:], test_coef, wavelength=WAVE, N_iter=N_iter,
                                                  readout_noise=True, RMS_readout=1. / SNR,
                                                            multiwave_slice=(2*(N_WAVES - 1), 2 * N_WAVES))
    last_rms = last_rms[-1][-1]
    last_mu, last_std = np.mean(last_rms), np.std(last_rms)

    print("\nImpact of Wavelength")
    print("Is the last channel (longest wavelength) driving the performance?")
    print("Model trained on 1st Channel only [%.2f microns] || RMS: %.2f +- %.2f nm" % (WAVE0, first_mu, first_std))
    print("Model trained on Last Channel only [%.2f microns] || RMS: %.2f +- %.2f nm" % (WAVEN, last_mu, last_std))
    print("Model trained on All %d Channels || RMS: %.2f +- %.2f nm" % (N_WAVES, MU, STD))

    _chan = np.linspace(1, N_WAVES, 20, endpoint=True)
    theory = first_mu / np.sqrt(_chan)
    # Does the number of channels we use matter?
    plt.figure()
    plt.errorbar([1], first_mu, yerr=first_std, fmt='o', label='Single Wavelength')
    plt.plot(_chan, theory, linestyle='--', color='black', label=r'1/$\sqrt{N}$ fit')
    plt.errorbar(wave_channels, mu_chans_noise[0], yerr=std_chans_noise[0], fmt='o', label='Multiwavelength')
    # plt.plot(wave_channels, mu_chans)
    plt.xlabel(r'Number of Wavelength Channels')
    plt.ylabel(r'RMS after calibration [nm]')
    plt.ylim(bottom=0)
    plt.legend()
    plt.xticks(np.arange(1, N_WAVES + 1))
    plt.show()

    ### Results as a function of SNR | N_WAVES 5 | Waves: 1.5 -> 2.0 microns
    # SNR           1 wave          5 waves         delta mean [%]
    # 100           60.3 +- 8.3     39.2 +- 4.2     21.1
    # 125           53.4 +- 6.9     37.4 +- 4.0     16.0
    # 250           34.4 +- 4.2     24.2 +- 2.6     10.2 [29.6%]
    # 500           22.9 +- 2.4     17.0 +- 1.5     5.9

    mu_one_wave = [60.3, 53.4, 34.4, 22.9]
    std_one_wave = [8.3, 6.9, 4.2, 2.4]

    ### ============================================================================================================ ###
    # Does the wavelength range matter?
    # What happens if we use the same number of wavelengths but extend WAVEN longer?
    N_WAVES = 5
    SNRs = [125, 250, 500]
    long_waves = [1.75, 2.0, 2.25]
    mu_long, std_long = [], []
    for wave_N in long_waves[:1]:
        _centres = psf.actuator_centres_multiwave(N_actuators=N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                  N_waves=N_WAVES, wave0=WAVE0, waveN=wave_N, wave_ref=WAVE)

        _matrices = psf.actuator_matrix_multiwave(centres=_centres, alpha_pc=alpha_pc, rho_aper=RHO_APER,
                                                  rho_obsc=RHO_OBSC, N_waves=N_WAVES, wave0=WAVE0, waveN=wave_N,
                                                  wave_ref=WAVE, N_PIX=N_PIX)
        _PSFs = psf.PointSpreadFunctionMultiwave(matrices=_matrices, N_waves=N_WAVES,
                                                 wave0=WAVE0, waveN=wave_N, wave_ref=WAVE,
                                                 N_pix=N_PIX, crop_pix=pix, diversity_coef=diversity_defocus)

        train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(_PSFs, N_train, N_test,
                                                                                  coef_strength, rescale)

        mu_snr, std_snr = [], []
        for _snr in SNRs[:1]:
            print("\nSNR: ", _snr)
            long_calib = calibration.Calibration(PSF_model=_PSFs)
            long_calib.create_cnn_model(layer_filers, kernel_size, name='LONG', activation='relu')
            losses = long_calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                    N_loops, epochs_loop, verbose=1, batch_size_keras=32,
                                                    plot_val_loss=False, readout_noise=True, RMS_readout=[1. / _snr],
                                                    readout_copies=readout_copies)

            RMS_evolution, residual = long_calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                                  readout_noise=True, RMS_readout=1. / _snr)
            mu_snr.append(np.mean(RMS_evolution[-1][-1]))
            std_snr.append(np.std(RMS_evolution[-1][-1]))
        mu_long.append(mu_snr)
        std_long.append(std_snr)

    MUS_LONG = np.array(mu_long)
    STD_LONG = np.array(std_long)

    blues = cm.Blues(np.linspace(0.25, 1.0, len(SNRs)))
    plt.figure()
    plt.errorbar(SNRs, mu_one_wave, yerr=std_one_wave, color='red', fmt='o', label=r'1 $\lambda$')
    plt.plot(SNRs, mu_one_wave, color='red')
    for i, wave_N in enumerate(long_waves):
        mus_waves = mu_long[i]
        print(mus_waves[-1])
        std_waves = std_long[i]
        label = r'%d $\lambda$ [%.2f microns]' % (N_WAVES, wave_N)
        plt.errorbar(SNRs, mus_waves, yerr=std_waves, color=blues[i], fmt='o', label=label)
        plt.plot(SNRs, mus_waves, color=blues[i])
    plt.legend()
    plt.xlabel('SNR')
    plt.ylabel('RMS after calibration [nm]')
    plt.show()














