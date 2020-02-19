"""

                          -||  MULTIWAVE  COARSE  ||-


Does the multiwavelength approach help for very coarse scales (10 - 20 mas)

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

# make sure we have even pix
if pix % 2 != 0:
    raise ValueError("pix must be even")

# The RHO_APER for 10 and 20 mas scales is bigger than 1.0
# we need to sample with 5 mas pixels and then average across
SPAX = 5.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
print("Nominal Parameters | Spaxel Scale and Wavelength")
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

N_actuators = 16                    # Number of actuators in [-1, 1] line
alpha_pc = 10                       # Height [percent] at the neighbour actuator (Gaussian Model)

WAVE0 = WAVE                        # Minimum wavelength
WAVEN = 2.00                        # Maximum wavelength
N_WAVES = 5                         # How many wavelength channels to consider

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
N_iter = 4                          # How many iterations to run the calibration (testing)

import importlib
importlib.reload(calibration)

if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    ### ============================================================================================================ ###
    #                                   Let's start single wavelength
    ### ============================================================================================================ ###


    centers = psf.actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act = len(centers[0])
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)

    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=np.zeros(N_act))
    # Create a Zernike model so we can mimic the defocus
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=5, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.0)
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))
    # Use Least Squares to find the actuator commands that mimic a Zernike defocus
    zernike_fit = calibration.Zernike_fit(PSF_zernike, PSF_actuators, wavelength=WAVE, rho_aper=RHO_APER)
    defocus_zernike = np.zeros((1, zernike_matrix.shape[-1]))
    defocus_zernike[0, 1] = 1.0
    defocus_actuators = zernike_fit.fit_zernike_wave_to_actuators(defocus_zernike, plot=True, cmap='bwr')[:, 0]
    diversity_defocus = diversity * defocus_actuators

    # Update the Diversity Map on the actuator model so that it matches Defocus
    PSF_actuators.define_diversity(diversity_defocus)

    ### ============================================================================================================ ###
    #                                   Generate the training sets
    ### ============================================================================================================ ###

    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                              coef_strength, rescale)

    directory = os.path.join(os.getcwd(), 'Multiwave_Coarse')
    np.save(os.path.join(directory, 'train_PSF'), train_PSF)
    np.save(os.path.join(directory, 'train_coef'), train_coef)
    np.save(os.path.join(directory, 'test_PSF'), test_PSF)
    np.save(os.path.join(directory, 'test_coef'), test_coef)

    train_PSF = np.load(os.path.join(directory, 'train_PSF.npy'))
    train_coef = np.load(os.path.join(directory, 'train_coef.npy'))
    test_PSF = np.load(os.path.join(directory, 'test_PSF.npy'))
    test_coef = np.load(os.path.join(directory, 'test_coef.npy'))

    # Downsample the arrays to show the effect of using a coarser scale
    down_train_PSF = PSF_actuators.downsample_datacube(train_PSF)
    down_test_PSF = PSF_actuators.downsample_datacube(test_PSF)
    utils.plot_images(train_PSF, 1)
    utils.plot_images(down_train_PSF, 1)
    plt.show()

    # Let's see if there's a difference in performance between Scales
    # Fine Scale (5 mas) model
    calib = calibration.Calibration(PSF_model=PSF_actuators)
    calib.create_cnn_model(layer_filters, kernel_size, name='SINGLE_FINE', activation='relu')
    losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    RMS_fine, residual = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                    readout_noise=True, RMS_readout=1./SNR)

    # Coarse Scale (10 mas) model
    coarse_calib = calibration.Calibration(PSF_model=PSF_actuators)
    # Override self.PSF_model.crop_pix to account for the downsampling
    coarse_calib.PSF_model.crop_pix = pix // 2
    coarse_calib.create_cnn_model(layer_filters, kernel_size, name='SINGLE_COARSE', activation='relu')
    coarse_calib.PSF_model.crop_pix = pix
    # Do NOT forget to use the downsampled arrays for training
    losses_coarse = coarse_calib.train_calibration_model(down_train_PSF, train_coef, down_test_PSF, test_coef,
                                                         N_loops, epochs_loop, verbose=1, batch_size_keras=32,
                                                         plot_val_loss=False, readout_noise=True,
                                                         RMS_readout=[1. / SNR], readout_copies=readout_copies)
    # Do NOT forget to tell the calibration model to downsample after the PSF updates
    RMS_coarse, _residual = coarse_calib.calibrate_iterations(down_test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                       readout_noise=True, RMS_readout=1./SNR,
                                                       downsample=True)

    # # We can see that the Coarse Scale significantly degrades the performance
    # calib.plot_RMS_evolution(RMS_fine)
    # coarse_calib.plot_RMS_evolution(RMS_coarse)

    utils.show_wavefronts_grid(PSF_actuators, residual, RHO_APER,
                         images=(3, 5), cmap='jet', title=None)
    utils.show_wavefronts_grid(PSF_actuators, _residual, RHO_APER,
                         images=(3, 5), cmap='jet', title=None)

    plt.show()

    ### ============================================================================================================ ###
    #                                   Multiwavelength!
    ### ============================================================================================================ ###


    centers_multiwave = psf.actuator_centres_multiwave(N_actuators=N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                       N_waves=N_WAVES, wave0=WAVE0, waveN=WAVEN, wave_ref=WAVE)
    N_act = len(centers_multiwave[0][0])

    actuator_matrices = psf.actuator_matrix_multiwave(centres=centers_multiwave, alpha_pc=alpha_pc, rho_aper=RHO_APER,
                                                      rho_obsc=RHO_OBSC, N_waves=N_WAVES,  wave0=WAVE0, waveN=WAVEN,
                                                      wave_ref=WAVE, N_PIX=N_PIX)

    PSFs = psf.PointSpreadFunctionMultiwave(matrices=actuator_matrices, N_waves=N_WAVES, wave0=WAVE0, waveN=WAVEN,
                                            wave_ref=WAVE, N_pix=N_PIX, crop_pix=pix, diversity_coef=diversity_defocus)


    # Generate a training set | calibration.generate_dataset automatically knows we are Multiwavelength
    train_PSF_wave, train_coef_wave, test_PSF_wave, test_coef_wave = calibration.generate_dataset(PSFs, N_train, N_test,
                                                                              coef_strength, rescale)

    np.save(os.path.join(directory, 'train_PSF_wave'), train_PSF_wave)
    np.save(os.path.join(directory, 'train_coef_wave'), train_coef_wave)
    np.save(os.path.join(directory, 'test_PSF_wave'), test_PSF_wave)
    np.save(os.path.join(directory, 'test_coef_wave'), test_coef_wave)

    down_train_PSF_wave = PSFs.downsample_datacube(train_PSF_wave)
    down_test_PSF_wave = PSFs.downsample_datacube(test_PSF_wave)
    utils.show_PSF_multiwave(train_PSF_wave, 1)
    utils.show_PSF_multiwave(down_train_PSF_wave, 1)
    plt.show()

    ### ============================================================================================================ ###
    # Let's see how the peformance changes with sampling

    fine_calib_wave = calibration.Calibration(PSF_model=PSFs)
    fine_calib_wave.create_cnn_model(layer_filters, kernel_size, name='FINE', activation='relu')
    losses = fine_calib_wave.train_calibration_model(train_PSF_wave, train_coef_wave, test_PSF_wave, test_coef_wave,
                                                N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    RMS_fine_wave, residual = fine_calib_wave.calibrate_iterations(test_PSF_wave, test_coef_wave, wavelength=WAVE, N_iter=N_iter,
                                                              readout_noise=True, RMS_readout=1./SNR)
    fine_calib_wave.plot_RMS_evolution(RMS_fine_wave)

    coarse_calib_wave = calibration.Calibration(PSF_model=PSFs)
    coarse_calib_wave.PSF_model.crop_pix = pix // 2
    coarse_calib_wave.create_cnn_model(layer_filters, kernel_size, name='COARSE', activation='relu')
    coarse_calib_wave.PSF_model.crop_pix = pix
    coarse_losses = coarse_calib_wave.train_calibration_model(down_train_PSF_wave, train_coef_wave, down_test_PSF_wave, test_coef_wave,
                                                N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    # Very important! Remember to tell calibrate_iterations to downsample after each PSF update
    RMS_coarse_wave, _residual = coarse_calib_wave.calibrate_iterations(down_test_PSF_wave, test_coef_wave, wavelength=WAVE, N_iter=N_iter,
                                                              readout_noise=True, RMS_readout=1./SNR,
                                                              downsample=True)

    ### ============================================================================================================ ###
    #                                  Compare the performance
    ### ============================================================================================================ ###

    # Single Wavelength
    mu_fine, std_fine = np.mean(RMS_fine[-1][-1]), np.std(RMS_fine[-1][-1])
    mu_coarse, std_coarse = np.mean(RMS_coarse[-1][-1]), np.std(RMS_coarse[-1][-1])

    # Multiwavelength
    mu_fine_wave, std_fine_wave = np.mean(RMS_fine_wave[-1][-1]), np.std(RMS_fine_wave[-1][-1])
    mu_coarse_wave, std_coarse_wave = np.mean(RMS_coarse_wave[-1][-1]), np.std(RMS_coarse_wave[-1][-1])

    print("\n       Performance Comparison:")
    print("\nSingle Wavelength Models | %.2f microns" % WAVE)
    print("Fine Scale:     %2.1f [mas] | RMS: %.1f +- %.1f nm" % (SPAX, mu_fine, std_fine))
    print("Coarse Scale:  %2.1f [mas] | RMS: %.1f +- %.1f nm" % (2*SPAX, mu_coarse, std_coarse))

    print("\nMulti Wavelength Models | %.2f - %.2f microns [%d waves]" % (WAVE0, WAVEN, N_WAVES))
    print("Fine Scale:     %2.1f [mas] | RMS: %.1f +- %.1f nm" % (SPAX, mu_fine_wave, std_fine_wave))
    print("Coarse Scale:  %2.1f [mas] | RMS: %.1f +- %.1f nm" % (2*SPAX, mu_coarse_wave, std_coarse_wave))

    # Example
    #        Performance Comparison:
    #
    # Single Wavelength Models | 1.50 microns
    # Fine Scale:     5.0 [mas] | RMS: 28.3 +- 2.6 nm
    # Coarse Scale:  10.0 [mas] | RMS: 74.3 +- 18.8 nm
    #
    # Multi Wavelength Models | 1.50 - 2.00 microns [5 waves]
    # Fine Scale:     5.0 [mas] | RMS: 20.8 +- 1.7 nm
    # Coarse Scale:  10.0 [mas] | RMS: 33.6 +- 3.1 nm

    # We can see that the multiwavelength approach really pays off at coarser scales
    # Possible reasons:
    # (1) less pixels -> readout noise contamination more severe
    # (2) coarse scale -> more beneficial to go towards longer wavelengths?

    # Show the performance profiles
    # Single Wavelength
    calib_title = r'Single Wavelength Model | %.2f [microns] | %.1f [mas]' % (WAVE0, SPAX)
    calib.plot_RMS_evolution(RMS_fine, colormap=cm.Blues, title=calib_title)

    coarse_title = r'Single Wavelength Model | %.2f [microns] | %.1f [mas]' % (WAVE0, 2*SPAX)
    coarse_calib.plot_RMS_evolution(RMS_coarse, colormap=cm.Reds, title=coarse_title)

    # Multiwavelength
    fine_wave_title = r'Multiwavelength Model | %.2f - %.2f [microns] %d Waves | %.1f [mas]' % (WAVE0, WAVEN, N_WAVES, SPAX)
    fine_calib_wave.plot_RMS_evolution(RMS_fine_wave, colormap=cm.Blues, title=fine_wave_title)

    coarse_wave_title = r'Multiwavelength Model | %.2f - %.2f [microns] %d Waves | %.1f [mas]' % (WAVE0, WAVEN, N_WAVES, 2*SPAX)
    coarse_calib_wave.plot_RMS_evolution(RMS_coarse_wave, colormap=cm.Reds, title=coarse_wave_title)



    ### ============================================================================================================ ###
    ### ============================================================================================================ ###
    # OLD BITS
    if False:
        from numpy.fft import fftshift, fft2
        def compute_psd(fine_residuals, coarse_residuals):
            N_samples = fine_residuals.shape[0]
            psd_fine, psd_coarse = [], []
            for k in range(N_samples):
                phase_fine = np.dot(PSF_actuators.model_matrix, fine_residuals[k])
                fft_fine = fftshift(fft2(phase_fine))
                psd_fine.append((np.abs(fft_fine)) ** 2)

                phase_coarse = np.dot(PSF_actuators.model_matrix, coarse_residuals[k])
                fft_coarse = fftshift(fft2(phase_coarse))
                psd_coarse.append((np.abs(fft_coarse)) ** 2)
            psd_fine = np.stack(psd_fine)
            rms_fine = np.std(psd_fine, axis=0)
            psd_fine = np.mean(psd_fine, axis=0)
            psd_coarse = np.stack(psd_coarse)
            rms_coarse = np.std(psd_coarse, axis=0)
            psd_coarse = np.mean(psd_coarse, axis=0)

            return psd_fine[N_PIX//2, N_PIX//2:], rms_fine[N_PIX//2, N_PIX//2:], psd_coarse[N_PIX//2, N_PIX//2:], rms_coarse[N_PIX//2, N_PIX//2:]

        psd_fine, rms_fine, psd_coarse, rms_coarse = compute_psd(residual, _residual)

        plt.figure()
        plt.errorbar(range(128), psd_fine, yerr=rms_fine/2)
        plt.errorbar(range(128), psd_coarse, yerr=rms_coarse/2)
        plt.yscale('log')
        plt.xscale('log')
        plt.show()





