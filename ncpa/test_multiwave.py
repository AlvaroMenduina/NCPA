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



