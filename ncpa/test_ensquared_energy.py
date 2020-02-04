"""

                               -||  ENSQUARED ENERGY  ||-

Test / Tutorial to investigate the effect of RMS NCPA on Ensquared Energy for several SPAXEL SCALES

Using oversampled PSF images (typically 0.5 mas pixels) we compute variations on Ensquared Energy
within the central pixel, assuming a new coarser scale (4, 10, 20 mas pixels).
These variations on Ensquared Energy are calculated as a function of RMS wavefront
using a wavefront model based on Zernike polynomials

That way we can derive requirements on how much RMS wavefront we can tolerate for each spaxel scale
specially interesting for the coarsest scales, were the PSF is not Nyquist sampled

Author: Alvaro Menduina
Date: Feb 2020

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import psf
import utils
import zernike


def ensquared_one_pix(array, pix_scale, new_scale=40, plot=True):
    """
    Given an oversampled PSF (typically 0.5-1.0 mas spaxels), it calculates
    the Ensquared Energy of the central spaxel in a new_scale (4, 10, 20 mas)

    It selects a window of size new_scale and adds up the Intensity of those pixels

    :param array: PSF
    :param pix_scale: size in [mas] of the spaxels in the PSF
    :param new_scale: new spaxel scale [mas]
    :param plot: whether to plot the array and windows
    :return:
    """

    n = int(new_scale // pix_scale)
    minPix, maxPix = (pix + 1 - n) // 2, (pix + 1 + n) // 2
    ens = array[minPix:maxPix, minPix:maxPix]
    # print(ens.shape)
    energy = np.sum(ens)

    if plot:
        mapp = 'viridis'
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1 = plt.subplot(1, 2, 1)
        square = Rectangle((minPix - 0.5, minPix - 0.5), n, n, linestyle='--', fill=None, color='white')
        ax1.add_patch(square)
        img1 = ax1.imshow(array, cmap=mapp)
        ax1.set_title('%.1f mas pixels' % pix_scale)
        img1.set_clim(0, 1)
        plt.colorbar(img1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(1, 2, 2)
        img2 = ax2.imshow(ens, cmap=mapp)
        ax2.set_title('%d mas window' % new_scale)
        img1.set_clim(0, 1)
        plt.colorbar(img2, ax=ax2, orientation='horizontal')

    return energy


def calculate_ensquared_energy(PSF_model, wave, N_trials, N_rms,
                               rms_amplitude, nominal_scale, spaxel_scales):
    """
    Calculates variations in Ensquared Energy for a given PSF model
    at a certain wavelength, as a function of the RMS aberrations

    Repeats the analysis for several spaxel scales

    :param PSF_model: PSF model to compute the PSF images with aberrations
    :param wave: nominal wavelength (to compute the RMS wavefront)
    :param N_trials: how many times to repeat the analysis (random wavefronts)
    :param N_rms: how many data points to sample the RMS range
    :param rms_amplitude: scales the aberration coefficients to keep the RMS reasonable
    :param nominal_scale: the default spaxel scale of the PSF model
    :param spaxel_scales: list of Spaxel Scales to consider
    :return:
    """

    N_zern = PSF_model.N_coef
    ensquared_results = []

    print("Calculating Ensquared Energy")
    for scale in spaxel_scales:     # Loop over each Spaxel Scale [mas]
        print("%d mas spaxel scale" % scale)

        data = np.zeros((2, N_trials * N_rms))
        amplitudes = np.linspace(0.0, rms_amplitude, N_rms)
        i = 0
        p0, s0 = PSF_model.compute_PSF(np.zeros(N_zern))  # Nominal PSF
        # Calculate the EE for the nominal PSF so that you can compare
        EE0 = ensquared_one_pix(p0, pix_scale=nominal_scale, new_scale=scale, plot=False)

        for amp in amplitudes:              # Loop over coefficient strength

            for k in range(N_trials):               # For each case, repeat N_trials

                c_act = np.random.uniform(-amp, amp, size=N_zern)
                phase_flat = np.dot(PSF_model.model_matrix_flat, c_act)
                rms = wave * 1e3 * np.std(phase_flat)
                p, s = PSF_model.compute_PSF(c_act)
                EE = ensquared_one_pix(p, pix_scale=nominal_scale, new_scale=scale, plot=False)
                dEE = EE / EE0 * 100
                data[:, i] = [rms, dEE]
                i += 1

        ensquared_results.append(data)

    return ensquared_results


### PARAMETERS ###

# Change any of these before running the code

pix = 150  # Pixels to crop the PSF
N_PIX = 512  # Pixels for the Fourier arrays

# WATCH OUT:
# We want the spaxel scale to be 4 mas at 1.5 um
# But that means that it won't be 4 mas at whatever other wavelength

# SPAXEL SCALE
wave0 = 1.5  # Nominal wavelength at which to force a given spaxel scale
SPAXEL_MAS = 0.5  # [mas] spaxel scale at wave0
RHO_MAS = utils.rho_spaxel_scale(spaxel_scale=SPAXEL_MAS, wavelength=wave0)

# Rescale the aperture radius to mimic the wavelength scaling of the PSF
wave = 1.5  # 1 micron
RHO_APER = wave0 / wave * RHO_MAS
RHO_OBSC = 0.3 * RHO_APER  # Central obscuration (30% of ELT)

# Decide whether you want to use Zernike polynomials
# "up to a certain row" row_by_row == False
# or whether you want it row_by_row == True
# This is because low order and high order aberrations behave differently wrt EE
row_by_row = False
min_Zernike = 4  # Row number
max_Zernike = 7
zernike_triangle = zernike.triangular_numbers(N_levels=max_Zernike)

spaxel_scales = [4, 10, 20]
colors = ['blue', 'green', 'red']

N_trials = 5           # How many data points per RMS case
N_rms = 50              # How many RMS points to consider
rms_amplitude = 0.1

if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    print("\n==================================================================")
    print("                       ENSQUARED ENERGY")
    print("==================================================================")

    print("Analysis of the impact of RMS wavefront on Ensquared Energy")
    print("Considering Zernike levels from radial order %d to %d" % (min_Zernike - 1, max_Zernike - 1))

    for zernike_level in np.arange(min_Zernike, max_Zernike, 2):
        print("\nZernike Level: %d | Radial Order rho^{%d}" % (zernike_level, zernike_level - 1))

        # Initialize the Zernike matrices
        zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=zernike_level, rho_aper=RHO_APER,
                                                                              rho_obsc=RHO_OBSC,
                                                                              N_PIX=N_PIX, radial_oversize=1.0)

        if row_by_row:      # Select only a specific row (radial order) of Zernike
            # Number of polynomials in that last Zernike row
            poly_in_row = zernike_triangle[zernike_level] - zernike_triangle[zernike_level - 1]
            zernike_matrix = zernike_matrix[:, :, -poly_in_row:]   # select only the high orders
            flat_zernike = flat_zernike[:, -poly_in_row:]

        N_zern = zernike_matrix.shape[-1]  # Update the N_zern after removing Piston and Tilts

        matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
        PSF_zern = psf.PointSpreadFunction(matrices, N_pix=N_PIX, crop_pix=pix, diversity_coef=np.zeros(N_zern))

        rms_amplitude_new = rms_amplitude * (7/ N_zern)
        ensquared = calculate_ensquared_energy(PSF_zern, wave, N_trials, N_rms, rms_amplitude,
                                               nominal_scale=SPAXEL_MAS, spaxel_scales=spaxel_scales)

        # maxes = []
        plt.figure()
        for data, scale, color in zip(ensquared, spaxel_scales, colors):
            plt.scatter(data[0], data[1], color=color, s=3, label=scale)
        #     maxRMS = np.max(data[1])
        #     maxes.append(maxRMS)
        # maxRMS = int(np.max(maxes))

        # plt.axhline(y=95, color='darksalmon', linestyle='--')
        # plt.axhline(y=90, color='lightsalmon', linestyle='-.')
        plt.xlabel(r'RMS wavefront [nm]')
        plt.ylabel(r'Relative Encircled Energy [per cent]')
        plt.legend(title='Spaxel [mas]', loc=3)
        plt.ylim([80, 100])
        plt.xlim([0, 100])
        plt.title(r'%d Zernike ($\rho^{%d}$) [%.2f $\mu m$]' % (N_zern, zernike_level - 1, wave))
        plt.grid(True)
        # plt.savefig('%d Zernike' % N_zern)

    # Show some examples

    p0, s0 = PSF_zern.compute_PSF(np.zeros(N_zern))  # Nominal PSF
    # Calculate the EE for the nominal PSF so that you can compare
    EE0 = ensquared_one_pix(p0, pix_scale=SPAXEL_MAS, new_scale=10, plot=True)

    plt.show()
