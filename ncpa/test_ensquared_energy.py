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
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

import psf
import utils
import zernike


def ensquared_rectangle(array, pix_scale, new_scale=30, plot=True):
    """
    For the 60x30 scale we need a rectangle
    :param array:
    :param pix_scale:
    :param new_scale:
    :param plot:
    :return:
    """

    n = int(new_scale // pix_scale)
    minPixX, maxPixX = (pix + 1 - 2*n) // 2, (pix + 1 + 2*n) // 2
    minPixY, maxPixY = (pix + 1 - n) // 2, (pix + 1 + n) // 2
    ens = array[minPixX:maxPixX, minPixY:maxPixY]
    # print(ens.shape)
    energy = np.sum(ens)

    if plot:
        mapp = 'viridis'
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1 = plt.subplot(1, 2, 1)
        square = Rectangle((minPixY - 0.5, minPixX - 0.5), n, 2*n, linestyle='--', fill=None, color='white')
        ax1.add_patch(square)
        img1 = ax1.imshow(array, cmap=mapp)
        ax1.set_title('%.1f mas pixels' % pix_scale)
        img1.set_clim(0, 1)
        plt.colorbar(img1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(1, 2, 2)
        img2 = ax2.imshow(ens, cmap=mapp)
        ax2.set_title('%d x %d mas window' % (new_scale, 2 * new_scale))
        img1.set_clim(0, 1)
        plt.colorbar(img2, ax=ax2, orientation='horizontal')

    return energy


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
                               rms_amplitude, nominal_scale, spaxel_scales, rescale_coeff=None):
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

    a_min = 1.0 if rescale_coeff is None else rescale_coeff

    print("Calculating Ensquared Energy")
    for scale in spaxel_scales:     # Loop over each Spaxel Scale [mas]
        print("%d mas spaxel scale" % scale)

        data = np.zeros((2, N_trials * N_rms))
        amplitudes = np.linspace(0.0, rms_amplitude, N_rms)
        i = 0
        p0, s0 = PSF_model.compute_PSF(np.zeros(N_zern))  # Nominal PSF
        # Calculate the EE for the nominal PSF so that you can compare
        EE0 = ensquared_one_pix(p0, pix_scale=nominal_scale, new_scale=2*scale, plot=False)

        # 60x120
        # EE0 = ensquared_rectangle(p0, pix_scale=nominal_scale, new_scale=60, plot=False)

        for amp in amplitudes:              # Loop over coefficient strength

            for k in range(N_trials):               # For each case, repeat N_trials

                c_act = np.random.uniform(-amp, amp, size=N_zern)
                rescale = np.linspace(1, a_min, N_zern)
                c_act *= rescale
                phase_flat = np.dot(PSF_model.model_matrix_flat, c_act)
                rms = wave * 1e3 * np.std(phase_flat)
                p, s = PSF_model.compute_PSF(c_act)
                EE = ensquared_one_pix(p, pix_scale=nominal_scale, new_scale=2*scale, plot=False)
                # EE = ensquared_rectangle(p, pix_scale=nominal_scale, new_scale=60, plot=False)
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
wave = 1.0  # 1 micron
RHO_APER = wave0 / wave * RHO_MAS
RHO_OBSC = 0.3 * RHO_APER  # Central obscuration (30% of ELT)

# Decide whether you want to use Zernike polynomials
# "up to a certain row" row_by_row == False
# or whether you want it row_by_row == True
# This is because low order and high order aberrations behave differently wrt EE
row_by_row = True
min_Zernike = 3  # Row number
max_Zernike = 7
zernike_triangle = zernike.triangular_numbers(N_levels=max_Zernike)

spaxel_scales = [4, 10, 20]
# colors = cm.Spectral(np.linspace(0, 1, 3))
colors = ['blue', 'green', 'red']

N_trials = 5           # How many data points per RMS case
N_rms = 50              # How many RMS points to consider
rms_amplitude = 0.15

if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    path_results = os.path.abspath("H:/PycharmProjects/NCPA/tests/test_ensquared_energy")
    box_folder = "box_2x2"
    mode_folder = "row_by_row" if row_by_row is True else "up_to_row"
    folder = os.path.join(path_results, box_folder)
    results_dir = os.path.join(folder, mode_folder)

    print("Results will be saved in: ", results_dir)
    if not os.path.exists(folder):
        print("Creating Directory")
        os.mkdir(folder)  # If not, create the directory to store results

    if not os.path.exists(results_dir):
        print("Creating Directory")
        os.mkdir(results_dir)  # If not, create the directory to store results


    print("\n==================================================================")
    print("                       ENSQUARED ENERGY")
    print("==================================================================")

    print("Analysis of the impact of RMS wavefront on Ensquared Energy")
    print("Considering Zernike levels from radial order %d to %d" % (min_Zernike - 1, max_Zernike - 1))

    zernike_levels = np.arange(min_Zernike, max_Zernike, 1)
    n_rows = zernike_levels.shape[0] // 2
    n_cols = n_rows
    fig, axes = plt.subplots(n_rows, n_cols)
    # spaxel_scales = [60]

    for zernike_level, ax in zip(zernike_levels, axes.flatten()):
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

        ensquared = calculate_ensquared_energy(PSF_zern, wave, N_trials, N_rms, rms_amplitude,
                                               nominal_scale=SPAXEL_MAS, spaxel_scales=spaxel_scales)

        # maxes = []
        # plt.figure()
        for data, scale, color in zip(ensquared, spaxel_scales, colors):
            ax.scatter(data[0], data[1], color=color, s=3, label=scale)
        #     maxRMS = np.max(data[1])
        #     maxes.append(maxRMS)
        # maxRMS = int(np.max(maxes))

        ax.set_xlabel(r'RMS wavefront [nm]')
        ax.set_ylabel(r'Relative Encircled Energy [per cent]')
        ax.legend(title='Spaxel [mas]', loc=3)
        ax.set_ylim([80, 100])
        ax.set_xlim([0, 150])
        ax.set_title(r'%d Zernike Polynomials ($\rho^{%d}$) [%.2f $\mu m$]' % (N_zern, zernike_level - 1, wave))
        ax.grid(True)

    # plt.tight_layout()
    fig_name = "Zernike_rowbyrow" if row_by_row else "Zernike_Level_uptorow"
    plt.savefig(os.path.join(results_dir, fig_name))

        # # plt.axhline(y=95, color='darksalmon', linestyle='--')
        # # plt.axhline(y=90, color='lightsalmon', linestyle='-.')
        # plt.xlabel(r'RMS wavefront [nm]')
        # plt.ylabel(r'Relative Encircled Energy [per cent]')
        # plt.legend(title='Spaxel [mas]', loc=3)
        # plt.ylim([80, 100])
        # plt.xlim([0, 150])
        # plt.title(r'%d Zernike Polynomials ($\rho^{%d}$) [%.2f $\mu m$]' % (N_zern, zernike_level - 1, wave))
        # plt.grid(True)
        # fig_name = "Zernike_Level%d_rowbyrow" % zernike_level if row_by_row else "Zernike_Level%d" % zernike_level
        # plt.savefig(os.path.join(results_dir, fig_name))

    # Show some example
    p0, s0 = PSF_zern.compute_PSF(0.15*np.random.normal(0, 1, N_zern))  # Nominal PSF
    # Calculate the EE for the nominal PSF so that you can compare
    EE0 = ensquared_one_pix(p0, pix_scale=SPAXEL_MAS, new_scale=40, plot=True)
    fig_name = "Example_40mas_window"
    plt.tight_layout()
    plt.savefig(os.path.join(path_results, fig_name))

    plt.show()


    # 60 x 30

    # Up to order

    max_Zernike = 14
    zernike_triangle = zernike.triangular_numbers(N_levels=max_Zernike)
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=max_Zernike, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.0)
    N_zern = zernike_matrix.shape[-1]
    matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zern = psf.PointSpreadFunction(matrices, N_pix=N_PIX, crop_pix=pix, diversity_coef=np.zeros(N_zern))
    N_trials = 10  # How many data points per RMS case
    N_rms = 200  # How many RMS points to consider

    fig, ax = plt.subplots(1, 1)
    for rms_amplitude, a_min, color in zip([0.1, 0.2, 0.3], [1.0, 0.10, 0.01], ['black', 'red', 'blue']):


        ensquared = calculate_ensquared_energy(PSF_zern, wave, N_trials, N_rms, rms_amplitude,
                                               nominal_scale=SPAXEL_MAS, spaxel_scales=[60], rescale_coeff=a_min)


        ax.scatter(ensquared[0][0], ensquared[0][1], color=color, s=3, label=r'$r=%.2f$' % a_min)
    ax.set_ylim([80, 100])
    ax.set_xlim([0, 500])
    ax.grid(True)
    plt.legend(title='Ratios')
    ax.set_xlabel(r'RMS wavefront [nm]')
    ax.set_ylabel(r'Relative Encircled Energy [per cent]')
    plt.show()

    # maxes = []
    # plt.figure()
    for data, scale, color in zip(ensquared, spaxel_scales, ['red']):
        ax.scatter(data[0], data[1], color='black', s=3, label=scale)






    row_by_row = True
    min_Zernike = 3  # Row number
    max_Zernike = 7
    zernike_triangle = zernike.triangular_numbers(N_levels=max_Zernike)

    zernike_level = 3
    # Initialize the Zernike matrices
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=zernike_level, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.0)

    if row_by_row:  # Select only a specific row (radial order) of Zernike
        # Number of polynomials in that last Zernike row
        poly_in_row = zernike_triangle[zernike_level] - zernike_triangle[zernike_level - 1]
        zernike_matrix = zernike_matrix[:, :, -poly_in_row:]  # select only the high orders
        flat_zernike = flat_zernike[:, -poly_in_row:]

    N_zern = zernike_matrix.shape[-1]  # Update the N_zern after removing Piston and Tilts

    matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zern = psf.PointSpreadFunction(matrices, N_pix=N_PIX, crop_pix=pix, diversity_coef=np.zeros(N_zern))

    # Show some example
    p0, s0 = PSF_zern.compute_PSF(0.1*np.random.normal(0, 1, N_zern))  # Nominal PSF
    # Calculate the EE for the nominal PSF so that you can compare
    EE0 = ensquared_rectangle(p0, pix_scale=SPAXEL_MAS, new_scale=60, plot=True)
    plt.tight_layout()

    ensquared = calculate_ensquared_energy(PSF_zern, wave, N_trials, N_rms, rms_amplitude,
                                           nominal_scale=SPAXEL_MAS, spaxel_scales=spaxel_scales)








    import seaborn as sns

    xy = np.random.normal(0, scale=1, size=(500, 2))
    cent_x, cent_y = np.mean(xy[:, 0]), np.mean(xy[:, 1])

    sns.set_style("white")
    joint = sns.jointplot(x=xy[:, 0], y=xy[:, 1], kind='kde')

    def calculate_marginal_fwhm(ax_marg, mode='X'):
        # We need the axis of the plot where
        # the marginal distribution lives

        path_marg = ax_marg.collections[0].get_paths()[0]
        vert_marg = path_marg.vertices
        N_vert = vert_marg.shape[0]
        # the plot is a closed shape so half of it is just a path along y=0
        if mode=='X':
            xdata = vert_marg[N_vert//2:, 0]
            ydata = vert_marg[N_vert//2:, 1]
        if mode=='Y':   # In Y mode the data is rotated 90deg so X data becomes Y
            xdata = vert_marg[N_vert//2:, 1]
            ydata = vert_marg[N_vert//2:, 0]

        peak = np.max(ydata)
        # as the sampling is constant we will count how many points are
        # above 50% of the peak, and how many below
        # and scale that ratio above/below by the range of the plot
        deltaX = np.max(xdata) - np.min(xdata)      # the range of our data
        above = np.argwhere(ydata >= 0.5 * peak).shape[0]
        below = np.argwhere(ydata < 0.5 * peak).shape[0]
        total = above + below
        # Above / Total = FWHM / Range
        fwhm = above / total * deltaX

        return fwhm

    fx = calculate_marginal_fwhm(joint.ax_marg_x, mode='X')
    fy = calculate_marginal_fwhm(joint.ax_marg_y, mode='Y')


    path_marg_x = joint.ax_marg_x.collections[0].get_paths()[0]
    vert_marg_x = path_marg_x.vertices
    N_x = vert_marg_x.shape[0]
    deltaX =
    plt.plot(vert_marg_x[N_x//2:, 0], vert_marg_x[N_x//2:, 1])
    plt.show()

    #



    fig, ax1 = plt.subplots(1, 1)
    sns.set_style("white")
    contour = sns.kdeplot(xy[:, 0], xy[:, 1], n_levels=10, ax=ax1)
    half_contour = contour.collections[5]

    path = half_contour.get_paths()[0]
    vertices = path.vertices
    ax1.scatter(vertices[:, 0], vertices[:, 1], color='red', s=7)
    half_segments = half_contour.get_segments()[0]
    ax1.scatter(cent_x, cent_y, label='Centroid', color='green')
    ax1.scatter(xy[:, 0], xy[:, 1], color='blue', s=3)
    # ax1.scatter(half_segments[:, 0], half_segments[:, 1], color='red', s=5)
    plt.show()

    radii = np.sqrt((half_segments[:, 0] - cent_x)**2 + (half_segments[:, 1] - cent_y)**2)
    mean_radius = np.mean(radii)

    # ax.scatter(xy[:, 0], xy[:, 1], color='red', s=3)
    # ax.collections[0].set_visible(True)

    segments = col.get_segments()[0]
    plt.scatter(segments[:, 0], segments[:, 1], color='red', s=5)

    plt.show()

    sns.kdeplot(xy[:, 0], xy[:, 1], cmap="Reds", shade=True, bw=.15)
    # sns.plt.show()

    plt.figure()
    plt.scatter(xy[:, 0], xy[:, 1], color='black', s=3)
    cont = plt.contour(xy)
    plt.show()
