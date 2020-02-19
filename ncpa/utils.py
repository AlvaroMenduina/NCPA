import os
import numpy as np
import matplotlib.pyplot as plt

from itertools import tee
from pyfiglet import Figlet
from datetime import date


def print_title(message='N C P A', font=None, random_font=False):

    nice_fonts_list = ['lean', 'slant', 'alligator', 'alligator2',
                       'c1______', 'colossal', 'utopiab', 'mayhem_d']

    # Check the day of the week
    today = date.today()
    day = int(today.strftime("%d"))

    if font is not None:
        custom_fig = Figlet(font=font)
        print(custom_fig.renderText(message))

    else:

        if random_font is False:
            # Randomly select the font according to day
            i_font = day % len(nice_fonts_list) - 1
            custom_fig = Figlet(font=nice_fonts_list[i_font])
            print(custom_fig.renderText(message))
        else:
            custom_fig = Figlet()
            all_fonts = custom_fig.getFonts()
            i_font = np.random.choice(len(all_fonts))
            print(all_fonts[i_font])
            custom_fig = Figlet(font=all_fonts[i_font])
            print(custom_fig.renderText(message))
    return

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def list_files_in_directory(directory):

    onlyfiles = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return onlyfiles

def find_substring(list, substring):
    match = [name for name in list if substring in name]
    return match

ELT_DIAM = 39
MILIARCSECS_IN_A_RAD = 206265000

def rho_spaxel_scale(spaxel_scale=4.0, wavelength=1.0):
    """
    Compute the aperture radius necessary to have a
    certain SPAXEL SCALE [in mas] at a certain WAVELENGTH [in microns]

    That would be the aperture radius in an array ranging from [-1, 1] in physical length
    For example, if rho = 0.5, then the necessary aperture is a circle of half the size of the array

    We can use the inverse of that to get the "oversize" in physical units in our arrays to match a given scale
    :param spaxel_scale: [mas]
    :param wavelength: [microns]
    :return:
    """

    scale_rad = spaxel_scale / MILIARCSECS_IN_A_RAD
    rho = scale_rad * ELT_DIAM / (wavelength * 1e-6)
    return rho


def check_spaxel_scale(rho_aper, wavelength):
    """
    Checks the spaxel scale at a certain wavelength, for a given aperture radius
    defined for a [-1, 1] physical array
    :param rho_aper: radius of the aperture, relative to an array of size [-1, 1]
    :param wavelength: wavelength of interest (the PSF grows in size with wavelength, changing the spaxel scale)
    :return:
    """

    SPAXEL_RAD = rho_aper * wavelength / ELT_DIAM * 1e-6
    SPAXEL_MAS = SPAXEL_RAD * MILIARCSECS_IN_A_RAD
    print('%.2f mas spaxels at %.2f microns' %(SPAXEL_MAS, wavelength))
    return

def compute_FWHM(wavelength):
    """
    Compute the Full Width Half Maximum of the PSF at a given Wavelength
    in miliarcseconds, for the ELT
    :param wavelength: [microns]
    :return:
    """
    FWHM_RAD = wavelength * 1e-6 / ELT_DIAM          # radians
    FWHM_MAS = FWHM_RAD * MILIARCSECS_IN_A_RAD
    return FWHM_MAS


def crop_array(array, crop):
    """
    Crops an array or datacube for various cases of shapes to a smaller dimesion
    Typically used to zoom in for the PSF arrays
    :param array: can be [Pix, Pix] or [:, Pix, Pix] or [:, Pix, Pix, :]
    :param crop:
    :return:
    """
    shape = array.shape

    if len(shape) == 2:         # Classic [Pix, Pix] array
        PIX = array.shape[0]
        if crop > PIX:
            raise ValueError("Array is only %d x %d pixels. Trying to crop to %d x %d" % (PIX, PIX, crop, crop))
        min_crop = (PIX + 1 - crop) // 2
        max_crop = (PIX + 1 + crop) // 2
        array_crop = array[min_crop:max_crop, min_crop:max_crop]

    if len(shape) == 3:         # [N_PSF, Pix, Pix] array
        N_PSF, PIX = array.shape[0], array.shape[1]
        if crop > PIX:
            raise ValueError("Array is only %d x %d pixels. Trying to crop to %d x %d" % (PIX, PIX, crop, crop))
        min_crop = (PIX + 1 - crop) // 2
        max_crop = (PIX + 1 + crop) // 2
        array_crop = array[:, min_crop:max_crop, min_crop:max_crop]

    if len(shape) == 4:         # [N_PSF, Pix, Pix, N_channels] array
        N_PSF, PIX = array.shape[0], array.shape[1]
        if crop > PIX:
            raise ValueError("Array is only %d x %d pixels. Trying to crop to %d x %d" % (PIX, PIX, crop, crop))
        min_crop = (PIX + 1 - crop) // 2
        max_crop = (PIX + 1 + crop) // 2
        array_crop = array[:, min_crop:max_crop, min_crop:max_crop, :]

    return array_crop


def plot_images(PSF_datacube, N_images=5):
    """
    Plots a given number of PSF images
    :param PSF_datacube:
    :param N_images:
    :return:
    """

    for i in range(N_images):

        cmap = 'hot'
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1 = plt.subplot(1, 2, 1)
        img1 = ax1.imshow(PSF_datacube[i, :, :, 0], cmap=cmap)
        ax1.set_title(r'Nominal PSF')
        # img1.set_clim(0, 1)
        plt.colorbar(img1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(1, 2, 2)
        img2 = ax2.imshow(PSF_datacube[i, :, :, 1], cmap=cmap)
        ax2.set_title(r'Diversity PSF')
        # img2.set_clim(0, 1)
        plt.colorbar(img2, ax=ax2, orientation='horizontal')

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

def plot_actuator_commands(commands, centers, rho_aper, PIX, cmap='Reds', delta0=4, min_val=0.9):

    cent, delta = centers
    x = np.linspace(-1.25*rho_aper, 1.25*rho_aper, PIX, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    image = np.zeros((PIX, PIX))
    for i, (xc, yc) in enumerate(cent):
        act_mask = (xx - xc)**2 + (yy - yc)**2 <= (delta/delta0)**2
        image += commands[i] * act_mask
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.colorbar()
    plt.clim(min_val*image[np.nonzero(image)].min(), image.max())
    plt.title(r'Residual Error Command')

    return

def show_wavefronts(PSF_model, coefs, rho_aper, images=(2, 3), cmap='jet'):
    """
    Show examples of wavefronts

    :param PSF_model: PSF model to compute the wavefront (can be Zernike or Actuators)
    :param coefs: coefficients of the wavefront
    :param rho_aper: aperture radius relative to [-1, 1] line. Used to clip the images
    :param images: (n_rows, n_columns)
    :param cmap: colormap
    :return:
    """
    n_rows, n_cols = images
    f, (axes) = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            k = n_cols * i + j
            print(k)
            ax = axes[i][j]

            wavefront = np.dot(PSF_model.model_matrix, coefs[k])
            cmin = min(np.min(wavefront), -np.max(wavefront))

            img = ax.imshow(wavefront, cmap=cmap, extent=[-1, 1, -1, 1])
            ax.set_title('Wavefront [waves]')
            ax.set_xlim([-1.1 * rho_aper, 1.1 * rho_aper])
            ax.set_ylim([-1.1 * rho_aper, 1.1 * rho_aper])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            img.set_clim(cmin, -cmin)
            plt.colorbar(img, ax=ax, orientation='horizontal')

    return


def show_wavefronts_grid(PSF_model, coefs, rho_aper,
                         images=(2, 4), cmap='jet', title=None):
    """
    Variation on "show_wavefront". Instead of creating a subplot for each image,
    this code combines all wavefront maps into a single image
    This helps avoid having multiple subtitles, colorbars, etc

    :param PSF_model: PSF model to compute the wavefront (can be Zernike or Actuators)
    :param coefs: coefficients of the wavefront
    :param rho_aper: aperture radius relative to [-1, 1] line. Used to clip the images
    :param images: (n_rows, n_columns)
    :param cmap: colormap
    :param title: str containing a customized title
    :return:
    """
    PIX = PSF_model.model_matrix.shape[0]
    size = int(PIX * 1.1*rho_aper)
    minpix = (PIX + 1 - size) // 2
    maxpix = (PIX + 1 + size) // 2
    n_rows, n_cols = images
    display_grid = np.zeros((size * n_rows, n_cols * size))
    for i in range(n_rows):
        for j in range(n_cols):
            k = n_cols * i + j
            wavefront = np.dot(PSF_model.model_matrix, coefs[k])
            crop_wavefront = wavefront[minpix:maxpix, minpix:maxpix]
            display_grid[i * size: (i + 1) * size, j * size: (j + 1) * size] = crop_wavefront

    fig, ax = plt.subplots()
    if title is None:
        plt.title(r'Wavefront [waves]')
    else:
        plt.title(title)
    plt.grid(False)
    imag = ax.imshow(display_grid, aspect='equal', cmap=cmap)
    cmin = min(np.min(display_grid), -np.max(display_grid))
    imag.set_clim(cmin, -cmin)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.colorbar(imag, orientation='horizontal')

    return

