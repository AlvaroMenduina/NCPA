import numpy as np

from itertools import tee

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


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