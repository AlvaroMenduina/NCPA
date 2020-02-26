"""




"""

import os
import numpy as np
from numpy.fft import fft2, fftshift, ifft2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import zoom
from skimage.transform import warp, AffineTransform
from time import time

import pycuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import skcuda
import skcuda.linalg as clinalg
import skcuda.fft as cu_fft

import utils
import psf

CENTRAL_OBS = 0.30          # ELT central obscuration


def pupil_mask(xx, yy, rho_aper, rho_obsc, anamorphic=False):

    if anamorphic == False:
        rho = np.sqrt(xx ** 2 + yy ** 2)
    elif anamorphic == True:
        rho = np.sqrt(xx ** 2 + (2 * yy) ** 2)
    pupil = (rho <= rho_aper) & (rho >= rho_obsc)
    return pupil


class SlicerModel(object):
    """
    Object that models the effect of Image Slicers in light propagation
    """

    def __init__(self, slicer_options, N_PIX, spaxel_scale, N_waves, wave0, waveN, wave_ref):

        self.N_slices = slicer_options["N_slices"]                                          # Number os Slices to use

        # Make sure spaxel_per_slice is even! For sampling purposes
        if slicer_options["spaxels_per_slice"] % 2 != 0:
            raise ValueError("Spaxels per slice must be even!")
        self.spaxels_per_slice = slicer_options["spaxels_per_slice"]                        # Size of the slice in Spaxels
        self.pupil_mirror_aperture = slicer_options["pupil_mirror_aperture"]                # Pupil Mirror Aperture
        N_rings = self.spaxels_per_slice / 2
        self.anamorphic = slicer_options["anamorphic"]                                      # Anamorphic Preoptics?
        self.slice_size_mas = self.spaxels_per_slice * spaxel_scale                         # Size of 1 slice in mas
        self.spaxel_scale = spaxel_scale                                                    # Spaxel scale [mas]
        self.N_PIX = N_PIX

        print("\nCreating Image Slicer Model")
        print("     Nominal Wavelength: %.2f microns" % wave0)
        print("     Spaxel Scale: %.2f mas" % spaxel_scale)
        print("     Number of slices: ", self.N_slices)
        print("     Spaxels per slices: ", self.spaxels_per_slice)
        FWHM = utils.compute_FWHM(wave0)
        self.FWHM_ratio = self.slice_size_mas / FWHM            # How much of the FWHM is covered by one slice
        print("     FWHM: %.2f mas" % FWHM)
        print("     One slice covers: %.2f mas / %.2f FWHM" % (self.slice_size_mas, self.FWHM_ratio))

        self.create_pupil_masks(spaxel_scale, N_waves, wave0, waveN, wave_ref)
        self.create_slicer_masks()
        self.create_pupil_mirror_apertures(N_rings=self.pupil_mirror_aperture * N_rings)

        self.compute_peak_nominal_PSFs()

    def compute_peak_nominal_PSFs(self):
        """
        Compute the PEAK of the PSF without aberrations so that we can
        normalize everything by it
        """
        self.nominal_PSFs, self.PEAKs = {}, {}
        for wavelength in self.wave_range:
            pupil_mask = self.pupil_masks[wavelength]
            nominal_pup = pupil_mask * np.exp(1j * 2 * np.pi * pupil_mask / wavelength)
            nominal_PSF = (np.abs(fftshift(fft2(nominal_pup, norm='ortho')))) ** 2
            self.nominal_PSFs[wavelength] = nominal_PSF
            self.PEAKs[wavelength] = np.max(nominal_PSF)
        return

    def create_pupil_masks(self, spaxel_scale, N_waves, wave0, waveN, wave_ref):
        """
        Creates a dictionary of pupil masks for a set of wavelengths
        The pupil masks change in size to account for the variation of PSF size with wavelength
        We set the spaxel_scale for the first wavelength [wave0]
        :param spaxel_scale: spaxel_scale [mas] for wave0
        :param N_waves: number of wavelengths to consider in the interval [wave0, waveN]
        :param wave0: nominal wavelength in [microns]
        :param waveN: maximum wavelength in [microns]
        :param wave_ref: reference wavelength at which we have self.spaxel_scale
        :return:
        """

        self.wave_range = np.linspace(wave0, waveN, N_waves, endpoint=True)
        self.wave_ref = wave_ref
        self.waves_ratio = self.wave_range / wave_ref

        rho_aper = utils.rho_spaxel_scale(spaxel_scale, wavelength=wave0)
        self.diameters = 1/rho_aper
        # check_spaxel_scale(rho_aper=rho_aper, wavelength=wave0)
        rho_obsc = CENTRAL_OBS * rho_aper

        self.rho_aper = rho_aper
        self.rho_obsc = rho_obsc

        self.pupil_masks, self.pupil_masks_fft = {}, {}
        x0 = np.linspace(-1., 1., self.N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x0, x0)

        print("\nCreating Pupil Masks:")
        for i, wave in enumerate(self.waves_ratio):
            wavelength = self.wave_range[i]
            print("Wavelength: %.2f microns" % wavelength)
            # pupil = (rho <= rho_aper / wave) & (rho >= rho_obsc / wave)
            _pupil = pupil_mask(xx, yy, rho_aper / wave, rho_obsc / wave, self.anamorphic)
            mask = np.array(_pupil).astype(np.float32)
            self.pupil_masks[wavelength] = mask
            self.pupil_masks_fft[wavelength] = np.stack(mask * self.N_slices)       # For the GPU calculations
        return

    def create_slicer_masks(self):
        """
        Creates a list of slicer masks that model the finite aperture por each slice
        effectively cropping the PSF at the focal plane
        :return:
        """

        # Check the size of the arrays
        U = self.N_PIX * self.spaxel_scale          # Size in [m.a.s] of the Slicer Plane
        u0 = np.linspace(-U/2., U/2., self.N_PIX, endpoint=True)
        uu, vv = np.meshgrid(u0, u0)

        slice_width = self.slice_size_mas
        self.slicer_masks, self.slice_boundaries = [], []
        self.slicer_masks_fftshift = []
        #TODO: what if N_slices is not odd??
        N_slices_above = (self.N_slices - 1) // 2
        lower_centre = -N_slices_above * slice_width
        for i in range(self.N_slices):
            centre = lower_centre + i * slice_width
            bottom = centre - slice_width/2
            self.slice_boundaries.append(bottom)
            up_mask = vv > (centre - slice_width/2)
            low_mask = vv < (centre + slice_width/2)
            mask = up_mask * low_mask

            self.slicer_masks.append(mask)
            self.slicer_masks_fftshift.append(fftshift(mask))           # For the GPU calculations
        self.slicer_masks_fftshift = np.array(self.slicer_masks_fftshift).astype(np.float32)

        return

    def create_pupil_mirror_apertures(self, N_rings):
        """
        Creates a mask to model the finite aperture of the pupil mirror,
        which effectively introduces fringe effects on the PSF at the exit slit
        :param aperture:
        :return:
        """
        # The number of PSF zeros (or rings) that we see in the Pupil Mirror plane is
        # equal to half the number of spaxels per slice we have defined, along each direction
        # i.e. if we have 20 spaxels_per_slice, we see 10 zeros above and 10 below the PSF core

        N_zeros = self.spaxels_per_slice / 2.
        aperture_ratio = N_rings / N_zeros
        x0 = np.linspace(-1., 1., self.N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x0, x0)

        self.pupil_mirror_mask, self.pupil_mirror_masks_fft = {}, {}

        print("\nCreating Pupil Mirror Apertures:")
        for i, wave in enumerate(self.waves_ratio):
            wavelength = self.wave_range[i]
            print("Wavelength: %.2f microns" % wavelength)
            _pupil = np.abs(yy) <= aperture_ratio / wave
            mask = np.array(_pupil).astype(np.float32)
            self.pupil_mirror_mask[wavelength] = mask
            self.pupil_mirror_masks_fft[wavelength] = np.stack([mask] * self.N_slices)       # For the GPU calculations

        return

    def propagate_pupil_to_slicer(self, wavelength, wavefront):
        """
        Takes the a given wavefront at the PUPIL plane at a given wavelength
        and propagates it to the focal plane, i.e. SLICER plane
        using standard Fourier transforms
        :param wavelength: wavelength to use for the propagation [microns]
        :param wavefront: a wavefront map in [microns]
        :return: complex electric field at the slicer
        """

        # print("Pupil Plane -> Image Slicer Plane")

        pupil_mask = self.pupil_masks[wavelength]
        # wavefront = pupil_mask
        complex_pupil = pupil_mask * np.exp(1j * 2*np.pi * wavefront / wavelength)

        # Propagate to Slicer plane
        complex_slicer = fftshift(fft2(complex_pupil, norm='ortho'))

        return complex_slicer

    def propagate_slicer_to_pupil_mirror(self, complex_slicer):
        """
        Using the SLICER MASKS, it masks the complex field at the SLICER plane
        and propagates each slice to the PUPIL MIRROR plane using inverse Fourier transform
        :param complex_slicer: complex electric field at the SLICER plane
        :return:
        """

        # print("Image Slicer Plane -> Pupil Mirror Plane")
        complex_mirror = []
        for i_slice in range(self.N_slices):        # Loop over the Slices
            mask = self.slicer_masks[i_slice]
            masked_complex_slicer = mask * complex_slicer
            # Pre FFT-Shift to put it back to the format the FFT uses
            _shifted = fftshift(masked_complex_slicer)
            # propagate to Pupil Mirror plane
            complex_mirror.append(ifft2(_shifted, norm='ortho'))
        return complex_mirror

    def propagate_pupil_mirror_to_exit_slit(self, complex_mirror, wavelength):
        """
        Using the PUPIL MIRROR MASK, it propagates each slice to the corresponding
        exit slit
        :param complex_mirror: complex electric field at the PUPIL MIRROR plane [a list of slices]
        :param wavelength: wavelength at which to rescale the PUPIL MIRROR apertures
        """

        # print("Pupil Mirror Plane -> Exit Slits")
        exit_slits = []
        for c_mirror in complex_mirror:
            masked_mirror = self.pupil_mirror_mask[wavelength] * c_mirror
            complex_slit = fftshift(fft2(masked_mirror, norm='ortho'))
            exit_slits.append((np.abs(complex_slit))**2)
        image = np.sum(np.stack(exit_slits), axis=0)

        return exit_slits, image

    def downsample_image(self, image, crop=None):
        """
        Take the image of the exit slit [N_PIX, N_PIX] and apply the pixelation
        so that you have 1 spaxel per slice

        It can handle both anamorphic and non-anamorphic cases

        :param image:
        :return:
        """

        if self.anamorphic is True:

            x_ratio = 1. / self.spaxels_per_slice
            y_ratio = 2. / self.spaxels_per_slice

            down_image = zoom(image, zoom=[x_ratio, y_ratio])

            X, Y = down_image.shape
            if X < Y:
                down_image = down_image[:, Y // 2 - X // 2: Y // 2 + X // 2]

            if crop is not None:
                img = utils.crop_array(down_image, crop=crop)
            else:
                img = down_image

            return img

        else:
            x_ratio = 1. / self.spaxels_per_slice
            down_image = zoom(image, zoom=x_ratio)
            return

    def propagate_gpu_wavelength(self, wavelength, wavefront, N):
        """
        Propagation from Pupil Plane to Exit Slit on the GPU for a single wavelength

        Repeated N times to show how it runs much faster on the GPU when we want to compute
        many PSF images
        :param wavefront:
        :return:
        """
        # It is a pain in the ass to handle the memory properly on the GPU when you have [N_slices, N_pix, N_pix]
        # arrays
        print("\nPropagating on the GPU")
        # GPU memory management
        free, total = cuda.mem_get_info()
        print("Memory Start | Free: %.2f percent" % (free/total*100))
        slicer_masks_gpu = gpuarray.to_gpu(self.slicer_masks_fftshift)
        mirror_mask_gpu = gpuarray.to_gpu(self.pupil_mirror_masks_fft[wavelength])

        plan_batch = cu_fft.Plan((self.N_PIX, self.N_PIX), np.complex64, np.complex64, self.N_slices)

        # Allocate GPU arrays that will be overwritten with skcuda.misc.set_realloc to save memory
        _pupil = np.zeros((self.N_PIX, self.N_PIX), dtype=np.complex64)
        complex_pupil_gpu = gpuarray.to_gpu(_pupil)

        _slicer = np.zeros((self.N_slices, self.N_PIX, self.N_PIX), dtype=np.complex64)
        complex_slicer_gpu = gpuarray.to_gpu(_slicer)

        PSF_images = []
        for i in range(N):
            print(i)

            # Pupil Plane -> Image Slicer
            pupil_mask = self.pupil_masks[wavelength]
            complex_pupil = pupil_mask * np.exp(1j * 2 * np.pi * wavefront[i] / wavelength)
            skcuda.misc.set_realloc(complex_pupil_gpu, np.asarray(complex_pupil, np.complex64))
            cu_fft.fft(complex_pupil_gpu, complex_pupil_gpu, plan_batch)

            # Add N_slices copies to be Masked
            complex_slicer_cpu = complex_pupil_gpu.get()
            complex_slicer_cpu = np.stack([complex_slicer_cpu] * self.N_slices)
            skcuda.misc.set_realloc(complex_slicer_gpu, complex_slicer_cpu)
            clinalg.multiply(slicer_masks_gpu, complex_slicer_gpu, overwrite=True)

            # Image Slicer -> Pupil Mirror
            cu_fft.ifft(complex_slicer_gpu, complex_slicer_gpu, plan_batch, True)
            clinalg.multiply(mirror_mask_gpu, complex_slicer_gpu, overwrite=True)

            # Pupil Mirror -> Exit Slits
            cu_fft.fft(complex_slicer_gpu, complex_slicer_gpu, plan_batch)

            # pycuda.cumath.fabs(complex_slicer_gpu, out=complex_slicer_gpu)

            _slits = complex_slicer_gpu.get()
            slits = np.sum((np.abs(_slits))**2, axis=0)
            PSF_images.append(slits)

            # free, total = cuda.mem_get_info()
            # print("Memory Usage | Free: %.2f percent" % (free / total * 100))

            # free, total = cuda.mem_get_info()
            # print("Memory End | Free: %.2f percent" % (free/total*100))

        # Make sure you clean up the memory so that it doesn't blow up!!
        complex_pupil_gpu.gpudata.free()
        complex_slicer_gpu.gpudata.free()
        slicer_masks_gpu.gpudata.free()
        mirror_mask_gpu.gpudata.free()
        free, total = cuda.mem_get_info()
        print("Memory Final | Free: %.2f percent" % (free / total * 100))

        return fftshift(np.array(PSF_images), axes=(1, 2))

    def propagate_eager(self, wavelength, wavefront):
        """
        'Not-Too-Good' version of the propagation on the GPU (lots of Memory issues...)
        Remove in the future
        :param wavelength:
        :param wavefront:
        :return:
        """

        N = self.N_PIX
        # free, total = cuda.mem_get_info()
        free, total = cuda.mem_get_info()
        print("Free: %.2f percent" %(free/total*100))

        # Pupil Plane -> Image Slicer
        complex_pupil = self.pupil_masks[wavelength] * np.exp(1j * 2 * np.pi * self.pupil_masks[wavelength] / wavelength)
        complex_pupil_gpu = gpuarray.to_gpu(np.asarray(complex_pupil, np.complex64))
        plan = cu_fft.Plan(complex_pupil_gpu.shape, np.complex64, np.complex64)
        cu_fft.fft(complex_pupil_gpu, complex_pupil_gpu, plan, scale=True)

        # Add N_slices copies to be Masked
        complex_slicer_cpu = complex_pupil_gpu.get()
        complex_pupil_gpu.gpudata.free()

        free, total = cuda.mem_get_info()
        print("*Free: %.2f percent" %(free/total*100))

        complex_slicer_cpu = np.stack([complex_slicer_cpu]*self.N_slices)
        complex_slicer_gpu = gpuarray.to_gpu(complex_slicer_cpu)
        slicer_masks_gpu = gpuarray.to_gpu(self.slicer_masks_fftshift)
        clinalg.multiply(slicer_masks_gpu, complex_slicer_gpu, overwrite=True)
        slicer_masks_gpu.gpudata.free()
        free, total = cuda.mem_get_info()
        print("**Free: %.2f percent" %(free/total*100))

       # Slicer -> Pupil Mirror
        plan = cu_fft.Plan((N, N), np.complex64, np.complex64, self.N_slices)
        cu_fft.ifft(complex_slicer_gpu, complex_slicer_gpu, plan, scale=True)
        mirror_mask_gpu = gpuarray.to_gpu(self.pupil_mirror_masks_fft)
        clinalg.multiply(mirror_mask_gpu, complex_slicer_gpu, overwrite=True)

        # Pupil Mirror -> Slits
        cu_fft.fft(complex_slicer_gpu, complex_slicer_gpu, plan)
        slits = complex_slicer_gpu.get()
        complex_slicer_gpu.gpudata.free()
        mirror_mask_gpu.gpudata.free()
        slit = fftshift(np.sum((np.abs(slits))**2, axis=0))

        free, total = cuda.mem_get_info()
        print("***Free: %.2f percent" % (free / total * 100))

        return slit

    def propagate_one_wavelength(self, wavelength, wavefront, plot=False, silent=False):
        """
        Run the propagation from PUPIL Plane to EXIT SLIT Plane for a given wavelength
        :param wavelength:
        :param wavefront:
        """

        if not silent:
            print("\nPropagating Wavelength: %.2f microns" % wavelength)
        complex_slicer = self.propagate_pupil_to_slicer(wavelength=wavelength, wavefront=wavefront)
        complex_mirror = self.propagate_slicer_to_pupil_mirror(complex_slicer)
        exit_slits, image_slit = self.propagate_pupil_mirror_to_exit_slit(complex_mirror, wavelength=wavelength)

        if plot:

            #___________________________________________________________________________________
            # Image Slicer plane
            slicer_size = self.N_PIX * self.spaxel_scale / 2
            slicer_extents = [-slicer_size, slicer_size, -slicer_size, slicer_size]
            zoom_size = self.N_slices * self.slice_size_mas / 2
            slicer_intensity = (np.abs(complex_slicer))**2

            plt.figure()
            plt.imshow(slicer_intensity, extent=slicer_extents)
            self.plot_slicer_boundaries()
            plt.xlim([-zoom_size, zoom_size])
            plt.ylim([-zoom_size, zoom_size])
            plt.colorbar()
            plt.title('Slicer Plane')

            plt.figure()
            plt.imshow(np.log10(slicer_intensity), extent=slicer_extents)
            plt.clim(vmin=-10)
            self.plot_slicer_boundaries()
            plt.xlim([-zoom_size, zoom_size])
            plt.ylim([-zoom_size, zoom_size])
            plt.colorbar()
            plt.title('Slicer Plane [log10]')

            #___________________________________________________________________________________
            # Pupil Mirror plane
            central_slice = (self.N_slices - 1)//2
            pupil_mirror = complex_mirror[central_slice]
            pupil_image = (np.abs(pupil_mirror))**2
            plt.figure()
            plt.imshow(pupil_image, extent=[-1, 1, -1, 1])
            plt.axhline(self.pupil_mirror_aperture, linestyle='--', color='white')
            plt.axhline(-self.pupil_mirror_aperture, linestyle='--', color='white')
            plt.title('Pupil Mirror [Central Slice]')

            plt.figure()
            plt.imshow(np.log10(pupil_image), extent=[-1, 1, -1, 1])
            plt.clim(vmin=-10)
            plt.axhline(self.pupil_mirror_aperture, linestyle='--', color='white')
            plt.axhline(-self.pupil_mirror_aperture, linestyle='--', color='white')
            plt.title('Pupil Mirror [Central Slice]')

            #___________________________________________________________________________________
            # Exit Slit plane

            plt.figure()
            plt.imshow(image_slit / self.PEAKs[wavelength], extent=slicer_extents)
            self.plot_slicer_boundaries()
            plt.xlim([-zoom_size, zoom_size])
            plt.ylim([-zoom_size, zoom_size])
            plt.colorbar()
            plt.title('Exit Slit')

            residual = (image_slit - self.nominal_PSFs[wavelength]) / self.PEAKs[wavelength]
            m_res = min(np.min(residual), -np.max(residual))
            plt.figure()
            plt.imshow(residual, extent=slicer_extents, cmap='bwr')
            plt.xlim([-zoom_size, zoom_size])
            plt.ylim([-zoom_size, zoom_size])
            self.plot_slicer_boundaries()
            plt.colorbar()
            plt.clim(m_res, -m_res)
            plt.title('Exit Slit - No Slicer')

        return complex_slicer, complex_mirror, image_slit, exit_slits

    def plot_slicer_boundaries(self):
        """
        Overlays the boundaries of each SLICE on the plots
        :return:
        """

        min_alpha, max_alpha = 0.15, 0.85
        half_slices = (self.N_slices - 1)//2
        alphas = np.linspace(min_alpha, max_alpha,  half_slices)
        alphas = np.concatenate([alphas, [max_alpha], alphas[::-1]])
        for y, alpha in zip(self.slice_boundaries, alphas):
            plt.axhline(y, linestyle='--', color='white', alpha=alpha)
        return


class HARMONI(SlicerModel):

    def __init__(self, N_PIX, pix, N_waves, wave0, waveN, wave_ref, anamorphic):

        self.pix = pix
        N_slices = 33
        spaxels_per_slice = 16
        slice_mas = 7.0
        spaxel_scale = slice_mas / spaxels_per_slice

        # HARMONI fits approximately 6 rings (at each side) at the Pupil Mirror at 1.5 microns
        # In Python we have 1/2 spaxels_per_slice rings at each side in the Pupil Mirror arrays
        N_rings = spaxels_per_slice / 2
        rings_HARMONI = 6
        pupil_mirror_aperture = rings_HARMONI / N_rings
        slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                          "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": anamorphic}

        SlicerModel.__init__(self, slicer_options=slicer_options, N_PIX=N_PIX, spaxel_scale=spaxel_scale,
                             N_waves=N_waves, wave0=wave0, waveN=waveN, wave_ref=wave_ref)

    def create_actuator_matrices(self, N_act_perD, alpha_pc, plot=False):

        RHO_APER = self.rho_aper
        RHO_OBSC = self.rho_obsc
        N_WAVES = self.wave_range.shape[0]
        WAVE0 = self.wave_range[0]
        WAVEN = self.wave_range[-1]
        WAVEREF = self.wave_ref

        N_actuators = int(N_act_perD / RHO_APER)

        centers_multiwave = psf.actuator_centres_multiwave(N_actuators=N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                           N_waves=N_WAVES, wave0=WAVE0, waveN=WAVEN, wave_ref=WAVEREF)
        self.N_act = len(centers_multiwave[0][0])
        print("%d Actuators" % self.N_act)

        matrices = psf.actuator_matrix_multiwave(centres=centers_multiwave, alpha_pc=alpha_pc, rho_aper=RHO_APER,
                                                          rho_obsc=RHO_OBSC, N_waves=N_WAVES, wave0=WAVE0, waveN=WAVEN,
                                                          wave_ref=WAVEREF, N_PIX=self.N_PIX)

        if self.anamorphic is True:
            print("Transforming Actuator Matrices to match Anamorphism")
            # Absolutely key! We need to apply a transformation to the circular actuator matrices, to make them
            # anamorphic, just like the pupil masks

            sx, sy = 1, 2       # sy = 2 to apply the anamorphic scaling
            transform = AffineTransform(scale=(sx, sy), translation=(0, -self.N_PIX // 2))

            new_matrices = []
            for i, wave in enumerate(self.wave_range):   # loop over wavelengths
                old_matrices = matrices[i]
                old_actuator = old_matrices[0]
                print(old_actuator.shape)
                new_actuator = np.zeros_like(old_actuator)
                for k in range(self.N_act):
                    new_actuator[:, :, k] = warp(image=old_actuator[:, :, k], inverse_map=transform)
                old_matrices[0] = new_actuator
                pupil_bool = self.pupil_masks[wave].copy().astype(bool)
                old_matrices[2] = new_actuator[pupil_bool]
                new_matrices.append(old_matrices)
        else:
            new_matrices = matrices

        if plot:
            for i, wave_r in enumerate(self.waves_ratio):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                circ1 = Circle((0, 0), RHO_APER / wave_r, linestyle='--', fill=None)
                circ2 = Circle((0, 0), RHO_OBSC / wave_r, linestyle='--', fill=None)
                ax.add_patch(circ1)
                ax.add_patch(circ2)
                for c in centers_multiwave[i][0]:
                    ax.scatter(c[0], c[1], color='red', s=10)
                    ax.scatter(c[0], c[1], color='black', s=10)
                ax.set_aspect('equal')
                plt.xlim([-1.1*RHO_APER / wave_r, 1.1*RHO_APER / wave_r])
                plt.ylim([-1.1*RHO_APER / wave_r, 1.1*RHO_APER / wave_r])
                # plt.title('%d actuators' %N_act)
                plt.title('Wavelength: %.2f microns' % self.wave_range[i])

        self.actuator_matrices = {}
        self.actuator_flats = {}
        for i, wave in enumerate(self.wave_range):
            self.actuator_matrices[wave] = new_matrices[i][0]
            self.actuator_flats[wave] = new_matrices[i][2]

        return

    def generate_PSF(self, coef, wavelengths):

        N_samples = coef.shape[0]
        print("\nGenerating %d PSF images" % N_samples)

        N_channels = len(wavelengths)
        dataset = np.empty((N_samples, self.pix, self.pix, N_channels))
        start = time()
        for k in range(N_samples):
            for j, wavelength in enumerate(wavelengths):
                wavefront = np.dot(self.actuator_matrices[wavelength], coef[k])
                _slicer, _mirror, slit, slits = self.propagate_one_wavelength(wavelength, wavefront, silent=True)
                img = self.downsample_image(slit, crop=self.pix)
                dataset[k, :, :, j] = img
        finish = time()
        duration = finish - start
        print("%.2f sec / %.2f sec per PSF" % (duration, duration / N_samples))
        return dataset



class SlicerAnalysis(object):

    def __init__(self, path_save):

        self.path_save = path_save
        return

    def pixelation_central_slice(self, N_slices, spaxels_per_slice, spaxel_mas, rings_we_want, N_PIX,
                                 N_waves, wave0, waveN, wave_ref, anamorphic=False):

        intensity = []
        for rings in rings_we_want:

            N_rings = spaxels_per_slice / 2
            pupil_mirror_aperture = rings / N_rings
            slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                              "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": anamorphic}

            _slicer = SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX, spaxel_scale=spaxel_mas,
                                  N_waves=N_waves, wave0=wave0, waveN=waveN, wave_ref=wave_ref)

            intensity_wave = []
            for wave in _slicer.wave_range:



                complex_slicer, complex_mirror, exit_slit, slits = _slicer.propagate_one_wavelength(wavelength=wave,
                                                                                                    wavefront=0)

                print(exit_slit.shape)
                min_pix = N_PIX//2 - (spaxels_per_slice - 1) // 2
                max_pix = N_PIX//2 + (spaxels_per_slice - 1) // 2

                sliced = exit_slit[min_pix:max_pix, min_pix:max_pix]
                print(np.mean(sliced))
                intensity_wave.append(np.mean(sliced))
            intensity.append(intensity_wave)

        return intensity

    def pixelation_effect(self, N_slices, spaxels_per_slice, spaxel_mas, rings_we_want, N_PIX,
                          N_waves, wave0, waveN, wave_ref, anamorphic, crop):

        img_rings, ring_grid = [], []
        for rings in rings_we_want:

            N_rings = spaxels_per_slice / 2
            pupil_mirror_aperture = rings / N_rings
            slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                              "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": anamorphic}

            _slicer = SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX, spaxel_scale=spaxel_mas,
                                  N_waves=N_waves, wave0=wave0, waveN=waveN, wave_ref=wave_ref)

            imgs_wave = []
            for wave in _slicer.wave_range:

                complex_slicer, complex_mirror, exit_slit, slits = _slicer.propagate_one_wavelength(wavelength=wave,
                                                                                                    wavefront=0)
                img = _slicer.downsample_image(exit_slit, crop=crop)

                print(img.shape)
                imgs_wave.append(img)
            wave_grid = np.concatenate(imgs_wave, axis=0)
            ring_grid.append(wave_grid)
            img_rings.append(imgs_wave)
        ring_grid = np.concatenate(ring_grid, axis=1)

        pix_array = np.array(img_rings)         # [N_rings, N_waves, pix1, pix2]
        for i, wave in enumerate(_slicer.wave_range):
            N_r = len(rings_we_want)
            nom_img = pix_array[0, i]
            PEAK = np.max(nom_img)
            nom_img /= PEAK

            fig, axes = plt.subplots(1, N_r)
            ax = axes[0]
            _image = ax.imshow(nom_img, cmap='hot', extent=[-1, 1, -1, 1])
            ax.set_title('%.1f microns | %d rings' % (wave, rings_we_want[0]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.colorbar(_image, ax=ax, orientation='horizontal')

            for k in np.arange(1, len(rings_we_want)):
                ax = axes[k]
                this_img = pix_array[k, i]
                this_img /= PEAK
                residual = this_img - nom_img
                residual *= 100
                cm = min(np.min(residual), -np.max(residual))
                # cm = -20
                _image = ax.imshow(residual, cmap='bwr', extent=[-1, 1, -1, 1])
                _image.set_clim(cm, -cm)
                ax.set_title('%d rings' % (rings_we_want[k]))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                plt.colorbar(_image, ax=ax, orientation='horizontal')
        plt.show()


        # plt.figure()
        # plt.imshow(ring_grid, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
        # for i, _ring in enumerate(rings_we_want):
        #     x = (0.5 + i) / len(rings_we_want) - 0.0025
        #     plt.text(x=x, y=0.01, s=str(_ring), color='white')
        #
        # for j, _wave in enumerate(_slicer.wave_range):
        #     y = (0.5 + j) / len(_slicer.wave_range) - 0.005
        #     plt.text(x=0.01, y=y, s=str(_wave) + r' $\mu$m', color='white')
        # plt.axes().set_aspect(1 / len(rings_we_want))
        # plt.axes().get_xaxis().set_visible(False)
        # plt.axes().get_yaxis().set_visible(False)

        return img_rings, pix_array


    def fancy_grid(self, N_slices, spaxels_per_slice, spaxel_mas, rings_we_want, N_PIX,
                         N_waves, wave0, waveN, wave_ref, anamorphic=False):


        ring_grid = []
        for rings in rings_we_want:

            N_rings = spaxels_per_slice / 2
            pupil_mirror_aperture = rings / N_rings
            slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                              "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": anamorphic}

            _slicer = SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX, spaxel_scale=spaxel_mas,
                                  N_waves=N_waves, wave0=wave0, waveN=waveN, wave_ref=wave_ref)

            wave_grid = []
            for wave in _slicer.wave_range:

                complex_slicer, complex_mirror, exit_slit, slits = _slicer.propagate_one_wavelength(wavelength=wave,
                                                                                                    wavefront=0)

                minPix_Y = (N_PIX + 1 - 2 * spaxels_per_slice) // 2  # Show 2 slices
                maxPix_Y = (N_PIX + 1 + 2 * spaxels_per_slice) // 2
                minPix_X = (N_PIX + 1 - 2 * N_waves * spaxels_per_slice) // 2
                maxPix_X = (N_PIX + 1 + 2 * N_waves * spaxels_per_slice) // 2

                masked_slit = exit_slit * _slicer.slicer_masks[N_slices // 2]
                masked_slit = masked_slit[minPix_Y: maxPix_Y, minPix_X: maxPix_X]
                # Renormalize because the total intensity changes with wavelength
                masked_slit /= np.max(masked_slit)
                wave_grid.append(masked_slit)
            wave_grid = np.concatenate(wave_grid, axis=0)
            ring_grid.append(wave_grid)
        ring_grid = np.concatenate(ring_grid, axis=1)

        plt.figure()
        plt.imshow(ring_grid, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
        for i, _ring in enumerate(rings_we_want):
            x = (0.5 + i) / len(rings_we_want) - 0.0025
            plt.text(x=x, y=0.01, s=str(_ring), color='white')

        for j, _wave in enumerate(_slicer.wave_range):
            y = (0.5 + j) / len(_slicer.wave_range) - 0.005
            plt.text(x=0.01, y=y, s=str(_wave) + r' $\mu$m', color='white')
        plt.axes().set_aspect(1 / len(rings_we_want))
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)

        return ring_grid


    def show_propagation(self, N_slices, spaxels_per_slice, spaxel_mas, rings_we_want, N_PIX,
                         N_waves, wave0, waveN, wave_ref, anamorphic=False):


        N_rings = spaxels_per_slice / 2
        pupil_mirror_aperture = rings_we_want / N_rings
        slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                          "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": anamorphic}

        _slicer = SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX, spaxel_scale=spaxel_mas,
                              N_waves=N_waves, wave0=wave0, waveN=waveN, wave_ref=wave_ref)


        for wave in _slicer.wave_range:

            complex_slicer, complex_mirror, exit_slit, slits = _slicer.propagate_one_wavelength(wavelength=wave,
                                                                                                wavefront=0)

            masked_slicer = (np.abs(complex_slicer)) ** 2 * _slicer.slicer_masks[N_slices // 2]
            masked_slicer /= np.max(masked_slicer)
            minPix_Y = (N_PIX + 1 - 2 * spaxels_per_slice) // 2  # Show 2 slices
            maxPix_Y = (N_PIX + 1 + 2 * spaxels_per_slice) // 2
            minPix_X = (N_PIX + 1 - 6 * spaxels_per_slice) // 2
            maxPix_X = (N_PIX + 1 + 6 * spaxels_per_slice) // 2
            masked_slicer = masked_slicer[minPix_Y:maxPix_Y, minPix_X:maxPix_X]

            dX, dY = maxPix_X - minPix_X, maxPix_Y - minPix_Y           # How many pixels in the masked window
            masX, masY = dX * spaxel_mas, dY * spaxel_mas
            masked_extent = [-masX/2, masX/2, -masY/2, masY/2]

            plt.figure()
            plt.imshow(masked_slicer, cmap='jet', extent=masked_extent)
            plt.colorbar(orientation='horizontal')
            plt.title('HARMONI Slicer: Central Slice @%.2f microns' % wave)
            plt.xlabel(r'X [m.a.s]')
            plt.ylabel(r'Y [m.a.s]')
            plt.savefig(os.path.join(self.path_save, 'Central_Slice_%.2fmicrons_%drings.png' % (wave, rings_we_want)))
            # plt.show()

            masked_pupil_mirror = (np.abs(complex_mirror[N_slices // 2])) ** 2 * _slicer.pupil_mirror_mask[wave]
            masked_pupil_mirror /= np.max(masked_pupil_mirror)
            plt.figure()
            plt.imshow(np.log10(masked_pupil_mirror), cmap='jet')
            plt.colorbar()
            plt.clim(vmin=-4)
            plt.title('Pupil Mirror: Aperture %.2f PSF zeros' % rings_we_want)
            plt.savefig(os.path.join(self.path_save, 'Pupil_Mirror_%.2fmicrons_%drings.png' % (wave, rings_we_want)))
            # plt.show()

            masked_slit = exit_slit * _slicer.slicer_masks[N_slices // 2]
            masked_slit = masked_slit[minPix_Y: maxPix_Y, minPix_X: maxPix_X]
            plt.figure()
            plt.imshow(masked_slit, cmap='jet', extent=masked_extent)
            plt.colorbar(orientation='horizontal')
            plt.xlabel(r'X [m.a.s]')
            plt.ylabel(r'Y [m.a.s]')
            plt.title('Exit Slit @%.2f microns (Pupil Mirror: %.2f PSF zeros)' % (wave, rings_we_want))
            plt.savefig(os.path.join(self.path_save, 'Exit_Slit_%.2fmicrons_%drings.png' % (wave, rings_we_want)))

        return


class SlicerPSFGenerator(object):

    def __init__(self, slicer_model, N_actuators, alpha_pc, N_PIX, crop_pix, plot):


        print("\nCreating PSF Generator")
        self.slicer_model = slicer_model
        self.N_PIX = N_PIX
        self.pix = crop_pix

        matrices = self.create_actuator_matrices(N_actuators, alpha_pc, N_PIX, plot)
        self.actuator_matrices = [x[0] for x in matrices]
        self.pupil_masks = [x[1] for x in matrices]
        self.actuator_flats = [x[2] for x in matrices]

        return

    def create_actuator_matrices(self, N_actuators, alpha_pc, N_PIX, plot=False):


        RHO_APER = self.slicer_model.rho_aper
        RHO_OBSC = self.slicer_model.rho_obsc
        N_WAVES = self.slicer_model.wave_range.shape[0]
        WAVE0 = self.slicer_model.wave_range[0]
        WAVEN = self.slicer_model.wave_range[-1]
        WAVEREF = self.slicer_model.wave_ref

        centers_multiwave = psf.actuator_centres_multiwave(N_actuators=N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                           N_waves=N_WAVES, wave0=WAVE0, waveN=WAVEN, wave_ref=WAVEREF)
        self.N_act = len(centers_multiwave[0][0])
        print(self.N_act)

        actuator_matrices = psf.actuator_matrix_multiwave(centres=centers_multiwave, alpha_pc=alpha_pc, rho_aper=RHO_APER,
                                                          rho_obsc=RHO_OBSC, N_waves=N_WAVES, wave0=WAVE0, waveN=WAVEN,
                                                          wave_ref=WAVEREF, N_PIX=N_PIX)

        if plot:
            for i, wave_r in enumerate(self.slicer_model.waves_ratio):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                circ1 = Circle((0, 0), RHO_APER / wave_r, linestyle='--', fill=None)
                circ2 = Circle((0, 0), RHO_OBSC / wave_r, linestyle='--', fill=None)
                ax.add_patch(circ1)
                ax.add_patch(circ2)
                for c in centers_multiwave[i][0]:
                    ax.scatter(c[0], c[1], color='red', s=10)
                    ax.scatter(c[0], c[1], color='black', s=10)
                ax.set_aspect('equal')
                plt.xlim([-1.1*RHO_APER / wave_r, 1.1*RHO_APER / wave_r])
                plt.ylim([-1.1*RHO_APER / wave_r, 1.1*RHO_APER / wave_r])
                # plt.title('%d actuators' %N_act)
                plt.title('Wavelength: %.2f microns' % self.slicer_model.wave_range[i])

        return actuator_matrices

    def generate_PSF(self, coef, wavelengths):

        N_samples = coef.shape[0]
        print("\nGenerating %d PSF images" % N_samples)

        N_channels = len(wavelengths)
        dataset = np.empty((N_samples, self.pix, self.pix, N_channels))

        for k in range(N_samples):
            print("     PSF #%d" % (k + 1))

            for j, wavelength in enumerate(wavelengths):
                wavefront = np.dot(self.actuator_matrices[j], coef[k])
                _slicer, _mirror, slit, slits = self.slicer_model.propagate_one_wavelength(wavelength, wavefront)
                crop_slit = utils.crop_array(slit, self.pix)
                dataset[k, :, :, j] = crop_slit / self.slicer_model.PEAKs[wavelength]
        return dataset

    def generate_PSF_gpu(self, coef, wavelength, crop=False):

        N_samples = coef.shape[0]
        print("\nGenerating %d PSF images on the GPU" % N_samples)
        j = np.argwhere(self.slicer_model.wave_range == wavelength)[0][0]

        wavefronts = []
        for k in range(N_samples):
            _phase = np.dot(self.actuator_matrices[j], coef[k])
            wavefronts.append(_phase)

        dataset = self.slicer_model.propagate_gpu_wavelength(wavelength, wavefronts, N_samples)
        dataset /= self.slicer_model.PEAKs[wavelength]

        if crop is True:
            dataset = utils.crop_array(dataset, self.pix)

        return dataset



    ### Old Bits
    #
    # def downsample_slits_old(self, exit_slits):
    #
    #     if self.anamorphic is True:
    #
    #         central_pix = self.N_PIX // 2
    #         pix_per_slice = self.spaxels_per_slice
    #
    #         # Average along the slice
    #         y_pix = self.spaxels_per_slice // 2
    #         N_y = self.N_PIX // y_pix
    #
    #         # TODO: what if N_slices is not odd??
    #         N_slices_above = (self.N_slices - 1) // 2
    #         lower_centre = central_pix - N_slices_above * pix_per_slice
    #         down_slits = []
    #         for i in range(self.N_slices):
    #             centre = lower_centre + i * pix_per_slice
    #             bottom = centre - pix_per_slice // 2
    #             top = centre + pix_per_slice // 2
    #             _slit = exit_slits[i]
    #             sliced_slit = _slit[bottom:top, :]
    #
    #             down_slit = []
    #             for k in range(N_y):
    #                 bott = k * y_pix
    #                 topp = (k + 1) * y_pix
    #                 down_slit.append(np.sum(sliced_slit[:, bott:topp]))
    #             down_slits.append(down_slit)
    #
    #         down_slits = np.stack(down_slits)
    #         minY = N_y // 2 - (self.N_slices - 1)//2 - 1
    #         maxY = N_y // 2 + (self.N_slices - 1)//2
    #         down_slits = down_slits[:, minY:maxY]
    #
    #         return down_slits
    #
    #     else:
    #
    #         down_slits = 0
    #         return down_slits




