

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time

import pycuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

import utils
import slicers

utils.print_title(message='\nN C P A', font=None, random_font=False)

if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # ================================================================================================================ #
    #                                      Speed Comparison on the GPU                                                 #
    # ================================================================================================================ #

    slicer_options = {"N_slices": 21, "spaxels_per_slice": 32,
                      "pupil_mirror_aperture": 0.85, "anamorphic": True}
    N_PIX = 1024
    spaxel_mas = 0.25        # to get a decent resolution

    slicer = slicers.SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX, spaxel_scale=spaxel_mas,
                                 N_waves=2, wave0=1.5, waveN=2.0, wave_ref=1.5)

    print("\n---------------------------------------------------------------------")
    print("Running on the CPU")
    N_PSF = 10
    cpu_start = time()
    for i in range(N_PSF):
        complex_slicer, complex_mirror, exit_slit, slits = slicer.propagate_one_wavelength(wavelength=1.5, wavefront=0, plot=False)
    cpu_end = time()
    cpu_time = cpu_end - cpu_start

    print("Time to propagate %d PSFs: %.2f seconds" % (N_PSF, cpu_time))
    print("Time to propagate 1 PSF (%d slices): %.2f seconds" % (slicer.N_slices, cpu_time / N_PSF))
    print("Time to propagate 1 slice: %.2f seconds" % (cpu_time / N_PSF / slicer.N_slices))

    N_PSF = 10
    gpu_start = time()
    exit_slits = slicer.propagate_gpu_wavelength(wavelength=1.5, wavefront=np.zeros((N_PIX, N_PIX)), N=N_PSF)
    gpu_end = time()
    gpu_time = gpu_end - gpu_start
    print("Time to propagate %d PSFs: %.2f seconds" % (N_PSF, gpu_time))
    print("Time to propagate 1 PSF (%d slices): %.2f seconds" % (slicer.N_slices, gpu_time / N_PSF))
    print("Time to propagate 1 slice: %.2f seconds" % (gpu_time / N_PSF / slicer.N_slices))

    free, total = cuda.mem_get_info()
    print("Free: %.2f percent" % (free / total * 100))

    plt.figure()
    plt.imshow(exit_slits[0])
    plt.colorbar()
    plt.show()

    # ================================================================================================================ #
    #                               Generic Slicer Analysis
    # ================================================================================================================ #

    N_slices = 25
    spaxels_per_slice = 32
    slice_mas = 7
    spaxel_mas = slice_mas / spaxels_per_slice

    """ (1) Impact of the Pupil Mirror Aperture in the number of Fringes"""

    cmap = 'jet'

    minPix_Y = (N_PIX + 1 - 2 * spaxels_per_slice) // 2  # Show 2 slices
    maxPix_Y = (N_PIX + 1 + 2 * spaxels_per_slice) // 2
    minPix_X = (N_PIX + 1 - 6 * spaxels_per_slice) // 2
    maxPix_X = (N_PIX + 1 + 6 * spaxels_per_slice) // 2

    dX, dY = maxPix_X - minPix_X, maxPix_Y - minPix_Y  # How many pixels in the masked window
    masX, masY = dX * spaxel_mas, dY * spaxel_mas

    slits_img, mirrors_img, psf_img = [], [], []
    psf_dif = []
    ring_list = np.arange(3, spaxels_per_slice//2)[::-1]
    for rings in ring_list:

        N_rings = spaxels_per_slice / 2
        # Select how many Fringes we want
        rings_we_want = rings
        pupil_mirror_aperture = rings_we_want / N_rings

        N_PIX = 2048
        wave0, waveN = 1.5, 3.0
        N_waves = 1

        slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                          "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": False}

        SLICER = slicers.SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX,
                                      spaxel_scale=spaxel_mas, N_waves=N_waves, wave0=wave0, waveN=waveN, wave_ref=wave0)


        for wave in SLICER.wave_range:
            complex_slicer, complex_mirror, exit_slit, slits = SLICER.propagate_one_wavelength(wavelength=wave,
                                                                                                wavefront=0)

            masked_pupil_mirror = (np.abs(complex_mirror[N_slices // 2])) ** 2 * SLICER.pupil_mirror_mask[wave]
            masked_pupil_mirror /= np.max(masked_pupil_mirror)
            plt.figure()
            plt.imshow(np.log10(masked_pupil_mirror), cmap=cmap)
            plt.colorbar()
            plt.clim(vmin=-4)
            plt.title('Pupil Mirror: Aperture %.2f PSF zeros' % rings_we_want)

            cropped_mirror = utils.crop_array(masked_pupil_mirror, crop=N_PIX)
            mirrors_img.append(cropped_mirror)

            masked_slit = exit_slit * SLICER.slicer_masks[N_slices // 2]
            masked_slit = masked_slit[minPix_Y: maxPix_Y, minPix_X: maxPix_X]
            slits_img.append(masked_slit)

            cropped_psf = utils.crop_array(exit_slit, crop=N_PIX//8)
            psf_img.append(cropped_psf)
            # plt.figure()
            # plt.imshow(masked_slit, cmap=cmap, extent=masked_extent)
            # plt.colorbar(orientation='horizontal')
            # plt.title('Exit Slit @%.2f microns (Pupil Mirror: %.2f PSF zeros)' % (wave, rings_we_want))
            # plt.xlabel('X [m.a.s]')
            # plt.ylabel('Y [m.a.s]')

        # nom = cropped_psf

        diff = cropped_psf - nom
        psf_dif.append(diff)

    R = len(ring_list)
    masked_extent = [-R*masX / 2, R*masX / 2, -masY / 2, masY / 2]
    mirrors_img = np.concatenate(mirrors_img, axis=1)
    slits_img = np.concatenate(slits_img, axis=1)
    psf_img = np.concatenate(psf_img, axis=1)

    plt.figure()
    plt.imshow(np.log10(mirrors_img), cmap=cmap)
    plt.colorbar(orientation='horizontal')
    plt.clim(vmin=-4)
    plt.title('Pupil Mirror Apertures @%.2f microns' % wave0)

    plt.figure()
    plt.imshow(slits_img, cmap=cmap, extent=masked_extent)
    plt.colorbar(orientation='horizontal')
    plt.title('Exit Slits @%.2f microns' % (wave0))
    plt.xlabel('X [m.a.s]')
    plt.ylabel('Y [m.a.s]')

    E = psf_img.shape[0] / 2 * spaxel_mas
    # mas = 20
    plt.figure()
    plt.imshow(psf_img, cmap=cmap, extent=[-R*E, R*E, -E, E])
    plt.colorbar(orientation='horizontal')
    plt.title('Exit PSFs @%.2f microns' % (wave0))
    plt.xlabel('X [m.a.s]')
    plt.ylabel('Y [m.a.s]')

    for diff in psf_dif:
        m = min(np.min(diff), -np.max(diff))
        plt.figure()
        plt.imshow(diff, cmap='bwr')
        plt.clim(m, -m)
        plt.colorbar()

    plt.show()

    directory = os.path.join(os.getcwd(), 'Slicers')
    analysis = slicers.SlicerAnalysis(path_save=directory)

    grid = analysis.fancy_grid(N_slices=15, spaxels_per_slice=64, spaxel_mas=slice_mas/64, rings_we_want=[5, 6, 7],
                              N_PIX=4096, N_waves=5, wave0=1.5, waveN=2.5, wave_ref=1.5, anamorphic=False)
    plt.show()


    # ================================================================================================================ #
    #                                    HARMONI comparison                                                            #
    # ================================================================================================================ #

    # First we have to match the Slice Size to have a proper clipping at the Slicer Plane
    # That comes from the product spaxels_per_slice * spaxel_mas

    N_slices = 31
    spaxels_per_slice = 32
    spaxel_mas = 0.21875  # to get a decent resolution

    # HARMONI fits approximately 6 rings (at each side) at the Pupil Mirror at 1.5 microns
    # that would
    # In Python we have 1/2 spaxels_per_slice rings at each side in the Pupil Mirror arrays
    N_rings = spaxels_per_slice / 2
    rings_we_want = 3
    pupil_mirror_aperture = rings_we_want / N_rings

    N_PIX = 1024
    wave0, waveN = 1.5, 3.0

    slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                      "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": True}

    HARMONI = slicers.SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX,
                                  spaxel_scale=spaxel_mas, N_waves=1, wave0=wave0, waveN=waveN, wave_ref=1.5)

    cmap = 'jet'

    for wave in HARMONI.wave_range:

        complex_slicer, complex_mirror, exit_slit, slits = HARMONI.propagate_one_wavelength(wavelength=wave, wavefront=0)

        masked_slicer = (np.abs(complex_slicer)) ** 2 * HARMONI.slicer_masks[N_slices // 2]
        masked_slicer /= np.max(masked_slicer)
        minPix_Y = (N_PIX + 1 - 2 * spaxels_per_slice) // 2         # Show 2 slices
        maxPix_Y = (N_PIX + 1 + 2 * spaxels_per_slice) // 2
        minPix_X = (N_PIX + 1 - 6 * spaxels_per_slice) // 2
        maxPix_X = (N_PIX + 1 + 6 * spaxels_per_slice) // 2
        masked_slicer = masked_slicer[minPix_Y:maxPix_Y, minPix_X:maxPix_X]

        dX, dY = maxPix_X - minPix_X, maxPix_Y - minPix_Y  # How many pixels in the masked window
        masX, masY = dX * spaxel_mas, dY * spaxel_mas
        masked_extent = [-masX / 2, masX / 2, -masY / 2, masY / 2]

        plt.figure()
        plt.imshow(masked_slicer, cmap=cmap, extent=masked_extent)
        plt.colorbar(orientation='horizontal')
        plt.title('HARMONI Slicer: Central Slice @%.2f microns' % wave)
        plt.xlabel('X [m.a.s]')
        plt.ylabel('Y [m.a.s]')
        # plt.show()

        masked_pupil_mirror = (np.abs(complex_mirror[N_slices // 2])) ** 2 * HARMONI.pupil_mirror_mask[wave]
        masked_pupil_mirror /= np.max(masked_pupil_mirror)
        plt.figure()
        plt.imshow(np.log10(masked_pupil_mirror), cmap=cmap, extent=masked_extent)
        plt.colorbar()
        plt.clim(vmin=-4)
        plt.title('Pupil Mirror: Aperture %.2f PSF zeros' % rings_we_want)

        # plt.show()

        masked_slit = exit_slit * HARMONI.slicer_masks[N_slices // 2]
        masked_slit = masked_slit[minPix_Y: maxPix_Y, minPix_X: maxPix_X]
        plt.figure()
        plt.imshow(masked_slit, cmap=cmap, extent=masked_extent)
        plt.colorbar(orientation='horizontal')
        plt.title('Exit Slit @%.2f microns (Pupil Mirror: %.2f PSF zeros)' % (wave, rings_we_want))
        plt.xlabel('X [m.a.s]')
        plt.ylabel('Y [m.a.s]')

        E = N_PIX/2 * spaxel_mas
        mas = 20
        plt.figure()
        plt.imshow(exit_slit, cmap=cmap, extent=[-E, E, -E, E])
        plt.xlim([-mas, mas])
        plt.ylim([-mas, mas])
        plt.title('PSF @%.2f microns' % wave)
        plt.xlabel('X [m.a.s]')
        plt.ylabel('Y [m.a.s]')
        # ax.set_aspect(0.5)
        # plt.colorbar(orientation='horizontal')

    plt.show()

    img = HARMONI.downsample_image(exit_slit)
    from scipy.ndimage import zoom
    img_z = zoom(img, zoom=0.5)
    plt.imshow(np.log(img))
    plt.colorbar()
    plt.show()

    # ================================================================================================================ #
    #                                       Convolution trick
    # ================================================================================================================ #
    from scipy.signal import convolve2d, convolve
    from numpy.fft import fft2, fftshift

    pi_s = HARMONI.slicer_masks[N_slices // 2]
    pi_m = HARMONI.pupil_mirror_mask[wave0]

    fpi = fftshift(fft2(pi_m, norm='ortho'))

    kernel = convolve2d(fpi, pi_s, mode='same')
    kernel_conj = convolve2d(np.conj(fpi), pi_s, mode='same')
    kern_psf = kernel * kernel_conj

    plt.figure()
    plt.imshow(np.real(fpi))
    plt.colorbar()

    plt.figure()
    plt.imshow(np.imag(fpi))
    plt.colorbar()

    plt.figure()
    plt.imshow(np.abs(fpi))
    plt.colorbar()

    plt.show()

    psf_kern = kernel * np.conj(kernel)
    plt.figure()
    plt.imshow(np.real(psf_kern))
    plt.colorbar()

    plt.show()

    pup = HARMONI.pupil_masks[wave0]
    pup_f = pup * np.exp(1j * pup)
    fpup = pi_s * fftshift(fft2(pup_f))
    P = convolve2d(fpi, fpup)

    kern_pup = kernel * fpup

    plt.figure()
    plt.imshow(np.real(P))
    plt.colorbar()

    plt.figure()
    plt.imshow(np.abs(P))
    plt.colorbar()
    plt.show()


    PSF = (np.abs(P))**2

    plt.figure()
    plt.imshow(PSF)
    plt.colorbar()
    plt.show()

    ###
    x = np.linspace(-2, 2, 1024)
    f = 4
    sinc = np.sin(np.pi * f * x) / (np.pi * f * x)
    plt.plot(x, sinc)
    # plt.show()

    h = np.abs(x) < 1
    # plt.plot(x, h)

    p = convolve(sinc, h, mode='same')
    p /= np.max(p)
    plt.figure()
    plt.plot(x, h)
    plt.plot(x, p)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(x, sinc)
    ax1.set_xlabel(r'X[ ]')

    ax2.plot(x, sinc)
    ax2.set_xlabel(r'X[ ]')


    # ================================================================================================================ #
    #                                               ANALYSIS                                                           #
    # ================================================================================================================ #

    directory = os.path.join(os.getcwd(), 'Slicers')
    analysis = slicers.SlicerAnalysis(path_save=directory)

    """ (1) Impact of Pupil Mirror Aperture """
    mas_slice = 7.0
    spaxels_per_slice = 32
    spaxel_mas = mas_slice / spaxels_per_slice
    for rings in np.arange(2, 3):
        analysis.show_propagation(N_slices=15, spaxels_per_slice=spaxels_per_slice, spaxel_mas=spaxel_mas,
                                  rings_we_want=rings, N_PIX=2048, N_waves=1, wave0=1.5, waveN=1.5, wave_ref=1.5,
                                  anamorphic=True)

    plt.show()

    """ (2) Impact of Wavelength """
    analysis.show_propagation(N_slices=15, spaxels_per_slice=spaxels_per_slice, spaxel_mas=spaxel_mas, rings_we_want=3,
                              N_PIX=2048, N_waves=6, wave0=1.5, waveN=3.0, wave_ref=1.5, anamorphic=True)

    grid = analysis.fancy_grid(N_slices=15, spaxels_per_slice=64, spaxel_mas=mas_slice/64, rings_we_want=[2, 3, 4],
                              N_PIX=4096, N_waves=5, wave0=1.5, waveN=2.5, wave_ref=1.5, anamorphic=False)
    plt.show()

    """ (3) Impact of Pupil Mirror Aperture on the Central Pixel Intensity """
    rings_list = np.arange(2, 11)
    sliced = analysis.pixelation_central_slice(N_slices=5, spaxels_per_slice=64, spaxel_mas=mas_slice/64,
                                               rings_we_want=rings_list, N_PIX=4096, N_waves=3,
                                               wave0=1.5, waveN=3.0, wave_ref=1.5, anamorphic=False)

    """ (4) Pixelation images """
    rings_list = np.arange(6, 10)[::-1]
    pix_img, pix_grid = analysis.pixelation_effect(N_slices=31, spaxels_per_slice=32, spaxel_mas=mas_slice/32,
                                         rings_we_want=rings_list, N_PIX=2048, N_waves=2, wave0=1.5, waveN=2.0,
                                         wave_ref=1.5, anamorphic=True, crop=16)


    # ================================================================================================================ #
    #                                          PSF Generator                                                           #
    # ================================================================================================================ #



    # harmoni = slicers.HARMONI(N_PIX=1024, pix=32, N_waves=1, wave0=1.5, waveN=1.5, wave_ref=1.5, anamorphic=True)
    # harmoni.create_actuator_matrices(N_act_perD=6, alpha_pc=10, plot=True)
    #
    # harmoni_fake = slicers.HARMONI(N_PIX=2048, pix=32, N_waves=1, wave0=1.5, waveN=1.5, wave_ref=1.5,  anamorphic=False)
    # harmoni_fake.create_actuator_matrices(N_act_perD=6, alpha_pc=10, plot=True)
    #
    # c = np.random.uniform(-1, 1, size=harmoni.N_act)
    # phase = np.dot(harmoni.actuator_matrices[1.5], c)
    # # phase_fake = np.dot(harmoni_fake.actuator_matrices[1.5], c)
    #
    # plt.figure()
    # plt.imshow(phase)
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(harmoni.pupil_masks[1.5])
    # plt.colorbar()
    # plt.show()
    #
    # pup = harmoni.actuator_matrices[1.5][:,:,0]
    # plt.imshow(pup)
    # plt.show()
    #


    # ================================================================================================================ #
    #                                         HARMONI Slicer PSFs                                                      #
    # ================================================================================================================ #
    diversity = 0.50 * 1.5
    harmoni = slicers.HARMONI(N_PIX=256, pix=32, N_waves=1, wave0=1.5, waveN=1.5, wave_ref=1.5, anamorphic=True)
    harmoni.create_actuator_matrices(N_act_perD=10, alpha_pc=10, plot=True)
    harmoni.define_diversity(diversity)

    # for wave in harmoni.wave_range:
    #     plt.figure()
    #     plt.imshow(harmoni.diversities[wave])
    #     plt.colorbar()
    # plt.show()

    N_PSF = 5000
    c = 0.25
    coef = np.random.uniform(low=-c, high=c, size=((N_PSF, harmoni.N_act)))

    psf_images = harmoni.generate_PSF(coef, wavelengths=[1.5])

    plt.figure()
    plt.imshow(p[0, :, :, 1])
    plt.colorbar()
    plt.show()





    psf_generator = slicers.SlicerPSFGenerator(slicer_model=HARMONI, N_actuators=150, alpha_pc=10, N_PIX=N_PIX,
                                               crop_pix=128, plot=True)
    plt.show()

    N_PSF = 50
    c = 0.25
    coef = np.random.uniform(low=-c, high=c, size=((N_PSF, psf_generator.N_act)))

    gpu_start = time()
    images = psf_generator.generate_PSF_gpu(coef, wavelength=wave0, crop=True)
    gpu_end = time()
    gpu_time = gpu_end - gpu_start
    print("Time to propagate %d PSFs: %.2f seconds" % (N_PSF, gpu_time))
    print("Time to propagate 1 PSF (%d slices): %.2f seconds" % (HARMONI.N_slices, gpu_time / N_PSF))
    print("Time to propagate 1 slice: %.2f seconds" % (gpu_time / N_PSF / HARMONI.N_slices))


    PEAK = np.max(images)
    images /= PEAK

    for i in range(10):
        plt.figure()
        plt.imshow(images[i, :, :], cmap='hot')
        plt.colorbar()
    plt.show()