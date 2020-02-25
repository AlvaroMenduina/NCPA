

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time

import pycuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import skcuda
import skcuda.linalg as clinalg
import skcuda.fft as cu_fft

import utils
import slicers

utils.print_title(message='\nN C P A', font=None, random_font=False)



if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # ================================================================================================================ #
    #                                      Speed Comparison on the GPU                                                 #
    # ================================================================================================================ #
    #
    # slicer_options = {"N_slices": 38, "spaxels_per_slice": 11,
    #                   "pupil_mirror_aperture": 0.85, "anamorphic": True}
    # N_PIX = 2048
    # spaxel_mas = 0.25        # to get a decent resolution
    #
    # slicer = slicers.SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX, spaxel_scale=spaxel_mas,
    #                              N_waves=2, wave0=1.5, waveN=2.0, wave_ref=1.5)
    #
    # # print("\n---------------------------------------------------------------------")
    # # print("Running on the CPU")
    # # N_PSF = 10
    # # cpu_start = time()
    # # for i in range(N_PSF):
    # #     complex_slicer, complex_mirror, exit_slit, slits = slicer.propagate_one_wavelength(wavelength=1.5, wavefront=0, plot=False)
    # # cpu_end = time()
    # # cpu_time = cpu_end - cpu_start
    # #
    # # print("Time to propagate %d PSFs: %.2f seconds" % (N_PSF, cpu_time))
    # # print("Time to propagate 1 PSF (%d slices): %.2f seconds" % (slicer.N_slices, cpu_time / N_PSF))
    # # print("Time to propagate 1 slice: %.2f seconds" % (cpu_time / N_PSF / slicer.N_slices))
    #
    # # N_PSF = 10
    # # gpu_start = time()
    # # exit_slits = slicer.propagate_gpu_wavelength(wavelength=1.5, wavefront=0, N=N_PSF)
    # # gpu_end = time()
    # # gpu_time = gpu_end - gpu_start
    # # print("Time to propagate %d PSFs: %.2f seconds" % (N_PSF, gpu_time))
    # # print("Time to propagate 1 PSF (%d slices): %.2f seconds" % (slicer.N_slices, gpu_time / N_PSF))
    # # print("Time to propagate 1 slice: %.2f seconds" % (gpu_time / N_PSF / slicer.N_slices))
    # #
    # # free, total = cuda.mem_get_info()
    # # print("Free: %.2f percent" % (free / total * 100))
    # #
    # # for i in range(N_PSF):
    # #     plt.figure()
    # #     plt.imshow(exit_slits[i])
    # #     plt.colorbar()
    # # plt.show()
    #
    # # ================================================================================================================ #
    # #                                    HARMONI comparison                                                            #
    # # ================================================================================================================ #
    #
    # # First we have to match the Slice Size to have a proper clipping at the Slicer Plane
    # # That comes from the product spaxels_per_slice * spaxel_mas
    #
    # N_slices = 15
    # spaxels_per_slice = 28
    # spaxel_mas = 0.25  # to get a decent resolution
    #
    # # HARMONI fits approximately 6 rings (at each side) at the Pupil Mirror at 1.5 microns
    # # that would
    # # In Python we have 1/2 spaxels_per_slice rings at each side in the Pupil Mirror arrays
    # N_rings = spaxels_per_slice / 2
    # rings_we_want = 3
    # pupil_mirror_aperture = rings_we_want / N_rings
    #
    # N_PIX = 2048
    # wave0, waveN = 1.5, 3.0
    #
    # slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
    #                   "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": True}
    #
    # HARMONI = slicers.SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX,
    #                               spaxel_scale=spaxel_mas, N_waves=1, wave0=wave0, waveN=waveN, wave_ref=wave0)
    #
    # for wave in HARMONI.wave_range:
    #
    #     complex_slicer, complex_mirror, exit_slit, slits = HARMONI.propagate_one_wavelength(wavelength=wave, wavefront=0)
    #
    #     masked_slicer = (np.abs(complex_slicer)) ** 2 * HARMONI.slicer_masks[N_slices // 2]
    #     masked_slicer /= np.max(masked_slicer)
    #     minPix_Y = (N_PIX + 1 - 2 * spaxels_per_slice) // 2         # Show 2 slices
    #     maxPix_Y = (N_PIX + 1 + 2 * spaxels_per_slice) // 2
    #     minPix_X = (N_PIX + 1 - 6 * spaxels_per_slice) // 2
    #     maxPix_X = (N_PIX + 1 + 6 * spaxels_per_slice) // 2
    #     masked_slicer = masked_slicer[minPix_Y:maxPix_Y, minPix_X:maxPix_X]
    #
    #     plt.figure()
    #     plt.imshow(masked_slicer, cmap='jet')
    #     plt.colorbar(orientation='horizontal')
    #     plt.title('HARMONI Slicer: Central Slice @%.2f microns' % wave)
    #     # plt.show()
    #
    #     masked_pupil_mirror = (np.abs(complex_mirror[N_slices // 2])) ** 2 * HARMONI.pupil_mirror_mask[wave]
    #     masked_pupil_mirror /= np.max(masked_pupil_mirror)
    #     plt.figure()
    #     plt.imshow(np.log10(masked_pupil_mirror), cmap='jet')
    #     plt.colorbar()
    #     plt.clim(vmin=-4)
    #     plt.title('Pupil Mirror: Aperture %.2f PSF zeros' % rings_we_want)
    #     # plt.show()
    #
    #     masked_slit = exit_slit * HARMONI.slicer_masks[N_slices // 2]
    #     masked_slit = masked_slit[minPix_Y: maxPix_Y, minPix_X: maxPix_X]
    #     plt.figure()
    #     plt.imshow(masked_slit, cmap='jet')
    #     plt.colorbar(orientation='horizontal')
    #     plt.title('Exit Slit @%.2f microns (Pupil Mirror: %.2f PSF zeros)' % (wave, rings_we_want))
    # plt.show()

    # ================================================================================================================ #
    #                                               ANALYSIS                                                           #
    # ================================================================================================================ #

    directory = os.path.join(os.getcwd(), 'Slicers')
    analysis = slicers.SlicerAnalysis(path_save=directory)

    # """ (1) Impact of Pupil Mirror Aperture """
    # for rings in np.arange(2, 3):
    #     analysis.show_propagation(N_slices=15, spaxels_per_slice=28, spaxel_mas=0.25, rings_we_want=rings,
    #                               N_PIX=2048, N_waves=1, wave0=1.5, waveN=1.5, wave_ref=1.5, anamorphic=True)
    #
    # plt.show()
    #
    # """ (2) Impact of Wavelength """
    # analysis.show_propagation(N_slices=15, spaxels_per_slice=28, spaxel_mas=0.25, rings_we_want=3,
    #                           N_PIX=2048, N_waves=6, wave0=1.5, waveN=3.0, wave_ref=1.5, anamorphic=True)

    grid = analysis.fancy_grid(N_slices=15, spaxels_per_slice=56, spaxel_mas=0.125, rings_we_want=[5, 6, 7],
                              N_PIX=4096, N_waves=6, wave0=1.5, waveN=3.0, wave_ref=1.5, anamorphic=False)
    plt.show()

    plt.figure()
    plt.imshow(grid, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
    rings_we_want = [2, 3, 4]
    for i, _ring in enumerate(rings_we_want):
        x = (0.5 + i) / len(rings_we_want) - 0.0025
        plt.text(x=x, y=0.01, s=str(_ring), color='white')

    waves = [1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
    for j, _wave in enumerate(waves):
        y = (0.5 + j) / len(waves) - 0.005
        plt.text(x=0.01, y=y, s=str(_wave) + r' $\mu$m', color='white')
    plt.axes().set_aspect(1/3)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)

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