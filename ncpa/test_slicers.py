

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


from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import constants
from win32com.client import CastTo


# Taken from Zemax example code
class PythonStandaloneApplication(object):
    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass

    def __init__(self):
        # make sure the Python wrappers are available for the COM client and
        # interfaces
        EnsureModule('{EA433010-2BAC-43C4-857C-7AEAC4A8CCE0}', 0, 1, 0)

        # EnsureModule('{F66684D7-AAFE-4A62-9156-FF7A7853F764}', 0, 1, 0)
        # Note - the above can also be accomplished using 'makepy.py' in the
        # following directory:
        #      {PythonEnv}\Lib\site-packages\wind32com\client\
        # Also note that the generate wrappers do not get refreshed when the
        # COM library changes.
        # To refresh the wrappers, you can manually delete everything in the
        # cache directory:
        #      {PythonEnv}\Lib\site-packages\win32com\gen_py\*.*

        self.TheConnection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException(
                "Unable to intialize COM connection to ZOSAPI")

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication.LicenseException(
                "License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None

        self.TheConnection = None

    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if (self.TheApplication.LicenseStatus is
                constants.LicenseStatusType_PremiumEdition):
            return "Premium"
        elif (self.TheApplication.LicenseStatus is
              constants.LicenseStatusType_ProfessionalEdition):
            return "Professional"
        elif (self.TheApplication.LicenseStatus is
              constants.LicenseStatusType_StandardEdition):
            return "Standard"
        else:
            return "Invalid"

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
    spaxel_mas = 0.21875/2  # to get a decent resolution

    # HARMONI fits approximately 6 rings (at each side) at the Pupil Mirror at 1.5 microns
    # that would
    # In Python we have 1/2 spaxels_per_slice rings at each side in the Pupil Mirror arrays
    N_rings = spaxels_per_slice / 2
    rings_we_want = 3
    pupil_mirror_aperture = rings_we_want / N_rings

    N_PIX = 2048
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

    N = 2048
    rho = 0.15
    x = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, x)
    pupil = (xx**2 + yy**2) <= rho**2
    defocus = 2 * (xx**2 + yy**2) / (rho**2) - 1
    defocus = pupil * defocus

    from numpy.fft import fft2, fftshift
    ft_foc = fftshift(fft2(defocus, norm='ortho'))

    masked_ft_foc = HARMONI.slicer_masks[N_slices//2] * ft_foc


    # ================================================================================================================ #
    #                                       ZOS API
    # ================================================================================================================ #


    class POPAnalysis(object):

        def __init__(self, zosapi):

            self.zosapi = zosapi

        def run_pop(self, zemax_path, zemax_file, settings):

            start = time()

            # Hard-coded values
            surf_pupil_mirror = 32
            surf_sma = 36
            pop_sampling = {32: 1, 64: 2, 128: 3, 256: 4, 512: 5, 1024: 6,
                            2048: 7, 4096: 8}
            beam_types = {'top_hat': 3}
            waist_x = 0.189
            waist_y = 0.095
            N_pix = settings['SAMPLING']
            N_slices = settings['N_SLICES']
            slice_size = 0.130

            # check that the file name is correct and the zemax file exists
            if os.path.exists(os.path.join(zemax_path, zemax_file)) is False:
                raise FileExistsError("%s does NOT exist" % zemax_file)

            print("\nOpening Zemax File: ", zemax_file)
            psa.OpenFile(os.path.join(zemax_path, zemax_file), False)
            file_name = zemax_file.split(".")[0]  # Remove the ".zmx" suffix

            # Get some info on the system
            system = psa.TheSystem  # The Optical System
            MCE = system.MCE  # Multi Configuration Editor
            LDE = system.LDE  # Lens Data Editor
            N_surfaces = LDE.NumberOfSurfaces
            N_configs = MCE.NumberOfConfigurations

            # Read the POP Settings
            config = settings['CONFIG']
            wave_idx = settings['WAVE_IDX'] if 'WAVE_IDX' in settings else 1
            field_idx = settings['FIELD_IDX'] if 'FIELD_IDX' in settings else 1
            x_halfwidth = settings['X_HALFWIDTH']
            y_halfwidth = settings['Y_HALFWIDTH']
            end_surface = settings['END_SURFACE'] if 'END_SURFACE' in settings else LDE.NumberOfSurfaces - 1
            sampling = pop_sampling[settings['SAMPLING']]
            beam_type = beam_types[settings['BEAM_TYPE']] if 'BEAM_TYPE' in settings else 3

            print("Zernike: ", system.LDE.GetSurfaceAt(2).TypeName)

            # (0) Info
            print("\nRunning POP analysis to Surface #%d: %s" % (end_surface, LDE.GetSurfaceAt(end_surface).Comment))

            # (1) Set the Configuration to the Slice of interest
            system.MCE.SetCurrentConfiguration(config)
            print("(1) Setting Configuration to #%d" % config)

            # (2) Set the Pupil Mirror Aperture
            pupil_mirror = system.LDE.GetSurfaceAt(surf_pupil_mirror)

            print("(2) Setting Pupil Mirror Aperture")
            # Read Current Aperture Settings
            apt_type = pupil_mirror.ApertureData.CurrentType
            if apt_type == 4:  # 4 is Rectangular aperture
                current_apt_sett = pupil_mirror.ApertureData.CurrentTypeSettings
                print("Current Settings:")
                print("X_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.XHalfWidth)
                print("Y_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.YHalfWidth)
                # Change Settings
                aperture_settings = pupil_mirror.ApertureData.CreateApertureTypeSettings(
                    constants.SurfaceApertureTypes_RectangularAperture)
                aperture_settings._S_RectangularAperture.XHalfWidth = x_halfwidth
                aperture_settings._S_RectangularAperture.YHalfWidth = y_halfwidth
                pupil_mirror.ApertureData.ChangeApertureTypeSettings(aperture_settings)

                current_apt_sett = pupil_mirror.ApertureData.CurrentTypeSettings
                print("New Settings:")
                print("X_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.XHalfWidth)
                print("Y_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.YHalfWidth)


            # (2.5) SMA Aperture
            sma = system.LDE.GetSurfaceAt(surf_sma)
            if sma.Comment != 'SMA':
                raise ValueError("Surface #%d is not the SMA" % surf_sma)
            # Read Current Aperture Settings
            apt_type = sma.ApertureData.CurrentType
            print("(2.5) Setting SMA Aperture")
            if apt_type == 4:  # 4 is Rectangular aperture
                current_apt_sett = sma.ApertureData.CurrentTypeSettings
                print("Current Settings:")
                print("X_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.XHalfWidth)
                print("Y_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.YHalfWidth)

            if end_surface == LDE.NumberOfSurfaces - 1:
                # (3) Set the Sampling at the last surface
                final_surface = LDE.GetSurfaceAt(end_surface)
                pop_data = final_surface.PhysicalOpticsData
                pop_data.ResampleAfterRefraction = True
                pop_data.AutoResample = False
                pop_data.XSampling = sampling - 1          # somehow the numbering for the sampling is different
                pop_data.YSampling = sampling - 1          # for the Resampling... stupid Zemax
                pop_data.XWidth = N_slices * slice_size
                pop_data.YWidth = N_slices * slice_size


            # (4) Set the POP Analysis Parameters
            pop = system.Analyses.New_Analysis_SettingsFirst(constants.AnalysisIDM_PhysicalOpticsPropagation)
            pop.Terminate()
            pop_setting = pop.GetSettings()
            pop_settings = CastTo(pop_setting, 'IAS_')

            if pop.HasAnalysisSpecificSettings == False:

                zemax_dir = os.path.abspath('H:\Zemax')
                cfg = zemax_dir + '\\Configs\\POP.CFG'

                pop_settings.ModifySettings(cfg, 'POP_END', end_surface)
                pop_settings.ModifySettings(cfg, 'POP_WAVE', wave_idx)
                pop_settings.ModifySettings(cfg, 'POP_FIELD:', field_idx)
                pop_settings.ModifySettings(cfg, 'POP_BEAMTYPE', beam_type)  # 3 for Top Hap beam
                pop_settings.ModifySettings(cfg, 'POP_POWER', 1)
                pop_settings.ModifySettings(cfg, 'POP_SAMPX', sampling)
                pop_settings.ModifySettings(cfg, 'POP_SAMPY', sampling)
                pop_settings.ModifySettings(cfg, 'POP_PARAM1', waist_x)  # Waist X
                pop_settings.ModifySettings(cfg, 'POP_PARAM2', waist_y)  # Waist Y
                pop_settings.ModifySettings(cfg, 'POP_AUTO', 1)  # needs to go after POP_PARAM
                pop_settings.LoadFrom(cfg)

            # (5) Run POP Analysis
            pop.ApplyAndWaitForCompletion()
            pop_results = pop.GetResults()
            # pop_results.GetTextFile(os.path.join(zemax_path, '_res.txt'))

            cresults = CastTo(pop_results, 'IAR_')
            data = np.array(cresults.GetDataGrid(0).Values)

            psa.CloseFile(save=False)

            total_time = time() - start
            print("Analysis finished in %.2f seconds" % total_time)

            return data, cresults

    # surface.GetSurfaceCell(constants.SurfaceColumn_Par14)
    # zern = system.LDE.GetSurfaceAt(2)
    #
    # surface_type = surface.TypeName
    # type_data = surface.TypeData



    # Create a Python Standalone Application
    psa = PythonStandaloneApplication()
    zemax_path = os.path.abspath("D:\Research\Python Simulations\Image Slicer Effects")
    zemax_file = "HARMONI_SLICER_EFFECTS_ZOSAPI.zmx"
    # Dictionary of the wavelengths defined in the zemax file
    wavelengths = {1: 1.50, 2: 1.75, 3: 2.00, 4: 2.5, 5: 3.00, 6:1.25}

    # (1) Show the Nominal PSF at each surface for HARMONI with POP
    pop_analysis = POPAnalysis(zosapi=psa)

    results_path = os.path.abspath("D:\Research\Python Simulations\Image Slicer Effects\ZOS API HARMONI")
    N_pix = 2048
    N_slices = 73
    pupil_mirror = 32
    y_half = 3.5 / 2
    slice_idx = 21
    for wave_idx in np.arange(1, 6)[:1]:
        print(wave_idx)
        pop_settings = {'CONFIG': slice_idx, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': y_half, 'SAMPLING': N_pix,
                        'N_SLICES': N_slices, 'END_SURFACE': pupil_mirror, 'WAVE_IDX': wave_idx}

        pm_data, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)
        minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
        extent = [minX, -minX, minY, -minY]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        zoom_ = 2
        pm_img1 = ax1.imshow(pm_data, cmap='jet', extent=extent)
        ax1.axhline(y=y_half, linestyle='--', color='white')
        ax1.axhline(y=-y_half, linestyle='--', color='white')
        ax1.set_xlim([-4/zoom_, 4/zoom_])
        ax1.set_ylim([-4/zoom_, 4/zoom_])
        ax1.set_xlabel(r'X [mm]')
        ax1.set_ylabel(r'Y [mm]')
        ax1.set_title(r'Pupil Mirror [%dx zoom]' % zoom_)
        ax1.set_xticks(np.arange(-zoom_, zoom_ + 1))
        ax1.set_yticks(np.arange(-zoom_, zoom_ + 1))

        pm_img2 = ax2.imshow(np.log10(pm_data), cmap='jet', extent=extent)
        pm_img2.set_clim(-6)
        ax2.axhline(y=y_half, linestyle='--', color='black')
        ax2.axhline(y=-y_half, linestyle='--', color='black')
        ax2.set_xlim([-4, 4])
        ax2.set_ylim([-4, 4])
        ax2.set_xlabel(r'X [mm]')
        ax2.set_title(r'Log scale')
        ax2.set_xticks(np.arange(-4, 5, 2))
        ax2.set_yticks(np.arange(-4, 5, 2))
        fig_name = "060wavesPV_PupilMirror_1500nm_slice%d" % slice_idx
        # fig_name = "PupilMirror_NominalHARMONI_slice%d_%dnm" % (slice_idx, 1e3 * wavelengths[wave_idx])
        fig.savefig(os.path.join(results_path, fig_name))


        # SMA
        sma = 36
        pop_settings = {'CONFIG': slice_idx, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': y_half, 'SAMPLING': N_pix,
                        'N_SLICES': N_slices, 'END_SURFACE': sma, 'WAVE_IDX': wave_idx}

        sma_data, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)
        minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
        extent = [minX, -minX, minY, -minY]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        zoom_ = 2
        sma_img1 = ax1.imshow(sma_data, cmap='jet', extent=extent)
        ax1.axhline(y=3.5, linestyle='--', color='white')
        ax1.axhline(y=-3.5, linestyle='--', color='white')
        ax1.set_xlim([-4/zoom_, 4/zoom_])
        ax1.set_ylim([-4/zoom_, 4/zoom_])
        ax1.set_xlabel(r'X [mm]')
        ax1.set_ylabel(r'Y [mm]')
        ax1.set_title(r'Slit Mirror [%dx zoom]' % zoom_)
        ax1.set_xticks(np.arange(-zoom_, zoom_ + 1))
        ax1.set_yticks(np.arange(-zoom_, zoom_ + 1))

        sma_img2 = ax2.imshow(np.log10(sma_data), cmap='jet', extent=extent)
        sma_img2.set_clim(-6)
        ax2.axhline(y=3.5, linestyle='--', color='black')
        ax2.axhline(y=-3.5, linestyle='--', color='black')
        ax2.set_xlim([-4, 4])
        ax2.set_ylim([-4, 4])
        ax2.set_xlabel(r'X [mm]')
        ax2.set_title(r'Log scale')
        ax2.set_xticks(np.arange(-4, 5, 2))
        ax2.set_yticks(np.arange(-4, 5, 2))
        # ax2.set_ylabel(r'Y [mm]')
        # fig_name = "SlitMirror_NominalHARMONI_slice%d_%dnm" % (slice_idx, 1e3 * wavelengths[wave_idx])
        fig_name = "060wavesPV_SlitMirror_1500nm_slice%d" % slice_idx
        fig.savefig(os.path.join(results_path, fig_name))

        # Exit Slit
        pop_settings = {'CONFIG': slice_idx, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': y_half, 'SAMPLING': N_pix,
                        'N_SLICES': N_slices, 'WAVE_IDX': wave_idx}
        slit_data, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)
        minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
        extent = [minX, -minX, minY, -minY]
        slice_dim = 0.13
        ss = 3

        fig, (ax1, ax2) = plt.subplots(1, 2)
        slit_img1 = ax1.imshow(slit_data, cmap='jet', extent=extent)
        ax1.axhline(y=slice_dim/2, linestyle='--', color='white')
        ax1.axhline(y=-slice_dim/2, linestyle='--', color='white')
        ax1.set_xlim([-ss*slice_dim, ss*slice_dim])
        ax1.set_ylim([-ss*slice_dim, ss*slice_dim])
        ax1.set_xlabel(r'X [mm]')
        ax1.set_ylabel(r'Y [mm]')
        ax1.set_title(r'Exit Slit')

        slit_img2 = ax2.imshow(np.log10(slit_data), cmap='jet', extent=extent)
        slit_img2.set_clim(-6)
        ax2.axhline(y=slice_dim/2, linestyle='--', color='white')
        ax2.axhline(y=-slice_dim/2, linestyle='--', color='white')
        ax2.set_xlim([-ss*slice_dim, ss*slice_dim])
        ax2.set_ylim([-ss*slice_dim, ss*slice_dim])
        ax2.set_xlabel(r'X [mm]')
        ax2.set_title(r'Log scale')
        # fig_name = "ExitSlit_NominalHARMONI_slice%d_%dnm" % (slice_idx, 1e3 * wavelengths[wave_idx])
        fig_name = "060wavesPV_ExitSlit_1500nm_slice%d" % slice_idx
        fig.savefig(os.path.join(results_path, fig_name))

    plt.show()
    # --------------------------------------------


    """ Complete PSF """
    slices = [15, 59, 17, 57, 19, 55, 21, 53, 23]
    wave_idx = 1
    psf = []
    for slice_idx in slices:

        pop_settings = {'CONFIG': slice_idx, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': 3.5, 'SAMPLING': N_pix,
                        'N_SLICES': N_slices, 'WAVE_IDX': wave_idx}
        slit_data, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)
        psf.append(slit_data)

    psf_array = np.array(psf)
    psf_array = np.sum(psf_array, axis=0)

    minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
    extent = [minX, -minX, minY, -minY]
    slice_dim = 0.13
    ss = len(slices) / 2

    fig, (ax1, ax2) = plt.subplots(1, 2)
    slit_img1 = ax1.imshow(psf_array, cmap='jet', extent=extent)
    # ax1.axhline(y=slice_dim/2, linestyle='--', color='white')
    # ax1.axhline(y=-slice_dim/2, linestyle='--', color='white')
    ax1.set_xlim([-ss*slice_dim, ss*slice_dim])
    ax1.set_ylim([-ss*slice_dim, ss*slice_dim])
    ax1.set_xlabel(r'X [mm]')
    ax1.set_ylabel(r'Y [mm]')
    ax1.set_title(r'Exit Slit')

    slit_img2 = ax2.imshow(np.log10(psf_array), cmap='jet', extent=extent)
    slit_img2.set_clim(-6)
    ax2.set_xlim([-ss*slice_dim, ss*slice_dim])
    ax2.set_ylim([-ss*slice_dim, ss*slice_dim])
    ax2.set_xlabel(r'X [mm]')
    ax2.set_title(r'Log scale')
    fig_name = "PSF_NominalHARMONI_%dnm" % (1e3 * wavelengths[wave_idx])
    fig.savefig(os.path.join(results_path, fig_name))
    plt.show()

    ### ====== Without the Slicer ===== ###

    pop_settings = {'CONFIG': 19, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': 10, 'SAMPLING': N_pix,
                    'N_SLICES': N_slices, 'WAVE_IDX': wave_idx}
    psf_without, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)


    peak_slice = np.max(psf_without)
    psf_without /= peak_slice
    psf_array /= peak_slice

    # psf_array = psf_foc.copy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    with_slicer = ax1.imshow(psf_array, cmap='jet', extent=extent)
    # ax1.axhline(y=slice_dim/2, linestyle='--', color='white')
    # ax1.axhline(y=-slice_dim/2, linestyle='--', color='white')
    ax1.set_xlim([-ss*slice_dim, ss*slice_dim])
    ax1.set_ylim([-ss*slice_dim, ss*slice_dim])
    ax1.set_xlabel(r'X [mm]')
    ax1.set_ylabel(r'Y [mm]')
    ax1.set_title(r'With Slicer')
    plt.colorbar(with_slicer, ax=ax1, orientation='horizontal')

    without_slicer = ax2.imshow(psf_without, cmap='jet', extent=extent)
    # slit_img2.set_clim(-6)
    ax2.set_xlim([-ss*slice_dim, ss*slice_dim])
    ax2.set_ylim([-ss*slice_dim, ss*slice_dim])
    ax2.set_xlabel(r'X [mm]')
    ax2.set_title(r'Without Slicer')
    ax2.yaxis.set_visible(False)
    plt.colorbar(without_slicer, ax=ax2, orientation='horizontal')

    residual = psf_array - psf_without
    min_res = min(np.min(residual), -np.max(residual))
    res_img = ax3.imshow(residual, cmap='seismic', extent=extent)
    res_img.set_clim(vmin=min_res, vmax=-min_res)
    ax3.set_xlim([-ss*slice_dim, ss*slice_dim])
    ax3.set_ylim([-ss*slice_dim, ss*slice_dim])
    ax3.set_xlabel(r'X [mm]')
    ax3.set_title(r'Difference')
    ax3.yaxis.set_visible(False)
    plt.colorbar(res_img, ax=ax3, orientation='horizontal')
    plt.show()


    fig_name = "PSF_NominalHARMONI_%dnm" % (1e3 * wavelengths[wave_idx])
    fig.savefig(os.path.join(results_path, fig_name))
    plt.show()

    ### ====== With and Without defocus ===== ###

    """ Complete PSF """
    slices = [15, 59, 17, 57, 19, 55, 21, 53, 23]
    wave_idx = 1
    psf = []
    for slice_idx in slices:

        pop_settings = {'CONFIG': slice_idx, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': 1.75, 'SAMPLING': N_pix,
                        'N_SLICES': N_slices, 'WAVE_IDX': wave_idx}
        slit_data, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)
        psf.append(slit_data)

    psf_array = np.array(psf)
    psf_nofoc = np.sum(psf_array, axis=0)
    peak_nofoc = np.max(psf_nofoc)

    # !!!
    # Remember to add the focus

    psf = []
    for slice_idx in slices:

        pop_settings = {'CONFIG': slice_idx, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': 1.75, 'SAMPLING': N_pix,
                        'N_SLICES': N_slices, 'WAVE_IDX': wave_idx}
        slit_data, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)
        psf.append(slit_data)

    psf_array = np.array(psf)
    psf_foc = np.sum(psf_array, axis=0)



    psf_nofoc /= peak_nofoc
    psf_foc /= peak_nofoc


    minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
    extent = [minX, -minX, minY, -minY]
    slice_dim = 0.13
    ss = len(slices) / 2

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    nofoc_img = ax1.imshow(psf_nofoc, cmap='jet', extent=extent)
    # ax1.axhline(y=slice_dim/2, linestyle='--', color='white')
    # ax1.axhline(y=-slice_dim/2, linestyle='--', color='white')
    ax1.set_xlim([-ss*slice_dim, ss*slice_dim])
    ax1.set_ylim([-ss*slice_dim, ss*slice_dim])
    ax1.set_xlabel(r'X [mm]')
    ax1.set_ylabel(r'Y [mm]')
    ax1.set_title(r'Nominal')
    plt.colorbar(nofoc_img, ax=ax1, orientation='horizontal')

    foc_img = ax2.imshow(psf_foc, cmap='jet', extent=extent)
    foc_img.set_clim(vmax=1.0)
    ax2.set_xlim([-ss*slice_dim, ss*slice_dim])
    ax2.set_ylim([-ss*slice_dim, ss*slice_dim])
    ax2.set_xlabel(r'X [mm]')
    ax2.set_title(r'Defocus')
    ax2.yaxis.set_visible(False)
    plt.colorbar(foc_img, ax=ax2, orientation='horizontal')

    residual = psf_foc - psf_nofoc
    min_res = min(np.min(residual), -np.max(residual))
    res_img = ax3.imshow(residual, cmap='seismic', extent=extent)
    res_img.set_clim(vmin=min_res, vmax=-min_res)
    ax3.set_xlim([-ss*slice_dim, ss*slice_dim])
    ax3.set_ylim([-ss*slice_dim, ss*slice_dim])
    ax3.set_xlabel(r'X [mm]')
    ax3.set_title(r'Difference')
    ax3.yaxis.set_visible(False)
    plt.colorbar(res_img, ax=ax3, orientation='horizontal')
    plt.show()



    # ================================================================
    """ Comparison with HARMONI """

    # 6 zeroes fit within a 3.5 aperture
    N_pix = 2048
    N_zeroes = 3
    N_slices = 73
    y_half = N_zeroes * 3.5 / 6
    wave_idx = 1
    slice_dim = 0.13
    plot_slices = 7

    # config_slice = 21
    # k_slice = 2
    slices = [15, 59, 17, 57, 19, 55, 21, 53, 23]
    k_slices = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    # Zemax POP - Exit Slit
    pop_psf = []
    for config_slice in slices:
        pop_settings = {'CONFIG': config_slice, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': y_half, 'SAMPLING': N_pix,
                        'N_SLICES': N_slices, 'WAVE_IDX': wave_idx}
        pop_slit, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)
        pop_psf.append(pop_slit)
    pop_psf = np.array(pop_psf)
    pop_psf = np.sum(pop_psf, axis=0)

    # pop_settings = {'CONFIG': config_slice, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': y_half, 'SAMPLING': N_pix,
    #                 'N_SLICES': N_slices, 'WAVE_IDX': wave_idx}
    # pop_slit, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)
    minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
    pop_extent = [minX, -minX, minY, -minY]

    # Python - Exit Slit
    N_slices = 31
    spaxels_per_slice = 32
    spaxel_mas = 0.21875  # to get a decent resolution
    N_rings = spaxels_per_slice / 2
    pupil_mirror_aperture = N_zeroes / N_rings
    wave0 = 1.5


    slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                      "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": True}
    HARMONI = slicers.SlicerModel(slicer_options=slicer_options, N_PIX=N_pix,
                                  spaxel_scale=spaxel_mas, N_waves=1, wave0=wave0, waveN=wave0, wave_ref=wave0)

    complex_slicer, complex_mirror, exit_slit, slits = HARMONI.propagate_one_wavelength(wavelength=wave0, wavefront=0)
    # masked_slicer = (np.abs(complex_slicer)) ** 2 * HARMONI.slicer_masks[N_slices // 2]
    # masked_slicer /= np.max(masked_slicer)
    minPix_Y = (N_pix + 1 - plot_slices * spaxels_per_slice) // 2         # Show 2 slices
    maxPix_Y = (N_pix + 1 + plot_slices * spaxels_per_slice) // 2
    minPix_X = (N_pix + 1 - plot_slices * spaxels_per_slice) // 2
    maxPix_X = (N_pix + 1 + plot_slices * spaxels_per_slice) // 2
    # masked_slicer = masked_slicer[minPix_Y:maxPix_Y, minPix_X:maxPix_X]

    dX, dY = maxPix_X - minPix_X, maxPix_Y - minPix_Y  # How many pixels in the masked window
    masX, masY = dX * spaxel_mas, dY * spaxel_mas
    python_extent = [-masX / 2, masX / 2, -masY / 2, masY / 2]

    plt.show()
    python_psf = []
    for k_slice in k_slices:
        python_slit = slits[N_slices // 2 + k_slice]
        python_slit = python_slit[minPix_Y: maxPix_Y, minPix_X: maxPix_X]
        python_psf.append(python_slit)
    python_psf = np.array(python_psf)
    python_psf = np.sum(python_psf, axis=0)

    # python_slit = slits[N_slices // 2 + k_slice]
    # python_slit = python_slit[minPix_Y: maxPix_Y, minPix_X: maxPix_X]

    pop_peak = np.max(pop_psf)
    python_peak = np.max(python_psf)

    pop_psf /= pop_peak
    python_psf /= python_peak

    results_path = os.path.abspath("D:\Research\Python Simulations\Image Slicer Effects\Python vs Zemax")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    pop_img = ax1.imshow(pop_psf, cmap='jet', extent=pop_extent)
    ax1.set_xlim([-plot_slices/2*slice_dim, plot_slices/2*slice_dim])
    ax1.set_ylim([-plot_slices/2*slice_dim, plot_slices/2*slice_dim])
    # ax1.set_xlabel(r'X [mm]')
    ax1.set_ylabel(r'Y [mm]')
    ax1.set_title(r'Zemax POP PSF')

    python_img = ax2.imshow(python_psf, cmap=cmap, extent=python_extent)
    # ax2.set_xlabel(r'X [mas]')
    ax2.set_ylabel(r'Y [mas]')
    ax2.set_title(r'Python PSF')
    # fig_name = "Slit_config%d_wave%dnm" % (config_slice, 1e3 * wavelengths[wave_idx])
    # fig.savefig(os.path.join(results_path, fig_name))
    # plt.show()

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    pop_img = ax3.imshow(np.log10(pop_psf), cmap='jet', extent=pop_extent)
    pop_img.set_clim(vmin=-8, vmax=0)
    ax3.set_xlim([-plot_slices/2*slice_dim, plot_slices/2*slice_dim])
    ax3.set_ylim([-plot_slices/2*slice_dim, plot_slices/2*slice_dim])
    ax3.set_xlabel(r'X [mm]')
    ax3.set_ylabel(r'Y [mm]')
    # ax3.set_title(r'Zemax POP Exit Slit')
    # plt.colorbar(pop_img, ax=ax3, orientation='horizontal')

    python_img = ax4.imshow(np.log10(python_psf), cmap=cmap, extent=python_extent)
    python_img.set_clim(vmin=-8, vmax=0)
    ax4.set_xlabel(r'X [mas]')
    ax4.set_ylabel(r'Y [mas]')
    # ax4.set_title(r'Python Exit Slit')
    # plt.colorbar(python_img, ax=ax4, orientation='horizontal')
    # fig_name = "AllSlit_config%d_wave%dnm" % (config_slice, 1e3 * wavelengths[wave_idx])
    fig_name = "PSF_wave%dnm" % (1e3 * wavelengths[wave_idx])
    fig.savefig(os.path.join(results_path, fig_name))
    plt.show()


    """ Zemax Vs Python - PUPIL MIRROR """

    N_pix = 2048
    N_slices = 73
    pupil_mirror = 32
    y_half = 3.5
    slice_idx = 19
    k_slice = 0

    pop_settings = {'CONFIG': slice_idx, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': y_half, 'SAMPLING': N_pix,
                    'N_SLICES': N_slices, 'END_SURFACE': pupil_mirror, 'WAVE_IDX': 1}

    pop_pupilmirror, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)
    minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
    pop_extent = [minX, -minX, minY, -minY]
    pop_pupilmirror /= np.max(pop_pupilmirror)

    N_slices = 31
    mm_per_ring = 3.5 / 6
    spaxels_per_slice = 32
    spaxel_mas = 0.21875  # to get a decent resolution
    N_rings = spaxels_per_slice / 2
    N_zeroes = 6
    pupil_mirror_aperture = N_zeroes / N_rings
    wave0 = 1.5

    slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                      "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": True}
    HARMONI = slicers.SlicerModel(slicer_options=slicer_options, N_PIX=N_pix,
                                  spaxel_scale=spaxel_mas, N_waves=1, wave0=wave0, waveN=wave0, wave_ref=wave0)

    complex_slicer, complex_mirror, exit_slit, slits = HARMONI.propagate_one_wavelength(wavelength=wave0, wavefront=0)
    pupil_extent = [-N_rings * mm_per_ring, N_rings * mm_per_ring, -N_rings * mm_per_ring, N_rings * mm_per_ring]
    python_pupil_mirror = (np.abs(complex_mirror[N_slices // 2 + k_slice])) ** 2 * HARMONI.pupil_mirror_mask[wave0]
    python_pupil_mirror /= np.max(python_pupil_mirror)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    pop_img = ax1.imshow(np.log10(pop_pupilmirror), cmap='jet', extent=pop_extent)
    pop_img.set_clim(vmin=-7)
    ax1.set_xlim([-4, 4])
    ax1.set_ylim([-4, 4])
    ax1.set_ylabel(r'Y [mm]')
    ax1.set_xlabel(r'X [mm]')
    ax1.set_title(r'Zemax POP Pupil Mirror')

    python_img = ax2.imshow(np.log10(python_pupil_mirror), cmap=cmap, extent=pupil_extent)
    python_img.set_clim(vmin=-7)
    ax2.set_xlim([-4, 4])
    ax2.set_ylim([-4, 4])
    # ax2.set_xlabel(r'X [mas]')
    # ax2.set_ylabel(r'Y [mas]')
    ax2.set_title(r'Python Pupil Mirror')
    # fig_name = "Slit_config%d_wave%dnm" % (config_slice, 1e3 * wavelengths[wave_idx])
    # fig.savefig(os.path.join(results_path, fig_name))
    plt.show()



    plt.figure()
    plt.imshow((masked_pupil_mirror), cmap=cmap, extent=pupil_extent)
    plt.colorbar()
    # plt.clim(vmin=-4)
    plt.ylim([-(N_zeroes + 1), N_zeroes + 1])
    plt.xlim([-(N_zeroes + 1), N_zeroes + 1])
    plt.title('Pupil Mirror: ')
    plt.show()

    # --------------------------------------------

    N_pix = 2048

    # Pupil Mirror
    for y_half in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0][:1]:
        pop_settings = {'CONFIG': 19, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': y_half, 'SAMPLING': N_pix,
                        'N_SLICES': 5}
        pop = POPAnalysis(zosapi=psa)
        data, cresults = pop.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)

        minX = cresults.GetDataGrid(0).MinX
        minY = cresults.GetDataGrid(0).MinY
        extent = [minX, -minX, minY, -minY]

        plt.figure()
        plt.imshow(data, cmap='jet', extent=extent)
        # plt.imshow(np.log10(data), cmap='jet', extent=extent)
        # plt.axhline(y=y_half, linestyle='--', color='black')
        # plt.axhline(y=-y_half, linestyle='--', color='black')
        plt.colorbar()
        plt.title('Aperture Y Half Width %.2f mm' % y_half)
        # plt.xlim([-4, 4])
        plt.xlabel(r'X [mm]')
        plt.ylabel(r'Y [mm]')
    plt.show()

    for y_half in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        pop_settings = {'CONFIG': 19, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': y_half, 'SAMPLING': N_pix,
                        'N_SLICES': 5}
        pop = POPAnalysis(zosapi=psa)
        data, cresults = pop.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)

        minX = cresults.GetDataGrid(0).MinX
        minY = cresults.GetDataGrid(0).MinY
        extent = [minX, -minX, minY, -minY]

        plt.figure()
        plt.imshow(data, cmap='jet', extent=extent)
        # plt.imshow(np.log10(data), cmap='jet', extent=extent)
        # plt.axhline(y=y_half, linestyle='--', color='black')
        # plt.axhline(y=-y_half, linestyle='--', color='black')
        plt.colorbar()
        plt.title('Aperture Y Half Width %.2f mm' % y_half)
        # plt.xlim([-4, 4])
        plt.xlabel(r'X [mm]')
        plt.ylabel(r'Y [mm]')
    plt.show()

    del psa
    psa = None

    # surf = system.







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

    ### defocus
    from numpy.fft import fft, fftshift
    Xmax = 20
    x = np.linspace(-Xmax, Xmax, 1024)
    f = 4
    sinc = np.sin(np.pi * f * x) / (np.pi * f * x)

    h = np.abs(x) < 1


    foc = 2 * x**2 - 1
    # plt.plot(x, h * foc)
    ft_foc = fftshift(fft(h*foc, norm='ortho'))
    plt.figure()
    plt.plot(np.abs(ft_foc))



    p = convolve(sinc, ft_foc, mode='same')
    p /= np.max(p)

    plt.figure()
    plt.plot(np.abs(p))
    plt.show()


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