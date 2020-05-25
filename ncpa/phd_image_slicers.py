"""
An attempt to clean and tidy up the simulations of Image Slicers for the PhD Thesis

Author: Alvaro Menduina
Date: May 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.cm as cm
from time import time
import utils
import slicers

from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import constants
from win32com.client import CastTo

# Parameters

# Let's remind us of the nominal apertures of the different surfaces
apertures = {'Pupil Mirror': {'X_HALF': 4.5, 'Y_HALF': 1.75}, 'SMA': {'X_HALF': 6.5, 'Y_HALF': 3.5}}

# Dictionary of the wavelengths defined in the zemax file
zemax_wavelengths = {1: 1.50, 2: 1.75, 3: 2.00, 4: 2.5, 5: 3.00, 6: 1.25}
N_slices = 31


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



class POPAnalysis(object):

    def __init__(self, zosapi):

        self.zosapi = zosapi

    def run_pop(self, zemax_path, zemax_file, settings, defocus_pv=None):

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
        slice_size = 0.130              # Size of a slice at the exit focal plane (slits)

        # check that the file name is correct and the zemax file exists
        if os.path.exists(os.path.join(zemax_path, zemax_file)) is False:
            raise FileExistsError("%s does NOT exist" % zemax_file)

        print("\nOpening Zemax File: ", zemax_file)
        self.zosapi.OpenFile(os.path.join(zemax_path, zemax_file), False)
        file_name = zemax_file.split(".")[0]  # Remove the ".zmx" suffix

        # Get some info on the system
        system = self.zosapi.TheSystem  # The Optical System
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
        zernike_phase = system.LDE.GetSurfaceAt(2)
        # Setting the N_terms to 0 removes all coefficients (sanity check)
        zernike_phase.GetSurfaceCell(constants.SurfaceColumn_Par13).IntegerValue = 0
        # Start with the terms again

        if defocus_pv is not None:
            zernike_phase.GetSurfaceCell(constants.SurfaceColumn_Par13).IntegerValue = 12
            # 15 is Piston, 18 is Defocus, 19 is Astigmatism
            zernike_phase.GetSurfaceCell(constants.SurfaceColumn_Par15).DoubleValue = 1.5 / 2 * defocus_pv
            zernike_phase.GetSurfaceCell(constants.SurfaceColumn_Par18).DoubleValue = 2.5 / 2 * defocus_pv
            zernike_phase.GetSurfaceCell(constants.SurfaceColumn_Par19).DoubleValue = -3.0 / 2 * defocus_pv

        print("Piston: ", zernike_phase.GetSurfaceCell(constants.SurfaceColumn_Par15))
        print("Defocus: ", zernike_phase.GetSurfaceCell(constants.SurfaceColumn_Par18))
        print("Astigm: ", zernike_phase.GetSurfaceCell(constants.SurfaceColumn_Par19))

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

        self.zosapi.CloseFile(save=False)

        total_time = time() - start
        print("Analysis finished in %.2f seconds" % total_time)

        return data, cresults


class CompareZemaxPython(object):

    def __init__(self, zosapi, zemax_path, zemax_file, N_pix):

        self.zosapi = zosapi
        self.pop_analysis = POPAnalysis(zosapi=self.zosapi)
        self.zemax_path = zemax_path
        self.zemax_file = zemax_file

        # 6 zeroes fit within a 3.5 aperture
        self.N_pix = N_pix
        N_zeroes = 3

        N_slices = 73
        y_half = N_zeroes * 3.5 / 6
        self.slice_dim = 0.13
        self.plot_slices = 7

        self.config_slices = {'-15': 71, '-14': 5, '-13': 69,
                              '-12': 7, '-11': 67, '-10': 9, '-9': 65, '-8': 11, '-7': 63,
                              '-6': 13, '-5': 61, '-4': 15, '-3': 59, '-2': 17, '-1': 57,
                              'Central': 19,
                              '+1': 55, '+2': 21, '+3': 53, '+4': 23, '+5': 51, '+6': 25,
                              '+7': 49, '+8': 27, '+9': 47, '+10': 29, '+11': 45, '+12': 31,
                              '+13': 43, '+14': 33, '+15': 41}
        self.index_slices = {'-5': -5, '-4': -4, '-3': -3, '-2': -2, '-1': -1,
                             'Central': 0, '+1': 1, '+2': 2, '+3': 3, '+4': 4, '+5': 5}
        self.slices = {'Zemax': self.config_slices, 'Python': self.index_slices}


    def pop_slice(self, slice_label, N_zeros, wave_idx):

        config_slice = self.slices['Zemax'][slice_label]

        y_half = N_zeros * 3.5 / 6
        pop_settings = {'CONFIG': config_slice, 'X_HALFWIDTH': apertures['Pupil Mirror']['X_HALF'],
                        'Y_HALFWIDTH': y_half, 'SAMPLING': self.N_pix, 'N_SLICES': N_slices, 'WAVE_IDX': wave_idx}
        pop_slit, cresults = self.pop_analysis.run_pop(zemax_path=self.zemax_path, zemax_file=self.zemax_file,
                                                       settings=pop_settings)
        pop_slit /= np.max(pop_slit)
        minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
        pop_extent = [minX, -minX, minY, -minY]

        return pop_slit, pop_extent

    def python_slice(self, slice_label, N_zeros, wave_idx):

        index_slice = self.slices['Python'][slice_label]

        # Python - Exit Slit
        N_slices = 31
        spaxels_per_slice = 32
        spaxel_mas = 0.21875  # to get a decent resolution
        N_rings = spaxels_per_slice / 2
        pupil_mirror_aperture = N_zeros / N_rings
        wave = zemax_wavelengths[wave_idx]

        slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                          "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": True}
        HARMONI = slicers.SlicerModel(slicer_options=slicer_options, N_PIX=self.N_pix,
                                      spaxel_scale=spaxel_mas, N_waves=1, wave0=wave, waveN=wave, wave_ref=wave)

        complex_slicer, complex_mirror, exit_slit, slits = HARMONI.propagate_one_wavelength(wavelength=wave,
                                                                                            wavefront=0)
        # masked_slicer = (np.abs(complex_slicer)) ** 2 * HARMONI.slicer_masks[N_slices // 2]
        # masked_slicer /= np.max(masked_slicer)
        minPix_Y = (self.N_pix + 1 - self.plot_slices * spaxels_per_slice) // 2  # Show 2 slices
        maxPix_Y = (self.N_pix + 1 + self.plot_slices * spaxels_per_slice) // 2
        minPix_X = (self.N_pix + 1 - self.plot_slices * spaxels_per_slice) // 2
        maxPix_X = (self.N_pix + 1 + self.plot_slices * spaxels_per_slice) // 2
        # masked_slicer = masked_slicer[minPix_Y:maxPix_Y, minPix_X:maxPix_X]

        dX, dY = maxPix_X - minPix_X, maxPix_Y - minPix_Y  # How many pixels in the masked window
        masX, masY = dX * spaxel_mas, dY * spaxel_mas
        python_extent = [-masX / 2, masX / 2, -masY / 2, masY / 2]

        python_slit = slits[N_slices // 2 + index_slice]
        python_slit = python_slit[minPix_Y: maxPix_Y, minPix_X: maxPix_X]
        python_slit /= np.max(python_slit)

        return python_slit, python_extent

    def compare_exit_slit(self, slice_label, N_zeros, wave_idx, cmap='jet'):

        pop_slice, pop_extent = self.pop_slice(slice_label, N_zeros, wave_idx)
        python_slice, python_extent = self.python_slice(slice_label, N_zeros, wave_idx)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        pop_img = ax1.imshow(pop_slice, cmap='jet', extent=pop_extent)
        ax1.set_xlim([-self.plot_slices / 2 * self.slice_dim, self.plot_slices / 2 * self.slice_dim])
        ax1.set_ylim([-self.plot_slices / 2 * self.slice_dim, self.plot_slices / 2 * self.slice_dim])
        # ax1.set_xlabel(r'X [mm]')
        ax1.set_ylabel(r'Y [mm]')
        ax1.set_title(r'Zemax POP Exit Slit %s' % slice_label)

        python_img = ax2.imshow(python_slice, cmap=cmap, extent=python_extent)
        # ax2.set_xlabel(r'X [mas]')
        ax2.set_ylabel(r'Y [mas]')
        ax2.set_title(r'Python Exit Slit %s' % slice_label)
        # fig_name = "Slit_config%d_wave%dnm" % (config_slice, 1e3 * wavelengths[wave_idx])
        # fig.savefig(os.path.join(results_path, fig_name))
        # plt.show()

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        pop_img = ax3.imshow(np.log10(pop_slice), cmap='jet', extent=pop_extent)
        pop_img.set_clim(vmin=-8, vmax=0)
        ax3.set_xlim([-self.plot_slices / 2 * self.slice_dim, self.plot_slices / 2 * self.slice_dim])
        ax3.set_ylim([-self.plot_slices / 2 * self.slice_dim, self.plot_slices / 2 * self.slice_dim])
        ax3.set_xlabel(r'X [mm]')
        ax3.set_ylabel(r'Y [mm]')
        # ax3.set_title(r'Zemax POP Exit Slit')
        # plt.colorbar(pop_img, ax=ax3, orientation='horizontal')

        python_img = ax4.imshow(np.log10(python_slice), cmap=cmap, extent=python_extent)
        python_img.set_clim(vmin=-8, vmax=0)
        ax4.set_xlabel(r'X [mas]')
        ax4.set_ylabel(r'Y [mas]')
        # ax4.set_title(r'Python Exit Slit')
        # plt.colorbar(python_img, ax=ax4, orientation='horizontal')
        # fig_name = "AllSlit_config%d_wave%dnm" % (config_slice, 1e3 * wavelengths[wave_idx])
        # fig_name = "PSF_wave%dnm" % (1e3 * zemax_wavelengths[wave_idx])
        # fig.savefig(os.path.join(results_path, fig_name))
        # plt.show()

    def pop_psf(self, N_zeros, wave_idx, defocus_pv=None):

        y_half = N_zeros * 3.5 / 6

        pop_psf = []
        for config_slice in self.config_slices.values():
            pop_settings = {'CONFIG': config_slice, 'X_HALFWIDTH': apertures['Pupil Mirror']['X_HALF'],
                            'Y_HALFWIDTH': y_half, 'SAMPLING': self.N_pix, 'N_SLICES': N_slices, 'WAVE_IDX': wave_idx}
            pop_slit, cresults = self.pop_analysis.run_pop(zemax_path=self.zemax_path, zemax_file=self.zemax_file,
                                                           settings=pop_settings, defocus_pv=defocus_pv)
            pop_psf.append(pop_slit)
        pop_psf = np.array(pop_psf)
        pop_psf = np.sum(pop_psf, axis=0)

        minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
        pop_extent = [minX, -minX, minY, -minY]

        pop_peak = np.max(pop_psf)
        pop_psf /= pop_peak

        return pop_psf, pop_extent

    def fix_anamorph(self, pop_psf, pop_extent):
        """
        The PSF at the exit slit is anarmophic, elongated along Y [across slices]
        we have to resample along that direction to go back to symmetric PSFs
        :param pop_psf:
        :param pop_extent:
        :return:
        """

        N_before = pop_psf.shape[0]
        N_after = N_before // 2

        new_psf = np.zeros((N_after, N_after))
        # First of all we crop by half on the X direction so that at the end we have equal number of pixels in both axes
        minX, maxX = N_before // 2 - N_after // 2, N_before // 2 + N_after // 2
        crop_psf = pop_psf[:, minX:maxX]
        # The we loop over each slice and average two pixels in the Y direction
        for i in range(N_after):
            cut = crop_psf[2 * i:2 * (i + 1), :]
            new_psf[i, :] = np.mean(cut, axis=0)
        return new_psf

    def python_psf(self, N_zeros, wave_idx):

        # Python - Exit Slit
        N_slices = 31
        spaxels_per_slice = 32
        spaxel_mas = 0.21875  # to get a decent resolution
        N_rings = spaxels_per_slice / 2
        pupil_mirror_aperture = N_zeros / N_rings
        wave = zemax_wavelengths[wave_idx]

        slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                          "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": True}
        HARMONI = slicers.SlicerModel(slicer_options=slicer_options, N_PIX=self.N_pix,
                                      spaxel_scale=spaxel_mas, N_waves=1, wave0=wave, waveN=wave, wave_ref=wave)

        complex_slicer, complex_mirror, exit_slit, slits = HARMONI.propagate_one_wavelength(wavelength=wave,
                                                                                            wavefront=0)
        # masked_slicer = (np.abs(complex_slicer)) ** 2 * HARMONI.slicer_masks[N_slices // 2]
        # masked_slicer /= np.max(masked_slicer)
        minPix_Y = (self.N_pix + 1 - self.plot_slices * spaxels_per_slice) // 2  # Show 2 slices
        maxPix_Y = (self.N_pix + 1 + self.plot_slices * spaxels_per_slice) // 2
        minPix_X = (self.N_pix + 1 - self.plot_slices * spaxels_per_slice) // 2
        maxPix_X = (self.N_pix + 1 + self.plot_slices * spaxels_per_slice) // 2
        # masked_slicer = masked_slicer[minPix_Y:maxPix_Y, minPix_X:maxPix_X]

        dX, dY = maxPix_X - minPix_X, maxPix_Y - minPix_Y  # How many pixels in the masked window
        masX, masY = dX * spaxel_mas, dY * spaxel_mas
        python_extent = [-masX / 2, masX / 2, -masY / 2, masY / 2]

        plt.show()
        python_psf = []
        for k_slice in self.index_slices.values():
            python_slit = slits[N_slices // 2 + k_slice]
            python_slit = python_slit[minPix_Y: maxPix_Y, minPix_X: maxPix_X]
            python_psf.append(python_slit)
        python_psf = np.array(python_psf)
        python_psf = np.sum(python_psf, axis=0)

        python_peak = np.max(python_psf)
        python_psf /= python_peak

        return python_psf, python_extent

    def compare_exit_psf(self, N_zeros, wave_idx, results_path, cmap='jet'):

        pop_psf, pop_extent = self.pop_psf(N_zeros, wave_idx)
        python_psf, python_extent = self.python_psf(N_zeros, wave_idx)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        pop_img = ax1.imshow(pop_psf, cmap='jet', extent=pop_extent)
        ax1.set_xlim([-self.plot_slices / 2 * self.slice_dim, self.plot_slices / 2 * self.slice_dim])
        ax1.set_ylim([-self.plot_slices / 2 * self.slice_dim, self.plot_slices / 2 * self.slice_dim])
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
        ax3.set_xlim([-self.plot_slices / 2 * self.slice_dim, self.plot_slices / 2 * self.slice_dim])
        ax3.set_ylim([-self.plot_slices / 2 * self.slice_dim, self.plot_slices / 2 * self.slice_dim])
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
        fig_name = "PSF_wave%dnm" % (1e3 * zemax_wavelengths[wave_idx])
        # fig.savefig(os.path.join(results_path, fig_name))
        plt.show()


utils.print_title(message='\nN C P A', font=None, random_font=False)

if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # ================================================================================================================ #
    #                                        HARMONI Model                                                             #
    # ================================================================================================================ #

    # First we have to match the Slice Size to have a proper clipping at the Slicer Plane
    # That comes from the product spaxels_per_slice * spaxel_mas


    spaxels_per_slice = 32
    spaxel_mas = 0.21875/2  # to get a decent resolution

    # HARMONI fits approximately 6 rings (at each side) at the Pupil Mirror at 1.5 microns
    # that would
    # In Python we have 1/2 spaxels_per_slice rings at each side in the Pupil Mirror arrays
    N_rings = spaxels_per_slice / 2
    rings_we_want = 3
    pupil_mirror_aperture = rings_we_want / N_rings

    # ================================================================================================================ #
    #                                        Zemax POP - HARMONI                                                       #
    # ================================================================================================================ #

    # Create a Python Standalone Application
    psa = PythonStandaloneApplication()
    zemax_path = os.path.abspath("D:\Research\Python Simulations\Image Slicer Effects")
    zemax_file = "HARMONI_SLICER_EFFECTS.zmx"
    results_path = os.path.abspath("D:\Research\Python Simulations\Image Slicer Effects\ZOS API HARMONI")
    # Dictionary of the wavelengths defined in the zemax file
    # wavelengths = {1: 1.50, 2: 1.75, 3: 2.00, 4: 2.5, 5: 3.00, 6: 1.25}
    #
    #
    #
    # # (1) Show the Nominal PSF at each surface for HARMONI with POP
    pop_analysis = POPAnalysis(zosapi=psa)
    N_pix = 2048
    # N_slices = 73
    pupil_mirror, sma = 32, 36
    y_half = 3.5
    slice_idx = 19
    defocus = None
    for wave_idx in [1]:
        print(wave_idx)
        wavelength = zemax_wavelengths[wave_idx]
        pop_settings = {'CONFIG': slice_idx, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': 3.5, 'SAMPLING': N_pix,
                        'N_SLICES': N_slices, 'END_SURFACE': pupil_mirror, 'WAVE_IDX': wave_idx}

        pm_data, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings,
                                                 defocus_pv=defocus)
        minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
        extent = [minX, -minX, minY, -minY]

        fig1, (ax1, ax2) = plt.subplots(1, 2)
        zoom_ = 2
        pm_img1 = ax1.imshow(pm_data, cmap='jet', extent=extent)
        ax1.axhline(y=y_half, linestyle='--', color='white')
        ax1.axhline(y=-y_half, linestyle='--', color='white')
        ax1.set_xlim([-4/zoom_, 4/zoom_])
        ax1.set_ylim([-4/zoom_, 4/zoom_])
        ax1.set_xlabel(r'X [mm]')
        ax1.set_ylabel(r'Y [mm]')
        ax1.set_title(r'Pupil Mirror [%dx zoom] | %.2f $\mu$m' % (zoom_, wavelength))
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

        # SMA
        pop_settings = {'CONFIG': slice_idx, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': 3.5, 'SAMPLING': N_pix,
                        'N_SLICES': N_slices, 'END_SURFACE': sma, 'WAVE_IDX': wave_idx}

        sma_data, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings,
                                                  defocus_pv=defocus)
        minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
        extent = [minX, -minX, minY, -minY]

        fig2, (ax1, ax2) = plt.subplots(1, 2)
        zoom_ = 2
        sma_img1 = ax1.imshow(sma_data, cmap='jet', extent=extent)
        ax1.axhline(y=3.5, linestyle='--', color='white')
        ax1.axhline(y=-3.5, linestyle='--', color='white')
        ax1.set_xlim([-4/zoom_, 4/zoom_])
        ax1.set_ylim([-4/zoom_, 4/zoom_])
        ax1.set_xlabel(r'X [mm]')
        ax1.set_ylabel(r'Y [mm]')
        ax1.set_title(r'Slit Mirror [%dx zoom] | %.2f $\mu$m' % (zoom_, wavelength))
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

        # Exit Slit
        pop_settings = {'CONFIG': slice_idx, 'X_HALFWIDTH': 4.5, 'Y_HALFWIDTH': 3.5, 'SAMPLING': N_pix,
                        'N_SLICES': N_slices, 'WAVE_IDX': wave_idx}
        slit_data, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings,
                                                   defocus_pv=defocus)
        minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
        extent = [minX, -minX, minY, -minY]
        slice_dim = 0.13
        ss = 3

        fig3, (ax1, ax2) = plt.subplots(1, 2)
        slit_img1 = ax1.imshow(slit_data, cmap='jet', extent=extent)
        ax1.axhline(y=slice_dim/2, linestyle='--', color='white')
        ax1.axhline(y=-slice_dim/2, linestyle='--', color='white')
        ax1.set_xlim([-ss*slice_dim, ss*slice_dim])
        ax1.set_ylim([-ss*slice_dim, ss*slice_dim])
        ax1.set_xlabel(r'X [mm]')
        ax1.set_ylabel(r'Y [mm]')
        ax1.set_title(r'Exit Slit | %.2f $\mu$m' % (wavelength))

        slit_img2 = ax2.imshow(np.log10(slit_data), cmap='jet', extent=extent)
        slit_img2.set_clim(-6)
        ax2.axhline(y=slice_dim/2, linestyle='--', color='white')
        ax2.axhline(y=-slice_dim/2, linestyle='--', color='white')
        ax2.set_xlim([-ss*slice_dim, ss*slice_dim])
        ax2.set_ylim([-ss*slice_dim, ss*slice_dim])
        ax2.set_xlabel(r'X [mm]')
        ax2.set_title(r'Log scale')

    plt.show()

    # ================================================================================================================ #

    # # Compare the exit slit for Python and Zemax
    # comparison = CompareZemaxPython(zosapi=psa, N_pix=2048, zemax_path=zemax_path, zemax_file=zemax_file)
    # for label in ['-1', 'Central', '+1']:
    #     comparison.compare_exit_slit(slice_label=label, N_zeros=3, wave_idx=1)
    # comparison.compare_exit_psf(N_zeros=3, wave_idx=1, results_path=results_path)

    # POP PSF for Phase Diversity
    pop = CompareZemaxPython(zosapi=psa, N_pix=2048, zemax_path=zemax_path, zemax_file=zemax_file)
    pop_psf, pop_extent = pop.pop_psf(N_zeros=3, wave_idx=1, defocus_pv=None)
    x_win = pop_extent[1]

    final_psf = pop.fix_anamorph(pop_psf, pop_extent)
    # After we fix the anamorph the size in X is halved
    x_final = pop_extent[1]
    x_sampling = x_final / final_psf.shape[0]       # mm / pixel

    fig, (ax1, ax2) = plt.subplots(1, 2)
    img1 = ax1.imshow(np.log10(pop_psf))
    ax1.set_title(r'Raw Anamorphic')
    img2 = ax2.imshow(np.log10(final_psf), extent=[-x_win/2, x_win/2, -x_win/2, x_win/2])
    ax2.set_title(r'Downsampled Y')
    plt.show()

    # Things to do: find out about the Nyquist sampling condition

    # Loop over different N_pix
    for N_pix in [512, 1024, 2048]:
        pop = CompareZemaxPython(zosapi=psa, N_pix=2*N_pix, zemax_path=zemax_path, zemax_file=zemax_file)
        pop_psf, pop_extent = pop.pop_psf(N_zeros=6, wave_idx=1)
        final_psf = pop.fix_anamorph(pop_psf, pop_extent)

        psf_name = 'PSF_%dPix_Nominal' % N_pix
        np.save(os.path.join(results_path, psf_name), final_psf)



    def save_fits(psf_array, results_path, file_name):

        hdu = fits.PrimaryHDU([psf_array, psf_array])
        hdr = hdu.header
        hdr['PIX'] = 2048
        hdu.writeto(os.path.join(results_path, file_name + '.fits'), overwrite=True)


    save_fits(final_psf, results_path, 'PSF_%dPix_Nominal' % N_pix)

    def read_fits(path_to_file):
        with fits.open(path_to_file) as hdul:
            hdul.info()
            hdr = hdul[0].header
            print(repr(hdr))
            print(list(hdr.keys()))

    read_fits(os.path.join(results_path, 'PSF_%dPix_Nominal' % N_pix + '.fits'))

    class PhaseDiversityPSF(object):

        def __init__(self, zosapi, zemax_path, zemax_file, results_path):

            self.zosapi = zosapi
            self.zemax_path = zemax_path
            self.zemax_file = zemax_file
            self.results_path = results_path

        @staticmethod
        def save_fits(list_psf, header_info, path_to_file):

            # create a HDU and add the [Nominal PSF, Defocused PSF]
            hdu = fits.PrimaryHDU(list_psf)
            # edit the header to add WAVELENGTH info
            for (key, value) in header_info.items():
                hdu.header[key] = value

            # save
            hdu.writeto(path_to_file, overwrite=True)

            return

        @staticmethod
        def read_fits(path_to_file):
            with fits.open(path_to_file) as hdul:
                hdul.info()
                hdr = hdul[0].header
                print(repr(hdr))
                print(list(hdr.keys()))

        def generate_fits(self, N_PIX, wavelength, N_zeros, defocus_pv):

            # Find the wave index for that wavelength in the dictionary
            wave_idx = list(zemax_wavelengths.keys())[list(zemax_wavelengths.values()).index(wavelength)]

            # POP PSF for Phase Diversity
            pop = CompareZemaxPython(zosapi=self.zosapi, N_pix=2 * N_PIX, zemax_path=self.zemax_path, zemax_file=self.zemax_file)
            raw_pop_nominal, raw_pop_nominal_extent = pop.pop_psf(N_zeros=N_zeros, wave_idx=wave_idx, defocus_pv=None)
            pop_nominal_psf = pop.fix_anamorph(raw_pop_nominal, raw_pop_nominal_extent)
            x_win = raw_pop_nominal_extent[1]

            raw_pop_defocus, raw_pop_defocus_extent = pop.pop_psf(N_zeros=N_zeros, wave_idx=wave_idx, defocus_pv=defocus_pv)
            pop_defocus_psf = pop.fix_anamorph(raw_pop_defocus, raw_pop_defocus_extent)

            # Header
            header = {}
            header['WAVE'] = (wavelength, 'Nominal Wavelength [microns]')
            header['LEN_X'] = (x_win, 'Physical size X [mm]')
            header['FOCUS_PV'] = (defocus_pv, 'Defocus Peak-to-Valley [waves]')

            # save
            file_name = 'PSF_%dPix_Wave_%.2fmicrons_Defocus_%.2fPV.fits' % (N_PIX, wavelength, defocus_pv)
            self.save_fits(list_psf=[pop_nominal_psf, pop_defocus_psf], header_info=header,
                           path_to_file=os.path.join(self.results_path, file_name))

            return

    pd = PhaseDiversityPSF(zosapi=psa, zemax_path=zemax_path, zemax_file=zemax_file, results_path=results_path)
    pd.generate_fits(N_PIX=256, wavelength=1.5, N_zeros=3, defocus_pv=0.10)
    pd.read_fits(os.path.join(results_path, 'PSF_256Pix_Wave_1.50microns_Defocus_0.10PV.fits'))


    del psa
    psa = None