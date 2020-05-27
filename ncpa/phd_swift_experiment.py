"""
POP for the SWIFT Image Slicer

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





class POPAnalysisSWIFT(object):
    """
    POP Analysis for the SWIFT Image Slicer using the Python ZOS API
    """

    def __init__(self, zosapi):

        self.zosapi = zosapi

    def run_pop(self, zemax_path, zemax_file, settings):

        start = time()

        # check that the file name is correct and the zemax file exists
        if os.path.exists(os.path.join(zemax_path, zemax_file)) is False:
            raise FileExistsError("%s does NOT exist" % zemax_file)

        print("\nOpening Zemax File: ", zemax_file)
        self.zosapi.OpenFile(os.path.join(zemax_path, zemax_file), False)
        file_name = zemax_file.split(".")[0]  # Remove the ".zmx" suffix

        pop_sampling = {32: 1, 64: 2, 128: 3, 256: 4, 512: 5, 1024: 6,
                        2048: 7, 4096: 8}
        beam_types = {'top_hat': 3}


        # Hard-coded values
        surf_pupil_mirror = 16
        waist_x = 0.125
        waist_y = 0.125
        slice_size = 0.47  # Size of a slice at the exit focal plane (slits)

        # Get some info on the system
        system = self.zosapi.TheSystem  # The Optical System
        MCE = system.MCE  # Multi Configuration Editor
        LDE = system.LDE  # Lens Data Editor
        N_surfaces = LDE.NumberOfSurfaces

        # Read the POP Settings
        config = settings['CONFIG']
        # N_slices = settings['N_SLICES']
        wave_idx = settings['WAVE_IDX'] if 'WAVE_IDX' in settings else 1
        field_idx = settings['FIELD_IDX'] if 'FIELD_IDX' in settings else 1
        XWidth = settings['X_WIDTH']
        YWidth = settings['Y_WIDTH']
        end_surface = settings['END_SURFACE'] if 'END_SURFACE' in settings else LDE.NumberOfSurfaces - 1
        N_PIX = pop_sampling[settings['N_PIX']]
        sampling = pop_sampling[settings['SAMPLING']]
        beam_type = beam_types[settings['BEAM_TYPE']] if 'BEAM_TYPE' in settings else 3

        # (0) Info
        print("\nRunning POP analysis to Surface #%d: %s" % (end_surface, LDE.GetSurfaceAt(end_surface).Comment))

        # (1) Set the Configuration to the Slice of interest
        system.MCE.SetCurrentConfiguration(config)
        print("(1) Setting Configuration to #%d" % config)


        if 'PM_X_HALFWIDTH' in settings:
            print("(2) Setting Pupil Mirror Aperture")
            x_halfwidth = settings['PM_X_HALFWIDTH']
            pupil_mirror = system.LDE.GetSurfaceAt(surf_pupil_mirror)
            # Read Current Aperture Settings
            apt_type = pupil_mirror.ApertureData.CurrentType
            if apt_type == 4:  # 4 is Rectangular aperture
                current_apt_sett = pupil_mirror.ApertureData.CurrentTypeSettings
                print("Current Settings:")
                print("X_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.XHalfWidth)
                # print("Y_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.YHalfWidth)
                # Change Settings
                aperture_settings = pupil_mirror.ApertureData.CreateApertureTypeSettings(
                    constants.SurfaceApertureTypes_RectangularAperture)
                aperture_settings._S_RectangularAperture.XHalfWidth = x_halfwidth
                # aperture_settings._S_RectangularAperture.YHalfWidth = y_halfwidth
                pupil_mirror.ApertureData.ChangeApertureTypeSettings(aperture_settings)

                current_apt_sett = pupil_mirror.ApertureData.CurrentTypeSettings
                print("New Settings:")
                print("X_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.XHalfWidth)
                # print("Y_HalfWidth = %.2f" % current_apt_sett._S_RectangularAperture.YHalfWidth)

        if end_surface == LDE.NumberOfSurfaces - 1:
            # (2) Set the Sampling at the last surface
            final_surface = LDE.GetSurfaceAt(end_surface)
            pop_data = final_surface.PhysicalOpticsData
            pop_data.ResampleAfterRefraction = True
            pop_data.AutoResample = False
            pop_data.XSampling = sampling - 1          # somehow the numbering for the sampling is different
            pop_data.YSampling = sampling - 1          # for the Resampling... stupid Zemax
            pop_data.XWidth = XWidth
            pop_data.YWidth = YWidth

        # (3) Set the POP Analysis Parameters
        theAnalyses = system.Analyses
        nanaly = theAnalyses.NumberOfAnalyses
        print("Number of Analyses already open: ", nanaly)
        pop = system.Analyses.New_Analysis_SettingsFirst(constants.AnalysisIDM_PhysicalOpticsPropagation)
        pop.Terminate()
        pop_setting = pop.GetSettings()
        pop_settings = CastTo(pop_setting, 'IAS_')

        if pop.HasAnalysisSpecificSettings == False:

            zemax_dir = os.path.abspath('H:\Zemax')
            # Remember not to mix the POP config with that of HARMONI
            cfg = zemax_dir + '\\Configs\\SWIFT_POP.CFG'

            pop_settings.ModifySettings(cfg, 'POP_END', end_surface)
            pop_settings.ModifySettings(cfg, 'POP_WAVE', wave_idx)
            pop_settings.ModifySettings(cfg, 'POP_FIELD:', field_idx)
            pop_settings.ModifySettings(cfg, 'POP_BEAMTYPE', beam_type)  # 3 for Top Hap beam
            pop_settings.ModifySettings(cfg, 'POP_POWER', 1)
            pop_settings.ModifySettings(cfg, 'POP_SAMPX', N_PIX)
            pop_settings.ModifySettings(cfg, 'POP_SAMPY', N_PIX)
            pop_settings.ModifySettings(cfg, 'POP_PARAM1', waist_x)  # Waist X
            pop_settings.ModifySettings(cfg, 'POP_PARAM2', waist_y)  # Waist Y
            pop_settings.ModifySettings(cfg, 'POP_AUTO', 1)  # needs to go after POP_PARAM
            pop_settings.LoadFrom(cfg)

        # (4) Run POP Analysis
        pop.ApplyAndWaitForCompletion()
        pop_results = pop.GetResults()
        cresults = CastTo(pop_results, 'IAR_')
        data = np.array(cresults.GetDataGrid(0).Values)

        # Close the Analysis to avoid piling up
        theAnalyses.CloseAnalysis(nanaly)

        self.zosapi.CloseFile(save=False)

        total_time = time() - start
        print("Analysis finished in %.2f seconds" % total_time)

        return data, cresults



utils.print_title(message='\nS W I F T', font=None, random_font=False)

if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)


    # Create a Python Standalone Application
    psa = PythonStandaloneApplication()
    zemax_path = os.path.abspath("D:\Research\Experimental\SWIFT Final System")
    zemax_file = "SWIFT_SLICER_FINAL_VERSION.zmx"
    results_path = os.path.join(zemax_path, 'POP')

    zemax_wavelengths = {1: 0.65, 2: 0.85, 3: 1.00}
    wave_idx = 2
    det_pix = 0.0025    # 13.5 microns
    N_pix = 1024
    backoptics_mag = 1.13
    PIX = 128
    x_width = det_pix * N_pix
    # we divide by

    # run some POP
    pop_analysis = POPAnalysisSWIFT(zosapi=psa)

    # Show all 4 slices
    # fig, axes = plt.subplots(1, 4)
    all_slices = []
    for i, config in enumerate([19, 20, 21, 22]):
        pop_settings = {'CONFIG': config, 'N_PIX': N_pix, 'SAMPLING': N_pix,
                        'X_WIDTH': x_width, 'Y_WIDTH': x_width, 'WAVE_IDX': wave_idx}
        pop_data, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file, settings=pop_settings)
        # We have to transpose the PSF because of the weird XY coordinates of the Zemax file
        all_slices.append(pop_data.T)
        minX, minY = cresults.GetDataGrid(0).MinY, cresults.GetDataGrid(0).MinX
        extent = [minX, -minX, minY, -minY]
        #
        # ax = axes[i]
        # img = ax.imshow(np.log10(pop_data.T), origin='lower', extent=extent)
        # # plt.colorbar(img, ax=ax)
        # ax.set_title(r'Slice #%d' % config)
        # ax.set_xlabel(r'X [mm]')
        # ax.set_ylabel(r'Y [mm]')

    psf = np.array(all_slices)
    psf = np.sum(psf, axis=0)

    cmap = 'hot'
    fig, (ax1, ax2) = plt.subplots(1, 2)
    img_psf = ax1.imshow(psf, origin='lower', extent=extent, cmap=cmap)
    ax1.set_xlabel(r'X [mm]')
    ax1.set_ylabel(r'Y [mm]')
    ax1.set_title(r'SWIFT PSF | %.2f $\mu$m' % zemax_wavelengths[wave_idx])

    log_psf = ax2.imshow(np.log10(psf), origin='lower', extent=extent, cmap=cmap)
    log_psf.set_clim(vmin=-4)
    ax2.set_xlabel(r'X [mm]')
    ax2.set_ylabel(r'Y [mm]')
    ax2.set_title(r'%d Pixels | Sampling: %.1f $\mu$m per pixel' % (N_pix, 1000 * det_pix))

    plt.show()

    # First Zero of the PSF is at +- 0.40 mm
    # 4 slices at the exit slit plane occupy +- 0.22 mm
    # so the


    # Impact of apertures


    def zeros_PSF(N_pix, SNR=50):

        # nominal aperture
        x_aper = 4.54  # this is half width
        # At 850 nm, for a half width X of 0.91 we can fit 3 PSF zeroes

        mm_per_zero = 0.91 / 3

        zeros = [2, 3, 4]
        fig, axes = plt.subplots(1, len(zeros))
        for j, N_zeros in enumerate(zeros):

            all_slices = []
            for i, config in enumerate([19, 20, 21]):
                pop_settings = {'CONFIG': config, 'N_PIX': N_pix, 'SAMPLING': N_pix, 'X_WIDTH': x_width, 'Y_WIDTH': x_width,
                                'WAVE_IDX': wave_idx,
                                'PM_X_HALFWIDTH': N_zeros * mm_per_zero}
                pop_data, cresults = pop_analysis.run_pop(zemax_path=zemax_path, zemax_file=zemax_file,
                                                          settings=pop_settings)
                # We have to transpose the PSF because of the weird XY coordinates of the Zemax file
                all_slices.append(pop_data.T)
                minX, minY = cresults.GetDataGrid(0).MinY, cresults.GetDataGrid(0).MinX
                extent = [minX, -minX, minY, -minY]

            psf = np.array(all_slices)
            psf = np.sum(psf, axis=0)
            psf /= np.max(psf)

            psf += np.random.normal(0.0, scale=1/SNR, size=(N_pix, N_pix))

            ax = axes[j]
            img = ax.imshow(psf, origin='lower', extent=extent, cmap=cmap)
            ax.set_xlabel(r'X [mm]')
            ax.set_ylabel(r'Y [mm]')
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_title(r'SWIFT PSF | %.2f $\mu$m | %d PSF Zeros | SNR %d' % (zemax_wavelengths[wave_idx], N_zeros, SNR))

    zeros_PSF(1024)

    plt.show()

    #
    # cmap = 'hot'
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # img_psf = ax1.imshow(psf, origin='lower', extent=extent, cmap=cmap)
    # ax1.set_xlabel(r'X [mm]')
    # ax1.set_ylabel(r'Y [mm]')
    # ax1.set_title(r'SWIFT PSF | %.2f $\mu$m' % zemax_wavelengths[wave_idx])
    #
    # log_psf = ax2.imshow(np.log10(psf), origin='lower', extent=extent, cmap=cmap)
    # log_psf.set_clim(vmin=-4)
    # ax2.set_xlabel(r'X [mm]')
    # ax2.set_ylabel(r'Y [mm]')
    # ax2.set_title(r'%d Pixels | Sampling: %.1f $\mu$m per pixel' % (N_pix, 1000 * det_pix))


    # del psa
    # psa = None

