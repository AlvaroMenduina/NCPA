"""
Effect of field-dependent aberrations on the PSF for HARMONI

Author: Alvaro Menduina
Date: August 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time
import zernike as zern

from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import constants
from win32com.client import CastTo


class ZernikePhase(object):

    def __init__(self, N_zern):
        N_PIX = 1024
        x = np.linspace(-1, 1, N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x, x)
        rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
        pupil = rho <= 1.0

        rho, theta = rho[pupil], theta[pupil]
        zernike = zern.ZernikeNaive(mask=pupil)
        _phase = zernike(coef=np.zeros(N_zern), rho=rho, theta=theta, normalize_noll=False,
                         mode='Jacobi', print_option='Silent')
        H_flat = zernike.model_matrix[:, 3:]
        self.H_matrix = zern.invert_model_matrix(H_flat, pupil)
        self.pupil_mask = pupil
        self.N_zern = self.H_matrix.shape[-1]

    def transform_zemax_coefficients(self, zemax_coeff):
        # Zemax Zernike Standard Phase: Defocus, horizontal Astig, oblique Astig, horizontal Coma, vertical Coma,
        # H_matrix: [-] oblique Astig, defocus, horiz Astig, oblique trefoil, horizontal coma

        # Dictionary that maps the Zemax coefficients to the H_matrix
        zemax_dict = {0: [1, 0],    # Defocus
                      1: [2, 1],    # Horizontal Astigmatism [sign flipped]
                      2: [0, 1],    # Oblique Astigmatism [s f]
                      3: [4, 0],    # Horizontal Coma
                      4: [5, 1],    # Vertical Coma [s f]
                      5: [9, 0],    # Spherical
                      6: [3, 1],    # Oblique Trefoil [s f]
                      7: [6, 0],    # Horizontal Trefoil
                      8: [10, 1],   # Secondary Horizontal Coma [s f]
                      9: [8, 1],    # Secondary Oblique Coma [s f]
                      10: [14, 0],  # Other coma
                      11: [15, 1],  # Other coma [s f]
                      }

        N_zemax = zemax_coeff.shape[0]
        zern_coef = np.zeros(self.N_zern)
        for i in range(N_zemax):
            h_index, sign_k = zemax_dict[i]
            zern_coef[h_index] = (-1) ** sign_k * zemax_coeff[i]
        return zern_coef

    def show_phase(self, zemax_coef):
        N_config = zemax_coef.shape[0]
        RMS = []
        fig, axes = plt.subplots(1, N_config)
        maxes = []
        for j_config in range(N_config):
            zern_coef = self.transform_zemax_coefficients(zemax_coef[j_config])
            phase = np.dot(self.H_matrix, zern_coef)
            rms = np.std(phase[self.pupil_mask])
            RMS.append(rms)
            cmax = max(np.max(phase), -np.min(phase))
            maxes.append(cmax)

        MAX = np.max(maxes)
        for j_config in range(N_config):
            zern_coef = self.transform_zemax_coefficients(zemax_coef[j_config])
            phase = np.dot(self.H_matrix, zern_coef)
            ax = axes[j_config]
            img = ax.imshow(phase, cmap='RdBu')
            img.set_clim(-MAX, MAX)
            # plt.colorbar(img, ax=ax)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        mean_RMS = np.mean(RMS)
        return mean_RMS


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

    def __init__(self, zosapi, N_pix):

        self.zosapi = zosapi
        self.N_pix = N_pix

        self.pm_zernike = 33        # The Surface number for the Zernike phase at the Pupil Mirror

        self.config_slices = {'-15': 71, '-14': 5, '-13': 69,
                              '-12': 7, '-11': 67, '-10': 9, '-9': 65, '-8': 11, '-7': 63,
                              '-6': 13, '-5': 61, '-4': 15, '-3': 59, '-2': 17, '-1': 57,
                              'Central': 19,
                              '+1': 55, '+2': 21, '+3': 53, '+4': 23, '+5': 51, '+6': 25,
                              '+7': 49, '+8': 27, '+9': 47, '+10': 29, '+11': 45, '+12': 31,
                              '+13': 43, '+14': 33, '+15': 41}

    def set_field_aberrations(self, system, N_zernike, N_slices, z_max):

        self.zernike = ZernikePhase(N_zern=20)

        MCE = system.MCE  # Multi Configuration Editor
        LDE = system.LDE  # Lens Data Editor

        # check that we are looking at the proper Zemax surface
        PM_Zernike = LDE.GetSurfaceAt(self.pm_zernike).Comment
        if PM_Zernike != "PM Zernike":
            raise ValueError("Surface #%d is not the PM Zernike" % self.pm_zernike)

        # we want to consider N_slices. Select which config numbers we need
        delta = (N_slices - 1) // 2
        config_keys = ['-%d' % (i + 1) for i in range(delta)] + ['Central'] + ['+%d' % (i + 1) for i in range(delta)]
        configs = [pop_analysis.config_slices[key] for key in config_keys]
        print(configs)

        # Create a set of random Zernike polynomials for each slice
        zern_coef = np.random.uniform(low=-z_max, high=z_max, size=(N_slices, N_zernike))

        # Check how many Operands we have in the MCE already
        Nops = MCE.NumberOfOperands
        print(MCE.NumberOfOperands)
        for k in range(N_zernike):
            zernike_op = MCE.AddOperand()
            zernike_op.ChangeType(constants.MultiConfigOperandType_PRAM)
            zernike_op.Param1 = self.pm_zernike         # the Surface
            zernike_op.Param2 = 19 + k                  # the Parameter. In this case the Zernike polynomial
            # We start at Param19 as that is the Zernike Defocus

            # Set the values of the Zernike coefficients for each slices
            for j, config in enumerate(configs):
                zernike_op.GetOperandCell(config).DoubleValue = zern_coef[j, k]

        print(MCE.NumberOfOperands)
        # for k in range(N_zernike):
        #     op = MCE.GetOperandAt(Nops + k + 1)
        #     for j, config in enumerate(configs):
        #         val = op.GetOperandCell(config).DoubleValue
        #         print(val)
        # self.zosapi.CloseFile(save=True)
        # for j, config in enumerate(configs):
        mean_rms = self.zernike.show_phase(zern_coef)

        return mean_rms

    def run_pop(self, system, settings, defocus_pv=None):

        start = time()

        pop_sampling = {32: 1, 64: 2, 128: 3, 256: 4, 512: 5, 1024: 6, 2048: 7, 4096: 8}
        beam_types = {'top_hat': 3}
        waist_x = 0.189
        waist_y = 0.095
        N_pix = settings['SAMPLING']
        N_slices = settings['N_SLICES']
        slice_size = 0.130              # Size of a slice at the exit focal plane (slits)

        # Get some info on the system
        # system = self.zosapi.TheSystem  # The Optical System
        MCE = system.MCE  # Multi Configuration Editor
        LDE = system.LDE  # Lens Data Editor

        # Read the POP Settings
        config = settings['CONFIG']
        wave_idx = settings['WAVE_IDX'] if 'WAVE_IDX' in settings else 1
        field_idx = settings['FIELD_IDX'] if 'FIELD_IDX' in settings else 1
        end_surface = settings['END_SURFACE'] if 'END_SURFACE' in settings else LDE.NumberOfSurfaces - 1
        sampling = pop_sampling[settings['SAMPLING']]
        beam_type = beam_types[settings['BEAM_TYPE']] if 'BEAM_TYPE' in settings else 3

        # print("Zernike: ", system.LDE.GetSurfaceAt(2).TypeName)
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
        theAnalyses = system.Analyses
        nanaly = theAnalyses.NumberOfAnalyses
        # for i in range(1, nanaly + 1)[::-1]:  # must close in reverse order
        #     theAnalyses.CloseAnalysis(i)
        print("Number of Analyses: ", nanaly)
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
        CastTo(pop, 'ISystemTool').Close()

        # for i in range(1, nanaly + 1)[::-1]:  # must close in reverse order

        theAnalyses.CloseAnalysis(nanaly)
        # self.zosapi.CloseFile(save=False)

        total_time = time() - start
        print("Analysis finished in %.2f seconds" % total_time)

        return data, cresults

    def fix_anamorph(self, pop_psf):
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

    def calculate_pop_psf(self, N_zernike, N_slices, z_max, wave_idx, defocus_pv):

        # check that the file name is correct and the zemax file exists
        if os.path.exists(os.path.join(zemax_path, zemax_file)) is False:
            raise FileExistsError("%s does NOT exist" % zemax_file)

        print("\nOpening Zemax File: ", zemax_file)
        self.zosapi.OpenFile(os.path.join(zemax_path, zemax_file), False)

        # Get some info on the system
        system = self.zosapi.TheSystem  # The Optical System

        if z_max != 0.0:
            # Set the Field-dependent aberrations
            mean_rms = self.set_field_aberrations(system, N_zernike, N_slices, z_max=z_max)
        else:
            mean_rms = 0.0

        delta = (N_slices - 1) // 2
        config_keys = ['-%d' % (i + 1) for i in range(delta)] + ['Central'] + ['+%d' % (i + 1) for i in range(delta)]
        configs = [pop_analysis.config_slices[key] for key in config_keys]

        # Run the POP PSF calculation
        pop_psf = []
        for config_slice in configs:
            pop_settings = {'CONFIG': config_slice, 'SAMPLING': self.N_pix, 'N_SLICES': N_slices, 'WAVE_IDX': wave_idx}
            pop_slit, cresults = self.run_pop(system=system, settings=pop_settings, defocus_pv=defocus_pv)
            pop_psf.append(pop_slit)
        pop_psf = np.array(pop_psf)
        pop_psf = np.sum(pop_psf, axis=0)

        # minX, minY = cresults.GetDataGrid(0).MinX, cresults.GetDataGrid(0).MinY
        # pop_extent = [minX, -minX, minY, -minY]

        pop_psf = self.fix_anamorph(pop_psf)

        self.zosapi.CloseFile(save=False)
        return pop_psf, mean_rms
#
# def zernike_phase(coef):
#
#     N_PIX = 1024
#     x = np.linspace(-1, 1, N_PIX, endpoint=True)
#     xx, yy = np.meshgrid(x, x)
#     rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
#     pupil = rho <= 1.0
#
#     rho, theta = rho[pupil], theta[pupil]
#     zernike = zern.ZernikeNaive(mask=pupil)
#     _phase = zernike(coef=np.zeros(N_zern), rho=rho / (radial_oversize * rho_aper), theta=theta, normalize_noll=False,
#                      mode='Jacobi', print_option='Silent')
#     H_flat = zernike.model_matrix[:, 3:]
#     H_matrix = zern.invert_model_matrix(H_flat, pupil)


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    zemax_path = os.path.abspath("D:\Research\Python Simulations\Tolerances")
    zemax_file = "HARMONI_TOLERANCES.zmx"
    results_path = os.path.abspath("D:\Research\Python Simulations\Tolerances\Results")

    # Create a Python Standalone Application
    psa = PythonStandaloneApplication()

    #
    N_pix = 512
    pop_analysis = POPAnalysis(zosapi=psa, N_pix=N_pix)

    N_zernike = 12
    N_slices = 7
    width = N_slices * 0.130
    extent = [-width/2, width/2, -width/2, width/2]
    psf, _r = pop_analysis.calculate_pop_psf(N_zernike, N_slices, z_max=0.0, wave_idx=1, defocus_pv=None)
    pop_peak = np.max(psf)
    psf /= pop_peak

    psf_field, mean_rms = pop_analysis.calculate_pop_psf(N_zernike, N_slices, z_max=0.01, wave_idx=1, defocus_pv=None)
    psf_field /= pop_peak

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    cmap = 'inferno'
    # Nominal PSF (No Field Aberrations
    img1 = ax1.imshow(psf, origin='lower', cmap=cmap, extent=extent)
    plt.colorbar(img1, ax=ax1, orientation='horizontal')
    # ax1.set_xlabel(r'X [mm]')
    ax1.set_xlabel(r'X [mm]')
    ax1.set_ylabel(r'Y [mm]')

    img2 = ax2.imshow(psf_field, origin='lower', cmap=cmap, extent=extent)
    plt.colorbar(img2, ax=ax2, orientation='horizontal')
    ax2.set_xlabel(r'X [mm]')
    ax2.set_ylabel(r'Y [mm]')

    diff = psf_field - psf
    cmax = max(np.max(diff), -np.min(diff))
    img3 = ax3.imshow(diff, origin='lower', cmap='bwr', extent=extent)
    plt.colorbar(img3, ax=ax3, orientation='horizontal')
    img3.set_clim(-cmax, cmax)
    plt.tight_layout()
    plt.show()

    # 

    from scipy.stats import reciprocal
    a = 1e-5
    b = 1e-1
    r = reciprocal.rvs(a, b, size=500)

    rms, mean_diff = [], []
    for z in r:
        psf_field, mean_rms = pop_analysis.calculate_pop_psf(N_zernike, N_slices, z_max=z, wave_idx=1,
                                                             defocus_pv=None)
        psf_field /= pop_peak
        diff = psf_field - psf

        rms.append(mean_rms)
        mean_diff.append(np.mean(np.abs(diff)))
        plt.close('all')

    plt.figure()
    plt.scatter(rms, mean_diff, s=2)
    # plt.xscale('log')
    plt.show()



