"""



"""

import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import psf
import utils
import calibration


class RealisticPSF(object):

    def __init__(self, psf_model_param, effects_dict):

        # Read the parameters of the nominal PSF model
        N_PIX = psf_model_param['N_PIX']
        pix = psf_model_param['pix']
        SPAX = psf_model_param['SPAX']
        WAVE = psf_model_param['WAVE']
        N_LEVELS = psf_model_param['N_LEVELS']
        diversity = psf_model_param['DIVERSITY']

        # Parameters
        self.N_PIX = N_PIX        # pixels for the Fourier arrays
        self.pix = pix              # Pixels in the FoV of the PSF images (we crop from N_PIX -> pix)
        self.WAVE = WAVE        # microns | reference wavelength
        self.SPAX = SPAX  # mas | spaxel scale
        self.RHO_APER = utils.rho_spaxel_scale(spaxel_scale=self.SPAX, wavelength=self.WAVE)
        self.RHO_OBSC = 0.30 * self.RHO_APER  # ELT central obscuration
        print("Nominal Parameters | Spaxel Scale and Wavelength")
        utils.check_spaxel_scale(rho_aper=self.RHO_APER, wavelength=self.WAVE)

        # Nominal PSF model
        self.define_nominal_PSF_model(N_levels=N_LEVELS, crop_pix=pix, diversity=diversity)

        # define the set of realistic effects that will be considered
        nyquist_errors = effects_dict['NYQUIST']        # Whether to account for Nyquist sampling errors

        return

    def define_nominal_PSF_model(self, N_levels, crop_pix, diversity):

        print("\nDefining a nominal PSF model")
        zernike_matrices = psf.zernike_matrix(N_levels=N_levels, rho_aper=self.RHO_APER, rho_obsc=self.RHO_APER,
                                              N_PIX=self.N_PIX, radial_oversize=1.0, anamorphic_ratio=1.0)
        zernike_matrix, pupil_mask_zernike, flat_zernike = zernike_matrices
        N_zern = zernike_matrix.shape[-1]
        self.nominal_PSF_model = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=self.N_PIX,
                                                         crop_pix=crop_pix, diversity_coef=np.zeros(N_zern))
        print("Number of aberration coefficients: ", self.nominal_PSF_model.N_coef)
        print("Defocus diversity: %.3f / (2 Pi)" % diversity)

        defocus_zernike = np.zeros(N_zern)
        defocus_zernike[1] = diversity / (2 * np.pi)
        self.nominal_PSF_model.define_diversity(defocus_zernike)

        return

    def generate_bad_PSF_model(self, nyquist=False, nyquist_errors=None):

        # First we check whether we want Nyquist sampling errors
        if nyquist is False:
            pass

        elif nyquist is True:

            SPAX_ERR = 0.05  # Percentage of error [10%]
            WAVE_BAD = WAVE * (1 + SPAX_ERR)


            centers_multiwave = psf.actuator_centres_multiwave(N_actuators=N_actuators, rho_aper=RHO_APER,
                                                               rho_obsc=RHO_OBSC,
                                                               N_waves=N_WAVES, wave0=WAVE, waveN=WAVE_BAD,
                                                               wave_ref=WAVE)
            N_act = len(centers_multiwave[0][0])

            rbf_matrices = psf.actuator_matrix_multiwave(centres=centers_multiwave, alpha_pc=alpha_pc,
                                                         rho_aper=RHO_APER,
                                                         rho_obsc=RHO_OBSC, N_waves=N_WAVES, wave0=WAVE, waveN=WAVE_BAD,
                                                         wave_ref=WAVE, N_PIX=N_PIX)

            PSF_nom = psf.PointSpreadFunction(rbf_matrices[0], N_pix=N_PIX, crop_pix=pix,
                                              diversity_coef=np.zeros(N_act))

    def generate_dataset(self, N_train, N_test, coef_strength, rescale=0.35):

        N_coef = self.nominal_PSF_model.N_coef
        pix = self.nominal_PSF_model.crop_pix
        N_samples = N_train + N_test

        dataset = np.empty((N_samples, pix, pix, 2))
        coefs = coef_strength * np.random.uniform(low=-1, high=1, size=(N_samples, N_coef))

        # Rescale the coefficients to cover a wider range of RMS (so we can iterate)
        rescale_train = np.linspace(1.0, rescale, N_train)
        rescale_test = np.linspace(1.0, 0.5, N_test)
        rescale_coef = np.concatenate([rescale_train, rescale_test])
        coefs *= rescale_coef[:, np.newaxis]

        print("\nGenerating datasets: %d PSF images" % N_samples)
        for i in range(N_samples):

            # First we draw a PSF model from the available


            im0, _s = PSF_model.compute_PSF(coefs[i])
            dataset[i, :, :, 0] = im0

            im_foc, _s = PSF_model.compute_PSF(coefs[i], diversity=True)
            dataset[i, :, :, 1] = im_foc

            if i % 500 == 0:
                print(i)