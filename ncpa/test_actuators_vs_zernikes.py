"""
Date: 20th November 2019

Comparison of Basis Function for Wavefront Definition in the context of ML calibration
________________________________________________________________________________________

Is there a basis that works better for calibration?
Is Zernike polynomials better-suited than Actuator Commands models?




"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

import psf
import utils
import calibration
import noise


class Zernike_fit(object):

    """
    Uses Least Squares to fit Wavefronts back and forth between two basis
    the Zernike polynomials and an Actuator Commands model

    """

    def __init__(self, PSF_zernike, PSF_actuator):
        self.PSF_zernike = PSF_zernike
        self.PSF_actuator = PSF_actuator

        # Get the Model matrices
        self.H_zernike = self.PSF_zernike.RBF_mat.copy()
        self.H_zernike_flat = self.PSF_zernike.RBF_flat.copy()
        self.H_actuator = self.PSF_actuator.RBF_mat.copy()
        self.H_actuator_flat = self.PSF_actuator.RBF_flat.copy()

        self.pupil_mask = self.PSF_zernike.pupil_mask.copy()

    def plot_example(self, zern_coef, actu_coef, ground_truth='zernike', k=0, cmap='bwr'):

        if ground_truth == "zernike":
            true_phase = WAVE * 1e3 * np.dot(self.H_zernike, zern_coef.T)[:, :, k]
            fit_phase = WAVE * 1e3 * np.dot(self.H_actuator, actu_coef)[:, :, k]
            names = ['Zernike', 'Actuator']
            print(0)

        elif ground_truth == "actuator":
            true_phase = WAVE * 1e3 * np.dot(self.H_actuator, actu_coef.T)[:, :, k]
            fit_phase = WAVE * 1e3 * np.dot(self.H_zernike, zern_coef)[:, :, k]
            names = ['Actuator', 'Zernike']
            print(1)

        residual = true_phase - fit_phase
        rms0 = np.std(true_phase[self.pupil_mask])
        rms = np.std(residual[self.pupil_mask])

        mins = min(true_phase.min(), fit_phase.min())
        maxs = max(true_phase.max(), fit_phase.max())
        m = min(mins, -maxs)
        # mapp = 'bwr'

        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1 = plt.subplot(1, 3, 1)
        img1 = ax1.imshow(true_phase, cmap=cmap, extent=[-1, 1, -1, 1])
        ax1.set_title('%s Wavefront [$\sigma=%.1f$ nm]' % (names[0], rms0))
        img1.set_clim(m, -m)
        ax1.set_xlim([-RHO_APER, RHO_APER])
        ax1.set_ylim([-RHO_APER, RHO_APER])
        plt.colorbar(img1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(1, 3, 2)
        img2 = ax2.imshow(fit_phase, cmap=cmap, extent=[-1, 1, -1, 1])
        ax2.set_title('%s Fit Wavefront' % names[1])
        img2.set_clim(m, -m)
        ax2.set_xlim([-RHO_APER, RHO_APER])
        ax2.set_ylim([-RHO_APER, RHO_APER])
        plt.colorbar(img2, ax=ax2, orientation='horizontal')

        ax3 = plt.subplot(1, 3, 3)
        img3 = ax3.imshow(residual, cmap=cmap, extent=[-1, 1, -1, 1])
        ax3.set_title('Residual [$\sigma=%.1f$ nm]' % rms)
        img3.set_clim(m, -m)
        ax3.set_xlim([-RHO_APER, RHO_APER])
        ax3.set_ylim([-RHO_APER, RHO_APER])
        plt.colorbar(img3, ax=ax3, orientation='horizontal')

    def fit_actuator_wave_to_zernikes(self, actu_coef, plot=False, cmap='bwr'):
        """
        Fit a Wavefront defined in terms of Actuator commands to
        Zernike polynomials

        :param actu_coef:
        :param plot: whether to plot an example to show the fitting error
        :return:
        """

        actu_wave = np.dot(self.H_actuator_flat, actu_coef.T)
        x_zern = self.least_squares(y_obs=actu_wave, H=self.H_zernike_flat)

        if plot:
            self.plot_example(x_zern, actu_coef, ground_truth="actuator", cmap=cmap)

        return x_zern

    def fit_zernike_wave_to_actuators(self, zern_coef, plot=False, cmap='bwr'):
        """
        Fit a Wavefront defined in terms of Zernike polynomials to the
        model of Actuator Commands

        :param zern_coef:
        :param plot: whether to plot an example to show the fitting error
        :return:
        """

        # Generate Zernike Wavefronts [N_pix, N_pix, N_samples]
        zern_wave = np.dot(self.H_zernike_flat, zern_coef.T)
        x_act = self.least_squares(y_obs=zern_wave, H=self.H_actuator_flat)

        if plot:
            self.plot_example(zern_coef, x_act, ground_truth="zernike", cmap=cmap)

        return x_act

    def least_squares(self, y_obs, H):
        """
        High level definition of the Least Squares fitting problem

        y_obs = H * x_fit + noise
        H.T * y_obs = (H.T * H) * x_fit
        with N = H.T * H
        x_fit = inv(N) * H.T * y_obs

        H is the model matrix that we use for the fit
        For instance, if the wavefront (y_obs) is defined in terms of Zernike polynomials
        H would be the Model Matrix for the Actuator Commands
        and x_act would be the best fit in terms of actuator commands that describe that wavefront
        :param y_obs:
        :param H:
        :return:
        """

        Ht = H.T
        Hty_obs = np.dot(Ht, y_obs)
        N = np.dot(Ht, H)
        invN = np.linalg.inv(N)
        x_fit = np.dot(invN, Hty_obs)

        return x_fit