"""



"""

import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time

import psf
import utils
import calibration

# PSF bits
N_PIX = 128  # pixels for the Fourier arrays
pix = 64  # pixels to crop the PSF images
WAVE = 1.5  # microns | reference wavelength
SPAX = 4.0  # mas | spaxel scale

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
N_levels = 10                        # Number of Zernike radial levels levels
coef_strength = 0.17                # Strength of Zernike aberrations
diversity = 1.0                     # Strength of the Defocus diversity
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
kernel_size = 3

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
        self.N_LEVELS = N_LEVELS
        self.diversity = diversity

        # Nominal PSF model
        self.define_nominal_PSF_model()

        # define the set of realistic effects that will be considered
        self.nyquist_errors = effects_dict['NYQUIST_ERRORS']        # Whether to account for Nyquist sampling errors
        self.diversity_errors = effects_dict['DEFOCUS_ERRORS']        # Whether to account for Diversity errors
        self.anamorphic_errors = effects_dict['ANAMORPHIC_ERRORS']        # Whether to account for Diversity errors


        return

    def define_nominal_PSF_model(self):

        print("\nDefining a nominal PSF model")
        zernike_matrices = psf.zernike_matrix(N_levels=self.N_LEVELS, rho_aper=self.RHO_APER, rho_obsc=self.RHO_OBSC,
                                              N_PIX=self.N_PIX, radial_oversize=1.0, anamorphic_ratio=1.0)
        zernike_matrix, pupil_mask_zernike, flat_zernike = zernike_matrices
        N_zern = zernike_matrix.shape[-1]
        self.nominal_PSF_model = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=self.N_PIX,
                                                         crop_pix=self.pix, diversity_coef=np.zeros(N_zern))
        print("Number of aberration coefficients: ", self.nominal_PSF_model.N_coef)
        print("Defocus diversity: %.3f / (2 Pi)" % self.diversity)

        defocus_zernike = np.zeros(N_zern)
        defocus_zernike[1] = self.diversity / (2 * np.pi)
        self.nominal_PSF_model.define_diversity(defocus_zernike)

        return

    def generate_bad_PSF_model(self, nyquist_errors=False, nyquist_error_value=None,
                               diversity_errors=False, diversity_error_value=None,
                               anamorphic_ratio_errors=False, anamorphic_ratio_value=None):

        # First we check whether we want Nyquist sampling errors
        if nyquist_errors is False:
            RHO_APER = self.RHO_APER
            RHO_OBSC = self.RHO_OBSC

        elif nyquist_errors is True:

            SPAX_ERR = nyquist_error_value
            NEW_SPAX = self.SPAX * (1 + SPAX_ERR)
            # print("Spaxel: %.3f mas" % NEW_SPAX)

            RHO_APER = utils.rho_spaxel_scale(spaxel_scale=NEW_SPAX, wavelength=self.WAVE)
            RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration

        # Now we check whether we want Diversity errors
        if diversity_errors is False:
            diversity_coef = self.diversity

        elif diversity_errors is True:
            diversity_coef = self.diversity * (1 + diversity_error_value)

        # Now we check whether we want Anamorphic magnification errors
        if anamorphic_ratio_errors is False:
            anamorphic_ratio = 1.0
        elif anamorphic_ratio_errors is True:
            anamorphic_ratio = 1 + anamorphic_ratio_value
            # print("Anamorphic ratio: %.3f" % anamorphic_ratio)

        zernike_matrices = psf.zernike_matrix(N_levels=self.N_LEVELS, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                              N_PIX=self.N_PIX, radial_oversize=1.0,
                                              anamorphic_ratio=anamorphic_ratio)
        zernike_matrix, pupil_mask_zernike, flat_zernike = zernike_matrices
        N_zern = zernike_matrix.shape[-1]
        PSF_model = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=self.N_PIX,
                                            crop_pix=self.pix, diversity_coef=np.zeros(N_zern))
        # print("Number of aberration coefficients: ", self.nominal_PSF_model.N_coef)
        # print("Defocus diversity: %.3f / (2 Pi)" % diversity_coef)

        defocus_zernike = np.zeros(N_zern)
        defocus_zernike[1] = diversity_coef / (2 * np.pi)
        PSF_model.define_diversity(defocus_zernike)

        return PSF_model

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

        # Keep track of the errors
        nyquist_errors = self.nyquist_errors["ON"]
        diversity_errors = self.diversity_errors["ON"]
        anamorphic_errors = self.anamorphic_errors["ON"]

        nyquist_array = np.random.uniform(low=self.nyquist_errors["MIN_ERR"],
                                                high=self.nyquist_errors["MAX_ERR"],
                                                size=N_samples) if nyquist_errors is True else None
        diversity_array = np.random.uniform(low=self.diversity_errors["MIN_ERR"],
                                            high=self.diversity_errors["MAX_ERR"],
                                            size=N_samples) if diversity_errors is True else None

        anamorphic_array = np.random.uniform(low=self.anamorphic_errors["MIN_ERR"],
                                              high=self.anamorphic_errors["MAX_ERR"],
                                              size=N_samples) if anamorphic_errors is True else None

        print("\nGenerating datasets: %d PSF images" % N_samples)
        print("Errors considered: ")
        print("Nyquist errors: ", self.nyquist_errors["ON"])
        print("Diversity errors: ", self.diversity_errors["ON"])
        print("Anamorphic errors: ", self.anamorphic_errors["ON"])
        start = time()
        for i in range(N_samples):

            if i % 500 == 0:
                print(i)

            nyquist_error_value = nyquist_array[i] if nyquist_errors is True else None
            diversity_error_value = diversity_array[i] if diversity_errors is True else None
            anamorphic_ratio_value = anamorphic_array[i] if anamorphic_errors is True else None

            PSF_model = self.generate_bad_PSF_model(nyquist_errors=nyquist_errors, nyquist_error_value=nyquist_error_value,
                                                    diversity_errors=diversity_errors, diversity_error_value=diversity_error_value,
                                                    anamorphic_ratio_errors=anamorphic_errors, anamorphic_ratio_value=anamorphic_ratio_value)

            im0, _s = PSF_model.compute_PSF(coefs[i])
            dataset[i, :, :, 0] = im0

            im_foc, _s = PSF_model.compute_PSF(coefs[i], diversity=True)
            dataset[i, :, :, 1] = im_foc

            if i % 100 == 0 and i > 0:
                delta_time = time() - start
                estimated_total = delta_time * N_samples / i
                remaining = estimated_total - delta_time
                message = "%d | t=%.1f sec | ETA: %.1f sec (%.1f min)" % (i, delta_time, remaining, remaining / 60)
                print(message)

        errors = [nyquist_array, diversity_array, anamorphic_array]

        return dataset[:N_train], coefs[:N_train], dataset[N_train:], coefs[N_train:], errors

    def update_PSF_images(self, coef, errors_array):


        N_PSF = coef.shape[0]
        new_PSF = np.zeros((N_PSF, self.pix, self.pix, 2))

        nyquist_array = errors_array[0]
        diversity_array = errors_array[1]
        anamorphic_array = errors_array[2]

        print("Updating PSF images")
        start = time()
        for i in range(N_PSF):

            if i % 100 == 0 and i > 0:
                delta_time = time() - start
                estimated_total = delta_time * N_PSF / i
                remaining = estimated_total - delta_time
                message = "%d | t=%.1f sec | ETA: %.1f sec (%.1f min)" % (i, delta_time, remaining, remaining / 60)
                print(message)

            # Get the proper PSF model
            bad_model = self.generate_bad_PSF_model(nyquist_errors=True, nyquist_error_value=nyquist_array[i],
                                                    diversity_errors=True, diversity_error_value=diversity_array[i],
                                                    anamorphic_ratio_errors=True, anamorphic_ratio_value=anamorphic_array[i])

            psf_nom, s_nom = bad_model.compute_PSF(coef[i])
            psf_foc, s_foc = bad_model.compute_PSF(coef[i], diversity=True)
            new_PSF[i, :, :, 0] = psf_nom
            new_PSF[i, :, :, 1] = psf_foc

        return new_PSF

    def calculate_roc(self, calibration_model, test_images, test_coef, errors_array, N_iter=3):

        N_test = test_coef.shape[0]
        N_eps = 200
        percentage = np.linspace(0, 100, N_eps)
        roc = np.zeros((N_iter, N_eps))

        # get the norm of the test coef
        norm_test0 = norm(test_coef, axis=1)

        psf0, coef0 = test_images, test_coef
        for k in range(N_iter):
            print("\nIteration #%d" % (k + 1))
            # predict the coefficients
            guess = calibration_model.predict(psf0)
            residual = coef0 - guess
            norm_residual = norm(residual, axis=1)
            # compare the norm of the residual to that of the original coefficients
            ratios = norm_residual / norm_test0 * 100
            # calculate the ROC
            for j in range(N_eps):
                roc[k, j] = np.sum(ratios < percentage[j]) / N_test

            # update the PSF
            new_PSF = self.update_PSF_images(residual, errors_array)

            # Overwrite the arrays for the next iteration
            psf0 = new_PSF
            coef0 = residual

        return roc, percentage, residual

    def calculate_cum_strehl(self, calibration_model, test_images, test_coef, errors_array, N_iter=3):

        N_PSF = test_coef.shape[0]
        N_strehl = 200
        strehl_threshold = np.linspace(0, 1, N_strehl)
        strehl_profiles = np.zeros((N_iter + 1, N_strehl))

        # calculate the original strehl
        s0 = self.calculate_strehl(test_coef)
        for j in range(N_strehl):
            strehl_profiles[0, j] = np.sum(s0 > strehl_threshold[j]) / N_PSF

        # Add Noise
        test_images = calibration_model.noise_effects.add_readout_noise(test_images, RMS_READ=1 / SNR)

        psf0, coef0 = test_images, test_coef
        for k in range(N_iter):
            print("\nIteration #%d" % (k + 1))
            # predict the coefficients
            guess = calibration_model.cnn_model.predict(psf0)
            residual = coef0 - guess

            new_strehl = self.calculate_strehl(residual)
            for j in range(N_strehl):
                strehl_profiles[k + 1, j] = np.sum(new_strehl > strehl_threshold[j]) / N_PSF

            # update the PSF
            new_PSF = self.update_PSF_images(residual, errors_array)
            new_PSF = calibration_model.noise_effects.add_readout_noise(new_PSF, RMS_READ=1 / SNR)

            # Overwrite the arrays for the next iteration
            psf0 = new_PSF
            coef0 = residual

        return strehl_profiles, strehl_threshold, residual


    def calculate_strehl(self, coef):

        N_PSF = coef.shape[0]
        strehl = np.zeros(N_PSF)
        for k in range(N_PSF):
            psf_nom, s_nom = self.nominal_PSF_model.compute_PSF(coef[k])
            strehl[k] = s_nom
        return strehl



if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Populate the dictionary with PSF model parameters
    psf_model_param = {"N_PIX": N_PIX,
                       "pix": pix,
                       "SPAX": SPAX,
                       "WAVE": WAVE,
                       "N_LEVELS": N_levels,
                       "DIVERSITY": diversity}

    effects_dict = {"NYQUIST_ERRORS": {"ON": True,
                                       "MIN_ERR": -0.10,
                                       "MAX_ERR": 0.10},
                    "DEFOCUS_ERRORS": {"ON": True,
                                       "MIN_ERR": -0.10,
                                       "MAX_ERR": 0.10},
                    "ANAMORPHIC_ERRORS": {"ON": True,
                                          "MIN_ERR": -0.10,
                                          "MAX_ERR": 0.10}}

    realistic_PSF = RealisticPSF(psf_model_param=psf_model_param, effects_dict=effects_dict)
    PSF_zernike = realistic_PSF.nominal_PSF_model

    N_train = 20000
    train_PSF, train_coef, test_PSF, test_coef, errors = realistic_PSF.generate_dataset(N_train, N_test, coef_strength, rescale)

    layer_filters = [32, 32]
    epochs = 20
    SNR = 250
    readout_copies = 5

    calib_zern = calibration.Calibration(PSF_model=realistic_PSF.nominal_PSF_model)
    calib_zern.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='relu', learning_rate=5e-4)
    # trainable_count = count_params(calib_zern.cnn_model.trainable_weights) / 1000
    # param_sigm[i, j] = trainable_count  # How many K parameters
    losses = calib_zern.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                N_loops=1, epochs_loop=5, verbose=1, batch_size_keras=32,
                                                plot_val_loss=False, readout_noise=True,
                                                RMS_readout=[1. / SNR], readout_copies=readout_copies)

    guess_coef = calib_zern.cnn_model.predict(test_PSF)
    residual_coef = test_coef - guess_coef
    norm_res = np.mean(norm(residual_coef, axis=1))

    # See if there is a dependence on the errors
    errors_array = np.array(errors)
    errors_array = errors_array[:, N_train:]

    N_roc = 5
    # roc, percentage, _res = realistic_PSF.calculate_roc(calib_zern.cnn_model, test_PSF, test_coef, errors_array, N_roc)

    strehl_roc, threshold, _res = realistic_PSF.calculate_cum_strehl(calib_zern, test_PSF, test_coef, errors_array, N_roc)

    # strehl = realistic_PSF.calculate_strehl(_res)
    #
    # colors = cm.Greens(np.linspace(0.5, 1, N_roc))
    # plt.figure()
    # for k in range(N_roc):
    #     roc_data = roc[k]
    #     auc = np.mean(roc_data)
    #     plt.plot(percentage, roc_data, color=colors[k], label="%d (%.3f)" % (k + 1, auc))
    # # plt.xscale('log')
    # plt.xlim([0, 100])
    # plt.grid(True)
    # plt.xticks(np.arange(0, 110, 10))
    # plt.xlabel(r'Ratio Norm Residuals / Norm Test Coef [percent]')
    # plt.ylabel(r'Cumulative')
    # plt.legend(title='Iteration (AUC)')
    # plt.title(r'ROC | Architecture: [Conv1, Conv2, FC]')
    # plt.show()

    colors = cm.Blues(np.linspace(0.5, 1, N_roc))
    plt.figure()
    for k in range(N_roc):
        strehl_data = strehl_roc[k]
        auc = np.mean(strehl_data)
        plt.plot(threshold, strehl_data, color=colors[k], label="%d (%.3f)" % (k + 1, auc))
    # plt.xscale('log')
    plt.xlim([0, 1])
    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r'Strehl ratio [ ]')
    plt.ylabel(r'Cumulative')
    plt.legend(title='Iteration (AUC)')
    plt.title(r'Strehl ROC | Architecture: [Conv1, Conv2, FC]')
    plt.show()



    residual_norm = norm(residual_coef, axis=1)

    r = norm(_res, axis=1)
    colors = cm.Reds(r/np.max(r))

    plt.figure()
    plt.scatter(errors_array[1], errors_array[2], color=colors, s=20)
    plt.show()

    test_PSF_noise = calib_zern.noise_effects.add_readout_noise(test_PSF, RMS_READ=1 / SNR)

    # Hyper
    N_zern = realistic_PSF.nominal_PSF_model.N_coef
    input_shape = (pix, pix, 2)

    import keras
    from keras.models import Sequential
    from keras.layers import Conv2D, Flatten, Dense

    # from tensorflow import keras
    # from tensorflow.keras.layers import (
    #     Conv2D,
    #     Dense,
    #     Dropout,
    #     Flatten,
    #     MaxPooling2D
    # )
    from kerastuner.tuners import Hyperband
    from kerastuner import HyperModel

    class CNNHyperModelMassive(HyperModel):

        def __init__(self, input_shape, num_classes):
            self.input_shape = input_shape
            self.num_classes = num_classes
            self.activation = 'relu'

        def build(self, hp):

            model = keras.Sequential()
            model.add(Conv2D(filters=hp.Choice('num_filters_0', values=[8, 16, 32, 64]),
                             kernel_size=hp.Choice('kernel_size', values=[3, 4, 5]),
                             activation=self.activation, input_shape=self.input_shape))

            for i in range(hp.Int('num_layers', 1, 3)):
                model.add(Conv2D(filters=hp.Choice('num_filters_%d' % (i), values=[8, 16, 32, 64]),
                                 kernel_size=hp.Choice('kernel_size_%d' % (i), values=[3, 4, 5]),
                                 activation=self.activation))
            model.add(Flatten())
            model.add(Dense(N_zern))
            model.summary()

            model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-4, 1e-4])),
                          loss='mean_squared_error')
            return model

    hypermodel_massive = CNNHyperModelMassive(input_shape=input_shape, num_classes=N_zern)

    HYPERBAND_MAX_EPOCHS = 10
    MAX_TRIALS = 20
    EXECUTION_PER_TRIAL = 2


    tuner = Hyperband(hypermodel_massive, max_epochs=HYPERBAND_MAX_EPOCHS, objective='val_loss',
                      executions_per_trial=EXECUTION_PER_TRIAL, directory='hyperband', project_name='massive')

    tuner.search_space_summary()


    train_PSF_noisy = np.zeros(())
    tuner.search(train_PSF_extra, train_coef_extra, epochs=10,
                 validation_data=(test_PSF_extra, test_coef_extra))

    tuner.results_summary()
    best_model = tuner.get_best_models(num_models=1)[0]

    # best HP
    best_hp = tuner.get_best_hyperparameters()[0]
    print(best_hp.values)

