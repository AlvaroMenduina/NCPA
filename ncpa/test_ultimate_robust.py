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
pix = 32  # pixels to crop the PSF images
WAVE = 1.5  # microns | reference wavelength
SPAX = 4.0  # mas | spaxel scale

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
N_levels = 8                        # Number of Zernike radial levels levels
coef_strength = 0.20                # Strength of Zernike aberrations
diversity = 1.0                     # Strength of the Defocus diversity
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
kernel_size = 3
SNR = 250
readout_copies = 5


class NoiseEffects(object):

    """
    A variety of Noise Effects to add to PSF images
    """

    def __init__(self):

        pass

    def add_readout_noise(self, PSF_image, RMS_READ, sigma_offset=5.0):
        """
        Add Readout Noise in the form of additive Gaussian noise
        at a given RMS_READ equal to 1 / SNR, the Signal To Noise ratio

        Assuming the perfect PSF has a peak of 1.0
        :param PSF_image: datacube of [pix, pix, 2] nominal and defocus PSF images
        :param RMS_READ: 1/ SNR
        :param sigma_offset: add [sigma_offset] * RMS_READ to avoid having negative values
        :return:
        """

        pix, N_chan = PSF_image.shape[0], PSF_image.shape[-1]
        PSF_image_noisy = np.zeros_like(PSF_image)
        # print("Adding Readout Noise with RMS: %.4f | SNR: %.1f" % (RMS_READ, 1./RMS_READ))
        for j in range(N_chan):
            read_out = np.random.normal(loc=0, scale=RMS_READ, size=(pix, pix))
            PSF_image_noisy[:, :, j] = PSF_image[:, :, j] + read_out

        # Add a X Sigma offset to avoid having negative values
        PSF_image_noisy += sigma_offset * RMS_READ

        return PSF_image_noisy

    def add_flat_field(self, PSF_image, flat_delta):
        """
        Add Flat Field uncertainties in the form of a multiplicative map
        of Uniform Noise [1 - delta, 1 + delta]

        Such that if delta is 10 %, the pixel sensitivity is known to +- 10 %
        :param PSF_image: datacube of [pix, pix, 2] nominal and defocus PSF images
        :param flat_delta: range of the Flat Field uncertainty [1- delta, 1 + delta]
        :return:
        """

        pix, N_chan = PSF_image.shape[0], PSF_image.shape[-1]
        new_PSF = np.zeros_like(PSF_image)
        # print(r"Adding Flat Field errors [1 - $\delta$, 1 + $\delta$]: $\delta$=%.3f" % flat_delta)
        # sigma_uniform = flat_delta / np.sqrt(3)
        flat_field = np.random.uniform(low=1 - flat_delta, high=1 + flat_delta, size=(pix, pix))
        for j in range(N_chan):
            new_PSF[:, :, j] = flat_field * PSF_image[:, :, j]

        return new_PSF


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

        self.noise_model = NoiseEffects()
        self.readout_noise = effects_dict['READOUT_NOISE']
        self.flat_field = effects_dict['FLAT_FIELD']

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
        flat_field = self.flat_field["ON"]
        readout_noise = self.readout_noise["ON"]

        nyquist_array = np.random.uniform(low=self.nyquist_errors["MIN_ERR"],
                                                high=self.nyquist_errors["MAX_ERR"],
                                                size=N_samples) if nyquist_errors is True else None
        diversity_array = np.random.uniform(low=self.diversity_errors["MIN_ERR"],
                                            high=self.diversity_errors["MAX_ERR"],
                                            size=N_samples) if diversity_errors is True else None

        anamorphic_array = np.random.uniform(low=self.anamorphic_errors["MIN_ERR"],
                                              high=self.anamorphic_errors["MAX_ERR"],
                                              size=N_samples) if anamorphic_errors is True else None

        flat_field_array = np.random.uniform(low=self.flat_field["MIN_ERR"],
                                              high=self.flat_field["MAX_ERR"],
                                              size=N_samples) if flat_field is True else None

        readout_noise_array = np.random.uniform(low=self.readout_noise["MIN_SNR"],
                                              high=self.readout_noise["MAX_SNR"],
                                              size=N_samples) if readout_noise is True else None

        print("\nGenerating datasets: %d PSF images" % N_samples)
        print("Errors considered: ")
        print("Nyquist errors: ", self.nyquist_errors["ON"])
        print("Diversity errors: ", self.diversity_errors["ON"])
        print("Anamorphic errors: ", self.anamorphic_errors["ON"])
        print("Readout Noise: ", self.readout_noise["ON"])
        print("Flat Field: ", self.flat_field["ON"])

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

            # print(np.std(PSF_model.diversity_phase))

            im0, _s = PSF_model.compute_PSF(coefs[i])
            dataset[i, :, :, 0] = im0

            im_foc, _s = PSF_model.compute_PSF(coefs[i], diversity=True)
            dataset[i, :, :, 1] = im_foc

            # Add Noise effects
            if flat_field == True:
                # img = dataset[i].copy()
                dataset[i] = self.noise_model.add_flat_field(dataset[i], flat_delta=flat_field_array[i])
                # delta = img - dataset[i]
                # plt.figure()
                # plt.imshow(delta[:, :, 0], cmap='RdBu')
                # plt.colorbar()
                # plt.show()

            if readout_noise == True:
                rms = 1 / readout_noise_array[i]
                dataset[i] = self.noise_model.add_readout_noise(dataset[i], RMS_READ=rms)

            if i % 100 == 0 and i > 0:
                delta_time = time() - start
                estimated_total = delta_time * N_samples / i
                remaining = estimated_total - delta_time
                message = "%d | t=%.1f sec | ETA: %.1f sec (%.1f min)" % (i, delta_time, remaining, remaining / 60)
                print(message)

        errors = [nyquist_array, diversity_array, anamorphic_array, flat_field_array, readout_noise_array]

        return dataset[:N_train], coefs[:N_train], dataset[N_train:], coefs[N_train:], errors

    def update_PSF_images(self, coef, errors_array):

        N_PSF = coef.shape[0]
        new_PSF = np.zeros((N_PSF, self.pix, self.pix, 2))

        nyquist_array = errors_array[0]
        diversity_array = errors_array[1]
        anamorphic_array = errors_array[2]
        flat_field = errors_array[3]
        readout_noise = errors_array[4]

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

            if flat_field is not None:
                new_PSF[i] = self.noise_model.add_flat_field(new_PSF[i], flat_delta=flat_field[i])

            if readout_noise is not None:
                rms = 1 / readout_noise[i]
                new_PSF[i] = self.noise_model.add_readout_noise(new_PSF[i], RMS_READ=rms)
                # print('A')

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

        # # Add Noise
        # test_images = calibration_model.noise_effects.add_readout_noise(test_images, RMS_READ=1 / SNR)

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
            # new_PSF = calibration_model.noise_effects.add_readout_noise(new_PSF, RMS_READ=1 / SNR)

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

    # Specify which realistic effects we want to include and their range
    effects_dict = {"NYQUIST_ERRORS": {"ON": True, "MIN_ERR": -0.05, "MAX_ERR": 0.05},
                    "DEFOCUS_ERRORS": {"ON": True, "MIN_ERR": -0.05, "MAX_ERR": 0.05},
                    "ANAMORPHIC_ERRORS": {"ON": True, "MIN_ERR": -0.05, "MAX_ERR": 0.05},
                    "FLAT_FIELD": {"ON": True, "MIN_ERR": -0.05, "MAX_ERR": 0.05},
                    "READOUT_NOISE": {"ON": True, "MIN_SNR": 250, "MAX_SNR": 500}}
    effects_label = ["Nyquist Err"]

    realistic_PSF = RealisticPSF(psf_model_param=psf_model_param, effects_dict=effects_dict)
    PSF_zernike = realistic_PSF.nominal_PSF_model
    N_zern = PSF_zernike.N_coef

    N_train = 25000
    N_test = 2500
    train_PSF, train_coef, test_PSF, test_coef, errors = realistic_PSF.generate_dataset(N_train, N_test, coef_strength, rescale)

    layer_filters = [32, 32]
    epochs = 20

    calib_zern = calibration.Calibration(PSF_model=realistic_PSF.nominal_PSF_model)
    # calib_zern.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='relu', learning_rate=5e-4)
    # # trainable_count = count_params(calib_zern.cnn_model.trainable_weights) / 1000
    # # param_sigm[i, j] = trainable_count  # How many K parameters
    # losses = calib_zern.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
    #                                             N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
    #                                             plot_val_loss=False, readout_noise=False,
    #                                             RMS_readout=[1. / SNR], readout_copies=readout_copies)
    #
    # guess_coef = calib_zern.cnn_model.predict(test_PSF)
    # residual_coef = test_coef - guess_coef
    # norm_res = norm(residual_coef, axis=1)
    # print(np.mean(norm_res))

    str0 = realistic_PSF.calculate_strehl(test_coef)

    # See if there is a dependence on the errors
    errors_array = np.array(errors)
    errors_array = errors_array[:, N_train:]
    errors_labels = list(effects_dict.keys())

    # ================================================================================================================ #
    #                          HYPERPARAMETER TUNING
    # ================================================================================================================ #

    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, Dense

    from kerastuner.tuners import Hyperband
    from kerastuner import HyperModel

    class CNNHyperModel(HyperModel):
        """
        Here we subclass HyperModel to define our own architecture
        """

        def __init__(self, input_shape, num_classes):
            self.input_shape = input_shape
            self.num_classes = num_classes
            # self.activation = 'relu'

        def build(self, hp):
            """
            Build a Keras model with varying hyper-parameters

            Things that we allow to vary:
                - Adam optimiser learning rate
                - Number of Conv. layers
                - Kernel size for each layer
                - Filters for each layer
                - Activation function for each layer
            :param hp:
            :return:
            """

            model = Sequential()
            model.add(Conv2D(filters=hp.Choice('num_filters_0', values=[8, 16, 32, 64]),
                             kernel_size=hp.Choice('kernel_size_0', values=[3, 4, 5]),
                             activation=hp.Choice('activation_0', values=['relu', 'tanh']),
                             input_shape=self.input_shape))

            for i in range(hp.Int('num_layers', 1, 3)):
                model.add(Conv2D(filters=hp.Choice('num_filters_%d' % (i + 1), values=[8, 16, 32, 64]),
                                 kernel_size=hp.Choice('kernel_size_%d' % (i + 1), values=[3, 4, 5]),
                                 activation=hp.Choice('activation_%d' % (i + 1), values=['relu', 'tanh'])))
            model.add(Flatten())
            model.add(Dense(N_zern))
            model.summary()

            model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])),
                          loss='mean_squared_error')
            return model


    input_shape = (pix, pix, 2,)
    hypermodel = CNNHyperModel(input_shape=input_shape, num_classes=N_zern)

    HYPERBAND_MAX_EPOCHS = 10
    MAX_TRIALS = 50
    EXECUTION_PER_TRIAL = 2
    tuner = Hyperband(
        hypermodel,
        max_epochs=HYPERBAND_MAX_EPOCHS,
        objective='val_loss',
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='hyperband',
        project_name='ultimate_robust'
    )

    tuner.search_space_summary()

    tuner.search(train_PSF, train_coef, epochs=10,
                 validation_data=(test_PSF, test_coef))

    tuner.results_summary()
    best_model = tuner.get_best_models(num_models=1)[0]

    # best HP
    best_hp = tuner.get_best_hyperparameters()[0]
    print(best_hp.values)

    guess_best = best_model.predict(test_PSF)
    residual_coef = test_coef - guess_best
    norm_res = np.mean(norm(residual_coef, axis=1))
    print(norm_res)

    calib_zern.cnn_model = best_model

    def corner_plot(variables, var_labels, values):

        N_var = variables.shape[0]
        N_row = N_var - 1
        N_col = N_row

        # values = values / np.max(values)
        colors = cm.Greens(values)

        fig, axes = plt.subplots(N_row, N_col)
        for i in range(N_row):
            for j in range(i + 1):
                ax = axes[i][j]
                x_data = variables[j]
                y_data = variables[i + 1]
                sc = ax.scatter(x_data, y_data, s=2, c=values, cmap=plt.cm.get_cmap('Greens'))
                sc.set_clim([np.min(values), np.max(values)])
                plt.colorbar(sc, ax=ax)
                ax.set_xlim([np.min(x_data), np.max(x_data)])
                ax.set_ylim([np.min(y_data), np.max(y_data)])
                if i == N_row - 1:
                    ax.set_xlabel(var_labels[j])
                elif i != N_row - 1:
                    ax.xaxis.set_visible(False)

                if j == 0:
                    ax.set_ylabel(var_labels[i + 1])
                elif j != 0:
                    ax.yaxis.set_visible(False)
        # Remove useless
        for i in range(N_row):
            for j in np.arange(i + 1, N_col):
                axes[i][j].remove()

        return

    strehls = realistic_PSF.calculate_strehl(_res)
    corner_plot(errors_array, var_labels=errors_labels, values=strehls)
    plt.show()

    # A new iteration
    new_PSF = realistic_PSF.update_PSF_images(residual_coef, errors_array)

    # Get the new predictions
    guess_coef2 = calib_zern.cnn_model.predict(new_PSF)
    residual_coef2 = residual_coef - guess_coef2
    norm_res2 = norm(residual_coef2, axis=1)
    print(np.mean(norm_res2))
    corner_plot(errors_array, var_labels=errors_labels, values=norm_res2)
    plt.show()

    # A new iteration
    new_PSF2 = realistic_PSF.update_PSF_images(residual_coef2, errors_array)
    guess_coef3 = calib_zern.cnn_model.predict(new_PSF2)
    residual_coef3 = residual_coef2 - guess_coef3
    norm_res3 = norm(residual_coef3, axis=1)
    print(np.mean(norm_res3))
    corner_plot(errors_array, var_labels=errors_labels, values=norm_res3)
    plt.show()

    bins = 50
    plt.figure()
    plt.hist(norm(test_coef, axis=1), bins=bins)
    plt.hist(norm_res, histtype='step', bins=bins)
    plt.hist(norm_res2, histtype='step', bins=bins)
    plt.hist(norm_res3, histtype='step', bins=bins)
    plt.show()

    N_errors = errors_array.shape[0]
    N_bins = 10
    list_effects = list(effects_dict.keys())
    fig, axes = plt.subplots(2, N_errors)
    for k in range(N_errors):

        ymin, ymax = np.min(strehls), np.max(strehls)

        effect = list_effects[k]
        keys = list(effects_dict[effect].keys())
        kmin, kmax = keys[1:]
        xmin, xmax = effects_dict[effect][kmin], effects_dict[effect][kmax]

        ax1 = axes[0][k]
        sc = ax1.scatter(errors_array[k], strehls, s=5, c=str0, cmap=plt.cm.get_cmap('Blues'))
        # sc.set_clim([np.min(str0), 1])
        # cb = plt.colorbar(sc, ax=ax1, orientation='horizontal')
        ax1.set_ylim([0.9 * ymin, 1.0])
        ax1.set_xlim([xmin, xmax])

        bins = np.linspace(xmin, xmax, N_bins)
        digitized = np.digitize(errors_array[k], bins)

        bin_means = [errors_array[k][digitized == i].mean() for i in range(1, len(bins))]
        mu_bins = np.array([strehls[digitized == i].mean() for i in range(1, len(bins))])
        q_05 = np.array([np.percentile(strehls[digitized == i], 5) for i in range(1, len(bins))])
        q_95 = np.array([np.percentile(strehls[digitized == i], 95) for i in range(1, len(bins))])
        std_bins = [strehls[digitized == i].std() for i in range(1, len(bins))]

        errorbar = np.zeros((2, N_bins - 1))
        errorbar[0] = mu_bins - q_05
        errorbar[1] = q_95 - mu_bins

        ax2 = axes[1][k]
        ax2.errorbar(x=bin_means, y=mu_bins, yerr=errorbar, fmt='o', capsize=3)
        ax2.set_ylim([0.9 * ymin, 1.0])
        ax2.set_xlim([xmin, xmax])

        axes[1][k].set_xlabel(effect)


    axes[0][0].set_ylabel(r'Strehl ratio')
    axes[1][0].set_ylabel(r'Strehl ratio')

    plt.show()

    norm_ratio = norm_res / np.max(norm_res)
    colors = cm.Reds(norm_ratio)
    for i in range(N_errors):
        for j in np.arange(i + 1, N_errors):
            plt.figure()
            plt.scatter(errors_array[i], errors_array[j], s=5, color=colors)
            plt.xlabel(r'Error #%d' % (i + 1))
            plt.ylabel(r'Error #%d' % (j + 1))
    plt.show()

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

    colors = cm.Blues(np.linspace(0.5, 1, N_roc + 1))
    plt.figure()
    for k in range(N_roc + 1):
        strehl_data = strehl_roc[k]
        auc = np.mean(strehl_data)
        plt.plot(threshold, strehl_data, color=colors[k], label="%d (%.3f)" % (k, auc))
    # plt.xscale('log')
    plt.xlim([0, 1])
    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r'Strehl ratio [ ]')
    plt.ylabel(r'Cumulative')
    plt.legend(title='Iteration (AUC)')
    # plt.title(r'Strehl ROC | Architecture: [Conv1, Conv2, FC]')
    plt.title(r'Strehl ROC | Robust Model')
    plt.show()

    # Now Test outside the range

    # Specify which realistic effects we want to include and their range
    effects_dict_range = {"NYQUIST_ERRORS": {"ON": True, "MIN_ERR": -0.10, "MAX_ERR": -0.05},
                    "DEFOCUS_ERRORS": {"ON": True, "MIN_ERR": -0.10, "MAX_ERR": -0.05},
                    "ANAMORPHIC_ERRORS": {"ON": True, "MIN_ERR": -0.10, "MAX_ERR": -0.05},
                    "FLAT_FIELD": {"ON": True, "MIN_ERR": -0.10, "MAX_ERR": -0.05},
                    "READOUT_NOISE": {"ON": True, "MIN_SNR": 250, "MAX_SNR": 500}}

    outrange_PSF = RealisticPSF(psf_model_param=psf_model_param, effects_dict=effects_dict_range)
    N_test = 2500
    _t, _c, test_PSF_range, test_coef_range, errors_range = outrange_PSF.generate_dataset(0, N_test, coef_strength, rescale)
    errors_range = np.array(errors_range)

    strehl_roc_range, threshold_range, _res_range = outrange_PSF.calculate_cum_strehl(calib_zern, test_PSF_range, test_coef_range,
                                                                                errors_range, N_roc)
    strehls_range = outrange_PSF.calculate_strehl(_res_range)

    guess_coef_range = calib_zern.cnn_model.predict(test_PSF_range)
    residual_coef_range = test_coef_range - guess_coef_range
    norm_res_range = norm(residual_coef_range, axis=1)
    print(np.mean(norm_res_range))

    str0 = realistic_PSF.calculate_strehl(test_coef_range)


    colors = cm.Blues(np.linspace(0.5, 1, N_roc + 1))
    plt.figure()
    for k in range(N_roc + 1):
        strehl_data = strehl_roc_range[k]
        auc = np.mean(strehl_data)
        plt.plot(threshold_range, strehl_data, color=colors[k], label="%d (%.3f)" % (k, auc))
    # plt.xscale('log')
    plt.xlim([0, 1])
    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r'Strehl ratio [ ]')
    plt.ylabel(r'Cumulative')
    plt.legend(title='Iteration (AUC)')
    # plt.title(r'Strehl ROC | Architecture: [Conv1, Conv2, FC]')
    plt.title(r'Strehl ROC | Robust Model')
    plt.show()

    N_errors = errors_range.shape[0]
    N_bins = 20
    list_effects = list(effects_dict_range.keys())
    fig, axes = plt.subplots(2, N_errors)
    for k in range(N_errors):

        ymin, ymax = np.min(strehls_range), np.max(strehls_range)
        ymin = 0.8/0.9

        effect = list_effects[k]
        keys = list(effects_dict_range[effect].keys())
        kmin, kmax = keys[1:]
        xmin, xmax = effects_dict_range[effect][kmin], effects_dict_range[effect][kmax]
        xmin_train, xmax_train = effects_dict[effect][kmin], effects_dict[effect][kmax]

        ax1 = axes[0][k]
        ax1.fill_between(x=[xmin_train, xmax_train], y1=0.9 * ymin, y2=1.0, color='lightgreen', alpha=0.5)
        # sc = ax1.scatter(errors_range[k], strehls_range, s=5, c='red', cmap=plt.cm.get_cmap('Blues'))
        sc = ax1.scatter(errors_range[k], strehls_range, s=5, color='red')
        # sc.set_clim([np.min(str0), 1])
        # cb = plt.colorbar(sc, ax=ax1, orientation='horizontal')
        ax1.set_ylim([0.9 * ymin, 1.0])
        ax1.set_xlim([xmin, xmax])

        bins = np.linspace(xmin, xmax, N_bins)
        digitized = np.digitize(errors_range[k], bins)

        bin_means = [errors_range[k][digitized == i].mean() for i in range(1, len(bins))]
        mu_bins = np.array([strehls_range[digitized == i].mean() for i in range(1, len(bins))])
        q_05 = np.array([np.percentile(strehls_range[digitized == i], 5) for i in range(1, len(bins))])
        q_95 = np.array([np.percentile(strehls_range[digitized == i], 95) for i in range(1, len(bins))])
        std_bins = [strehls_range[digitized == i].std() for i in range(1, len(bins))]

        errorbar = np.zeros((2, N_bins - 1))
        errorbar[0] = mu_bins - q_05
        errorbar[1] = q_95 - mu_bins

        ax2 = axes[1][k]
        ax2.fill_between(x=[xmin_train, xmax_train], y1=0.9 * ymin, y2=1.0, color='lightgreen', alpha=0.5)
        ax2.errorbar(x=bin_means, y=mu_bins, yerr=errorbar, fmt='o', capsize=3)
        ax2.set_ylim([0.9 * ymin, 1.0])
        ax2.set_xlim([xmin, xmax])

        axes[1][k].set_xlabel(effect)

    axes[0][0].set_ylabel(r'Strehl ratio')
    axes[1][0].set_ylabel(r'Strehl ratio')

    plt.show()


    # ratio between nominal and defocus peaks, versus the defocus diversity errora

    # Populate the dictionary with PSF model parameters
    N_divs = 150
    diversity_errors = np.linspace(-0.20, 0.20, N_divs)
    coef = test_coef[N_test // 2: N_test // 2 + 1]
    c_guesses = np.zeros((N_divs, N_zern))
    maxs = np.zeros(N_divs)
    for k in range(N_divs):
        new_diversity = diversity * (1 + diversity_errors[k])
        psf_model_param = {"N_PIX": N_PIX, "pix": pix, "SPAX": SPAX, "WAVE": WAVE, "N_LEVELS": N_levels,
                           "DIVERSITY": new_diversity}

        diversity_PSF = RealisticPSF(psf_model_param=psf_model_param, effects_dict=effects_dict)
        PSF_div = diversity_PSF.nominal_PSF_model
        # print(PSF_div.no)
        p = diversity_PSF.update_PSF_images(coef, errors_array)
        c_guess = calib_zern.cnn_model.predict(p)
        c_guesses[k] = c_guess[0]

        # maxs[k] = np.max(p[0, :, :, 1])
        # # print(np.max(p))
        # plt.figure()
        # plt.imshow(p[0, :, :, 1])
        # plt.clim(0, 0.5)
        # plt.colorbar()
    # plt.show()

    N_row, N_col = 4, 8
    fig, axes = plt.subplots(N_row, N_col)
    for i in range(N_row):
        for j in range(N_col):
            ax = axes[i][j]
            k_aberr = i * N_col + j
            sc = ax.scatter(diversity_errors, c_guesses[:, k_aberr], s=6)
            ax.axhline(coef[0, k_aberr], color='k', linestyle='-.')
            ax.yaxis.set_visible(False)
            if i != N_row - 1:
                ax.xaxis.set_visible(False)

    plt.show()


    plt.figure()
    plt.
    plt.axhline(coef[0, k_aberr], color='k', linestyle='-.')
    plt.show()

