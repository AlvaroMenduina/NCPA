import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from time import time

from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.models import Sequential
from keras import backend as K
from numpy.linalg import norm as norm

import noise


def create_cnn_model(layer_filers, kernel_size, input_shape, N_classes, name, activation='relu'):
    """
    Creates a CNN model for NCPA calibration
    :return:
    """
    # input_shape = (pix, pix, 2,)        # Multiple Wavelength Channels
    model = Sequential()
    model.name = name
    model.add(Conv2D(layer_filers[0], kernel_size=(kernel_size, kernel_size), strides=(1, 1),
                     activation=activation, input_shape=input_shape))
    for N_filters in layer_filers[1:]:
        model.add(Conv2D(N_filters, (kernel_size, kernel_size), activation=activation))

    model.add(Flatten())
    model.add(Dense(N_classes))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def generate_dataset_and_save(PSF_model, N_train, N_test, coef_strength, rescale, dir_to_save):


    # Check whether the directory already exists
    if not os.path.exists(dir_to_save):
        print("Creating New Directory")
        os.makedirs(dir_to_save)
    else:
        print("Directory Already Exists")

    train_PSF, train_coef, test_PSF, test_coef = generate_dataset(PSF_model, N_train, N_test, coef_strength, rescale)

    print("Saving Datasets in: ")
    for dataset, file_name in zip([train_PSF, train_coef, test_PSF, test_coef],
                                  ["train_PSF", "train_coef", "test_PSF", "test_coef"]):

        name_str = os.path.join(dir_to_save, file_name)
        print(name_str + '.npy')
        np.save(name_str, dataset)

    return train_PSF, train_coef, test_PSF, test_coef

def load_datasets(dir_to_load, file_names):

    datasets = []
    for file in file_names:

        data = np.load(os.path.join(dir_to_load, file + '.npy'))
        datasets.append(data)

    return datasets


def generate_dataset(PSF_model, N_train, N_test, coef_strength, rescale=0.35):
    """

    :param PSF_model:
    :param N_train:
    :param N_test:
    :param coef_strength:
    :param rescale:
    :return:
    """

    # Check whether PSF_model is single wavelength or multiwave
    try:
        N_waves = PSF_model.N_waves
        print("Multiwavelength Model | N_WAVES: %d" % N_waves)
    except AttributeError:
        N_waves = 1

    N_coef = PSF_model.N_coef
    pix = PSF_model.crop_pix
    N_samples = N_train + N_test

    dataset = np.empty((N_samples, pix, pix, 2 * N_waves))
    coefs = coef_strength * np.random.uniform(low=-1, high=1, size=(N_samples, N_coef))

    # Rescale the coefficients to cover a wider range of RMS (so we can iterate)
    rescale_train = np.linspace(1.0, rescale, N_train)
    rescale_test = np.linspace(1.0, 0.5, N_test)
    rescale_coef = np.concatenate([rescale_train, rescale_test])
    coefs *= rescale_coef[:, np.newaxis]

    print("\nGenerating datasets: %d PSF images" % N_samples)
    if N_waves == 1:
        for i in range(N_samples):

            im0, _s = PSF_model.compute_PSF(coefs[i])
            dataset[i, :, :, 0] = im0

            im_foc, _s = PSF_model.compute_PSF(coefs[i], diversity=True)
            dataset[i, :, :, 1] = im_foc

            if i % 500 == 0:
                print(i)

    else:   # MULTIWAVELENGTH    # This can take a long time
        start = time()
        for i in range(N_samples):
            for wave_idx in range(N_waves):
                im0, _s = PSF_model.compute_PSF(coefs[i], wave_idx=wave_idx)
                dataset[i, :, :, 2*wave_idx] = im0

                im_foc, _s = PSF_model.compute_PSF(coefs[i], wave_idx=wave_idx, diversity=True)
                dataset[i, :, :, 2*wave_idx + 1] = im_foc

            if i % 100 == 0 and i > 0:
                delta_time = time() - start
                estimated_total = delta_time * N_samples / i
                remaining = estimated_total - delta_time
                message = "%d | t=%.1f sec | ETA: %.1f sec (%.1f min)" % (i, delta_time, remaining, remaining / 60)
                print(message)

    return dataset[:N_train], coefs[:N_train], dataset[N_train:], coefs[N_train:]

def robust_diversity(PSF_model, N_train, N_test, coef_strength, rescale=0.35, diversity_error=0.10, N_copies=5):
    """
    Generates datasets to train a model to be robust against Diversity uncertainties

    We cover an interval of +- X % of diversity errors
    For each copy we generate a training set with a different diversity wavefront
    That way we hope to train the model to be robust against uncertainties

    :param PSF_model:
    :param N_train:
    :param N_test:
    :param coef_strength:
    :param rescale:
    :param diversity_error:
    :return:
    """

    diversity_copy = PSF_model.diversity_phase.copy()

    train_PSF, train_coef, test_PSF, test_coef = [], [], [], []
    diversities = np.linspace(-diversity_error, diversity_error, N_copies, endpoint=True)
    print("\nRobust Training || Diversity Errors")
    print("Generating %d copies of the datasets" % N_copies)
    print("Ranging from -%.1f to +%.1f per cent Diversity Error" % (100*diversity_error, 100 * diversity_error))
    for i, div in enumerate(diversities):
        print("Copy #%d/%d" % (i+1, N_copies))

        PSF_model.diversity_phase = (1 + div) * diversity_copy          # modify the diversity
        _train_PSF, _train_coef, _test_PSF, _test_coef = generate_dataset(PSF_model, N_train, N_test,
                                                                          coef_strength, rescale)
        train_PSF.append(_train_PSF)
        train_coef.append(_train_coef)
        test_PSF.append(_test_PSF)
        test_coef.append(_test_coef)

    train_PSF = np.concatenate(train_PSF, axis=0)
    train_coef = np.concatenate(train_coef, axis=0)
    test_PSF = np.concatenate(test_PSF, axis=0)
    test_coef = np.concatenate(test_coef, axis=0)

    # Watch out, we need to go back to the previous diversity
    PSF_model.diversity_phase = diversity_copy

    return train_PSF, train_coef, test_PSF, test_coef


class Calibration(object):

    def __init__(self, PSF_model, noise_effects=noise.NoiseEffects()):

        self.PSF_model = PSF_model
        self.noise_effects = noise_effects

    def generate_dataset_batches(self, N_batches, N_train, N_test, coef_strength, rescale=0.35):
        """
        Generate the [clean] datasets necessary to train a calibration model
        We split the training in N_batches of size N_train PSF images

        The function generates a total of (N_batches * N_train + N_test) PSF images with random aberrations


        :param N_batches: How many batches of size N_train to create
        :param N_train: Number of PSF images within a batch
        :param N_test: Number of PSF images to test the performance
        :param coef_strength: Strength of the aberration coefficients, used to modulate the RMS
        :param rescale: Rescale the Training Coefficients between [1] and [rescale] to cover a wider range of RMS
        :return:
        """

        N_coef = self.PSF_model.N_coef
        pix = self.PSF_model.crop_pix
        N_samples = N_batches * N_train + N_test

        dataset = np.empty((N_samples, pix, pix, 2))
        coefs = coef_strength * np.random.uniform(low=-1, high=1, size=(N_samples, N_coef))

        # Rescale the coefficients to cover a wider range of RMS (so we can iterate)
        rescale_train = np.linspace(1.0, rescale, N_batches * N_train)
        rescale_test = np.linspace(1.0, 0.5, N_test)
        rescale_coef = np.concatenate([rescale_train, rescale_test])
        coefs *= rescale_coef[:, np.newaxis]

        # Loop over the samples, generating nominal and defocus images
        print("\nGenerating %d PSF images" % N_samples)
        for i in range(N_samples):
            if i % 500 == 0:
                print(i)

            im0, _s = self.PSF_model.compute_PSF(coefs[i])
            dataset[i, :, :, 0] = im0

            im_foc, _s = self.PSF_model.compute_PSF(coefs[i], diversity=True)
            dataset[i, :, :, 1] = im_foc

        # Separate the batches
        train_batches = [dataset[i * N_train:(i + 1) * N_train] for i in range(N_batches)]
        coef_batches = [coefs[i * N_train:(i + 1) * N_train] for i in range(N_batches)]
        test_images, test_coefs = dataset[N_batches * N_train:], coefs[N_batches * N_train:]
        print("Splitting into %d batches of %d images" % (N_batches, N_train))
        print("Finished")
        return train_batches, coef_batches, test_images, test_coefs

    def update_PSF(self, coefs):
        """
        Updates the PSF images after calibration. We use this to run the calibration
        for several iterations until it converges
        :param coefs: residual coefficients
        :return:
        """

        print("\nUpdating the PSF images")
        N_channels = 2                          # [Nominal, Defocused]
        pix = self.PSF_model.crop_pix
        N_samples = coefs.shape[0]
        dataset = np.zeros((N_samples, pix, pix, N_channels))
        for i in range(N_samples):
            if i % 500 == 0:
                print(i)

            im0, _s = self.PSF_model.compute_PSF(coefs[i])                      # Nominal PSF
            dataset[i, :, :, 0] = im0

            im_foc, _s = self.PSF_model.compute_PSF(coefs[i], diversity=True)   # Defocused PSF
            dataset[i, :, :, 1] = im_foc
        print("Updated")
        return dataset

    def update_PSF_multiwave(self, coefs):
        """
        Multiwave version of Update_PSF
        :param coefs:
        :return:
        """

        print("\nUpdating the PSF images | Multiwave")
        N_waves = self.PSF_model.N_waves
        pix = self.PSF_model.crop_pix
        N_samples = coefs.shape[0]
        dataset = np.zeros((N_samples, pix, pix, 2*N_waves))

        start = time()
        for i in range(N_samples):                  # Loop over the PSF images
            for wave_idx in range(N_waves):         # Loop over the Wavelength Channels [Nom, Foc, Nom, Foc...]
                im0, _s = self.PSF_model.compute_PSF(coefs[i], wave_idx=wave_idx)
                dataset[i, :, :, 2*wave_idx] = im0

                im_foc, _s = self.PSF_model.compute_PSF(coefs[i], wave_idx=wave_idx, diversity=True)
                dataset[i, :, :, 2*wave_idx + 1] = im_foc

            if i % 100 == 0 and i > 0:
                delta_time = time() - start
                estimated_total = delta_time * N_samples / i
                remaining = estimated_total - delta_time
                message = "%d | t=%.1f sec | ETA: %.1f sec (%.1f min)" % (i, delta_time, remaining, remaining / 60)
                print(message)
        print("Updated")
        return dataset

    def create_cnn_model(self, layer_filters, kernel_size, name, activation='relu', dropout=None):
        """
        Creates a CNN model for NCPA calibration

        It accepts both [Single Wavelength] and [Multiwavelength] approaches
        if [Single Wavelength] the input datacube is [:, pix, pix, 2] Nominal and Defocus PSF images
        if [Multiwavelength] the input datacube is [:, pix, pix, 2 x N_waves] Nom+Defoc for each wavelength channel

        :param layer_filters: list containing number of CNN filters per layer
        :param kernel_size: list of CNN layer kernel sizes
        :param name: (str) name of the CNN model
        :param activation: (str) type of activation function (default: ReLu)
        :param dropout: dropout rate. If NOT None, then all layers have dropout
        :return:
        """

        # Check whether PSF_model is single wavelength or multiwave
        try:
            N_waves = self.PSF_model.N_waves
            print("Multiwavelength Model | N_WAVES: %d" % N_waves)
        except AttributeError:
            N_waves = 1

        pix = self.PSF_model.crop_pix
        input_shape = (pix, pix, 2 * N_waves,)        # Multiple Wavelength Channels
        model = Sequential()
        model.name = name
        model.add(Conv2D(layer_filters[0], kernel_size=(kernel_size, kernel_size), strides=(1, 1),
                         activation=activation, input_shape=input_shape))

        if dropout is not None:
            model.add(Dropout(rate=dropout))
            for N_filters in layer_filters[1:]:
                model.add(Conv2D(N_filters, (kernel_size, kernel_size), activation=activation))
                model.add(Dropout(rate=dropout))
            model.add(Flatten())
            model.add(Dropout(rate=dropout))
        else:
            for N_filters in layer_filters[1:]:
                model.add(Conv2D(N_filters, (kernel_size, kernel_size), activation=activation))
            model.add(Flatten())

        model.add(Dense(self.PSF_model.N_coef))         # N_classes is the number of NCPA coefficients of the PSF model
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')

        self.cnn_model = model

        return


    def validation_loss(self, test_images, test_coefs):
        """
        Validation loss to see how the model performs
        We compare the norm of the residual coefficients to the initial norm
        :param test_images:
        :param test_coefs:
        :return:
        """
        guess_coef = self.cnn_model.predict(test_images)
        residual = test_coefs - guess_coef
        norm_coefs = norm(test_coefs, axis=1)
        norm_residual = norm(residual, axis=1)
        ratio = np.mean(norm_residual / norm_coefs) * 100
        return ratio

    def train_calibration_model(self, train_images, train_coefs, test_images, test_coefs,
                                N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                readout_noise=False, RMS_readout=[1. / 100], readout_copies=3):

        loss, val_loss = [], []
        print("\nTraining the Calibration Model")
        for i_times in range(N_loops):  # Loop over all images N_loops times
            print("\nIteration %d / %d" % (i_times + 1, N_loops))
            if readout_noise is True:
                i_noise = np.random.choice(range(len(RMS_readout)))  # Randomly select the RMS Readout
                train, coef = [], []
                for k in range(readout_copies):  # Get multiple random copies
                    _train_images = self.noise_effects.add_readout_noise(train_images, RMS_READ=RMS_readout[i_noise])
                    train.append(_train_images)
                    coef.append(train_coefs)
                train_noisy_images = np.concatenate(train, axis=0)
                train_noisy_coefs = np.concatenate(coef, axis=0)
                test_noisy_images = self.noise_effects.add_readout_noise(test_images, RMS_READ=RMS_readout[i_noise])
            else:
                train_noisy_images = train_images
                train_noisy_coefs = train_coefs
                test_noisy_images = test_images

            train_history = self.cnn_model.fit(x=train_noisy_images, y=train_noisy_coefs,
                                               validation_data=(test_noisy_images, test_coefs),
                                               epochs=epochs_loop, batch_size=batch_size_keras,
                                               shuffle=True, verbose=verbose)

            loss.extend(train_history.history['loss'])
            val_loss.append(self.validation_loss(test_noisy_images, test_coefs))

        if plot_val_loss:
            plt.figure()
            plt.plot(val_loss)
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')

        return loss, val_loss


    def train_calibration_model_batches(self, images_batches, coefs_batches, test_images, test_coefs,
                                N_loops=10, epochs_loop=50, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                readout_noise=False, RMS_readout=[1. / 100], readout_copies=3):

        """
        Train a CNN calibration model to estimate the aberrations from PSF images

        We use batches so that we can add random instances of noise effects

        In other words, if N_loops = 5, N_batches = 10, N_train = 1000,
        we loop 5 times over a training set of 10,000 images, in 10 batches of 1,000.
        This is very helpful as it allow us to add NOISE EFFECTS multiple times (random instances)
        and make the calibration more robust to such effects

        NOISE_EFFECTS | We can add:
            - READOUT NOISE
        for each case of noise, we can cover a certain range, by randomly selecting from the list

        :param images_batches: list of the form [train_images1, train_images2, ...] ("clean")
        :param coefs_batches: list of the form [train_coefs1, train_coefs2, ...] ("clean")
        :param test_images: datacube of test PSF images ("clean")
        :param test_coefs: datacube of test coefficients
        :param N_loops: Number of loops to run over all batches of images
        :param epochs_loop: Number of epochs per loop to train the model
        :param verbose: Keras verbose option: 0 silent, 1 display stuff
        :param plot_val_loss: whether to plot the validation loss evolution at the end
        :param readout_noise: whether to add READOUT NOISE
        :param RMS_readout: how much READOUT NOISE to add, list of RMS [1/SNR1, 1/ SNR2]
        :return:
        """

        N_batches = len(images_batches)
        N_train = images_batches[0].shape[0]
        loss, val_loss = [], []
        print("\nTraining the Calibration Model")
        for i_times in range(N_loops):  # Loop over all batches N_loops times
            for k_batch in range(N_batches):  # Loop over each training batch
                print("\nIteration %d || Batch #%d (%d samples)" % (i_times + 1, k_batch + 1, N_train))

                train_images = images_batches[k_batch]
                train_coefs = coefs_batches[k_batch]

                if readout_noise is True:
                    i_noise = np.random.choice(range(len(RMS_readout)))  # Randomly select the RMS Readout
                    train, coef = [], []
                    for k in range(readout_copies):     # Get multiple random copies
                        _train_images = self.noise_effects.add_readout_noise(train_images, RMS_READ=RMS_readout[i_noise])
                        train.append(_train_images)
                        coef.append(train_coefs)
                    train_images = np.concatenate(train, axis=0)
                    train_coefs = np.concatenate(coef, axis=0)
                    test_images = self.noise_effects.add_readout_noise(test_images, RMS_READ=RMS_readout[i_noise])

                # Allergen Notice: At this point clean_images may contain noise
                train_history = self.cnn_model.fit(x=train_images, y=train_coefs,
                                                   validation_data=(test_images, test_coefs),
                                                   epochs=epochs_loop, batch_size=batch_size_keras,
                                                   shuffle=True, verbose=verbose)
                loss.extend(train_history.history['loss'])
                val_loss.append(self.validation_loss(test_images, test_coefs))

        if plot_val_loss:
            plt.figure()
            plt.plot(val_loss)
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')

        return loss, val_loss

    def calculate_RMS(self, coef_before, coef_after, wavelength):
        """
        Using the PSF model, calculate the RMS wavefront BEFORE and AFTER corrections
        We use this to loop over the calibration
        :param coef_before:
        :param coef_after:
        :param wavelength: working wavelength [microns]
        :return:
        """
        if self.PSF_model.N_waves == 1:
            model_matrix_flat = self.PSF_model.model_matrix_flat
        else:
            i_wave = np.argwhere(self.PSF_model.wavelengths == wavelength)[0][0]
            print(i_wave)
            model_matrix_flat = self.PSF_model.model_matrices_flat[i_wave]

        N_samples = coef_before.shape[0]
        print("\nCalculating RMS before / after for %d samples" % N_samples)
        RMS0, RMS = [], []
        for k in range(N_samples):
            wavef_before = wavelength * 1e3 * np.dot(model_matrix_flat, coef_before[k])
            wavef_after = wavelength * 1e3 * np.dot(model_matrix_flat, coef_after[k])
            RMS0.append(np.std(wavef_before))
            RMS.append(np.std(wavef_after))
        mu0, mu = np.mean(RMS0), np.mean(RMS)
        med0, med = np.median(RMS0), np.median(RMS)
        std0, std = np.std(RMS0), np.std(RMS)
        print("RMS Before: %.1f +- %.1f nm (%.1f median)" % (mu0, std0, med0))
        print("RMS  After: %.1f +- %.1f nm (%.1f median)" % (mu, std, med))
        return RMS0, RMS

    def calibrate_iterations(self, test_images, test_coefs, wavelength, N_iter=3,
                             readout_noise=False, RMS_readout=1./100, dropout=False, N_samples_drop=None):

        """
        Run the calibration for several iterations
        :param test_images: datacube of test PSF images ("clean")
        :param test_coefs: datacube of test coefficients ("clean")
        :param N_iter: how many iterations to run
        :param readout_noise: whether to add READOUT NOISE
        :param RMS_readout: how much READOUT NOISE to add, RMS 1/SNR
        :param dropout: whether to use a CNN with dropout
        :param N_samples_drop: number of times to sample the posterior [if dropout is True]
        :return:
        """
        if readout_noise is True:
            images_before = self.noise_effects.add_readout_noise(test_images, RMS_READ=RMS_readout)
        else:
            images_before = test_images

        coefs_before = test_coefs
        RMS_evolution = []
        for k in range(N_iter):
            print("\nNCPA Calibration | Iteration %d/%d" % (k + 1, N_iter))

            if dropout is False:
                predicted_coefs = self.cnn_model.predict(images_before)
            else:
                _pred, mean_pred, _uncert = self.predict_with_uncertainty(images_before, N_samples_drop)
                predicted_coefs = mean_pred

            coefs_after = coefs_before - predicted_coefs
            rms_before, rms_after = self.calculate_RMS(coefs_before, coefs_after, wavelength)
            rms_pair = [rms_before, rms_after]
            RMS_evolution.append(rms_pair)

            if k == N_iter - 1:
                break
            # Update the PSF and coefs
            if self.PSF_model.N_waves == 1:
                images_before = self.update_PSF(coefs_after)
            else:
                images_before = self.update_PSF_multiwave(coefs_after)
            coefs_before = coefs_after

            if readout_noise is True:
                images_before = self.noise_effects.add_readout_noise(images_before, RMS_READ=RMS_readout)

        final_residuals = coefs_after

        return RMS_evolution, final_residuals

    def predict_with_uncertainty(self, test_images, N_samples):
        """
        Makes use of the fact that our model has Dropout to sample from
        the posterior of predictions to get an estimate of the uncertainty of the predictions
        :param dropout_model: a CNN model with Dropout
        :param test_images: the test images to be used
        :param N_samples: number of times to sample the posterior
        :return:
        """

        print("\nUsing Dropout model to predict with Uncertainty")
        print("%d Samples of the posterior" % N_samples)
        dropout_model = self.cnn_model
        # force_training_mode: a Keras function that forces the model to act on "training mode" because Keras
        # freezes the Dropout during testing
        force_training_mode = K.function([dropout_model.layers[0].input, K.learning_phase()],
                                         [dropout_model.layers[-1].output])
        N_classes = self.PSF_model.N_coef

        # results is [N_samples, N_PSF, N_coef] where axis=0 is the Posterior of the predictions
        result = np.zeros((N_samples,) + (test_images.shape[0], N_classes))

        for i in range(N_samples):
            result[i, :, :] = force_training_mode((test_images, 1))[0]
        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)

        return result, prediction, uncertainty

    @staticmethod
    def plot_RMS_evolution(RMS_evolution):
        """
        Plot the evolution of RMS wavefront with calibration iterations
        :param RMS_evolution: list of pairs of RMS [(BEFORE, AFTER)_0, ..., (BEFORE, AFTER)_N]
        """

        N_pairs = len(RMS_evolution)  # Pairs of [Before, After]
        blues = cm.Blues(np.linspace(0.5, 1.0, N_pairs))

        plt.figure()
        for i, rms_pair in enumerate(RMS_evolution):
            before, after = rms_pair[0], rms_pair[1]
            print(len(before), len(after))
            label = r"%d | $\mu$=%.1f $\pm$ %.1f nm" % (i + 1, np.mean(after), np.std(after))
            plt.scatter(before, after, s=4, color=blues[i], label=label)

        plt.xlim([0, 300])
        plt.ylim([0, 200])
        plt.grid(True)
        plt.legend(title='Iteration', loc=2)
        plt.xlabel(r'RMS wavefront BEFORE [nm]')
        plt.ylabel(r'RMS wavefront AFTER [nm]')


class CalibrationEnsemble(Calibration):

    """
    Class Inheritance for Calibration Object
    It allows us to train an Ensemble of Calibration models on the same dataset
    """

    def generate_ensemble_models(self, N_models, layer_filters, kernel_size, name, activation='relu', drop_out=None):
        """
        Create a list of CNN models for calibration
        :param N_models:
        :param layer_filters:
        :param kernel_size:
        :param name:
        :param activation:
        :return:
        """

        self.ensemble_models = []
        for k in range(N_models):
            new_name = name + '_%d' % (k + 1)
            self.create_cnn_model(layer_filters, kernel_size, new_name, activation, drop_out)
            self.ensemble_models.append(self.cnn_model)
        del self.cnn_model

    def train_ensemble_models(self, train_images, train_coefs, test_images, test_coefs,
                                N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                readout_noise=False, RMS_readout=[1. / 100], readout_copies=3):

        """
        Loop over the models in the list of Ensemble Models and train them
        """

        for _model in self.ensemble_models:
            self.cnn_model = _model
            print("\nTraining Ensemble Model | ", _model.name)
            _loss = self.train_calibration_model(train_images, train_coefs, test_images, test_coefs,
                                N_loops, epochs_loop, verbose, batch_size_keras, plot_val_loss,
                                readout_noise, RMS_readout, readout_copies)

    def average_predictions(self, images, how_many_models, dropout=False, N_samples_drop=None):
        """
        Use the models in the list of Ensemble Models to get multiple predictions for a given test set
        and average across those predictions
        :param images: test PSF images (already noisy)
        :param how_many_models: how many ensemble models to average across
        :param dropout: if True (models have Dropout) we predict with uncertainty, i.e. we sample the posterior
        :param N_samples_drop: if dropout is True, how many samples for the posterior
        :return:
        """

        _predictions = []
        for _model in self.ensemble_models[:how_many_models]:
            print("\nPredicting with model: ", _model.name)
            if dropout is False:
                _predictions.append(_model.predict(images))
            else:
                self.cnn_model = _model
                _pred, mean_pred, _uncert = self.predict_with_uncertainty(images, N_samples_drop)
                _predictions.append(mean_pred)

        _predictions = np.stack(_predictions)
        print(_predictions.shape)

        return np.mean(_predictions, axis=0)

    def calibrate_iterations_ensemble(self, how_many_models, test_images, test_coefs, wavelength, N_iter=3,
                                      readout_noise=False, RMS_readout=1./100, dropout=False, N_samples_drop=None):

        """
        Modified self.calibrate_iterations to account for the existence of multiple Ensemble Models
        :param: how_many_models: how many models from the ensemble list to use


        :param test_images: datacube of test PSF images ("clean")
        :param test_coefs: datacube of test coefficients ("clean")
        :param N_iter: how many iterations to run
        :param readout_noise: whether to add READOUT NOISE
        :param RMS_readout: how much READOUT NOISE to add, RMS 1/SNR
        :return:
        """
        if readout_noise is True:
            images_before = self.noise_effects.add_readout_noise(test_images, RMS_READ=RMS_readout)
        else:
            images_before = test_images

        coefs_before = test_coefs
        RMS_evolution = []
        for k in range(N_iter):
            print("\nNCPA Calibration | Iteration %d/%d" % (k + 1, N_iter))

            #### This is the bit we are overriding!
            predicted_coefs = self.average_predictions(images_before, how_many_models, dropout, N_samples_drop)
            ####

            coefs_after = coefs_before - predicted_coefs
            rms_before, rms_after = self.calculate_RMS(coefs_before, coefs_after, wavelength)
            rms_pair = [rms_before, rms_after]
            RMS_evolution.append(rms_pair)

            if k == N_iter - 1:
                break
            # Update the PSF and coefs
            images_before = self.update_PSF(coefs_after)
            coefs_before = coefs_after

            if readout_noise is True:
                images_before = self.noise_effects.add_readout_noise(images_before, RMS_READ=RMS_readout)

        final_residuals = coefs_after

        return RMS_evolution, final_residuals



class Zernike_fit(object):

    """
    Uses Least Squares to fit Wavefronts back and forth between two basis
    the Zernike polynomials and an Actuator Commands model
    """

    def __init__(self, PSF_zernike, PSF_actuator, wavelength, rho_aper):
        self.PSF_zernike = PSF_zernike
        self.PSF_actuator = PSF_actuator

        # Get the Model matrices
        self.H_zernike = self.PSF_zernike.model_matrix.copy()
        self.H_zernike_flat = self.PSF_zernike.model_matrix_flat.copy()
        self.H_actuator = self.PSF_actuator.model_matrix.copy()
        self.H_actuator_flat = self.PSF_actuator.model_matrix_flat.copy()

        self.pupil_mask = self.PSF_zernike.pupil_mask.copy()

        self.wavelength = wavelength
        self.rho_aper = rho_aper

    def plot_example(self, zern_coef, actu_coef, ground_truth='zernike', k=0, cmap='bwr'):

        if ground_truth == "zernike":
            true_phase = self.wavelength * 1e3 * np.dot(self.H_zernike, zern_coef.T)[:, :, k]
            fit_phase = self.wavelength * 1e3 * np.dot(self.H_actuator, actu_coef)[:, :, k]
            names = ['Zernike', 'Actuator']
            print(0)

        elif ground_truth == "actuator":
            true_phase = self.wavelength * 1e3 * np.dot(self.H_actuator, actu_coef.T)[:, :, k]
            fit_phase = self.wavelength * 1e3 * np.dot(self.H_zernike, zern_coef)[:, :, k]
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
        ax1.set_xlim([-self.rho_aper, self.rho_aper])
        ax1.set_ylim([-self.rho_aper, self.rho_aper])
        plt.colorbar(img1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(1, 3, 2)
        img2 = ax2.imshow(fit_phase, cmap=cmap, extent=[-1, 1, -1, 1])
        ax2.set_title('%s Fit Wavefront' % names[1])
        img2.set_clim(m, -m)
        ax2.set_xlim([-self.rho_aper, self.rho_aper])
        ax2.set_ylim([-self.rho_aper, self.rho_aper])
        plt.colorbar(img2, ax=ax2, orientation='horizontal')

        ax3 = plt.subplot(1, 3, 3)
        img3 = ax3.imshow(residual, cmap=cmap, extent=[-1, 1, -1, 1])
        ax3.set_title('Residual [$\sigma=%.1f$ nm]' % rms)
        img3.set_clim(m, -m)
        ax3.set_xlim([-self.rho_aper, self.rho_aper])
        ax3.set_ylim([-self.rho_aper, self.rho_aper])
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