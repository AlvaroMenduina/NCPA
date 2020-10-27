import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from time import time
import copy
import h5py

import keras
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Sequential, Input, Model, load_model
from keras import backend as K
from numpy.linalg import norm as norm

import noise
import zernike
import utils


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

    def update_PSF(self, coefs, downsample=False):
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

        if downsample:
            dataset = self.PSF_model.downsample_datacube(dataset)

        print("Updated")
        return dataset

    def update_PSF_multiwave(self, coefs, multiwave_slice=None, downsample=False):
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

        if downsample:
            dataset = self.PSF_model.downsample_datacube(dataset)

        if multiwave_slice is not None:
            a, b = multiwave_slice
            dataset = dataset[:, :, :, a:b]
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
        try:
            N_waves = self.PSF_model.N_waves
            print("Multiwavelength Model | N_WAVES: %d" % N_waves)
            i_wave = np.argwhere(self.PSF_model.wavelengths == wavelength)[0][0]
            print(i_wave)
            model_matrix_flat = self.PSF_model.model_matrices_flat[i_wave]
        except AttributeError:
            model_matrix_flat = self.PSF_model.model_matrix_flat

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
                             readout_noise=False, RMS_readout=1./100, dropout=False, N_samples_drop=None,
                             multiwave_slice=None, downsample=False):

        """
        Run the calibration for several iterations
        :param test_images: datacube of test PSF images ("clean")
        :param test_coefs: datacube of test coefficients ("clean")
        :param N_iter: how many iterations to run
        :param readout_noise: whether to add READOUT NOISE
        :param RMS_readout: how much READOUT NOISE to add, RMS 1/SNR
        :param dropout: whether to use a CNN with dropout
        :param N_samples_drop: number of times to sample the posterior [if dropout is True]
        :param multiwave_slice: (a, b) tuple: which wavelength channels to slice
        :param downsample: bool, whether to downsample to a 2x coarser scale after updating the PSF images
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

            try:
                N_waves = self.PSF_model.N_waves
                images_before = self.update_PSF_multiwave(coefs_after, multiwave_slice, downsample)
            except AttributeError:
                images_before = self.update_PSF(coefs_after, downsample)

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
    def plot_RMS_evolution(RMS_evolution, colormap=cm.Blues, title=None):
        """
        Plot the evolution of RMS wavefront with calibration iterations
        :param RMS_evolution: list of pairs of RMS [(BEFORE, AFTER)_0, ..., (BEFORE, AFTER)_N]
        """

        N_pairs = len(RMS_evolution)  # Pairs of [Before, After]
        colors = colormap(np.linspace(0.5, 1.0, N_pairs))

        plt.figure()
        for i, rms_pair in enumerate(RMS_evolution):
            before, after = rms_pair[0], rms_pair[1]
            print(len(before), len(after))
            label = r"%d | $\mu$=%.1f $\pm$ %.1f nm" % (i + 1, np.mean(after), np.std(after))
            plt.scatter(before, after, s=4, color=colors[i], label=label)

        plt.xlim([0, 300])
        plt.ylim([0, 200])
        plt.grid(True)
        plt.legend(title='Iteration', loc=2)
        plt.xlabel(r'RMS wavefront BEFORE [nm]')
        plt.ylabel(r'RMS wavefront AFTER [nm]')
        if title is not None:
            plt.title(title)


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
            self.create_cnn_model(layer_filters[k], kernel_size, new_name, activation, drop_out)
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

class CalibrationAutoencoder(object):

    def __init__(self, PSF_model, N_autoencoders, encoder_filters, decoder_filters,
                 kernel_size, name, activation='relu', loss='binary_crossentropy',
                 load_directory=None,
                 noise_effects=noise.NoiseEffects()):
        """
        Object for Autoencoder-based Calibration

        Rationale: if we want to calibrate many aberrrations (~100 Zernike polynomials) we might run into trouble
        the calibration networks start to struggle in that regime. One possible alternative is to split the
        calibration task among multiple networks, each trained on a subset of Zernike polynomials.
        For example, 2 networks: one trained to identify low order aberrations, one trained to identify high order
        In that situation, if we show those networks the whole PSF with all types of aberrations, we can have
        feature contamination issues: i.e., the network trained to recognize low orders gets confused by the
        presence of high order features in the PSF images.
        We solve that by using Autoencoders to "denoise" the images. One Autoencoder would look at the PSF with
        all kinds of aberrations (high and low) and "clean" the image by removing a particular set of features

                         _____ Autoencoder LOW ----> [Clean Image] 1  ----> || Calibration || ---> Low Order Coeff
                        |                            # No High Orders
                        |
        |PSF image| -----
                        |
                        |_____Autoencoder HIGH ----> [Clean Image] 2  ----> || Calibration || ---> High Order Coeff
                                                     # No Low Orders

        We can then use the clean images to calibrate the subset of aberrations, without worrying about contamination

        Parameters

        :param PSF_model: PSF model to generate the images
        :param N_autoencoders: How many Autoencoders we will use to split the calibration
        :param encoder_filters: list containing the N filters for the Encoder section of the Autoencoder
        :param decoder_filters: list containing the N filters for the Decoder section of the Autoencoder
        :param kernel_size: kernel size for filters
        :param name: (str) name used to identify the Autoencoders
        :param activation: (str) activation function. Default: ReLu
        :param loss: (str) loss function for the Autoencoders. Default: Binary Crossentropy
        :param load_directory: path to load the trained models from
        :param noise_effects: Noise object to add readout noise to the images
        """

        self.PSF_model = PSF_model
        self.noise_effects = noise_effects

        self.N_autoencoders = N_autoencoders
        self.autoencoder_models = []
        if load_directory is None:  # Create the necessary autoencoders
            for k in range(N_autoencoders):
                print("Creating Autoencoder")
                self.autoencoder_models.append(self.create_autoencoder_model(encoder_filters, decoder_filters,
                                                                             kernel_size, name + '_%d' % (k + 1),
                                                                             activation, loss))
        else:   # Load the pre-trained models
            for k in range(N_autoencoders):
                file_name = os.path.join(load_directory, name + '_%d.h5' % (k + 1))
                print("Loading Trained Model:", file_name)
                self.autoencoder_models.append(load_model(file_name))

        # Get the Encoder part of the Autoencoders
        self.get_encoders(encoder_filters)

    def compute_PSF(self, coefs):
        """
        Calculate the PSF images given a set of aberration coefficients
        We use it to generate the training set for the Autoencoder models
        :param coefs:
        :return:
        """

        try:        # Check whether PSF_model is single wavelength or multiwave
            N_waves = self.PSF_model.N_waves
            print("Multiwavelength Model | N_WAVES: %d" % N_waves)
        except AttributeError:
            N_waves = 1

        pix = self.PSF_model.crop_pix
        N_samples = coefs.shape[0]
        dataset = np.empty((N_samples, pix, pix, 2 * N_waves))

        print("\nGenerating datasets: %d PSF images" % N_samples)
        if N_waves == 1:
            for i in range(N_samples):

                im0, _s = self.PSF_model.compute_PSF(coefs[i])
                dataset[i, :, :, 0] = im0

                im_foc, _s = self.PSF_model.compute_PSF(coefs[i], diversity=True)
                dataset[i, :, :, 1] = im_foc

                if i % 500 == 0:
                    print(i)
        return dataset

    def slice_zernike_polynomials(self, max_level=20):
        """
        We need to split the task of estimating the aberrations between N_autoencoders
        We typically want to do that by Zernike radial levels, i.e. some autoencoders will do low orders
        and others higher orders. But at the same time, we want to keep an (approximately) constant
        number of coefficients that each autoencoder has to estimate.

        This is not straightforward because as we go through the Zernike pyramid we have more and more polynomials
        per row (or radial order). So we have to find a way to split the Zernike polynomials between the autoencoders

        For that purpose, we identify the particular radial orders that divide the total number of polynomials
        in N_autoencoder groups of approximately the same number, while ensuring we do not mix aberrations of
        other orders

        At the end we return a list of indices for the model matrix that we can use to slice the zernikes
        :return:
        """

        N_coef = self.PSF_model.N_coef
        print("\nDividing %d Zernike polynomials among %d autoencoders" % (N_coef, self.N_autoencoders))
        # Create a dictionary that contains the Total Number of Polynomials "up to a Zernike radial order"
        zernike_dict = zernike.triangular_numbers(N_levels=max_level)
        # List the TNP
        total_poly = np.array([zernike_dict[key] - 3 for key in np.arange(3, max_level + 1)])
        # max_level = np.argwhere(total_poly == N_coef)[0][0]
        # frac contains the fractions we need to divide an interval in N_autoencoder chunks
        # if N=2 [1/2]
        # if N=3 [1/3, 2/3]
        # if N=4 [1/4, 2/4, 3/4] and so on...
        frac = [x / self.N_autoencoders for x in np.arange(1, self.N_autoencoders)]
        # Find the indices of total_poly that (approximately) split the total number of polynomials by those fractions
        i_cut = [np.argmin(np.abs(total_poly - f * N_coef)) for f in frac]
        # in other words, i_cut are the particular Zernike radial orders that fulfill our condition
        slices = [0]
        for i in i_cut:     # Create a list of indices to slice the model matrix and zernike coefficients
            print("Cutting at Zernike levels: ", i)
            print("Total polynomials up to that level: ", total_poly[i])
            slices.append(total_poly[i])
        slices.append(N_coef)
        return slices

    def generate_datasets_autoencoder(self, N_train, N_test, coef_strength, rescale):
        """
        The training of the autoencoders requires pairs of "noisy" vs "clean" images
        For N_autoencoders we need (N_autoendocers + 1) datasets for training because
        the "noisy" is shared between all autoencoders, as that is the PSF with all types of aberrations

        Here we compute (N_autoendocers + 1) datasets by generating a set of random aberration
        coefficients, and splitting them according to the slices from self.slice_zernike_polynomials()

        :param N_train:
        :param N_test:
        :param coef_strength:
        :param rescale:
        :return:
        """

        N_coef = self.PSF_model.N_coef
        N_samples = N_train + N_test

        # Calculate the TOTAL dataset of coefficients (i.e. with ALL aberrations)
        coefs = coef_strength * np.random.uniform(low=-1, high=1, size=(N_samples, N_coef))
        # Rescale the coefficients to cover a wider range of RMS (so we can iterate)
        rescale_train = np.linspace(1.0, rescale, N_train)
        rescale_test = np.linspace(1.0, 0.5, N_test)
        rescale_coef = np.concatenate([rescale_train, rescale_test])
        coefs *= rescale_coef[:, np.newaxis]

        # "Noisy" dataset -> PSF images with ALL aberrations
        nominal_dataset = self.compute_PSF(coefs)
        PSF_AE = [nominal_dataset]
        all_coefs = [coefs]

        # How to split the aberrations across autoencoders?
        slices_zernike = self.slice_zernike_polynomials()
        print(slices_zernike)
        for _min, _max in utils.pairwise(slices_zernike):
            zeroed_coef = coefs.copy()
            sliced_coef = coefs[:, _min:_max]
            zeroed_coef[:, :_min] *= 0.0    # remove the other coefficients
            zeroed_coef[:, _max:] *= 0.0
            # print(zeroed_coef[0])
            # In order to reuse the same model to generate all datasets we simply mask [zero] the coefficients
            PSF_AE.append(self.compute_PSF(zeroed_coef))
            all_coefs.append(sliced_coef)
        return PSF_AE, all_coefs

    def create_autoencoder_model(self, encoder_filters, decoder_filters, kernel_size, name, activation='relu',
                                 loss='binary_crossentropy'):
        """
        Create a list of self.N_autoencoders
        :param encoder_filters: list containing the N filters for the Encoder section of the Autoencoder
        :param decoder_filters: list containing the N filters for the Decoder section of the Autoencoder
        :param kernel_size: kernel size for the filters
        :param name: (str) name used to identify the Autoencoders
        :param activation: (str) activation function. Default: ReLu
        :param loss: (str) loss function. Default: Binary Crossentropy
        :return:
        """

        try:        # Check whether PSF_model is single wavelength or multiwave
            N_waves = self.PSF_model.N_waves
            print("Multiwavelength Model | N_WAVES: %d" % N_waves)
        except AttributeError:
            N_waves = 1

        pix = self.PSF_model.crop_pix
        input_shape = (pix, pix, 2 * N_waves,)  # Multiple Wavelength Channels

        model = Sequential()
        model.name = name
        # Encoder Part
        model.add(Conv2D(encoder_filters[0], kernel_size=(kernel_size, kernel_size), strides=(1, 1),
                         activation=activation, input_shape=input_shape, padding='same'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        for N_filters in encoder_filters[1:]:
            model.add(Conv2D(N_filters, (kernel_size, kernel_size), activation=activation, padding='same'))
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(AveragePooling2D(pool_size=(2, 2)))

        # Decoder Part
        for N_filters in decoder_filters:
            model.add(Conv2D(N_filters, (kernel_size, kernel_size), activation=activation, padding='same'))
            model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(2 * N_waves, (kernel_size, kernel_size), activation='sigmoid', padding='same'))
        model.summary()
        model.compile(optimizer='adam', loss=loss)

        return model

    def train_autoencoder_models(self, datasets, N_train, N_test, N_loops=5 ,epochs_per_loop=100,
                                 readout_noise=False, readout_copies=2, RMS_readout=1./100,
                                 save_directory=None):

        nominal_dataset = datasets[0]
        self.metrics = []
        for _model, clean_dataset in zip(self.autoencoder_models, datasets[1:]):
            print("Training Autoencoder: ", _model.name)
            train_before = nominal_dataset[:N_train]
            test_before = nominal_dataset[N_train:]
            train_after = clean_dataset[:N_train]
            test_after = clean_dataset[N_train:]
            _metric = Metrics()
            self.metrics.append(_metric)

            for i_times in range(N_loops):  # Loop over all images N_loops times
                print("\nIteration %d / %d" % (i_times + 1, N_loops))
                if readout_noise is True:
                    _before, _after = [], []
                    for k in range(readout_copies):
                        _train_images = self.noise_effects.add_readout_noise(train_before, RMS_READ=RMS_readout)
                        _before.append(_train_images)
                        # _train_images = self.noise_effects.add_readout_noise(train_after, RMS_READ=RMS_readout)
                        _after.append(train_after)
                    train_before_noisy = np.concatenate(_before, axis=0)
                    train_after_noisy = np.concatenate(_after, axis=0)

                    test_before_noisy = self.noise_effects.add_readout_noise(test_before, RMS_READ=RMS_readout)
                    test_after_noisy = test_after
                else:
                    train_before_noisy = train_before
                    train_after_noisy = train_after
                    test_before_noisy = test_before
                    test_after_noisy = test_after

                _model.fit(train_before_noisy, train_after_noisy, epochs=epochs_per_loop, shuffle=True,
                          verbose=1, validation_data=(test_before_noisy, test_after_noisy), callbacks=[_metric])

            # Save the models after training
            if save_directory is not None:
                file_name = os.path.join(save_directory, _model.name + '.h5')
                print("Saving Trained Model: ", file_name)
                _model.save(file_name)

    def get_encoders(self, encoder_filters):
        """
        It is sometimes interesting to look at the Encoded images
        to understand how the Autoencoder compresses information

        For that purpose we can extract the Encoder part of the Autoencoder
        and generate a model that takes the noisy images as input and
        outputs the encoded arrays
        :param encoder_filters:
        :return:
        """

        try:  # Check whether PSF_model is single wavelength or multiwave
            N_waves = self.PSF_model.N_waves
            print("Multiwavelength Model | N_WAVES: %d" % N_waves)
        except AttributeError:
            N_waves = 1

        pix = self.PSF_model.crop_pix
        input_shape = (pix, pix, 2 * N_waves,)  # Multiple Wavelength Channels

        self.encoders = []
        for _model in self.autoencoder_models:
            print("\nExtracting the Encoder part of the Autoencoder")
            input_img = Input(shape=input_shape)
            _input = input_img
            for k in range(2 * len(encoder_filters)):  # 2x because of the Conv2D + Pooling2D
                encoded_layer = _model.layers[k]
                _output = encoded_layer(_input)
                _input = _output

            _encoder = Model(input_img, _input)
            _encoder.summary()
            self.encoders.append(_encoder)
        return

    def create_calibration_models(self, layer_filters, kernel_size, name, activation='relu',
                                  mode='Decoded', load_directory=None):
        """
        The Autoencoders are in charge of "denoising" the PSF images to avoid feature contamination
        but do not provide any estimation of the aberrations

        For the proper calibration, we need to generate Calibration Models and train them with the
        output of the Autoencoders, the "clean" images

        We can load pre-trained models to save time
        :param layer_filters:
        :param kernel_size:
        :param name:
        :param activation:
        :param load_directory: path where the pre-trained models are stored. Default: None leads to training new models
        :return:
        """


        slices_zernike = self.slice_zernike_polynomials()      # How to split the Model Matrix across autoencoders?
        self.calibration_models = []
        for i, (_min, _max) in enumerate(utils.pairwise(slices_zernike)):
            print("Creating calibration network")

            # Copy the PSF model
            _PSF_copy = copy.deepcopy(self.PSF_model)
            N_zern = _max - _min                # How many Zernikes is this Autoencoder in charge of?
            _PSF_copy.N_coef = N_zern
            _PSF_copy.model_matrix = _PSF_copy.model_matrix[:, :, _min:_max]        # Slice the Model Matrix
            _PSF_copy.model_matrix_flat = _PSF_copy.model_matrix_flat[:, _min:_max]      # Slice the Model Matrix flat
            print(_PSF_copy.model_matrix_flat.shape)

            if mode == 'Encoded':
                # Find out what is the output shape of the Encoder
                encoder_shape = self.encoders[0].layers[-1].output_shape
                _crop = encoder_shape[1]
                _channels = encoder_shape[-1]

                crop_copy = _PSF_copy.crop_pix
                # Force the shapes for the input of the Calibration Models
                _PSF_copy.crop_pix = _crop
                _PSF_copy.N_waves = _channels // 2

                _calib = Calibration(PSF_model=_PSF_copy)

                if load_directory is None:  # Create the Models

                    _calib.create_cnn_model(layer_filters, kernel_size, name=name + '_%d' % (i + 1),
                                            activation=activation)
                else:  # Load the pre-trained models
                    file_name = os.path.join(load_directory, name + '_%d.h5' % (i + 1))
                    print("Loading Trained Model:", file_name)
                    _calib.cnn_model = load_model(file_name)

                # Restore the values to properly compute the PSF
                _PSF_copy.crop_pix = crop_copy
                _PSF_copy.N_waves = 1

            elif mode == 'Decoded': # We work with the same dimensions. Do nothing

                _calib = Calibration(PSF_model=_PSF_copy)

                if load_directory is None:  # Create the Models

                    _calib.create_cnn_model(layer_filters, kernel_size, name=name + '_%d' % (i + 1),
                                            activation=activation)
                else:  # Load the pre-trained models
                    file_name = os.path.join(load_directory, name + '_%d.h5' % (i + 1))
                    print("Loading Trained Model:", file_name)
                    _calib.cnn_model = load_model(file_name)

            self.calibration_models.append(_calib)

        return

    def clean_datasets_for_training(self, images, coefs, copies, RMS_readout,
                                    mode='Decoded', show_images=False):
        """
        Using the PSF images, generate the datasets necessary to train the calibration models
        If mode == 'Decoded'
            After adding readout noise to the nominal images, we 'predict' with the Autoencoder how the images
            should look like after cleaning the feature contamination
        if mode == 'Encoded'
            After adding readout noise, we 'encode' the images with the Encoder

        :param images: list of (N_autoencoders + 1) with the PSF images
        :param coefs: list of len (N_autoencoders + 1) with the coefficients for the PSF images and their decoded equivalent
        :param copies: how many copies with Readout Noise to add
        :param RMS_readout: RMS (1 / SNR) to use for the copies
        :param mode: "Decoded" we use the whole Autoencoder or "Encoded" we use only the Encoder part
        :param show_images: bool whether to show some examples of how the Autoencoder performs
        :return:
        """

        print("\nGenerating Datasets to train the Calibration Models")

        # Get the nominal PSF datacube [N_samples, pix, pix, 2] the PSF images with ALL aberrations
        nominal_imgs = images[0]
        truth = images[1:]              # What the Autoencoder should recover if it were perfect
        nominal_coefs = coefs[0]

        images_copies, coefs_copies = [], []

        # Add copies of the Nominal
        _nom_img, _nom_coef = [], []
        for n in range(copies):
            # noisy = self.noise_effects.add_readout_noise(nominal_imgs, RMS_READ=RMS_readout)
            noisy = nominal_imgs
            _nom_img.append(noisy)
            _nom_coef.append(nominal_coefs)
        images_copies.append(np.concatenate(_nom_img, axis=0))
        coefs_copies.append(np.concatenate(_nom_coef, axis=0))

        # Add copies of the Clean
        for k in range(self.N_autoencoders):        # Loop over the Autoencoders

            _predictions, _coef = [], []
            for i in range(copies):                 # Add multiple copies of Noisy Input, Coefficients, Clean Output
                noisy = self.noise_effects.add_readout_noise(nominal_imgs, RMS_READ=RMS_readout)

                if mode == 'Decoded':               # We work the whole Autoencoder
                    print("Cleaning the PSF images with Autoencoder")
                    clean = self.autoencoder_models[k].predict(noisy)
                    print(clean.shape)
                elif mode == 'Encoded':             # We use only the Encoder part
                    print("Encoding the PSF with Encoder")
                    clean = self.encoders[k].predict(noisy)
                    print(clean.shape)

                _predictions.append(clean)
                _coef.append(coefs[k + 1])

            images_copies.append(np.concatenate(_predictions, axis=0))
            coefs_copies.append(np.concatenate(_coef, axis=0))

            if show_images:

                list_images = [noisy, truth[k], images_copies[k + 1]]
                list_names = ['Noisy', 'Clean Truth', 'Clean Guess']
                list_foc = ['_Nom', '_Foc']

                # show the performance
                for m in range(5):
                    fig, axes = plt.subplots(2, 3)
                    for i in range(2):
                        for j in range(3):
                            idx = 3 * i + j
                            # print(idx)
                            ax = axes[i][j]
                            _array = list_images[j]
                            img = ax.imshow(_array[m*50, :, :, i], cmap='hot')
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                            plt.colorbar(img, ax=ax, orientation='horizontal')
                            ax.set_title(list_names[j] + list_foc[i] + '_AE_%d' % (k + 1))
        return images_copies, coefs_copies


    def train_calibration_models(self, images, coefs, N_train, N_test, N_loops, epochs_loop, verbose, batch_size_keras,
                                 plot_val_loss, readout_noise, RMS_readout, readout_copies, save_directory=None):
        """

        :param images: a list of len(N_autoencoders) with the 'clean' PSF images. Output of Autoencoder or Encoder
        :param coefs: a list of len(N_autoencoders) containing the aberration coefficients for each dataset of PSF images
        :param N_train: how many images to use for training
        :param N_test: how many images to use for testing
        :param N_loops: how many loops to cycle over the complete datasets (to add random instances of readout noise)
        :param epochs_loop: epochs to train per loop
        :param verbose: classic Keras verbose option. 1: shows iterations
        :param batch_size_keras:
        :param plot_val_loss:
        :param readout_noise:
        :param RMS_readout:
        :param readout_copies:
        :param save_directory: path where we will save the trained models after finishing
        :return:
        """

        for i, _model in enumerate(self.calibration_models):
            _image, _coef = images[i], coefs[i]
            print("Training Calibration Model: ", _model.cnn_model.name)
            train_images, test_images = _image[:N_train], _image[N_train:]
            train_coefs, test_coefs = _coef[:N_train], _coef[N_train:]
            _model.train_calibration_model(train_images, train_coefs, test_images, test_coefs,
                                           N_loops=N_loops, epochs_loop=epochs_loop, verbose=verbose,
                                           batch_size_keras=batch_size_keras, plot_val_loss=plot_val_loss,
                                           readout_noise=readout_noise, RMS_readout=RMS_readout, readout_copies=readout_copies)
            if save_directory is not None:
                file_name = os.path.join(save_directory, _model.cnn_model.name + '.h5')
                print("Saving Trained Model: ", file_name)
                _model.cnn_model.save(file_name)

    def calibrate_iterations_autoencoder(self, test_images, test_coefs, N_iter, wavelength,
                                         readout_noise=False, RMS_readout=None, mode='Decoded'):
        """
        Calibrate the aberrations iteratively using the Autoencoder approach
        At each iteration we add some readout noise and then show the images to each Autoencoder
        if mode == 'Decoded'
            each Autoencoder will output a 'clean' PSF image without feature contamination or readout noise
        if mode == 'Encoded'
            each Encoder will output an 'encoded' PSF image
        with such outputs, each Calibration Model will predict the aberrations for the subset of polynomials
        it is in charge of. After aggregating all predictions into a single guess of size self.PSF_model.N_coef
        we will apply a correction and update the PSF images

        :param test_images:
        :param test_coefs:
        :param N_iter:
        :param wavelength:
        :param readout_noise:
        :param RMS_readout:
        :param mode:
        :return:
        """

        nominal_test_images = test_images[0]
        nominal_test_coefs = test_coefs[0]
        before_imgs = nominal_test_images
        before_coef = nominal_test_coefs
        print("\nCalibration with Autoencoders")
        RMS_evolution = []
        # Dummy Calib is used simply to calculate the evolution of RMS for the whole PSF (all aberrations)
        self.dummy_calib = Calibration(PSF_model=self.PSF_model)

        # How to split the aberrations across autoencoders?
        slices_zernike = self.slice_zernike_polynomials()

        for k in range(N_iter):         # Loop over N iterations

            print("\nIteration #%d" % (k + 1))
            if readout_noise is True:
                noisy_before = self.noise_effects.add_readout_noise(before_imgs, RMS_READ=RMS_readout)
            else:
                noisy_before = before_imgs

            guess_total = []
            for i, (_min, _max) in enumerate(utils.pairwise(slices_zernike)):        # Loop over each Autoencoder

                if mode == 'Decoded':       # We work with the complete Autoencoder
                    # (1) We clean the images with the Autoencoder
                    print("Cleaning PSF images with Autoencoder: ", self.autoencoder_models[i].name)
                    clean_imgs = self.autoencoder_models[i].predict(noisy_before)

                elif mode == 'Encoded':     # We use only the Encoder part of the Autoencoder
                    print("Encoding the PSF images: ")
                    clean_imgs = self.encoders[i].predict(noisy_before)

                # (2) We estimate the aberrations with the Calibration Model
                print("Estimating aberrations with Calibration Model: ", self.calibration_models[i].cnn_model.name)
                guess_coef = self.calibration_models[i].cnn_model.predict(clean_imgs)
                guess_total.append(guess_coef)

                # Double Check that the predictions are actually useful
                sliced_coef = before_coef[:, _min:_max]
                norm_before = norm(sliced_coef)
                norm_after = norm(sliced_coef - guess_coef)
                print("\nLocal Norm Variation")
                print(norm_before)
                print(norm_after)

            # (3) We join the estimations of all Autoencoders to correct the aberrations in a single step
            guess_total = np.concatenate(guess_total, axis=-1)
            # print(before_coef[0])
            # print(guess_total[0])
            residual = before_coef - guess_total
            print("\nTotal Norm Variation")
            print(norm(before_coef))
            print(norm(residual))

            rms_before, rms_after = self.dummy_calib.calculate_RMS(before_coef, residual, wavelength)
            rms_pair = [rms_before, rms_after]
            RMS_evolution.append(rms_pair)

            if k == N_iter - 1:
                break

            new_images = self.compute_PSF(residual)
            before_imgs = new_images
            before_coef = residual

        return RMS_evolution, residual

    def validation(self, datasets, N_train, N_test, RMS_readout, k_image=0):
        nominal_dataset = datasets[0]
        for _model, clean_dataset in zip(self.autoencoder_models, datasets[1:]):
            test_noisy = self.noise_effects.add_readout_noise(nominal_dataset[N_train:], RMS_READ=RMS_readout)
            test_clean = clean_dataset[N_train:]
            guess_clean = _model.predict(test_noisy)
            for _img in [test_noisy[k_image, :, :, 0], test_clean[k_image, :, :, 0], guess_clean[k_image, :, :, 0]]:
                plt.figure()
                plt.imshow(_img, cmap='hot')
                plt.colorbar()

            for _img in [test_noisy[k_image, :, :, 1], test_clean[k_image, :, :, 1], guess_clean[k_image, :, :, 1]]:
                plt.figure()
                plt.imshow(_img, cmap='hot')
                plt.colorbar()

class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.validation_loss = []

    def on_epoch_end(self, epoch, logs={}):
        guess_clean = self.model.predict(self.validation_data[0])
        true_clean = self.validation_data[1]
        MSE = np.mean(np.sum((guess_clean - true_clean)**2, axis=(1, 2, 3)), axis=0)
        print(MSE)
        self.validation_loss.append(MSE)

        return

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

