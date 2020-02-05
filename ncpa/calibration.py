import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
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


def generate_dataset(PSF_model, N_train, N_test, coef_strength, rescale=0.35):
    """

    :param PSF_model:
    :param N_train:
    :param N_test:
    :param coef_strength:
    :param rescale:
    :return:
    """

    N_coef = PSF_model.N_coef
    pix = PSF_model.crop_pix
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

        im0, _s = PSF_model.compute_PSF(coefs[i])
        dataset[i, :, :, 0] = im0

        im_foc, _s = PSF_model.compute_PSF(coefs[i], diversity=True)
        dataset[i, :, :, 1] = im_foc

        if i % 500 == 0:
            print(i)

    return dataset[:N_train], coefs[:N_train], dataset[N_train:], coefs[N_train:]


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
        N_channels = 2
        pix = self.PSF_model.crop_pix
        N_samples = coefs.shape[0]
        dataset = np.zeros((N_samples, pix, pix, N_channels))
        for i in range(N_samples):
            if i % 500 == 0:
                print(i)

            im0, _s = self.PSF_model.compute_PSF(coefs[i])
            dataset[i, :, :, 0] = im0

            im_foc, _s = self.PSF_model.compute_PSF(coefs[i], diversity=True)
            dataset[i, :, :, 1] = im_foc
        print("Updated")
        return dataset

    def create_cnn_model(self, layer_filers, kernel_size, name, activation='relu'):
        """
        Creates a CNN model for NCPA calibration
        :return:
        """
        pix = self.PSF_model.crop_pix
        input_shape = (pix, pix, 2,)        # Multiple Wavelength Channels
        model = Sequential()
        model.name = name
        model.add(Conv2D(layer_filers[0], kernel_size=(kernel_size, kernel_size), strides=(1, 1),
                         activation=activation, input_shape=input_shape))
        for N_filters in layer_filers[1:]:
            model.add(Conv2D(N_filters, (kernel_size, kernel_size), activation=activation))

        model.add(Flatten())
        model.add(Dense(self.PSF_model.N_coef))
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
        residual = test_coefs + guess_coef
        norm_coefs = norm(test_coefs, axis=1)
        norm_residual = norm(residual, axis=1)
        ratio = np.mean(norm_residual / norm_coefs) * 100
        return ratio

    def train_calibration_model(self, images_batches, coefs_batches, test_images, test_coefs,
                                N_loops=10, epochs_loop=50, verbose=1, plot_val_loss=False,
                                readout_noise=False, RMS_readout=[1. / 100]):

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
                    train_images = self.noise_effects.add_readout_noise(train_images, RMS_READ=RMS_readout[i_noise])
                    test_images = self.noise_effects.add_readout_noise(test_images, RMS_READ=RMS_readout[i_noise])

                # Allergen Notice: At this point clean_images may contain noise
                train_history = self.cnn_model.fit(x=train_images, y=train_coefs,
                                                   validation_data=(test_images, test_coefs),
                                                   epochs=epochs_loop, batch_size=N_train,
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
        :param coef_before:
        :param coef_after:
        :param wavelength: working wavelength [microns]
        :return:
        """

        N_samples = coef_before.shape[0]
        print("\nCalculating RMS before / after for %d samples" % N_samples)
        RMS0, RMS = [], []
        for k in range(N_samples):
            wavef_before = wavelength * 1e3 * np.dot(self.PSF_model.RBF_flat, coef_before[k])
            wavef_after = wavelength * 1e3 * np.dot(self.PSF_model.RBF_flat, coef_after[k])
            RMS0.append(np.std(wavef_before))
            RMS.append(np.std(wavef_after))
        mu0, mu = np.mean(RMS0), np.mean(RMS)
        med0, med = np.median(RMS0), np.median(RMS)
        std0, std = np.std(RMS0), np.std(RMS)
        print("RMS Before: %.1f +- %.1f nm (%.1f median)" % (mu0, std0, med0))
        print("RMS  After: %.1f +- %.1f nm (%.1f median)" % (mu, std, med))
        return RMS0, RMS

    def calibrate_iterations(self, test_images, test_coefs, wavelength, N_iter=3,
                             readout_noise=False, RMS_readout=1./100):

        """
        Run the calibration for several iterations
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
            predicted_coefs = self.cnn_model.predict(images_before)
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

        return RMS_evolution

    def plot_RMS_evolution(self, RMS_evolution):
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

