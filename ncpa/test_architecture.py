"""


                    -||  ARCHITECTURE  ||-

Investigation into different CNN Architecture Hyper-parameters

- Hyper-parameter tuning (Keras-tuner Hyperband)

Author: Alvaro Menduina
Date: Nov 2020


"""

import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from keras.backend import clear_session
from keras.utils.layer_utils import count_params

import psf
import utils
import calibration

# In case we modify Calibration while running this experiment, we can reload the changes
import importlib
importlib.reload(calibration)

# PSF bits
N_PIX = 256                         # pixels for the Fourier arrays
pix = 32                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength
SPAX = 4.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
print("Nominal Parameters | Spaxel Scale and Wavelength")
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
N_levels = 8                        # Number of Zernike radial levels levels
coef_strength = 0.2                # Strength of Zernike aberrations
diversity = 1.0                     # Strength of the Defocus diversity
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
kernel_size = 3
input_shape = (pix, pix, 2,)
SNR = 500                           # SNR for the Readout Noise
N_loops, epochs_loop = 5, 5         # How many times to loop over the training
readout_copies = 2                  # How many copies with Readout Noise to use
N_iter = 3                          # How many iterations to run the calibration (testing)


def calculate_roc(calibration_model, PSF_model, test_images, test_coef, N_iter=3, noise=False, SNR_noise=None):
    """
    Calculate the 'equivalent' of Receiving Operating Characteristic curve for a calibration model

    At each iteration, we calculate the ratio between the norm of the residual coefficients
    and the norm of the original test coefficients. Then we calculate the fraction of PSF cases
    for which that ratio is below a certain threshold.

    This gives us an idea of the performance of the model

    :param calibration_model:
    :param PSF_model:
    :param test_images:
    :param test_coef:
    :param N_iter:
    :param noise:
    :param SNR_noise:
    :return:
    """

    N_PSF = test_coef.shape[0]                              # How many PSF cases do we have
    N_eps = 200                                             # How many thresholds to sample
    percentage = np.linspace(0, 100, N_eps)                 # Array of percentage thresholds
    roc = np.zeros((N_iter, N_eps))                         # ROC array

    # get the norm of the test coef at the beginning
    norm_test0 = norm(test_coef, axis=1)

    if noise:
        # Add readout noise to the test images
        test_images = calibration_model.noise_effects.add_readout_noise(test_images, RMS_READ=1 / SNR_noise)

    # placeholder arrays for the images and coefficients
    psf0, coef0 = test_images, test_coef
    for k in range(N_iter):
        print("\nIteration #%d" % (k + 1))
        # predict the coefficients
        guess = calibration_model.cnn_model.predict(psf0)
        residual = coef0 - guess
        norm_residual = norm(residual, axis=1)
        # compare the norm of the residual to that of the original coefficients
        ratios = norm_residual / norm_test0 * 100

        # calculate the ROC
        for j in range(N_eps):
            # compute the fraction of cases whose ratio norm residual / norm test coef is smaller than threshold
            roc[k, j] = np.sum(ratios < percentage[j]) / N_PSF

        # update the PSF images for the next iteration
        new_PSF = np.zeros_like(test_images)
        for i in range(N_PSF):
            if i % 500 == 0:
                print(i)
            psf_nom, s_nom = PSF_model.compute_PSF(residual[i])
            psf_foc, s_foc = PSF_model.compute_PSF(residual[i], diversity=True)
            new_PSF[i, :, :, 0] = psf_nom
            new_PSF[i, :, :, 1] = psf_foc

        if noise:
            # Add readout noise to the test images
            new_PSF = calibration_model.noise_effects.add_readout_noise(new_PSF, RMS_READ=1 / SNR_noise)

        # Overwrite the placeholder arrays for the next iteration
        psf0 = new_PSF
        coef0 = residual

    return roc, percentage, residual


def calculate_strehl(PSF_model, coef):
    """
    Calculate the Strehl ratio for a set of aberration coefficients
    :param PSF_model:
    :param coef:
    :return:
    """

    N_PSF = coef.shape[0]
    strehl = np.zeros(N_PSF)

    for k in range(N_PSF):
        psf_nom, s_nom = PSF_model.compute_PSF(coef[k])
        strehl[k] = s_nom

    return strehl


def calculate_cum_strehl(calibration_model, PSF_model, test_images, test_coef, N_iter=3, noise=False, SNR_noise=None):
    """
    Calculate the cumulative distribution of Strehl ratios as a function of iterations
    Kind of similar to the ROC curve from above, but we use Strehl rather than ratio of coefficient norms

    At each iteration, we calculate the Strehl ratio for all PSF images in the test set.
    Then we calculate the fraction of PSF cases for which the Strehl ratio
    is higher than a certain threshold.

    This gives us an idea of the performance of the model

    :param calibration_model:
    :param PSF_model:
    :param test_images:
    :param test_coef:
    :param N_iter:
    :param noise:
    :param SNR_noise:
    :return:
    """

    N_PSF = test_coef.shape[0]                              # How many PSF cases do we have
    N_strehl = 200                                          # How many Strehl thresholds to sample
    strehl_threshold = np.linspace(0, 1, N_strehl)          # Array of Strehl thresholds
    strehl_profiles = np.zeros((N_iter + 1, N_strehl))      # Cumulative distributions of Strehls

    # calculate the original strehl
    s0 = calculate_strehl(PSF_model, test_coef)
    for j in range(N_strehl):
        strehl_profiles[0, j] = np.sum(s0 > strehl_threshold[j]) / N_PSF

    # Add Noise
    if noise:
        test_images = calibration_model.noise_effects.add_readout_noise(test_images, RMS_READ=1 / SNR_noise)

    # Placeholder arrays for the iterations
    psf0, coef0 = test_images, test_coef
    for k in range(N_iter):
        print("\nIteration #%d" % (k + 1))
        # predict the coefficients
        guess = calibration_model.cnn_model.predict(psf0)
        residual = coef0 - guess

        # Update the Strehl ratios
        new_strehl = calculate_strehl(PSF_model, residual)
        for j in range(N_strehl):
            strehl_profiles[k + 1, j] = np.sum(new_strehl > strehl_threshold[j]) / N_PSF

        # update the PSF
        new_PSF = np.zeros_like(test_images)
        for i in range(N_test):
            if i % 500 == 0:
                print(i)
            psf_nom, s_nom = PSF_model.compute_PSF(residual[i])
            psf_foc, s_foc = PSF_model.compute_PSF(residual[i], diversity=True)
            new_PSF[i, :, :, 0] = psf_nom
            new_PSF[i, :, :, 1] = psf_foc

        if noise:
            new_PSF = calibration_model.noise_effects.add_readout_noise(new_PSF, RMS_READ=1 / SNR_noise)

        # Overwrite the arrays for the next iteration
        psf0 = new_PSF
        coef0 = residual

    return strehl_profiles, strehl_threshold, residual


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    ### (0) Define a nominal PSF Zernike model with no anamorphic mag. Perfectly circular pupil
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=N_levels, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC, N_PIX=N_PIX,
                                                                          radial_oversize=1.0, anamorphic_ratio=1.0)

    N_zern = zernike_matrix.shape[-1]
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))

    defocus_zernike = np.zeros(zernike_matrix.shape[-1])
    defocus_zernike[1] = diversity / (2 * np.pi)
    PSF_zernike.define_diversity(defocus_zernike)

    # ================================================================================================================ #
    #                                         ARCHITECTURE investigation
    # ================================================================================================================ #

    # Generate some training dataset
    N_train = 20000
    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                              coef_strength, rescale)

    # show the distribution of Strehls
    peaks_train, peaks_test = np.max(train_PSF[:, :, :, 0], axis=(1, 2)), np.max(test_PSF[:, :, :, 0], axis=(1, 2))
    n_bins = 25
    plt.figure()
    plt.hist(peaks_train, bins=2*n_bins, density=True, label=r'Training')
    plt.hist(peaks_test, histtype='step', bins=n_bins, color='black',  density=True, label='Testing')
    plt.xlim([0, 1])
    plt.xlabel(r'Strehl ratio')
    plt.ylabel(r'PDF')
    plt.legend(title='Dataset', loc=1)
    plt.show()

    epochs = 15

    # Impact of the number of the Number of FILTERS of Convolutional Layers
    # We test an architecture with 2 CNN Layers + 1 Dense layer, and see how the N FILTERS affects the performance
    calib_zern = calibration.Calibration(PSF_model=PSF_zernike)
    layer = np.array([4, 8, 16, 32])
    N_layers = layer.shape[0]
    perf = np.zeros((N_layers, N_layers))           # Performance (norm of the residual coefficients)
    param = np.zeros((N_layers, N_layers))          # Number of trainable parameters of each model

    for i in range(N_layers):
        for j in range(N_layers):

            layer_filters = [layer[i], layer[j]]
            calib_zern.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='relu')
            trainable_count = count_params(calib_zern.cnn_model.trainable_weights) / 1000
            param[i, j] = trainable_count           # How many Keras parameters
            losses = calib_zern.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                        N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                        plot_val_loss=False, readout_noise=False,
                                                        RMS_readout=[1. / SNR], readout_copies=readout_copies)

            # test_PSF_noise = calib_zern.noise_effects.add_readout_noise(test_PSF, RMS_READ=1 / SNR)
            guess_coef = calib_zern.cnn_model.predict(test_PSF)
            residual_coef = test_coef - guess_coef
            norm_res = np.mean(norm(residual_coef, axis=1))
            perf[i, j] = norm_res
            # ?? Delete the models?
            del calib_zern.cnn_model
            clear_session()

    # Make an plt.imshow() and include labels showing the number of trainable parameters for each model

    # Limits for the extent
    x_start, x_end = 0, N_layers
    y_start, y_end = 0, N_layers
    size = N_layers

    jump_x, jump_y = (x_end - x_start) / (2.0 * size), (y_end - y_start) / (2.0 * size)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)
    ticks = np.linspace(0.5, N_layers - 0.5, N_layers)

    plt.figure()
    plt.imshow(perf, cmap='Reds', origin='lower', extent=[x_start, x_end, y_start, y_end])
    plt.colorbar()
    plt.xticks(ticks=ticks, labels=layer)
    plt.yticks(ticks=ticks, labels=layer)
    plt.xlabel('Conv2 [Filters]')
    plt.ylabel('Conv1 [Filters]')
    plt.title(r'Norm Residual Coefficients')

    # Add the text with the N trainable params
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = param[y_index, x_index]
            if label / 1000 < 1.0:      # Hundreds of Thousands
                s = "%dK" % label
            elif label / 1000 > 1.0:    # Millions
                s = "%.1fM" % (label / 1000)

            text_x, text_y = x + jump_x, y + jump_y
            plt.text(text_x, text_y, s, color='black', ha='center', va='center')

    plt.show()

    # ================================================================================================================ #
    # # Show a comparison between a model with 2 layers [Conv1, Conv2] and a model with 3 layers [32, Conv1
    #
    # results = [perf, perf_32, perf_64]
    # params = [param, param_32, param_64]
    #

    # [WATCH OUT!] because this bit is "hardcoded", we just run it multiple times changing the parameters

    results = [perf_relu, perf_tanh]
    params = [param_relu, param_tanh]
    minc = np.min([np.min(x) for x in results])
    maxc = np.max([np.max(x) for x in results])

    fig, (ax1, ax2) = plt.subplots(1, 2)

    img1 = ax1.imshow(perf_relu, cmap='Blues', origin='lower', extent=[x_start, x_end, y_start, y_end])
    img1.set_clim(minc, maxc)
    plt.colorbar(img1, ax=ax1, orientation='horizontal')
    # ax1.set_title(r'Architecture [Conv1, Conv2, FC]')
    ax1.set_title(r'Architecture [Conv1, Conv2, FC] | ReLu')

    img2 = ax2.imshow(perf_tanh, cmap='Blues', origin='lower', extent=[x_start, x_end, y_start, y_end])
    img2.set_clim(minc, maxc)
    plt.colorbar(img2, ax=ax2, orientation='horizontal')
    # ax2.set_title(r'Architecture [Conv0(32), Conv1, Conv2, FC]')
    ax2.set_title(r'Architecture [Conv0(32), Conv1, Conv2, FC] | Tanh')

    # img3 = ax3.imshow(perf_sigm, cmap='Reds', origin='lower', extent=[x_start, x_end, y_start, y_end])
    # img3.set_clim(minc, maxc)
    # plt.colorbar(img3, ax=ax3, orientation='horizontal')
    # ax3.set_title(r'Architecture [Conv0(64), Conv1, Conv2, FC]')

    for ax, data in zip([ax1, ax2], [param_relu, param_tanh]):
        ax.set_xlabel('Conv2 [Filters]')
        ax.set_ylabel('Conv1 [Filters]')

        ax.set_xticks(ticks=ticks)
        ax.set_xticklabels(labels=layer)
        ax.set_yticks(ticks=ticks)
        ax.set_yticklabels(labels=layer)

        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
                label = data[y_index, x_index]
                if label / 1000 < 1.0:      # Hundreds of Thousands
                    s = "%dK" % label
                elif label / 1000 > 1.0:    # Millions
                    s = "%.1fM" % (label / 1000)

                text_x = x + jump_x
                text_y = y + jump_y
                ax.text(text_x, text_y, s, color='black', ha='center', va='center')

    plt.show()
    # ================================================================================================================ #
    # Impact of FULLY-CONNECTED layers
    # We add multiple Dense Layers after the Convolutional Layers, and see how the performance changes

    calib_dense = calibration.Calibration(PSF_model=PSF_zernike)
    dense_layers = np.array([1, 2, 3, 4])
    N_dense = dense_layers.shape[0]
    dense_activation = 'tanh'           # We can also choose the activation for the Dense Layers (except the last one(
    cnn_layers = [8, 4]
    perf_dense = np.zeros(N_dense)
    param_dense = np.zeros(N_dense)

    for j in range(N_dense):
        layer_filters = cnn_layers
        calib_dense.create_cnn_model(layer_filters, kernel_size, N_dense=dense_layers[j], name='DENSE',
                                     activation='relu', dense_acti=dense_activation)
        trainable_count = count_params(calib_dense.cnn_model.trainable_weights) / 1000
        param_dense[j] = trainable_count           # How many K parameters
        losses = calib_dense.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                    N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                    plot_val_loss=False, readout_noise=False,
                                                    RMS_readout=[1. / SNR], readout_copies=readout_copies)

        # test_PSF_noise = calib_dense.noise_effects.add_readout_noise(test_PSF, RMS_READ=1 / SNR)
        guess_coef = calib_dense.cnn_model.predict(test_PSF)
        residual_coef = test_coef - guess_coef
        norm_res = np.mean(norm(residual_coef, axis=1))
        perf_dense[j] = norm_res
        # ?? Delete the models?
        del calib_zern.cnn_model
        clear_session()

    print("\nPerformance as function of N of dense layers:")
    print(perf_dense)

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

    hypermodel = CNNHyperModel(input_shape=input_shape, num_classes=N_zern)

    HYPERBAND_MAX_EPOCHS = 15
    MAX_TRIALS = 50
    EXECUTION_PER_TRIAL = 2
    tuner = Hyperband(
        hypermodel,
        max_epochs=HYPERBAND_MAX_EPOCHS,
        objective='val_loss',
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='hyperband',
        project_name='noisy'
    )

    tuner.search_space_summary()

    calib_noisy = calibration.Calibration(PSF_model=PSF_zernike)

    # Get some noisy
    N_copies = 1
    readout_copies = 3
    SNRs = [250, 500, 750]
    # SNRs = [50, 100, 200]
    train_PSF_noisy, train_coef_noisy = [], []
    test_PSF_noisy, test_coef_noisy = [], []
    for k in range(readout_copies):
        for j in range(N_copies):
            noisy_train = calib_noisy.noise_effects.add_readout_noise(train_PSF, RMS_READ=1 / SNRs[k])
            train_PSF_noisy.append(noisy_train)
            train_coef_noisy.append(train_coef)

            noisy_test = calib_noisy.noise_effects.add_readout_noise(test_PSF, RMS_READ=1 / SNRs[k])
            test_PSF_noisy.append(noisy_test)
            test_coef_noisy.append(test_coef)

    train_PSF_noisy = np.concatenate(train_PSF_noisy, axis=0)
    train_coef_noisy = np.concatenate(train_coef_noisy, axis=0)
    test_PSF_noisy = np.concatenate(test_PSF_noisy, axis=0)
    test_coef_noisy = np.concatenate(test_coef_noisy, axis=0)

    tuner.search(train_PSF_noisy, train_coef_noisy, epochs=10,
                 validation_data=(test_PSF_noisy, test_coef_noisy))

    tuner.results_summary()
    best_model = tuner.get_best_models(num_models=1)[0]

    # best HP
    best_hp = tuner.get_best_hyperparameters()[0]
    print(best_hp.values)

    guess_best = best_model.predict(test_PSF_noisy)
    residual_coef = test_coef_noisy - guess_best
    norm_res = np.mean(norm(residual_coef, axis=1))
    print(norm_res)

    calib_noisy.cnn_model = best_model

    N_roc = 5
    # SNRs = [250, 500, 750]
    S = len(SNRs)
    colors_array = [cm.Reds(np.linspace(0.5, 1, N_roc)),
                    cm.Blues(np.linspace(0.5, 1, N_roc)),
                    cm.Greens(np.linspace(0.5, 1, N_roc))]
    fig, axes = plt.subplots(1, len(SNRs))
    for i in range(S):

        _snr = SNRs[i]

        roc, percentage, _res = calculate_roc(calib_noisy, PSF_zernike, test_PSF, test_coef, N_roc,
                                              noise=True, SNR_noise=_snr)

        ax = axes[i]
        for k in range(N_roc):
            roc_data = roc[k]
            auc = np.mean(roc_data)
            ax.plot(percentage, roc_data, color=colors_array[i][k], label="%d (%.3f)" % (k + 1, auc))
        # plt.xscale('log')
        ax.set_xlim([0, 100])
        ax.grid(True)
        ax.set_xticks(np.arange(0, 110, 10))
        ax.set_xlabel(r'Ratio Norm Residuals / Norm Test Coef [percent]')
        ax.set_ylabel(r'Cumulative')
        ax.legend(title='Iteration (AUC)', loc=4)
        ax.set_title(r'ROC | HyperParam Architecture | SNR %d' % _snr)
    plt.show()

    fig, axes = plt.subplots(1, len(SNRs))
    for i in range(S):
        _snr = SNRs[i]

        strehl_roc, threshold, _res = calculate_cum_strehl(calib_noisy, PSF_zernike, test_PSF, test_coef,
                                                           N_iter=N_roc, noise=True, SNR_noise=_snr)

        ax = axes[i]
        for k in range(N_roc):
            strehl_data = strehl_roc[k]
            auc = np.mean(strehl_data)
            ax.plot(threshold, strehl_data, color=colors_array[i][k], label="%d (%.3f)" % (k + 1, auc))
        # plt.xscale('log')
        ax.set_xlim([0, 1])
        ax.grid(True)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel(r'Strehl ratio [ ]')
        ax.set_ylabel(r'Cumulative')
        ax.legend(title='Iteration (AUC)', loc=3)
        ax.set_title(r'ROC-Strehl | HyperParam Architecture | SNR %d' % _snr)
    plt.show()

    # train a reference model
    layer_filters = [32, 32]

    calib_noisy.create_cnn_model(layer_filters, kernel_size, name='NOISE', activation='relu')
    losses = calib_noisy.train_calibration_model(train_PSF_noisy, train_coef_noisy,
                                                 test_PSF_noisy, test_coef_noisy,
                                                 N_loops=1, epochs_loop=5, verbose=1, batch_size_keras=32,
                                                 plot_val_loss=False, readout_noise=False,
                                                 RMS_readout=[1. / SNR], readout_copies=readout_copies)


    # class MyTuner(Hyperband):
    #     def run_trial(self, trial, *args, **kwargs):
    #         # You can add additional HyperParameters for preprocessing and custom training loops
    #         # via overriding `run_trial`
    #         kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32)
    #         super(MyTuner, self).run_trial(trial, *args, **kwargs)
    #
    #
    # tuner = MyTuner(hypermodel,
    #     max_epochs=HYPERBAND_MAX_EPOCHS,
    #     objective='val_loss',
    #     executions_per_trial=EXECUTION_PER_TRIAL,
    #     directory='hyperband',
    #     project_name='batch_size')
    # # Don't pass epochs or batch_size here, let the Tuner tune them.
    # tuner.search(...)



    ### https://github.com/keras-team/keras-tuner/issues/122
    # class MyTuner(kerastuner.tuners.BayesianOptimization):
    #     def run_trial(self, trial, *args, **kwargs):
    #         # You can add additional HyperParameters for preprocessing and custom training loops
    #         # via overriding `run_trial`
    #         kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32)
    #         kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
    #         super(MyTuner, self).run_trial(trial, *args, **kwargs)
    #
    #
    # # Uses same arguments as the BayesianOptimization Tuner.
    # tuner = MyTuner(...)
    # # Don't pass epochs or batch_size here, let the Tuner tune them.
    # tuner.search(...)

    # ================================================================================================================ #

    # Flat Field ??

    calib_flat = calibration.Calibration(PSF_model=PSF_zernike)
    layer_filters = [32, 32]
    SNR = 500
    calib_flat.create_cnn_model(layer_filters, kernel_size, name='FLATS', activation='relu', learning_rate=1e-3)
    # noisy_train = calib_flat.noise_effects.add_readout_noise(train_PSF, RMS_READ=1 / SNR)
    losses = calib_flat.train_calibration_model(train_PSF, train_coef,
                                                test_PSF, test_coef,
                                                N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                plot_val_loss=False, readout_noise=False,
                                                RMS_readout=[1. / SNR], readout_copies=readout_copies)

    def add_flat_field(test_images, max_dev=0.10, noise='Uniform', sigma=1):

        N_PSF = test_images.shape[0]
        pix = test_images.shape[1]
        flat_images = np.zeros_like(test_images)

        for k in range(N_PSF):

            psf_nom = test_images[k, :, :, 0]
            psf_foc = test_images[k, :, :, 1]

            # flat = np.random.uniform(low=1 - max_dev, high=1 + max_dev, size=(pix, pix))
            # flat_foc = np.random.uniform(low=1 - max_dev, high=1 + max_dev, size=(pix, pix))

            if noise == "Uniform":
                flat = np.random.uniform(low=1 - max_dev, high=1 + max_dev, size=(pix, pix))
                # flat_foc = np.random.uniform(low=1 - max_dev, high=1 + max_dev, size=(pix, pix))
            elif noise == "Gaussian":
                flat = np.random.normal(loc=1, scale=max_dev / sigma, size=(pix, pix))
                # flat_foc = np.random.normal(loc=1, scale=max_dev / sigma, size=(pix, pix))
            # flat_foc = np.random.normal(loc=1, scale=max_dev/3, size=(pix, pix))

            flat_images[k, :, :, 0] = flat * psf_nom
            flat_images[k, :, :, 1] = flat * psf_foc

        return flat_images

    # Check possible correlation between the starting Strehl ratio and the impact of flat field
    max_dev = 0.10

    N_figs = 5
    fig, axes = plt.subplots(2, 5)
    for k in range(N_figs):

        flat = np.random.uniform(low=1 - max_dev, high=1 + max_dev, size=(pix, pix))
        p0 = train_PSF[k, :, :, 0]
        s0 = np.max(p0)
        pf = train_PSF[-k -1, :, :, 0]
        sf = np.max(pf)

        img0 = p0 - p0 * flat
        c0 = max(-np.min(img0), np.max(img0))

        imgf = pf - pf * flat
        cf = max(-np.min(imgf), np.max(imgf))

        ax1 = axes[0][k]
        img1 = ax1.imshow(img0, cmap='bwr')
        img1.set_clim(-c0, c0)
        plt.colorbar(img1, ax=ax1, orientation='horizontal')
        ax1.set_title(r'PSF #%d | Strehl %.2f' % (k + 1, s0))

        ax2 = axes[1][k]
        img2 = ax2.imshow(imgf, cmap='bwr')
        img2.set_clim(-cf, cf)
        plt.colorbar(img2, ax=ax2, orientation='horizontal')
        ax2.set_title(r'PSF #%d | Strehl %.2f' % (N_train - k, sf))

    plt.show()

    # We generate a test set with at least 10k images to really get an idea of what's going on
    N = 10000
    _p, _c, new_test_PSF, new_test_coef = calibration.generate_dataset(PSF_zernike, 0, N, coef_strength, rescale)

    # Calculate the initial Strehl ratios. we want to see if the original Strehl correlates with a bad response to flat errors
    peaks_test = np.max(new_test_PSF[:, :, :, 0], axis=(1, 2))
    cc = cm.Reds(peaks_test[::-1])          # flip it so we have the Red at low Strehl and get a cooler plot
    devs = np.array([0.01, 0.05, 0.10])
    N_devs = devs.shape[0]

    # First we calculate the nominal performance. What is the residual when test on "CLEAN" images
    residual_coef0 = new_test_coef - calib_flat.cnn_model.predict(new_test_PSF)
    norm_nom = norm(residual_coef0, axis=1)

    # Calculate the Strehl ratio after correction
    strehl_nominal = np.zeros(N)
    for i in range(N):
        if i % 500 == 0:
            print(i)
        psf_nom, s_nom = PSF_zernike.compute_PSF(residual_coef0[i])
        strehl_nominal[i] = s_nom

    # Now for each value of Flat Field error, we calculate the Strehl after correction
    # and compare it to the Strehl for the nominal case, see if it's higher or lower.
    strehl_diffs = np.zeros((N_devs, N))

    for k in range(N_devs):

        # First we add the Flat Field to the PSF images
        test_PSF_flat_uniform = add_flat_field(new_test_PSF, max_dev=devs[k], noise='Uniform')
        # We predict the aberrations and apply the correction
        residual_coef_uniform = new_test_coef - calib_flat.cnn_model.predict(test_PSF_flat_uniform)
        # norm_uni = norm(residual_coef_uniform, axis=1)

        # We update the Strehl ratio
        strehl_flat = np.zeros(N)
        for i in range(N):
            if i % 500 == 0:
                print(i)
            psf_flat, s_flat = PSF_zernike.compute_PSF(residual_coef_uniform[i])
            strehl_flat[i] = s_flat

        # And calculate the difference
        diff_strehl = strehl_flat - strehl_nominal
        strehl_diffs[k] = diff_strehl
        # diff_norm = (norm_uni - norm_nom) / norm_nom * 100

    fig, axes = plt.subplots(1, N_devs + 1, figsize=(20, 5))
    ax = axes[0]
    ax.scatter(peaks_test * 100, strehl_nominal * 100, color=cc, s=3, label=r'Nominal')
    ax.set_ylim([0, 100])
    ax.set_xlim([0, 100])
    ax.set_xticks(np.arange(0, 110, 10))
    ax.set_xlabel(r'Initial Strehl ratio')
    ax.set_ylabel(r'Corrected Strehl ratio')
    # ax.legend(title='Flat Field', loc=4)
    ax.set_title(r"Nominal Model")
    for k in range(N_devs):
        data = strehl_diffs[k] * 100
        cmax = max(-np.min(data), np.max(data))
        worse = np.sum(data < 0.0) / N
        ax = axes[k + 1]
        ax.fill_between(x=[0, 100], y1=-1.1*cmax, y2=0.0, color='orange', alpha=0.5)
        # ax.fill_between(x=[0, 1], y1=0, y2=1.1*cmax, color='lightgreen', alpha=0.5)
        ax.scatter(peaks_test * 100, data, color=cc, s=3, label=r'$\Delta = %.1f$ pc' % (100*devs[k]))
        ax.axhline(0, linestyle='-.', color='k')
        ax.set_ylim([-1.1*cmax, 1.1*cmax])
        ax.set_xlim([0, 100])
        ax.set_xticks(np.arange(0, 110, 10))
        ax.set_xlabel(r'Initial Strehl ratio')
        ax.legend(title='Flat Field', loc=4)
        ax.set_title(r"Samples with worse Strehl: %d percent" % worse)
        if k == 0:
            ax.set_ylabel(r'Change in Strehl')

    # Why are some of the predictions better when there is flat field?

    # Iterate

    # plt.plot(range(N_iter_flat + 1), strehl_iter)

    j_example = 0
    N_trials = 20
    N_iter_flat = 3
    N = 100
    strehl_data = np.zeros((N_iter_flat, N_trials, N))
    for i in range(N):
        for j in range(N_trials):

            test_image = new_test_PSF[i:i + 1]
            coef_image = new_test_coef[i:i + 1]

            # strehl_iter = [np.max(test_image[0, :, :, 0], axis=(0, 1))]
            test_image = add_flat_field(test_image, max_dev=0.10, noise='Uniform')
            for k in range(N_iter_flat):
                r = coef_image - calib_flat.cnn_model.predict(test_image)
                new_image = np.zeros_like(test_image)
                psf_nom, s_ = PSF_zernike.compute_PSF(r[0])
                strehl_data[k, j, i] = s_
                # strehl_iter.append(s_)
                psf_foc, s__ = PSF_zernike.compute_PSF(r[0], diversity=True)
                new_image[0, :, :, 0] = psf_nom
                new_image[0, :, :, 1] = psf_foc
                new_image = add_flat_field(new_image, max_dev=0.10, noise='Uniform')
                test_image = new_image
                coef_image = r
        # strehl_iter = np.array(strehl_iter)
        # plt.plot(range(N_iter_flat + 1), strehl_iter, color='red')


    import seaborn as sns
    import pandas as pd

    strehl_data = strehl_data.reshape((N_iter_flat, -1))
    data = pd.DataFrame(strehl_data.T, columns=range(N_iter_flat))

    # Violinplot
    fig_violin, ax = plt.subplots(1, 1)
    sns.violinplot(data=data, ax=ax, hue_order=range(N_iter_flat), palette=sns.color_palette("Reds"))
    plt.show()


    # First we add the Flat Field to the PSF images
    test_PSF_flat_uniform = add_flat_field(new_test_PSF, max_dev=devs[k], noise='Uniform')
    # We predict the aberrations and apply the correction
    residual_coef_uniform = new_test_coef - calib_flat.cnn_model.predict(test_PSF_flat_uniform)
    # norm_uni = norm(residual_coef_uniform, axis=1)

    # We update the Strehl ratio
    strehl_flat = np.zeros(N)
    for i in range(N):
        if i % 500 == 0:
            print(i)
        psf_flat, s_flat = PSF_zernike.compute_PSF(residual_coef_uniform[i])
        strehl_flat[i] = s_flat

    residual_coef0 = test_coef - calib_flat.cnn_model.predict(test_PSF)
    norm_res0 = np.mean(norm(residual_coef0, axis=1))

    flat_devs = np.linspace(1e-3, 25, 100) / 100
    N_flats = flat_devs.shape[0]

    norms_flat_uniform = np.zeros(N_flats)
    norms_flat_gaussian2sig = np.zeros(N_flats)
    norms_flat_gaussian3sig = np.zeros(N_flats)

    for k in range(N_flats):

        test_PSF_flat_uniform = add_flat_field(test_PSF, max_dev=flat_devs[k], noise='Uniform')
        test_PSF_flat_gaussian2sig = add_flat_field(test_PSF, max_dev=flat_devs[k], noise='Gaussian', sigma=2)
        test_PSF_flat_gaussian3sig = add_flat_field(test_PSF, max_dev=flat_devs[k], noise='Gaussian', sigma=3)

        residual_coef_uniform = test_coef - calib_flat.cnn_model.predict(test_PSF_flat_uniform)
        norm_res_uniform = np.mean(norm(residual_coef_uniform, axis=1))
        norms_flat_uniform[k] = (norm_res_uniform / norm_res0 - 1) * 100

        residual_coef_gaussian = test_coef - calib_flat.cnn_model.predict(test_PSF_flat_gaussian2sig)
        norm_res_gaussian = np.mean(norm(residual_coef_gaussian, axis=1))
        norms_flat_gaussian2sig[k] = (norm_res_gaussian / norm_res0 - 1) * 100

        residual_coef_gaussian = test_coef - calib_flat.cnn_model.predict(test_PSF_flat_gaussian3sig)
        norm_res_gaussian = np.mean(norm(residual_coef_gaussian, axis=1))
        norms_flat_gaussian3sig[k] = (norm_res_gaussian / norm_res0 - 1) * 100

    colors = cm.Reds(np.linspace(0.5, 1, 3))
    s = 10
    plt.figure()
    plt.scatter(flat_devs * 100, norms_flat_uniform, label=r'$\mathcal{U}\left[1-\Delta, 1+\Delta \right]$',
                color=colors[0], s=s)
    # plt.scatter(flat_devs * 100, norms_flat_gaussian2sig, label=r'$\mathcal{N}(1, %d\sigma=\Delta)$' % 2,
    #             color=colors[1], s=s)
    # plt.scatter(flat_devs * 100, norms_flat_gaussian3sig, label=r'$\mathcal{N}(1, %d\sigma=\Delta)$' % 3,
    #             color=colors[2], s=s)
    plt.xlabel(r'Flat Uncertainty $\Delta$ [percent]')
    plt.ylabel(r'Change in Residual Norm [percent]')
    plt.legend(title='Flat Field Model')
    plt.grid(True)
    plt.xlim([0, 25])
    plt.ylim([0, 5])
    plt.show()

    # We have done this analysis for perfect PSF images. Only adding the flat fields.
    # But what if we couple this with readout noise?

    # we train a Calibration Model on Noisy Images with NO flat field
    # then we test the performance by adding Flat Field errors, and then the Noise on top

    calib_flat_noisy = calibration.Calibration(PSF_model=PSF_zernike)
    # SNRs = [50, 100, 200]
    SNRs = [250, 500, 750]
    flat_devs = np.linspace(1e-3, 25, 100) / 100
    N_flats = flat_devs.shape[0]
    norms = np.zeros((3, N_flats))
    norms_rel = np.zeros((3, N_flats))
    for i, _snr in enumerate(SNRs):
        calib_flat_noisy.create_cnn_model(layer_filters, kernel_size, name='FLATS', activation='relu')
        # noisy_train = calib_flat.noise_effects.add_readout_noise(train_PSF, RMS_READ=1 / SNR)
        losses = calib_flat_noisy.train_calibration_model(train_PSF, train_coef,
                                                    test_PSF, test_coef,
                                                    N_loops=1, epochs_loop=4, verbose=1, batch_size_keras=32,
                                                    plot_val_loss=False, readout_noise=True,
                                                    RMS_readout=[1. / _snr], readout_copies=readout_copies)

        # repeat to get a good value
        m = []
        for j in range(10):
            noisy_test = calib_flat_noisy.noise_effects.add_readout_noise(test_PSF, RMS_READ=1 / _snr)
            residual_coef0 = test_coef - calib_flat_noisy.cnn_model.predict(noisy_test)
            m.append(np.mean(norm(residual_coef0, axis=1)))

        norm_res0 = np.mean(m)

        norms_flat_uniform = np.zeros(N_flats)
        # norms_flat_gaussian2sig = np.zeros(N_flats)
        # norms_flat_gaussian3sig = np.zeros(N_flats)

        for k in range(N_flats):
            test_PSF_flat_uniform = add_flat_field(test_PSF, max_dev=flat_devs[k], noise='Uniform')
            test_PSF_flat_uniform = calib_flat_noisy.noise_effects.add_readout_noise(test_PSF_flat_uniform, RMS_READ=1 / _snr)

            residual_coef_uniform = test_coef - calib_flat_noisy.cnn_model.predict(test_PSF_flat_uniform)
            norm_res_uniform = np.mean(norm(residual_coef_uniform, axis=1))
            # norms_flat_uniform[k] = (norm_res_uniform / norm_res0 - 1) * 100
            norms_flat_uniform[k] = norm_res_uniform
            norms[i, k] = norm_res_uniform
            norms_rel[i, k] = (norm_res_uniform / norm_res0 - 1) * 100


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for i, _snr in enumerate(SNRs):
        ax1.scatter(flat_devs * 100, norms[i], label=r'SNR %d' % _snr,
                    color=colors[i], s=s)
        ax2.scatter(flat_devs * 100, norms_rel[i], label=r'SNR %d' % _snr,
                    color=colors[i], s=s)
    ax1.set_xlabel(r'Flat Uncertainty $\Delta$ [percent]')
    ax2.set_xlabel(r'Flat Uncertainty $\Delta$ [percent]')
    ax1.set_ylabel(r'Residual Coeff. Norm')
    ax2.set_ylabel(r'Change in Residual Norm [percent]')
    for ax in [ax1, ax2]:
        ax.legend(title='Readout Noise', loc=2)
        ax.set_title(r'Flat Field Model: U$\left[1-\Delta, 1+\Delta \right]$')
        ax.grid(True)
        ax.set_xlim([0, 25])
    # plt.ylim(bottom=0)
    plt.show()

    # spatially correlated noise
    max_dev = 0.10
    flat = np.random.uniform(low=1 - max_dev, high=1 + max_dev, size=(pix, pix))
    std_flat = np.std(flat)
    x0 = np.linspace(-1, 1, pix)
    xx, yy = np.meshgrid(x0, x0)
    a, b, c, d, e = np.random.uniform(low=-1, high=1, size=5)
    z = a * xx **2 + b * yy **2 + c * xx * yy + d * xx + e * yy
    z -= np.mean(z)
    z_max = max(-np.min(z), np.max(z))
    # Normalize to +- 1.0
    z /= (z_max)

    z *= max_dev

    noise = flat + z

    # std_noise = np.std(noise)
    # noise = noise / std_noise * std_flat

    fig, axes = plt.subplots(1, 3)
    data = [flat, z, noise]
    imgs = []
    for i in range(3):
        ax = axes[i]
        img = ax.imshow(data[i], cmap='Reds', origin='lower')
        imgs.append(img)
        plt.colorbar(img, ax=ax, orientation='horizontal')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Pixel')

    axes[0].set_title(r'Base Flat Field U$\left[1-\Delta, 1+\Delta \right]$')
    axes[1].set_title(r'Spatially Correlated Term')
    axes[2].set_title(r'Combined Flat Field')

    imgs[0].set_clim(1-max_dev, 1+max_dev)
    imgs[1].set_clim(-max_dev, max_dev)
    imgs[2].set_clim(np.min(noise), np.max(noise))
    # ax3.set_title('Mean: %.2f | Std: %.2f' % (np.mean(noise), np.std(noise)))
    plt.show()

    def add_spatial_flat_field(test_images, max_dev=0.10, noise='Uniform', sigma=1):

        N_PSF = test_images.shape[0]
        pix = test_images.shape[1]
        flat_images = np.zeros_like(test_images)

        # Spatially correlated component
        x0 = np.linspace(-1, 1, pix)
        xx, yy = np.meshgrid(x0, x0)

        for k in range(N_PSF):

            psf_nom = test_images[k, :, :, 0]
            psf_foc = test_images[k, :, :, 1]

            # flat = np.random.uniform(low=1 - max_dev, high=1 + max_dev, size=(pix, pix))
            # flat_foc = np.random.uniform(low=1 - max_dev, high=1 + max_dev, size=(pix, pix))

            if noise == "Uniform":
                flat = np.random.uniform(low=1 - max_dev, high=1 + max_dev, size=(pix, pix))
                # flat_foc = np.random.uniform(low=1 - max_dev, high=1 + max_dev, size=(pix, pix))
            elif noise == "Gaussian":
                flat = np.random.normal(loc=1, scale=max_dev / sigma, size=(pix, pix))
                # flat_foc = np.random.normal(loc=1, scale=max_dev / sigma, size=(pix, pix))
            # flat_foc = np.random.normal(loc=1, scale=max_dev/3, size=(pix, pix))

            # Spatially correlated
            a, b, c, d, e = np.random.uniform(low=-1, high=1, size=5)
            z = a * xx ** 2 + b * yy ** 2 + c * xx * yy + d * xx + e * yy
            z -= np.mean(z)
            z_max = max(-np.min(z), np.max(z))
            # Normalize to +- 1.0
            z /= (z_max)

            z *= max_dev
            noise = flat + z
            # if k == 0:
            #     plt.figure()
            #     plt.imshow(noise, cmap='Reds')
            #     plt.colorbar()


            flat_images[k, :, :, 0] = noise * psf_nom
            flat_images[k, :, :, 1] = noise * psf_foc

        return flat_images


    residual_coef0 = test_coef - calib_flat.cnn_model.predict(test_PSF)
    norm_res0 = np.mean(norm(residual_coef0, axis=1))

    flat_devs = np.linspace(1e-3, 15, 200) / 100
    N_flats = flat_devs.shape[0]

    norms_flat_uniform = np.zeros(N_flats)
    norms_flat_uniform_spa = np.zeros(N_flats)
    # norms_flat_gaussian2sig = np.zeros(N_flats)
    # norms_flat_gaussian3sig = np.zeros(N_flats)

    for k in range(N_flats):

        test_PSF_flat_uniform = add_flat_field(test_PSF, max_dev=flat_devs[k], noise='Uniform')
        test_PSF_flat_uniform_spa = add_spatial_flat_field(test_PSF, max_dev=flat_devs[k], noise='Uniform')
        # test_PSF_flat_gaussian2sig = add_flat_field(test_PSF, max_dev=flat_devs[k], noise='Gaussian', sigma=2)
        # test_PSF_flat_gaussian3sig = add_flat_field(test_PSF, max_dev=flat_devs[k], noise='Gaussian', sigma=3)

        residual_coef_uniform = test_coef - calib_flat.cnn_model.predict(test_PSF_flat_uniform)
        norm_res_uniform = np.mean(norm(residual_coef_uniform, axis=1))
        norms_flat_uniform[k] = (norm_res_uniform / norm_res0 - 1) * 100

        residual_coef_uniform = test_coef - calib_flat.cnn_model.predict(test_PSF_flat_uniform_spa)
        norm_res_uniform = np.mean(norm(residual_coef_uniform, axis=1))
        norms_flat_uniform_spa[k] = (norm_res_uniform / norm_res0 - 1) * 100


    colors = cm.Blues(np.linspace(0.5, 1, 2))
    s = 10
    plt.figure()
    # for i in range(2):
    plt.scatter(flat_devs * 100, norms_flat_uniform, label=r'Uniform',
                color=colors[0], s=s)
    plt.scatter(flat_devs * 100, norms_flat_uniform_spa, label=r'Spatially Correlated',
                color=colors[1], s=s)

    plt.xlabel(r'Flat Uncertainty $\Delta$ [percent]')
    plt.ylabel(r'Change in Residual Norm [percent]')
    plt.legend(title='Flat Field Model')
    plt.grid(True)
    plt.xlim([0, 15])
    plt.ylim(bottom=0)
    plt.xticks(np.arange(0, 16, 1))
    # plt.ylim([0, 5])
    plt.show()

    # ================================================================================================================ #
    N_roc = 5
    roc, percentage, _res = calculate_roc(best_model, PSF_zernike, test_PSF, test_coef, N_roc)

    colors = cm.Greens(np.linspace(0.5, 1, N_roc))
    plt.figure()
    for k in range(N_roc):
        roc_data = roc[k]
        auc = np.mean(roc_data)
        plt.plot(percentage, roc_data, color=colors[k], label="%d (%.3f)" % (k + 1, auc))
    # plt.xscale('log')
    plt.xlim([0, 100])
    plt.grid(True)
    plt.xticks(np.arange(0, 110, 10))
    plt.xlabel(r'Ratio Norm Residuals / Norm Test Coef [percent]')
    plt.ylabel(r'Cumulative')
    plt.legend(title='Iteration (AUC)')
    plt.title(r'ROC | Architecture: [Conv1(64), Conv2(64)]')
    plt.show()

    calib_roc = calibration.Calibration(PSF_model=PSF_zernike)
    layer = np.array([4, 8, 16, 32, 64])
    N_layers = layer.shape[0]
    perf_roc = np.zeros((N_layers, N_layers))
    param_roc = np.zeros((N_layers, N_layers))
    for i in range(N_layers):
        for j in range(N_layers):
            layer_filters = [layer[i], layer[j]]

            calib_roc.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='relu')
            trainable_count = count_params(calib_roc.cnn_model.trainable_weights) / 1000
            param_roc[i, j] = trainable_count           # How many K parameters
            losses = calib_roc.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                        N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                        plot_val_loss=False, readout_noise=False,
                                                        RMS_readout=[1. / SNR], readout_copies=readout_copies)

            roc, percentage, _res = calculate_roc(calib_roc.cnn_model, PSF_zernike, test_PSF, test_coef, N_roc)
            print("\nROC: ", np.mean(roc[-1]))
            perf_roc[i, j] = np.mean(roc[-1])
            # ?? Delete the models?
            del calib_roc.cnn_model
            clear_session()

    # Limits for the extent
    x_start = 0
    x_end = N_layers
    y_start = 0
    y_end = N_layers
    size = N_layers

    jump_x = (x_end - x_start) / (2.0 * size)
    jump_y = (y_end - y_start) / (2.0 * size)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)
    ticks = np.linspace(0.5, N_layers - 0.5, N_layers)

    plt.figure()
    plt.imshow(perf_roc, cmap='Greens', origin='lower', extent=[x_start, x_end, y_start, y_end])
    plt.colorbar()
    plt.xticks(ticks=ticks, labels=layer)
    plt.yticks(ticks=ticks, labels=layer)
    plt.xlabel('Conv2 [filters]')
    plt.ylabel('Conv1 [filters]')
    plt.title(r'ROC | Architecture [Conv1, Conv2, FC]')

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = perf_roc[y_index, x_index]
            s = "%.3f" % label
            # if label / 1000 < 1.0:      # Hundreds of Thousands
            #     s = "%dK" % label
            # elif label / 1000 > 1.0:    # Millions
            #     s = "%.1fM" % (label / 1000)

            text_x = x + jump_x
            text_y = y + jump_y
            plt.text(text_x, text_y, s, color='black', ha='center', va='center')

    plt.show()

    # ================================================================================================================ #
    #                                   Impact of the NUMBER OF TRAINING EXAMPLES
    # ================================================================================================================ #

    N_train_extra = int(10**5)
    N_test_extra = 1000
    datasets = calibration.generate_dataset(PSF_zernike, N_train_extra, N_test_extra, coef_strength, rescale)
    train_PSF_extra, train_coef_extra, test_PSF_extra, test_coef_extra = datasets

    # N_training = np.arange(2000, N_train_extra + 2000, 2000)
    n = np.arange(10000, N_train_extra + 5000, 5000)
    N_training = np.concatenate([np.array([1000, 2000, 3000, 4000, 5000, 7500]), n])
    N_cases = N_training.shape[0]
    res_norm_extra = np.zeros(N_cases)
    roc_extra = np.zeros(N_cases)
    roc_data_extra = []
    for k in range(N_cases):
        calib_extra = calibration.Calibration(PSF_model=PSF_zernike)
        layer_filters = [32, 32]

        calib_extra.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='relu', learning_rate=1e-4)
        # trainable_count = count_params(calib_extra.cnn_model.trainable_weights) / 1000
        # randomly select the dataset
        choice = np.random.choice(range(N_train_extra), size=N_training[k], replace=False)
        losses = calib_extra.train_calibration_model(train_PSF_extra[choice], train_coef_extra[choice],
                                                     test_PSF_extra, test_coef_extra,
                                                     N_loops=1, epochs_loop=20, verbose=1, batch_size_keras=32,
                                                   plot_val_loss=False, readout_noise=False,
                                                   RMS_readout=[1. / SNR], readout_copies=readout_copies)

        # guess_coef = calib_extra.cnn_model.predict(test_PSF_extra)
        # residual_coef = test_coef_extra - guess_coef

        roc, percentage, _res = calculate_roc(calib_extra.cnn_model, PSF_zernike, test_PSF_extra, test_coef_extra, N_roc)
        norm_res = np.mean(norm(_res, axis=1))
        res_norm_extra[k] = norm_res

        roc_data_extra.append(roc[-1])
        roc_extra[k] = np.mean(roc[-1])
        print("\nROC: ", np.mean(roc[-1]))
        del calib_extra.cnn_model
        clear_session()


    plt.figure()
    plt.plot(N_training, res_norm_extra)
    plt.xlim(right=80000)
    plt.show()

    # Let's find the optimum hyper-parameters for the super massive training set.

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

            model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                          loss='mean_squared_error')
            return model

    hypermodel_massive = CNNHyperModelMassive(input_shape=input_shape, num_classes=N_zern)

    HYPERBAND_MAX_EPOCHS = 40
    MAX_TRIALS = 20
    EXECUTION_PER_TRIAL = 2


    tuner = Hyperband(hypermodel_massive, max_epochs=HYPERBAND_MAX_EPOCHS, objective='val_loss',
                      executions_per_trial=EXECUTION_PER_TRIAL, directory='hyperband', project_name='massive')

    tuner.search_space_summary()

    tuner.search(train_PSF_extra, train_coef_extra, epochs=10,
                 validation_data=(test_PSF_extra, test_coef_extra))

    tuner.results_summary()
    best_model = tuner.get_best_models(num_models=1)[0]

    # best HP
    best_hp = tuner.get_best_hyperparameters()[0]
    print(best_hp.values)



    # ================================================================================================================ #

    ### Old stuff


    # ================================================================================================================ #
    # Now with Noise

    SNR = 250
    readout_copies = 3
    N_roc = 4

    calib_noisy = calibration.Calibration(PSF_model=PSF_zernike)
    layer = np.array([4, 8, 16, 32])
    N_layers = layer.shape[0]
    perf_noisy = np.zeros((N_layers, N_layers))
    param_noisy = np.zeros((N_layers, N_layers))
    for i in range(N_layers):
        for j in range(N_layers):
            layer_filters = [layer[i], layer[j]]

            calib_noisy.create_cnn_model(layer_filters, kernel_size, name='NOISE', activation='relu')
            trainable_count = count_params(calib_noisy.cnn_model.trainable_weights) / 1000
            param_noisy[i, j] = trainable_count           # How many K parameters
            losses = calib_noisy.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                        N_loops=3, epochs_loop=3, verbose=1, batch_size_keras=32,
                                                        plot_val_loss=False, readout_noise=True,
                                                        RMS_readout=[1. / SNR], readout_copies=readout_copies)

            # test_PSF_noise = calib_noisy.noise_effects.add_readout_noise(test_PSF, RMS_READ=1 / SNR)
            # guess_coef = calib_noisy.cnn_model.predict(test_PSF_noise)
            # residual_coef = test_coef - guess_coef
            # norm_res = np.mean(norm(residual_coef, axis=1))

            roc, percentage, _res = calculate_roc(calib_noisy, PSF_zernike, test_PSF, test_coef, N_roc, noise=True)

            perf_noisy[i, j] = norm_res
            # ?? Delete the models?
            del calib_noisy.cnn_model
            clear_session()

    # Limits for the extent
    x_start = 0
    x_end = N_layers
    y_start = 0
    y_end = N_layers
    size = N_layers

    jump_x = (x_end - x_start) / (2.0 * size)
    jump_y = (y_end - y_start) / (2.0 * size)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)
    ticks = np.linspace(0.5, N_layers - 0.5, N_layers)

    plt.figure()
    plt.imshow(perf_noisy, cmap='Reds', origin='lower', extent=[x_start, x_end, y_start, y_end])
    plt.colorbar()
    plt.xticks(ticks=ticks, labels=layer)
    plt.yticks(ticks=ticks, labels=layer)
    plt.xlabel('Conv2')
    plt.ylabel('Conv1')
    plt.title(r'Norm Residual Coefficients')

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = param_noisy[y_index, x_index]
            if label / 1000 < 1.0:      # Hundreds of Thousands
                s = "%dK" % label
            elif label / 1000 > 1.0:    # Millions
                s = "%.1fM" % (label / 1000)

            text_x = x + jump_x
            text_y = y + jump_y
            plt.text(text_x, text_y, s, color='black', ha='center', va='center')

    plt.show()


    ## === ''' ### ---==

    # First we see whether the number of filters matters, for a constant number of layers
    layers = [[8, 4],
              [16, 8],
              [32, 16],
              [64, 32]]

    norm_residuals = []
    mean_stds = []
    for layer_filters in layers:
        calib_zern = calibration.Calibration(PSF_model=PSF_zernike)
        calib_zern.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='relu')
        losses = calib_zern.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                    N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                    plot_val_loss=False, readout_noise=False,
                                                    RMS_readout=[1. / SNR], readout_copies=readout_copies)

        # test_PSF_noise = calib_zern.noise_effects.add_readout_noise(test_PSF, RMS_READ=1 / SNR)
        guess_coef = calib_zern.cnn_model.predict(test_PSF)
        residual_coef = test_coef - guess_coef
        norm_res = np.mean(norm(residual_coef, axis=1))
        print(norm_res)
        norm_residuals.append(norm_res)
        stds = np.mean(np.std(residual_coef, axis=0))
        mean_stds.append(stds)

    for k, layer_filters in enumerate(layers):
        print("\nModel Architecture")
        print(layer_filters)
        print(norm_residuals[k])
        print(mean_stds[k])

    # Now we see how the layers affect
    layers = [[8], [16, 8], [32, 16, 8], [64, 32, 16, 8], [128, 64, 32, 16, 8]]
    norm_residuals = []
    mean_stds = []
    for layer_filters in layers:
        calib_zern = calibration.Calibration(PSF_model=PSF_zernike)
        calib_zern.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='relu')
        losses = calib_zern.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                    N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                    plot_val_loss=False, readout_noise=False,
                                                    RMS_readout=[1. / SNR], readout_copies=readout_copies)

        # test_PSF_noise = calib_zern.noise_effects.add_readout_noise(test_PSF, RMS_READ=1 / SNR)
        guess_coef = calib_zern.cnn_model.predict(test_PSF)
        residual_coef = test_coef - guess_coef
        norm_res = np.mean(norm(residual_coef, axis=1))
        print(norm_res)
        norm_residuals.append(norm_res)
        stds = np.mean(np.std(residual_coef, axis=0))
        mean_stds.append(stds)

    for k, layer_filters in enumerate(layers):
        print("\nModel Architecture")
        print(layer_filters)
        print(norm_residuals[k])
        print(mean_stds[k])
