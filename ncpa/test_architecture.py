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


# PSF bits
N_PIX = 256                         # pixels for the Fourier arrays
pix = 64                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength
SPAX = 4.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
print("Nominal Parameters | Spaxel Scale and Wavelength")
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
N_levels = 10                        # Number of Zernike radial levels levels
coef_strength = 0.17                # Strength of Zernike aberrations
diversity = 1.0                     # Strength of the Defocus diversity
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
kernel_size = 3
input_shape = (pix, pix, 2,)
SNR = 500                           # SNR for the Readout Noise
N_loops, epochs_loop = 5, 5         # How many times to loop over the training
readout_copies = 2                  # How many copies with Readout Noise to use
N_iter = 3                          # How many iterations to run the calibration (testing)

import importlib
importlib.reload(calibration)

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
    # Does the number of pixels used to crop the images matter?
    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_zernike, N_train, N_test,
                                                                              coef_strength, rescale)

    epochs = 10

    from keras.backend import clear_session
    from keras.utils.layer_utils import count_params

    calib_zern = calibration.Calibration(PSF_model=PSF_zernike)
    layer = np.array([4, 8, 16, 32])
    N_layers = layer.shape[0]
    perf_sigm = np.zeros((N_layers, N_layers))
    param_sigm = np.zeros((N_layers, N_layers))
    for i in range(N_layers):
        for j in range(N_layers):
            layer_filters = [layer[i], layer[j]]

            calib_zern.create_cnn_model(layer_filters, kernel_size, name='NOM_ZERN', activation='sigmoid')
            trainable_count = count_params(calib_zern.cnn_model.trainable_weights) / 1000
            param_sigm[i, j] = trainable_count           # How many K parameters
            losses = calib_zern.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                        N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                        plot_val_loss=False, readout_noise=False,
                                                        RMS_readout=[1. / SNR], readout_copies=readout_copies)

            # test_PSF_noise = calib_zern.noise_effects.add_readout_noise(test_PSF, RMS_READ=1 / SNR)
            guess_coef = calib_zern.cnn_model.predict(test_PSF)
            residual_coef = test_coef - guess_coef
            norm_res = np.mean(norm(residual_coef, axis=1))
            perf_sigm[i, j] = norm_res
            # ?? Delete the models?
            del calib_zern.cnn_model
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
    plt.imshow(perf_sigm, cmap='Reds', origin='lower', extent=[x_start, x_end, y_start, y_end])
    plt.colorbar()
    plt.xticks(ticks=ticks, labels=layer)
    plt.yticks(ticks=ticks, labels=layer)
    plt.xlabel('Conv2')
    plt.ylabel('Conv1')
    plt.title(r'Norm Residual Coefficients')

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = param_tanh[y_index, x_index]
            if label / 1000 < 1.0:      # Hundreds of Thousands
                s = "%dK" % label
            elif label / 1000 > 1.0:    # Millions
                s = "%.1fM" % (label / 1000)

            text_x = x + jump_x
            text_y = y + jump_y
            plt.text(text_x, text_y, s, color='black', ha='center', va='center')

    plt.show()

    # ================================================================================================================ #
    # # Show a comparison between a model with 2 layers [Conv1, Conv2] and a model with 3 layers [32, Conv1
    #
    # results = [perf, perf_32, perf_64]
    # params = [param, param_32, param_64]
    #
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

    calib_dense = calibration.Calibration(PSF_model=PSF_zernike)
    dense_layers = np.array([1, 2, 3, 4])
    N_dense = dense_layers.shape[0]
    cnn_layers = [8, 4]
    # N_cnn = len(cnn_layers)
    perf_dense = np.zeros((N_dense))
    param_dense = np.zeros((N_dense))

    for j in range(N_dense):
        layer_filters = cnn_layers
        calib_dense.create_cnn_model(layer_filters, kernel_size, N_dense=dense_layers[j], name='DENSE',
                                     activation='relu', dense_acti='tanh')
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

    for i in range(N_cnn):
        plt.plot(dense_layers, perf_dense[i])
    plt.show()


    # ================================================================================================================ #
    # Now with Noise

    SNR = 250

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

            test_PSF_noise = calib_noisy.noise_effects.add_readout_noise(test_PSF, RMS_READ=1 / SNR)
            guess_coef = calib_noisy.cnn_model.predict(test_PSF_noise)
            residual_coef = test_coef - guess_coef
            norm_res = np.mean(norm(residual_coef, axis=1))
            perf_noisy[i, j] = norm_res
            # ?? Delete the models?
            del calib_noisy.cnn_model
            clear_session()

    # ================================================================================================================ #
    #                   HYPERPARAMETER TUNING
    # ================================================================================================================ #

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

    class CNNHyperModel(HyperModel):

        def __init__(self, input_shape, num_classes):
            self.input_shape = input_shape
            self.num_classes = num_classes
            self.activation = 'relu'

        def build(self, hp):

            model = keras.Sequential()
            model.add(Conv2D(filters=hp.Choice('num_filters_0', values=[8, 16, 32, 64], default=64,),
                             kernel_size=3, activation=self.activation, input_shape=self.input_shape))

            for i in range(hp.Int('num_layers', 1, 4)):
                model.add(Conv2D(filters=hp.Choice('num_filters_%d' % (i), values=[8, 16, 32, 64], default=64,),
                                 kernel_size=(kernel_size, kernel_size), activation=self.activation))
            model.add(Flatten())
            model.add(Dense(N_zern))
            model.summary()

            model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])),
                          loss='mean_squared_error')
            return model

    hypermodel = CNNHyperModel(input_shape=input_shape, num_classes=N_zern)

    HYPERBAND_MAX_EPOCHS = 40
    MAX_TRIALS = 20
    EXECUTION_PER_TRIAL = 2


    tuner = Hyperband(
        hypermodel,
        max_epochs=HYPERBAND_MAX_EPOCHS,
        objective='val_loss',
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='hyperband',
        project_name='hyperparam'
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

    ### https://github.com/keras-team/keras-tuner/issues/122
    class MyTuner(kerastuner.tuners.BayesianOptimization):
        def run_trial(self, trial, *args, **kwargs):
            # You can add additional HyperParameters for preprocessing and custom training loops
            # via overriding `run_trial`
            kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32)
            kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
            super(MyTuner, self).run_trial(trial, *args, **kwargs)


    # Uses same arguments as the BayesianOptimization Tuner.
    tuner = MyTuner(...)
    # Don't pass epochs or batch_size here, let the Tuner tune them.
    tuner.search(...)



    # ROC
    def calculate_roc(test_coef, guess_coef):

        eps_min = 0.01
        N_eps = 50
        percentage = np.linspace(0, eps_min, N_eps)

        N_samples = test_coef.shape[0]
        N_coef = test_coef.shape[-1]
        roc = np.zeros((N_coef, N_eps))
        rel_diff = np.zeros((N_samples, N_coef))
        for k in range(N_coef):
            for i in range(N_samples):
                res = test_coef[i, k] - guess_coef[i, k]
                rel_diff[i, k] = np.abs(res / test_coef[i, k]) * 100

            values = rel_diff[:, k]
            for j in range(N_eps):
                threshold = percentage[j]
                num = values > threshold
                roc[k, j] = np.sum(num) / N_samples


        return rel_diff, roc

    rel_diff, roc = calculate_roc(test_coef, guess_best)

    eps_min = 0.01
    N_eps = 50
    percentage = np.linspace(0, eps_min, N_eps)
    plt.figure()
    plt.plot(percentage, roc[0])
    plt.show()





    # ================================================================================================================ #

    ### Old stuff

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
