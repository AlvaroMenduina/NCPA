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
pix = 32                            # pixels to crop the PSF images
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
    N_train = 20000
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

    # ================================================================================================================ #
    #                   HYPERPARAMETER TUNING
    # ================================================================================================================ #

    def calculate_roc(calibration_model, PSF_model, test_images, test_coef, N_iter=3, noise=False, SNR_noise=None):

        N_test = test_coef.shape[0]
        N_eps = 200
        percentage = np.linspace(0, 100, N_eps)
        roc = np.zeros((N_iter, N_eps))

        # get the norm of the test coef
        norm_test0 = norm(test_coef, axis=1)

        if noise:
            test_images = calibration_model.noise_effects.add_readout_noise(test_images, RMS_READ=1 / SNR_noise)

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
                roc[k, j] = np.sum(ratios < percentage[j]) / N_test

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

        return roc, percentage, residual

    def calculate_strehl(PSF_model, coef):

        N_PSF = coef.shape[0]
        strehl = np.zeros(N_PSF)
        for k in range(N_PSF):
            psf_nom, s_nom = PSF_model.compute_PSF(coef[k])
            strehl[k] = s_nom
        return strehl

    def calculate_cum_strehl(calibration_model, PSF_model, test_images, test_coef, N_iter=3, noise=False, SNR_noise=None):

        N_PSF = test_coef.shape[0]
        N_strehl = 200
        strehl_threshold = np.linspace(0, 1, N_strehl)
        strehl_profiles = np.zeros((N_iter + 1, N_strehl))

        # calculate the original strehl
        s0 = calculate_strehl(PSF_model, test_coef)
        for j in range(N_strehl):
            strehl_profiles[0, j] = np.sum(s0 > strehl_threshold[j]) / N_PSF

        # Add Noise
        if noise:
            test_images = calibration_model.noise_effects.add_readout_noise(test_images, RMS_READ=1 / SNR_noise)

        psf0, coef0 = test_images, test_coef
        for k in range(N_iter):
            print("\nIteration #%d" % (k + 1))
            # predict the coefficients
            guess = calibration_model.cnn_model.predict(psf0)
            residual = coef0 - guess

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

    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, Dense

    from kerastuner.tuners import Hyperband
    from kerastuner import HyperModel

    class CNNHyperModel(HyperModel):

        def __init__(self, input_shape, num_classes):
            self.input_shape = input_shape
            self.num_classes = num_classes
            self.activation = 'relu'

        def build(self, hp):

            model = Sequential()
            model.add(Conv2D(filters=hp.Choice('num_filters_0', values=[8, 16, 32, 64]),
                             kernel_size=hp.Choice('kernel_size', values=[3, 4, 5]),
                             activation=self.activation, input_shape=self.input_shape))

            for i in range(hp.Int('num_layers', 1, 3)):
                model.add(Conv2D(filters=hp.Choice('num_filters_%d' % (i + 1), values=[8, 16, 32, 64]),
                                 kernel_size=(kernel_size, kernel_size), activation=self.activation))
            model.add(Flatten())
            model.add(Dense(N_zern))
            model.summary()

            model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])),
                          loss='mean_squared_error')
            return model

    hypermodel = CNNHyperModel(input_shape=input_shape, num_classes=N_zern)

    HYPERBAND_MAX_EPOCHS = 7
    MAX_TRIALS = 40
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

    # Get some noisy
    readout_copies = 3
    SNRs = [250, 500, 750]
    train_PSF_noisy, train_coef_noisy = [], []
    test_PSF_noisy, test_coef_noisy = [], []
    for k in range(readout_copies):
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
