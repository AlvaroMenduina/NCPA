"""

                               -||  ENSEMBLE  &  DROPOUT ||-

Does it help to use an Ensemble of calibration models to average across the predictions?
In this script we look at 2 common approaches in Machine Learning for performance enhancement

    (1) ENSEMBLE: training multiple models with the same datasets can be used to improve performance
        because of randomized training conditions, multiple models will end up with different behaviours
        even after being exposed to the same training set.
        After training, one can average across the predictions of many models to obtain robust predictions

        We test this approach to show that we can gain some performance by training an ensemble of models

    (2) DROPOUT:

Author: Alvaro Menduina
Date: Feb 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import psf
import utils
import calibration

utils.print_title(message='\nN C P A', font=None, random_font=False)

# PSF bits
N_PIX = 256                         # pixels for the Fourier arrays
pix = 30                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength
SPAX = 4.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
print("Nominal Parameters | Spaxel Scale and Wavelength")
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

N_actuators = 20                    # Number of actuators in [-1, 1] line
alpha_pc = 10                       # Height [percent] at the neighbour actuator (Gaussian Model)

# Machine Learning bits
N_train, N_test = 10000, 1000       # Samples for the training of the models
coef_strength = 0.40                # Strength of the actuator coefficients
diversity = 0.65                    # Strength of extra diversity commands
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filters = [64, 32, 16, 8]      # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
N_loops, epochs_loop = 5, 5         # How many times to loop over the training
readout_copies = 2                  # How many copies with Readout Noise to use
N_iter = 3                          # How many iterations to run the calibration (testing)

directory = os.path.join(os.getcwd(), 'Ensemble')

import importlib
importlib.reload(calibration)


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Calculate the Actuator Centres
    centers = psf.actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act = len(centers[0])
    psf.plot_actuators(centers, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)

    plt.show()

    # Calculate the Actuator Model Matrix (Influence Functions)
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)

    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]
    diversity_actuators = 1 / (2 * np.pi) * np.random.uniform(-1, 1, size=N_act)

    # Create the PSF model using the Actuator Model for the wavefront
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=diversity_actuators)

    plt.show()

    # ================================================================================================================ #
    #                                    Machine Learning | Single Calibration Model
    # ================================================================================================================ #

    # Let us begin with a baseline design. One calibration model with No Dropout or anything fancy


    # Generate training and test datasets (clean PSF images)
    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                              coef_strength, rescale)

    # Single Calibration Model || Baseline design
    calib = calibration.Calibration(PSF_model=PSF_actuators)
    calib.create_cnn_model(layer_filters, kernel_size, name='SINGLE_MODEL', activation='relu')
    losses = calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                           N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    RMS_evolution, residual = calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                         readout_noise=True, RMS_readout=1./SNR)

    calib.plot_RMS_evolution(RMS_evolution)

    # ================================================================================================================ #
    #                             Ensemble Approach | Multiple Calibration Models
    # ================================================================================================================ #

    SNR = 150
    readout_copies = 5
    N_ensemble = 5
    layer_filters_ensemb = [[64, 32, 16, 8], [64, 32, 16], [64, 32], [32, 16], [16, 8]]
    ensemb_calib = calibration.CalibrationEnsemble(PSF_model=PSF_actuators)
    ensemb_calib.generate_ensemble_models(N_models=N_ensemble, layer_filters=layer_filters_ensemb, kernel_size=kernel_size,
                                          name='ENSEMBLE', activation='relu')
    ensemb_calib.train_ensemble_models(train_PSF, train_coef, test_PSF, test_coef,
                                           N_iter, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    # Let's see what happens to the performance as a function of the Number of ensemble models we average across
    mus_ens, sts_ens = [], []
    print("\nTesting Ensemble Models")
    # for how_many in np.arange(1, N_ensemble + 1):
    for how_many in [1, N_ensemble + 1]:
        print("\nHow many models to use: ", how_many)
        _RMS, _res = ensemb_calib.calibrate_iterations_ensemble(how_many, test_PSF, test_coef, wavelength=WAVE,
                                                                N_iter=N_iter, readout_noise=True, RMS_readout=1./SNR)
        final_RMS = _RMS[-1][-1]
        avg, std = np.mean(final_RMS), np.std(final_RMS)
        mus_ens.append(avg)
        sts_ens.append(std)

    # How much did the performance improve?
    improv_mu = (mus_ens[0] - mus_ens[-1]) / mus_ens[0] * 100
    improv_rms = (sts_ens[0] - sts_ens[-1]) / sts_ens[0] * 100
    print("\nDid the performance improve?")
    print(" 1 Model  | RMS after: %.2f +- %.2f nm" % (mus_ens[0], sts_ens[0]))
    print("%d Models | RMS after: %.2f +- %.2f nm" % (N_ensemble, mus_ens[-1], sts_ens[-1]))
    print("Relative improvement: %.1f per cent" % improv_mu)

    # Plot of how the performance improves with N models
    plt.figure()
    plt.errorbar(np.arange(1, N_ensemble + 1), mus_ens, yerr=sts_ens, fmt='o')
    plt.ylim(bottom=0)
    plt.show()

    # ================================================================================================================ #
    #                                   Dropout
    # ================================================================================================================ #

    # Let's train a calibration model that has Dropout after each CNN layer
    # Quick way to model "Bayesian Neural Networks" and get access to uncertainties in the predictions

    # Dropout Calibration Model ||
    readout_copies = 3
    SNR = 250
    drop_calib = calibration.Calibration(PSF_model=PSF_actuators)
    drop_calib.create_cnn_model(layer_filters, kernel_size, name='DROPOUT', activation='relu', dropout=0.25)
    losses = drop_calib.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                N_loops, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                                readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    # Run iterations with Dropout. At each iteration, we average across the Posterior of the predictions
    N_iter = 5
    RMS_evo_drop, _drop_resid = drop_calib.calibrate_iterations(test_PSF, test_coef, wavelength=WAVE, N_iter=N_iter,
                                                                readout_noise=True, RMS_readout=1./SNR,
                                                                dropout=True, N_samples_drop=500)
    #
    # # Compare the performance between the Nominal Model and the one with Dropout
    calib.plot_RMS_evolution(RMS_evolution)
    drop_calib.plot_RMS_evolution(RMS_evo_drop)
    plt.show()

    # noisy_test_PSF = drop_calib.noise_effects.add_readout_noise(test_PSF, RMS_READ=1./SNR)
    _drop_results, mean_drop, uncert_drop = drop_calib.predict_with_uncertainty(noisy_test_PSF, N_samples=500)

    nom_pred = calib.cnn_model.predict(noisy_test_PSF)
    bins = np.linspace(-coef_strength, coef_strength, 50)
    for k in range(5):
        plt.figure()
        plt.hist(_drop_results[:, 0, k], bins=bins, histtype='step', label='Dropout Posterior')
        std_k = np.std(_drop_results[:, :, k], axis=0)
        print(np.mean(std_k))
        plt.axvline(nom_pred[0, k], linestyle='--', color='blue', label='Nominal Model')
        plt.axvline(test_coef[0, k], color='black', label='Ground Truth')
        plt.legend()
    plt.show()

    # check what has more bias: the nominal model, or the dropout model
    biases_nom = np.zeros(N_act)
    biases_drop = np.zeros(N_act)
    for k in range(N_act):

        bias_nom = np.abs(test_coef[:, k] - nom_pred[0, k])
        biases_nom[k] = np.mean(bias_nom)

        mu_drop = np.mean(_drop_results[:, :, k], axis=0)
        bias_drop = np.abs(test_coef[:, k] - mu_drop)
        biases_drop[k] = np.mean(bias_drop)



    residual_nom = test_coef - nom_pred
    std_nom = np.std(residual_nom, axis=0)
    residual_drop = test_coef - mean_drop
    std_drop = np.std(residual_drop, axis=0)

    n_grid = int(np.floor(np.sqrt(N_act)))


    # ================================================================================================================ #
    #                                   Ensemble of Dropout
    # ================================================================================================================ #

    # Let's see if an ensemble of models WITH Dropout perform better than an ensemble of models WITHOUT it

    N_ens_drop = 5
    ensemb_drop_calib = calibration.CalibrationEnsemble(PSF_model=PSF_actuators)
    ensemb_drop_calib.generate_ensemble_models(N_models=N_ens_drop, layer_filters=[layer_filters for k in range(N_ens_drop)], kernel_size=kernel_size,
                                               name='ENSEMBLE_DROPOUT', activation='relu', drop_out=0.10)
    ensemb_drop_calib.train_ensemble_models(train_PSF, train_coef, test_PSF, test_coef,
                                           N_iter, epochs_loop, verbose=1, batch_size_keras=32, plot_val_loss=False,
                                           readout_noise=True, RMS_readout=[1. / SNR], readout_copies=readout_copies)

    RMS_ens_drop, _rr = ensemb_drop_calib.calibrate_iterations_ensemble(N_ens_drop, test_PSF, test_coef, wavelength=WAVE,
                                                                        N_iter=N_iter, readout_noise=True, RMS_readout=1./SNR,
                                                                        dropout=True, N_samples_drop=500)

    final_RMS = RMS_ens_drop[-1][-1]
    mu_ens_drop = np.mean(final_RMS)
    std_ens_drop = np.std(final_RMS)

    print("\nPerformance Summary")
    print("1 Model  | RMS after: %.2f +- %.2f nm" % (mus_ens[0], sts_ens[0]))
    print("%d Models | RMS after: %.2f +- %.2f nm" % (N_ensemble, mus_ens[-1], sts_ens[-1]))
    print("[Dropout] %d Models | RMS after: %.2f +- %.2f nm" % (N_ens_drop,mu_ens_drop, std_ens_drop))

    # # Check whether the predictions vary across the models in the list
    noisy_test_PSF = ensemb_drop_calib.noise_effects.add_readout_noise(test_PSF, RMS_READ=1./SNR)
    k = 1
    hists = []
    plt.figure()
    for _model in ensemb_drop_calib.ensemble_models:
        drop_calib.cnn_model = _model
        _drop_results, mean_drop, uncert_drop = drop_calib.predict_with_uncertainty(noisy_test_PSF, N_samples=500)
        x = _drop_results[:, 0, k]
        h = np.histogram(x, bins=bins)
        hists.append(h[0])
        plt.hist(x, bins=bins, histtype='step', label=_model.name)
    plt.axvline(test_coef[0, k], color='black', label='Ground Truth')
    plt.legend()
    plt.show()

    from scipy.stats import binned_statistic
    x = _drop_results[:, 0, k]
    binx = binned_statistic(x, x, bins=bins)

    digitized = np.digitize(x, bins)

    hs = np.array(hists)
    pdf = np.prod(hs, axis=0)
    peak = np.max(pdf)
    pdf = pdf / peak










