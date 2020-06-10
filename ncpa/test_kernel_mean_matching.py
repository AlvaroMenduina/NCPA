"""
Kernel Mean Matching experiment



KMM Method based on https://github.com/skyblueutd/KMM
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as sk
from cvxopt import matrix, solvers

import psf
import utils
import calibration
import noise

gamma = 1.0


class KernelMeanMatching(object):

    def __init__(self):

        return

    def calculate_kernel_width(self, training_set):

        N_samples = training_set.shape[0]
        distances = []

        for i in range(N_samples):
            for j in range(i + 1, N_samples):
                data_i = training_set[i]
                data_j = training_set[j]
                dist = np.sum((data_i - data_j) ** 2)
                distances.append(np.sqrt(dist))

        kernel_width = np.median(np.array(distances))

        return kernel_width

    def calculate_beta_weights(self, training_set, test_set):

        N_train = training_set.shape[0]
        N_test = test_set.shape[0]

        # Calculate the Gamma to use for the RBF kernel
        # gamma = self.calculate_kernel_width(training_set)
        # gamma = 1.0
        print("Kernel width: ", gamma)

        # Calculate Kernel
        print('Computing kernel for training data ...')
        K_ns = sk.rbf_kernel(training_set, training_set, gamma)
        # print(K_ns.shape)
        K = 0.9 * (K_ns + K_ns.T)

        # Calculate Kappa
        print('Computing kernel for kappa ...')
        kappa_r = sk.rbf_kernel(training_set, test_set, gamma)
        ones = np.ones(shape=(N_test, 1))
        kappa = np.dot(kappa_r, ones)
        kappa = -(N_train / N_test) * kappa

        # calculate eps
        eps = (np.sqrt(N_train) - 1) / np.sqrt(N_train)

        # constraints
        A0 = np.ones(shape=(1, N_train))
        A1 = -np.ones(shape=(1, N_train))
        A = np.vstack([A0, A1, -np.eye(N_train), np.eye(N_train)])
        b = np.array([[N_train * (eps + 1), N_train * (eps - 1)]])
        b = np.vstack([b.T, -np.zeros(shape=(N_train, 1)), np.ones(shape=(N_train, 1)) * 100])

        print('Solving quadratic program for beta ...')
        P = matrix(K, tc='d')
        q = matrix(kappa, tc='d')
        G = matrix(A, tc='d')
        h = matrix(b, tc='d')
        beta = solvers.qp(P, q, G, h)
        return [i for i in beta['x']]

### PARAMETERS ###

# PSF bits
N_PIX = 128                         # pixels for the Fourier arrays
pix = 32                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength
SPAX = 4.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)
N_actuators = 12                    # Number of actuators in [-1, 1] line
alpha_pc = 20                       # Height [percent] at the neighbour actuator (Gaussian Model)

N_train = 7500
N_test = 1500
layer_filers = [256, 128, 32, 8]    # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
epochs = 50                         # Training epochs



if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Calculate the Actuator Centres
    centers = psf.actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act = len(centers[0])
    psf.plot_actuators(centers, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)

    # Calculate the Actuator Model Matrix (Influence Functions)
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)

    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]
    diversity_actuators = 2 / (2 * np.pi) * np.random.uniform(-1, 1, size=N_act)

    # Create the PSF model using the Actuator Model for the wavefront
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=diversity_actuators)

    plt.show()

    # Generate training and test datasets (clean PSF images)
    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                              coef_strength=0.25, rescale=0.5)

    random_order = np.arange(N_train)
    np.random.shuffle(random_order)
    # Shuffle the training set
    shuffle_train_PSF = np.zeros_like(train_PSF)
    shuffle_train_coef = np.zeros_like(train_coef)
    for i in range(N_train):
        _psf = train_PSF[random_order[i]]
        _c = train_coef[random_order[i]]
        shuffle_train_PSF[i] = _psf
        shuffle_train_coef[i] = _c


    # check the peaks
    peaks_nom = np.max(shuffle_train_PSF[:, :, :, 0], axis=(1, 2))
    peaks_foc = np.max(shuffle_train_PSF[:, :, :, 1], axis=(1, 2))

    # ================================================================================================================ #
    #                                         Kernel Mean Matching
    # ================================================================================================================ #

    # Get the nominal PSF, and flatten the features
    training_set = shuffle_train_PSF[:, :, :, 0].reshape((N_train, -1))
    test_set = test_PSF[:, :, :, 0].reshape((N_test, -1))

    noisy_train_PSF = np.zeros_like(train_PSF)
    noisy_test_PSF = np.zeros_like(test_PSF)

    # Add Noise

    # Test set
    sigma_test = 0.2
    noisy_test_set = np.zeros_like(test_set)
    for k in range(N_test):
        _noise = np.random.normal(0.0, scale=sigma_test, size=pix**2)
        noisy_test_set[k] = test_set[k] + _noise
        noisy_test_PSF[k, :, :, 0] = test_PSF[k, :, :, 0] + _noise.reshape((pix, pix))
        noisy_test_PSF[k, :, :, 1] = test_PSF[k, :, :, 1] + _noise.reshape((pix, pix))

    # Training set
    sigma_train = np.linspace(0, 1.0 * sigma_test, N_train)
    mu = 0.001
    mu_train = np.linspace(-mu, mu, N_train, endpoint=True)
    noisy_training_set = np.zeros_like(training_set)
    for k in range(N_train):
        mu_t, sigma_t = mu_train[k], sigma_train[k]
        _noise = np.random.normal(mu_t, scale=sigma_t, size=pix ** 2)
        noisy_training_set[k] = training_set[k] + _noise
        noisy_train_PSF[k, :, :, 0] = shuffle_train_PSF[k, :, :, 0] + _noise.reshape((pix, pix))
        noisy_train_PSF[k, :, :, 1] = shuffle_train_PSF[k, :, :, 1] + _noise.reshape((pix, pix))

    # Calculate the betas with KMM
    KMM = KernelMeanMatching()
    beta = KMM.calculate_beta_weights(noisy_training_set, noisy_test_set)

    beta = np.array(beta)
    # beta /= np.max(beta)

    fig, ax = plt.subplots(1, 1)
    ax.plot(range(N_train), beta, label='%.3f' % gamma)
    ax.set_ylabel(r'$\beta$ sample weight')
    ax.set_xlabel(r'Sample')
    ax.legend()


    np.save('beta', beta)
    np.save('noisy_train_PSF', noisy_train_PSF)
    np.save('train_coef', shuffle_train_coef)
    np.save('noisy_test_PSF', noisy_test_PSF)
    np.save('test_coef', test_coef)
    np.save('diversity_actuators', diversity_actuators)

    plt.show()

    # ================================================================================================================ #
    #                                         Train the ML models
    # ================================================================================================================ #


    # Initialize Convolutional Neural Network model for calibration
    nominal_model = calibration.create_cnn_model(layer_filers, kernel_size, input_shape,
                                                     N_classes=N_act, name='CALIBR', activation='relu')

    # Train the calibration model
    train_history = nominal_model.fit(x=noisy_train_PSF, y=train_coef,
                                          validation_data=(noisy_test_PSF, test_coef),
                                          epochs=epochs, batch_size=32, shuffle=True, verbose=1, sample_weight=None)

    # Evaluate performance
    guess_coef = nominal_model.predict(noisy_test_PSF)
    residual_coef = test_coef - guess_coef
    norm_before = np.mean(norm(test_coef, axis=1))
    norm_after = np.mean(norm(residual_coef, axis=1))
    print("\nPerformance:")
    print("Average Norm Coefficients")
    print("Before: %.2f" % norm_before)
    print("After : %.2f" % norm_after)

    # Now use the sample weights from KMM
    # Initialize Convolutional Neural Network model for calibration
    kmm_model = calibration.create_cnn_model(layer_filers, kernel_size, input_shape,
                                             N_classes=N_act, name='CALIBR', activation='relu')

    # Train the calibration model
    alpha = beta + 1.0
    train_history_kmm = kmm_model.fit(x=noisy_train_PSF, y=train_coef,
                                          validation_data=(noisy_test_PSF, test_coef),
                                          epochs=2, batch_size=32, shuffle=True, verbose=1,
                                      sample_weight=None)


    # Evaluate performance
    guess_coef_kmm = kmm_model.predict(noisy_test_PSF)
    residual_coef_kmm = test_coef - guess_coef_kmm
    norm_before = np.mean(norm(test_coef, axis=1))
    norm_after = np.mean(norm(residual_coef_kmm, axis=1))
    print("\nPerformance:")
    print("Average Norm Coefficients")
    print("Before: %.2f" % norm_before)
    print("After : %.2f" % norm_after)



    #
    # # Generate random Test Set with fixed Sigma
    # sigma = 0.5
    # test_set = np.zeros((N_test, N_pix**2))
    # for k in range(N_test):
    #     test_set[k] = np.random.normal(0.0, sigma, size=N_pix**2)
    #
    # # Generate a random Training set with increasing sigma towards the true one
    # sigma_train = np.linspace(sigma, 2*sigma, N_train)
    # # mu_train = np.linspace(0.0, 1.0, N_train)
    # training_set = np.zeros((N_train, N_pix**2))
    # for j in range(N_train):train_PSF
    #     training_set[j] = np.random.normal(0.0, sigma_train[j], size=N_pix**2)
    #
    # # Calculate the betas with KMM
    # KMM = KernelMeanMatching()
    # beta = KMM.calculate_beta_weights(training_set, test_set)