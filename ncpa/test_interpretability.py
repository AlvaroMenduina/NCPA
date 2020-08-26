"""

        ~~ || Interpreting the CNN predictions || ~~

Author: Alvaro Menduina
Date: August 2020

Description: with this script, we try to interpret and understand the predictions
of a calibration model (CNN) that uses an Actuator-based wavefront description

For this purpose we use Shapley values, a game theory method to distribute
payouts among player of a collective game, based on the total payout and their
relative contributions.

In this case, the 'payout' would be the (marginal) predictions of the CNN for
the actuator commands, and the 'players' would be the pixel features of the
PSF images (in-focus and defocused).

From that perspective, the Shapley values are a measure of feature importance
and tell us what particular pixel features the calibration model is using
to predict the coefficients

At the end of the script we also visualize the activation of the convolutional layers
to get a feel for what the model is doing
"""

from time import time
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from keras import models

import psf
import utils
import calibration
import noise
import shap


# PSF bits
N_PIX = 256                         # pixels for the Fourier arrays
pix = 32                            # pixels to crop the PSF images
WAVE = 1.5                          # microns | reference wavelength
# We oversample to 2.0 mas pixels to properly see the PSF features in the Shapley values
SPAX = 2.0                          # mas | spaxel scale
RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
RHO_OBSC = 0.30 * RHO_APER  # ELT central obscuration
utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)
N_actuators = 2*16                    # Number of actuators in [-1, 1] line
alpha_pc = 15                       # Height [percent] at the neighbour actuator (Gaussian Model)
diversity = 0.30                    # Strength of extra diversity commands

# Machine Learning bits
N_train, N_test = 10000, 500       # Samples for the training of the models
coef_strength = 1.8 / (2 * np.pi)     # Strength of the actuator coefficients
rescale = 0.35                      # Rescale the coefficients to cover a wide range of RMS
layer_filters = [16, 8]   # How many filters per layer
kernel_size = 3
input_shape = (pix, pix, 2,)
epochs = 10                         # Training epochs
SNR = 500

def hide_axes(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # (1) We begin by creating a Zernike PSF model with Defocus as diversity
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=5, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.0)
    zernike_matrices = [zernike_matrix, pupil_mask_zernike, flat_zernike]
    PSF_zernike = psf.PointSpreadFunction(matrices=zernike_matrices, N_pix=N_PIX,
                                          crop_pix=pix, diversity_coef=np.zeros(zernike_matrix.shape[-1]))

    defocus_zernike = np.zeros(zernike_matrix.shape[-1])
    defocus_zernike[1] = diversity

    # Calculate the Actuator Centres
    centers = psf.actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act = len(centers[0])
    psf.plot_actuators(centers, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)

    plt.show()

    # Calculate the Actuator Model Matrix (Influence Functions)
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)

    actuator_matrices = [actuator_matrix, pupil_mask, flat_actuator]

    # Create the PSF model using the Actuator Model for the wavefront
    PSF_actuators = psf.PointSpreadFunction(matrices=actuator_matrices, N_pix=N_PIX,
                                            crop_pix=pix, diversity_coef=np.zeros(N_act))

    # Use Least Squares to find the actuator commands that mimic the Zernike defocus
    zernike_fit = calibration.Zernike_fit(PSF_zernike, PSF_actuators, wavelength=WAVE, rho_aper=RHO_APER)
    defocus_zernike = np.zeros((1, zernike_matrix.shape[-1]))
    defocus_zernike[0, 1] = diversity
    defocus_actuators = zernike_fit.fit_zernike_wave_to_actuators(defocus_zernike, plot=True, cmap='Reds')[:, 0]

    # Update the Diversity Map on the actuator model so that it matches Defocus
    diversity_defocus = defocus_actuators
    PSF_actuators.define_diversity(diversity_defocus)

    # At this point we have our PSF actuator model ready

    # ================================================================================================================ #
    #                                         Train the Machine Learning model                                         #
    # ================================================================================================================ #

    calib_actu = calibration.Calibration(PSF_model=PSF_actuators)
    calib_actu.create_cnn_model(layer_filters, kernel_size, name='NOM_ACTU', activation='relu')

    # We create the different PSF images for training and testing the performance
    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                              coef_strength=coef_strength, rescale=rescale)

    # We train the calibration model [NO NOISE]
    losses = calib_actu.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                plot_val_loss=False, readout_noise=False)

    # evaluate the RMS performance
    guess_actu = calib_actu.cnn_model.predict(test_PSF)
    residual_coef = test_coef - guess_actu

    RMS0, RMS = np.zeros(N_test), np.zeros(N_test)
    for k in range(N_test):
        ini_phase = np.dot(PSF_actuators.model_matrix, test_coef[k])
        res_phase = np.dot(PSF_actuators.model_matrix, residual_coef[k])
        RMS0[k] = np.std(ini_phase[PSF_actuators.pupil_mask])
        RMS[k] = np.std(res_phase[PSF_actuators.pupil_mask])
    meanRMS0 = np.mean(RMS0)
    meanRMS = np.mean(RMS)
    print("\nCalibration Model Performance:")
    print("Mean RMS before calibration: %.4f rad" % meanRMS0)
    print("Mean RMS after calibration: %.4f rad" % meanRMS)

    # Show an example of the true wavefront and the one guessed by the calibration model
    fig, (ax1, ax2) = plt.subplots(1, 2)
    phase0 = np.dot(PSF_actuators.model_matrix, test_coef[0])
    clim = max(-np.min(phase0), np.max(phase0))
    im1 = ax1.imshow(phase0, cmap='RdBu', extent=[-1, 1, -1, 1])
    im1.set_clim([-clim, clim])
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlim([-RHO_APER, RHO_APER])
    ax1.set_ylim([-RHO_APER, RHO_APER])

    im2 = ax2.imshow(np.dot(PSF_actuators.model_matrix, guess_actu[0]), cmap='RdBu', extent=[-1, 1, -1, 1])
    im2.set_clim([-clim, clim])
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlim([-RHO_APER, RHO_APER])
    ax2.set_ylim([-RHO_APER, RHO_APER])
    plt.show()

    # ================================================================================================================ #
    #                                               Shapley values                                                     #
    # ================================================================================================================ #

    start = time()
    N_background = 50           # How many PSF images from training we use as Background for the Shapley
    N_shap_samples = 50         # How many samples from the test set to use in the Shapley calculations
    clean_background = train_PSF[np.random.choice(N_train, N_background, replace=False)]
    deep_expainer = shap.DeepExplainer(calib_actu.cnn_model, clean_background)

    # Select only the first N_shap from the test set
    test_PSF_shap = test_PSF[:N_shap_samples]
    test_coef_shap = test_coef[:N_shap_samples]
    guess_coef_shap = guess_actu[:N_shap_samples]
    shap_values = deep_expainer.shap_values(test_PSF_shap)
    total_time = time() - start
    print("Computed SHAP values for %d samples | %d background in: %.1f sec" % (N_shap_samples, N_background, total_time))


    def show_shap_example(shap_values, test_PSF_shap, test_coef, guess_coef, i_exa, j_actu):
        """
        Show an example of the Shapley value

        It will show both in-focus and defocused PSF (linear and log scale)
        for a chosen example from the test set (i_exa) followed by the
        Shapley values for a particular actuator (j_actu)

        :param shap_values: a list of len(N_act) containing the Shapley values [N_PSF, pix, pix, 2]
        :param test_PSF_shap: the test PSF images associated with the Shapley values [N_PSF, pix, pix, 2]
        :param test_coef: the true actuator coefficients for those test PSF images [N_PSF, N_act]
        :param guess_coef: the guessed coefficients from the calibration mode [N_PSF, N_act]
        :param i_exa: index for the example that we want to show
        :param j_actu: index for the actuator command for which we want the Shapley values
        :return:
        """

        # Compute how that actuator affects the PSF

        shap_val_nom = shap_values[j_actu][i_exa, :, :, 0]          # Shapley values for the In-focus channel
        shap_val_foc = shap_values[j_actu][i_exa, :, :, 1]          # Shapley values for the Defocused channel
        # Use the same clim for both Shapley maps to properly compare the two channels
        smax_nom = max(-np.min(shap_val_nom), np.max(shap_val_nom))
        smax_foc = max(-np.min(shap_val_foc), np.max(shap_val_foc))
        MAX = max(smax_nom, smax_foc)

        # Get the PSF image chanels
        img_nom = test_PSF_shap[i_exa, :, :, 0]
        img_foc = test_PSF_shap[i_exa, :, :, 1]

        cmap = 'plasma'
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
        im1 = ax1.imshow(img_nom, cmap=cmap)
        plt.colorbar(im1, ax=ax1)
        ax1.set_title(r'In-focus PSF')

        im2 = ax2.imshow(np.log10(np.abs(img_nom)), cmap=cmap)
        plt.colorbar(im2, ax=ax2)
        ax2.set_title(r'In-focus PSF [log10]')

        im3 = ax3.imshow(shap_val_nom, cmap='bwr')
        im3.set_clim(-MAX, MAX)
        ax3.set_title(r'SHAP [Actuator #%d] Command: %.3f' % (j_actu + 1, test_coef[i_exa, j_actu]))
        plt.colorbar(im3, ax=ax3)

        # --------

        im4 = ax4.imshow(img_foc, cmap=cmap)
        ax4.set_title(r'Defocused PSF')
        plt.colorbar(im4, ax=ax4)

        im5 = ax5.imshow(np.log10(np.abs(img_foc)), cmap=cmap)
        ax5.set_title(r'Defocused PSF [log10]')
        plt.colorbar(im5, ax=ax5)

        im6 = ax6.imshow(shap_val_foc, cmap='bwr')
        im6.set_clim(-MAX, MAX)
        ax6.set_title(r'SHAP [Actuator #%d] Prediction: %.3f' % (j_actu + 1, guess_coef[i_exa, j_actu]))
        plt.colorbar(im6, ax=ax6)

    # Show a couple of examples for a given actuator command
    for k in range(3):
        show_shap_example(shap_values=shap_values, test_PSF_shap=test_PSF_shap, test_coef=test_coef_shap,
                          guess_coef=guess_coef_shap, i_exa=k, j_actu=0)
    plt.show()

    # If we want to compare actuators that are diametrically opposed, we can find the index for the opposite actuator
    eps = 1e-4
    k_act = 0
    act_cent = np.array(centers[0])
    xc, yc = act_cent[k_act]
    k_x = np.argwhere(np.abs(act_cent[:, 0] + xc) < eps)
    k_y = np.argwhere(np.abs(act_cent[:, 1] + yc) < eps)
    k_opp = np.intersect1d(k_x, k_y)[0]

    print("Actuator #%d: [%.4f, %.4f]" % (k_act, xc, yc))
    print("Actuator #%d: [%.4f, %.4f]" % (k_opp, act_cent[k_opp][0], act_cent[k_opp][1]))

    def show_multiple_examples(shap_values, test_coef_shap, j_actu, n_row=4, n_col=5):
        """
        Here we show the Shapley values for a particular actuator
        across multiple PSF images (only in-focus channel), to see if
        some of the features are repeated

        To facilitate this, we split the Shapley values by their sign
        according to the value of the actuator command for a given PSF image
        In other words, if the actuator command is positive we only show the positive Shapley
        to see which features contributed positively to that prediction

        :param shap_values: a list of len(N_act) containing the Shapley values [N_PSF, pix, pix, 2]
        :param test_coef_shap: the true actuator coefficients for those test PSF images [N_PSF, N_act]
        :param j_actu: index for the actuator command for which we want the Shapley values
        :param n_row: how many rows for the image grid
        :param n_col: how many columns for the image grid
        :return:
        """

        cc = shap_values[j_actu][:, :, :, 0]  # show the In-Focus Shapley
        fig, axes = plt.subplots(n_row, n_col, dpi=150)
        for k in range(n_row * n_col):
            ax = axes.flatten()[k]
            c = cc[k]  # We get the Shapley for that test PSF image
            command = test_coef_shap[k, j_actu]  # We get the actuator command for that test PSF image

            if command > 0.0:
                # If the command is positive, we mask out the negative value and show the positive Shapley in red
                c_mask = c > 0.0
                img = ax.imshow(np.abs(c) * c_mask, cmap='Reds', origin='lower')
                clim = max(0, np.max(c))
                img.set_clim(0, clim)

            if command < 0.0:
                # If the command is negative, we mask out the positive value and show the negative Shapley in blue
                c_mask = c < 0.0
                img = ax.imshow(np.abs(c) * c_mask, cmap='Blues', origin='lower')
                clim = max(0, -np.min(c))
                img.set_clim(0, clim)
            # We specify the value of the actuator command so that we can compare similar cases
            ax.set_title("%.3f" % command)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        plt.tight_layout()
        plt.show()

    show_multiple_examples(shap_values, test_coef_shap, j_actu=0)


    def reds_and_blues(shap_values, test_coef_shap, j_actu, channel):
        """
        Get the Shapley values and the test_coef
        and split the values according to the sign of the command
        for that particular test PSF

        We need this to average the positive / negative Shapley values
        across many test PSF cases to see which features are preferentially
        used to identify a given actuator

        :param shap_values: a list of len(N_act) containing the Shapley values [N_PSF, pix, pix, 2]
        :param test_coef_shap: the true actuator coefficients for those test PSF images [N_PSF, N_act]
        :param j_actu: index for the actuator command for which we want the Shapley values
        :param channel: which channel to use, 0: in-focus, 1: defocused
        :return:
        """

        reds, blues = [], []
        for k_exa in range(N_shap_samples):
            s_val = shap_values[j_actu][k_exa, :, :, channel]
            command = test_coef_shap[k_exa, j_actu]
            if command > 0.0:
                # we select only the positive Shapley, masking the negative values
                c_mask = s_val > 0.0
                red = c_mask * s_val
                reds.append(red)
            if command < 0.0:
                c_mask = s_val < 0.0
                blue = c_mask * s_val
                blues.append(blue)

        pos_shap = np.mean(np.array(reds), axis=0)
        neg_shap = np.mean(np.array(blues), axis=0)

        return pos_shap, neg_shap

    def average_delta_psf(j_actu, channel):
        """
        The changes in the PSF induced by poking a particular actuator
        (what we call the differential features), slightly depend on the
        underlying wavefront map

        Thus, we want to average that out by simulating many PSF images
        with random wavefront errors and poking the actuator

        :param j_actu: index for the actuator we want to poke
        :param channel: which PSF channel to simulate
        :return:
        """


        diffs = []
        for k in range(50):
            # Create some random wavefront error
            new_coef = np.random.uniform(low=-coef_strength, high=coef_strength, size=N_act)
            if channel == 1:        # the defocus
                new_coef += diversity_defocus
            psf0, s0 = PSF_actuators.compute_PSF(new_coef)

            # Now calculate the PSF after slightly poking the chosen actuator
            new_coef[j_actu] += 0.01
            psf_actu, sactu = PSF_actuators.compute_PSF(new_coef)
            psf_diff = psf_actu - psf0
            diffs.append(psf_diff)

        # Calculate the average change in PSF features when poking that actuator
        mean_diff = np.mean(np.array(diffs), axis=0)
        mean_diff /= np.max(np.abs(mean_diff))

        return mean_diff

    def compare_shapley_to_diff_feat(shap_values, test_coef_shap, j_actu):

        # Calculate the average Shapley values across all PSF images for both channels
        pos_shap_nom, neg_shap_nom = reds_and_blues(shap_values, test_coef_shap, j_actu, channel=0)
        pos_shap_foc, neg_shap_foc = reds_and_blues(shap_values, test_coef_shap, j_actu, channel=1)

        # Calculate the average change in PSF features when poking that actuator
        mean_diff_nom = average_delta_psf(j_actu, channel=0)
        mean_diff_foc = average_delta_psf(j_actu, channel=1)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, dpi=100)
        img1 = ax1.imshow(pos_shap_nom, cmap='Reds', origin='lower')
        ax1.set_title(r'Mean (+) Shapley')
        ax1.text(5.5, 0.9, 'In-focus',{'color': 'black', 'fontsize': 10, 'ha': 'center', 'va': 'center',
                                   'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})

        img2 = ax2.imshow(-neg_shap_nom, cmap='Blues', origin='lower')
        ax2.set_title(r'Mean (-) Shapley')

        # Turn it into a modified log scale to highlight the features
        img3 = ax3.imshow(np.log10(1 + 5*np.abs(mean_diff_nom)), cmap='Reds', origin='lower')
        # img3 = ax3.imshow(np.abs(mean_diff), cmap='Reds', origin='lower')
        ax3.set_title(r'Mean PSF change')

        img4 = ax4.imshow(pos_shap_foc, cmap='Reds', origin='lower')
        ax4.text(5.50, 0.9, 'Defocus',{'color': 'black', 'fontsize': 10, 'ha': 'center', 'va': 'center',
                                   'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})

        img5 = ax5.imshow(-neg_shap_foc, cmap='Blues', origin='lower')

        img6 = ax6.imshow(np.log10(1 + 5*np.abs(mean_diff_foc)), cmap='Reds', origin='lower')

        # Hide the pixel coordinates and axes
        for ax in fig.axes:
            hide_axes(ax)

        return

    # Show the comparison between differential features and Shapley values
    for j_actu in [-1, -2]:
        compare_shapley_to_diff_feat(shap_values, test_coef_shap, j_actu)

    plt.show()

    # ================================================================================================================ #
    #                         Visualising the Activation for each Convolutional Layer                                  #
    # ================================================================================================================ #


    def calculate_activations(calibration_model, img_tensor, plot=False):
        """
        Calculate the Activation for each convolution layer
        in the calibration model. We give a PSF image (in-focus + defocused channels)
        and visualize the outputs of the convolutional layers

        :param calibration_model:
        :param img_tensor: a [1, pix, pix, 2] PSF image datacube
        :param plot:
        :return:
        """

        model = calibration_model.cnn_model
        layer_outputs = [layer.get_output_at(0) for layer in model.layers]      # get the outputs of the layers
        # Construct a model that receives the input of the calibration CNN model and produces as output
        # the activations
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

        # Get the activations for the given image tensor
        activations = activation_model.predict(img_tensor)
        print(activations[-1])  # the guessed

        layer_names = ['Convolution_1', 'Convolution_2']

        # To properly see what areas are being masked with 0.0
        # we use the colormap Reds, and will "set as white values under X" with X = epsilon
        cmap = plt.get_cmap('Reds')
        cmap.set_under('white')

        images_per_row = 4
        grids = []

        for k, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):  # Displays the feature maps

            n_features = layer_activation.shape[-1]  # Number of features in the feature map
            size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
            print(size)
            n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))

            for col in range(n_cols):  # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0, :, :, col * images_per_row + row]
                    # channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                    # channel_image /= channel_image.std()
                    # channel_image *= 64
                    # channel_image += 128
                    # channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size: (col + 1) * size,  # Displays the grid
                    row * size: (row + 1) * size] = channel_image[::-1, :]  # origin 'lower' convention

            grids.append(display_grid)

            if plot:
                scale = 2. / size
                fig, (ax1, ax2) = plt.subplots(1, 2)
                img1 = ax1.imshow(img_tensor[0, :, :, 0], cmap='plasma', origin='lower')
                ax1.set_title(r'In-focus PSF')
                img2 = ax2.imshow(display_grid, cmap=cmap, vmin=np.spacing(0.0))
                ax2.set_title(r'Convolution Layer #%d Outputs' % (k + 1))
                for ax in fig.axes:
                    hide_axes(ax)

        return grids

    grids = calculate_activations(calibration_model=calib_actu, img_tensor=test_PSF[0:1], plot=True)
    plt.show()

    def compare_activations(calibration_model, j_actu):
        """
        We investigate how poking one actuator modifies the activations
        Calculate the PSF before and after slighly poking the actuator
        and take the difference between the convolutional layer
        activations

        :param calibration_model:
        :param j_actu:
        :return:
        """

        layer_activations = []
        # rand_coef = np.random.uniform(low=-coef_strength, high=coef_strength, size=N_act)
        for alpha in [0.5, 0.55]:

            # Compute the in-focus and defocused PSF images for a given actuator
            coef_zero = np.zeros(N_act)
            coef_zero[j_actu] = alpha * coef_strength
            PSF, _s = PSF_actuators.compute_PSF(coef_zero)
            PSF_foc, _s = PSF_actuators.compute_PSF(coef_zero + diversity_defocus)
            PSF_array = np.zeros((1, pix, pix, 2))
            PSF_array[:, :, :, 0] = PSF
            PSF_array[:, :, :, 1] = PSF_foc
            img_tensor = PSF_array

            # Calculate the activations
            activations = calculate_activations(calibration_model, img_tensor, False)
            layer_activations.append(activations)

        nominal_activation = layer_activations[0]
        new_activation = layer_activations[1]
        N_layers = len(nominal_activation)
        layer_names = ['Convolutional #%d Differential Output' % (i + 1) for i in range(N_layers)]

        for k in range(N_layers):

            size = nominal_activation[k].shape[1]
            scale = 8. / size
            plt.figure(figsize=(scale * new_activation[k].shape[1], scale * new_activation[k].shape[0]))
            plt.grid(False)
            im = new_activation[k] - nominal_activation[k]
            plt.imshow(im, aspect='auto', cmap='seismic')
            im_max = max(-np.min(im), np.max(im))
            plt.clim(-im_max, im_max)
            plt.title(layer_names[k] + r' | Actuator #%d' % (j_actu + 1))
            plt.axes().xaxis.set_visible(False)
            plt.axes().yaxis.set_visible(False)

        return

    compare_activations(calib_actu, j_actu=0)
    plt.show()

    # ================================================================================================================ #

    ### Average of importance across all Actuator commands
    pos_array_nom = np.zeros((N_act, pix, pix))
    pos_array_foc = np.zeros((N_act, pix, pix))
    for j_act in range(N_act):

        pos_shap_nom, neg_shap_nom = reds_and_blues(shap_values, test_shap_coef, j_act, channel=0)
        pos_shap_foc, neg_shap_foc = reds_and_blues(shap_values, test_shap_coef, j_act, channel=1)
        pos_array_nom[j_act] = pos_shap_nom
        pos_array_foc[j_act] = pos_shap_foc

    # pos_array_nom_noisy = pos_array_nom.copy()
    # pos_array_foc_noisy = pos_array_foc.copy()
    # np.save('pos_array_nom_noisy', pos_array_nom_noisy)
    # np.save('pos_array_foc_noisy', pos_array_foc_noisy)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    MAX = max([np.max(np.mean(x, axis=0)) for x in [pos_array_nom, pos_array_foc, pos_array_nom_noisy, pos_array_foc_noisy]])
    A = np.mean(pos_array_nom, axis=0)
    img1 = ax1.imshow(A * 1e3, cmap='Reds', origin='lower')
    # img1.set_clim(0, 1)
    plt.colorbar(img1, ax=ax1)
    ax1.set_title(r'Mean (+) Shapley [In-focus]')
    ax1.text(5.5, 0.9, 'No Noise', {'color': 'black', 'fontsize': 10, 'ha': 'center', 'va': 'center',
                                    'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})

    B = np.mean(pos_array_foc, axis=0)
    # p0, s0 = PSF_actuators.compute_PSF(np.zeros(N_act))
    img2 = ax2.imshow(B * 1e3, cmap='Reds', origin='lower')
    # img2.set_clim(0, 1)
    plt.colorbar(img2, ax=ax2)
    ax2.set_title(r'Mean (+) Shapley [Defocused]')

    C = np.mean(pos_array_nom_noisy, axis=0)
    img3 = ax3.imshow(C * 1e3, cmap='Reds', origin='lower')
    # img3.set_clim(0, 1)
    plt.colorbar(img3, ax=ax3)
    # ax3.set_title(r'Mean (+) Shapley [In-focus]')
    ax3.text(5.5, 0.9, 'Noise SNR 250', {'color': 'black', 'fontsize': 10, 'ha': 'center', 'va': 'center',
                                    'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})

    D = np.mean(pos_array_foc_noisy, axis=0)
    img4 = ax4.imshow(D * 1e3, cmap='Reds', origin='lower')
    # img4.set_clim(0, 1)
    plt.colorbar(img4, ax=ax4)
    # ax4.set_title(r'Mean (+) Shapley [Defocused]')

    # Hide the pixel coordinates and axes
    # for ax in fig.axes:
    #     hide_axes(ax)
    plt.show()








    nominal_grid = []

    k = 0
    images_per_row = 4
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        print(size)
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                # channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                # channel_image /= channel_image.std()
                # channel_image *= 64
                # channel_image += 128
                # channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image[::-1, :]
        scale = 2. / size
        mask_grid = display_grid == 0.0
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        # plt.imshow(mask_grid, cmap='Reds')
        im = display_grid - nominal_grid[k]
        plt.imshow(im, aspect='auto', cmap='seismic')
        im_max = max(-np.min(im), np.max(im))
        plt.clim(-im_max, im_max)
        k += 1
        # nominal_grid.append(display_grid)
    plt.show()

    # Noise
    # Add some Readout Noise to the PSF images to spice things up
    # SNR_READOUT = 750
    SNR = 250
    noise_model = noise.NoiseEffects()
    train_PSF_readout = noise_model.add_readout_noise(train_PSF, RMS_READ=1./SNR)
    test_PSF_readout = noise_model.add_readout_noise(test_PSF, RMS_READ=1./SNR)

    losses = calib_actu.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                N_loops=1, epochs_loop=epochs, verbose=1, batch_size_keras=32,
                                                plot_val_loss=False, readout_noise=False, RMS_readout=[1. / SNR], readout_copies=3)
