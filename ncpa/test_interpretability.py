"""


"""


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


def generate_pixel_ids(N_pixels):
    """
    Generate Labels for each pixel according to their (i,j) index
    """
    central_pix = N_pixels // 2
    x = list(range(N_pixels))
    xx, yy = np.meshgrid(x, x)
    xid = xx.reshape((N_pixels * N_pixels))
    yid = yy.reshape((N_pixels * N_pixels))

    labels = ["(%d, %d)" % (x - central_pix, y - central_pix) for (x, y) in zip(xid, yid)]
    return labels


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # (1) We begin by creating a Zernike PSF model with Defocus as diversity
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=5, rho_aper=RHO_APER,
                                                                          rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.1)
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
    diversity_actuators = 1 / (2 * np.pi) * np.random.uniform(-1, 1, size=N_act)

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

    plt.figure()
    plt.imshow(PSF_actuators.diversity_phase, cmap='RdBu')
    plt.colorbar()
    plt.title(r'Diversity Map | Defocus [rad]')
    plt.show()

    calib_actu = calibration.Calibration(PSF_model=PSF_actuators)
    calib_actu.create_cnn_model(layer_filters, kernel_size, name='NOM_ACTU', activation='relu')
    epochs = 10
    # rms_train = []
    # for k in range(5):

    train_PSF, train_coef, test_PSF, test_coef = calibration.generate_dataset(PSF_actuators, N_train, N_test,
                                                                              coef_strength=coef_strength,
                                                                              rescale=rescale)

    # Add some Readout Noise to the PSF images to spice things up
    # SNR_READOUT = 750
    SNR = 250
    noise_model = noise.NoiseEffects()
    train_PSF_readout = noise_model.add_readout_noise(train_PSF, RMS_READ=1./SNR)
    test_PSF_readout = noise_model.add_readout_noise(test_PSF, RMS_READ=1./SNR)


    losses = calib_actu.train_calibration_model(train_PSF, train_coef, test_PSF, test_coef,
                                                N_loops=3, epochs_loop=5, verbose=1, batch_size_keras=32,
                                                plot_val_loss=False,
                                                readout_noise=True, RMS_readout=[1. / SNR], readout_copies=3)

    # evaluate the RMS performance
    guess_actu = calib_actu.cnn_model.predict(test_PSF_readout)
    residual_coef = test_coef - guess_actu

    RMS = np.zeros(N_test)
    for k in range(N_test):
        res_phase = np.dot(PSF_actuators.model_matrix, residual_coef[k])
        RMS[k] = np.std(res_phase[PSF_actuators.pupil_mask])
    meanRMS = np.mean(RMS)
    # rms_train.append(meanRMS)
    print(meanRMS)

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
    #                                  SHAP values
    # ================================================================================================================ #

    from time import time
    start = time()
    N_background = 50
    N_shap_samples = 50
    clean_background = train_PSF[np.random.choice(N_train, N_background, replace=False)]
    clean_explainer = shap.DeepExplainer(calib_actu.cnn_model, clean_background)

    # Select only the first N_shap from the test set
    test_shap = test_PSF[:N_shap_samples]
    test_shap_coef = test_coef[:N_shap_samples]
    guess_shap_coef = guess_actu[:N_shap_samples]
    clean_shap_values = clean_explainer.shap_values(test_shap)
    total_time = time() - start
    print("Computed SHAP values for %d samples | %d background in: %.1f sec" % (N_shap_samples, N_background, total_time))

    # (1) The impact of pixel values.
    pix_label = generate_pixel_ids(pix)
    for k_chan in [0, 1]:
        j_act = 2
        shap_val_chan = [x[:, :, :, k_chan].reshape((N_shap_samples, pix*pix)) for x in clean_shap_values]
        features_chan = test_shap[:, :, :, k_chan].reshape((N_shap_samples, pix*pix))

        shap.summary_plot(shap_values=shap_val_chan[j_act], features=features_chan,
                          feature_names=pix_label)

    def show_shap_example(shap_values, test_PSF_shap, test_coef, guess_coef, i_exa, j_actu):

        # Compute how that actuator affects the PSF

        shap_val_nom = shap_values[j_actu][i_exa, :, :, 0]
        shap_val_foc = shap_values[j_actu][i_exa, :, :, 1]
        smax_nom = max(-np.min(shap_val_nom), np.max(shap_val_nom))
        smax_foc = max(-np.min(shap_val_foc), np.max(shap_val_foc))
        MAX = max(smax_nom, smax_foc)


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


    for k in range(3):
        show_shap_example(shap_values=clean_shap_values, test_PSF_shap=test_shap, test_coef=test_shap_coef,
                          guess_coef=guess_shap_coef, i_exa=k+6, j_actu=0)
    plt.show()

    eps = 1e-4
    k_act = 0
    act_cent = np.array(centers[0])
    xc, yc = act_cent[k_act]
    k_x = np.argwhere(np.abs(act_cent[:, 0] + xc) < eps)
    k_y = np.argwhere(np.abs(act_cent[:, 1] + yc) < eps)
    k_opp = np.intersect1d(k_x, k_y)[0]
    # k_opp = k_act

    print("Actuator #%d: [%.4f, %.4f]" % (k_act, xc, yc))
    print("Actuator #%d: [%.4f, %.4f]" % (k_opp, act_cent[k_opp][0], act_cent[k_opp][1]))


    # ( ) For a given actuator, we show what features have been determined the predicted value
    # across multiple test PSF images. If the command was positive, we only look at the positive Shapley (in Red)

    j_act = 0       # Which actuator we want to look at
    cc = clean_shap_values[j_act][:, :, :, 0]       # show the In-Focus Shapley
    n_row, n_col = 4, 5
    fig, axes = plt.subplots(n_row, n_col, dpi=150)
    for k in range(n_row * n_col):
        ax = axes.flatten()[k]
        c = cc[k + 20]   # We get the Shapley for that test PSF image
        command = test_shap_coef[k + 20, j_act]      # We get the actuator command for that test PSF image

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

    # ( )

    def reds_and_blues(clean_shap_values, test_shap_coef, j_act, channel):

        reds, blues = [], []
        for k_exa in range(N_shap_samples):
            s_val = clean_shap_values[j_act][k_exa, :, :, channel]
            command = test_shap_coef[k_exa, j_act]
            if command > 0.0:
                # we select the
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

    def average_delta_psf(j_act, channel):

        # Calculate the average change in PSF features when poking that actuator
        diffs = []
        for k in range(50):
            new_coef = np.random.uniform(low=-coef_strength, high=coef_strength, size=N_act)
            if channel == 1:        # the defocus
                new_coef += diversity_defocus
            psf0, s0 = PSF_actuators.compute_PSF(new_coef)
            new_coef[j_act] += 0.01
            psf_actu, sactu = PSF_actuators.compute_PSF(new_coef)
            psf_diff = psf_actu - psf0
            diffs.append(psf_diff)

        mean_diff = np.mean(np.array(diffs), axis=0)
        mean_diff /= np.max(np.abs(mean_diff))

        return mean_diff

    def hide_axes(ax):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    for j_act in [-1, -2]:

        pos_shap_nom, neg_shap_nom = reds_and_blues(clean_shap_values, test_shap_coef, j_act, channel=0)
        pos_shap_foc, neg_shap_foc = reds_and_blues(clean_shap_values, test_shap_coef, j_act, channel=1)
        # pos_array[j_act] = pos_shap_nom

        # Calculate the average change in PSF features when poking that actuator
        mean_diff_nom = average_delta_psf(j_act, channel=0)
        mean_diff_foc = average_delta_psf(j_act, channel=1)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, dpi=100)
        img1 = ax1.imshow(pos_shap_nom, cmap='Reds', origin='lower')
        # plt.colorbar(img1, ax=ax1)
        ax1.set_title(r'Mean (+) Shapley')
        ax1.text(5.5, 0.9, 'In-focus',{'color': 'black', 'fontsize': 10, 'ha': 'center', 'va': 'center',
                                   'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})


        img2 = ax2.imshow(-neg_shap_nom, cmap='Blues', origin='lower')
        # plt.colorbar(img2, ax=ax2)
        ax2.set_title(r'Mean (-) Shapley')

        img3 = ax3.imshow(np.log10(1 + 5*np.abs(mean_diff_nom)), cmap='Reds', origin='lower')
        # img3 = ax3.imshow(np.abs(mean_diff), cmap='Reds', origin='lower')
        # plt.colorbar(img3, ax=ax3)
        ax3.set_title(r'Mean PSF change')

        img4 = ax4.imshow(pos_shap_foc, cmap='Reds', origin='lower')
        # plt.colorbar(img1, ax=ax1)
        ax4.text(5.50, 0.9, 'Defocus',{'color': 'black', 'fontsize': 10, 'ha': 'center', 'va': 'center',
                                   'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})

        img5 = ax5.imshow(-neg_shap_foc, cmap='Blues', origin='lower')
        # plt.colorbar(img2, ax=ax2)

        img6 = ax6.imshow(np.log10(1 + 5*np.abs(mean_diff_foc)), cmap='Reds', origin='lower')

        # Hide the pixel coordinates and axes
        for ax in fig.axes:
            hide_axes(ax)

    plt.show()

    ### Average of importance across all Actuator commands
    pos_array_nom = np.zeros((N_act, pix, pix))
    pos_array_foc = np.zeros((N_act, pix, pix))
    for j_act in range(N_act):

        pos_shap_nom, neg_shap_nom = reds_and_blues(clean_shap_values, test_shap_coef, j_act, channel=0)
        pos_shap_foc, neg_shap_foc = reds_and_blues(clean_shap_values, test_shap_coef, j_act, channel=1)
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

    j_act = 3
    diffs = []
    for k in range(25):
        new_coef = np.random.uniform(low=-coef_strength, high=coef_strength, size=N_act)
        # new_coef += diversity_defocus
        psf0, s0 = PSF_actuators.compute_PSF(new_coef)
        new_coef[j_act] += 0.01
        psf_actu, sactu = PSF_actuators.compute_PSF(new_coef)
        psf_diff = psf_actu - psf0
        diffs.append(psf_diff)

        # diff_max = max(-np.min(psf_diff), np.max(psf_diff))
        # fig, ax1 = plt.subplots(1, 1)
        #
        # im1 = ax1.imshow(psf_diff, cmap='bwr')
        # im1.set_clim(-diff_max, diff_max)
        # plt.colorbar(im1, ax=ax1)
    # plt.show()

    mean_diff = np.mean(np.array(diffs), axis=0)
    mean_diff /= np.max(mean_diff)
    plt.figure()
    plt.imshow(np.log10(1 + np.abs(mean_diff)), cmap='Reds')
    plt.show()

    # correlation between channels?

    for j_act in range(N_act):
        shap_nom = clean_shap_values[j_act][:, :, :, 0].reshape((N_shap_samples, pix**2))
        shap_foc = clean_shap_values[j_act][:, :, :, 1].reshape((N_shap_samples, pix**2))
        corr_shap = np.zeros(N_shap_samples)
        for k in range(N_shap_samples):
            # plt.scatter(shap_nom[k], shap_foc[k], s=3)
            corr_shap[k] = np.corrcoef(shap_nom[k], shap_foc[k])[0, 1]
        plt.hist(corr_shap, bins=15, histtype='step')
    plt.show()

    shsh = np.array(clean_shap_values)[:, :, :, :, 0]
    shsh = shsh.reshape((N_act, -1))
    csh = np.corrcoef(shsh)

    from sklearn.decomposition import PCA

    N_comp = 10
    pca = PCA(n_components=N_comp)
    pca.fit(X=shap_nom)

    for i in range(N_comp):
        comp = pca.components_[i].reshape((pix, pix))
        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(comp, cmap='bwr')
        clim = max(-np.min(comp), np.max(cnomp))
        img.set_clim(-clim, clim)
        plt.colorbar(img, ax=ax)


    # layer_names = []
    # for layer in model.layers[:2]:
    #     layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot


    def calculate_activations(img_tensor, plot=False):

        model = calib_actu.cnn_model
        layer_outputs = [layer.get_output_at(0) for layer in model.layers]  # Extracts the outputs of the top 12 layers
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

        activations = activation_model.predict(img_tensor)
        print(activations[-1])
        first_layer_activation = activations[0]
        print(first_layer_activation.shape)

        layer_names = ['Convolution_1', 'Convolution_2']

        cmap = plt.get_cmap('Reds')
        cmap.set_under('white')

        images_per_row = 4
        grids = []
        k = 0
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
                # plt.figure(figsize=(scale * display_grid.shape[1],
                #                     scale * display_grid.shape[0]))
                # plt.title(layer_name)
                # plt.grid(False)
                # # plt.imshow(mask_grid, cmap='Reds')
                # im = display_grid
                # plt.imshow(im, aspect='auto', cmap=cmap, vmin=np.spacing(0.0))
            k += 1

        return grids

    grids = calculate_activations(img_tensor=test_PSF_readout[0:1], plot=True)
    plt.show()


    def compare_activations(j_act):

        layer_activations = []
        rand_coef = np.random.uniform(low=-coef_strength, high=coef_strength, size=N_act)
        for alpha in [0.0, coef_strength]:

            # Compute the in-focus and defocused PSF images for a given actuator
            coef_zero = np.zeros(N_act)

            coef_zero[j_act] = alpha * coef_strength
            PSF, _s = PSF_actuators.compute_PSF(coef_zero)
            PSF_foc, _s = PSF_actuators.compute_PSF(coef_zero + diversity_defocus)
            PSF_array = np.zeros((1, pix, pix, 2))
            PSF_array[:, :, :, 0] = PSF
            PSF_array[:, :, :, 1] = PSF_foc
            img_tensor = noise_model.add_readout_noise(PSF_array, RMS_READ=1./(2*SNR))

            # Calculate the activations
            activations = calculate_activations(img_tensor, True)
            layer_activations.append(activations)

        nominal_activation = layer_activations[0]
        new_activation = layer_activations[1]
        N_layers = len(nominal_activation)
        layer_names = ['Convolutional #%d Output' % (i + 1) for i in range(N_layers)]
        for k in range(N_layers):

            size = nominal_activation[k].shape[1]
            scale = 8. / size
            plt.figure(figsize=(scale * new_activation[k].shape[1],
                                scale * new_activation[k].shape[0]))
            plt.grid(False)
            # plt.imshow(mask_grid, cmap='Reds')
            im = new_activation[k] - nominal_activation[k]
            plt.imshow(im, aspect='auto', cmap='seismic')
            im_max = max(-np.min(im), np.max(im))
            plt.clim(-im_max, im_max)
            plt.title(layer_names[k] + r' | Actuator #%d' % (j_act + 1))
            plt.axes().xaxis.set_visible(False)
            plt.axes().yaxis.set_visible(False)

        return

    compare_activations(3)
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
