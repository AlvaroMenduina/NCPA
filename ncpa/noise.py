import numpy as np


class NoiseEffects(object):

    """
    A variety of Noise Effects to add to PSF images
    """

    def __init__(self):

        pass

    def add_readout_noise(self, PSF_images, RMS_READ, sigma_offset=5.0):
        """
        Add Readout Noise in the form of additive Gaussian noise
        at a given RMS_READ equal to 1 / SNR, the Signal To Noise ratio

        Assuming the perfect PSF has a peak of 1.0
        :param PSF_images: datacube of [N_samples, pix, pix, 2] nominal and defocus PSF images
        :param RMS_READ: 1/ SNR
        :param sigma_offset: add [sigma_offset] * RMS_READ to avoid having negative values
        :return:
        """

        N_samples, pix, N_chan = PSF_images.shape[0], PSF_images.shape[1], PSF_images.shape[-1]
        PSF_images_noisy = np.zeros_like(PSF_images)
        print("Adding Readout Noise with RMS: %.4f | SNR: %.1f" % (RMS_READ, 1./RMS_READ))
        for k in range(N_samples):
            for j in range(N_chan):
                read_out = np.random.normal(loc=0, scale=RMS_READ, size=(pix, pix))
                PSF_images_noisy[k, :, :, j] = PSF_images[k, :, :, j] + read_out

        # Add a X Sigma offset to avoid having negative values
        PSF_images_noisy += sigma_offset * RMS_READ

        return PSF_images_noisy

    def add_flat_field(self, PSF_images, flat_delta):
        """
        Add Flat Field uncertainties in the form of a multiplicative map
        of Uniform Noise [1 - delta, 1 + delta]

        Such that if delta is 10 %, the pixel sensitivity is known to +- 10 %
        :param PSF_images: datacube of [N_samples, pix, pix, 2] nominal and defocus PSF images
        :param flat_delta: range of the Flat Field uncertainty [1- delta, 1 + delta]
        :return:
        """

        N_samples, pix, N_chan = PSF_images.shape[0], PSF_images.shape[1], PSF_images.shape[-1]
        print(r"Adding Flat Field errors [1 - $\delta$, 1 + $\delta$]: $\delta$=%.3f" % flat_delta)
        # sigma_uniform = flat_delta / np.sqrt(3)

        for k in range(N_samples):
            for j in range(N_chan):
                flat_field = np.random.uniform(low= 1 -flat_delta, high=1 + flat_delta, size=(pix, pix))
                PSF_images[k, :, :, j] *= flat_field
        return PSF_images

    def add_bad_pixels(self, PSF_images, p_bad=0.20, max_bad_pixels=3, BAD_PIX=1.0):

        print("Adding Bad Pixels")

        N_samples, pix, N_chan = PSF_images.shape[0], PSF_images.shape[1], PSF_images.shape[-1]

        # Randomly select which PSF images [nom, foc] will have bad_pixels
        which_images_nom = np.random.binomial(n=1, p=p_bad, size=N_samples)
        index_nom = np.argwhere(which_images_nom == 1)
        which_images_foc = np.random.binomial(n=1, p=p_bad, size=N_samples)
        index_foc = np.argwhere(which_images_foc == 1)

        # Nominal Channel
        for i_nom in index_nom:
            how_many = np.random.randint(low=1, high=max_bad_pixels)        # How many bad pixels?
            for j in range(how_many):                                       # Which (x, y) pixels?
                x_bad, y_bad = np.random.choice(pix, size=1)[0], np.random.choice(pix, size=1)[0]
                PSF_images[i_nom[0], x_bad, y_bad, 0] = BAD_PIX

        # Defocus Channel
        for i_foc in index_foc:
            how_many = np.random.randint(low=1, high=max_bad_pixels)        # How many bad pixels?
            for j in range(how_many):                                       # Which (x, y) pixels?
                x_bad, y_bad = np.random.choice(pix, size=1)[0], np.random.choice(pix, size=1)[0]
                PSF_images[i_foc[0], x_bad, y_bad, 1] = BAD_PIX

        return PSF_images