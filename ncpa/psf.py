
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import zernike as zern



def plot_actuators(centers, rho_aper, rho_obsc, zoom=True):
    """
    Plot the actuator centres to visualize the model
    :param centers: list of [x, y] coordinates for the actuator centres
    :return:
    """
    N_act = len(centers[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    circ1 = Circle((0, 0), rho_aper, linestyle='--', fill=None)
    circ2 = Circle((0, 0), rho_obsc, linestyle='--', fill=None)
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    for c in centers[0]:
        ax.scatter(c[0], c[1], color='red')
    ax.set_aspect('equal')
    if zoom is True:
        plt.xlim([-1.25 * rho_aper, 1.25 * rho_aper])
        plt.ylim([-1.25 * rho_aper, 1.25 * rho_aper])
    if zoom is False:
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
    plt.title('Actuator Centres | %d actuators' % N_act)
    return

def actuator_centres(N_actuators, rho_aper, rho_obsc, radial=True):
    """
    Computes the (Xc, Yc) coordinates of actuator centres
    inside a circle of rho_aper, assuming there are N_actuators
    along the [-1, 1] line

    :param N_actuators: Number of actuators along the [-1, 1] line
    :param rho_aper: aperture radius relative to [-1, 1]
    :param rho_obsc: radius of central obscuration
    :param radial: whether to include actuators at the boundaries
    :return:
    """

    x0 = np.linspace(-1., 1., N_actuators, endpoint=True)
    delta = x0[1] - x0[0]
    N_in_D = 2*rho_aper/delta
    print("\nCalculating Actuator Centres")
    print("%.2f actuators per Diameter" % N_in_D)        # How many actuators per diameter
    xx, yy = np.meshgrid(x0, x0)
    x_f = xx.flatten()
    y_f = yy.flatten()

    act = []
    for x_c, y_c in zip(x_f, y_f):
        r = np.sqrt(x_c ** 2 + y_c ** 2)
        if r < rho_aper - delta/2 and r > rho_obsc + delta/2:
            act.append([x_c, y_c])

    if radial:
        for r in [rho_aper, rho_obsc]:
            N_radial = int(np.floor(2*np.pi*r/delta))
            d_theta = 2*np.pi / N_radial
            theta = np.linspace(0, 2*np.pi - d_theta, N_radial)
            # Super important to do 2Pi - d_theta to avoid placing 2 actuators in the same spot... Degeneracy!
            for t in theta:
                act.append([r*np.cos(t), r*np.sin(t)])

    total_act = len(act)
    print('Total Actuators: ', total_act)
    return act, delta

def actuator_centres_multiwave(N_actuators, rho_aper, rho_obsc,
                               N_waves, wave0, waveN, wave_ref, radial=True):
    """
    Computes the (Xc, Yc) coordinates of actuator centres
    inside a circle of rho_aper, assuming there are N_actuators
    along the [-1, 1] line

    Multiwave version of actuator_centres. It computes the centres for each
    wavelength in the interval [wave0, ..., waveN]. The assumption is that
    rho_aper, rho_obsc are the radii that give us the Spaxel Scale of interest
    at a reference wavelength wave_ref

    The code rescales the apertures with respect to wave_ref to account for wavelength scaling
    in the PSF, while keeping the actuator model identical

    :param N_actuators: Number of actuators along the [-1, 1] line
    :param rho_aper: Relative size of the aperture wrt [-1, 1] for reference wavelength
    :param rho_obsc: Relative size of the obscuration
    :param N_waves: how many wavelengths to consider
    :param wave0: shortest wavelength [microns]
    :param waveN: longest wavelength [microns]
    :param wave_ref: reference wavelength at which we have a given Spaxel Scale (typically 1.5 microns)
    :param radial: if True, we add actuators at the boundaries rho_aper, rho_obsc
    :return: [act (list of actuator centres), delta (actuator separation)] for each wavelength
    """
    print("\nCalculating Actuator Centres")
    print("Wavelenght range: [%.2f, ..., %.2f] microns | %d wavelengths" % (wave0, waveN, N_waves))
    print("Reference wavelength: %.2f microns" % wave_ref)
    waves = np.linspace(wave0, waveN, N_waves, endpoint=True)
    waves_ratio = waves / wave_ref
    centres = []
    for wave_r in waves_ratio:
        x0 = np.linspace(-1./wave_r, 1./wave_r, N_actuators, endpoint=True)
        delta = x0[1] - x0[0]
        xx, yy = np.meshgrid(x0, x0)
        x_f = xx.flatten()
        y_f = yy.flatten()

        act = []    # List of actuator centres (Xc, Yc)
        for x_c, y_c in zip(x_f, y_f):
            r = np.sqrt(x_c ** 2 + y_c ** 2)
            if r < (rho_aper / wave_r - delta / 2) and r > (rho_obsc / wave_r + delta / 2):   # Leave some margin close to the boundary
                act.append([x_c, y_c])

        if radial:  # Add actuators at the boundaries, keeping a constant angular distance
            for r in [rho_aper / wave_r, rho_obsc / wave_r]:
                N_radial = int(np.floor(2*np.pi*r/delta))
                d_theta = 2*np.pi / N_radial
                theta = np.linspace(0, 2*np.pi - d_theta, N_radial)
                # Super important to do 2Pi - d_theta to avoid placing 2 actuators in the same spot... Degeneracy!
                for t in theta:
                    act.append([r*np.cos(t), r*np.sin(t)])

        centres.append([act, delta])
    return centres

def actuator_matrix(centres, alpha_pc, rho_aper, rho_obsc, N_PIX):
    """
    Computes the matrix containing the Influence Function of each actuator
    Returns a matrix of size [N_PIX, N_PIX, N_actuators] where each [N_PIX, N_PIX, k] slice
    represents the effect of "poking" one actuator

    Current model: Gaussian function

    :param centres: [act, delta] list from actuator_centres containing the centres and the spacing
    :param alpha: scaling factor to control the tail of the Gaussian and avoid overlap / crosstalk
    :param rho_aper: Relative size of the aperture wrt [-1, 1] for reference wavelength
    :param rho_obsc: Relative size of the obscuration
    :return: a list containing [actuator matrix, pupil_mask, flattened actuator_matrix[pupil_mask]
    """

    print("\nCalculating Actuator Matrix (Influence Functions)")
    print("Gaussian Model")

    # alpha_pc is the height of the Gaussian at a distance of 1 Delta in %
    alpha = 1/np.sqrt(np.log(100/alpha_pc))

    # Read the coordinates of actuator centres (cent) and the actuator sampling (delta)
    cent, delta = centres
    N_act = len(cent)
    matrix = np.empty((N_PIX, N_PIX, N_act))
    x0 = np.linspace(-1., 1., N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x0, x0)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    pupil = (rho <= rho_aper) & (rho >= rho_obsc)

    for k in range(N_act):
        xc, yc = cent[k][0], cent[k][1]
        r2 = (xx - xc) ** 2 + (yy - yc) ** 2
        matrix[:, :, k] = pupil * np.exp(-r2 / (alpha * delta) ** 2)        # Modify this to use other model

    mat_flat = matrix[pupil]

    return matrix, pupil, mat_flat


def actuator_matrix_multiwave(centres, alpha_pc, rho_aper, rho_obsc,
                              N_waves, wave0, waveN, wave_ref, N_PIX):
    """
    Multiwavelength equivalent of actuator matrix
    Computes the matrix containing the Influence Function of each actuator
    Creatres a matrix of size [N_PIX, N_PIX, N_actuators] where each [N_PIX, N_PIX, k] slice
    represents the effect of "poking" one actuator

    All of this is repeated for each wavelength in [wave0, ..., waveN]

    :param centres: output of actuator_centres_multiwave
    :param alpha_pc: [%] height of the Gaussian at the neighbour actuator
    :param rho_aper: Relative size of the aperture wrt [-1, 1] for reference wavelength
    :param rho_obsc: Relative size of the obscuration
    :param N_waves: how many waves to consider (it should match the N_waves used in actuator_centres_multiwave
    :param wave0: shortest wavelength [microns]
    :param waveN: longest wavelength [microns]
    :param wave_ref: reference wavelength for the particular Spaxel Scale being used
    :param N_PIX: Number of pixels to use in the arrays
    :return: a list of lists [[actuator_matrix, pupil_mask, flattened actuator_matrix[pupil_mask]] for each wavelength]
    """

    print("\nCalculating Actuator Matrix (Influence Functions)")
    print("Gaussian Model")
    print("Wavelenght range: [%.2f, ..., %.2f] microns | %d wavelengths" % (wave0, waveN, N_waves))
    print("Reference wavelength: %.2f microns" % wave_ref)

    waves = np.linspace(wave0, waveN, N_waves, endpoint=True)
    waves_ratio = waves / wave_ref

    # alpha_pc is the height of the Gaussian at a distance of 1 Delta in %
    alpha = 1/np.sqrt(np.log(100/alpha_pc))

    matrices = []
    for i, wave in enumerate(waves_ratio):      # Loop over the wavelengths

        cent, delta = centres[i]
        N_act = len(cent)
        matrix = np.empty((N_PIX, N_PIX, N_act))
        x0 = np.linspace(-1., 1., N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x0, x0)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        pupil = (rho <= rho_aper/wave) & (rho >= rho_obsc/wave)

        for k in range(N_act):
            xc, yc = cent[k][0], cent[k][1]
            r2 = (xx - xc) ** 2 + (yy - yc) ** 2
            matrix[:, :, k] = pupil * np.exp(-r2 / (alpha * delta) ** 2)

        mat_flat = matrix[pupil]
        matrices.append([matrix, pupil, mat_flat])

    return matrices


def zernike_matrix(N_levels, rho_aper, rho_obsc, N_PIX, radial_oversize=1.0):
    """
    Equivalent of actuator_matrix but for a Zernike wavefront model
    It computes the Zernike Model Matrix [N_PIX, N_PIX, N_ZERN]

    By default it removes piston and tilt terms

    :param N_levels: how many Zernike pyramid levels to consider
    :param rho_aper:
    :param rho_obsc:
    :param N_PIX:
    :param radial_oversize:
    :return:
    """

    # Calculate how many Zernikes we need, to reach N_levels
    levels = zern.triangular_numbers(N_levels)
    N_zern = levels[N_levels]


    x = np.linspace(-1, 1, N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
    pupil = (rho <= rho_aper) & (rho >= rho_obsc)

    rho, theta = rho[pupil], theta[pupil]
    zernike = zern.ZernikeNaive(mask=pupil)
    _phase = zernike(coef=np.zeros(N_zern), rho=rho / (radial_oversize * rho_aper), theta=theta, normalize_noll=False,
                     mode='Jacobi', print_option='Silent')
    H_flat = zernike.model_matrix[:, 3:]
    H_matrix = zern.invert_model_matrix(H_flat, pupil)
    N_zern = H_matrix.shape[-1]         # Update the total number of Zernikes

    print("\nCalculating Zernike Wavefront Matrix (Influence Functions)")

    return H_matrix, pupil, H_flat


