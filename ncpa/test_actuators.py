"""
A Test to show how to use the Actuator Model and Zernike Model to generate Wavefront maps


"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import psf
import utils


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    WAVE = 1.5          # microns | reference wavelength
    N_PIX = 1024

    # Select Spaxel Scale
    SPAX = 2.0          # mas | spaxel scale

    RHO_APER = utils.rho_spaxel_scale(spaxel_scale=SPAX, wavelength=WAVE)
    RHO_OBSC = 0.30 * RHO_APER      # ELT central obscuration

    utils.check_spaxel_scale(rho_aper=RHO_APER, wavelength=WAVE)

    # FHWM in [mas] at that wavelength
    FWHM = utils.compute_FWHM(wavelength=WAVE)

    # Actuator Models
    N_act_diam = 8          # how many actuators per diameter
    N_actuators = int(N_act_diam / RHO_APER)

    # Calculate the Actuator Centres
    centers = psf.actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True)
    N_act = len(centers[0])
    psf.plot_actuators(centers, rho_aper=RHO_APER, rho_obsc=RHO_OBSC)
    # plt.show()

    # Calculate the Actuator Model Matrix (Influence Functions)
    alpha_pc = 20       # Height [percent] at the neighbour actuator (Gaussian Model)
    actuator_matrix, pupil_mask, flat_actuator = psf.actuator_matrix(centres=centers, alpha_pc=alpha_pc,
                                                                     rho_aper=RHO_APER, rho_obsc=RHO_OBSC, N_PIX=N_PIX)
    print("Actuator Matrix: ", actuator_matrix.shape)
    print("Pupil Mask: ", pupil_mask.shape)
    print("Flattened Actuator Matrix: ", flat_actuator.shape)

    for i in np.random.choice(N_act, size=5):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(actuator_matrix[:, :, i], extent=[-1, 1, -1, 1], cmap='Reds')
        circ1 = Circle((0, 0), RHO_APER, linestyle='--', fill=None)
        circ2 = Circle((0, 0), RHO_OBSC, linestyle='--', fill=None)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        plt.title("Influence Function | Actuator #%d" % (i + 1))
        plt.xlim([-1.25 * RHO_APER, 1.25 * RHO_APER])
        plt.ylim([-1.25 * RHO_APER, 1.25 * RHO_APER])
    # plt.show()

    # Create a random wavefront [Actuator Model]
    c_act = 1/(2*np.pi) * np.random.uniform(-1, 1, size=N_act)
    phase0 = np.dot(actuator_matrix, c_act)
    p0 = min(np.min(phase0), -np.max(phase0))

    plt.figure()
    plt.imshow(phase0, extent=(-1, 1, -1, 1), cmap='bwr')
    plt.colorbar()
    plt.clim(p0, -p0)
    for c in centers[0]:
        plt.scatter(c[0], c[1], color='black', s=4)
    plt.xlim([-1.1*RHO_APER, 1.1*RHO_APER])
    plt.ylim([-1.1*RHO_APER, 1.1*RHO_APER])
    plt.title(r'%d Actuators | Wavefront [$\lambda$]' % N_act)
    # plt.show()

    # Create a random wavefront [Actuator Model]
    zernike_matrix, pupil_mask_zernike, flat_zernike = psf.zernike_matrix(N_levels=5, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                                                                          N_PIX=N_PIX, radial_oversize=1.0)
    print("Zernike Matrix: ", zernike_matrix.shape)
    print("Pupil Mask: ", pupil_mask_zernike.shape)
    print("Flattened Zernike Matrix: ", flat_zernike.shape)
    N_zern = zernike_matrix.shape[-1]

    for i in range(N_zern):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        image = ax.imshow(zernike_matrix[:, :, i], extent=[-1, 1, -1, 1], cmap='jet')
        circ1 = Circle((0, 0), RHO_APER, linestyle='--', fill=None)
        circ2 = Circle((0, 0), RHO_OBSC, linestyle='--', fill=None)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        plt.xlim([-1.1 * RHO_APER, 1.1 * RHO_APER])
        plt.ylim([-1.1 * RHO_APER, 1.1 * RHO_APER])
        plt.colorbar(image)
    plt.show()



