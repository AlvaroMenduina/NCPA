"""
Date: 20th November 2019

Comparison of Basis Function for Wavefront Definition in the context of ML calibration
________________________________________________________________________________________

Is there a basis that works better for calibration?
Is Zernike polynomials better-suited than Actuator Commands models?




"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

import psf
import utils
import calibration
import noise

