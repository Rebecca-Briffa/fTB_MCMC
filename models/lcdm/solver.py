from scipy import optimize
import numpy as np
from scipy import interpolate
import math
from mpmath import *
import time



def solver(H_0, Om, z):
    h = H_0 / 100
    OR = 4.1534 * (10 ** (-5)) / h ** 2
    sqrt_H = H_0**2*(Om*(1. + z)**3 + OR*(1. + z)**4 + (1. - Om - OR))
    # catch error here
    if any(H < 0 for H in sqrt_H):
        return -1.e-60
    else:
        H = np.sqrt(sqrt_H)
        return H
