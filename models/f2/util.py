from models.f2.solver import *
from pandas import read_table
from scipy import interpolate
import time

# Obtain an approx for the comoving sound horizon at the baryon drag epoch (https://arxiv.org/pdf/2106.00428.pdf)
def r_drag(H_0 ,Om):

    h = H_0 / 100
    O_b = 0.02166/ (h**2)
    w_b = O_b*h*h
    w_m = Om*h*h

    a1 = 0.00257366
    a2 = 0.05032
    a3 = 0.013
    a4 = 0.7720642
    a5 = 0.24346362
    a6 = 0.00641072
    a7 = 0.5350899
    a8 = 32.7525
    a9 = 0.315473

    rs_den = (a1*(w_b**a2) + (a3*(w_b**a4)*(w_m**a5)) + (a6*w_m**a7))
    rs = 1/rs_den - (a8/w_m**a9)
    return rs


# the speed of light in km/s^2
c_g = 299792458.  # speed of light in m/s
c = c_g / 1000.  # speed of light in km/s



# calculates the luminosity distance D_L = D_C * (1+z)
def distances(z, Hz, z_int, d_C):

    dC_interpolation = interpolate.interp1d(z_int, d_C)
    # obtaining the function for d_C and then obtaining values at each z
    t12 = time.time()
    Dc = dC_interpolation(z)

    d_L = Dc * (1 + z)

    # calculates angular diameter distance d_A
    d_A = Dc / (1 + z)

    # calculates the comoving angular diameter D_M = D_A * (1+z)
    d_M = d_A * (1+z)
    t13 = time.time()
    # print('distances', t13 - t12, H_0, dH_0, d2H_0, Om, q)
    return  d_A, d_M,  d_L

#
def v_weight_eff(z, Hz, z_int, d_C):
    Hz_inter = interpolate.interp1d(z_int, Hz)
    H_z = Hz_inter(z)
    cosmo_dist = distances(z, Hz, z_int, d_C)

    d_M = cosmo_dist[1]
    d_V = pow(d_M * d_M * c * z / H_z, 1. / 3.)
    return d_V

# read the matrix for the Supernoave data
def read_matrix(path):
    with open(path, 'r') as text:
        length = int(text.readline())
    matrix = read_table(path).to_numpy().reshape((length, length))
    return matrix


def read_light_curve_parameters(path):

    with open(path, 'r') as text:
        clean_first_line = text.readline()[1:].strip()
        names = [e.strip().replace('3rd', 'third')
                 for e in clean_first_line.split( )]

    lc_parameters = read_table(
        path, sep=' ', names=names, header=0, index_col=False)
    return lc_parameters
