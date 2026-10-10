from scipy import optimize
import numpy as np
from scipy import interpolate
import math
from mpmath import *
import time

def Hz(y, z, H_0, Om, p):
    H, dH = y
    dH_0 = 40
    d2H_0 = 40
    T = -6 * H ** 2
    # print(T)
    B = -6 * ((3 * H ** 2) - (1 + z) * H * dH)
    B0 = -6 * ((3 * H_0 ** 2) - (H_0 * dH_0))
    # print(B, B0)

    Bdot_0 = -6 * ((-5 * H_0 ** 2 * dH_0 )+ (H_0 * dH_0 ** 2) + (H_0 ** 2 * d2H_0))
    h = H_0 / 100
    # define omega_radiation as from Planck 18
    OR = 4.1534 * (10 ** (-5)) / h ** 2
    alpha = (6 * H_0 ** 2 * mp.exp(p) * (Om + OR - 1)) / (-1 - (p / 2) + (6 * H_0 * p * (1 + p) * Bdot_0) / (4 * B0 ** 2))
    #print(alpha)

    part1 = (-mp.exp(p * mp.sqrt(B / B0))*B0*B) / (9 * (1 + z) ** 2 * H ** 3 * alpha * p * (p + mp.sqrt(B0/B)))
    part2 = (6 * H_0 ** 2 * (Om * (1 + z) ** 3 + OR * (1 + z) ** 4)) + T
    part3 = alpha * mp.exp(-p * mp.sqrt(B / B0))
    part4 = 1 + (p / 2 * mp.sqrt(B / B0)) - (54 * p * H ** 3 * dH * (1 + z) * ((1 / mp.sqrt(B0 * B ** 3)) + (p / (B * B0))))
    part5 = -dH / (1 + z) - dH ** 2 / H
    dHdz = [dH, part1 * (part2 + (part3 * part4)) + part5]

    return dHdz

def solver(H_0, Om, q, z):
    # H_0, dH_0, d2H_0, Om, q =
    dH_0 = 40
    y0 = [H_0, dH_0]

    t1 = time.time()
    f = odefun(lambda z, y: Hz(y, z, H_0, Om, q), 0, y0, tol = 0.01)
    t2 = time.time()
    # print('solver', t2-t1)
    HH = np.zeros((len(z),2))
    H = np.zeros(len(z))

    for i in range(0, len(z)):

        #print(z[i], f(z[i]))
        complex_H = mp.mpmathify(f(z[i])[0])
        H_r = np.real(complex_H)
        H_i = np.imag(complex_H)
       # print(H_i)
        if H_i != 0:
            H = "Complex"
            break
        elif H_r < 0:
            H[i] = -1.e-60
            H = "Negative H"
            break
        else:
            # HH[i] = f(z[i])
            H[i] = H_r
    return H
