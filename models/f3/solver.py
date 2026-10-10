from scipy import optimize
import numpy as np
from scipy import interpolate
import math
from mpmath import *
import time

def Hz(y, z, H_0,  Om, q):
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
    alpha = (6* H_0**2 * (Om + OR -1)) / ((6*H_0*Bdot_0/ B0**2) -1 - mp.ln(q))
    #print(alpha)

    part1 = -B**2/ (36* (1+z)**2 * H**3*alpha)
    part2 = (6 * H_0 ** 2 * (Om * (1 + z) ** 3 + OR * (1 + z) ** 4)) + T
    part3 = (alpha*mp.ln(q*B0/B)) + alpha - (216*H**3*dH*(1+z)*alpha/B**2)
    part4 = -dH / (1 + z) - dH ** 2 / H
    dHdz = [dH, part1 * (part2 + part3) + part4]

    return dHdz

def solver(H_0, Om, q, z):
    # H_0, dH_0, d2H_0, Om, q = theta
    dH_0 = 40
    y0 = [H_0, dH_0]
    t1 = time.time()
    f = odefun(lambda z, y: Hz(y, z, H_0, Om, q), 0, y0, tol = 0.01)
    t2 = time.time()
    # print('solver', t2-t1)
    HH = np.zeros((len(z),2))
    H = np.zeros(len(z))

    t3 =time.time()
    for i in range(0, len(z)):
        # print(z[i], f(z[i]))
        t7= time.time()
        HH[i] = f(z[i])
       # print(i, HH[i])
        # print(HH[i,0])
        t8 = time.time()
        # print('HH', t8-t7)
        if HH[i,0] < 0:
            H[i] = -1.e-60
        else:
            H[i] = (HH[i,0])

            #print(H)
        # print('loop loop', t9-t8)
    t4 = time.time()
    # print('solver loop', t4-t3)
    return H
