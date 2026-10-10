from scipy import optimize
import numpy as np
from scipy import interpolate
import math
from mpmath import *
import time


def Hz(y, z, H_0, Om, q):
    #print(z)
    H = y[0]
    dH = y[1]
    T = -6*H**2
    T0 = -6*H_0**2
    dH_0 = 40
    d2H_0 = 40
    #print(T)
    B = -6*((3*H**2) - (1+z)*H*dH)

    #print('B',B)
    B0 = -6*((3*H_0**2) - (H_0*dH_0))

    Bdot_0 = -6*(-5*H_0**2*dH_0 + H_0*dH_0**2 + H_0**2*d2H_0)

    h = H_0 / 100
    # define omega_radiation as from Planck 18
    OR = 4.1534 * (10 ** (-5)) / h ** 2

    # the constant Beta
    alpha1 = (q-1)*(-B0)**(q) + 6*H_0*(q*(q-1)*(-B0)**(q-2)*Bdot_0)
    # print('alpha',alpha1)
    alpha2 = 6*H_0**2*(Om+OR-1) / alpha1
    # print('alpha2', alpha2)

    # Friedmann equation
    part1 = 1/ (36*(1+z)**2*H**3* alpha2*q*(1-q)*(-B)**(q-2))
    # print('part1', part1)
    part2 = 6*H_0**2*(Om*(1+z)**3 + OR*(1+z)**4) + T
    # print('part2', part2)
    part3 = alpha2 *(1-q)*(-B)**q
    # print('part3', part3)
    part4 = -216*H**3*dH*(1+z)*alpha2*q*(q-1)*(-B)**(q-2)
    # print('part4', part4)
    part5 = -dH/(1+z) - dH**2/H
    # print('part5', part5)

    dHdz = [dH, part1*(part2+part3+part4)+part5]

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

    for i in range(0, len(z)):
        #print(z[i], f(z[i]))
        complex_H = mp.mpmathify(f(z[i])[0])
        H_i = np.imag(complex_H)
        #print(H_i)
        if H_i != 0:
            H[i] = -1.e-60
        elif f(z[i])[0] < 0:
            H[i] = -1.e-60
        else:
            HH[i] = f(z[i])
            H[i] = (HH[i,0])
    return H
