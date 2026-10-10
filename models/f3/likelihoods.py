import numpy as np
from numpy import loadtxt
from scipy import integrate
from models.f3.util import *
from models.f3.paths import data_file
import scipy.linalg as la
import numexpr as ne
from pandas import read_table
from scipy import interpolate
import time

class Likelihood_clocks:
    '''
    Cosmic chronometers likelihood
    '''

    def __init__(self):
                # Text file for Cosmic Chronometer (CC) data
                H_file = data_file('H_' + 'CC' + 'noH0' + '.dat')

                # load the CC H(z) data
                (z_H_dat, H_z_mean, H_z_err) = loadtxt(H_file, unpack='True')

                # Store the columns separately
                self.z = z_H_dat
                self.Hz = H_z_mean
                self.err = H_z_err


    def loglkl(self, theta, z_int, Hz):
        # define theta as the free parameters
        H_0, Om, q, M = theta

        # computing chi2 by first starting with chi2 = 0 and then adding to this chi2
        chi2 = 0


        # compute the interpolation at the chosen redshifts
        Hz_inter = interpolate.interp1d(z_int, Hz)
        # obtain 31 theoretical points
        H_z = Hz_inter(self.z)

        #compute chi**2
        for index, z in enumerate(self.z):
            chi2 += (self.Hz[index] - H_z[index]) ** 2 / self.err[index] ** 2

        return -0.5 * chi2


# obtain the covariance matrix for the Pantheon file
C00_init = read_matrix(data_file('covfile.txt'))
light_curve_parameters_init = read_light_curve_parameters(data_file('lcparam_full_long.txt'))

class Likelihood_Pantheon():
    '''
    Pantheon Supernovae dataset
    '''
    def __init__(self):

        # Load matrices from text files, whose names were read in the
        # configuration file#
        self.C00 = C00_init

        # Reading light-curve parameters from self.data_file (lcparam_full_long.txt)
        self.light_curve_params = light_curve_parameters_init

        C00 = self.C00
        covm = ne.evaluate("C00")

        sn = self.light_curve_params

        # Update the diagonal terms of the covariance matrix with the
        # statistical error
        covm += np.diag(sn.dmb ** 2)

        # Whiten the residuals, in two steps.
        # Step 1) Compute the Cholesky decomposition of the covariance matrix, in
        # place. This is a time expensive (0.015 seconds) part, which is why it is
        # now done in init. Note that this is different to JLA, where it needed to
        # be done inside the loglkl function.
        self.cov = la.cholesky(covm, lower=True, overwrite_a=True)
        # Step 2) depends on point in parameter space -> done in loglkl calculation

    def loglkl(self, theta, z_int, Hz, d_C):
        H_0, Om, q, M = theta

        # obtain the redshifts
        z_SN = self.light_curve_params.zcmb

        red_size = z_SN.size

        # obtain the luminosity distance
        cosmo_dist = distances(z_SN, Hz, z_int, d_C)
        lumin_dist = cosmo_dist[2]


       #compute the moduli to be able to compare with data
        moduli = 5 * np.log10(lumin_dist) + 25

        # Compute the residuals (estimate of distance moduli - exact moduli)
        residuals = np.empty((red_size,))
        sn = self.light_curve_params
        # This operation loops over all supernovae!
        # Compute the approximate moduli
        residuals = sn.mb - M
        # Remove from the approximate moduli the one computed from CLASS
        residuals -= moduli

        # Step 2) (Step 1 is done in the init) Solve the triangular system, also time expensive (0.02 seconds)
        residuals = la.solve_triangular(self.cov, residuals, lower=True, check_finite=False)

        # Finally, compute the chi2 as the sum of the squared residuals
        chi2 = (residuals ** 2).sum()

        return -0.5 * chi2


class Likelihood_BAO_Alam:

    '''
    BOSS data (DR12 galaxy sample) - from Alam et.al - https://arxiv.org/abs/1607.03155
    '''

    def __init__(self):

        # define arrays for values of z and data points
        self.z_Alam = np.array([], 'float64')
        self.dM_Alam = np.array([], 'float64')
        self.H_Alam = np.array([], 'float64')


        # read redshifts and data points for D_M and H
        with open(data_file('BAO_consensus_results_dM_Hz.txt'), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    # load redshifts and D_M * (r_s / r_s_fid)^-1 in Mpc
                    if this_line[1] == 'dM(rsfid/rs)':
                        self.z_Alam = np.append(self.z_Alam, float(this_line[0]))
                        self.dM_Alam = np.append(self.dM_Alam, float(this_line[2]))
                    # load H(z) * (r_s / r_s_fid) in km s^-1 Mpc^-1
                    elif this_line[1] == 'Hz(rs/rsfid)':
                        self.H_Alam = np.append(
                            self.H_Alam, float(this_line[2]))

        # read covariance matrix
        self.covdata = np.loadtxt(data_file('BAO_consensus_covtot_dM_Hz.txt'))

        # computing chi2

    def loglkl(self, theta, z_int, Hz, d_C):
        H_0, Om, q, M = theta
        # load the value for r_drag

        r_s = r_drag(H_0, Om)
        # define r_fiducial
        r_fid = 147.78

        # Obtain D_M in Mpc
        DM = self.dM_Alam
        # Obtain H(z) in in km s^-1 Mpc^-1
        H_z = self.H_Alam

        # obtain the comoving angular diameter
        cosmo_dist = distances(self.z_Alam, Hz, z_int, d_C)
        D_M = cosmo_dist[1]

        # compute the interpolation at the chosen redshifts
        Hz_interpolate = interpolate.interp1d(z_int, Hz)

        # obtain the theoretical points
        H_cosmo = Hz_interpolate(self.z_Alam)
        H_theo = H_cosmo * (r_s / r_fid)


        # calculating the theoretical comoving angular distance diameter
        theo_DM = D_M * (r_fid / r_s)
        # print(theo_DM)

        data_array = np.array([], 'float64')
        for index, z in enumerate(self.z_Alam):

            # calculate difference between the sampled point and observations for D_M and H (to be used in chi2)
            DM_diff = theo_DM[index] - DM[index]
            H_diff = H_theo[index] - H_z[index]
            # appending to the data array in order s.t (DM, H, DM, H,...)
            data_array = np.append(data_array, DM_diff)
            data_array = np.append(data_array, H_diff)

        cov_inv = np.linalg.inv(self.covdata)
        chi2 = np.dot(np.dot(data_array, cov_inv), data_array)

        return -0.5 * chi2


class Likelihood_BAO_DR14:
    '''
    eBOSS data (DR14 quasars sample) - from Zhao et. al - https://arxiv.org/abs/1801.03043
    '''

    def __init__(self):

       # define arrays for values of z and data points
        self.z_DR14 = np.array([], 'float64')
        self.dA_DR14 = np.array([], 'float64')
        self.H_DR14 = np.array([], 'float64')


        # read redshifts and data points
        with open(data_file('BAO_DR14Q_results_dA_Hz.txt'), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    # load redshifts and D_A * (r_s / r_s_fid)^-1 in Mpc
                    if this_line[1] == 'dA(rsfid/rs)':
                        self.z_DR14= np.append(self.z_DR14, float(this_line[0]))
                        self.dA_DR14 = np.append(
                            self.dA_DR14, float(this_line[2]))
                    # load H(z) * (r_s / r_s_fid) in km s^-1 Mpc^-1
                    elif this_line[1] == 'Hz(rs/rsfid)':
                        self.H_DR14 = np.append(
                            self.H_DR14, float(this_line[2]))

        # read correlation matrix
        rho_BAO = np.loadtxt(data_file('BAO_DR14Q_covtot_dA_Hz.txt'))

        # 1 sigma of DA and H in that order
        H_DA_sigma = [284.93, 14.63, 159.67, 12.42, 96.59, 12.75, 146.46, 14.79]

        # compute the covariance matrix
        self.BAO_cov = [[0 for x in range(len(H_DA_sigma))] for y in range(len(H_DA_sigma))]
        for ii in range(0, len(H_DA_sigma)):
            for jj in range(0, len(H_DA_sigma)):
                self.BAO_cov[ii][jj] = H_DA_sigma[ii] * rho_BAO[ii][jj] * H_DA_sigma[jj]


 # computing chi2

    def loglkl(self, theta, z_int, Hz, d_C):
        H_0, Om, q, M = theta
        # load the value for r_drag
        r_s = r_drag(H_0, Om)

        # define r_fiducial
        r_fid = 147.78

        # Obtain D_A in in Mpc from data
        DA = self.dA_DR14
        # Obtain H(z) in in km s^-1 Mpc^-1 from data
        H_z = self.H_DR14

        data_array = np.array([], 'float64')

        # obtain the angular diameter distance d_A
        cosmo_dist = distances(self.z_DR14, Hz, z_int, d_C)
        D_A = cosmo_dist[0]
        theo_DA = D_A * (r_fid / r_s)

        # compute the interpolation at the chosen redshifts
        Hz_interpolate = interpolate.interp1d(z_int, Hz)
        # obtain the theoretical points
        H_cosmo = Hz_interpolate(self.z_DR14)
        H_theo = H_cosmo * (r_s / r_fid)


        for index, z in enumerate(self.z_DR14):
            # calculate difference between the sampled point and observations for D_M and H (to be used in chi2)
            DA_diff = theo_DA[index] - DA[index]
            H_diff = H_theo[index] - H_z[index]

            # appending to the data array in order s.t (DA, H, DA, H,...)
            data_array = np.append(data_array, DA_diff)
            data_array = np.append(data_array, H_diff)

        # compute the inverse of the covariance matrix
        cov_inv = np.linalg.inv(self.BAO_cov)
        # compute chi^2
        chi2 = np.dot(np.dot(data_array, cov_inv), data_array)

        return -0.5 * chi2
class Likelihood_bao_boss:
    '''
    6DF, Beutler et. al 2011 - https://academic.oup.com/mnras/article/416/4/3017/976636
    SDSS DR7 MGS, Ross et al. - 1409.3242v1
    SDSS III Lymann alpha, - from Bourbous et. al - https://arxiv.org/abs/1708.02225
    '''

    def __init__(self):

        # define array for values of z and data points
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        self.type = np.array([], 'int')

        # read redshifts and data points
        with open(data_file('bao_2014.txt'), 'r') as filein:
            for line in filein:
                if line.strip() and line.find('#') == -1:
                    # the first entry of the line is the identifier
                    this_line = line.split()

                    if this_line[0] == '6DF' or 'MGS' or 'Lya':
                        self.z = np.append(self.z, float(this_line[1]))
                        self.data = np.append(self.data, float(this_line[2]))
                        self.error = np.append(self.error, float(this_line[3]))
                        self.type = np.append(self.type, int(this_line[4]))

        # number of data points
        self.num_points = np.shape(self.z)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, theta, z_int, Hz, d_C):
        H_0, Om, q, M = theta
        chi2 = 0.


        # compute the interpolation at the chosen redshifts
        Hz_interpolate = interpolate.interp1d(z_int, Hz)
        # obtain the theoretical points
        H_cosmo = Hz_interpolate(self.z)
        # for each point, compute angular distance da,
        # volume distance dv, sound horizon at baryon drag rs_d,
        # theoretical prediction and chi2 contribution
        for i in range(self.num_points):
            t5 = time.time()
            dv = v_weight_eff(self.z[i], Hz, z_int, d_C)
            t6 = time.time()
            # print('vol eff time', t6-t5)
            c_g = 299792458.  # speed of light in m/s
            c = c_g / 1000.  # speed of light in km/s
            dH = c / H_cosmo[i]
            rs = r_drag(H_0, Om)

            if self.type[i] == 1:
                theo = dv / rs

            elif self.type[i] == 2:
                theo = rs / dv

            elif self.type[i] == 3:
                theo = dH / rs

            chi2 += ((theo - self.data[i]) / self.error[i]) ** 2

            return -0.5 * chi2

def likelihood_H0_Riess21(theta, z_int, Hz):
    H_0, Om, q, M = theta
    ''' Riess21 H_0 prior'''
    # Defining the H0_prior, H_0 = 73.30 +- 1.04 km/s/Mpc  (r21 prior - arxiv:2112.04510)

    chi2 = 0
    z = 0

    # compute the interpolation at the chosen redshifts
    Hz_inter = interpolate.interp1d(z_int, Hz)
    # obtain 31 theoretical points
    H_z = Hz_inter(z)
    H0_R = 73.30
    H0_R_err = 1.04

    chi2 += (H_z - H0_R)**2/ H0_R_err**2


    return -0.5*chi2


def likelihood_H0_W21(theta, z_int, Hz):
    H_0, Om, q, M = theta
    ''' H_0 prior'''
    # Defining the H0_prior, H_0 = 69.8 +- 1.7 km/s/Mpc  (W21 prior - arxiv:2106.15656)

    chi2 = 0
    z = 0

    # compute the interpolation at the chosen redshifts
    Hz_inter = interpolate.interp1d(z_int, Hz)
    # obtain 31 theoretical points
    H_z = Hz_inter(z)
    H0_W = 69.8
    H0_W_err = 1.7

    chi2 += (H_z - H0_W)**2/ H0_W_err**2


    return -0.5*chi2
