import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import emcee
from multiprocessing import Pool, cpu_count
from models.lcdm.likelihoods import *
ncpu = cpu_count()


def log_prior(theta):
    H_0, Om, M = theta
    if 0.25 < Om < 0.35 and 50. < H_0 < 100. and  -80 < M < 20:
        return 0.0
    else:
        return -np.inf

def log_probability(theta):
    H_0, Om, M = theta
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    n_interpolation = 100
    z_min = 0.0
    z_max = 2.4
    z_int = np.linspace(z_min, z_max, n_interpolation)
    Hz = solver(H_0, Om,  z_int)
    d_H = c / Hz
    integrand = interpolate.interp1d(z_int, d_H)
    d_C = []
    for j in range(len(z_int)):
        d_C.append(integrate.romberg(integrand, 0, z_int[j], divmax=20))
    lp = lp   + Likelihood_Pantheon().loglkl(theta, z_int, Hz, d_C) + Likelihood_clocks().loglkl(theta, z_int, Hz) \
         + Likelihood_BAO_Alam().loglkl(theta,z_int, Hz,d_C) + Likelihood_BAO_DR14().loglkl(theta,z_int, Hz,d_C) + Likelihood_bao_boss().loglkl(theta, z_int, Hz, d_C)
    if np.isnan(lp) == True:
        return -np.inf
    return lp

ndim, nwalkers = 3, 20  #25
niter = 100000#20
p0 = [[74., 0.3,  -19] + [5., 5.e-3,  20] * np.random.randn(ndim) for i in range(nwalkers)]

if __name__ == '__main__':
    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        print('\nRunning burn-in...')
        p0, _, _ = sampler.run_mcmc(p0, 50, progress=True)
        sampler.reset()
        print('Running production...')
        old_tau = np.inf
        for sample in sampler.sample(p0, iterations = niter, progress=True):
            if sampler.iteration % 100:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau*100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau)/ tau < 0.01)
            if converged:
                break
            old_tau = tau
        np.savetxt('lcdm_cc_sn_bao_chain.txt', sampler.get_chain(flat=True))
        np.savetxt('lcdm_cc_sn_bao_log_prob.txt', sampler.get_log_prob(flat=True))
