import emcee
import numpy as np
import os
from scipy.optimize import minimize

def log_prior(theta):
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

def log_likelihood(theta, x, y, yerr):
        m, b, log_f = theta
        model = m * x + b
        sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def run_sampler(m_true, b_true, f_true, parameters, num):
    N = 50
    x = np.sort(10 * np.random.rand(N))
    yerr = 0.1 + 0.5 * np.random.rand(N)
    y = m_true * x + b_true
    y += np.abs(f_true * y) * np.random.randn(N)
    y += yerr * np.random.randn(N)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([m_true, b_true, np.log(f_true)]) + 0.1 * np.random.randn(3)
    soln = minimize(nll, initial, args=(x, y, yerr))

    pos = soln.x + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
    sampler.run_mcmc(pos, 10000, progress=True)
    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)
    np.savetxt(
        "emcee_output/chain_{}.dat".format(num), samples, delimiter="\t",
        header="\t".join(parameters),
    )

nchains = 8
m_true = -0.9594
b_true = 4.294
f_true = 0.534
header = ["m", "b", "logf"]
os.makedirs("emcee_output", exist_ok=True)
np.savetxt(
    "emcee_output/injected.txt", [[m_true, b_true, f_true]], delimiter="\t",
    header="\t".join(header)
)
for num in range(nchains):
    run_sampler(m_true, b_true, f_true, header, num)
