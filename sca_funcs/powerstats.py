import numpy as np
import scipy as sp
#from scipy.stats import expon

## maybe we want to include the plotting tools here too? Or in a separate
## folder for all the different plotting tools.

def MC_KS_exp(x, N_draws, N_samples=None, do_sims=False):
    """ Monte Carlo simulation of KS test comparing data distribution and theoretical
    distribution whose parameters were estimated using the data.

    When comparing the inferred empirical distribution from the theoretical distribution, KS test
    tends to overestimate the test statistic (inflated p-value). Hence, we perform MC simulations
    to infer a distribution of test statistics.

    Parameters
    ----------
    x : 1-D array
        Data to be tested, in this particular context, spectral power series.
    N_draws : int
        Number of times to perform simulation.
    N_samples : int
        Number of samples to draw in each simulation. Defaults to None, which
        uses the len(x).
    do_sims : bool
        Whether to perform MC simulations. Defaults to False.

    Returns
    -------
    obs_stat: float
        Observed KS test statistic comparing data and theoretical distribution.

    obs_pval: float
        Observed KS test p-value.

    sim_pval: float
        Updated KS test p-value based on simulated test-statistic distribution.

    sim_statistics: 1-D array, 1 x N_draws
        vector of KS test statistic from simulated tests.
    """


    # if N_samples is not given, default to same number as # of power values
    if N_samples is None:
        N_samples = len(x)

    # initialize test statistic array
    sim_statistics = np.zeros((N_draws))

    # fit theoretical dist to data, get one set of params
    obs_param = sp.stats.expon.fit(x, floc=0)

    # compute empirical ks-test statistic and p-val on data
    obs_stat, obs_pval = sp.stats.kstest(x, 'expon', args=obs_param)

    if do_sims:
        # perform MC sim
        for n in range(N_draws):
            # draw from theoretical distribution
            simulated = np.random.exponential(scale=obs_param[1],size=N_samples)

            # test randomly drawn data against theoretical for statistic and p-val
            stat, pval = sp.stats.kstest(simulated, 'expon', args=obs_param)
            sim_statistics[n] = stat

        # find simulated p-value
        sim_pval = len([s for s in sim_statistics if s>obs_stat])/N_draws
    else:
        sim_pval = np.nan
        sim_statistics = np.nan*np.ones((N_draws))

    return obs_stat, obs_pval, sim_pval, sim_statistics


def spg_mc_KSexp(spg, N_draws=1000, N_samples=None, do_sims=False):
    """Perform KS test assuming exponential distribution on spectrogram data.

    Parameters
    ----------
    spg : 2-D array, frequency x time
        Spectrogram data (or any 2D array). Tests each column using MC_KS_exp.
    N_draws : int
        Number of times to perform simulation.
    N_samples : int
        Number of samples to draw in each simulation. Defaults to None, which
        uses the len(x).
    do_sims : bool
        Whether to perform MC simulations. Defaults to False.

    Returns
    -------
    p_sims: 1-D array
        Vector of simulation-corrected p-values.
    p_emp: 1-D array
        Vector of observed (point-estimate) p-values.
    t_emp: 1-D array
        Vector of observed (point-estimate) test statistics.

    """
    N_freqs, N_data = spg.shape
    if N_samples is None:
        N_samples = N_data

    # init vectors
    p_sims = np.zeros(N_freqs)
    p_emp = np.zeros(N_freqs)
    t_emp = np.zeros(N_freqs)

    for i in range(N_freqs):
        t_emp[i], p_emp[i], p_sims[i], sim_stat = MC_KS_exp(spg_[i,:],N_draws=N_draws,N_samples=N_samples, do_sims=do_sims)

    return p_sims, p_emp, t_emp
