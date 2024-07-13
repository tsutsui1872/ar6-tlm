import logging
import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)

def sampling(df, target_mean, target_cov, seed=0, maxlen=None):
    """
    Apply the Metropolis test with the condition of detailed balance
    in terms of proposed (input) and target probability density

    Parameters
    ----------
    df
        Input series (multivariate or univariate)
    target_mean
        Means of target distributions
    target_cov
        Covariance or variance of target distributions
    seed, optional
        Seed for random module
    maxlen, optional
        If None, the length of the input series is reduced to this value
        for kernel density evaluation

    Returns
    -------
        Indexes of accepted states
    """
    slc = slice(None, maxlen)

    # Probability in the proposed distribution
    if df.ndim == 2:
        kernel = stats.gaussian_kde(df.iloc[slc].T)
        fp = kernel(df.T)
    else:
        kernel = stats.gaussian_kde(df.iloc[slc])
        fp = kernel(df)

    # Probability in the target distribution
    if df.ndim == 2:
        rv_target = stats.multivariate_normal(mean=target_mean, cov=target_cov)
    else:
        rv_target = stats.norm(loc=target_mean, scale=np.sqrt(target_cov))

    ft = rv_target.pdf(df)

    np.random.seed(seed)

    i = 0
    ret = [i]
    naccept = 1

    for ip in range(1, len(df)):
        r = np.random.uniform(0, 1)
        if ft[ip] * fp[i] / (ft[i] * fp[ip]) > r:
            i = ip
            naccept += 1

        ret.append(i)

    logger.info('acceptance rate {}'.format(naccept/len(df)))

    return ret


def asymmetric_gaussian(central, unc90, size=10000, random_state=1):
    """
    Generate samples from an Gaussian distribution
    and adjust values according to asymmetry

    Parameters
    ----------
    central
        Central value
    unc90
        90% uncertainty range
    size, optional
        The number of samples, by default 10000
    random_state, optional
        Seed to initialize the random generator, by default 1

    Returns
    -------
        Generated samples
    """
    NINETY_TO_ONESIGMA = stats.norm.ppf(0.95)

    d1 = stats.norm.rvs(
        size=size,
        loc=np.ones(size),
        scale=np.ones(size) * (unc90[1]/central-1.) / NINETY_TO_ONESIGMA,
        random_state=random_state,
    )
    d1 = np.where(
        d1 > 1., d1,
        (d1 - 1.) * ((central-unc90[0])/(unc90[1]-central)) + 1.,
    ) * central

    return d1
