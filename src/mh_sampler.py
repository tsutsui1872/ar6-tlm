import logging
import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)

def sampling(df, target_mean, target_cov, seed=0):
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

    Returns
    -------
        Indexes of accepted states
    """
    # Probability in the proposed distribution
    if df.ndim == 2:
        kernel = stats.gaussian_kde(df.T)
        fp = kernel(df.T)
    else:
        kernel = stats.gaussian_kde(df)
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