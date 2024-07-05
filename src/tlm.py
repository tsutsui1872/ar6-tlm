import numpy as np
import pandas as pd
import scipy.stats as stats

from mce.core.climate import IrmBase

class DriverMCE:
    """
    Custom MCE driver
    """
    def __init__(self):
        self.climate = IrmBase()
        self.parms = {}

        # Conversion factor borrowed from the AR6 two-layer model
        SECPERYEAR = 60 * 60 * 24 * 365
        EARTHRADIUS = 6371000
        # self.ntoa2joule = 4*np.pi * EARTHRADIUS * EARTHRADIUS * SECPERYEAR * 1e-22
        # modified to covert to ZJ unit
        self.ntoa2joule = 4*np.pi * EARTHRADIUS * EARTHRADIUS * SECPERYEAR * 1e-21

    def calib(self, parms):
        """Update climate parameters

        Parameters
        ----------
        parms
            Two-layer model parameters
        """
        lamg = parms['lamg']
        gamma_p = parms['gamma_2l']
        cmix = parms['cmix']
        cdeep_p = parms['cdeep']
        eff = parms.get('eff', 1.)

        if eff != 1.:
            gamma_p *= eff
            cdeep_p *= eff

        tauj, akj = self.climate.ebm_to_irm([lamg, gamma_p], [cmix, cdeep_p])
        self.climate.parms.update(asj=akj[0], tauj=tauj, lamb=lamg)

        self.parms.update(tauj=tauj, akj=akj, **parms.to_dict())

    def run(self, qin, tkjlast=None, qlast=None):
        """Perform time integration

        Parameters
        ----------
        qin
            Yearly forcing

        Returns
        -------
            Climate response
        """
        p = self.parms
        if tkjlast is None:
            # Times are defined as mid-year points, and initial zero is added
            time = np.hstack([0., np.arange(len(qin))+0.5])
            qvals = np.hstack([0., qin.values])
            tlev = self.climate.response(time, qvals, asj=p['akj'])
        else:
            time = np.arange(len(qin)+1) + 0.5
            qvals = np.hstack([qlast, qin.values])
            self.climate.tkjlast = tkjlast
            tlev = self.climate.response(time, qvals, asj=p['akj'], init=False)

        eff = p.get('eff', 1.)

        hflux = qvals - p['lamg'] * tlev[:, 0]
        if eff != 1.:
            hflux = hflux - (eff - 1.) * p['gamma_2l'] * (tlev[:, 0] - tlev[:, 1])

        ohc = (tlev * np.array([p['cmix'], p['cdeep']])).sum(axis=1) * self.ntoa2joule

        return pd.DataFrame({
            'tg': tlev[1:, 0],
            'ohc': ohc[1:],
            'hflux': hflux[1:],
        }, index=qin.index)


def ebm_to_irm(df):
    """
    Conversion from energy balance models to impulse response models

    Parameters
    ----------
    df
        Input parameters

    Returns
    -------
        Converted parameters
    """
    eff = df['eff'].fillna(1.)
    msyst = np.array([
        [
            (df['lamg'] + df['gamma_2l'] * eff) / df['cmix'],
            - df['gamma_2l'] * eff / df['cmix'],
        ],
        [
            - df['gamma_2l'] * eff / (df['cdeep'] * eff),
            df['gamma_2l'] * eff / (df['cdeep'] * eff),
        ],
    ]).transpose([2, 0, 1])
    eigval, eigvec = np.linalg.eig(msyst)

    tauk = 1./eigval

    def mkdiag(xin):
        ndim = xin.shape[1]
        x = np.zeros(xin.shape + (ndim,))
        for i in range(ndim):
            x[:, i, i] = xin[:, i]
        return x

    akj = np.linalg.solve(
        eigvec[:, :, :] / eigvec[:, 0:1, :],
        mkdiag(
            df['lamg'].values.reshape((-1, 1))
            / np.array([df['cmix'], df['cdeep']*eff]).transpose([1, 0])
        ),
    )
    akj = np.stack([
        akj[:, 0] / eigval[:, 0].reshape((-1, 1)),
        akj[:, 1] / eigval[:, 1].reshape((-1, 1)),
    ]).transpose(1, 2, 0)

    return pd.DataFrame({
        'tau0': tauk[:, 0],
        'tau1': tauk[:, 1],
        'a0': akj[:, 0, 0],
        'a1': akj[:, 0, 1],
    }, index=df.index)


def irm_to_ebm(df):
    """
    Conversion from impulse response models to energy balance models

    Parameters
    ----------
    df
        Input parameters

    Returns
    -------
        Converted parameters
    """
    asj = df[['a0', 'a1']].values
    tauj = df[['tau0', 'tau1']].values
    lamb = df['lamg'].values
    eff = df['eff'].fillna(1.).values

    xitot = (asj*tauj).sum(axis=1) * lamb
    xis = lamb / (asj/tauj).sum(axis=1)

    xi1 = xitot - xis
    lamb1 = xis * xi1 / lamb / tauj.prod(axis=1)

    return pd.DataFrame({
        'gamma_2l': lamb1 / eff,
        'cmix': xis,
        'cdeep': xi1 / eff,
    }, index=df.index)


def add_ecs_tcr(df, approx=False, inplace=True):
    """
    Add analytically derived ECS and TCR
    Use the following approximate formula when approx=True
    tcr/ecs = lamg / (lamg + gamma_2l * eff)

    Parameters
    ----------
    df
        Parameter DataFrame
    """
    ecs = df['q2x'] / df['lamg']

    if approx:
        eff = df['eff'].fillna(1.)
        rwf = df['lamg'] / (df['lamg'] + df['gamma_2l'] * eff)
    else:
        aj = df[['a0', 'a1']].rename(columns=lambda x: x.replace('a', ''))
        tauj = df[['tau0', 'tau1']].rename(columns=lambda x: x.replace('tau', ''))
        t70 = np.log(2) / np.log(1.01)
        rwf = 1. - (aj * tauj * (1 - np.exp(-t70/tauj))).sum(axis=1) / t70

    tcr = ecs * rwf

    if inplace:
        df['ecs'] = ecs
        df['tcr'] = tcr
    else:
        return pd.DataFrame({'ecs': ecs, 'tcr': tcr})

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