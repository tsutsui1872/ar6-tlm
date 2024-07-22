import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from mce.core.forcing import RfAll
from mce.core.climate import IrmBase

class DriverMCE:
    """
    Custom MCE driver for the two-layer model
    with or without ocean heat uptake efficacy
    """
    def __init__(self):
        self.forcing = RfAll()
        self.climate = IrmBase()
        self.parms = {}

        # Conversion factor borrowed from the AR6 two-layer model
        SECPERYEAR = 60 * 60 * 24 * 365
        EARTHRADIUS = 6371000
        # self.ntoa2joule = 4*np.pi * EARTHRADIUS * EARTHRADIUS * SECPERYEAR * 1e-22
        # modified to covert to ZJ unit
        self.ntoa2joule = 4*np.pi * EARTHRADIUS * EARTHRADIUS * SECPERYEAR * 1e-21

        self.cco2_pi_1750 = self.forcing.parms_ar6_ghg.C0_1750
        self.q4x_ref = self.forcing.c2erf_ar6('CO2', self.cco2_pi_1750 * 4.)
        self.interp_cco2 = None

    def mk_interp_cco2(self, xlim=(-1., 12.)):
        """
        Define an interpolation object to get equivalent CO2 concentrations
        to well-mixed GHG forcing levels using the AR6 formula of CO2 forcing
        to relate concentrations to forcing levels.

        Parameters
        ----------
        xlim, optional
            Nominal range of well-mixed GHG forcing in W/m2 across assumed scenarios,
            by default (-1., 12.)
        """
        cco2_2019 = 409.85
        cn2o_2019 = 332.091
        erf_co2_2019 = self.forcing.c2erf_ar6('CO2', cco2_2019, cn2o=cn2o_2019)

        # y-value: equivalent CO2 concentrations
        cco2_pi = self.cco2_pi_1750
        alpha = erf_co2_2019 / np.log(cco2_2019 / cco2_pi)
        yp = cco2_pi * np.exp(np.linspace(*xlim, **{'num': 500}) / alpha)

        # x-value: well-mixed GHG forcing levels
        xp = self.forcing.c2erf_ar6('CO2', yp)

        self.interp_cco2 = interp1d(xp, yp)

    def calib(self, parms):
        """Update climate parameters

        Parameters
        ----------
        parms
            Two-layer model parameters
        """
        p = parms.dropna()

        lamg = p['lamg']
        gamma_p = p['gamma_2l']
        cmix = p['cmix']
        cdeep_p = p['cdeep']
        eff = p.get('eff', 1.)

        if eff != 1.:
            gamma_p = gamma_p * eff
            cdeep_p = cdeep_p * eff

        tauj, akj = self.climate.ebm_to_irm([lamg, gamma_p], [cmix, cdeep_p])
        self.climate.parms.update(asj=akj[0], tauj=tauj, lamb=lamg)

        # MCE forcing parameters
        # q2x = alpha * np.log(2)
        # q4x = alpha * np.log(4) * beta
        co2_alpha = p['q2x'] / np.log(2)
        co2_beta = p.get('co2_beta', 0.5*p['q4x']/p['q2x'])
        self.forcing.parms.update(alpha=co2_alpha, beta=co2_beta)

        self.parms = {'tauj': tauj, 'akj': akj, **p.to_dict()}

    def run(self, qin, tkjlast=None, qlast=None):
        """Time integration

        Parameters
        ----------
        qin
            Yearly forcing

        Returns
        -------
            Thermal response
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

        df = {
            'tg': tlev[1:, 0],
            'ohc': ohc[1:],
            'hflux': hflux[1:],
        }
        return pd.DataFrame(df, index=qin.index)

    def run_ideal_co2(self, time, qin):
        """Time integration for instantaneous or transient forcing changes

        Parameters
        ----------
        time
            Time points, unequal spacing allowed
        qin
            Instantaneous forcing change by scalar
            or transient forcing changes by array

        Returns
        -------
            Thermal response
        """
        p = self.parms

        if np.isscalar(qin):
            tlev = (
                self.climate.response_ideal(time, 'step', 'tres', asj=p['akj'])
                * (qin / p['lamg'])
            )
            hflux = self.climate.response_ideal(time, 'step', 'flux') * qin
        else:
            tlev = self.climate.response(time, qin, asj=p['akj'])
            hflux = qin - p['lamg'] * tlev[:, 0]

        eff = p.get('eff', 1.)

        if eff != 1.:
            hflux = hflux - (eff - 1.) * p['gamma_2l'] * (tlev[:, 0] - tlev[:, 1])

        ohc = (tlev * np.array([p['cmix'], p['cdeep']])).sum(axis=1) * self.ntoa2joule

        df = {
            'tg': tlev[:, 0],
            'ohc': ohc,
            'hflux': hflux,
        }
        return pd.DataFrame(df, index=time)

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
