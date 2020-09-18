import numpy as np
from scipy.integrate import quad
from scipy.signal.spectral import _median_bias
from scipy.special import beta


def likelihood_integrand(psd, nu, n_segments, old=1):
    """
    Evaluate (22) of arXiv:2006.05292

    Parameters
    ----------
    psd: float
        The PSD, equivalent to variance = 1 / Q
    nu: float
        The whitened strain
    n_segments: int
        The number of segments being averaged over
    old: float
        An optional guess at the result, this is used in the recursive integration
        to improve convergence

    Returns
    -------
    float: the integrand
    """
    mm = (n_segments - 1) / 2
    return (
        (
            1
            / psd ** 3
            * (1 - np.exp(-1 / 2 / psd)) ** mm
            * np.exp(-(abs(nu) ** 2 / _median_bias(n=n_segments) / 2 + mm + 1) / 2 / psd)
        )
        / beta(mm + 1, mm + 1)
        / old
    )


def whitening_integrand(psd, nu, n_segments, old=1):
    """
    Evaluate (24) of arXiv:2006.05292

    Parameters
    ----------
    psd: float
        The PSD, equivalent to variance = 1 / Q
    nu: float
        The whitened strain
    n_segments: int
        The number of segments being averaged over
    old: float
        An optional guess at the result, this is used in the recursive integration
        to improve convergence

    Returns
    -------
    float: the integrand
    """
    mm = (n_segments - 1) / 2
    return (
        (
            1
            / psd ** 2.5
            * (1 - np.exp(-1 / 2 / psd)) ** mm
            * np.exp(-(abs(nu) ** 2 / _median_bias(n=n_segments) / 2 + mm + 1) / 2 / psd)
        )
        / beta(mm + 1, mm + 1)
        / old
    )


def recursively_integrate(integrand, nu, n_segments, old_guess):
    """Recursively evaluate an integral based on an initial guess"""
    for _ in range(2):
        new_guess = quad(
            integrand, 0, np.inf, (nu, int(n_segments), old_guess), limit=100
        )[0]
        output = new_guess * old_guess
        old_guess = new_guess
    return output
