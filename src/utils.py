import numpy as np
from scipy.integrate import quad
from scipy.signal.spectral import _median_bias
from scipy.special import beta


def likelihood_integrand(x, xx, n_segments, old=1):
    mm = (n_segments - 1) / 2
    return (
        (
            1
            / x ** 3
            * (1 - np.exp(-1 / 2 / x)) ** mm
            * np.exp(-(abs(xx) ** 2 / _median_bias(n=n_segments) / 2 + mm + 1) / 2 / x)
        )
        / beta(mm + 1, mm + 1)
        / old
    )


def whitening_integrand(x, xx, n_segments, old=1):
    mm = (n_segments - 1) / 2
    return (
        (
            1
            / x ** 2.5
            * (1 - np.exp(-1 / 2 / x)) ** mm
            * np.exp(-(abs(xx) ** 2 / _median_bias(n=n_segments) / 2 + mm + 1) / 2 / x)
        )
        / beta(mm + 1, mm + 1)
        / old
    )


def recursively_integrate(integrand, xx, n_segments, old_guess):
    for _ in range(2):
        new_guess = quad(
            integrand, 0, np.inf, (xx, int(n_segments), old_guess), limit=100
        )[0]
        output = new_guess * old_guess
        old_guess = new_guess
    return output
