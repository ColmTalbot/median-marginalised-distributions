import os

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.stats import rv_continuous

__dir__ = os.path.abspath(os.path.dirname(__file__))


class _InterpolatedDistribution(rv_continuous):
    def _logpdf(self, x, n_average):
        n_average = self._sanitize_n_average(n_average)
        return self._cached_logpdf_interpolants[n_average](abs(x))

    def _pdf(self, x, n_average):
        return np.exp(self._logpdf(x, n_average))

    def _cdf(self, x, n_average):
        n_average = self._sanitize_n_average(n_average)
        return self._cached_cdf_interpolants[n_average](x)

    def _ppf(self, x, n_average):
        n_average = self._sanitize_n_average(n_average)
        return self._cached_inverse_cdf_interpolants[n_average](x)

    def _sanitize_n_average(self, n_average):
        if isinstance(n_average, np.ndarray):
            if n_average.size > 1:
                n_average = n_average[0]
        n_average = int(n_average)
        if n_average not in self._cached_cdf_interpolants:
            self._add_interpolants(n_average)
        return n_average

    def _add_interpolants(self, n_average):
        self._cached_logpdf_interpolants[n_average] = interp1d(
            self._data["x"],
            self._data[str(n_average)],
            bounds_error=False,
            fill_value=0,
        )
        cdf = cumtrapz(
            np.exp(self._cached_logpdf_interpolants[n_average](np.abs(self.x_values))),
            self.x_values,
            initial=0,
        )
        cdf /= cdf[-1]
        self._cached_cdf_interpolants[n_average] = interp1d(
            self.x_values, cdf, bounds_error=False, fill_value=(0, 1)
        )
        self._cached_inverse_cdf_interpolants[n_average] = interp1d(cdf, self.x_values)


class MedianMarginalisedDistribution(_InterpolatedDistribution):
    _data = pd.read_csv(os.path.join(__dir__, "whitening.dat"), sep="\t")
    _cached_logpdf_interpolants = dict()
    _cached_cdf_interpolants = dict()
    _cached_inverse_cdf_interpolants = dict()
    x_values = np.linspace(-100, 100, 10000)


class MedianMarginalisedLikelihood(_InterpolatedDistribution):
    _data = pd.read_csv(os.path.join(__dir__, "likelihood.dat"), sep="\t")
    _cached_logpdf_interpolants = dict()
    _cached_cdf_interpolants = dict()
    _cached_inverse_cdf_interpolants = dict()
    x_values = np.linspace(0, 100, 10000)


median_marginalised = MedianMarginalisedDistribution()
median_marginalised_likelihood = MedianMarginalisedLikelihood()
