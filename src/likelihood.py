import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import betaln

from bilby.gw.likelihood import GravitationalWaveTransient

__dir__ = os.path.abspath(os.path.dirname(__file__))


class PSDMarginalizedGravitationalWaveTransient(GravitationalWaveTransient):
    def __init__(
        self,
        interferometers,
        waveform_generator,
        n_segments,
        reference_frame="H1L1",
        time_reference="geocent",
        **kwargs
    ):
        """

        A likelihood object, able to compute the likelihood of the data given
        some model parameters

        This class should be subclassed and a method which calculates the PSD
        which has been marginalised over the uncertainty in the PSD should be
        defined.

        Parameters
        ----------
        interferometers: list
            A list of `bilby.gw.detector.Interferometer` instances - contains
            the detector data and power spectral densities
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            An object which computes the frequency-domain strain of the signal,
            given some set of parameters
        n_segments: int
            Number of segments from which the PSD was generated.
            These are assumed to be non-overlapping.

        """
        super(PSDMarginalizedGravitationalWaveTransient, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=None,
            distance_marginalization=False,
            phase_marginalization=False,
            time_marginalization=False,
            jitter_time=False,
            reference_frame=reference_frame,
            time_reference=time_reference,
        )
        self.n_segments = n_segments
        self._noise_log_likelihood = None

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(interferometers={},\n\twaveform_generator={},\n\tn_segments={})".format(
                self.interferometers, self.waveform_generator, self.n_segments
            )
        )

    def noise_log_likelihood(self):
        """Calculates the real part of noise log-likelihood

        This has been hacked to make the normalisation agree with the GWT.

        Returns
        -------
        float: The real part of the noise log likelihood

        """
        if self._noise_log_likelihood is None:
            log_l = 0
            for ifo in self.interferometers:
                whitened_strain = (
                    abs(ifo.frequency_domain_strain[ifo.frequency_mask])
                    / ifo.amplitude_spectral_density_array[ifo.frequency_mask]
                )
                whitened_strain *= (4 / ifo.strain_data.duration) ** 0.5
                log_l += sum(self._logpdf(abs(whitened_strain)))
            self._noise_log_likelihood = float(np.real(log_l))
        return self._noise_log_likelihood

    def log_likelihood(self):
        """Calculates the real part of log-likelihood value

        Returns
        -------
        float: The real part of the log likelihood

        """
        try:
            self.parameters.update(self.get_sky_frame_parameters())
        except AttributeError:
            pass

        log_l = 0
        waveform = self.waveform_generator.frequency_domain_strain(
            self.parameters.copy()
        )
        if waveform is None:
            return np.nan_to_num(-np.inf)
        log_l = 0
        for ifo in self.interferometers:
            response = ifo.get_detector_response(waveform, self.parameters)[
                ifo.frequency_mask
            ]
            whitened_strain = (
                abs(ifo.frequency_domain_strain[ifo.frequency_mask] - response)
                / ifo.amplitude_spectral_density_array[ifo.frequency_mask]
            )
            whitened_strain *= (4 / ifo.strain_data.duration) ** 0.5
            log_l += sum(self._logpdf(abs(whitened_strain)))
        log_l = float(np.real(log_l))
        return log_l.real

    def _logpdf(self, whitened_strain):
        raise NotImplementedError()


class MeanMarginalizedGravitationalWaveTransient(
    PSDMarginalizedGravitationalWaveTransient
):
    def __init__(
        self,
        interferometers,
        waveform_generator,
        n_segments,
        reference_frame="H1L1",
        time_reference="geocent",
    ):
        super(MeanMarginalizedGravitationalWaveTransient, self).__init__(
            interferometers,
            waveform_generator,
            n_segments,
            reference_frame=reference_frame,
            time_reference=time_reference,
        )

    def _logpdf(self, whitened_strain):
        ln_likelihood = (
            np.log1p(abs(whitened_strain) ** 2 / 2 / self.n_segments)
            * (-1 - self.n_segments)
            + betaln(self.n_segments, 1)
            + np.log(self.n_segments / 2)
            - np.log(np.pi)
        )
        return ln_likelihood


class MedianMarginalizedGravitationalWaveTransient(
    PSDMarginalizedGravitationalWaveTransient
):

    data = pd.read_csv(os.path.join(__dir__, "likelihood.dat"), sep="\t")

    def __init__(
        self,
        interferometers,
        waveform_generator,
        n_segments,
        reference_frame="H1L1",
        time_reference="geocent",
    ):
        """
        A likelihood object, able to compute the likelihood of the data given
        some model parameters

        This includes a user implemented method of marginalising over the
        uncertainty in the PSD.

        This class assumes the PSD was calculated using a median average.

        Parameters
        ----------
        interferometers: list
            A list of `bilby.gw.detector.Interferometer` instances - contains
            the detector data and power spectral densities
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            An object which computes the frequency-domain strain of the signal,
            given some set of parameters
        n_segments: int
            Number of segments from which the PSD was generated.
            These are assumed to be non-overlapping.

        """
        super(MedianMarginalizedGravitationalWaveTransient, self).__init__(
            interferometers,
            waveform_generator,
            n_segments,
            reference_frame=reference_frame,
            time_reference=time_reference,
        )
        self._interpolated_logpdf = interp1d(
            self.data["x"] ** 0.5,
            self.data[str(self.n_segments)],
            bounds_error=False,
            fill_value=(-np.log(2), -np.inf),
        )

    def _logpdf(self, whitened_strain):
        return self._interpolated_logpdf(whitened_strain) - np.log(np.pi)
