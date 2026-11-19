"""Numerical checks for continuous Morlet power and sparse time sampling."""
import unittest

import numpy as np

from src.spectrograms import morlet_kernel, rate_spectrogram


class MorletSpectrogramTests(unittest.TestCase):
    def test_windows_span_at_least_the_requested_periods(self):
        dt = 0.005
        frequencies, _, _ = rate_spectrogram(np.zeros(2000))
        np.testing.assert_allclose(frequencies, np.arange(1, 20.01, 0.25))
        for periods in (3, 4, 6):
            for frequency in frequencies:
                kernel = morlet_kernel(frequency, bin_s=dt, window_periods=periods)
                duration = (len(kernel) - 1) * dt
                requested = periods / frequency
                self.assertGreaterEqual(duration + 1e-12, requested)
                self.assertLess(duration, requested + 2 * dt + 1e-12)
        for frequency, expected in ((1, 3), (5, 0.6), (10, 0.3), (20, 0.15)):
            duration = (len(morlet_kernel(frequency)) - 1) * dt
            self.assertAlmostEqual(duration, expected)

    def test_wavelets_shrink_with_frequency_and_reject_dc(self):
        widths = []
        for frequency in (1, 5, 10, 20):
            wavelet = morlet_kernel(frequency)
            widths.append(len(wavelet))
            self.assertAlmostEqual(np.linalg.norm(wavelet), 1)
            self.assertLess(abs(wavelet.sum()), 1e-12)
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(widths[1], widths[2])
        _, _, powers = rate_spectrogram(np.ones(1000))
        self.assertLess(max(p.max() for p in powers), 1e-25)

    def test_support_overlap_is_bounded_at_every_frequency(self):
        for overlap in (0, 0.1, 0.5):
            frequencies, times, powers = rate_spectrogram(np.zeros(2000), overlap=overlap)
            for frequency, time, power in zip(frequencies, times, powers):
                length = len(morlet_kernel(frequency))
                hops = np.diff(time) / 0.005
                self.assertTrue(np.all((length - hops) / length <= overlap + 1e-12))
                self.assertEqual(len(time), len(power))
                self.assertAlmostEqual(time[0], 0.0025)
                self.assertLess(time[-1], 10)
            self.assertLess(len(times[0]), len(times[-1]))

    def test_known_tone_and_continuous_onset_match_direct_transform(self):
        dt = 0.005
        time = (np.arange(2000) + 0.5) * dt
        rate = np.where(time >= 3.017, np.cos(2 * np.pi * 10 * time), 0)
        frequencies, times, powers = rate_spectrogram(rate)
        mean_power = [p[(t > 4) & (t < 8)].mean() for t, p in zip(times, powers)]
        self.assertAlmostEqual(frequencies[np.argmax(mean_power)], 10, delta=0.75)
        i = np.flatnonzero(frequencies == 10)[0]
        wavelet = morlet_kernel(10)
        half = len(wavelet) // 2
        padded = np.pad(rate, (half, half), mode="reflect")
        direct = np.convolve(padded, wavelet.conj()[::-1], mode="valid")
        centers = np.rint(times[i] / dt - 0.5).astype(int)
        np.testing.assert_allclose(powers[i], abs(direct[centers]) ** 2, atol=1e-12)
        # Mixing is confined to wavelet support, without a separate pre/post transform.
        self.assertLess(powers[i][times[i] < 3.017 - half * dt].max(), 1e-25)

    def test_invalid_parameters_are_rejected(self):
        for kwargs in ({"overlap": -0.1}, {"overlap": 1}, {"window_periods": 0},
                       {"bin_s": 0}, {"frequencies": [10, 5]},
                       {"frequencies": [10, 100]}):
            with self.assertRaises(ValueError):
                rate_spectrogram(np.zeros(1000), **kwargs)


if __name__ == "__main__":
    unittest.main()
