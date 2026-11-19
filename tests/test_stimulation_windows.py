"""Numerical checks for window boundaries, spectral peaks and pooled phase vectors."""
import unittest

import numpy as np

from src.stimulation_windows import compare_windows


class WindowComparisonTests(unittest.TestCase):
    def test_half_open_partition_and_rate(self):
        result = compare_windows([1099, 1100, 2299, 2300, 3499, 3500], 10, 1100, 3500)
        self.assertEqual([r['n_spikes'] for r in result['records']], [2, 2, 4])
        np.testing.assert_array_equal(np.concatenate(result['rates'][:2]), result['rates'][2])
        self.assertAlmostEqual(result['records'][2]['mean_rate_hz'], 4 / 2.4 / 10)

    def test_full_plv_uses_complex_vectors_not_mean_magnitudes(self):
        # First half spikes at maxima; second half spikes half a cycle later.
        times = np.r_[np.arange(0, 1200, 100), np.arange(1250, 2400, 100)]
        rows = compare_windows(times, 1, 0, 2400)['records']
        np.testing.assert_allclose([r['plv'] for r in rows], [1, 1, 0], atol=1e-12)
        z = [complex(r['plv_real'], r['plv_imag']) for r in rows]
        self.assertAlmostEqual(abs(z[2] - (z[0]+z[1])/2), 0, places=12)

    def test_2000_ms_period_has_two_equal_halves_and_2hz_grid(self):
        result = compare_windows([1500, 2499, 2500, 3499], 10, 1500, 3500)
        self.assertEqual([r['n_spikes'] for r in result['records']], [2, 2, 4])
        self.assertEqual([r['duration_ms'] for r in result['records']], [1000, 1000, 2000])
        self.assertTrue(np.isclose(result['frequency_hz'], 2).any())

    def test_psd_finds_known_two_and_ten_hz_signals(self):
        t_ms = np.arange(2.5, 2400, 5)
        for frequency in (2, 10):
            counts = np.rint(100 + 80*np.cos(2*np.pi*frequency*t_ms/1000)).astype(int)
            result = compare_windows(np.repeat(t_ms, counts), 100, 0, 2400)
            f = result['frequency_hz']
            band = (f >= 1) & (f <= 20)
            peak = f[band][np.argmax(result['native_psd'][2, band])]
            self.assertAlmostEqual(peak, frequency, delta=0.2)
            self.assertTrue(np.all(result['common_psd'] >= 0))
            self.assertEqual([r['welch_segments'] for r in result['records']], [1, 1, 3])

    def test_empty_window_is_undefined_plv_and_zero_psd(self):
        result = compare_windows([], 10, 0, 2400)
        self.assertTrue(all(np.isnan(r['plv']) for r in result['records']))
        self.assertTrue(np.all(result['common_psd'] == 0))

    def test_invalid_inputs_rejected(self):
        for spikes, n, start, stop in [([np.nan], 1, 0, 2400), ([], 0, 0, 2400),
                                       ([], 1, 2400, 0), ([], 1, 0, 2405)]:
            with self.assertRaises(ValueError):
                compare_windows(spikes, n, start, stop)


if __name__ == '__main__':
    unittest.main()
