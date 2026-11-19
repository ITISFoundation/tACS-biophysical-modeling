"""Checks for onset alignment and averaging rates before nonlinear power."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.spectrogram_panels import average_rates, load_amplitude_averages
from src.spectrogram_comparisons import morlet_power


class AmplitudeAverageTests(unittest.TestCase):
    def test_common_interval_aligns_onsets_and_keeps_equal_seed_weights(self):
        time = (np.arange(500) + 0.5)*0.005
        records = [dict(rate_time=time, rate=time-1.5),
                   dict(rate_time=time, rate=3*(time-1.1))]
        common, mean, seeds = average_rates(records, [1.5, 1.1])
        np.testing.assert_allclose(common[[0, -1]], [-1.0975, 0.9975])
        np.testing.assert_allclose(mean, 2*common, atol=1e-12)
        np.testing.assert_allclose(seeds[0], common, atol=1e-12)
        np.testing.assert_allclose(seeds[1], 3*common, atol=1e-12)
        self.assertEqual(seeds.shape, (2, 420))

    def test_rejects_fractional_bin_shifts_and_disjoint_recordings(self):
        record = dict(rate_time=(np.arange(10)+0.5)*0.005, rate=np.ones(10))
        with self.assertRaisesRegex(ValueError, 'align'):
            average_rates([record], [0.001])
        with self.assertRaisesRegex(ValueError, 'common interval'):
            average_rates([record, record], [0, 10])

    def test_loader_transforms_mean_rate_not_mean_seed_power(self):
        time = (np.arange(500)+0.5)*0.005
        oscillation = np.sin(2*np.pi*10*time)
        rates = [4+oscillation, 4-oscillation]
        records = [dict(rate_time=time, rate=rate, n_neurons=100, stop_s=2.5,
                        observed_stop=2.5, warnings=[]) for rate in rates]
        configs = [dict(run=dict(random_seed=seed), inputs={'stim': dict(
            input_type='extracellular_stimulation', delay=0, duration=2500,
            fields=[dict(Ex=1, Ey=0, Ez=0, frequency=10, phase=0)])}) for seed in (1, 2)]
        rows = pd.DataFrame([dict(stim_amp=1., sim_id=f'10Hz1Vm_seed{s}',
                                  source_file=f'data/run{s}/config.json') for s in (1, 2)])
        with patch('pathlib.Path.read_text', side_effect=[json.dumps({}), *map(json.dumps, configs)]), \
                patch('src.spectrogram_panels.load_record', side_effect=records):
            panels, audit = load_amplitude_averages('.', {'low': rows}, 'nodes.json', [2, 10, 30])
        run = panels['low'][1.]
        np.testing.assert_allclose(run['rate'], 4)
        self.assertLess(run['power'].max(), 1e-25)
        self.assertGreater(np.mean([morlet_power(rate, [2, 10, 30])[1].mean() for rate in rates]), 0.1)
        self.assertEqual(run['n_seeds'], 2)
        self.assertEqual(run['seeds'], [1, 2])
        self.assertEqual(len(audit), 2)


if __name__ == '__main__':
    unittest.main()
