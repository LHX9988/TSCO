import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from tsco.features import Normalization
from tsco.simulation import scenario_forecasts
from tsco.training import calibrate


class StrictControllerTests(unittest.TestCase):
    def test_forecasts_require_exact_supplied_day(self):
        case=SimpleNamespace(n_bus=2,n_renewable=1,n_storage=0)
        data=SimpleNamespace(planning_scenarios=[])
        with self.assertRaisesRegex(ValueError,'planning_scenarios'):
            scenario_forecasts(data,case,0,1)
        record=dict(start_interval=0,demand_mw=[[[2.,3.],[4.,5.]]],
                    renewable_mw=[[[1.],[2.]]],probabilities=[1.],
                    storage_modes=[[[],[]]],hour_dt=1.)
        data.planning_scenarios=[record]
        demand,res,prob,modes,expected=scenario_forecasts(data,case,0,1)
        np.testing.assert_array_equal(demand,[[[2.,3.]]])
        np.testing.assert_array_equal(res,[[[1.]]])
        self.assertEqual(modes.shape,(1,1,0))
        self.assertIsNone(expected)
        with self.assertRaisesRegex(ValueError,'interval 12'):
            scenario_forecasts(data,case,12,1)
        record['probabilities']=None
        with self.assertRaisesRegex(ValueError,'probabilities'):
            scenario_forecasts(data,case,0,1)

    def test_calibration_requires_supplied_samples_without_rollout(self):
        with patch('tsco.training.Simulator') as sim:
            with self.assertRaisesRegex(ValueError,'calibration sample file'):
                calibrate('unused',{},'unused')
            sim.assert_not_called()

    def test_calibration_uses_supplied_training_statistics_and_q95(self):
        sample=dict(nodes=[[1.,2.]],edges=[[[3.,4.]]],adjacency=[[True]],
                    task=[5.,6.],global_features=[7.,8.],mask=[True,False,False,False])
        payload=dict(provenance={'split':'train'},observations=[sample],
                     rewards=[dict(revenue=1.,cost=2.,carbon=3.,tardiness=4.),
                              dict(revenue=3.,cost=4.,carbon=5.,tardiness=6.)],
                     feature_scales={k:[2.,2.] for k in Normalization.fields},
                     feature_offsets={k:[1.,1.] for k in Normalization.fields})
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'samples.json';path.write_text(json.dumps(payload))
            with patch('tsco.training.Simulator') as sim:
                norm,obs=calibrate('unused',{},Path(temporary)/'out',samples_path=path)
                sim.assert_not_called()
            self.assertAlmostEqual(norm.reward_scales['revenue'],2.9)
            np.testing.assert_array_equal(norm.transform(obs).nodes,[[0.,.5]])
            payload['provenance']['split']='test';path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'training partition'):
                calibrate('unused',{},Path(temporary)/'out',samples_path=path)

    def test_blank_normalization_is_not_replaced_with_unit_scales(self):
        with self.assertRaisesRegex(ValueError,'feature_scales'):
            Normalization(provenance={'split':'train'}).validate()


if __name__=='__main__':unittest.main()
