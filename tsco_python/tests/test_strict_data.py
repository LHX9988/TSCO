"""Input completeness, chronological partitioning, and explicit state checks."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tsco.data import (load_case, load_dataset, nominal_case, save_case,
                       _parse_job_records, _read_initial_states, _read_planning_scenarios)
from test_grid import small_case


class StrictDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.case = small_case()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, payload):
        path = self.path/name
        path.write_text(json.dumps(payload))
        return path

    def test_no_implicit_nominal_grid(self):
        with self.assertRaisesRegex(ValueError, 'case.json'):
            nominal_case()
        template = Path(__file__).parents[1]/'inputs'/'case.json'
        with self.assertRaisesRegex(ValueError, '尚未填写'):
            load_case(template)

    def test_blank_jobs_are_not_zero_workload(self):
        for payload in ({'jobs': None}, {'jobs': []}, []):
            with self.assertRaisesRegex(ValueError, '作业'):
                _parse_job_records(payload)

    def test_dependency_edge_volume_is_required(self):
        task = {'id': 'a', 'workload': 10., 'demands': [1, 0, 0],
                'compatible': [0], 'predecessor_ids': [], 'data_gb': 1., 'revenue': 2.}
        successor = dict(task, id='b', predecessor_ids=['a'])
        job = {'id': 'j', 'arrival_seconds': 0., 'deadline_seconds': 100.,
               'tasks': [task, successor]}
        with self.assertRaisesRegex(ValueError, 'dependency edge'):
            _parse_job_records([job])
        successor['edge_data_gb'] = {'a': 1.}
        self.assertEqual(len(_parse_job_records([job])[0].tasks), 2)
        task['predecessor_ids'] = ['b']; task['edge_data_gb'] = {'b': 1.}
        with self.assertRaisesRegex(ValueError, 'acyclic'):
            _parse_job_records([job])

    def test_initial_duration_must_be_provided(self):
        state = {'start_interval': 0, 'commitment': [1], 'generation_mw': [8.],
                 'storage_energy_mwh': [], 'cpn_empty': True}
        path = self.write('states.json', {'initial_states': [state]})
        with self.assertRaisesRegex(ValueError, 'initial_status_hours'):
            _read_initial_states(path, self.case, 10)
        state['initial_status_hours'] = [3.]
        self.write('states.json', {'initial_states': [state]})
        self.assertEqual(_read_initial_states(path, self.case, 10)[0]['initial_status_hours'][0], 3.)

    def test_planning_storage_modes_and_component_are_explicit(self):
        case = small_case(storage=True)
        record = {'start_interval': 0, 'hour_dt': 1., 'demand_mw': [[[0., 8.]]],
                  'renewable_mw': [[[2.]]], 'probabilities': [1.]}
        path = self.write('plans.json', {'planning_scenarios': [record]})
        with self.assertRaisesRegex(ValueError, 'storage_modes'):
            _read_planning_scenarios(path, case, 10)
        record['storage_modes'] = [[[0]]]
        self.write('plans.json', {'planning_scenarios': [record]})
        result = _read_planning_scenarios(path, case, 10)[0]
        self.assertIsNone(result['cpn_demand_mw'])
        self.assertIsNone(result['expected_value_storage_modes'])

    def test_chronology_exact_boundaries_and_whole_jobs(self):
        save_case(self.case, self.path/'case.json')
        origin = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with (self.path/'signals.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['timestamp', 'base_bus_0_mw', 'base_bus_1_mw', 'res_0_mw'])
            for i in range(2880):
                writer.writerow([(origin+timedelta(seconds=300*i)).isoformat(), 0., 8., 2.])
        states = [{'start_interval': i, 'commitment': [1], 'generation_mw': [8.],
                   'storage_energy_mwh': [], 'initial_status_hours': [3.], 'cpn_empty': True}
                  for i in (0, 2016, 2448)]
        plans = [{'start_interval': i, 'hour_dt': 1., 'demand_mw': [[[0., 8.]]],
                  'renewable_mw': [[[2.]]], 'probabilities': [1.], 'storage_modes': [[[]]]}
                 for i in (0, 2016, 2448)]
        self.write('initial_states.json', {'initial_states': states})
        self.write('planning_scenarios.json', {'planning_scenarios': plans})
        jobs = []
        for i, (arrival, deadline) in enumerate([(0, 600), (2016*300-1, 2016*300+1),
                                               (2016*300, 2016*300+600), (2448*300, 2448*300+600)]):
            jobs.append({'id': str(i), 'arrival_seconds': arrival, 'deadline_seconds': deadline,
                         'tasks': [{'id': f't{i}', 'workload': 10., 'demands': [1, 0, 0],
                                    'compatible': [0], 'predecessor_ids': [], 'data_gb': 0., 'revenue': 1.}]})
        self.write('jobs.json', {'jobs': jobs})
        with patch('tsco.cpn.CPNSimulator'):
            data = [load_dataset(self.path, split) for split in ('train', 'validation', 'test')]
        self.assertEqual([len(x.times) for x in data], [2016, 432, 432])
        self.assertEqual([len(x.jobs) for x in data], [1, 1, 1])
        self.assertEqual([x.source_start_index for x in data], [0, 2016, 2448])
        for x in data:
            self.assertEqual(x.initial_state['start_interval'], 0)
            self.assertEqual(x.planning_scenarios[0]['start_interval'], 0)
            self.assertEqual(x.storage_modes.shape, (len(x.times), 0))
        with self.assertRaisesRegex(ValueError, 'explicit record'):
            data[2].window(1, 12)
        self.assertEqual(len(data[2].window(0, 12).times), 12)


if __name__ == '__main__':
    unittest.main()
