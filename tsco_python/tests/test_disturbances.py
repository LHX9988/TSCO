import unittest
import numpy as np
from tsco.schema import GridCase
from tsco.disturbances import Disturbance, Schedule


def triangle_case():
    """Mathematical connectivity fixture."""
    empty = np.empty(0)
    return GridCase(
        name='connectivity unit fixture', base_mva=100., base_load=np.ones(3),
        branch_from=np.array([0, 1, 2]), branch_to=np.array([1, 2, 0]),
        branch_x=np.full(3, .1), branch_tap=np.ones(3), branch_limit=np.full(3, 30.),
        gen_bus=np.array([0, 1]), gen_pmin=np.zeros(2), gen_pmax=np.ones(2),
        gen_ramp=np.ones(2), gen_a=np.zeros(2), gen_b=np.ones(2), gen_c=np.zeros(2),
        gen_emission=np.ones(2), gen_startup=np.ones(2), gen_shutdown=np.ones(2),
        storage_bus=np.empty(0, int), storage_power=empty, storage_capacity=empty,
        storage_eta_c=empty, storage_eta_d=empty, storage_min=empty, storage_max=empty,
        renewable_bus=np.empty(0, int), renewable_capacity=empty, cpn_bus=np.array([0]))


class DisturbanceTests(unittest.TestCase):
    def test_generator_mask_half_open_window_and_automatic_restore(self):
        case = triangle_case()
        schedule = Schedule(case, [Disturbance('generator_outage', index=1,
                                               start_hour=1., duration_hours=1.)])
        self.assertTrue(schedule.at(11)[1].all())
        changed, availability, _ = schedule.at(12)
        self.assertFalse(availability[1])
        self.assertEqual(availability.sum(), case.n_gen - 1)
        np.testing.assert_array_equal(changed.gen_pmax, case.gen_pmax)
        self.assertFalse(schedule.at(23)[1][1])
        self.assertTrue(schedule.at(24)[1].all())
        description = schedule.descriptions()[0]
        self.assertEqual(description['equipment_id'], 2)
        self.assertEqual(description['end_hour_exclusive'], 2.0)

    def test_nonislanding_line_copy_and_one_based_description(self):
        case = triangle_case()
        schedule = Schedule(case, [Disturbance('line_outage', index=0,
                                               start_hour=0., duration_hours=1.)])
        changed, _, _ = schedule.at(0)
        self.assertEqual(changed.branch_limit[0], 0)
        self.assertEqual(case.branch_limit[0], 30)
        changed.branch_limit[1] = 99
        self.assertEqual(schedule.at(0)[0].branch_limit[1], 30)
        self.assertEqual(schedule.at(12)[0].branch_limit[0], 30)
        self.assertEqual(schedule.descriptions()[0]['from_bus'], 1)
        self.assertEqual(schedule.descriptions()[0]['to_bus'], 2)

    def test_bridge_and_overlapping_safe_lines_rejected(self):
        case = triangle_case()
        case.branch_limit[2] = 0  # The remaining network is a chain.
        with self.assertRaisesRegex(ValueError, 'islanding'):
            Schedule(case, [Disturbance('line_outage', index=0, start_hour=0, duration_hours=1)])
        case = triangle_case()
        with self.assertRaisesRegex(ValueError, 'combination'):
            Schedule(case, [Disturbance('line_outage', index=0, start_hour=0, duration_hours=1),
                            Disturbance('line_outage', index=1, start_hour=.5, duration_hours=1)])
        # Touching half-open windows do not overlap, and both single outages are safe.
        schedule = Schedule(case, [Disturbance('line_outage', index=0, start_hour=0, duration_hours=1),
                                   Disturbance('line_outage', index=1, start_hour=1, duration_hours=1)])
        self.assertEqual(np.count_nonzero(schedule.at(12)[0].branch_limit == 0), 1)

    def test_demand_multiplication_and_validation(self):
        case = triangle_case()
        schedule = Schedule(case, [Disturbance('demand_scale', value=1.1, start_hour=0, duration_hours=1),
                                   Disturbance('demand_scale', value=.9, start_hour=0, duration_hours=1)])
        self.assertAlmostEqual(schedule.at(0)[2], .99)
        self.assertAlmostEqual(schedule.at(12)[2], 1.)
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                Schedule(case, [Disturbance('demand_scale', value=value, start_hour=0, duration_hours=1)])
        with self.assertRaises(ValueError):
            Schedule(case, [Disturbance('demand_scale', index=1)])

    def test_time_alignment_indices_and_mapping_input(self):
        case = triangle_case()
        for event in [Disturbance('generator_outage', index=0, start_hour=.01),
                      Disturbance('generator_outage', index=0, duration_hours=0),
                      Disturbance('generator_outage', index=-1),
                      Disturbance('line_outage', index=100),
                      Disturbance('generator_outage', index=True),
                      Disturbance('unknown')]:
            with self.assertRaises(ValueError): Schedule(case, [event])
        schedule = Schedule(case, [{'kind': 'demand_scale', 'value': 1.2,
                                   'start_hour': 1/12, 'duration_hours': 1/12}])
        self.assertEqual(schedule.at(0)[2], 1.)
        self.assertEqual(schedule.at(1)[2], 1.2)
        self.assertEqual(schedule.at(2)[2], 1.)
        with self.assertRaises(ValueError): schedule.at(-1)
        with self.assertRaises(ValueError): schedule.at(0, dt_seconds=60)

    def test_line_resolves_explicit_bus_ids_without_inventing_row_index(self):
        case = triangle_case()
        event = Disturbance('line_outage', start_hour=96, duration_hours=24,
                            from_bus_id=1, to_bus_id=2)
        schedule = Schedule(case, [event])
        self.assertEqual(schedule.events[0].index, 0)
        self.assertEqual(schedule.at(96 * 12)[0].branch_limit[0], 0)
        with self.assertRaisesRegex(ValueError, 'finite'):
            Schedule(case, [Disturbance('generator_outage', index=1)])


if __name__ == '__main__': unittest.main()
