"""Physical invariants and event semantics for the CPN simulator.
"""
import math
import unittest
import numpy as np
from tsco.cpn import CPNSimulator, Job, Task, jobs_from_records, jobs_to_records


def make_task(tid, job="j", predecessors=(), workload=10.0, demand=1.0, data=0.0,
              edge_data=None):
    return Task(tid, job, workload, (demand, 0., 0.), (0,), tuple(predecessors),
                data, revenue=1.0, edge_data_gb=edge_data)


def config(**updates):
    c = {"capacities": [2, 2, 2], "flops_per_unit_second": [1, 1, 1],
         "idle_mw": [0.1, 0.1, 0.1], "peak_mw": [1, 1, 1],
         "alpha": [.5, .5, .5], "bandwidth_gbps": 2,
         "latency_seconds": .5, "deferral_seconds": 300,
         "reservation_fraction": 1.0}
    c.update(updates)
    return {"cpn": c}


class CPNTests(unittest.TestCase):
    def test_precedence_and_resource_release(self):
        a, b, c = make_task("a"), make_task("b", predecessors=("a",), edge_data={"a": 0.}), make_task("c")
        env = CPNSimulator(2, config(), [Job("j", 0, 100, [a, b, c])])
        self.assertEqual([t.id for t in env.ready_tasks()], ["a", "c"])
        self.assertEqual(env.check("b", 0, 0).reason, "predecessor_incomplete")
        env.assign("a", 0, 0)
        env.assign("c", 0, 0)
        self.assertEqual(env.used_resources()[0, 0], 2)
        self.assertAlmostEqual(env.next_event_time(), 10)
        result = env.advance_to(10)
        self.assertEqual(result["completed_task_count"], 2)
        self.assertEqual(env.used_resources()[0, 0], 0)
        self.assertEqual([t.id for t in env.ready_tasks()], ["b"])
        env.assign("b", 0, 0)
        env.advance_to(20)
        self.assertEqual(env.metrics(100)["job_success_pct"], 100)

    def test_gb_seconds_and_link_serialization(self):
        a = make_task("a", workload=1)
        b = make_task("b", predecessors=("a",), workload=1, data=8, edge_data={"a": 8.})
        c = make_task("c", predecessors=("a",), workload=1, data=8, edge_data={"a": 8.})
        env = CPNSimulator(2, config(), [Job("j", 0, 100, [a, b, c])])
        env.assign("a", 0, 0)
        env.advance_to(1)
        b_info = env.assign("b", 1, 0)
        c_info = env.assign("c", 1, 0)
        self.assertAlmostEqual(b_info.start, 5.5)  # 8 GB / 2 GB/s + .5 s
        self.assertAlmostEqual(c_info.start, 9.5)
        self.assertAlmostEqual(c_info.transfers[0].start, 5.0)  # propagation does not hold link
        self.assertEqual(env.used_resources()[1, 0], 2)
        self.assertAlmostEqual(env.power()[1], .3)  # waiting inputs only consume idle compute power
        env.advance_to(5.5)
        self.assertGreater(env.power()[1], .3)
        env.advance_to(10.5)
        m = env.metrics(100)
        self.assertEqual(m["cross_node_transfers_per_completed_job"], 2)
        self.assertAlmostEqual(m["cross_node_input_workload_pct"], 200 / 3)

    def test_atomic_invalid_assignment_and_clone(self):
        a, b = make_task("a", demand=2), make_task("b", demand=1)
        env = CPNSimulator(1, config(), [Job("j", 0, 100, [a, b])])
        env.assign("a", 0, 0)
        state = env.clone()
        self.assertEqual(env.check("b", 0, 0).reason, "resource_capacity")
        with self.assertRaises(ValueError):
            env.assign("b", 0, 0)
        self.assertEqual(env.executions, state.executions)
        self.assertEqual(env.task_status, state.task_status)
        clone = env.clone()
        clone.advance_to(10)
        self.assertEqual(env.now, 0)
        self.assertEqual(env.task_status["a"], "running")

    def test_job_level_success_and_completion_based_tardiness(self):
        j1 = Job("j1", 0, 20, [make_task("a", job="j1", workload=10)])
        j2 = Job("j2", 0, 5, [make_task("b", job="j2", workload=10)])
        j3 = Job("j3", 0, 15, [make_task("c", job="j3", workload=10)])
        j4 = Job("j4", 0, 200, [make_task("d", job="j4", workload=10)])
        env = CPNSimulator(2, config(), [j1, j2, j3, j4])
        env.assign("a", 0, 0)
        env.assign("b", 1, 0)
        env.advance_to(30)
        m = env.metrics(100)
        self.assertEqual(m["evaluated_jobs"], 3)
        self.assertEqual(m["completed_jobs"], 2)
        self.assertAlmostEqual(m["job_success_pct"], 100 / 3)
        self.assertEqual(m["average_tardiness_s"], 2.5)

    def test_exact_power_integral_and_linear_ablation(self):
        job = Job("j", 0, 100, [make_task("a", workload=10)])
        env = CPNSimulator(1, config(), [job])
        linear = CPNSimulator(1, config(), [job], linear_power=True)
        env.assign("a", 0, 0)
        linear.assign("a", 0, 0)
        self.assertLess(env.power()[0], linear.power()[0])
        p = env.power()[0]
        env.advance_to(20)
        self.assertAlmostEqual(env.energy_mwh[0], (p * 10 + .3 * 10) / 3600)
        energy = env.energy_mwh.copy()
        env.advance_to(20)
        np.testing.assert_array_equal(env.energy_mwh, energy)

    def test_deferral_future_arrival_and_json_roundtrip(self):
        jobs = [Job("j", 3, 1000, [make_task("a")])]
        env = CPNSimulator(1, config(), jobs_from_records(jobs_to_records(jobs)))
        self.assertFalse(env.ready_tasks())
        self.assertEqual(env.next_event_time(), 3)
        env.advance_to(3)
        env.defer("a")
        self.assertFalse(env.ready_tasks())
        self.assertEqual(env.next_event_time(), 303)
        env.advance_to(303)
        self.assertEqual(len(env.ready_tasks()), 1)
        with self.assertRaises(ValueError):
            env.defer("a", 303)

    def test_cycle_and_incompatible_hardware(self):
        with self.assertRaises(ValueError):
            CPNSimulator(1, config(), [Job("j", 0, 100,
                [make_task("a", predecessors=("b",), edge_data={"b": 0.}),
                 make_task("b", predecessors=("a",), edge_data={"a": 0.})])])
        env = CPNSimulator(1, config(), [Job("j", 0, 100, [make_task("a")])])
        self.assertEqual(env.check("a", 0, 1).reason, "incompatible_hardware")

    def test_multi_parent_share_and_zero_link(self):
        a, b = make_task("a", workload=1), make_task("b", workload=1)
        c = make_task("c", predecessors=("a", "b"), workload=2, data=8,
                      edge_data={"a": 4., "b": 4.})
        env = CPNSimulator(3, config(), [Job("j", 0, 100, [a, b, c])])
        env.assign("a", 0, 0); env.assign("b", 1, 0); env.advance_to(1)
        cand = env.check("c", 2, 0)
        self.assertEqual([r.data_gb for r in cand.transfers], [4, 4])
        self.assertAlmostEqual(cand.start, 3.5)
        env.bandwidth[0, 2] = 0
        self.assertEqual(env.check("c", 2, 0).reason, "no_communication_link")

    def test_missing_physical_parameters_are_not_filled(self):
        for key in ("idle_mw", "peak_mw", "alpha", "flops_per_unit_second",
                    "reservation_fraction", "bandwidth_gbps", "latency_seconds"):
            for remove in (True, False):
                with self.subTest(key=key, omitted=remove):
                    values = config()
                    if remove:
                        del values["cpn"][key]
                    else:
                        values["cpn"][key] = None
                    with self.assertRaisesRegex(ValueError, key):
                        CPNSimulator(1, values, [])

    def test_revenue_and_dependency_volumes_must_be_explicit(self):
        task = Task("a", "j", 1., (1., 0., 0.), (0,), (), 0.)
        with self.assertRaisesRegex(ValueError, "explicit revenue"):
            CPNSimulator(1, config(), [Job("j", 0, 100, [task])])
        a = make_task("a")
        b = make_task("b", predecessors=("a",), data=8)
        with self.assertRaisesRegex(ValueError, "explicit edge_data_gb"):
            CPNSimulator(1, config(), [Job("j", 0, 100, [a, b])])

    def test_unspecified_deferral_requires_explicit_end(self):
        env = CPNSimulator(1, config(deferral_seconds=None),
                           [Job("j", 0, 100, [make_task("a")])])
        with self.assertRaisesRegex(ValueError, "end time is unspecified"):
            env.defer("a")
        env.defer("a", until=7)
        self.assertEqual(env.next_event_time(), 7)


if __name__ == "__main__":
    unittest.main()
