"""End-to-end invariants for physical accounting and trial-policy isolation."""
import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np

from tsco.config import load_config
from tsco.schema import GridCase, Observation
from tsco.cpn import Task, Job
from tsco.data import Dataset
from tsco.features import Normalization
from tsco.simulation import Simulator


def case_fixture():
    f=lambda x:np.asarray(x,float)
    i=lambda x:np.asarray(x,int)
    return GridCase('controller fixture',100.,f([0,8]),i([0]),i([1]),
                    f([.1]),f([1]),f([10]),i([0]),f([0]),f([100]),f([120]),
                    f([0]),f([10]),f([0]),f([.8]),f([1]),f([1]),
                    i([]),f([]),f([]),f([]),f([]),f([]),f([]),
                    i([1]),f([20]),i([1]))


def fixture(intervals=2):
    case=case_fixture()
    cfg=load_config()
    cfg['cpn'].update(capacities=[1,1,1],idle_mw=[0,0,0],peak_mw=[1,1,1],
                      alpha=[0,0,0],flops_per_unit_second=[1,1,1],
                      reservation_fraction=1.,bandwidth_gbps=1.,latency_seconds=0.)
    cfg['solver'].update(backend='scipy',startup_ramp_mw=[100.],shutdown_ramp_mw=[100.])
    cfg['closed_loop'].update(max_iterations=4,epsilon_mw=.001)
    task=Task('t','j',60.,(1.,0.,0.),(0,),(),0.,1.,{})
    job=Job('j',0.,100.,[task])
    dataset=Dataset(case,np.tile([0.,8.],(intervals,1)),np.zeros((intervals,1)),
                    np.arange(intervals)*300.,[job],300.,
                    {},'2024-01-01T00:00:00Z','test')
    dataset.initial_state=dict(commitment=[1],generation_mw=[8.],storage_energy_mwh=[],
                               initial_status_hours=[2.],cpn_empty=True)
    dataset.storage_modes=np.empty((intervals,0),int)
    train=deepcopy(dataset);train.split='train'
    return dataset,train,cfg


class WrongHardwarePolicy:
    def __init__(self):
        self.seen_masks=[]
    def action_values(self,obs,explore=False):
        self.seen_masks.append(obs.mask.copy())
        # CPU action1 is compatible, GPU2 deliberately preferred but invalid.
        return np.array([-100.,1.,2.,0.])


def normalizer():
    dims={'nodes':14,'edges':2,'task':10,'global_features':8}
    return Normalization({k:np.ones(v) for k,v in dims.items()},
                         dict(revenue=1.,cost=1.,carbon=1.,tardiness=1.),
                         {'split':'train'},{k:np.zeros(v) for k,v in dims.items()})


def simulator(data,train,config,variant='tsco'):
    sim=Simulator(data,train,config,variant=variant,normalization=normalizer())
    # Explicit all-committed plan is a unit-test fixture, not a planning default.
    sim._plan=lambda k,remaining:(np.ones((1,1),int),0.)
    return sim


class SimulationTests(unittest.TestCase):
    def test_trials_do_not_multiply_energy_or_queue_updates(self):
        data,train,cfg=fixture(1)
        cfg['simulation']['carbon_budget_t']=1.
        one=deepcopy(cfg);one['closed_loop']['max_iterations']=1
        many=deepcopy(cfg);many['closed_loop']['max_iterations']=4
        s1=simulator(data,train,one)
        s2=simulator(data,train,many)
        r1=s1.run(WrongHardwarePolicy(),hours=1/12)
        r2=s2.run(WrongHardwarePolicy(),hours=1/12)
        self.assertFalse(r1.metrics['failed']);self.assertFalse(r2.metrics['failed'])
        self.assertEqual(r2.metrics['completed_jobs'],1)
        self.assertAlmostEqual(r2.metrics['cpn_energy_mwh'],1/60)
        self.assertAlmostEqual(r1.metrics['emissions_t'],r2.metrics['emissions_t'])
        expected=.8*(8/12+1/60)
        self.assertAlmostEqual(r2.metrics['emissions_t'],expected)
        self.assertAlmostEqual(s2.queue,expected-1/2016)
        self.assertEqual(r2.metrics['carbon_queue_updates'],1)
        self.assertGreater(r2.metrics['trial_scheduling_decisions'],r2.metrics['scheduling_decisions'])
        row=r2.intervals[0]
        self.assertAlmostEqual(row['thermal_mw']+row['used_res_mw']+row['bess_discharge_mw'],
                               row['base_mw']+row['cpn_mw']+row['bess_charge_mw'])
        self.assertLess(row['balance_residual_mw'],1e-7)
        self.assertIsNone(r2.metrics['terminal_budget_violation'])

    def test_mask_and_rejection_are_distinct_and_no_fs_does_not_leak(self):
        data,train,cfg=fixture(1)
        fullpolicy=WrongHardwarePolicy()
        full=simulator(data,train,cfg).run(fullpolicy,hours=1/12)
        self.assertEqual(full.metrics['job_success_pct'],100.)
        self.assertEqual(full.metrics['preferred_action_infeasibility_pct'],100.)
        self.assertEqual(full.metrics['selected_action_infeasibility_pct'],0.)
        self.assertTrue(all(not m[2] for m in fullpolicy.seen_masks))
        nofspolicy=WrongHardwarePolicy()
        without=simulator(data,train,cfg,variant='no_fs').run(nofspolicy,hours=1/12)
        self.assertEqual(without.metrics['job_success_pct'],0.)
        self.assertEqual(without.metrics['selected_action_infeasibility_pct'],100.)
        self.assertEqual(without.metrics['executed_action_infeasibility_pct'],0.)
        self.assertEqual(without.metrics['cpn_energy_mwh'],0.)
        self.assertTrue(all(m.all() for m in nofspolicy.seen_masks))

    def test_explicit_truncation_is_not_full_completion(self):
        data,train,cfg=fixture(3)
        second=Task('t2','j2',60.,(1.,0.,0.),(0,),(),0.,1.,{})
        data.jobs.append(Job('j2',350.,450.,[second]))
        transitions=[]
        result=simulator(data,train,cfg).run(
            WrongHardwarePolicy(),hours=3/12,max_decisions=1,transition_callback=transitions.append)
        self.assertEqual(result.metrics['status'],'truncated')
        self.assertTrue(result.metrics['truncated'])
        self.assertEqual(result.metrics['completed_intervals'],2)
        self.assertEqual(result.metrics['evaluated_jobs'],2)
        self.assertEqual(len(transitions),1)
        self.assertFalse(transitions[0].done)
        self.assertIsNone(result.metrics['terminal_budget_violation'])

    def test_normalization_training_only_and_fixed_after_transform(self):
        obs=Observation(np.array([[1.,2.]]),np.array([[[3.,4.]]]),np.ones((1,1),bool),
                        np.array([5.,6.]),np.array([7.,8.]),np.array([True,False,False,False]))
        rewards=[dict(revenue=1.,cost=2.,carbon=.1,tardiness=3.)]
        with self.assertRaises(ValueError):Normalization.fit([obs],rewards,{'split':'test'})
        norm=Normalization.fit([obs],rewards,{'split':'train'},
                               {k:getattr(obs,k).reshape(-1,2)[0] for k in Normalization.fields},
                               {k:np.zeros(2) for k in Normalization.fields})
        before=deepcopy(norm.to_dict())
        transformed=norm.transform(obs)
        self.assertEqual(before,norm.to_dict())
        np.testing.assert_allclose(transformed.nodes,1.)
        np.testing.assert_array_equal(transformed.mask,obs.mask)


if __name__=='__main__':unittest.main()
