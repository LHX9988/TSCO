"""Training-budget, validation isolation, and checkpoint eligibility checks."""
import importlib.util
import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from tsco.config import load_config
from tsco.features import Normalization
from tsco.schema import Observation,Transition
from tsco.training import validation_plateau,preserve_training_rng,train_agent


def observation():
    return Observation(np.ones((1,14),np.float32),np.ones((1,1,2),np.float32),
                       np.ones((1,1),bool),np.ones(10,np.float32),
                       np.ones(8,np.float32),np.array([True,True,False,False]))


class TrainingLogicTests(unittest.TestCase):
    def test_early_stop_requires_consecutive_feasible_checks(self):
        record=lambda eligible=True,value=100.:{'eligible_full_horizon_checkpoint':eligible,'return':value}
        self.assertTrue(validation_plateau([record(),record(),record()],2,.01))
        self.assertFalse(validation_plateau([record(),record(False),record(),record()],2,.01))
        self.assertFalse(validation_plateau([record(),record(value=105),record(value=106)],2,.01))
        self.assertFalse(validation_plateau([record(),record()],2,.01))


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch optional dependency not installed')
class TrainingTorchTests(unittest.TestCase):
    def test_validation_restores_rngs_and_network_modes_on_exception(self):
        import torch
        agent=SimpleNamespace(rng=np.random.default_rng(1),
                              replay=SimpleNamespace(rng=np.random.default_rng(2)),
                              online=torch.nn.Sequential(torch.nn.Linear(1,1)),
                              target=torch.nn.Sequential(torch.nn.Linear(1,1)))
        agent.online.train();agent.target.eval()
        torch_state=torch.get_rng_state().clone()
        global_np=deepcopy(np.random.get_state());global_py=random.getstate()
        generator_state=deepcopy(agent.rng.bit_generator.state)
        replay_state=deepcopy(agent.replay.rng.bit_generator.state)
        with self.assertRaises(RuntimeError):
            with preserve_training_rng(agent):
                torch.rand(20);np.random.rand(20);random.random()
                agent.rng.random(20);agent.replay.rng.random(20)
                agent.online.eval();agent.target.train()
                raise RuntimeError('simulated validation failure')
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        np.testing.assert_array_equal(global_np[1],np.random.get_state()[1])
        self.assertEqual(global_py,random.getstate())
        self.assertEqual(generator_state,agent.rng.bit_generator.state)
        self.assertEqual(replay_state,agent.replay.rng.bit_generator.state)
        self.assertTrue(agent.online.training);self.assertFalse(agent.target.training)

    def test_step_cap_flushes_observed_tails_and_short_validation_not_feasible(self):
        obs=observation()
        norm=Normalization({k:np.ones(getattr(obs,k).shape[-1]) for k in Normalization.fields},
                           {'revenue':1.,'cost':1.,'carbon':1.,'tardiness':1.},
                           {'split':'train'},
                           {k:np.zeros(getattr(obs,k).shape[-1]) for k in Normalization.fields})
        cfg=load_config();cfg['drl'].update(batch_size=2,warmup_steps=9999,
                                          replay_capacity=100,n_step=3,
                                          validation_interval=2,save_replay=True,
                                          training_episode_starts=[0],validation_start_interval=0,
                                          hardware_embedding_dim=7,value_hidden_dims=[11],
                                          gat_negative_slope=.1,per_priority_epsilon=1e-5,
                                          adam_betas=[.8,.98],adam_eps=1e-7,adam_weight_decay=0.,
                                          per_beta_steps=100,anneal_rule='linear',torch_threads=1)
        class FakeData:
            base_load=np.zeros((24,2))
            manifest={}
            case=SimpleNamespace(n_cpn=1)
            def window(self,*args):return self
        class FakeSimulator:
            def __init__(self,*args,**kwargs):pass
            def run(self,policy,hours=None,explore=False,transition_callback=None,max_decisions=None):
                n=7 if explore else 2
                for i in range(n):
                    transition_callback(Transition(obs,0,1.,obs,i==n-1))
                return SimpleNamespace(metrics={'failed':False,'failure_reason':'',
                                                'ed_invocations':10,'budget_horizon_complete':False,
                                                'terminal_budget_violation':None,'job_success_pct':99.})
        with tempfile.TemporaryDirectory() as directory:
            normpath=Path(directory)/'input_normalization.json';norm.save(normpath)
            with patch('tsco.training.load_dataset',return_value=FakeData()),\
                    patch('tsco.training.Simulator',FakeSimulator),\
                    patch('tsco.training.save_run'):
                agent,metadata=train_agent('unused',cfg,directory,steps=5,
                                            episode_hours=1/12,validation_hours=1/12,
                                            normalization_path=normpath)
            self.assertEqual(agent.steps,5)
            self.assertEqual(len(agent.pending),0)
            self.assertEqual(len(agent.replay),5)
            self.assertFalse(metadata['validated_full_budget_horizon'])
            self.assertFalse(metadata['best_feasible_found'])
            self.assertFalse((Path(directory)/'best_feasible.pt').exists())
            self.assertEqual(metadata['ed_invocations_training'],10)
            self.assertEqual(metadata['ed_invocations_validation'],20)
            self.assertNotIn('ed_invocations_initial_calibration',metadata)
            # The fifth accepted transition had an observed next state; the
            # artificial training-budget cap must not invent done=True.
            self.assertTrue(all(not tr.done for tr in agent.replay.data if tr is not None))


if __name__=='__main__':unittest.main()
