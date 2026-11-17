"""Training-only feature/reward calibration; immutable during evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .schema import Observation


def make_observation(cpn, task, lmp, mci, headroom, energy, case,
                     base_load, renewable, remaining_budget, queue, mask):
    free = cpn.free_resources()
    used = cpn.used_resources()
    waiting = np.zeros(cpn.n_sites)
    for ex in cpn.executions.values():
        if ex.start > cpn.now:
            waiting[ex.node] += 1
    nodes = np.column_stack((free, used / cpn.capacities, waiting,
                             cpn.power(), lmp[case.cpn_bus],
                             mci[case.cpn_bus],
                             cpn.capacities * cpn.reservation_fraction - used,
                             headroom[case.cpn_bus]))
    if not np.isfinite(nodes).all():
        raise ValueError('Node features contain missing or nonfinite values.')
    available = cpn.available_bandwidth()
    # CPN communication matrices use source,destination; GAT uses destination,source.
    edges = np.stack((available.T, cpn.latency.T), axis=-1)
    adjacency = (cpn.bandwidth.T > 0) | np.eye(cpn.n_sites, dtype=bool)
    if task is None:
        tx = np.zeros(10)
    else:
        compatible = np.zeros(3)
        compatible[list(task.compatible)] = 1
        tx = np.r_[task.workload, task.data_gb, task.demands, compatible,
                   cpn.jobs[task.job_id].deadline_seconds - cpn.now,
                   cpn.critical_path_seconds(task.id)]
    phase = 2 * np.pi * (cpn.now % 86400) / 86400
    gx = np.r_[renewable.sum(), np.mean(energy / case.storage_capacity)
               if case.n_storage else 0.0, base_load.sum(), cpn.power().sum(),
               np.sin(phase), np.cos(phase), remaining_budget, queue]
    if not all(np.isfinite(v).all() for v in (edges,tx,gx)):
        raise ValueError('Observation features contain missing or nonfinite values.')
    return Observation(nodes.astype(np.float32), edges.astype(np.float32),
                       adjacency, tx.astype(np.float32), gx.astype(np.float32),
                       np.asarray(mask, dtype=bool))


class Normalization:
    fields = ('nodes', 'edges', 'task', 'global_features')

    reward_fields=('revenue','cost','carbon','tardiness')

    def __init__(self, feature_scales=None, reward_scales=None, provenance=None, feature_offsets=None):
        self.feature_scales = {k: None if v is None else np.asarray(v, float)
                               for k, v in (feature_scales or {}).items()}
        self.feature_offsets = {k: None if v is None else np.asarray(v, float)
                                for k, v in (feature_offsets or {}).items()}
        self.reward_scales = dict(reward_scales or {})
        self.provenance = dict(provenance or {})

    def validate(self):
        if self.provenance.get('split')!='train':
            raise ValueError('Normalization must identify the training partition.')
        for name in self.fields:
            scale=self.feature_scales.get(name);offset=self.feature_offsets.get(name)
            if scale is None or offset is None:
                raise ValueError('Missing normalization feature_scales/feature_offsets: '+name)
            if scale.ndim!=1 or offset.shape!=scale.shape or not np.isfinite(scale).all() or not np.isfinite(offset).all() or np.any(scale<=0):
                raise ValueError('Invalid supplied normalization feature statistics: '+name)
        for name in self.reward_fields:
            scale=self.reward_scales.get(name)
            if scale is None or not np.isfinite(scale) or scale<=0:
                raise ValueError('Missing or invalid normalization reward_scales.'+name)
        return self

    @classmethod
    def fit(cls, observations, rewards, provenance, feature_scales=None, feature_offsets=None):
        if provenance.get('split') != 'train':
            raise ValueError('Normalization may only be fitted on the training partition.')
        if not observations or not rewards:
            raise ValueError('Calibration needs at least one training observation.')
        if feature_scales is None or feature_offsets is None:
            raise ValueError('Supply training-derived feature_scales and feature_offsets; their estimator is not specified.')
        for name in cls.fields:
            arrays = [np.asarray(getattr(o, name), float).reshape(-1, getattr(o, name).shape[-1])
                      for o in observations]
            if not np.isfinite(np.concatenate(arrays)).all():
                raise ValueError('Calibration observations contain missing or nonfinite values.')
        samples={}
        for name in cls.reward_fields:
            if any(r.get(name) is None for r in rewards):
                raise ValueError('Missing supplied calibration reward component: '+name)
            samples[name]=np.asarray([r[name] for r in rewards],float)
            if not np.isfinite(samples[name]).all():raise ValueError('Nonfinite calibration reward: '+name)
        rs = {name: max(float(np.quantile(np.abs(values), .95)), 1e-8)
              for name,values in samples.items()}
        return cls(feature_scales, rs, provenance,feature_offsets).validate()

    def transform(self, obs):
        self.validate()
        if not all(np.isfinite(np.asarray(getattr(obs,k))).all() for k in self.fields):
            raise ValueError('Cannot normalize missing or nonfinite observations.')
        values = {k: np.clip((np.asarray(getattr(obs, k))-self.feature_offsets[k]) / self.feature_scales[k], -5, 5).astype(np.float32)
                  for k in self.fields}
        return Observation(values['nodes'], values['edges'], obs.adjacency.copy(),
                           values['task'], values['global_features'], obs.mask.copy())

    def reward(self, raw, weights, queue_ratio=0.0):
        self.validate()
        for key in (*self.reward_fields,'deferral','failure'):
            if raw.get(key) is None or not np.isfinite(raw[key]):
                raise ValueError('Missing reward component: '+key)
        z = {k: float(np.clip(raw[k] / self.reward_scales[k], -5, 5))
             for k in self.reward_fields}
        return (weights['revenue']*z['revenue'] - weights['cost']*z['cost']
                - weights['carbon']*z['carbon'] - weights['tardiness']*z['tardiness']
                - weights['queue']*queue_ratio*z['carbon']
                - weights['deferral']*raw['deferral']
                - weights['failure']*raw['failure'])

    def to_dict(self):
        return {'feature_scales': {k:None if v is None else v.tolist() for k,v in self.feature_scales.items()},
                'feature_offsets': {k:None if v is None else v.tolist() for k,v in self.feature_offsets.items()},
                'reward_scales':self.reward_scales, 'provenance':self.provenance}

    @classmethod
    def from_dict(cls, data):
        return cls(data.get('feature_scales'), data.get('reward_scales'),
                   data.get('provenance'),data.get('feature_offsets')).validate()

    def save(self,path):
        Path(path).write_text(json.dumps(self.to_dict(),indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


