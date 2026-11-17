"""Configuration and input validation for TSCO."""
from __future__ import annotations
import copy
import json
from pathlib import Path

DEFAULTS = {
    'simulation': {'dt_seconds': 300, 'horizon_hours': 168,
                   'carbon_price': 50.0, 'carbon_budget_t': 1450.0,
                   'budget_horizon_hours': 168, 'seed': 0,
                   'max_decisions_per_interval': None,
                   'max_events_per_interval': None},
    'solver': {'backend': 'gurobi', 'feasibility_tol': 1e-7,
               'optimality_tol': 1e-7, 'int_feas_tol': 1e-6,
               'mip_gap': 1e-4, 'threads': 8, 'ed_threads': 1,
               'ed_method': 1, 'lp_warm_start': 2, 'time_limit': None,
               'pwl_segments': 20, 'mci_delta_mw': 0.1,
               'startup_ramp_mw': None, 'shutdown_ramp_mw': None},
    'planning': {'relative_gap': 0.002, 'max_iterations': None,
                 'time_limit_seconds': None, 'min_up_hours': None,
                 'min_down_hours': None, 'terminal_storage': None,
                 'scenario_count': 100, 'horizon_hours': 24},
    'closed_loop': {'rho': 0.5, 'epsilon_mw': 0.01, 'max_iterations': 10},
    'cpn': {'capacities': [128, 64, 32], 'site_peak_mw': 4.0,
            'idle_mw': None, 'peak_mw': None, 'alpha': None,
            'flops_per_unit_second': None, 'bandwidth_gbps': None,
            'latency_seconds': None, 'reservation_fraction': None,
            'deferral_seconds': None},
    'workload': {'jobs_per_day': 800},
    'data': {'split': [0.70, 0.15, 0.15]},
    'drl': {'gamma': 0.99, 'n_step': 3, 'batch_size': 256,
            'replay_capacity': 500000, 'warmup_steps': 20000,
            'learning_rate': 1e-4, 'grad_clip': 10.0,
            'per_alpha': 0.6, 'per_beta_start': 0.4, 'per_beta_end': 1.0,
            'per_beta_steps': None, 'anneal_rule': None,
            'per_priority_epsilon': None, 'noisy_sigma': 0.5,
            'hardware_embedding_dim': None, 'value_hidden_dims': None,
            'gat_negative_slope': None, 'adam_betas': None,
            'adam_eps': None, 'adam_weight_decay': None,
            'target_update': 5000, 'n_quantiles': 32,
            'huber_kappa': 1.0, 'dropout': 0.1,
            'max_steps': 2000000, 'validation_interval': 25000,
            'early_stop_checks': 10, 'early_stop_fraction': 0.01,
            'validation_qos_min': 98.0, 'seed': 0,
            'epsilon_decay_steps': 200000, 'epsilon_final': 0.05,
            'training_episode_starts': None, 'validation_start_interval': None},
    'reward': {'weights': {'revenue': 1.0, 'cost': 0.30,
                           'carbon': 0.40, 'tardiness': 1.0,
                           'queue': 0.50, 'deferral': 0.05,
                           'failure': 2.0}, 'clip': 5.0},
}

# None disables an optional execution cap; it never fills physical input data.
OPTIONAL_EMPTY = {'simulation.max_decisions_per_interval',
                  'simulation.max_events_per_interval', 'solver.time_limit',
                  'planning.max_iterations', 'planning.time_limit_seconds'}


def merge_config(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def missing_fields(value, prefix='', optional=()):
    """List empty fields without assigning replacement values."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return [] if prefix in optional else [prefix]
    if isinstance(value, dict):
        return [field for key, item in value.items()
                for field in missing_fields(item, f'{prefix}.{key}'.strip('.'), optional)]
    if isinstance(value, (list, tuple)):
        return [field for i, item in enumerate(value)
                for field in missing_fields(item, f'{prefix}[{i}]', optional)]
    return []


def load_config(path=None):
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, encoding='utf-8') as handle:
            config = merge_config(config, json.load(handle))
    validate_config(config)
    return config


def validate_config(config, require_complete=False):
    if config['simulation']['dt_seconds'] != 300:
        raise ValueError('simulation.dt_seconds must be 300.')
    if not 0 < config['closed_loop']['rho'] <= 1:
        raise ValueError('closed_loop.rho must lie in (0,1].')
    if config['closed_loop']['max_iterations'] < 1:
        raise ValueError('closed_loop.max_iterations must be positive.')
    if config['simulation']['carbon_budget_t'] <= 0:
        raise ValueError('simulation.carbon_budget_t must be positive.')
    fractions = config['data']['split']
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or abs(sum(fractions)-1) > 1e-9:
        raise ValueError('data.split must contain three positive fractions summing to one.')
    for key in ('capacities','idle_mw','peak_mw','alpha','flops_per_unit_second'):
        value = config['cpn'][key]
        if value is not None and len(value) != 3:
            raise ValueError('CPU/GPU/TPU configuration requires three entries: cpn.'+key)
    if require_complete:
        missing = missing_fields(config, optional=OPTIONAL_EMPTY)
        if missing:
            raise ValueError('Missing configuration fields: '+', '.join(missing))


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
