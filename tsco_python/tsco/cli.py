"""Command-line entry points for input checks, training and evaluation."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import json
import sys
from .config import load_config, missing_fields, OPTIONAL_EMPTY
from .data import load_dataset
from .features import Normalization
from .simulation import Simulator, VARIANTS
from .training import calibrate, train_agent
from .statistics import summarize_runs
from .artifacts import save_run, write_json, plot_run, environment_report


def parser():
    p = argparse.ArgumentParser(description='TSCO CPN-grid scheduling')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('check-inputs', help='Check configuration and input fields')
    q.add_argument('--data', required=True)
    q.add_argument('--config', default='configs/nominal.json')
    q = sub.add_parser('calibrate', help='Compute normalization from supplied training samples')
    q.add_argument('--data', required=True)
    q.add_argument('--config', default='configs/nominal.json')
    q.add_argument('--output', required=True)
    q.add_argument('--samples')
    for name in ('train', 'evaluate'):
        q = sub.add_parser(name)
        q.add_argument('--data', required=True)
        q.add_argument('--config', default='configs/nominal.json')
        q.add_argument('--output', required=True)
        q.add_argument('--planning', choices=['benders', 'extensive'], default='benders')
        q.add_argument('--seed', type=int, default=0)
        q.add_argument('--variant', choices=VARIANTS, default='tsco')
        q.add_argument('--device', default='cpu')
        if name == 'train':
            q.add_argument('--steps', type=int)
            q.add_argument('--episode-hours', type=float)
            q.add_argument('--validation-hours', type=float)
            q.add_argument('--normalization')
        else:
            q.add_argument('--checkpoint', required=True,
                           help='Checkpoint path; {seed} selects a separate trained model per seed')
            q.add_argument('--hours', type=float, default=168)
            q.add_argument('--split', choices=['validation', 'test'], default='test')
            q.add_argument('--seeds', type=int, nargs='+')
            q.add_argument('--plots', action='store_true')
            q.add_argument('--events', help='JSON event configuration')
    q = sub.add_parser('plot'); q.add_argument('--run', required=True)
    q = sub.add_parser('environment'); q.add_argument('--output')
    return p


def check_inputs(directory, config):
    directory = Path(directory)
    errors = ['config.' + key for key in missing_fields(config, optional=OPTIONAL_EMPTY)]
    for name in ('case.json', 'jobs.json', 'manifest.json', 'planning_scenarios.json',
                 'initial_states.json', 'normalization.json'):
        path = directory/name
        if not path.is_file():
            errors.append(name)
            continue
        try:
            values = json.loads(path.read_text(encoding='utf-8'))
            errors.extend(key for key in missing_fields(values, name)
                          if not key.endswith(('.expected_value_storage_modes', '.cpn_demand_mw')))
        except (ValueError, TypeError) as exc:
            errors.append(f'{name}: {exc}')
    signal_path = directory/'signals.csv'
    if not signal_path.is_file():
        errors.append('signals.csv')
    else:
        import csv
        with signal_path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            errors.append('signals.csv: data rows')
        elif any(any(value is None or not value.strip() for value in row.values()) for row in rows):
            errors.append('signals.csv: empty cells')
    if errors:
        print(json.dumps({'complete': False, 'missing_fields': errors}, ensure_ascii=False, indent=2))
        return False
    for split in ('train', 'validation', 'test'):
        load_dataset(directory, split, config)
    Normalization.load(directory/'normalization.json')
    print(json.dumps({'complete': True}, ensure_ascii=False))
    return True


def evaluate(args, config):
    train = load_dataset(args.data, 'train', config)
    data = load_dataset(args.data, args.split, config)
    if args.hours * 3600 / data.dt_seconds > len(data.base_load):
        raise ValueError(f'Requested horizon exceeds the {args.split} partition.')
    seeds = args.seeds if args.seeds is not None else [args.seed]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    runs = []
    from .learning import DQNAgent
    for seed in seeds:
        conf = deepcopy(config)
        conf['simulation']['seed'] = seed; conf['drl']['seed'] = seed
        checkpoint = Path(args.checkpoint.format(seed=seed))
        policy = DQNAgent.load(checkpoint, args.device)
        if 'normalization' not in policy.metadata:
            raise ValueError('Checkpoint lacks normalization.')
        normalizer = Normalization.from_dict(policy.metadata['normalization'])
        if policy.metadata.get('variant') != args.variant:
            raise ValueError('Checkpoint variant does not match --variant.')
        disturbances = None
        if getattr(args, 'events', None):
            from .disturbances import Schedule
            disturbances = Schedule(data.case, json.loads(Path(args.events).read_text(encoding='utf-8')))
        result = Simulator(data, train, conf, args.variant, normalizer,
                           args.planning, disturbances).run(policy, hours=args.hours)
        result.metrics['evaluation_policy'] = 'trained_checkpoint'
        run_dir = output/f'seed_{seed:02d}'
        save_run(result, run_dir, conf, data.manifest)
        if args.plots and result.intervals:
            plot_run(run_dir)
        runs.append(result.metrics)
        print(json.dumps({'seed': seed, 'status': result.metrics['status'],
                          'cost_kusd': result.metrics['operational_cost_kusd'],
                          'emissions_t': result.metrics['emissions_t'],
                          'failure_reason': result.metrics['failure_reason']}, ensure_ascii=False))
    write_json(output/'runs.json', runs)
    write_json(output/'summary.json', summarize_runs(runs))
    return runs


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'plot':
            plot_run(args.run); return 0
        if args.command == 'environment':
            report = environment_report()
            if args.output:
                write_json(args.output, report)
            print(json.dumps(report, indent=2)); return 0
        config = load_config(args.config)
        if args.command == 'check-inputs':
            return 0 if check_inputs(args.data, config) else 2
        if args.command == 'calibrate':
            calibrate(args.data, config, args.output, samples_path=args.samples)
            return 0
        config['simulation']['seed'] = args.seed
        config['drl']['seed'] = args.seed
        if args.command == 'train':
            _, summary = train_agent(args.data, config, args.output,
                variant=args.variant, device=args.device, steps=args.steps,
                episode_hours=args.episode_hours, validation_hours=args.validation_hours,
                planning_mode=args.planning, normalization_path=args.normalization)
            print(json.dumps({k: v for k, v in summary.items()
                              if k not in ('normalization', 'input_manifest')}, indent=2))
        elif args.command == 'evaluate':
            if any(run['failed'] for run in evaluate(args, config)):
                return 2
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
