"""Training/validation orchestration with held-out data and auditable checkpoints."""
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager
import random
import time
import json
import numpy as np
from .data import load_dataset
from .features import Normalization
from .schema import Observation
from .simulation import Simulator
from .artifacts import save_run,write_json


def validation_plateau(records, patience, fraction):
    """Require consecutive feasible checks, not a filtered feasible history.

    One initial check and ``patience`` subsequent checks define the window.
    Any infeasible or incomplete-budget check interrupts the sequence.
    """
    patience=int(patience)
    if patience<1:raise ValueError('early_stop_checks must be positive')
    recent=records[-(patience+1):]
    if len(recent)<patience+1 or not all(
            r.get('eligible_full_horizon_checkpoint',False) for r in recent):
        return False
    returns=np.asarray([r['return'] for r in recent],float)
    if not np.isfinite(returns).all():return False
    improvement=(float(np.max(returns[1:]))-returns[0])/max(abs(returns[0]),1.)
    return bool(improvement<float(fraction))


@contextmanager
def preserve_training_rng(agent):
    """Validation has no side effects on training RNGs, even when it raises."""
    import torch
    py_state=random.getstate();np_state=np.random.get_state()
    torch_state=torch.get_rng_state()
    cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    agent_state=deepcopy(agent.rng.bit_generator.state)
    replay_state=deepcopy(agent.replay.rng.bit_generator.state)
    modules=list(agent.online.modules())+list(agent.target.modules())
    modes=[m.training for m in modules]
    noise=[getattr(m,'force_noise',None) for m in modules]
    try:
        yield
    finally:
        random.setstate(py_state);np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:torch.cuda.set_rng_state_all(cuda_state)
        agent.rng.bit_generator.state=agent_state
        agent.replay.rng.bit_generator.state=replay_state
        for module,mode,force in zip(modules,modes,noise):
            module.training=mode
            if force is not None:module.force_noise=force


def flush_observed_tails(agent):
    """Flush shorter n-step prefixes at a training cap without inventing termination.

    Every prefix bootstraps from its last *observed* next state. This neither
    adds an environment transition nor labels a time-limit cut as physical
    termination. Normal complete episodes are already flushed by observe().
    """
    while agent.pending:
        agent._emit_prefix()


def observation_template(normalizer,n_sites):
    """Construct array shapes for network initialization, not experimental inputs."""
    normalizer.validate()
    expected={'nodes':14,'edges':2,'task':10,'global_features':8}
    for field,width in expected.items():
        if normalizer.feature_scales[field].shape!=(width,):
            raise ValueError(f'Normalization {field} requires {width} feature entries.')
    return Observation(np.zeros((n_sites,14),np.float32),
                       np.zeros((n_sites,n_sites,2),np.float32),
                       np.eye(n_sites,dtype=bool),np.zeros(10,np.float32),
                       np.zeros(8,np.float32),np.ones(1+3*n_sites,bool))


def calibrate(directory,config,output,samples_path=None):
    """Use supplied training samples and feature statistics; never run a policy."""
    began=time.perf_counter()
    if samples_path is None:
        raise ValueError('Supply a calibration sample file; automatic calibration rollouts are not defined.')
    data=json.loads(Path(samples_path).read_text(encoding='utf-8'))
    if not data.get('observations') or not data.get('rewards'):
        raise ValueError('Supply nonempty training observations and raw reward samples.')
    observations=[]
    for item in data['observations']:
        fields=('nodes','edges','adjacency','task','global_features','mask')
        if any(item.get(k) is None for k in fields):
            raise ValueError('Every calibration observation must provide all six observation fields.')
        observations.append(Observation(*(np.asarray(item[k],dtype=bool if k in ('adjacency','mask') else np.float32)
                                          for k in fields)))
    provenance=dict(data.get('provenance') or {})
    provenance['sample_file']=str(Path(samples_path))
    normalization=Normalization.fit(observations,data['rewards'],provenance,
                                    data.get('feature_scales'),data.get('feature_offsets'))
    normalization.provenance['calibration_wall_clock_seconds']=time.perf_counter()-began
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    normalization.save(output/'normalization.json')
    return normalization,observations[0]


def train_agent(directory,config,output,variant='tsco',device='cpu',steps=None,
                episode_hours=None,validation_hours=None,
                planning_mode='benders',normalization_path=None):
    overall_began=time.perf_counter()
    from .learning import DQNAgent
    import torch
    if config['drl'].get('torch_threads') is not None:
        torch.set_num_threads(int(config['drl']['torch_threads']))
    if normalization_path is None:
        raise ValueError('Supply normalization_path with training-derived normalization statistics.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    train=load_dataset(directory,'train',config);validation=load_dataset(directory,'validation',config)
    normalizer=Normalization.load(normalization_path)
    template=observation_template(normalizer,train.case.n_cpn)
    normalizer.save(output/'normalization.json')
    if normalizer.provenance.get('split')!='train':raise ValueError('Untrusted normalization split.')
    architecture=variant if variant in ('flat_dqn','no_quantile') else 'tsco'
    if variant=='no_fs':template.mask[:]=True
    agent=DQNAgent(template,config,architecture,device)
    target_steps=int(config['drl']['max_steps'] if steps is None else steps)
    episode_hours=float(config['simulation']['horizon_hours'] if episode_hours is None else episode_hours)
    validation_hours=float(episode_hours if validation_hours is None else validation_hours)
    if target_steps<1 or not np.isfinite([episode_hours,validation_hours]).all() or min(episode_hours,validation_hours)<=0:
        raise ValueError('Training steps and episode/validation horizons must be positive.')
    if episode_hours*12>len(train.base_load) or validation_hours*12>len(validation.base_load):
        raise ValueError('Partitions are shorter than the requested episode/validation horizon.')
    starts=config['drl'].get('training_episode_starts')
    validation_start=config['drl'].get('validation_start_interval')
    if not isinstance(starts,list) or not starts or any(type(x) is not int or x<0 for x in starts):
        raise ValueError('Supply drl.training_episode_starts as an explicit sequence of interval indices.')
    if type(validation_start) is not int or validation_start<0:
        raise ValueError('Supply drl.validation_start_interval.')
    losses=[];validations=[];episodes=[];best_return=-np.inf;early_stop=False
    began=time.perf_counter();total_ed_calls=0;training_ed_calls=0;validation_ed_calls=0
    validation_seconds=0.
    metadata={'variant':variant,'normalization':normalizer.to_dict(),
              'input_manifest':train.manifest,'training_complete':False,
              'validated_full_budget_horizon':False}
    val_data=validation.window(validation_start,int(validation_hours*12))

    def validate():
        nonlocal best_return,early_stop,total_ed_calls,validation_ed_calls,validation_seconds
        # Evaluation is deterministic and must not perturb exploration RNG state.
        conf=deepcopy(config)
        rewards=[]
        def observe_only(tr):rewards.append(tr.reward)
        validation_began=time.perf_counter()
        with preserve_training_rng(agent):
            sim=Simulator(val_data,train,conf,variant,normalizer,planning_mode)
            result=sim.run(agent,hours=validation_hours,transition_callback=observe_only)
        validation_seconds+=time.perf_counter()-validation_began
        total_ed_calls+=result.metrics['ed_invocations']
        validation_ed_calls+=result.metrics['ed_invocations']
        record={**result.metrics,'step':agent.steps,'return':float(sum(rewards))}
        validations.append(record)
        save_run(result,output/f'validation_{agent.steps:09d}',conf,val_data.manifest)
        full=result.metrics['budget_horizon_complete']
        qos=result.metrics['job_success_pct']
        eligible=(not result.metrics['failed'] and full and
                  result.metrics['terminal_budget_violation'] is False and
                  np.isfinite(qos) and qos>=config['drl']['validation_qos_min'])
        record['eligible_full_horizon_checkpoint']=eligible
        if eligible and record['return']>best_return:
            best_return=record['return']
            agent.save(output/'best_feasible.pt',
                       {**metadata,'validated_full_budget_horizon':True,'last_validation':record})
        # A short smoke validation is not promoted to a full-budget feasible checkpoint.
        agent.save(output/'latest_validation.pt',
                   {**metadata,'validated_full_budget_horizon':bool(eligible),'last_validation':record})
        patience=config['drl']['early_stop_checks']
        early_stop=validation_plateau(validations,patience,config['drl']['early_stop_fraction'])

    def learn_transition(transition):
        if agent.steps>=target_steps or early_stop:return
        agent.observe(transition)
        update=agent.learn()
        if update is not None:losses.append({'step':agent.steps,**update})
        if agent.steps%config['drl']['validation_interval']==0:validate()

    episode=0
    while agent.steps<target_steps and not early_stop:
        start=starts[episode%len(starts)]
        data=train.window(start,int(episode_hours*12))
        conf=deepcopy(config)
        sim=Simulator(data,train,conf,variant,normalizer,planning_mode)
        before=agent.steps
        result=sim.run(agent,hours=episode_hours,explore=True,transition_callback=learn_transition,
                       max_decisions=target_steps-agent.steps)
        flush_observed_tails(agent)
        total_ed_calls+=result.metrics['ed_invocations'];episode+=1
        training_ed_calls+=result.metrics['ed_invocations']
        episodes.append({**result.metrics,'episode':episode,'steps_after':agent.steps,'window_start_interval':start})
        if result.metrics['failed']:
            # Preserve diagnosis and stop; do not silently discard physically failed runs.
            save_run(result,output/'failed_training_episode',conf,data.manifest)
            metadata['training_failure']=result.metrics['failure_reason'];break
        if agent.steps==before:
            raise RuntimeError('Training episode produced no decisions; inspect workload/window.')
        agent.save(output/'last.pt',metadata)
    final_validation_current=bool(validations and validations[-1]['step']==agent.steps
                                  and validations[-1].get('eligible_full_horizon_checkpoint',False))
    metadata.update(training_complete=agent.steps>=target_steps or early_stop,
                    actual_transitions=agent.steps,optimizer_updates=agent.updates,
                    ed_invocations_training_and_validation=total_ed_calls,
                    ed_invocations_training=training_ed_calls,
                    ed_invocations_validation=validation_ed_calls,
                    initialization_hours=(began-overall_began)/3600,
                    validation_rollout_hours=validation_seconds/3600,
                    wall_clock_hours=(time.perf_counter()-began)/3600,
                    wall_clock_scope='Training loop including validation and checkpoint I/O; excludes input normalization preparation and initialization',
                    total_job_wall_clock_hours=(time.perf_counter()-overall_began)/3600,
                    validated_full_budget_horizon=final_validation_current,
                    best_feasible_found=bool(np.isfinite(best_return)),
                    early_stopped=early_stop,requested_transitions=target_steps)
    agent.save(output/'last.pt',metadata)
    write_json(output/'training_summary.json',metadata)
    write_json(output/'training_episodes.json',episodes)
    write_json(output/'validation_history.json',validations)
    write_json(output/'learning_curve.json',losses)
    return agent,metadata
