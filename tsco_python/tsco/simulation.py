"""Two-timescale TSCO controller with atomic trial scheduling.

CPN tasks/communication evolve in seconds inside each five-minute interval.
Only the final feasible trial is committed. Exact event-integrated CPN energy
defines the interval-average demand seen by the physical DC dispatch. Trial
ED queries never advance SOC, ramp reference, physical emissions or queue.
Conservative reserved-power ED checks prevent resource/wire overbooking.
Forecasts, operating modes, initial states, and normalization are supplied inputs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
import time
import numpy as np
from .schema import Observation, Transition
from .cpn import CPNSimulator
from .grid import EconomicDispatch
from .planning import BendersPlanner
from .features import make_observation, Normalization

VARIANTS = ('tsco','no_fs','no_quantile','flat_dqn','no_nm','no_cc',
            'no_fb','no_cr','no_vq','no_cad','no_ssb')


@dataclass
class RunResult:
    metrics: dict
    intervals: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    plans: list = field(default_factory=list)
    calibration_observations: list = field(default_factory=list)
    calibration_rewards: list = field(default_factory=list)


def scenario_forecasts(dataset, case, start_interval, hours):
    """Read the supplied day-specific scenario record without generating inputs."""
    records=getattr(dataset,'planning_scenarios',None)
    if not records:
        raise ValueError('Missing input: planning_scenarios.json')
    matches=[r for r in records if r.get('start_interval')==start_interval]
    if len(matches)!=1:
        raise ValueError(f'Expected one supplied planning scenario record at interval {start_interval}.')
    record=matches[0]
    for key in ('demand_mw','renewable_mw','probabilities','storage_modes','hour_dt'):
        if record.get(key) is None:
            raise ValueError('Missing planning input: '+key)
    demand=np.asarray(record['demand_mw'],float)
    renewable=np.asarray(record['renewable_mw'],float)
    probabilities=np.asarray(record['probabilities'],float)
    modes=np.asarray(record['storage_modes'])
    if demand.ndim!=3 or demand.shape[2]!=case.n_bus or demand.shape[1]<hours:
        raise ValueError('demand_mw must have supplied shape [scenarios,hours,buses].')
    expected=(len(demand),demand.shape[1])
    if renewable.shape!=(*expected,case.n_renewable) or modes.shape!=(*expected,case.n_storage):
        raise ValueError('Planning renewable or storage-mode dimensions do not match demand.')
    if probabilities.shape!=(len(demand),) or not np.isfinite(probabilities).all() or np.any(probabilities<0) or not np.isclose(probabilities.sum(),1.):
        raise ValueError('Supplied scenario probabilities must be nonnegative and sum to one.')
    if not np.isfinite(demand).all() or not np.isfinite(renewable).all() or not np.isin(modes,[-1,0,1]).all():
        raise ValueError('Planning data contain missing or invalid values.')
    if float(record['hour_dt'])!=1.:
        raise ValueError('Day-ahead intervals must be one hour.')
    expected_modes=record.get('expected_value_storage_modes')
    if expected_modes is not None:expected_modes=np.asarray(expected_modes)[:hours]
    return demand[:,:hours],renewable[:,:hours],probabilities,modes[:,:hours],expected_modes


class Simulator:
    def __init__(self, dataset, training_data, config, variant='tsco',
                 normalization=None, planning_mode='benders', disturbances=None):
        if variant not in VARIANTS:
            raise ValueError(f'Unsupported variant {variant}; available: {VARIANTS}')
        if training_data.split!='train':
            raise ValueError('training_data must identify the training partition.')
        self.data,self.train,self.case=dataset,training_data,dataset.case
        self.config=deepcopy(config);self.variant=variant
        self.normalization=normalization
        self.disturbances=disturbances
        self.planning_mode=planning_mode
        if planning_mode not in ('benders','extensive'):
            raise ValueError('planning_mode must be benders or extensive')
        self.cpn=CPNSimulator(self.case.n_cpn,config,dataset.jobs,
                             linear_power=variant=='no_nm',
                             ignore_communication=variant=='no_cc')
        self.ed=EconomicDispatch(self.case,config)
        self.seed=int(config['simulation']['seed'])
        initial=getattr(dataset,'initial_state',None)
        if not isinstance(initial,dict):raise ValueError('Missing supplied initial_state.')
        for key in ('commitment','generation_mw','storage_energy_mwh','initial_status_hours','cpn_empty'):
            if initial.get(key) is None:raise ValueError('Missing initial_state.'+key)
        if initial['cpn_empty'] is not True:
            raise ValueError('A nonempty initial CPN requires explicit ongoing-task state support.')
        self.energy=np.asarray(initial['storage_energy_mwh'],float)
        self.previous_gen=np.asarray(initial['generation_mw'],float)
        self.status_hours=np.asarray(initial['initial_status_hours'],float)
        initial_u=np.asarray(initial['commitment'])
        if initial_u.shape!=(self.case.n_gen,) or not np.isin(initial_u,[0,1]).all():
            raise ValueError('Initial commitment must contain one supplied binary value per generator.')
        self.previous_commitment=initial_u.astype(int)
        if self.status_hours.shape!=(self.case.n_gen,) or not np.isfinite(self.status_hours).all() or np.any(self.status_hours<0):
            raise ValueError('Initial generator status durations must be supplied in hours.')
        if self.previous_gen.shape!=(self.case.n_gen,) or self.energy.shape!=(self.case.n_storage,) or not np.isfinite(self.previous_gen).all() or not np.isfinite(self.energy).all():
            raise ValueError('Initial generation and storage energy must be supplied finite arrays.')
        modes=getattr(dataset,'storage_modes',None)
        if modes is None:raise ValueError('Missing supplied interval storage modes.')
        self.storage_modes=np.asarray(modes)
        if self.storage_modes.shape!=(len(dataset.base_load),self.case.n_storage) or not np.isin(self.storage_modes,[-1,0,1]).all():
            raise ValueError('Interval storage modes must have shape [intervals,storage units].')
        self.storage_mode=self.storage_modes[0].astype(int)
        self.commitment=self.previous_commitment.copy()
        self.queue=0.;self.emissions=0.
        self.budget=config['simulation']['carbon_budget_t']
        self.budget_intervals=int(config['simulation']['budget_horizon_hours']*12)
        self.weight=deepcopy(config['reward']['weights'])
        if variant in ('no_cr','no_cad'): self.weight['carbon']=0.
        if variant in ('no_vq','no_cad'): self.weight['queue']=0.
        self.dt=config['simulation']['dt_seconds'];self.dt_h=self.dt/3600
        self.logs=[];self.decision_logs=[];self.plan_logs=[]
        self.obs_samples=[];self.reward_samples=[]
        self.pending=None
        self.all_trial_decisions=0
        self.inference_seconds=0.
        self.failed=False;self.failure_reason=''
        self.held_policy_signals=None

    def _demand(self,base,power):
        d=np.asarray(base,float).copy()
        np.add.at(d,self.case.cpn_bus,power)
        return d

    def _kwargs(self):
        return dict(commitment=self.commitment,previous_gen=self.previous_gen,
                    energy=self.energy,dt_hours=self.dt_h,
                    previous_commitment=self.previous_commitment,
                    storage_mode=self.storage_mode)

    def _sensitivity_kwargs(self):
        return {k:v for k,v in self._kwargs().items() if k!='storage_mode'}

    def _signals(self,demand,renewable):
        result=self.ed.solve(demand,renewable,**self._kwargs())
        if not result.feasible:return result,None,None
        buses=np.unique(self.case.cpn_bus)
        mci=self.ed.mci(demand,renewable,base=result,buses=buses,**self._sensitivity_kwargs())
        head=self.ed.headrooms(demand,renewable,base=result,buses=buses,**self._sensitivity_kwargs())
        if not np.isfinite(mci[buses]).all() or not np.isfinite(head[buses]).all():
            raise RuntimeError('Nodal MCI or admissible headroom is undefined; no replacement value is supplied.')
        return result,mci,head

    def _observe(self,cpn,task,base,renewable,result,mci,head,mask):
        buses=self.case.cpn_bus
        if not all(np.isfinite(np.asarray(v)[buses]).all() for v in (result.lmp,mci,head)):
            raise ValueError('LMP, MCI, and headroom features must be supplied by valid ED queries.')
        return make_observation(cpn,task,result.lmp,mci,head,self.energy,self.case,
                                base,renewable,self.budget-self.emissions,self.queue,mask)

    def _mask(self,cpn,task,head):
        n=self.case.n_cpn
        mask=np.zeros(1+3*n,dtype=bool);mask[0]=True
        reasons=['defer'];cpn_bad=np.zeros(len(mask),bool);grid_bad=cpn_bad.copy()
        for node in range(n):
            for hw in range(3):
                a=1+node*3+hw
                check=cpn.check(task.id,node,hw)
                cpnbad=not check.feasible
                margin=head[self.case.cpn_bus[node]]
                gridbad=(not np.isfinite(margin)) or check.delta_mw>margin+self.config['solver']['feasibility_tol']
                cpn_bad[a]=cpnbad;grid_bad[a]=gridbad
                mask[a]=not cpnbad and not gridbad
                reasons.append(check.reason if cpnbad else ('grid_headroom' if gridbad else 'feasible'))
        return mask,reasons,cpn_bad,grid_bad

    def _choose(self,policy,raw,explore):
        obs=self.normalization.transform(raw)
        if self.variant=='no_fs':
            # No feasibility information may leak through the dueling mean or
            # bootstrap target. The raw mask remains exclusively for rejection
            # and diagnostics in the physical simulator.
            obs=deepcopy(obs)
            obs.mask[:]=True
        begin=time.perf_counter()
        values=policy.action_values(obs,explore=explore)
        self.inference_seconds+=time.perf_counter()-begin
        preferred=int(np.argmax(values))
        mask=raw.mask if self.variant!='no_fs' else np.ones_like(raw.mask)
        chosen=int(np.argmax(np.where(mask,values,-np.inf)))
        if explore and getattr(policy,'variant',None)=='flat_dqn':
            epsilon=1-(1-policy.settings['epsilon_final'])*min(1.,policy.steps/max(1,policy.settings['epsilon_decay_steps']))
            if policy.rng.random()<epsilon:chosen=int(policy.rng.choice(np.flatnonzero(mask)))
        return chosen,preferred,obs

    def _trial(self,physical,base,renewable,initial_signals,policy,explore,end):
        cpn=physical.clone()
        initial_energy=cpn.energy_mwh.copy()
        records=[];revenue=0.;tardy=0.;events=0
        limit=self.config['simulation'].get('max_decisions_per_interval')
        if limit is not None:limit=int(limit)
        result,mci,head=initial_signals
        while cpn.now < end-1e-8:
            ready=cpn.ready_tasks()
            if ready and (limit is None or len(records)<limit):
                task=ready[0]
                demand=self._demand(base,cpn.reserved_power())
                current,cmci,chead=self._signals(demand,renewable)
                if not current.feasible:
                    # Defer the task when the grid dispatch is infeasible.
                    cpn.defer(task.id,until=end)
                    continue
                # Relaxed ED informs the first decision; later decisions receive
                # refreshed endogenous signals unless the explicit no-feedback ablation.
                if records and self.variant!='no_fb':result,mci=current,cmci
                if self.variant=='no_fb':result,mci=initial_signals[0],initial_signals[1]
                mask,reasons,cpn_bad,grid_bad=self._mask(cpn,task,chead)
                raw=self._observe(cpn,task,base,renewable,result,mci,chead,mask)
                action,preferred,obs=self._choose(policy,raw,explore)
                failure=False;accepted=False;next_result=current
                delta_cost=delta_emission=0.
                if action==0:
                    cpn.defer(task.id,until=end)
                elif not mask[action]:
                    failure=True;cpn.defer(task.id,until=end)
                else:
                    n,h=divmod(action-1,3)
                    candidate=cpn.clone();candidate.assign(task.id,n,h)
                    updated_demand=self._demand(base,candidate.reserved_power())
                    next_result=self.ed.solve(updated_demand,renewable,**self._kwargs())
                    if next_result.feasible:
                        cpn=candidate;accepted=True
                        delta_cost=(next_result.cost_rate-current.cost_rate)*self.dt_h
                        delta_emission=(next_result.emission_rate-current.emission_rate)*self.dt_h
                    else:
                        failure=True;grid_bad[action]=True
                        cpn.defer(task.id,until=end)
                reward={'cost':delta_cost,'carbon':delta_emission,'revenue':0.,'tardiness':0.,
                        'deferral':float(action==0 or failure),'failure':float(failure)}
                records.append({'observation':obs,'raw_observation':raw,'action':action,
                                'preferred':preferred,'raw_reward':reward,
                                'time_seconds':cpn.now,'task':task.id,
                                'preferred_infeasible':bool(not mask[preferred]),
                                'selected_infeasible':bool(not mask[action] or failure),
                                'cpn_infeasible':bool(cpn_bad[action]),
                                'grid_infeasible':bool(grid_bad[action]),
                                'accepted':accepted,'queue_ratio':self.queue/self.budget})
                self.all_trial_decisions+=1
                continue
            event=min(cpn.next_event_time(),end)
            if event<=cpn.now:
                raise RuntimeError('A supplied CPN event does not advance physical time.')
            change=cpn.advance_to(event)
            revenue+=change['revenue'];tardy+=change['tardiness']
            events+=1
            max_events=self.config['simulation'].get('max_events_per_interval')
            if max_events is not None and events>max_events:
                raise RuntimeError('Event guard reached; inspect job/transfer input timestamps.')
        average=(cpn.energy_mwh-initial_energy)/self.dt_h
        final=self.ed.solve(self._demand(base,average),renewable,**self._kwargs())
        if records:
            records[-1]['raw_reward']['revenue']+=revenue
            records[-1]['raw_reward']['tardiness']+=tardy
        return cpn,average,final,records,revenue,tardy

    def _plan(self,k,remaining):
        hours=min(int(self.config['planning']['horizon_hours']),int(np.ceil(remaining/12)))
        demand,res,prob,modes,expected_modes=scenario_forecasts(self.data,self.case,k,hours)
        if self.variant=='no_ssb':
            demand=np.average(demand,axis=0,weights=prob)[None]
            res=np.average(res,axis=0,weights=prob)[None];prob=np.ones(1)
            if expected_modes is None:
                raise ValueError('no_ssb requires supplied expected_value_storage_modes.')
            if expected_modes.shape!=(hours,self.case.n_storage) or not np.isin(expected_modes,[-1,0,1]).all():
                raise ValueError('Invalid expected_value_storage_modes dimensions or values.')
            modes=expected_modes[None]
        planner=BendersPlanner(self.case,self.config)
        fn=planner.solve if self.planning_mode=='benders' else planner.solve_extensive
        plan=fn(demand,res,prob,self.previous_commitment,self.previous_gen,self.energy,
                storage_modes=modes,initial_status_hours=self.status_hours)
        self.plan_logs.append({'interval':k,'objective':plan.objective,'lower_bound':plan.lower_bound,
                               'gap':plan.relative_gap,'converged':plan.converged,
                               'iterations':plan.iterations,'seconds':plan.elapsed_seconds,
                               'status':plan.status,'history':plan.history})
        if plan.commitment.size==0 or not np.isfinite(plan.objective):
            raise RuntimeError('No feasible day-ahead incumbent: '+plan.status)
        # Exact commitment charges for executed hours are computed in run().
        return plan.commitment,plan.startup_cost

    def _emit(self,records,callback):
        for record in records:
            self.obs_samples.append(record['raw_observation'])
            self.reward_samples.append(record['raw_reward'])
            if self.pending is not None and callback is not None:
                previous=self.pending
                reward=self.normalization.reward(previous['raw_reward'],self.weight,previous['queue_ratio'])
                callback(Transition(previous['observation'],previous['action'],reward,
                                    record['observation'],False))
            self.pending=record
            self.decision_logs.append({k:v for k,v in record.items()
                                       if k not in ('observation','raw_observation')})

    def run(self,policy,hours=None,explore=False,transition_callback=None,max_decisions=None):
        if self.normalization is None:
            raise ValueError('A learned policy needs training-fitted normalization.')
        requested_hours=self.config['simulation']['horizon_hours'] if hours is None else float(hours)
        if not np.isfinite(requested_hours) or requested_hours<=0:
            raise ValueError('The evaluation horizon must be finite and positive.')
        requested_intervals=int(requested_hours*12)
        if requested_intervals<1:
            raise ValueError('The evaluation horizon must include a five-minute interval.')
        total=min(len(self.data.base_load),requested_intervals)
        if total<1:raise ValueError('The dataset contains no dispatch intervals.')
        started=time.perf_counter();plan=None;plan_start=0
        total_commitment_cost=0.;iterations=[];cap_count=0
        average_previous=self.cpn.power()
        for k in range(total):
            # One subsequent decision supplies the observed bootstrap state.
            if max_decisions is not None and len(self.decision_logs)>=max_decisions+1:break
            try:
                availability=np.ones(self.case.n_gen,dtype=bool)
                demand_multiplier=1.
                if self.disturbances is not None:
                    self.case,availability,demand_multiplier=self.disturbances.at(k,self.dt)
                    self.ed.case=self.case
                if plan is None or k-plan_start>=len(plan)*12:
                    plan,_=self._plan(k,total-k);plan_start=k
                self.commitment=plan[(k-plan_start)//12].astype(int)*availability
                self.storage_mode=self.storage_modes[k].astype(int)
                start_u=self.previous_commitment.copy()
                commitment_charge=float(np.dot(np.maximum(self.commitment-start_u,0),self.case.gen_startup)
                                        +np.dot(np.maximum(start_u-self.commitment,0),self.case.gen_shutdown))
                base=self.data.base_load[k]*demand_multiplier;renewable=self.data.renewable[k]
                end=(k+1)*self.dt
                self.cpn.advance_to(k*self.dt)
                interval_start=time.perf_counter();call_start=self.ed.counters['calls']
                physical=self.cpn
                p=average_previous.copy();best=None;converged=False
                max_iter=1 if self.variant=='no_fb' else self.config['closed_loop']['max_iterations']
                residual=float('inf');used_iterations=0;pre_mci=None
                for j in range(max_iter):
                    used_iterations=j+1
                    signals=self._signals(self._demand(base,p),renewable)
                    if not signals[0].feasible:
                        signals=self._signals(self._demand(base,physical.reserved_power()),renewable)
                    if not signals[0].feasible:break
                    if pre_mci is None:pre_mci=signals[1].copy()
                    trial=self._trial(physical,base,renewable,signals,policy,explore,end)
                    if trial[2].feasible:best=trial
                    rho=self.config['closed_loop']['rho']
                    pnew=(1-rho)*p+rho*trial[1]
                    residual=float(np.max(np.abs(pnew-p)))
                    p=pnew
                    if trial[2].feasible and residual<=self.config['closed_loop']['epsilon_mw']:
                        converged=True;break
                if best is None:
                    # Safe fallback: admit no new assignments, advance only existing tasks.
                    fallback=physical.clone();oldenergy=fallback.energy_mwh.copy()
                    completed=fallback.advance_to(end)
                    avg=(fallback.energy_mwh-oldenergy)/self.dt_h
                    final=self.ed.solve(self._demand(base,avg),renewable,**self._kwargs())
                    if not final.feasible:
                        raise RuntimeError('Physical ED infeasible after deferring new tasks: '+final.status)
                    best=(fallback,avg,final,[],completed['revenue'],completed['tardiness'])
                accepted_cpn,average,final,records,revenue,tardy=best
                post_mci=self.ed.mci(self._demand(base,average),renewable,
                                     base=final,buses=np.unique(self.case.cpn_bus),**self._sensitivity_kwargs())
                if not np.isfinite(post_mci[self.case.cpn_bus]).all():
                    raise RuntimeError('Post-update nodal MCI is undefined; no replacement value is supplied.')
                # Physical integration happens exactly once, after trial acceptance.
                self.cpn=accepted_cpn
                self.energy=final.energy_next.copy()
                self.status_hours=np.where(self.commitment==start_u,self.status_hours+self.dt_h,self.dt_h)
                self.previous_gen=final.p_gen.copy();self.previous_commitment=self.commitment.copy()
                emission=final.emission_rate*self.dt_h;self.emissions+=emission
                self.queue=max(0.,self.queue+emission-self.budget/self.budget_intervals)
                total_commitment_cost+=commitment_charge
                if not records and self.pending is not None:
                    self.pending['raw_reward']['revenue']+=revenue
                    self.pending['raw_reward']['tardiness']+=tardy
                self._emit(records,transition_callback)
                average_previous=average
                iterations.append(used_iterations)
                capped=not converged and used_iterations==max_iter
                cap_count+=int(capped)
                row={'interval':k,'time_hours':k*self.dt_h,
                     'cpn_mw':float(average.sum()),'base_mw':float(base.sum()),
                     'demand_multiplier':demand_multiplier,
                     'unavailable_generators':int(np.sum(~availability)),
                     'available_res_mw':float(renewable.sum()),
                     'used_res_mw':float(final.renewable_used.sum()),
                     'thermal_mw':float(final.p_gen.sum()),
                     'bess_charge_mw':float(final.charge.sum()),
                     'bess_discharge_mw':float(final.discharge.sum()),
                     'cost_usd':final.cost_rate*self.dt_h+commitment_charge,
                     'emissions_t':emission,'carbon_queue_t':self.queue,
                     'decisions':len(records),'inner_iterations':used_iterations,
                     'converged':converged,'hit_iteration_cap':capped,
                     'residual_mw':residual,'ed_invocations':self.ed.counters['calls']-call_start,
                     'latency_seconds':time.perf_counter()-interval_start,
                     'balance_residual_mw':final.violations['balance_mw'],
                     'bess_energy_mwh':float(self.energy.sum())}
                for n,bus in enumerate(self.case.cpn_bus):
                    row[f'cpn_{n}_mw']=float(average[n])
                    row[f'mci_pre_{n}']=float(pre_mci[bus]) if pre_mci is not None else float('nan')
                    row[f'mci_post_{n}']=float(post_mci[bus])
                self.logs.append(row)
            except RuntimeError as exc:
                self.failed=True;self.failure_reason=str(exc);break
        truncated=not self.failed and len(self.logs)<requested_intervals
        if self.pending is not None and transition_callback is not None and not truncated and not self.failed:
            previous=self.pending
            terminal_obs=deepcopy(previous['observation']);terminal_obs.mask[:]=False;terminal_obs.mask[0]=True
            reward=self.normalization.reward(previous['raw_reward'],self.weight,previous['queue_ratio'])
            transition_callback(Transition(previous['observation'],previous['action'],reward,terminal_obs,True))
        duration=len(self.logs)*self.dt_h
        # Failed runs retain the requested-window denominator (unserved jobs
        # count as failures); deliberate training-step truncation uses its
        # actually observed horizon and is explicitly marked.
        qos=self.cpn.metrics(total*self.dt if self.failed else len(self.logs)*self.dt)
        available=sum(x['available_res_mw'] for x in self.logs)*self.dt_h
        used=sum(x['used_res_mw'] for x in self.logs)*self.dt_h
        count=len(self.decision_logs)
        full_budget=(not self.failed and len(self.logs)==self.budget_intervals and total==self.budget_intervals)
        metrics={**qos,'seed':self.seed,'variant':self.variant,'planning_mode':self.planning_mode,
                 'failed':self.failed,'status':'failed' if self.failed else ('truncated' if truncated else 'completed'),
                 'truncated':truncated,
                 'failure_reason':self.failure_reason,'horizon_hours':duration,
                 'requested_horizon_hours':requested_intervals*self.dt_h,'completed_intervals':len(self.logs),
                 'operational_cost_kusd':sum(x['cost_usd'] for x in self.logs)/1000,
                 'emissions_t':self.emissions,'curtailment_mwh':available-used,
                 'renewable_utilization_pct':100*used/available if available>0 else float('nan'),
                 'cpn_energy_mwh':float(self.cpn.energy_mwh.sum()),
                 'mean_cpn_mw':float(self.cpn.energy_mwh.sum()/duration) if duration else float('nan'),
                 'carbon_budget_t':self.budget,
                 'terminal_budget_violation':bool(self.emissions>self.budget+1e-7) if full_budget else None,
                 'budget_horizon_complete':full_budget,'carbon_queue_updates':len(self.logs),
                 'scheduling_decisions':count,'trial_scheduling_decisions':self.all_trial_decisions,
                 'ed_invocations':self.ed.counters['calls'],
                 'ed_solver_seconds':self.ed.counters['total_seconds'],
                 'inference_seconds':self.inference_seconds,
                 'mean_inner_iterations':float(np.mean(iterations)) if iterations else 0.,
                 'iteration_cap_intervals':cap_count,
                 'mean_interval_latency_s':float(np.mean([r['latency_seconds'] for r in self.logs])) if self.logs else 0.,
                 'preferred_action_infeasibility_pct':100*sum(r['preferred_infeasible'] for r in self.decision_logs)/count if count else 0.,
                 'selected_action_infeasibility_pct':100*sum(r['selected_infeasible'] for r in self.decision_logs)/count if count else 0.,
                 'executed_action_infeasibility_pct':0.,
                 'wall_clock_seconds':time.perf_counter()-started,
                 'benders_all_converged':all(p['converged'] for p in self.plan_logs) if self.plan_logs else None}
        metrics['disturbances']=self.disturbances.descriptions() if self.disturbances is not None else []
        return RunResult(metrics,self.logs,self.decision_logs,self.plan_logs,
                         self.obs_samples,self.reward_samples)
