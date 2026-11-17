"""Train independent seeded checkpoints and evaluate shared held-out inputs."""
import argparse
from pathlib import Path
from copy import deepcopy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tsco.config import load_config
from tsco.training import train_agent
from tsco.cli import evaluate

p=argparse.ArgumentParser()
p.add_argument('--data',required=True)
p.add_argument('--output',required=True)
p.add_argument('--config',default='configs/nominal.json')
p.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
p.add_argument('--variant',default='tsco')
p.add_argument('--steps',type=int,default=2_000_000)
p.add_argument('--hours',type=float,default=168)
p.add_argument('--device',default='cpu')
p.add_argument('--normalization')
p.add_argument('--checkpoint-name',default='best_feasible.pt',choices=['best_feasible.pt','last.pt'])
p.add_argument('--evaluate-only',action='store_true')
a=p.parse_args();root=Path(a.output);cfg=load_config(a.config)
if not a.evaluate_only:
    for seed in a.seeds:
        c=deepcopy(cfg);c['simulation']['seed']=seed;c['drl']['seed']=seed
        train_agent(a.data,c,root/f'seed_{seed}',a.variant,a.device,a.steps,
                    a.hours,a.hours,planning_mode='benders',normalization_path=a.normalization)
args=argparse.Namespace(data=a.data,split='test',hours=a.hours,seeds=a.seeds,seed=0,
                       output=str(root/'evaluation'),checkpoint=str(root/'seed_{seed}'/a.checkpoint_name),
                       device=a.device,variant=a.variant,planning='benders',plots=False)
evaluate(args,cfg)
