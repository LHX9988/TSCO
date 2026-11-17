"""Write inspectable run results without embedding manuscript outcome values."""
import json
from pathlib import Path
import math
import platform
from importlib.metadata import version, PackageNotFoundError
import numpy as np
import pandas as pd


def json_safe(value):
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(x) for x in value]
    return value


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(json_safe(data),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def environment_report():
    packages={}
    for name in ['numpy','scipy','pandas','matplotlib','torch','pypsa','gurobipy']:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':packages}


def save_run(result,directory,config,dataset_manifest):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'metrics.json',result.metrics)
    write_json(directory/'config.json',config)
    write_json(directory/'environment.json',environment_report())
    write_json(directory/'input_manifest.json',dataset_manifest)
    write_json(directory/'plans.json',result.plans)
    pd.DataFrame(result.intervals).to_csv(directory/'intervals.csv',index=False)
    pd.DataFrame(result.decisions).to_csv(directory/'decisions.csv',index=False)


def plot_run(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory)
    data=pd.read_csv(directory/'intervals.csv')
    if data.empty:raise ValueError('No completed physical interval to plot.')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'axes.labelsize':11,'axes.titlesize':12,
                         'axes.titleweight':'bold','pdf.fonttype':42})
    fig,ax=plt.subplots(2,2,figsize=(9,5.6),constrained_layout=True)
    x=data.time_hours
    ax[0,0].plot(x,data.cpn_mw);ax[0,0].set(ylabel='CPN demand (MW)',title='CPN demand')
    dt=(x.iloc[1]-x.iloc[0]) if len(x)>1 else 1/12
    ax[0,1].plot(np.r_[x.iloc[0],x.to_numpy()+dt],np.r_[0.,data.emissions_t.cumsum()])
    ax[0,1].set(ylabel='Cumulative CO2 (t)',title='Realized emissions')
    ax[1,0].plot(x,data.available_res_mw-data.used_res_mw);ax[1,0].set(ylabel='Curtailment (MW)',title='Renewable curtailment')
    ax[1,1].plot(x,data.latency_seconds);ax[1,1].set(ylabel='Latency (s/interval)',title='Measured control latency')
    for a in ax.flat:a.set_xlabel('Time (h)');a.grid(alpha=.2)
    fig.savefig(directory/'run_profiles.pdf');plt.close(fig)
    cols=[c for c in data if c.startswith('mci_pre_')]
    fig,ax=plt.subplots(figsize=(7,3),constrained_layout=True)
    for i,c in enumerate(cols):
        color=f'C{i%10}'
        ax.step(x,data[c],where='post',ls='--',c=color,alpha=.65)
        ax.step(x,data[c.replace('pre','post')],where='post',c=color,label=f'CPN {i+1}')
    ax.set(xlabel='Time (h)',ylabel='MCI (tCO2/MWh)',title='Dashed: pre-update; solid: post-update')
    ax.legend(ncol=min(5,len(cols)),fontsize=8);ax.grid(alpha=.2)
    fig.savefig(directory/'mci_profiles.pdf');plt.close(fig)
