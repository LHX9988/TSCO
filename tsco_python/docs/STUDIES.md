# 敏感性实验

先填写 `inputs/`、配置文件及归一化统计量，并完成策略训练。`StudyRunner` 使用相同的处理后输入和配对随机种子，输出实际运行的指标、逐区间日志及均值/95%置信区间。

## 新能源容量与碳预算

```bash
python examples/run_studies.py \
  --data inputs --config configs/nominal.json \
  --checkpoint outputs/train/best_feasible.pt \
  --grid res_capacity --values .6 .8 1 1.2 1.4 \
  --seeds 0 1 2 3 4 5 6 7 8 9 --hours 168 \
  --output outputs/res_sensitivity --plots
```

新能源容量实验同步缩放设备容量、实际可用功率和提供的日前场景可用功率，保持归一化曲线、工作量、物理储能模式及策略不变。碳预算实验改用 `--grid carbon_budget --values 1300 1375 1450 1525 1600`。冻结策略不进行训练更新；每次评估使用检查点中保存的训练归一化参数。仅完成整个预算周期后报告终端预算是否违反。

## CPN 电功率缩放

`--grid cpn_power_scale --values .5 1 1.5` 同步缩放提供的 CPU/GPU/TPU 空闲功率、峰值功率、站点电功率限额，以及日前场景中的 CPN 需求分量。使用前须填写每条场景记录的 `cpn_demand_mw`，该分量从总需求中分离后单独缩放；缺失时停止，不估计 CPN 预测值。

此实验保留站点数量、计算资源、处理速度、作业和通信参数。若实验同时改变资源数量或到达强度，应另外提供对应配置与完整输入。

## 奖励权重

支持三个组：`reward_cost`（cost）、`reward_carbon`（carbon、queue）和 `reward_qos`（tardiness、deferral、failure）。各组乘数为 0.5、0.75、1、1.25、1.5，其余权重不变。每个配置须独立训练，并使用同一训练归一化文件。

将各配置检查点路径填入 JSON 映射，例如：

```json
{
  "0.5": "reward_cost_05/seed_{seed}/best_feasible.pt",
  "0.75": "reward_cost_075/seed_{seed}/best_feasible.pt",
  "1.0": "reward_cost_10/seed_{seed}/best_feasible.pt",
  "1.25": "reward_cost_125/seed_{seed}/best_feasible.pt",
  "1.5": "reward_cost_15/seed_{seed}/best_feasible.pt"
}
```

路径相对于映射文件；`{seed}` 对应随机种子。命令中用 `--checkpoint-map` 代替 `--checkpoint`，并选择相应的 `--grid`。运行器检查训练权重与当前配置相符，各乘数使用不同的检查点，且同一配对种子的归一化参数一致。

## Python 调用

```python
from tsco.config import load_config
from tsco.data import load_dataset
from tsco.studies import StudyRunner

config = load_config("configs/nominal.json")
training = load_dataset("inputs", "train", config)
test = load_dataset("inputs", "test", config)
runner = StudyRunner(config, training, test, "outputs/train/best_feasible.pt")
runs = runner.run_grid("res_capacity", [.6, .8, 1., 1.2, 1.4],
                       range(10), "outputs/res_sensitivity", hours=168.)
```

输出包括逐设置/种子的日志、配置、`metrics.csv`、`runs.json` 和 `summary.json`。失败运行单独计数，不以部分时段成本替代完整周期结果。`plot_study()` 在成功完成的对比结果上生成 PDF。
