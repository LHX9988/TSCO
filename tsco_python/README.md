# TSCO Python 实现与使用说明

代码包含日前场景优化、Benders 分解、五分钟实时 ED、节点 LMP/MCI、DAG 任务调度、可行性屏蔽、CS-GAQR-DQN 训练与评价。

名义配置位于 `configs/nominal.json`，输入模板位于 `inputs/`。运行前需补齐标记为 `null` 的必要参数及 CSV 数据行。

## 1. 环境

论文环境为 Python 3.11.9、PyPSA 0.30.3、PyTorch 2.4.1（CUDA 12.1）、Gurobi 11.0.3。

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell 使用 .venv\Scripts\Activate.ps1
python -m pip install -r requirements-paper.txt
python -m pip install -e .
```

默认求解器为 Gurobi，需要可用许可证。使用 GPU 时，安装与 CUDA 12.1 对应的 PyTorch 构建，并在运行命令中指定 `--device cuda`。

## 2. 参数与输入

| 文件 | 填写内容 |
|---|---|
| `configs/nominal.json` | 硬件功率与算力、通信参数、机组运行约束、其余训练参数 |
| `inputs/case.json` | 母线、线路、机组、BESS、RES 与 CPN 映射及设备参数 |
| `inputs/signals.csv` | UTC 时间戳、五分钟节点非 CPN 负荷、RES 可用功率、BESS 模式 |
| `inputs/jobs.json` | 到达时刻、截止时间、DAG、工作量、硬件需求、依赖边数据量和收益 |
| `inputs/job_record_template.json` | 单个作业的字段格式 |
| `inputs/planning_scenarios.json` | 各日前窗口的负荷/RES 场景、概率及 BESS 模式 |
| `inputs/initial_states.json` | 各仿真窗口的初始机组状态、出力、状态持续时间及储能电量 |
| `inputs/normalization.json` | 训练分区的特征与奖励归一化统计量 |
| `inputs/calibration_samples.json` | 计算归一化所需的训练样本 |
| `inputs/manifest.json` | 输入时间范围及数据处理信息 |

字段、单位和数组维度见 `docs/DATA.md`。

已填写的名义设置包括：30 母线、6 台火电机组、4 个 BESS、5 个 CPN 站点；每站 4 MW、128/64/32 个 CPU/GPU/TPU 归一化资源单位；RES 总容量 650 MW，其中光伏 330 MW、风电 320 MW；碳价 50 美元/tCO₂，168 小时碳预算 1450 tCO₂。

日前场景数为 100、规划窗口为 24 小时，实时步长为 300 秒。闭环参数为 `rho=0.5`、`epsilon_mw=0.01`、`max_iterations=10`；Benders 相对 gap 为 0.002；MCI 扰动为 0.1 MW；机组成本采用 20 段线性化。

数据按时间先划分为 70%/15%/15% 的训练、验证和测试分区。归一化统计量只来自训练分区。各分区需要覆盖实际选取的完整实验窗口；作业时刻保留秒级精度。训练输入中的经验预测残差和 [0.8,1.2] 负荷强度扰动应在数据准备阶段处理，并提供对应窗口与场景。

检查填写情况：

```bash
python -m tsco check-inputs --data inputs --config configs/nominal.json
```

未填齐时，该命令列出字段并返回状态码 2；不会开始训练。填写完成后还会检查数组维度、时间间隔、容量和数据分区。

`planning.terminal_storage` 填 `true` 时约束期末储能电量等于期初，填 `false` 时不添加期末等式。`solver.startup_ramp_mw` 与 `shutdown_ramp_mw` 为机组启动、停机过渡的出力限额。可选计算上限保留 `null` 表示不额外设限；物理参数保留 `null` 则不能执行。

## 3. 归一化、训练和测试

特征偏移和尺度需填写训练分区统计量。奖励尺度可直接填写到 `normalization.json`，也可提供训练奖励样本后计算：

```bash
python -m tsco calibrate --data inputs --config configs/nominal.json \
  --samples inputs/calibration_samples.json --output inputs
```

奖励的收益、成本、排放和延迟分量使用训练样本绝对值的 95% 分位数，分母下限为 `1e-8`，并裁剪至 `[-5,5]`。特征归一化尺度由输入文件提供。

填齐配置和输入后运行：

```bash
python -m tsco train --data inputs --config configs/nominal.json \
  --normalization inputs/normalization.json --output models/seed_0 \
  --seed 0 --variant tsco --steps 2000000 \
  --episode-hours 168 --validation-hours 168 --device cuda

python -m tsco evaluate --data inputs --config configs/nominal.json \
  --checkpoint models/seed_0/best_feasible.pt --output results/nominal \
  --seed 0 --variant tsco --hours 168 --plots --device cuda
```

训练最多执行 200 万个环境转移，每 25,000 步验证一次。`best_feasible.pt` 仅在完整预算周期满足预算和 QoS 条件时保存。训练日志分别记录环境转移、ED 调用和墙钟时间。

十个种子分别训练并使用相同测试输入：

```bash
python examples/run_seeds.py --data inputs --config configs/nominal.json \
  --normalization inputs/normalization.json --output models/paired \
  --seeds 0 1 2 3 4 5 6 7 8 9 --device cuda
```

## 4. 消融与敏感性

`--variant` 可选择：

| 参数 | 对应设置 |
|---|---|
| `tsco` | 完整模型 |
| `no_nm` | 线性 CPN 功率模型 |
| `no_cc` | 去除通信约束 |
| `no_cad` | 移除碳奖励和虚拟队列奖励项 |
| `no_fb` | 去除间隔内调度–ED 反馈 |
| `no_ssb` | 使用期望值日前子问题 |
| `no_cr` / `no_vq` | 分别移除碳奖励、虚拟队列奖励项 |
| `no_fs` | 移除动作选择前的可行性屏蔽 |
| `flat_dqn` | 展平特征及标量 DQN |
| `no_quantile` | 图编码保留，分位数头改为标量 double-dueling 头 |

各变体分别训练；评价时配置必须与 checkpoint 一致。`no_fs` 中无效动作仍由模拟器拒绝，分别记录屏蔽前偏好、选择和实际执行动作。

RES、碳预算及奖励权重检查通过 `examples/run_studies.py` 执行，具体命令见 `docs/STUDIES.md`。机组退出、线路检修和需求变化配置位于 `examples/`；跨拓扑测试需要对应系统填写完成的输入目录。

## 5. 代码结构

| 模块 | 内容 |
|---|---|
| `config.py` / `data.py` / `schema.py` | 配置、输入检查和数据结构 |
| `cpn.py` | DAG、资源、通信时延和 CPN 功率 |
| `optimization.py` / `grid.py` | LP/MILP、DC ED、LMP、MCI 和可用功率余量 |
| `planning.py` | 多场景 Benders 与同一 SUC 模型的完整 MILP |
| `features.py` / `learning.py` | 归一化、GATv2、分位数网络、PER、NoisyNet |
| `simulation.py` / `training.py` | 两阶段控制、训练、验证和 checkpoint |
| `statistics.py` | 十次运行的 95% CI、Welch 检验和 Holm 校正 |
| `studies.py` / `disturbances.py` | 敏感性检查和冻结策略扰动 |
| `artifacts.py` / `cli.py` | 日志、PDF 图和命令入口 |

## 6. 运行输出

每次运行保存 `metrics.json`、`intervals.csv`、`decisions.csv`、`plans.json`，以及使用的配置和环境信息。`--plots` 直接从运行日志生成 PDF。多种子评价另存 `summary.json`。

成本率和排放率按物理间隔积分一次；内层试探不重复累计。作业成功率以窗口内到期作业为分母，平均延迟统计已完成作业并包含零延迟。置信区间按运行级指标计算，`ci95` 表示均值 95% 置信区间的半宽。

```bash
python -m unittest discover -s tests -v
python -m tsco environment --output environment.json
```
