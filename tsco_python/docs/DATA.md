# 输入数据

运行前需补齐 `inputs/` 下的必要输入。`null` 和空 CSV 数据区表示尚未提供数据。

## 论文中已明确的设置

| 项目 | 数值 | 对应文件 |
|---|---:|---|
| 电网、常规机组、储能 | 30 个母线、6 台机组、4 台 BESS | `case.json` |
| CPN | 5 站，每站 4 MW，总计 20 MW | `case.json`、配置文件 |
| 每站资源 | CPU/GPU/TPU：128/64/32 个分别归一化的单位 | 配置文件 |
| 新能源 | 5 站，3 光伏、2 风电；650 MW=330+320 MW | `case.json` |
| 名义峰值终端负荷 | 300 MW | `case.json` |
| 时间步长 | 日前 1 h，实时 300 s | 场景文件、配置文件 |
| 日前情景数 | 100 | 配置文件 |
| 数据划分 | 按时间先后 70%/15%/15% | 配置文件 |

未给出的设备参数及时间序列保留为空。文中汇总量不用于自动生成逐时 CPN 轨迹。

## `case.json`

`case` 对象与 `tsco.schema.GridCase` 字段对应。母线下标从 **0** 开始；设备数组均按固定设备顺序填写。`renewable_types` 填写每站对应的 `solar` 或 `wind`，顺序与新能源数组一致。

- `base_mva`：DC 模型的基准容量；`base_load[B]`：各母线非 CPN 名义负荷，MW。
- `branch_from/to[L]`：支路两端；`branch_x[L]`：标幺电抗；`branch_tap[L]`：变比；`branch_limit[L]`：有功限额，MW。
- `gen_bus[G]`、`gen_pmin/pmax[G]`：位置和 MW 限额；`gen_ramp[G]`：MW/h；`gen_a/b/c[G]`：成本函数 `a P²+b P+c` 的系数，函数值为 $/h；`gen_emission[G]`：tCO2/MWh；`gen_startup/shutdown[G]`：每次启停费用，$。
- `storage_bus[K]`、`storage_power[K]`：位置和 MW 限额；`storage_capacity/min/max[K]`：容量和运行能量上下界，MWh；`storage_eta_c/d[K]`：分别提供充、放电效率，不能直接将往返效率填入两者。
- `renewable_bus/capacity[R]`：位置和 MW 容量；`cpn_bus[N]`：CPN 站与母线的映射。

`known_paper_settings` 中保留总体设置，读取时核对设备数量、650 MW 总容量及光伏/风电分项容量。不同拓扑应提供对应的完整 case 文件，不由 30-bus 数据扩展生成。若实际没有储能，相关数组应明确为 `[]`。

## `signals.csv`

每一行表示一个五分钟区间的平均功率。`timestamp` 为区间起点，必须包含 UTC 时区（例如 `2026-01-01T00:00:00Z`），相邻行相差 300 秒。

| 列 | 内容 |
|---|---|
| `base_bus_0_mw` … `base_bus_{B-1}_mw` | 各母线非 CPN 负荷，MW |
| `res_0_mw` … `res_{R-1}_mw` | 各新能源站可用功率，MW |
| `storage_0_mode` … `storage_{K-1}_mode` | 各储能实际区间模式：-1 放电、0 空闲、+1 充电 |

不通过总负荷自动分配母线负荷。分辨率转换应在准备实际数据时完成，并保持原区间能量；加载器不插值或补齐缺失行。源数据处理方法、日期及划分说明可填入 `manifest.json`。

## `jobs.json`

将 `jobs` 填写为作业列表，字段形式见 `job_record_template.json`。全局时间以 `signals.csv` 首个时间戳为零，单位秒。

每个作业提供唯一 `id`、`arrival_seconds`、`deadline_seconds` 和完整 `tasks`。每个子任务提供全局唯一 `id`、`workload`（FLOPs）、`demands`（CPU/GPU/TPU 三项资源需求）、`compatible`（硬件索引 0/1/2）、`predecessor_ids`、`data_gb` 和 `revenue`。有前驱时，`edge_data_gb` 必须逐边给出 GB 数据量，键为全部前驱 ID，不自动平均分配。根子任务可填写空前驱列表和空边映射。

读取器检查引用和 DAG 无环性。全局 `jobs: null` 或空列表表示尚未提供作业，不解释为零负载。划分后只保留到达时间与截止时间均位于该区间的完整作业，跨界作业不截断拼接。

## `planning_scenarios.json`

`planning_scenarios` 是按起点排列的记录列表。每条记录：

| 字段 | 形状及含义 |
|---|---|
| `start_interval` | 在完整 `signals.csv` 中的区间下标，从 0 开始 |
| `hour_dt` | 1.0 |
| `demand_mw` | `[S,H,B]`：场景总需求，包含预测 CPN 需求 |
| `renewable_mw` | `[S,H,R]`：场景新能源可用功率 |
| `probabilities` | `[S]`：非负、总和为 1 |
| `storage_modes` | `[S,H,K]`：-1/0/+1，对应日前场景中的储能模式 |
| `expected_value_storage_modes` | `[H,K]`：仅 `w/o SSB` 所需，须单独提供 |
| `cpn_demand_mw` | `[S,H,B]`：总需求中的 CPN 分量，仅 CPN 电功率缩放实验必需 |

日前场景须由实际数据处理流程提供。执行器按每条记录的 `start_interval` 启用该记录；各规划日都应提供覆盖相应时段的记录。实际区间储能模式与日前场景模式分别输入，不能相互推断。

## `initial_states.json`

`initial_states` 为记录列表。每个可用起点提供 `start_interval`、`commitment[G]`（0/1）、`generation_mw[G]`、`initial_status_hours[G]`（每台机组当前开/停状态已持续的小时数）、`storage_energy_mwh[K]` 和 `cpn_empty`。

当前运行器要求 `cpn_empty: true`，表示该起点确实没有在执行或排队的历史作业。若存在历史 CPN 状态，应先扩展状态恢复接口，不能将其当作空状态运行。

## 划分、窗口与归一化

先按原始行数确定 70/15/15 分界，再选择完整作业和输入场景。训练、验证、测试起点以及每个训练 episode 起点都必须有对应初始状态及日前场景记录。加载后记录的 `start_interval` 转为相对当前分区的下标；`Dataset.window()` 进一步转为窗口内下标。运行器不替缺失起点制造初始状态。

`normalization.json` 填写训练分区计算得到的特征尺度、偏移和奖励尺度，具体格式见该文件。验证、测试及敏感性实验复用训练得到的值。同一次对比的所有随机种子复用同一份处理后输入。

```python
from tsco.config import load_config
from tsco.data import load_dataset

config = load_config("configs/nominal.json")
training = load_dataset("inputs", "train", config)
test = load_dataset("inputs", "test", config)
```
