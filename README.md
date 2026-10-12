# 名创优品 IP 潮玩首单与返单智能决策算法 V3.2

本仓库已生成 ROUND 1–9 的本地实现。各阶段是独立入口；生成完整代码不等于一次执行全部算法，也不代表代码已在公司服务器、真实数据或 MySQL 上跑通。

正式依据只有《名创优品算法方案V3.2.pdf》和《数据字段说明.pdf》。V3.1 及历史示例数字不作为实现门槛。

## 架构边界

- `src/data/` 是唯一允许知道原始表名、物理字段和 Parquet 读取方式的目录。
- `restore/`、`comparable/`、`placement/`、`factors/`、`first_order/`、`reorder/`、`training/` 只接收标准化 DataFrame/domain objects。
- `config/source_tables.json` 保存交付表字段契约；`src/data/decision_adapter.py` 保存统一物理到领域映射。
- `config/algorithm_v3_2.json` 保存版本化算法参数、保护带和“待业务确认”状态。
- 国内缺少正式 MOQ、承诺产能、交期日历、计划铺货量、实际上架日和正式促销标识。代码不自动拿相似字段代替；参考值会在报告中明确标记。
- 库存中心表是对账快照，不作为出入库流水。
- 每个算法入口只执行自己的阶段，读取前一阶段标准化产物。

## 独立入口

| 阶段               | 入口                                     | 输出目录（默认）                        |
| ---------------- | -------------------------------------- | ------------------------------- |
| 环境/源表探测          | `scripts/server_probe.py`              | 用户指定                            |
| MySQL 建表         | `scripts/init_database.py`             | MySQL                           |
| 数据契约验证           | `scripts/validate_dataset.py`          | `artifacts/data_validation`     |
| 标准 IP 映射核对       | `scripts/build_standard_ip_mapping.py` | `artifacts/standard_ip_mapping` |
| ROUND 3 cohort   | `scripts/build_training_dataset.py`    | `artifacts/round3_dataset`      |
| M2 历史还原          | `scripts/run_restore.py`               | `artifacts/round4_restore`      |
| M1 可比款           | `scripts/run_comparable.py`            | `artifacts/round5_comparable`   |
| M3/M4 铺货         | `scripts/run_placement.py`             | `artifacts/round6_placement`    |
| M5 因子            | `scripts/run_factors.py`               | `artifacts/round7_factors`      |
| M6/M7 首单         | `scripts/run_first_order.py`           | `artifacts/round7_first_order`  |
| 训练集拼装            | `scripts/build_modeling_dataset.py`    | `artifacts/training`            |
| OOF + full Train | `scripts/train.py`                     | `artifacts/round8_training`     |
| 锁模 Test replay   | `scripts/backtest.py`                  | `artifacts/round8_backtest`     |
| N1–N6 返单         | `scripts/run_reorder.py`               | `artifacts/round9_reorder`      |
| 可选落库             | `scripts/persist_results.py`           | MySQL append-only 表             |

主要入口支持 `--limit-skus`、可重复的 `--sku-code` 和 `--dry-run`；决策阶段支持 `--decision-cutoff-time`。`run_restore.py` 还兼容旧参数 `--cutoff-time`、`--sku-limit`。

## 已实现公式范围

- M1：Path0–Path4 硬优先级、产品类型/市场硬过滤、内部 Top-K、相似度可用权重重归一、质量加权 softmax、每个历史 SKU 仅一个 `alpha`、外部方向系数中性降级。
- M2：有符号冲销、PIT、异常大单、缺货/有货未陈列、严格三态暴露、价格中和、有效暴露收缩、档期中和和门店层级速度。
- M3/M4：历史铺货先验、表现概率、经验贝叶斯收缩、`rho/K/Kg`、高等级优先配额、门店潜力、缺失项权重重归一和生命周期曲线。
- M5：MAD 稳健标准化、比例 winsor、计数 `log1p`、组内/组间缺失重归一、相对分位差、价格弹性、竞争规则、保护带和 `Cdemand`。
- M6/M7：覆盖期、门店需求/安全库存/陈列、`Qbase/Qnat/Qplan/Qfeas/Qopp`、策略、MOQ/模具/箱规/产能顺序、SABC、输入完整度和可靠度；没有 OOF 时区间明确为未校准。
- ROUND 8：固定 Train/Test、同首销日不拆分、B1–B5 扩展窗口、B2–B5 OOF、长尾权重、时间半衰期、非负 Elastic Net 加权模型选择、full Train 锁模、Test 单次 replay、WAPE/Bias/等级命中/混淆矩阵/OOF 区间。
- N1–N6：Actual/Plan 主偏离、`lambda/Mdev`、趋势斜率、状态、`Tecon`、有限窗口、可售库存扣减、安全库存差额、新店、质量护栏、`QRplan/QRfeas/QRopp`、动作和正负归因。

## 服务器逐步测试顺序

以下命令每次只运行一个阶段。先上传整个项目到 `/tmp/code/xujing`，不要再嵌套一层目录。

```bash
conda activate py310
cd /tmp/code/xujing
python -m pytest -q
```

1. 只读探测和数据契约：

```bash
python scripts/server_probe.py \
  --data-root /data01/yeyingshi/data \
  --output-dir artifacts/server_probe_round2

python scripts/validate_dataset.py \
  --data-root /data01/yeyingshi/data \
  --probe-dir artifacts/server_probe_round2 \
  --output-dir artifacts/data_validation \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00'
```

1. 如需重建 ROUND 3：

```bash
python scripts/build_training_dataset.py \
  --data-root /data01/yeyingshi/data \
  --probe-dir artifacts/server_probe_round2 \
  --output-dir artifacts/round3_dataset \
  --cutoff-time '2026-09-30T23:59:59+08:00'
```

1. M2 候选池 smoke。可比款需要历史候选池，不应只还原两个目标 SKU：

```bash
python scripts/run_restore.py \
  --data-root /data01/yeyingshi/data \
  --probe-dir artifacts/server_probe_round2 \
  --cohort-file artifacts/round3_dataset/cohort.csv \
  --output-dir artifacts/round4_restore_pool \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00' \
  --limit-skus 30 \
  --batch-skus 5
```

M2 会按 `--batch-skus` 串行批量读取 SKU（默认每批 5 个，不启动多个  
SKU 任务并行）。销售、库存、铺货的日期分区和文件目录元数据只在当前  
进程扫描一次。每个完成批次会保存到  
`artifacts/round4_restore_pool/_checkpoints/`；运行中断后，使用完全相同的  
命令即可自动跳过已完成批次。日志会显示批次、数据源、行数、分区数、  
耗时、模型阶段及文件写出阶段。

如果服务器内存较小，可改为 `--batch-skus 2`；批量大小只影响读取方式，  
不会改变最终所有 SKU 的联合 M2 拟合口径。若历史分区被回补或修正，需要  
强制重新读取原始数据，则增加 `--no-resume`。SKU 选择、cutoff、观察窗口  
或批量大小变化时会生成新的检查点标识，不会误用旧批次。

1. 对两个指定 SKU 单独运行 M1：

先将 `reference/sku_standard_ip_mapping.csv` 与决策时点的服务器商品分区按  
`sku_code` 精确连接并检查覆盖率。映射中的 `excel_exact` 来自业务补充表；  
`text_inferred_reference_alias` 是有表内已标注别名佐证的文字提取；  
`text_candidate_needs_review` 和 `unmapped` 不允许触发严格同 IP 的 PATH1。

```bash
python scripts/build_standard_ip_mapping.py \
  --data-root /data01/yeyingshi/data \
  --cohort-file artifacts/round3_dataset/cohort.csv \
  --mapping-file reference/sku_standard_ip_mapping.csv \
  --alias-config config/standard_ip_aliases.json \
  --output-dir artifacts/standard_ip_mapping \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00'
```

该入口只处理 Round 3 cohort 中已经满足数据集准入条件的 SKU，而不是商品表  
全量 SKU。它输出 cohort 全量映射明细（无法映射的 SKU 也保留空 IP 行）、  
逐 SKU 覆盖审计和汇总报告。名称只用于审计和有证据的补充推断；服务器连接键  
始终是 `sku_code`。

```bash
python scripts/run_comparable.py \
  --data-root /data01/yeyingshi/data \
  --cohort-file artifacts/round3_dataset/cohort.csv \
  --restore-dir artifacts/round4_restore_pool \
  --resolved-standard-ip-file artifacts/standard_ip_mapping/cohort_sku_ip_mapping.csv \
  --output-dir artifacts/round5_comparable_yoyo \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00' \
  --sku-code 2028156310109 \
  --sku-code 2029025310107
```

M1 只按 `sku_code` 查询已经生成的版本化标准 IP 快照，不会再次执行名称匹配或  
文字推断。快照的 `dataset_version`、cutoff、商品分区、cohort SKU 范围和报告计数  
必须与本次运行一致，否则脚本会报错。标准 IP 参考、别名或映射逻辑变化时，应先  
显式重跑 `build_standard_ip_mapping.py` 并生成一个新的快照版本。

1. 只运行铺货。默认即可运行：数据适配层按配置中的 28 天首发窗口读取  
   `first_store_arrival_time`，引擎按 V3.2 计算 `p_prior`、`p_perf`、收缩后的 `p*`、  
   铺货率和门店排序。需要人工确认/后续 Train 校准的参数集中放在  
   `config/algorithm_v3_2.json` 的 `placement.initial_defaults`；  
   `CategoryFit`/`Execution` 暂无完整快照时保持缺失，并按方案重归一权重：

```bash
python scripts/run_placement.py \
  --data-root /data01/yeyingshi/data \
  --comparable-dir artifacts/round5_comparable_yoyo \
  --restore-dir artifacts/round4_restore_pool \
  --output-dir artifacts/round6_placement_yoyo \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00'
```

业务参数或经过训练的证据准备好后，可分别用 `--placement-policy-file`、  
`--placement-evidence-file`、`--target-store-feature-file` 覆盖默认输入；模板位于  
`config/*_v3_2_template.csv`。每次运行都会输出 `v3_2_compliance.csv`，逐项说明  
数据计算、版本化默认值、缺失重归一和未交付硬约束，不会把默认值伪装成已校准值。

生命周期曲线按门店层级整条选择同一个来源：先尝试已选可比款，再依次尝试  
L4/L3/L2/L1/全局品类历史。每条曲线必须通过首周锚点、观测周数、样本数和倍率  
上限检查；同一条曲线不会逐周拼接不同来源。所有历史来源均不合格时，输出明确  
标记的平坦初始曲线（各周倍率为 1），等待后续 Train 校准。Path4 的历史首周锚点  
若为 0、近似 0 或样本不足，也会继续向高层级回退，最终使用正的版本化静态锚点。

1. 只运行因子：

```bash
python scripts/run_factors.py \
  --data-root /data01/yeyingshi/data \
  --comparable-dir artifacts/round5_comparable_yoyo \
  --restore-dir artifacts/round4_restore_pool \
  --output-dir artifacts/round7_factors_yoyo \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00'
```

1. 只运行首单。未传业务约束文件时可以做代码 smoke，但 MOQ/交期/产能结论不是正式业务结果：

```bash
python scripts/run_first_order.py \
  --data-root /data01/yeyingshi/data \
  --placement-dir artifacts/round6_placement_yoyo \
  --factor-dir artifacts/round7_factors_yoyo \
  --comparable-dir artifacts/round5_comparable_yoyo \
  --output-dir artifacts/round7_first_order_yoyo \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00'
```

1. ROUND 8 先拼训练表，再训练，最后单独回放 Test。真实训练必须扩大到完整 Train 样本，两个 SKU 不能训练：

```bash
python scripts/build_modeling_dataset.py \
  --cohort-file artifacts/round3_dataset/cohort.csv \
  --feature-artifact comparable=artifacts/round5_comparable_all/comparable_summary.csv \
  --feature-artifact placement=artifacts/round6_placement_all/placement_summary.csv \
  --feature-artifact factors=artifacts/round7_factors_all/demand_adjustments.csv \
  --output-dir artifacts/training \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00'

python scripts/train.py \
  --modeling-dataset artifacts/training/modeling_dataset.csv \
  --output-dir artifacts/round8_training \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00'

python scripts/backtest.py \
  --modeling-dataset artifacts/training/modeling_dataset.csv \
  --training-dir artifacts/round8_training \
  --output-dir artifacts/round8_backtest \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00'
```

1. N1–N6 历史 smoke 读取冻结的首单结果和 M2 日明细：

```bash
python scripts/run_reorder.py \
  --first-order-dir artifacts/round7_first_order_yoyo \
  --placement-dir artifacts/round6_placement_yoyo \
  --actual-file artifacts/round4_restore_yoyo/daily_exposure_audit.csv \
  --output-dir artifacts/round9_reorder_yoyo \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00'
```

正式在线返单必须改用该返单决策时点可见的实际日数据、冻结首单快照以及业务确认的返单交期/MOQ/库存字段，不能把历史 smoke 当成正式建议。

1. MySQL 可选且独立。先 `init_database.py --dry-run`，取得应用账号后正式初始化，再选择已完成阶段落库：

```bash
python scripts/init_database.py --dry-run
python scripts/init_database.py

python scripts/persist_results.py \
  --stage first_order \
  --artifact-dir artifacts/round7_first_order_yoyo \
  --decision-cutoff-time '2026-09-30T23:59:59+08:00'
```

## 当前状态

### IMPLEMENTED LOCALLY

数据适配、独立 CLI、原子化产物、日志、append-only MySQL writer、synthetic tests，  
以及 M1–M7/N1–N6 的可审计代码骨架。不能严格按 V3.2 落地的部分统一记录在  
`config/v3_2_implementation_gaps.json`；各阶段不得再用 `implemented_output` 掩盖  
缺失输入或未训练参数。

### NOT VERIFIED ON SERVER

本次新增的 M1、M3–M7、ROUND 8、N1–N6 及 MySQL writer 均未在公司服务器执行。

### REQUIRES SERVER DATA

真实候选覆盖、字段值域、门店铺货规模、因子缺失比例、首单与返单实际窗口、真实库存和日更延迟。

### REQUIRES MYSQL

九张表的真实初始化、权限、事务写入、重复运行和 superseded 链验证。

### REQUIRES REAL TRAINING

相似度权重、温度、K、因子 beta、生命周期随机效应、保护带、时间半衰期、OOF  
误差区间、返单残差模型、阈值和可靠度校准。配置中的参考值只供服务器 smoke，  
不应直接发布为正式参数。

## 本地验证

```bash
python -m pytest -q
```

合成数据只验证公式、接口、PIT、去重、缺失处理、多 IP 防重复、训练隔离和事务逻辑，不用于真实参数或业务结论。
