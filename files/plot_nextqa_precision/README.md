# NExT-QA FSDP2 参数精度对比图

读取 **verl console 日志**（例如训练命令经过 `2>&1 | tee run.log` 保存的文件）。
无需 TensorBoard、PyTorch 或服务器 NPU 环境，只需 Python 3.10+ 和 matplotlib：

```bash
pip install matplotlib
python files/plot_nextqa_precision/plot_nextqa_precision.py \
  --bf16 /path/to/fsdp2_bf16.log \
  --fp32 /path/to/fsdp2_fp32.log \
  --output-dir /path/to/precision_figures
```

两条曲线图例分别为 `bf16` 和 `fp32`，使用蓝色和朱红色。
横轴为 `Training step`，纵轴为 `val-reward(mean@1)`；默认不显示标题、图注或网格。
该标注要求实验确实只改变原始参数精度，并保留 BF16 前后向计算。

## 日志格式与指标

兼容 verl 的标准 console 格式，允许 Ray 前缀、ANSI 颜色码、科学计数法，以及 `np.float64(0.84)`、`numpy.float32(...)` 等 NumPy 数值标量：

```text
(TaskRunner pid=123) step:10 - val-core/nextqa/reward/mean@1:0.78 - timing_s/step:320.5 - perf/max_memory_allocated_gb:48.2
```

也可读取带 `step` / `global_step` / `training/global_step` 的 Python 字典日志。
每组应对应一次实验；同一指标同一步完全重复会去重，值冲突会报错，避免默默混合续训或不同实验。
不要使用只含进度条、没有指标行的日志。终端视觉换行不影响解析；手工拆成多行的指标行不支持。

默认指标：

- reward：`val-core/nextqa/reward/mean@1`
- 单步耗时：`timing_s/step`，单位秒
- 显存：`perf/max_memory_allocated_gb`，默认按 GiB 解释（字节除以 1024³）

**某些版本不会输出显存指标**。先查看实际日志中的名字：

```bash
python files/plot_nextqa_precision/plot_nextqa_precision.py \
  --bf16 /path/to/fsdp2_bf16.log --fp32 /path/to/fsdp2_fp32.log --list-tags
```

按实际名称覆盖 `--reward-tag`、`--time-tag`、`--memory-tag`。
若显存日志单位不是 GiB，设置 `--memory-unit bytes` / `MiB` / `GB`。
缺少 reward 会报错；缺少耗时或显存会警告并显示 N/A，不会估算或填零。
若选择 `max_memory_reserved`，需在 PR 中明确它是 reserved memory，不能与 allocated memory 混比。

## 统计口径

reward 图显示原始验证点，不平滑、不插值、不补缺失步；横轴默认截止到两组最后一个 reward step 的较小值。
例如两组分别到 100 和 60 步，图和 `reward_points.csv` 都只保留到 60 步的数据。
用 `--reward-max-step 50` 可进一步缩短显示范围；大于较短运行的上限不会延长横轴。
实际截止步记录在 `plot_metadata.json` 的 `reward_end_step` 中。

耗时/显存表默认从第 5 步开始，到两组共同记录的最后一个耗时 step；
仅使用两组都存在的相同 step。耗时显示均值、中位数及样本量，显存显示窗口内日志值最大值及样本量。
可显式指定窗口并排除验证、保存或 profiling 步：

```bash
python files/plot_nextqa_precision/plot_nextqa_precision.py \
  --bf16 /path/to/fsdp2_bf16.log --fp32 /path/to/fsdp2_fp32.log \
  --stats-start 5 --stats-end 50 \
  --exclude-steps 10 20 30 40 50 \
  --reward-max-step 100 --output-dir /path/to/precision_figures
```

排除只作用于成本统计，**不会删掉 reward 验证点**。
没有共同耗时记录时，可设置 `--stats-end` 仍统计显存。
日志显存峰值可能是进程累计峰值，未必在每步重置；它也未必代表所有 rank 的最大值。
日志耗时可能包含验证/保存开销，不通过日志 wall-clock 时间差估算。
不要把带 profiler 的运行与普通运行直接作为性能对照。

输出：

- `reward.png`、`reward.pdf`：主图。
- `cost_summary.png`、`cost_summary.csv`、`cost_summary.md`：成本表。
- `reward_points.csv`：用于作图的原始数值。
- `plot_metadata.json`：输入路径、指标名、实际统计 step 列表及参数。

同一输出目录重复执行会覆盖这些文件。图和表默认不带标题或图注；
如有需要，可通过 `--note '...'` 和 `--title '...'` 为 reward 图添加图注和标题。
统计口径和样本量仍保留在 CSV、Markdown 和 JSON 中。
PR 附上硬件、依赖、模型和数据版本，确认学习率、batch、总训练步数、seed、评估采样配置一致。
输出元数据含输入文件路径，公开分享前检查路径是否含私人信息。

可选 CSV 输入：长表列为 `step,tag,value`；宽表首列为 `step`，其他列名为完整指标名，缺失值留空。
这不是必须的转换步骤，直接传 `.log` 文件即可。

## 本地回归检查

```bash
python -m unittest discover -s files/plot_nextqa_precision -p 'test_*.py'
```

测试使用合成数据，验证日志解析、重复冲突检测、统计窗口及实际图片输出；不代表真实实验结果。

## 仅复制文本的 PR 指标提取

不方便传输日志或上传文件时，在仓库根目录运行（仅需 Python 标准库）：

```bash
python files/plot_nextqa_precision/extract_pr_metrics.py /path/to/fsdp2_bf16.log --label fsdp2_bf16
python files/plot_nextqa_precision/extract_pr_metrics.py /path/to/fsdp2_fp32.log --label fsdp2_fp32
python files/plot_nextqa_precision/extract_pr_metrics.py /path/to/veomni.log --label veomni_fp32
```

复制 `BEGIN_METRICS` 到 `END_METRICS` 的终端文本即可，无需上传文件。
标签应反映实际实验配置；它不会从日志推断后端或参数精度。
输出包含逐步耗时及分项、actor allocated/reserved 显存、验证 reward、训练 reward、训推一致性及其他辅助指标。
没有验证的步数保留空值，不根据 step 是否整除 10 推断验证结果，也不插值。
显存优先选择带 `actor/` 前缀的指标，并在 sources 中注明实际来源；保持原始日志单位与统计范围。
`step_s` 与 `val_s` 分开保留，不自动相减（验证耗时不一定包含在 step 计时器内）。
使用 `--start 0 --end 50`、`--start 51 --end 100` 分段复制较长结果；每段保留头尾。
可加 `--note 'hardware=...; seed=...; batch=...; precision=...'` 说明配置或实验差异。
输出为 8 位有效数字，不输出原始日志路径或提示词。
FSDP2 精度 PR 比较 fsdp2_bf16 与 fsdp2_fp32；后端适配 PR 比较配置对齐的 fsdp2_fp32 与 veomni_fp32。

## 一次生成两份 PR 的图和摘要

```bash
pip install matplotlib
python files/plot_nextqa_precision/compare_pr_experiments.py \
  --fsdp2-bf16 /path/to/fsdp2_bf16.log \
  --fsdp2-fp32 /path/to/fsdp2_fp32.log \
  --veomni /path/to/veomni.log \
  --output-dir pr_figures
```

- `pr_figures/precision/`：FSDP2 bf16 与 fp32 的 reward 图和耗时/显存表。
- `pr_figures/backend/`：FSDP2 fp32 与 VeOmni 的 reward 图和耗时/显存表。
- 复制终端 `BEGIN_PR_SUMMARY` 到 `END_PR_SUMMARY` 的内容用于撰写 PR，无需传完整日志。

VeOmni 对照要求同样使用 FP32 原始参数、BF16 计算，并核对硬件、batch、数据、学习率、seed 和评估配置。
摘要中的 reward 数值为 `[step, value]`，起始/最终/最佳只使用两组共同的验证步。
图保留截止步以内各自的原始点；耗时窗口默认第 5 步至较短运行最后一个验证步，仅比较双方都有记录的步数。
可指定 `--stats-start`、`--stats-end`、`--exclude-steps`、`--reward-max-step`；统计上限不会超过 reward 截止步。
缺失指标为 null；不补零、不扣除验证计时、不推断未记录的配置。
`left_over_right_speedup` 为左侧平均单步时间 / 右侧平均单步时间。
显存沿用日志单位和作用域；两个来源不同的指标在据此得出结论前应核对其定义。
训练精度/后端以命令行输入所代表的实验为准，脚本不会验证实际训练环境。
