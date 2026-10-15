# Qwen3-Omni / Ascend 视觉编码器等待复现

使用 `../install_nextqa_a3_npu_manual.sh` 已创建的 Conda 环境，不安装或升级任何包。
针对其中的 torch 2.10、vLLM 0.30 和 vLLM-Omni
`12e9280010afc41ce5c6a1f46900fc329cf08377` 接口编写。
使用原始完整模型目录，运行一个 Thinker TP=4 副本；不启动训练、Ray、Talker 或权重同步。
默认 eager、保留异步调度，每次一个视频，连续两次请求，各生成至多 16 token。
模型权重规模没有缩小，仍需四张有足够可用内存的 NPU；`--memory` 是每张卡的内存预算比例。

## 运行

在 server-transfer 根目录，先激活原环境。Conda 中保存的 Ascend 和动态库变量继续生效。
指定当前空闲的四张卡；不要与原训练任务占用同一组卡。

```bash
git pull --ff-only
conda activate /path/to/env-root/conda
MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct
VIDEO=/path/to/problem-video.mp4

# 只核对包元信息、路径并生成配置；不加载模型或初始化 NPU。
bash files/repro_vision_wait/run.sh --model "$MODEL" --devices 0,1,2,3 --preflight-only

# 第一轮：实际视频开头抽取 4 帧、缩放到 224x224，只记录日志，不添加同步。
bash files/repro_vision_wait/run.sh --model "$MODEL" --video "$VIDEO" \
  --devices 0,1,2,3 --probe log

# 第二轮：同一输入，在视觉编码器前几步加入 current-stream 同步。
bash files/repro_vision_wait/run.sh --model "$MODEL" --video "$VIDEO" \
  --devices 0,1,2,3 --probe sync

# 第三轮（独立对照）：强制同步下发，但不增加显式分段同步。
bash files/repro_vision_wait/run.sh --model "$MODEL" --video "$VIDEO" \
  --devices 0,1,2,3 --probe log --launch-blocking 1
```

各轮单独运行，上一轮退出、释放 NPU 后再启动下一轮。卡号是 `ASCEND_RT_VISIBLE_DEVICES`
的物理 ID；脚本生成的 stage 配置使用对应的逻辑 ID。`--tp` 与卡号数量必须相等。
先不提供 `--video` 可以使用固定随机种子的合成帧，排除视频读取问题；它不能代替原输入复现。
`--frames 16 --size 448 --frame-stride 4` 可以增加视觉负载；`--max-model-len` 也可能需要随之增加。
实际视频只抽取开头的帧，不包含音频，也不复刻训练数据集的采样逻辑。

若仅原任务卡住，按次序一次改变一个条件：使用原视频/接近原帧数与分辨率、增加
`--requests`，然后尝试 `--no-eager`（PIECEWISE，capture size=1）。
`--no-async-scheduling` 可作为异步调度对照，`--probe off` 可关闭视觉打点。
显式同步会改变时序；某个诊断模式能跑通并不代表已修复。
单请求测试也无法排除原任务的并发、权重更新、sleep/wake 或流间依赖问题。

## 日志

默认输出到当前目录的 `vision-repro-logs/`：

- `console-*.log`：完整控制台、错误栈和 worker PID。
- 每次运行的子目录：`environment.json`、`deploy.json`、每个 worker 的
  `vision-<pid>.jsonl` 和原始 `vision-source-<pid>.txt`。
- 每条记录包含时间、PID、TP rank、操作名称、shape/dtype/device，不打印张量内容。
- `log` 模式的 `after ...` 只说明 Python 调用返回，不说明设备完成。
- `sync` 模式只有 `sync_begin` 而没有 `sync_done` 的位置，是该次采样找到的等待区间。
  如果最前面的 `before x.to(...)` 就无法同步，应检查进入视觉编码器之前已提交的任务。

若卡住，保留进程并另开终端：

```bash
py-spy dump --native --pid <日志里的worker_PID>
# 若需 sudo 且 py-spy 安装在当前环境：
sudo "$(command -v py-spy)" dump --native --pid <日志里的worker_PID>
```

也可以向日志中明确显示 `[vision-probe] installed` 的 worker 发送
`kill -USR1 <worker_PID>`，Python 全线程栈会写入控制台日志。
不要向未注册该信号处理器的其他进程发送这个信号。
脚本不会自动超时杀进程，以便抓取现场；退出时先 Ctrl+C，并确认本次 worker 已退出后再重跑。

诊断通过 vLLM 的 `worker_extension_cls` 在每个 worker 中临时改写 `forward`，
只插入打印/可选同步，保留原计算步骤；不修改环境中的源码文件。若源码结构不匹配则报错，
不悄悄退化成未打点运行。退出进程即撤销插桩。

本地验证覆盖语法、配置生成和插桩计算顺序；完整 NPU 模型推理需要在服务器验证。
