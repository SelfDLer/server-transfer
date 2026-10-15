#!/usr/bin/env python3
"""One TP replica, Thinker only, one video per request, no training or Ray."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path, help="Existing full Qwen3-Omni checkpoint directory")
    p.add_argument("--video", type=Path, help="Existing video; omit for deterministic synthetic frames")
    p.add_argument("--devices", default="0,1,2,3", help="Physical NPU IDs dedicated to this run")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=1, help="Sample every Nth decoded frame from the start")
    p.add_argument("--size", type=int, default=224, help="Resize frames to this square size")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--requests", type=int, default=2, help="Sequential single-video requests")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--memory", type=float, default=0.6)
    p.add_argument("--probe", choices=("off", "log", "sync"), default="log")
    p.add_argument("--launch-blocking", choices=("0", "1"), default="0")
    execution = p.add_mutually_exclusive_group()
    execution.add_argument("--eager", action=argparse.BooleanOptionalAction, default=True)
    execution.add_argument("--execution", choices=("eager", "compile", "piecewise", "graph"),
                           help="Explicit eager / compile without graphs / PIECEWISE / FULL_AND_PIECEWISE")
    p.add_argument("--input-kind", choices=("video", "text"), default="video")
    p.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--init-timeout", type=int, default=600)
    p.add_argument("--output", type=Path, default=Path("vision-repro-logs"))
    p.add_argument("--preflight-only", action="store_true", help="Check packages/config without loading weights or NPU")
    args = p.parse_args()
    if args.execution is not None:
        args.eager = args.execution == "eager"
    for key in ("tp", "frames", "frame_stride", "size", "max_tokens", "requests", "max_model_len", "init_timeout"):
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if not 0 < args.memory < 1:
        p.error("--memory must be between 0 and 1")
    devices = args.devices.split(",")
    if len(devices) != args.tp or len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
        p.error("--devices must contain exactly --tp distinct numeric NPU IDs")
    args.model = args.model.expanduser().resolve()
    if not (args.model / "config.json").is_file():
        p.error("--model must contain config.json")
    if args.video:
        args.video = args.video.expanduser().resolve()
        if not args.video.is_file():
            p.error("--video does not exist")
    return args


def make_deploy(args):
    extras = {
        "model_arch": "Qwen3OmniMoeThinkerForConditionalGeneration",
        "worker_extension_cls": "vision_probe.VisionProbeWorkerExtension",
        "hf_overrides": {"enable_audio_output": False},
        "additional_config": {"weight_nz_mode": 0},
        "dtype": "bfloat16", "distributed_executor_backend": "mp",
        "limit_mm_per_prompt": {"video": 1, "image": 0, "audio": 0},
        "mm_processor_cache_gb": 0,
        "mm_processor_kwargs": {"use_audio_in_video": False},
    }
    stage = {
        "stage_id": 0, "devices": ",".join(str(i) for i in range(args.tp)),
        "tensor_parallel_size": args.tp, "max_num_seqs": 1,
        "max_model_len": args.max_model_len, "max_num_batched_tokens": args.max_model_len,
        "gpu_memory_utilization": args.memory, "enforce_eager": args.eager,
        "async_scheduling": args.async_scheduling, "enable_prefix_caching": False,
        "engine_extras": extras,
    }
    if not args.eager:
        stage["compilation_config"] = {"cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1]}
    if args.execution is not None:
        mode = {"eager": "NONE", "compile": "NONE", "piecewise": "PIECEWISE",
                "graph": "FULL_AND_PIECEWISE"}[args.execution]
        stage["compilation_config"] = {"mode": 0 if args.eager else 3, "cudagraph_mode": mode}
        if mode != "NONE":
            stage["compilation_config"]["cudagraph_capture_sizes"] = [1]
    return {"pipeline": "qwen3_omni_moe_thinker_only", "async_chunk": False, "stages": [stage]}


def load_frames(args):
    import numpy as np
    if not args.video:
        return np.random.default_rng(42).integers(0, 256, (args.frames, args.size, args.size, 3), dtype=np.uint8)
    import av
    frames = []
    with av.open(str(args.video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i % args.frame_stride == 0:
                frames.append(frame.reformat(width=args.size, height=args.size, format="rgb24").to_ndarray())
            if len(frames) == args.frames:
                break
    if not frames:
        raise RuntimeError("Video contains no decodable frames")
    if len(frames) < args.frames:
        print(f"Only {len(frames)} frames available; repeating the last frame", flush=True)
        frames.extend([frames[-1]] * (args.frames - len(frames)))
    return np.stack(frames)


def main():
    args = arguments()
    run = args.output.resolve() / (time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    run.mkdir(parents=True, exist_ok=False)
    # Set before importing torch/vLLM; also inherited by the engine and TP workers.
    os.environ.update({
        "ASCEND_RT_VISIBLE_DEVICES": args.devices,
        "ASCEND_LAUNCH_BLOCKING": args.launch_blocking,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_ASCEND_ENABLE_NZ": "0",
        "VISION_REPRO_LOG_DIR": str(run), "VISION_REPRO_PROBE": args.probe,
        "TOKENIZERS_PARALLELISM": "false",
    })
    module_dir = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [module_dir, os.environ.get("PYTHONPATH", "")]))
    packages = ("torch", "torch-npu", "vllm", "vllm-ascend", "vllm-omni", "verl", "verl-omni", "av")
    versions = {name: metadata.version(name) for name in packages}
    print("Run directory:", run, "\nPython:", sys.executable, "\nPackages:", versions, flush=True)
    report = {"args": vars(args), "python": sys.executable, "packages": versions,
              "environment": {k: os.environ.get(k) for k in (
                  "CONDA_PREFIX", "LD_LIBRARY_PATH", "LD_PRELOAD", "ASCEND_HOME_PATH",
                  "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_LAUNCH_BLOCKING", "SRC_DIR", "VERL_OMNI_DIR")}}
    source_root = os.environ.get("SRC_DIR")
    report["source_shas"] = {}
    if source_root:
        for name in ("vllm", "vllm-ascend", "vllm-omni", "verl"):
            result = subprocess.run(["git", "-C", str(Path(source_root) / name), "rev-parse", "HEAD"],
                                    capture_output=True, text=True)
            report["source_shas"][name] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    (run / "environment.json").write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")
    deploy = run / "deploy.json"
    deploy.write_text(json.dumps(make_deploy(args), indent=2), encoding="utf-8")
    if args.preflight_only:
        print("Preflight OK (package metadata/config only; no model, NPU or imports validated).", flush=True)
        return
    from vllm import SamplingParams
    from vllm_omni.config.pipeline_registry import register_pipeline
    from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_THINKER_ONLY_PIPELINE
    from vllm_omni.entrypoints.omni import Omni
    register_pipeline(QWEN3_OMNI_THINKER_ONLY_PIPELINE)
    frames = load_frames(args) if args.input_kind == "video" else None
    def phase(name, **details):
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": name, **details}
        with (run / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print("[repro]", json.dumps(record), flush=True)

    phase("engine_init_begin", input_kind=args.input_kind,
          video_shape=list(frames.shape) if frames is not None else None)
    engine = Omni(model=str(args.model), deploy_config=str(deploy),
                  hf_overrides={"enable_audio_output": False},
                  init_timeout=args.init_timeout, stage_init_timeout=args.init_timeout)
    phase("engine_init_done")
    try:
        if engine.num_stages != 1:
            raise RuntimeError(f"Expected Thinker only; got {engine.num_stages} stages")
        for index in range(args.requests):
            video_marker = "<|vision_start|><|video_pad|><|vision_end|>" if frames is not None else ""
            question = "Describe this video briefly." if frames is not None else "Count from one to five."
            prompt = {
                "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                          f"<|im_start|>user\n{video_marker}"
                          f"{question} Request {index + 1}.<|im_end|>\n"
                          "<|im_start|>assistant\n",
            }
            if frames is not None:
                prompt.update(multi_modal_data={"video": frames},
                              mm_processor_kwargs={"use_audio_in_video": False})
            phase("request_begin", request=index + 1)
            print(f"REQUEST {index + 1} BEGIN", flush=True)
            outputs = engine.generate([prompt], [SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=42)])
            print(f"REQUEST {index + 1} DONE: {outputs}", flush=True)
            phase("request_done", request=index + 1)
        print("REPRO COMPLETED: this reduced case did not hang.", flush=True)
    finally:
        phase("engine_close_begin")
        engine.close()
        phase("engine_close_done")


if __name__ == "__main__":
    main()
