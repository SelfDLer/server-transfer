"""Process-local vision instrumentation, imported by vLLM's worker extension."""
import ast
import functools
import inspect
import json
import os
from pathlib import Path
import textwrap
import time


def checkpoint(model, label, scope):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank

    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": os.getpid(), "tp_rank": get_tensor_model_parallel_rank(),
        "label": label, "target": str(model.device),
    }
    for name in ("x", "grid_thw", "grid_thw_tensor", "hidden_states"):
        value = scope.get(name)
        if isinstance(value, torch.Tensor):
            record[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                            "device": str(value.device)}
    path = Path(os.environ["VISION_REPRO_LOG_DIR"]) / f"vision-{os.getpid()}.jsonl"

    def emit(phase):
        item = {**record, "phase": phase, "monotonic": time.monotonic()}
        line = json.dumps(item, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print("[vision-probe] " + line, flush=True)

    emit("reached")
    if os.environ.get("VISION_REPRO_PROBE") == "sync":
        emit("sync_begin")
        torch.npu.current_stream(device=model.device).synchronize()
        emit("sync_done")


def instrument_forward(original):
    """Instrument the installed implementation rather than copy model code."""
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.decorator_list:
        raise RuntimeError("Unexpected vision forward structure; inspect the installed source.")
    required = ("self.patch_embed(hidden_states)", "self.rot_pos_emb(grid_thw)",
                "grid_thw.to(self.device)")
    if not all(fragment in source for fragment in required):
        raise RuntimeError("Vision forward differs from the diagnosed version; refusing to probe another path.")

    def call(label, location):
        node = ast.Expr(ast.Call(ast.Name("_vision_repro_checkpoint", ast.Load()),
                                [ast.Name("self", ast.Load()), ast.Constant(label),
                                 ast.Call(ast.Name("locals", ast.Load()), [], [])], []))
        return ast.copy_location(node, location)

    def prefix(statements):
        result = []
        for statement in statements:
            # Only instrument the initial operations through grid.to, before cu_seqlens.
            if isinstance(statement, ast.Assign):
                label = ast.unparse(statement).split(" = ", 1)[-1]
                result.extend([call("before " + label, statement), statement,
                               call("after " + label, statement)])
            elif isinstance(statement, ast.If):
                statement.body = prefix(statement.body)
                statement.orelse = prefix(statement.orelse)
                result.append(statement)
            else:
                raise RuntimeError("Unexpected statement in vision prefix; inspect source before using probe.")
        return result

    stop = next((i for i, node in enumerate(function.body) if isinstance(node, ast.Try)), None)
    if stop is None:
        raise RuntimeError("Cannot find cu_seqlens boundary in vision forward.")
    function.body = prefix(function.body[:stop]) + function.body[stop:]
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, _vision_repro_checkpoint=checkpoint)
    exec(compile(tree, "<vision-repro-forward>", "exec"), namespace)
    return functools.wraps(original)(namespace[function.name])


def install():
    import faulthandler
    import signal
    from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni_moe_thinker as module
    from vllm_omni.platforms.npu.worker.npu_ar_worker import NPUARWorker

    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    instrument_worker(NPUARWorker)
    cls = module.Qwen3Omni_VisionTransformer
    log_dir = Path(os.environ["VISION_REPRO_LOG_DIR"])
    original = cls.forward
    (log_dir / f"vision-source-{os.getpid()}.txt").write_text(
        f"file: {inspect.getsourcefile(original)}\n" + inspect.getsource(original), encoding="utf-8")
    if os.environ.get("VISION_REPRO_PROBE", "log") != "off":
        cls.forward = instrument_forward(original)
    print(f"[vision-probe] installed pid={os.getpid()} mode={os.environ.get('VISION_REPRO_PROBE')} "
          f"source={module.__file__}", flush=True)


def worker_state(worker, phase, error=None):
    config = worker.vllm_config
    compilation = config.compilation_config
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(),
        "rank": getattr(worker, "rank", None), "phase": phase,
        "enforce_eager": config.model_config.enforce_eager,
        "async_scheduling": config.scheduler_config.async_scheduling,
        "compilation_mode": getattr(compilation.mode, "name", str(compilation.mode)),
        "cudagraph_mode": getattr(compilation.cudagraph_mode, "name", str(compilation.cudagraph_mode)),
        "backend": compilation.backend,
        "capture_sizes": compilation.cudagraph_capture_sizes,
        "compilation_config": repr(compilation), "error": error,
    }
    path = Path(os.environ["VISION_REPRO_LOG_DIR"]) / f"worker-state-{os.getpid()}.jsonl"
    line = json.dumps(record, default=str)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("[worker-state] " + line, flush=True)


def instrument_worker(cls):
    # Observe resolved config before/after loading and warmup, including when the
    # vision probe is off. Do not synchronize devices or change returned values.
    def wrap(original, name):
        @functools.wraps(original)
        def observed(self, *args, **kwargs):
            first_execute = name == "execute_model"
            if first_execute and getattr(self, "_vision_repro_seen_execute", False):
                return original(self, *args, **kwargs)
            if first_execute:
                self._vision_repro_seen_execute = True
            worker_state(self, name + ":begin")
            try:
                result = original(self, *args, **kwargs)
            except Exception as exc:
                worker_state(self, name + ":error", repr(exc))
                raise
            worker_state(self, name + ":returned")
            return result
        return observed

    for name in ("load_model", "compile_or_warm_up_model", "execute_model"):
        original = getattr(cls, name)
        setattr(cls, name, wrap(original, name))


class VisionProbeWorkerExtension:
    """Importing this extension installs the probe before worker model loading."""


# vLLM resolves worker_extension_cls in each worker before load_model().
if os.environ.get("VISION_REPRO_LOG_DIR"):
    install()
