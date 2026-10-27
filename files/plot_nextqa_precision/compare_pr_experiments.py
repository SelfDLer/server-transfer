#!/usr/bin/env python3
"""Generate both PR comparisons and print compact evidence from three console logs."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from extract_pr_metrics import extract
from plot_nextqa_precision import read_metrics


def describe(values):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsdp2-bf16", type=Path, required=True)
    parser.add_argument("--fsdp2-fp32", type=Path, required=True)
    parser.add_argument(
        "--veomni",
        type=Path,
        required=True,
        help="Run with FP32 stored parameters and matching BF16 compute policy",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("pr_figures"))
    parser.add_argument("--stats-start", type=int, default=5)
    parser.add_argument("--stats-end", type=int)
    parser.add_argument("--exclude-steps", nargs="*", type=int, default=[])
    parser.add_argument("--reward-max-step", type=int)
    parser.add_argument(
        "--note",
        default="",
        help="Hardware and configuration facts for the summary, not the figure",
    )
    args = parser.parse_args()
    if args.stats_start < 0 or (
        args.stats_end is not None and args.stats_end < args.stats_start
    ):
        parser.error("Invalid statistics range")
    if args.reward_max_step is not None and args.reward_max_step < 0:
        parser.error("Invalid reward cutoff")
    paths = {
        "fsdp2_bf16": args.fsdp2_bf16,
        "fsdp2_fp32": args.fsdp2_fp32,
        "veomni": args.veomni,
    }
    runs = {name: extract(read_metrics(path)) for name, path in paths.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "note": args.note,
        "comparisons": {},
        "limitations": [
            "Single-run descriptive statistics; no significance claim.",
            "Memory retains logged units/scope; it is not necessarily an all-rank or per-step peak.",
            "Step and validation timers are separate; validation time is not subtracted.",
            "Verify matching hardware, batch, data, seed, evaluation, and compute precision externally.",
        ],
    }
    for name, pair, labels in [
        ("precision", ("fsdp2_bf16", "fsdp2_fp32"), ("bf16", "fp32")),
        ("backend", ("fsdp2_fp32", "veomni"), ("fsdp2", "veomni")),
    ]:
        data = [runs[key][0] for key in pair]
        if any(not run["val"] for run in data):
            parser.error(f"{name}: missing val-core/nextqa/reward/mean@1")
        end = min(max(run["val"]) for run in data)
        if args.reward_max_step is not None:
            end = min(end, args.reward_max_step)
        common_val = sorted(set(data[0]["val"]) & set(data[1]["val"]))
        common_val = [s for s in common_val if s <= end]
        if not common_val:
            parser.error(f"{name}: no common validation steps within cutoff")
        # Use the same observed training horizon as the reward comparison by default.
        stats_end = min(end, args.stats_end) if args.stats_end is not None else end
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        # Normalize aliases once, keeping source tags in the text report.
        normalized = []
        import csv

        for key, run in zip(pair, data):
            path = out / f"{key}_metrics.csv"
            fields = {
                "val": "val-core/nextqa/reward/mean@1",
                "step_s": "timing_s/step",
                "alloc_gb": "perf/max_memory_allocated_gb",
            }
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["step", "tag", "value"])
                for short, tag in fields.items():
                    for step, value in sorted(run[short].items()):
                        writer.writerow([step, tag, value])
            normalized.append(path)
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("plot_nextqa_precision.py")),
            "--bf16",
            str(normalized[0]),
            "--fp32",
            str(normalized[1]),
            "--labels",
            *labels,
            "--output-dir",
            str(out),
            "--reward-max-step",
            str(end),
            "--stats-start",
            str(args.stats_start),
            "--stats-end",
            str(stats_end),
            "--exclude-steps",
            *map(str, args.exclude_steps),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        result = {
            "reward_cutoff": end,
            "paired_validation_steps": common_val,
            "cost_window": [args.stats_start, stats_end],
            "excluded_steps": args.exclude_steps,
            "runs": {},
        }
        for key, run in zip(pair, data):
            vals = [(s, run["val"][s]) for s in common_val]
            best = max(vals, key=lambda item: item[1])
            item = {
                "reward_initial": vals[0],
                "reward_final": vals[-1],
                "reward_best": best,
                "reward_gain": vals[-1][1] - vals[0][1],
                "sources": runs[key][1],
            }
            for metric in (
                "step_s",
                "gen_s",
                "actor_s",
                "logprob_s",
                "weights_s",
                "alloc_gb",
                "reserved_gb",
                "diff_mean",
                "pearson",
            ):
                steps = sorted(set(data[0][metric]) & set(data[1][metric]))
                steps = [
                    s
                    for s in steps
                    if args.stats_start <= s <= stats_end
                    and s not in args.exclude_steps
                ]
                if metric in ("diff_mean", "pearson"):
                    steps = [
                        s
                        for s in steps
                        if all(d["diff_valid"].get(s, 1) == 1 for d in data)
                    ]
                item[metric] = describe([run[metric][s] for s in steps])
            result["runs"][key] = item
        left, right = [result["runs"][key]["step_s"] for key in pair]
        if left and right and left["mean"] > 0 and right["mean"] > 0:
            result["left_over_right_speedup"] = left["mean"] / right["mean"]
            result["right_time_reduction_percent"] = 100 * (
                1 - right["mean"] / left["mean"]
            )
        report["comparisons"][name] = result

    # Compact numeric precision for copying; source paths and prompts are omitted.
    def rounded(value):
        if isinstance(value, float):
            return float(format(value, ".8g"))
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [rounded(v) for v in value]
        return value

    text = json.dumps(rounded(report), ensure_ascii=False, indent=2)
    (args.output_dir / "pr_summary.json").write_text(text + "\n", encoding="utf-8")
    print("BEGIN_PR_SUMMARY\n" + text + "\nEND_PR_SUMMARY")


if __name__ == "__main__":
    main()
