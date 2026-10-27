#!/usr/bin/env python3
"""Print compact PR experiment metrics as copyable CSV text (standard library only)."""

import argparse
import csv
import sys
from pathlib import Path

from plot_nextqa_precision import read_metrics

# Keep the original metric names recoverable while shortening the pasted rows.
METRICS = {
    "val": ("val-core/nextqa/reward/mean@1",),
    "step_s": ("timing_s/step", "perf/time_per_step"),
    "gen_s": ("timing_s/gen",),
    "logprob_s": ("timing_s/old_log_prob",),
    "actor_s": ("timing_s/update_actor",),
    "weights_s": ("timing_s/update_weights",),
    "val_s": ("timing_s/testing",),
    "alloc_gb": ("actor/perf/max_memory_allocated_gb", "perf/max_memory_allocated_gb"),
    "reserved_gb": ("actor/perf/max_memory_reserved_gb", "perf/max_memory_reserved_gb"),
    "diff_valid": ("training/rollout_probs_diff_valid",),
    "diff_mean": ("training/rollout_probs_diff_mean",),
    "diff_max": ("training/rollout_probs_diff_max",),
    "pearson": ("training/rollout_actor_probs_pearson_corr",),
    "kl": ("rollout_corr/kl",),
    "grad_norm": ("actor/grad_norm",),
    "lr": ("actor/lr",),
    "train_reward": ("critic/rewards/mean",),
    "response_len": ("response_length/mean",),
    "tokens": ("perf/total_num_tokens",),
}


def extract(data):
    selected = {}
    sources = {}
    for name, candidates in METRICS.items():
        # Choose one source for the whole run; do not mix memory scopes or timers.
        tag = next((tag for tag in candidates if data.get(tag)), None)
        selected[name] = data.get(tag, {})
        sources[name] = tag
    return selected, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--label", required=True, help="e.g. fsdp2_bf16, fsdp2_fp32, veomni_fp32"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Inclusive first step to print"
    )
    parser.add_argument("--end", type=int, help="Inclusive last step to print")
    parser.add_argument(
        "--note",
        default="",
        help="Optional hardware/config differences; printed verbatim",
    )
    args = parser.parse_args()
    if args.start < 0 or (args.end is not None and args.end < args.start):
        parser.error("Invalid step range")
    selected, sources = extract(read_metrics(args.log))
    steps = sorted(
        {
            s
            for values in selected.values()
            for s in values
            if s >= args.start and (args.end is None or s <= args.end)
        }
    )
    if not steps:
        parser.error("No matching metrics in the requested range")
    print(f"BEGIN_METRICS {args.label}")
    if args.note:
        print(f"note: {args.note}")
    print(
        "sources: "
        + "; ".join(f"{name}={tag or 'MISSING'}" for name, tag in sources.items())
    )
    print("# Empty cells are missing measurements. Values use 8 significant digits.")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["step", *METRICS])
    for step in steps:
        writer.writerow(
            [
                step,
                *(
                    format(selected[name][step], ".8g")
                    if step in selected[name]
                    else ""
                    for name in METRICS
                ),
            ]
        )
    print(f"END_METRICS {args.label}")


if __name__ == "__main__":
    main()
