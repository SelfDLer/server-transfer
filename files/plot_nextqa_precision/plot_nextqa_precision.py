#!/usr/bin/env python3
"""Plot FSDP2 precision experiments from verl console logs or metric CSV files."""

import argparse
import ast
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path


def read_metrics(path):
    """Return tag -> step -> value; reject ambiguous duplicate measurements."""
    data = {}

    def add(tag, step, value):
        step_float, value = float(step), float(value)
        if not step_float.is_integer() or step_float < 0 or not math.isfinite(value):
            raise ValueError(f"Invalid measurement: {tag}, step={step}, value={value}")
        step = int(step_float)
        points = data.setdefault(tag, {})
        if step in points and points[step] != value:
            raise ValueError(
                f"Conflicting values for {tag} at step {step}; select one run/segment."
            )
        points[step] = value

    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if "step" not in fields:
                raise ValueError("CSV requires a 'step' column (see README).")
            long_format = "tag" in fields and "value" in fields
            for row in reader:
                if long_format:
                    add(row["tag"], row["step"], row["value"])
                else:
                    for tag in fields:
                        if tag != "step" and row[tag].strip():
                            add(tag, row["step"], row[tag])
    else:
        ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
        step_re = re.compile(
            r"(?:^|[\s,{])['\"]?(?:step|global_step|training/global_step)['\"]?\s*[:=]\s*(\d+)\b"
        )
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        pair_re = re.compile(r"(?:^|\s+-\s+)([\w/@.\-]+)\s*:\s*(" + number + r")\s*$")
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, raw in enumerate(stream, 1):
                line = ansi.sub("", raw)
                match = step_re.search(line)
                if not match:
                    continue
                step = int(match.group(1))
                # Standard LocalLogger format: step:N - key:value - key:value.
                payload = {}
                for item in line[match.start() :].strip().split(" - "):
                    pair = pair_re.fullmatch(item.strip())
                    if pair:
                        payload[pair.group(1)] = float(pair.group(2))
                # Some logging wrappers print a Python dict or JSON instead.
                if "{" in line and "}" in line:
                    try:
                        candidate = ast.literal_eval(
                            line[line.index("{") : line.rindex("}") + 1]
                        )
                    except (ValueError, SyntaxError):
                        candidate = None
                    if isinstance(candidate, dict):
                        payload.update(
                            {
                                k: v
                                for k, v in candidate.items()
                                if isinstance(v, (int, float))
                                and not isinstance(v, bool)
                            }
                        )
                for tag, value in payload.items():
                    if tag not in ("step", "global_step", "training/global_step"):
                        try:
                            add(tag, step, value)
                        except ValueError as error:
                            raise ValueError(
                                f"{path}:{line_number}: {error}"
                            ) from error
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bf16",
        type=Path,
        required=True,
        help="BF16-parameter run: verl console log or CSV",
    )
    parser.add_argument(
        "--fp32",
        type=Path,
        required=True,
        help="FP32-parameter run: verl console log or CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nextqa_precision_figures")
    )
    parser.add_argument("--reward-tag", default="val-core/nextqa/reward/mean@1")
    parser.add_argument(
        "--time-tag", default="timing_s/step", help="Logged step duration in seconds"
    )
    parser.add_argument("--memory-tag", default="perf/max_memory_allocated_gb")
    parser.add_argument(
        "--memory-unit", choices=["GiB", "GB", "MiB", "bytes"], default="GiB"
    )
    parser.add_argument(
        "--stats-start",
        type=int,
        default=5,
        help="First step for cost statistics, inclusive",
    )
    parser.add_argument(
        "--stats-end",
        type=int,
        help="Last step for cost statistics; default common final timing step",
    )
    parser.add_argument(
        "--exclude-steps",
        type=int,
        nargs="*",
        default=[],
        help="Exclude validation/checkpoint/profile steps from cost stats",
    )
    parser.add_argument(
        "--reward-max-step",
        type=int,
        help="Optional reward plot cutoff; no interpolation",
    )
    parser.add_argument(
        "--list-tags", action="store_true", help="Print metric tags and exit"
    )
    parser.add_argument("--title", default="NExT-QA: FSDP2 parameter precision")
    parser.add_argument(
        "--note", default="Single run per setting; raw validation points, no smoothing."
    )
    args = parser.parse_args()
    if args.stats_start < 0 or (
        args.stats_end is not None and args.stats_end < args.stats_start
    ):
        parser.error("Invalid statistics step range")
    runs = [read_metrics(args.bf16), read_metrics(args.fp32)]
    labels = ["FSDP2: BF16 parameters", "FSDP2: FP32 parameters + BF16 compute"]
    if args.list_tags:
        for label, run in zip(labels, runs):
            print(label + "\n" + "\n".join(sorted(run)))
        return
    for label, run in zip(labels, runs):
        if not run.get(args.reward_tag):
            raise ValueError(
                f"{label}: missing reward tag {args.reward_tag!r}; use --list-tags"
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    colors = ["#D97706", "#2563EB"]
    fig, ax = plt.subplots(figsize=(8.5, 5.1))
    reward_rows = []
    for label, color, run in zip(labels, colors, runs):
        points = sorted(
            (s, v)
            for s, v in run[args.reward_tag].items()
            if args.reward_max_step is None or s <= args.reward_max_step
        )
        if not points:
            raise ValueError(f"No reward points within cutoff: {label}")
        ax.plot(
            *zip(*points),
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=label,
        )
        reward_rows.extend({"run": label, "step": s, "reward": v} for s, v in points)
    ax.set(title=args.title, xlabel="Global step", ylabel="Validation reward (mean@1)")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.text(0.5, 0.02, args.note, ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"reward.{extension}", dpi=220)
    plt.close(fig)

    # Pair timing samples by step to avoid silently comparing different windows.
    timing_steps = set(runs[0].get(args.time_tag, {})) & set(
        runs[1].get(args.time_tag, {})
    )
    end = (
        args.stats_end
        if args.stats_end is not None
        else max(timing_steps, default=args.stats_start)
    )
    selected = sorted(
        s
        for s in timing_steps
        if args.stats_start <= s <= end and s not in args.exclude_steps
    )
    memory_sets = [set(run.get(args.memory_tag, {})) for run in runs]
    memory_steps = sorted(
        s
        for s in memory_sets[0] & memory_sets[1]
        if args.stats_start <= s <= end and s not in args.exclude_steps
    )
    if not selected:
        print(
            f"WARNING: no paired timing samples for {args.time_tag}; timing is N/A.",
            file=sys.stderr,
        )
    if not memory_steps:
        print(
            f"WARNING: no paired memory samples for {args.memory_tag}; memory is N/A.",
            file=sys.stderr,
        )
    divisor = {"GiB": 1, "GB": 2**30 / 1e9, "MiB": 1024, "bytes": 2**30}[
        args.memory_unit
    ]
    rows = []
    for label, run in zip(labels, runs):
        times = [run[args.time_tag][s] for s in selected]
        memory = [run[args.memory_tag][s] / divisor for s in memory_steps]
        if any(v < 0 for v in times + memory):
            raise ValueError("Negative time/memory value; check selected tags")
        rows.append(
            {
                "run": label,
                "timing_samples": len(times),
                "mean_step_s": statistics.mean(times) if times else None,
                "median_step_s": statistics.median(times) if times else None,
                "memory_samples": len(memory),
                "peak_logged_memory_GiB": max(memory) if memory else None,
            }
        )
    for filename, records in (
        ("reward_points.csv", reward_rows),
        ("cost_summary.csv", rows),
    ):
        with (out / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    columns = [
        "Run",
        "Time N",
        "Mean step (s)",
        "Median step (s)",
        "Memory N",
        "Peak logged (GiB)",
    ]
    cells = [
        [
            str(v) if isinstance(v, (str, int)) else "N/A" if v is None else f"{v:.2f}"
            for v in row.values()
        ]
        for row in rows
    ]
    note = (
        f"Cost window: steps {args.stats_start}–{end}; common logged steps only. "
        "Memory is the maximum logged value, not necessarily a per-step or all-rank peak. "
        "Logged step duration may include validation/checkpoint overhead; exclude those steps explicitly."
    )
    markdown = (
        "| "
        + " | ".join(columns)
        + " |\n| "
        + " | ".join(["---"] * len(columns))
        + " |\n"
    )
    markdown += "".join("| " + " | ".join(row) + " |\n" for row in cells)
    (out / "cost_summary.md").write_text(
        markdown + "\n" + note + "\n", encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(13, 2.7))
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.37, 0.08, 0.14, 0.15, 0.08, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor("#E2E8F0")
    ax.set_title("NExT-QA: training cost", pad=10)
    fig.text(
        0.5,
        0.04,
        f"Steps {args.stats_start}–{end}; paired samples. Memory: maximum logged value; see cost_summary.md.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "cost_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(timing_steps=selected, memory_steps=memory_steps, cost_note=note)
    (out / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(markdown + "\n" + note)
    print(f"Saved figures and tables to {out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(str(error))
