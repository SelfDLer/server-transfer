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
        # Unwrap only numeric literals from known NumPy scalar reprs. Never eval logs.
        numpy_scalar = re.compile(
            r"(?<![\w.])(?:np|numpy)\.(?:float(?:16|32|64)|u?int(?:8|16|32|64))"
            r"\(\s*(" + number + r")\s*\)"
        )
        pair_re = re.compile(r"(?:^|\s+-\s+)([\w/@.\-]+)\s*:\s*(" + number + r")\s*$")
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, raw in enumerate(stream, 1):
                line = numpy_scalar.sub(r"\1", ansi.sub("", raw))
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
        help="Further limit the shorter run cutoff; no interpolation",
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
    if args.reward_max_step is not None and args.reward_max_step < 0:
        parser.error("Reward cutoff must be nonnegative")
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
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "text.color": "#243247",
            "axes.labelcolor": "#435268",
            "xtick.color": "#64748B",
            "ytick.color": "#64748B",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    colors = ["#C77936", "#277C8E"]
    display_labels = ["BF16 parameters", "FP32 parameters · BF16 compute"]
    # Compare only the observed reward range covered by the shorter run.
    reward_end = min(max(run[args.reward_tag]) for run in runs)
    if args.reward_max_step is not None:
        reward_end = min(reward_end, args.reward_max_step)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.20, top=0.73)
    fig.text(0.12, 0.94, args.title, fontsize=17, weight="bold", va="top")
    fig.text(
        0.12,
        0.88,
        f"Validation reward  /  Both runs shown through step {reward_end}",
        fontsize=10,
        color="#64748B",
        va="top",
    )
    reward_rows = []
    for label, display_label, color, marker, run in zip(
        labels, display_labels, colors, ["o", "s"], runs
    ):
        points = sorted(
            (s, v) for s, v in run[args.reward_tag].items() if s <= reward_end
        )
        if not points:
            raise ValueError(f"No reward points within cutoff: {label}")
        ax.plot(
            *zip(*points),
            marker=marker,
            markersize=5,
            markeredgecolor="white",
            clip_on=False,
            markeredgewidth=1,
            linewidth=2.2,
            color=color,
            label=display_label,
        )
        reward_rows.extend({"run": label, "step": s, "reward": v} for s, v in points)
    reward_start = min(row["step"] for row in reward_rows)
    # Keep a single-point run visible without identical axis limits.
    ax.set_xlim(
        reward_start if reward_start < reward_end else reward_end - 0.5,
        reward_end if reward_start < reward_end else reward_end + 0.5,
    )
    ax.set(xlabel="Training step", ylabel="Validation reward (mean@1)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ax.margins(y=0.18)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5EAF0", linewidth=0.8)
    ax.tick_params(length=0, pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.04),
        ncol=2,
        borderaxespad=0,
        frameon=False,
        fontsize=10,
        handlelength=2.4,
    )
    fig.text(0.12, 0.065, args.note, fontsize=9, color="#64748B", wrap=True)
    for extension in ("png", "pdf"):
        fig.savefig(out / f"reward.{extension}", dpi=300)
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
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.30, top=0.72)
    ax.axis("off")
    fig.text(0.05, 0.93, "NExT-QA: training cost", fontsize=17, weight="bold", va="top")
    fig.text(
        0.05,
        0.82,
        f"Steps {args.stats_start}–{end}  /  Paired measurements",
        fontsize=10,
        color="#64748B",
        va="top",
    )
    display_cells = [
        [display_labels[i], row[2], row[3], row[5]] for i, row in enumerate(cells)
    ]
    table = ax.table(
        cellText=display_cells,
        colLabels=[
            "Parameter precision",
            "Mean step (s)",
            "Median step (s)",
            "Peak logged (GiB)",
        ],
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0, 1, 1],
        colWidths=[0.40, 0.19, 0.19, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2)
        cell.PAD = 0.07
        cell.set_facecolor("#EFF3F7" if row == 0 else "#FAFBFC")
        if row == 0:
            cell.set_text_props(weight="bold", color="#435268")
        elif col == 0:
            cell.set_text_props(weight="bold", color=colors[row - 1])
    fig.text(
        0.05,
        0.19,
        f"Paired samples per run: time {len(selected)} · memory {len(memory_steps)}",
        fontsize=9,
        color="#64748B",
    )
    fig.text(
        0.05,
        0.10,
        "Memory is the maximum logged value. Measurement details: cost_summary.md.",
        fontsize=9,
        color="#64748B",
    )
    fig.savefig(out / "cost_summary.png", dpi=300)
    plt.close(fig)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(
        reward_end_step=reward_end,
        timing_steps=selected,
        memory_steps=memory_steps,
        cost_note=note,
    )
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
