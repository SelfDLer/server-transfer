"""Synthetic regression checks; no real experiment results included."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from plot_nextqa_precision import read_metrics


class PlotTests(unittest.TestCase):
    def test_console_formats_and_conflicting_restarts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.log"
            path.write_text(
                "noise\n\x1b[36m(TaskRunner pid=42)\x1b[0m step:0 - val-core/nextqa/reward/mean@1:7.59e-1\n"
                "step:0 - val-core/nextqa/reward/mean@1:0.759\n"
                "prefix {'training/global_step': 1, 'timing_s/step': 320.0}\n",
                encoding="utf-8",
            )
            data = read_metrics(path)
            self.assertEqual(data["val-core/nextqa/reward/mean@1"], {0: 0.759})
            self.assertEqual(data["timing_s/step"], {1: 320.0})
            with path.open("a") as stream:
                stream.write("step:0 - val-core/nextqa/reward/mean@1:0.8\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                read_metrics(path)

    def test_numpy_scalar_console_and_dictionary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "numpy.log"
            path.write_text(
                "(TaskRunner pid=42) step:10 - val-core/nextqa/reward/mean@1:np.float64(0.84)"
                " - timing_s/step:numpy.float32(2.2e2) - perf/max_memory_allocated_gb:np.int64(48)\n"
                "prefix {'training/global_step': np.int64(20), "
                "'val-core/nextqa/reward/mean@1': numpy.float64(8.5e-1)}\n"
                "step:30 - actor/pg_loss:np.float32(-1.2e-3)\n",
                encoding="utf-8",
            )
            data = read_metrics(path)
            self.assertEqual(
                data["val-core/nextqa/reward/mean@1"], {10: 0.84, 20: 0.85}
            )
            self.assertEqual(data["timing_s/step"], {10: 220.0})
            self.assertEqual(data["perf/max_memory_allocated_gb"], {10: 48.0})
            self.assertEqual(data["actor/pg_loss"], {30: -0.0012})

    def test_end_to_end_paired_cost_and_raw_reward(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, duration, reward in (("bf16", 320, 0.79), ("fp32", 330, 0.84)):
                lines = ["step:0 - val-core/nextqa/reward/mean@1:0.76"]
                for step in range(1, 11):
                    lines.append(
                        f"step:{step} - timing_s/step:{duration + step} - perf/max_memory_allocated_gb:{40 + step}"
                    )
                lines.append(
                    f"step:10 - val-core/nextqa/reward/mean@1:np.float64({reward})"
                )
                (root / f"{name}.log").write_text("\n".join(lines), encoding="utf-8")
            out = root / "out"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("plot_nextqa_precision.py")),
                    "--bf16",
                    str(root / "bf16.log"),
                    "--fp32",
                    str(root / "fp32.log"),
                    "--stats-start",
                    "5",
                    "--exclude-steps",
                    "10",
                    "--output-dir",
                    str(out),
                ],
                check=True,
                capture_output=True,
            )
            with (out / "cost_summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]["mean_step_s"]), 327)
            self.assertEqual(float(rows[1]["mean_step_s"]), 337)
            self.assertEqual(float(rows[0]["peak_logged_memory_GiB"]), 49)
            self.assertEqual(
                json.loads((out / "plot_metadata.json").read_text(encoding="utf-8"))[
                    "timing_steps"
                ],
                [5, 6, 7, 8, 9],
            )
            with (out / "reward_points.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            for name in ("reward.png", "reward.pdf", "cost_summary.png"):
                self.assertGreater((out / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
