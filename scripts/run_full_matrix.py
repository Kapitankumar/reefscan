#!/usr/bin/env python
"""Drive the full E1–E6 experiment matrix sequentially on one machine.

Idempotent: every train.py call uses --continue, so completed runs (manifest
present) are skipped. Safe to re-run after an interruption.

Phases (each isolated — a failure logs and continues to the next phase):
  E1  coral FE screen x4 + evaluate            (NOAA coral)
  E2  coral fine-tune top-2 (+partial winner)
  E3  seg U-Net train + evaluate               (DeepFish)
  E4  fish FE screen x4                        (F4K crops)
  E5  fish fine-tune winner
  E6  fish/coral evaluate + enhancement + XAI
  FIN ledger + comparison + artifact zip

Usage:  python scripts/run_full_matrix.py [--platform gpu] [--skip-eval]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = ROOT / "logs" / "matrix.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd: list[str], phase: str, check: bool = True) -> int:
    log(f"{phase}: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    result = subprocess.run([sys.executable, *cmd], cwd=str(ROOT), check=False)
    dt = time.time() - t0
    status = "OK" if result.returncode == 0 else f"EXIT {result.returncode}"
    log(f"{phase}: done in {dt / 60:.1f} min — {status}")
    if check and result.returncode != 0:
        raise RuntimeError(f"{phase} failed with exit {result.returncode}")
    return result.returncode


def fe_rows_with_manifests(task: str) -> set[str]:
    """Backbones of `task` that already have a FE manifest (--continue handles skips;
    this set is only used to decide whether the screen phase can be skipped)."""
    runs = ROOT / "runs" / task
    if not runs.exists():
        return set()
    return {p.name.rsplit("_feature_extraction", 1)[0]
            for p in runs.glob("*_feature_extraction/manifest.json")}


def leaderboard_top(task: str, k: int) -> list[str]:
    import pandas as pd

    lb_path = ROOT / "results" / task / "model_leaderboard.csv"
    lb = pd.read_csv(lb_path)
    fe = lb[lb["strategy"] == "feature_extraction"]
    return fe.nlargest(k, "weighted_f1")["model"].tolist()


def phase_e1_coral_screen() -> None:
    manifest_count = len(fe_rows_with_manifests("coral"))
    if manifest_count >= 4:
        log("E1: all 4 coral FE runs have manifests — skipping screen")
        return
    run(["train.py", "--task", "coral", "--model", "all",
         "--strategy", "feature_extraction", "--platform", "gpu", "--continue"], "E1 coral FE x4")
    run(["evaluate.py", "--task", "coral", "--platform", "gpu"], "E1 coral evaluate")


def phase_e2_coral_finetune() -> None:
    top2 = leaderboard_top("coral", 2)
    log(f"E2: coral fine-tune top-2 = {top2}")
    for model in top2:
        run(["train.py", "--task", "coral", "--model", model,
             "--strategy", "finetune", "--platform", "gpu", "--continue"], f"E2 coral FT {model}")
    winner = top2[0]
    log("E2: optional partial_finetune on the E1 winner (non-fatal)")
    run(["train.py", "--task", "coral", "--model", winner,
         "--strategy", "partial_finetune", "--platform", "gpu", "--continue"],
        "E2 coral partial (winner)", check=False)
    run(["evaluate.py", "--task", "coral", "--platform", "gpu"], "E2 coral evaluate")


def phase_e3_seg() -> None:
    run(["train.py", "--task", "seg", "--platform", "gpu", "--continue"], "E3 seg train")
    run(["evaluate.py", "--task", "seg", "--platform", "gpu"], "E3 seg evaluate")


def phase_e4_fish_screen() -> None:
    manifest_count = len(fe_rows_with_manifests("fish"))
    if manifest_count >= 4:
        log("E4: all 4 fish FE runs have manifests — skipping screen")
        return
    run(["train.py", "--task", "fish", "--model", "all",
         "--strategy", "feature_extraction", "--platform", "gpu", "--continue"], "E4 fish FE x4")
    run(["evaluate.py", "--task", "fish", "--platform", "gpu"], "E4 fish evaluate")


def phase_e5_fish_finetune() -> None:
    winner = leaderboard_top("fish", 1)[0]
    log(f"E5: fish fine-tune winner = {winner}")
    run(["train.py", "--task", "fish", "--model", winner,
         "--strategy", "finetune", "--platform", "gpu", "--continue"], f"E5 fish FT {winner}")
    run(["evaluate.py", "--task", "fish", "--platform", "gpu"], "E5 fish evaluate")


def phase_e6_extras() -> None:
    samples = (list((ROOT / "datasets" / "f4k" / "curated" / "test").rglob("*.png"))
               + list((ROOT / "datasets" / "f4k" / "curated" / "train").rglob("*.png")))
    if samples:
        run(["evaluate.py", "--enhancement",
             "--data", str((ROOT / "datasets" / "f4k" / "curated" / "train"))],
            "E6 enhancement benchmark", check=False)
        run(["evaluate.py", "--xai", "--image", str(samples[0])],
            "E6 XAI grid", check=False)
    else:
        log("E6: no F4K sample found — skipping enhancement/XAI")


def phase_final() -> None:
    run(["scripts/log_experiment.py"], "FIN ledger", check=False)
    run(["scripts/compare_runs.py"], "FIN comparison", check=False)
    run(["scripts/export_artifacts.py"], "FIN export", check=False)


PHASES = [
    ("E1", phase_e1_coral_screen),
    ("E2", phase_e2_coral_finetune),
    ("E3", phase_e3_seg),
    ("E4", phase_e4_fish_screen),
    ("E5", phase_e5_fish_finetune),
    ("E6", phase_e6_extras),
    ("FIN", phase_final),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--only", default=None,
                        help="comma list of phases to run (e.g. E3,E4)")
    args = parser.parse_args()

    selected = PHASES
    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",")}
        selected = [(name, fn) for name, fn in PHASES if name in wanted]

    results = {}
    for name, fn in selected:
        log(f"===== PHASE {name} START =====")
        try:
            fn()
            results[name] = "OK"
        except Exception as exc:  # noqa: BLE001 — one phase must not kill the rest
            results[name] = f"FAILED: {type(exc).__name__}: {exc}"
            log(f"===== PHASE {name} FAILED: {exc} =====")

    log(f"===== MATRIX SUMMARY: {results} =====")
    return 0 if all(v == "OK" for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
