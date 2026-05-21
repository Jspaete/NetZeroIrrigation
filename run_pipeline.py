"""
NetZeroIrrigation preprocessing pipeline

Processes irrigation, pump, energy carrier, and solar PV data into
Zen-Garden compatible inputs.

Pipeline order:
  1. create_system_parameters  — county nodes & neighbour edges
  2. water_demand_data         — hourly water demand per county (p75 filtered)
  3. water_pumps               — pump conversion factors & existing capacities
  4. energy_carrieres          — electricity/diesel prices & carbon intensity
  5. prepare_PV_data           — solar PV capacity factors per county

Usage:
  python run_pipeline.py               # run all steps (controlled by run_pipeline.toml [run] flags)
  python run_pipeline.py 3 4           # run steps 3 and 4 by number
  python run_pipeline.py water_pumps   # run a single step by name

Requires: conda activate s2z_js
"""

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

# All modules use '../'-relative paths that assume CWD = moduls/
MODULS_DIR = Path(__file__).parent / "moduls"
os.chdir(MODULS_DIR)
sys.path.insert(0, str(MODULS_DIR))

CONFIG_PATH = Path(__file__).parent / "run_pipeline.toml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# (display_name, module_file_name)
STEPS = [
    ("create_system_parameters", "create_system_parameters"),
    ("water_demand_data",        "water_demand_data"),
    ("water_pumps",              "water_pumps"),
    ("energy_carrieres",         "energy_carrieres"),
    ("prepare_PV_data",          "prepare_PV_data"),
]

# Steps that require a specific upstream step to have succeeded
_STEP_DEPS: dict[str, str] = {
    "water_demand_data": "create_system_parameters",
    "water_pumps":       "water_demand_data",
    "energy_carrieres":  "water_demand_data",
    "prepare_PV_data":   "water_demand_data",
}


def load_config() -> dict:
    """Load run_pipeline.toml; return empty dict if unavailable."""
    if tomllib is None:
        log.warning("tomllib not available — using module-level path defaults")
        return {}
    if not CONFIG_PATH.exists():
        log.warning(f"Config not found: {CONFIG_PATH} — using module-level path defaults")
        return {}
    with open(CONFIG_PATH, "rb") as fh:
        cfg = tomllib.load(fh)
    log.info(f"Loaded config: {CONFIG_PATH}")
    return cfg


def run_step(label: str, module_name: str, config: dict) -> bool:
    """Import and run one pipeline step. Returns True on success."""
    log.info("")
    log.info("=" * 65)
    log.info(f"STEP START: {label}")
    log.info("=" * 65)
    t0 = time.time()
    try:
        mod = __import__(module_name)
        mod.main(config)
        elapsed = time.time() - t0
        log.info("=" * 65)
        log.info(f"STEP DONE:  {label}  ({elapsed:.1f}s)")
        log.info("=" * 65)
        return True
    except Exception as e:
        elapsed = time.time() - t0
        log.error("=" * 65)
        log.error(f"STEP FAILED: {label}  ({elapsed:.1f}s)")
        log.error(f"  {type(e).__name__}: {e}")
        for line in traceback.format_exc().splitlines():
            log.error(f"  {line}")
        log.error("=" * 65)
        return False


def main(selected: list[str] | None = None) -> None:
    config = load_config()
    step_labels = [label for label, _ in STEPS]

    if selected is not None:
        # CLI selection overrides TOML [run] flags
        steps_to_run = set(selected)
    else:
        run_flags = config.get("run", {})
        if run_flags:
            steps_to_run = {lbl for lbl in step_labels if run_flags.get(lbl, True)}
        else:
            steps_to_run = set(step_labels)

    results: dict[str, str] = {}

    for label, module_name in STEPS:
        if label not in steps_to_run:
            continue

        dep = _STEP_DEPS.get(label)
        if dep and results.get(dep) in ("FAILED", "SKIPPED"):
            log.warning(f'SKIP: {label} — prerequisite "{dep}" failed or was skipped')
            results[label] = "SKIPPED"
            continue

        results[label] = "OK" if run_step(label, module_name, config) else "FAILED"

    log.info("")
    log.info("=" * 65)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 65)
    icons = {"OK": "✓", "FAILED": "✗", "SKIPPED": "–"}
    for label, _ in STEPS:
        if label not in results:
            continue
        status = results[label]
        log.info(f"  {icons[status]}  {label}: {status}")
    log.info("=" * 65)

    if any(v == "FAILED" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    step_labels = [label for label, _ in STEPS]
    steps_help = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(step_labels))

    parser = argparse.ArgumentParser(
        description="NetZeroIrrigation preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available steps:\n{steps_help}",
    )
    parser.add_argument(
        "steps",
        nargs="*",
        help="Steps to run by name or number (default: all)",
    )
    args = parser.parse_args()

    if not args.steps:
        selected = None
    else:
        selected = []
        for s in args.steps:
            if s.isdigit():
                idx = int(s) - 1
                if 0 <= idx < len(step_labels):
                    selected.append(step_labels[idx])
                else:
                    parser.error(f"Step number {s} out of range (1–{len(step_labels)})")
            elif s in step_labels:
                selected.append(s)
            else:
                parser.error(f'Unknown step "{s}". Choose from: {step_labels}')

    main(selected)
