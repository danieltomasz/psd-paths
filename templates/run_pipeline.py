"""
Pipeline runner for the PATHS EEG analysis.

Usage examples:
    # Run all subjects, all steps
    python templates/run_pipeline.py

    # Or import and call programmatically:
    from templates.run_pipeline import run_pipeline, show_status
    run_pipeline(subjects_to_run=['101', '114'], steps_to_run=[1, 2, 3])
    show_status()
"""
import json
import papermill as pm
from pathlib import Path
from joblib import Parallel, delayed
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Load configuration from settings.toml
# ---------------------------------------------------------------------------
# We use find_project_root so the script works regardless of CWD
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "spectral"))
from spectral.utils import load_config, find_project_root

_PROJECT_ROOT = find_project_root(Path(__file__).parent)
_cfg = load_config(_PROJECT_ROOT / "settings.toml")
_pipe = _cfg.get("pipeline", {})

BIDS_ROOT    = _PROJECT_ROOT / _cfg["paths"].get("bids_root", "data/bids")
OUTPUT_ROOT  = _PROJECT_ROOT / _pipe.get("output_root",  "outputs/pipeline")
OUTPUT_LOG   = _PROJECT_ROOT / _pipe.get("output_log",   "outputs/log")
STATUS_FILE  = _PROJECT_ROOT / _pipe.get("status_file",  "outputs/pipeline_status.json")
N_JOBS       = int(_pipe.get("n_jobs", 4))
KERNEL_NAME  = _pipe.get("kernel_name", "psd-paths-3.13")

_TEMPLATES   = _PROJECT_ROOT / "templates"
STEPS_CONFIG = {
    1: {"template": str(_TEMPLATES / "01_Preprocessing.ipynb"), "suffix": "01_Preprocessing"},
    2: {"template": str(_TEMPLATES / "02_Epochs.ipynb"),        "suffix": "02_Epochs"},
    3: {"template": str(_TEMPLATES / "03_ICA.ipynb"),           "suffix": "03_ICA"},
    4: {"template": str(_TEMPLATES / "04_Specparam.ipynb"),     "suffix": "04_Specparam"},
}

STEP_NAMES = {1: "Preprocessing", 2: "Epochs", 3: "ICA", 4: "Specparam"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubjectResult:
    """Container for a single subject's pipeline results."""
    subject_id: str
    success: bool
    completed_steps: int
    total_steps: int
    duration_seconds: float
    logs: list = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Status file helpers
# ---------------------------------------------------------------------------

def _load_status() -> dict:
    """Load the pipeline_status.json file, returning {} if it doesn't exist."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {}


def _save_status(status: dict):
    """Write the pipeline_status.json file atomically."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    tmp.replace(STATUS_FILE)


def _mark_step(subject_id: str, step: int, result: str, duration_s: float = 0.0, error: str = ""):
    """Update a single subject/step entry in the status file (thread-safe per subject)."""
    # NOTE: parallel workers update their own in-memory copy; the final merge
    # happens in run_pipeline() after all workers finish to avoid race conditions.
    pass  # See run_pipeline() for the actual write logic.


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class LogCollector:
    """Collects log messages in memory for later writing."""

    def __init__(self, subject_id: str):
        self.subject_id = subject_id
        self.entries = []

    def _log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} | {level:7} | {message}"
        self.entries.append(line)
        print(line)

    def info(self, message: str):    self._log("INFO",    message)
    def warning(self, message: str): self._log("WARNING", message)
    def error(self, message: str):   self._log("ERROR",   message)


def setup_logging() -> Path:
    OUTPUT_LOG.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_LOG / f"pipeline_{timestamp}.log"


def write_logs_to_file(log_file: Path, header_logs: list, results: list, footer_logs: list):
    with open(log_file, "w") as f:
        for line in header_logs:
            f.write(line + "\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("SUBJECT LOGS\n")
        f.write("=" * 70 + "\n\n")
        for result in sorted(results, key=lambda r: r.subject_id):
            status = "SUCCESS" if result.success else "FAILED"
            f.write(f"--- Sub-{result.subject_id} | {status} | {result.completed_steps}/{result.total_steps} steps | {result.duration_seconds:.1f}s ---\n")
            for log_line in result.logs:
                f.write(f"    {log_line}\n")
            f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n")
        for line in footer_logs:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------

def process_subject_pipeline(
    subject_id: str,
    steps_to_run: list,
    skip_completed: bool = True,
    existing_status: dict = None,
    log_output: bool = False,
) -> tuple[SubjectResult, dict]:
    """
    Run selected steps for a single subject.

    Returns (SubjectResult, step_statuses_dict) so the caller can merge
    step statuses into the master status file without race conditions.
    """
    log = LogCollector(subject_id)
    start_time = datetime.now()

    subject_output_dir = OUTPUT_ROOT / f"sub-{subject_id}"
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    subject_prior = (existing_status or {}).get(subject_id, {})
    step_statuses = {}   # steps we ran/skipped in this call

    success_count = 0
    total_requested = len(steps_to_run)
    error_msg = None

    for step_num in steps_to_run:
        if step_num not in STEPS_CONFIG:
            log.warning(f"Sub-{subject_id} | Step {step_num} not in STEPS_CONFIG, skipping.")
            total_requested -= 1
            continue

        step_key = str(step_num)

        # --- skip if already successfully completed ---
        if skip_completed and subject_prior.get(step_key, {}).get("status") == "success":
            log.info(f"Sub-{subject_id} | Step {step_num} ({STEP_NAMES[step_num]}) | Already done, skipping.")
            step_statuses[step_key] = subject_prior[step_key]  # preserve existing record
            success_count += 1
            continue

        cfg = STEPS_CONFIG[step_num]
        output_path = subject_output_dir / f"sub-{subject_id}_{cfg['suffix']}.ipynb"
        step_start = datetime.now()

        try:
            log.info(f"Sub-{subject_id} | Step {step_num} ({STEP_NAMES[step_num]}) | Starting...")
            pm.execute_notebook(
                input_path=cfg["template"],
                output_path=str(output_path),
                parameters={"subject_id": subject_id},
                kernel_name=KERNEL_NAME,
                progress_bar=False,
                log_output=log_output,
            )
            duration = (datetime.now() - step_start).total_seconds()
            log.info(f"Sub-{subject_id} | Step {step_num} ({STEP_NAMES[step_num]}) | Completed in {duration:.0f}s")
            step_statuses[step_key] = {
                "status": "success",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(duration, 1),
            }
            success_count += 1

        except Exception as e:
            duration = (datetime.now() - step_start).total_seconds()
            error_msg = str(e)[:500]
            log.error(f"Sub-{subject_id} | Step {step_num} ({STEP_NAMES[step_num]}) | FAILED: {error_msg}")
            step_statuses[step_key] = {
                "status": "failed",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(duration, 1),
                "error": error_msg,
            }
            break  # stop processing further steps for this subject

    total_duration = (datetime.now() - start_time).total_seconds()

    result = SubjectResult(
        subject_id=subject_id,
        success=(success_count == total_requested),
        completed_steps=success_count,
        total_steps=total_requested,
        duration_seconds=total_duration,
        logs=log.entries,
        error=error_msg,
    )
    return result, step_statuses


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status():
    """Print a colour-coded ASCII table of pipeline status to the terminal."""
    status = _load_status()
    if not status:
        print("No status file found. Run the pipeline first.")
        return

    col_w = 14
    header = f"{'Subject':<10}" + "".join(f"{STEP_NAMES[s]:^{col_w}}" for s in sorted(STEPS_CONFIG))
    sep = "-" * len(header)
    print("\nPATHS Pipeline Status")
    print(sep)
    print(header)
    print(sep)

    icons = {"success": "done", "failed": "FAILED", "not_run": "...", "skipped": "skip"}

    for sub_id in sorted(status.keys()):
        row = f"sub-{sub_id:<6}"
        for step_num in sorted(STEPS_CONFIG.keys()):
            entry = status[sub_id].get(str(step_num), {})
            s = entry.get("status", "not_run")
            label = icons.get(s, s)
            row += f"{label:^{col_w}}"
        print(row)

    print(sep)
    print(f"Status file: {STATUS_FILE}\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_pipeline(
    n_subjects: int = None,
    steps_to_run: list = None,
    subjects_to_run: list = None,
    subject_steps_map: dict = None,
    skip_completed: bool = True,
) -> list[SubjectResult]:
    """
    Main pipeline runner.

    Args:
        n_subjects:        Process only the first N subjects (sorted).
        steps_to_run:      Steps to run for every subject (default: all).
        subjects_to_run:   Explicit list of subject IDs to process.
        subject_steps_map: Per-subject step overrides, e.g. {'101': [2, 3], '127': [3]}.
        skip_completed:    If True (default), skip steps already marked 'success'
                           in the status file.
    """
    if steps_to_run is None and subject_steps_map is None:
        steps_to_run = sorted(STEPS_CONFIG.keys())

    # Discover subjects
    all_subjects = sorted(p.name.replace("sub-", "") for p in BIDS_ROOT.glob("sub-*"))

    if subject_steps_map is not None:
        subjects = [s for s in subject_steps_map if s in all_subjects]
    elif subjects_to_run is not None:
        subjects = [s for s in subjects_to_run if s in all_subjects]
    elif n_subjects:
        subjects = all_subjects[:n_subjects]
    else:
        subjects = all_subjects

    if not subjects:
        print("No subjects found!")
        return []

    # Load existing status (shared read-only by all workers)
    existing_status = _load_status()

    log_file = setup_logging()
    start_time = datetime.now()

    steps_display = "custom per subject" if subject_steps_map else str(steps_to_run)

    header_logs = [
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    | Pipeline Started",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    |    Subjects: {len(subjects)} {subjects if len(subjects) <= 10 else subjects[:10]}",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    |    Steps: {steps_display}",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    |    Parallel jobs: {N_JOBS}",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    |    Skip completed: {skip_completed}",
    ]
    for line in header_logs:
        print(line)

    # Parallel execution
    if subject_steps_map is not None:
        raw_results = Parallel(n_jobs=N_JOBS)(
            delayed(process_subject_pipeline)(
                sub, subject_steps_map.get(sub, steps_to_run or [1, 2, 3]),
                skip_completed, existing_status,
            )
            for sub in subjects
        )
    else:
        raw_results = Parallel(n_jobs=N_JOBS)(
            delayed(process_subject_pipeline)(
                sub, steps_to_run,
                skip_completed, existing_status,
            )
            for sub in subjects
        )

    # Unpack (result, step_statuses) tuples and merge into status file
    results = []
    updated_status = dict(existing_status)  # start from existing, merge new data
    for result, step_statuses in raw_results:
        results.append(result)
        sub_entry = dict(updated_status.get(result.subject_id, {}))
        sub_entry.update(step_statuses)
        updated_status[result.subject_id] = sub_entry

    _save_status(updated_status)

    # Summary
    duration = datetime.now() - start_time
    successful = sum(1 for r in results if r.success)
    failed_subjects = [r.subject_id for r in results if not r.success]

    footer_logs = [
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    | Pipeline Finished",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    |    Total duration: {duration}",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    |    Success: {successful}/{len(subjects)} subjects",
    ]
    if failed_subjects:
        footer_logs.append(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | WARNING |    Failed: {failed_subjects}"
        )
    for line in footer_logs:
        print(line)

    write_logs_to_file(log_file, header_logs, results, footer_logs)
    print(f"\nLog saved to: {log_file}")
    print(f"Status file:  {STATUS_FILE}")

    return results


def rerun_failed(steps_to_run: list = None) -> list[SubjectResult]:
    """Convenience: rerun only subjects/steps that are currently marked as failed."""
    status = _load_status()
    subject_steps_map = {}
    for sub_id, steps in status.items():
        failed_steps = [int(k) for k, v in steps.items() if v.get("status") == "failed"]
        if failed_steps:
            subject_steps_map[sub_id] = steps_to_run or failed_steps
    if not subject_steps_map:
        print("No failed subjects found.")
        return []
    print(f"Rerunning failed steps for: {list(subject_steps_map.keys())}")
    return run_pipeline(subject_steps_map=subject_steps_map, skip_completed=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default: run all subjects, all steps (skips already-completed ones)
    results = run_pipeline(subjects_to_run=['101', '114', '127', '128'], steps_to_run=[2, 3, 4])

    # Show status table after run
    show_status()

    # --- Other usage examples (uncomment as needed) ---

    # Run specific subjects and steps:
    # results = run_pipeline(subjects_to_run=['101', '127'], steps_to_run=[2, 3])

    # Run different steps per subject:
    # results = run_pipeline(subject_steps_map={
    #     '101': [1, 2, 3, 4],
    #     '127': [2, 3, 4],
    #     '128': [3, 4],
    # })

    # Rerun only subjects that failed last time:
    # results = rerun_failed()

    # Force rerun even if steps are already marked done:
    # results = run_pipeline(subjects_to_run=['127'], steps_to_run=[2, 3], skip_completed=False)

    # Show current status without running anything:
    # show_status()
