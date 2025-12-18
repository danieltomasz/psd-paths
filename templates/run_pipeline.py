import papermill as pm
from pathlib import Path
from joblib import Parallel, delayed
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# --- CONFIGURATION ---
BIDS_ROOT = Path("/Users/daniel/PhD/Projects/psd-paths/data/bids")
OUTPUT_ROOT = Path("/Users/daniel/PhD/Projects/psd-paths/outputs/pipeline")
OUTPUT_LOG = Path("/Users/daniel/PhD/Projects/psd-paths/outputs/log")
N_JOBS = 4
KERNEL_NAME = "psd-paths-3.13"

STEPS_CONFIG = {
    1: {"template": "/Users/daniel/PhD/Projects/psd-paths/templates/01_template-step1.ipynb", "suffix": "step1-preprocessing"},
    2: {"template": "/Users/daniel/PhD/Projects/psd-paths/templates/02_Epochs.ipynb", "suffix": "step1b-epochs-psd-analysis"},
    3: {"template": "/Users/daniel/PhD/Projects/psd-paths/templates/03_Template_ICA_OLD_style.ipynb", "suffix": "step2-psd-analysis"}
}


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


class LogCollector:
    """Collects log messages in memory for later writing."""
    
    def __init__(self, subject_id: str):
        self.subject_id = subject_id
        self.entries = []
    
    def _log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.entries.append(f"{timestamp} | {level:7} | {message}")
        # Also print to console for real-time feedback
        print(f"{timestamp} | {level:7} | {message}")
    
    def info(self, message: str):
        self._log("INFO", message)
    
    def warning(self, message: str):
        self._log("WARNING", message)
    
    def error(self, message: str):
        self._log("ERROR", message)


def setup_logging() -> Path:
    """Creates log directory and returns the log file path."""
    OUTPUT_LOG.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_LOG / f"pipeline_{timestamp}.log"


def write_logs_to_file(log_file: Path, header_logs: list, results: list[SubjectResult], footer_logs: list):
    """Writes all collected logs to file in a clean, organized format."""
    with open(log_file, 'w') as f:
        # Write header (pipeline start info)
        for line in header_logs:
            f.write(line + "\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("SUBJECT LOGS\n")
        f.write("=" * 70 + "\n\n")
        
        # Write each subject's logs as a block
        for result in sorted(results, key=lambda r: r.subject_id):
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            f.write(f"--- Sub-{result.subject_id} | {status} | {result.completed_steps}/{result.total_steps} steps | {result.duration_seconds:.1f}s ---\n")
            for log_line in result.logs:
                f.write(f"    {log_line}\n")
            f.write("\n")
        
        # Write footer (summary)
        f.write("=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n")
        for line in footer_logs:
            f.write(line + "\n")


def process_subject_pipeline(subject_id: str, steps_to_run: list, log_output: bool = False) -> SubjectResult:
    """Runs selected steps for a single subject, collecting logs in memory."""
    log = LogCollector(subject_id)
    start_time = datetime.now()
    
    subject_output_dir = OUTPUT_ROOT / f"sub-{subject_id}"
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_requested = len(steps_to_run)
    error_msg = None

    for step_num in steps_to_run:
        if step_num not in STEPS_CONFIG:
            log.warning(f"Sub-{subject_id} | Step {step_num} missing in config, skipping.")
            continue
            
        cfg = STEPS_CONFIG[step_num]
        input_path = cfg["template"]
        output_path = subject_output_dir / f"sub-{subject_id}_{cfg['suffix']}.ipynb"
        
        try:
            log.info(f"Sub-{subject_id} | Step {step_num} | Starting...")
            
            pm.execute_notebook(
                input_path=input_path,
                output_path=str(output_path),
                parameters={"subject_id": subject_id},
                kernel_name=KERNEL_NAME,
                progress_bar=False,
                log_output=log_output 
            )
            
            log.info(f"Sub-{subject_id} | Step {step_num} | ✅ COMPLETED")
            success_count += 1
            
        except Exception as e:
            error_msg = str(e)[:300]
            log.error(f"Sub-{subject_id} | Step {step_num} | ❌ FAILED: {error_msg}")
            break
    
    duration = (datetime.now() - start_time).total_seconds()
    
    return SubjectResult(
        subject_id=subject_id,
        success=(success_count == total_requested),
        completed_steps=success_count,
        total_steps=total_requested,
        duration_seconds=duration,
        logs=log.entries,
        error=error_msg
    )


def run_pipeline(n_subjects: int = None, steps_to_run: list = None) -> list[SubjectResult]:
    """Main pipeline runner with deferred log writing."""
    if steps_to_run is None:
        steps_to_run = sorted(STEPS_CONFIG.keys())
    
    # Discover subjects
    subject_paths = sorted(list(BIDS_ROOT.glob("sub-*")))
    subjects = [p.name.replace("sub-", "") for p in subject_paths]
    if n_subjects:
        subjects = subjects[:n_subjects]
    
    if not subjects:
        print("⚠️ No subjects found!")
        return []
    
    # Setup
    log_file = setup_logging()
    start_time = datetime.now()
    
    # Header logs
    header_logs = [
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    | 🚀 Pipeline Started",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    |    Subjects: {len(subjects)}",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    |    Steps: {steps_to_run}",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    |    Parallel jobs: {N_JOBS}",
    ]
    for line in header_logs:
        print(line)

    # Parallel execution - each worker returns its logs
    results: list[SubjectResult] = Parallel(n_jobs=N_JOBS)(
        delayed(process_subject_pipeline)(sub, steps_to_run) for sub in subjects
    )

    # Compute summary
    duration = datetime.now() - start_time
    successful = sum(1 for r in results if r.success)
    failed_subjects = [r.subject_id for r in results if not r.success]
    
    # Footer logs
    footer_logs = [
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    | 🏁 Pipeline Finished",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    |    Total duration: {duration}",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO    |    Success: {successful}/{len(subjects)} subjects",
    ]
    if failed_subjects:
        footer_logs.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | WARNING |    Failed: {failed_subjects}")
    
    for line in footer_logs:
        print(line)

    # Write everything to file at once
    write_logs_to_file(log_file, header_logs, results, footer_logs)
    print(f"\n📄 Log saved to: {log_file}")

    return results


if __name__ == "__main__":
    results = run_pipeline(steps_to_run=[1,2,3])