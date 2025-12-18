import papermill as pm
from pathlib import Path
from joblib import Parallel, delayed
import logging
from datetime import datetime

# --- CONFIGURATION ---
BIDS_ROOT = Path("/Users/daniel/PhD/Projects/psd-paths/data/bids")
OUTPUT_ROOT = Path("/Users/daniel/PhD/Projects/psd-paths/outputs/pipeline")
OUTPUT_LOG = Path("/Users/daniel/PhD/Projects/psd-paths/outputs/log")
N_JOBS = 4
KERNEL_NAME = "psd-paths-3.13"

STEPS_CONFIG = {
     1: {
         "template": "/Users/daniel/PhD/Projects/psd-paths/WIP/01_template-step1.ipynb",
         "suffix": "step1-preprocessing"
     },
          2: {
         "template": "/Users/daniel/PhD/Projects/psd-paths/WIP/02_Epochs.ipynb",
         "suffix": "step1b-epochs-psd-analysis"
     },
     3: {
         "template": "/Users/daniel/PhD/Projects/psd-paths/WIP/03_Template_ICA_OLD_style.ipynb",
         "suffix": "step2-psd-analysis"
     }
}

def setup_logging():
    """Set up logging to both console and file with timestamp."""
    # Create log directory if it doesn't exist
    OUTPUT_LOG.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_LOG / f"pipeline_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def run_notebook_step(subject_id, step_num, input_note, output_note):
    """Helper to wrap the papermill execution logic."""
    logging.info(f"⏳ Sub-{subject_id} | Step {step_num} | Starting...")
    pm.execute_notebook(
        input_path=input_note,
        output_path=output_note,
        parameters={"subject_id": subject_id},
        kernel_name=KERNEL_NAME,
        progress_bar=False
    )

def process_subject_pipeline(subject_id, steps_to_run):
    """
    Runs selected steps for a single subject.
    """
    subject_output_dir = OUTPUT_ROOT / f"sub-{subject_id}"
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Loop only through the steps requested in steps_to_run
        for step_num in steps_to_run:
            if step_num not in STEPS_CONFIG:
                logging.warning(f"⚠️ Step {step_num} not found in STEPS_CONFIG. Skipping.")
                continue
                
            cfg = STEPS_CONFIG[step_num]
            input_path = cfg["template"]
            output_path = subject_output_dir / f"sub-{subject_id}_{cfg['suffix']}.ipynb"
            
            run_notebook_step(subject_id, step_num, input_path, output_path)
            
        logging.info(f"✅ Sub-{subject_id} | Selected Steps Complete")
        return True
    except Exception as e:
        logging.error(f"❌ Sub-{subject_id} | FAILED at Step {step_num}: {e}")
        return False

def run_pipeline(n_subjects=None, steps_to_run=None):
    """
    Run the pipeline for subjects.
    
    Args:
        n_subjects: Number of subjects to process.
        steps_to_run: List of step integers (e.g., [1] or [1, 2]). 
                  If None, runs all steps in STEPS_CONFIG.
    """
    # Default to all steps if none specified
    if steps_to_run is None:
        steps_to_run = sorted(STEPS_CONFIG.keys())
    
    # Get subjects
    subject_paths = sorted(list(BIDS_ROOT.glob("sub-*")))
    subjects = [p.name.replace("sub-", "") for p in subject_paths]
    
    if n_subjects is not None:
        subjects = subjects[:n_subjects]
    
    start_time = datetime.now()
    logging.info(f"🚀 Starting Pipeline for {len(subjects)} subjects.")
    logging.info(f"🏃 Running Steps: {steps_to_run}")
    logging.info(f"⏰ Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Pass steps_to_run into the parallel worker
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_subject_pipeline)(sub, steps_to_run) for sub in subjects
    )

    end_time = datetime.now()
    duration = end_time - start_time
    
    logging.info(f"\nPipeline Finished. Successfully processed: {sum(results)} / {len(subjects)}")
    logging.info(f"⏰ End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"⏱️ Duration: {duration}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Set up logging first
    log_file = setup_logging()
    logging.info(f"📝 Logging to: {log_file}")
    
    # EXAMPLES:
    # run_pipeline(n_subjects=2)             # Runs all steps for 2 subjects
    # run_pipeline(steps_to_run=[2])         # Runs ONLY Step 2 for all subjects
    # run_pipeline(n_subjects=1, steps_to_run=[1]) # Runs ONLY Step 1 for 1 subject
    
    run_pipeline(steps_to_run=[3])