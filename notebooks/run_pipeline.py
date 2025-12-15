import papermill as pm
from pathlib import Path
from joblib import Parallel, delayed
import time

# --- CONFIGURATION ---
INPUT_NOTEBOOK = "/Users/daniel/PhD/Projects/psd-paths/notebooks/01_template-step1.ipynb"  # Your template notebook
BIDS_ROOT = Path("/Users/daniel/PhD/Projects/psd-paths/data/bids")               # Where your subjects are
OUTPUT_DIR = Path("/Users/daniel/PhD/Projects/psd-paths/pipeline")      # Where to save the finished notebooks
N_JOBS = 4                                 # -1 = use all cores, 2 = use 2 cores
KERNEL_NAME = "psd-paths-3.13"

def process_subject(subject_id):
    """
    Runs the notebook for a single subject using Papermill.
    """
    output_path = OUTPUT_DIR / f"sub-{subject_id}_1step_analysis.ipynb"
    
    print(f"⏳ Starting Subject {subject_id}...")
    
    try:
        pm.execute_notebook(
            input_path=INPUT_NOTEBOOK,
            output_path=output_path,
            parameters={"subject_id": subject_id}, # Injects variable into 'parameters' cell
            kernel_name="python3",
            progress_bar=False  # Turn off per-notebook bars to keep terminal clean
        )
        print(f"✅ Finished Subject {subject_id}")
        return True
    except Exception as e:
        print(f"❌ FAILED Subject {subject_id}: {e}")
        return False

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Get list of subjects automatically from BIDS folder
    # Looks for folders named 'sub-101', 'sub-102', etc.
    subject_paths = sorted(list(BIDS_ROOT.glob("sub-*")))
    subjects = [p.name.replace("sub-", "") for p in subject_paths]
    
    print(f"Found {len(subjects)} subjects: {subjects}")

    # 3. Run in Parallel
    # n_jobs=-1 uses all available CPU cores
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_subject)(sub) for sub in subjects
    )

    print(f"\nTotal completed: {sum(results)} / {len(subjects)}")