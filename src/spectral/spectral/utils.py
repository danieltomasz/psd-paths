"""Utils for helping the analysis of the data"""

import os
from datetime import datetime
from pathlib import Path
import toml  # Use toml instead of ConfigParser
from typing import Dict, Optional, Union
import mne

def print_timestamp(prefix: str = ""):
    """
    Print the current timestamp with an optional prefix message.
    
    Args:
        prefix: Optional message to print before the timestamp
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    if prefix:
        print(f"{prefix}: {timestamp}")
    else:
        print(timestamp)


def find_project_root(start_path: Optional[Path] = None, 
                     marker_file: str = 'settings.toml') -> Path:
    """
    Find the project root by looking for a marker file (like settings.toml).
    
    This function starts from the current directory (or a specified path) and
    walks up the directory tree until it finds the marker file or reaches the
    filesystem root.
    
    Args:
        start_path: Starting directory (defaults to current working directory)
        marker_file: Name of file that marks the project root
        
    Returns:
        Path to the project root directory
        
    Raises:
        FileNotFoundError: If marker file cannot be found
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()
    
    current = start_path
    
    # Walk up the directory tree
    while current != current.parent:  # Stop at filesystem root
        if (current / marker_file).exists():
            return current
        current = current.parent
    
    # If we get here, we didn't find the marker file
    raise FileNotFoundError(
        f"Could not find '{marker_file}' in any parent directory of {start_path}"
    )



def load_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load configuration from a TOML file.
    
    This function loads all your analysis parameters from a single source,
    making your analysis reproducible and easy to modify.
    """
    if config_path is None:
        project_root = find_project_root()
        config_path = project_root / 'settings.toml'
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load and return the TOML configuration
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    # You can add validation here if needed
    _validate_config(config)
    
    return config

def _validate_config(config: Dict):
    """
    Validate that the configuration has all required fields.
    
    This helps catch configuration errors early, before they cause
    mysterious failures deep in your analysis pipeline.
    """
    required_sections = ['paths', 'preprocessing']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config: [{section}]")
    
    # Check that channels_to_remove is a list
    if 'channels_to_remove' in config['preprocessing']:
        if not isinstance(config['preprocessing']['channels_to_remove'], list):
            raise ValueError("channels_to_remove must be a list")
 

class ProjectPaths:
    """
    Simple path management for EEG analysis projects.
    
    Organizes files into just three main categories:
    - data: your raw files
    - processing: intermediate files while you work
    - outputs: final results to share
    """
    
    def __init__(self, subject_id: Union[str, int], config_path: Optional[Path] = None):
        """Initialize paths for a specific subject."""
        # Format subject ID with leading zeros (1 becomes "001")
        self.subject_id = f"{int(subject_id):03d}"
        self.subject = f"sub-{self.subject_id}"
        
        # Get project root from config
        self.config = load_config(config_path)
        self.root = Path(self.config['paths']['project_root']).resolve()
        
        # Set up the simple structure
        self._setup_paths()
    
    def _setup_paths(self):
        """Create a simple, logical structure for your project."""
        
        # Raw data - keep it safe and separate
        self.data = self.root / 'data' / 'raw'   / self.subject
        self.data_bids = self.data / 'raw'   # BIDS formatted data
        # Processing - all intermediate files go here
        self.processing = self.root /'data' / 'derrivatives' / self.subject
        self.preprocessed = self.processing / 'preprocessed'  # Cleaned/filtered data
        self.epochs = self.processing / 'epochs'              # Epoched data
        self.analysis = self.processing / 'analysis'          # PSD, connectivity, etc.
        self.ica = self.analysis / 'ica'                      # ICA results
        # Outputs - things you want to keep and share
        self.outputs = self.root / 'outputs' 
        self.figures = self.outputs / 'figures' / self.subject
        self.reports = self.outputs / 'reports' / self.subject
        self.specparam = self.outputs / 'specparam' / self.subject # Spectral parameters
        # One place for all the miscellaneous stuff
        self.logs = self.root /'data'  / 'logs'
    
    def make_filename(self, description: str, extension: str = '.fif') -> str:
        """
        Create consistent filenames.
        Example: make_filename('filtered-1-45Hz') → 'sub-001_filtered-1-45Hz.fif'
        """
        return f"{self.subject}_{description}{extension}"
    
    def create_directories(self):
        """Create all directories for this subject."""
        all_dirs = [
            self.data,
            self.preprocessed,
            self.epochs, 
            self.analysis,
            self.figures,
            self.reports,
            self.specparam,
            self.ica,
            self.logs
        ]
        
        for directory in all_dirs:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Created directories for {self.subject}")
        print(f"Project root: {self.root}")
    
    def clean_empty_directories(self):
        """
        Remove empty directories in the derivatives folder.
        
        Useful for keeping your project tidy by removing folders
        from analyses you didn't run.
        """
        for root, dirs, files in os.walk(self.subject_derivatives, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    print(f"✗ Removed empty: {dir_path.relative_to(self.project_root)}")
    
    def show(self, selection: Optional[list[str]] = None):
        """
        Displays specified or all project paths in a formatted way.

        Args:
            selection (Optional[list[str]]): A list of path attribute names to display.
                                             If None, all paths will be shown.
                                             Example: ['data', 'figures', 'epochs']
        """
        print("─" * 60)
        print(f"Paths for {self.subject}")
        print(f"Project Root: {self.root}")
        print("─" * 60)

        # 1. Discover all attributes that are Path objects
        all_path_attrs = {
            key: val
            for key, val in self.__dict__.items()
            if isinstance(val, Path) and key != 'root'
        }

        # 2. Determine which paths to display based on user selection
        paths_to_display = {}
        if selection is None:
            # If no selection, use all discovered paths
            paths_to_display = all_path_attrs
        else:
            # If user made a selection, filter for those
            for name in selection:
                if name in all_path_attrs:
                    paths_to_display[name] = all_path_attrs[name]
                else:
                    print(f"  ✗ Warning: '{name}' is not a valid path attribute.")
        
        if not paths_to_display:
            return # Exit if nothing to show

        # 3. Format and print the output
        max_key_length = max(len(key) for key in paths_to_display.keys()) + 1
        
        for key, path in sorted(paths_to_display.items()):
            print(f"  {key:<{max_key_length}}: {path}")
            
        print("─" * 60)