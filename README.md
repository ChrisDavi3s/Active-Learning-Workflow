# Active Learning Workflow

A two-step workflow for running Active Learning with a committee of ASE-compatible calculators.

The high-level idea of the workflow:

1) From an XYZ file (easily modifiable to be any list of structures), run a committee of ML potentials to generate trajectories and their associated errors (force, stress, energy uncertainties).  
2) Aggregate the results, produce overall statistics, and identify "worst frames" for further inspection or retraining.

A suggested starting point:

```text
.
├── run_active_learning.py         # Main workflow script
├── input_structures.xyz           # Input structures file
├── models/                        # Model directory
│   ├── deployed_model_0.pth       # Committee model 1
│   ├── deployed_model_1.pth       # Committee model 2
│   └── deployed_model_2.pth       # Committee model 3

```
--------------------------------------------------------------------------------
# (1) run_active_learning.py

Needs to run with a GPU (well, no, but ideally yes).

On Young (an HPC node) you might request an interactive node with something like:
```bash
qrsh -pe smp 8 -l mem=2056M,h_rt=2:00:00,gpu=1,tmpfs=50G -now no -A MCC_bulk_isl
```

This script implements a workflow for:
1. Structure relaxation (optional).  
2. NPT molecular dynamics.  
3. Uncertainty analysis using a committee of models.

### Requirements
- Python 3.8+  
- ASE  
- NequIP (or another ML potential package)  
- NumPy  
- Matplotlib  
- tqdm  

### Directory Structure


Generated output in workflow_results/:

```text
workflow_results/              # Generated output
├── logs/                      # Workflow logs
├── run_0000/                  # Individual run results
│   ├── relaxation.traj
│   ├── npt.extxyz
│   ├── core_stats.npz
│   ├── atom_data.npz
│   └── run_statistics.png
├── run_0001/
├── run_N/
```

### Calculator Configuration

The workflow requires ASE-compatible calculators. By default, we rely on a small "committee" of ML potentials. For example:

```python
# Multiple NequIP models:
from nequip.ase import NequIPCalculator
ASE_CALCULATORS = [NequIPCalculator.from_deployed_model(model_path=f"deployed_model_{i}.pth", 
                                                       device="cuda") for i in range(3)]

# Single MACE model:
from mace.calculators import MACECalculator
ASE_CALCULATORS = [MACECalculator(model_path="model.pt", device="cuda")]
```

Customizing your committee is straightforward:  
• Include any ASE-compatible calculator(s).  
• The first calculator in the list is used for the MD run. The script computes uncertainties by comparing the predictions among all listed calculators.

### Usage

Below is a snippet of configuration for run_active_learning.py:

```python
WORKFLOW_CONFIG = {
    'PATHS': {
        'BASE_DIR': Path('workflow_results'),
        'INPUT_XYZ': Path('generated_structures.xyz'),
        'LOG_DIR': None
    },
    'RELAXATION': { # Optional relaxation, just set this to 0 if you don't want to relax
        'STEPS': 5,
        'FORCE_CONVERGENCE': 0.01
    },
    'NPT': {
        'STEPS': 20,
        'TIME_STEP': 2 * units.fs,
        'PRESSURE': 1.01325 * units.bar,
        'SAVE_INTERVAL': 10,
        'TEMPERATURE': 600,
        'THERMOSTAT_TIME': 25 * units.fs,
        'BAROSTAT_TIME': 100 * units.fs
    },
    'ANALYSIS': {
        'DPI': 300,
        'FIGURE_SIZES': {
            'MAIN': (15, 15),
            'SPECIES': (10, 6)
        }
    },
    'CALCULATORS': ASE_CALCULATORS
}
```

Simply run:
```bash
$ python run_active_learning.py
```

The script will:  
1) Load each frame of generated_structures.xyz.  
2) Optionally perform a relaxation if STEPS > 0 in RELAXATION.  
3) Run an NPT simulation, producing npt.extxyz.  
4) Evaluate uncertainties from the committee and save results as core_stats.npz and atom_data.npz.  
5) Generate a quick figure (run_statistics.png) summarizing forces, uncertainties, and species-specific data.  
6) Store everything in workflow_results/run_XXXX/ directories.

### Example Analysis Plots
Below is an example style of the per-run figure (run_statistics.png):

![Example analysis plots](./images/run_statistics.png)

--------------------------------------------------------------------------------
# (2) analyze_active_learning.py

This script processes all the runs in workflow_results/, generates aggregated plots, and (optionally) identifies "worst frames" based on user-defined criteria.

Workflow steps:
1. Search for run_* subdirectories (e.g., run_0000, run_0001, etc.).  
2. Collect statistics from core_stats.npz and atom_data.npz across all runs.  
3. Generate combined histograms, time series, species-based analysis, etc.  
4. Select "worst frames" (high force or high uncertainty) according to configurable CRITERIA.  
5. Output summary plots (analysis.png, time_series.png) and optionally gather worst frames in a single worst_frames.xyz.

### Usage
After you have completed your runs with run_active_learning.py, invoke:
```bash
$ python analyze_active_learning.py
```
It reads configuration from RUN_CONFIG in analyze_active_learning.py, which includes:

• PATHS.BASE_DIR: the same directory used previously.  
• ANALYSIS.N_SKIP_FRAMES: frames to ignore (e.g., equilibrating).  
• WORST_FRAMES.ENABLED: True or False to enable/disable "worst frame" selection.  
• WORST_FRAMES.CRITERIA: A list of dictionaries specifying which metrics to look at (force vs. energy vs. stress, and max vs. mean, value vs. uncertainty), how many frames to pick, species filters, etc.  
• WORST_FRAMES.WINDOWS: "WITHIN_CRITERION" and "GLOBAL" windows that exclude neighboring frames once one is chosen.

### Directory Structure

```text
workflow_results/
├── analysis/
│   ├── plots/
│   │   ├── analysis.png
│   │   └── time_series.png
│   └── worst_runs/
│       ├──  ** PLOT OF CRITERIA 1 **
│       ├──  worst_frames.json
│       └──  worst_frames.xyz
└── run_*/                 # Original run directories
```
### Example Criterion
```python
'WORST_FRAMES': {
    'ENABLED': True,
    'CRITERIA': [
        {
            'metric': 'force',
            'stat_type': 'max',
            'measure': 'value',
            'n_frames': 10,
            'frame_range': (100, None),
            'within_window': 50
        },
        {
            'metric': 'force',
            'stat_type': 'mean',
            'measure': 'uncertainty',
            'species': 'Zn',
            'n_frames': 20,
            'frame_range': (0, 500),
            'within_window': 25
        }
    ],
    'WINDOWS': {
        'WITHIN_CRITERION': 5,
        'GLOBAL': 2
    }
}
```
Once run, it produces aggregated plots in workflow_results/analysis/plots/ (analysis.png and time_series.png), plus worst-frames data in analysis/worst_runs/.

### Example Analysis Plots

The script generates a summary figure (analysis.png) and a time series plot (time_series.png) across all runs:

analysis.png:
![Example analysis plots](./images/analysis.png)

time_series.png:
![Example time series plot](./images/time_series.png)


--------------------------------------------------------------------------------
# Concluding Remarks

This two-part pipeline allows you to:  
• Run a loop of relaxations + NPT simulations and measure uncertainty with multiple ML potentials.  
• Consolidate results, create summary plots across all runs, and automatically flag frames worth further investigation or higher accuracy calculations.  


# Acknowledgements

This code was written by Chris Davies 2025; if you use this for a paper, please give appropriate credit.

Peace.
