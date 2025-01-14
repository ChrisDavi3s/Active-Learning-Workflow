# Active Learning Workflow
An Active Learning workflow for a committee of ASE calculators
The general idea is:

1) From an XYZ file, run a committee, generate trajectories and their associated errors
2) Generate overall statistics and find worst frames in these generated runs.

# 1) run_active_learning.py

THIS NEEDS TO RUN WITH A GPU (well no but yes)

On Young login node run this to get an interactive node:
```bash
qrsh -pe smp 8 -l mem=2056M,h_rt=2:00:00,gpu=1,tmpfs=50G -now no -A MCC_bulk_isl
```

This script implements a workflow for:
1. Structure relaxation
2. NPT molecular dynamics
3. Uncertainty analysis using model committees

The idea is, we have run a loop on a set of structures that have been generated:

1) [optional] We run a relaxation on the structure
2) We run a NPT simulation on the structure
3) We calculate the uncertainty of the forces on the structure

### Requirements
- Python 3.8+
- ASE
- NequIP (or other ML potential package)
- NumPy
- Matplotlib
- tqdm

### Directory Structure
```text
.
├── run_active_learning.py         # Main workflow script
├── input_structures.xyz           # Input structures file
├── models/                        # Model directory
│   ├── deployed_model_0.pth       # Committee model 1
│   ├── deployed_model_1.pth       # Committee model 2
│   └── deployed_model_2.pth       # Committee model 3
└── workflow_results/              # Generated output
    ├── logs/                      # Workflow logs
    │   └── 
    ├── run_0000/                  # Individual run results
    │   ├── relaxation.traj
    │   ├── npt.extxyz
    │   └── 
    ├── run_0001/
    └── run_N/
```

### Calculator Configuration
The workflow requires ASE-compatible calculators. 

Due to me being lazy, the first committee model is used to run the MD. (todo)

```python
# Multiple NequIP models:
from nequip.ase import NequIPCalculator
ASE_CALCULATORS = [NequIPCalculator.from_deployed_model(model_path=f"deployed_model_{i}.pth", 
                                                       device="cuda") for i in range(3)]

# Single MACE model:
from mace.calculators import MACECalculator
ASE_CALCULATORS = [MACECalculator(model_path="model.pt", device="cuda")]
```

### Usage

Config
```python
WORKFLOW_CONFIG = {
    'PATHS': {
        # Root directory for all output files and analysis
        'BASE_DIR': Path('workflow_results'),

        # Path to input structure file which contains initial structures
        'INPUT_XYZ': Path('generated_structures.xyz'),

        # Directory for log files. If None, creates 'logs' in BASE_DIR
        'LOG_DIR': None
    },

    'RELAXATION': {
        # Maximum number of geometry optimization steps
        # Set to 0 to skip relaxation entirely
        'STEPS': 5,

        # Force convergence criterion in eV/Å
        # Relaxation stops when max force is below this value (or STEPS is reached)
        'FORCE_CONVERGENCE': 0.01
    },

    'NPT': {
        # Total number of MD steps to run
        # Actual simulation time = STEPS * TIME_STEP
        'STEPS': 20,

        # MD integration timestep in femtoseconds
        'TIME_STEP': 2 * units.fs,

        # Target pressure in bar (1.01325 bar = 1 atm)
        'PRESSURE': 1.01325 * units.bar,

        # Save structure to trajectory every N steps
        'SAVE_INTERVAL': 10,

        # Target temperature in Kelvin
        'TEMPERATURE': 600,

        # Thermostat coupling time - controls temperature fluctuations
        # Larger values = slower coupling but more stable
        'THERMOSTAT_TIME': 25 * units.fs,

        # Barostat coupling time - controls pressure fluctuations
        # Should typically be 3-4x thermostat time
        'BAROSTAT_TIME': 100 * units.fs
    },

    'ANALYSIS': {
        # Resolution of saved plots in dots per inch
        # 300 is publication quality
        'DPI': 300,
        'FIGURE_SIZES': {
            # Size of main analysis plots in inches (width, height)
            'MAIN': (15, 15),
            # Size of species-specific plots
            'SPECIES': (10, 6)
        }
    },

    # List of ASE calculators for committee predictions
    'CALCULATORS': ASE_CALCULATORS
}
```

```bash
$ python run_active_learning.py
```
### Example Analysis Plots
Below are example plots generated for each run:

![Example analysis plots](./images/run_statistics.png)

# 2) analyze_active_learning.py

