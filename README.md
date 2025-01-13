# active_learning_workflow
An Active Learning workflow for a committee of ASE calculators

## run_active_learning.py

This script implements a workflow for:
1. Structure relaxation
2. NPT molecular dynamics
3. Uncertainty analysis using model committees

### Requirements
- Python 3.8+
- ASE
- NequIP (or other ML potential package)
- PyTorch (with CUDA support)
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

$ python run_active_learning.py

Key Configuration Parameters:
--------------------------
BASE_DIR          : Output directory (default: "workflow_results")
INPUT_XYZ         : Input structure file (default: "generated_structures.xyz")
RELAXATION_STEPS  : Number of relaxation steps (default: 50)
NPT_STEPS         : Number of NPT simulation steps (default: 4000)
NPT_TEMPERATURE   : Temperature for NPT simulation (default: 600K)

Output:
-------
Each run directory contains:
- relaxation.traj  : Relaxation trajectory
- relaxed.xyz      : Final relaxed structure
- npt.extxyz      : NPT trajectory
- core_stats.npz   : Compressed analysis results
- run_statistics.png : Analysis plots
"""

## Example Analysis Plots
Below are example plots generated for each run:

![Example analysis plots](./images/run_statistics.png)

The plots show:
1. Force vs Uncertainty correlation (top left)
2. Force evolution during NPT (top right) 
3. Force distribution (middle left)
4. Uncertainty distribution (middle right)
5. Species-specific forces (bottom left)
6. Species-specific uncertainties (bottom right)



