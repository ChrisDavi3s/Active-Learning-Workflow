# Copyright Chris Davies 2025 
# Usage under attached LICENSE

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
from datetime import datetime
import seaborn as sns
import json
from typing import Dict, List, Any, Tuple, Union
import heapq
import sys
from tqdm import tqdm
from ase.io import read, write

RUN_CONFIG = {
    'PATHS': {
        'BASE_DIR': Path('workflow_results'),
        'OUTPUT_DIR': None  # Will default to BASE_DIR/analysis if None
    },
    'ANALYSIS': {
        'N_SKIP_FRAMES': 0,     # Number of initial frames to skip in analysis/plots (e.g. for equilibration)
        'DPI': 300,
        'FIGURE_SIZES': {
            'MAIN': (15, 25),
            'TIME_SERIES': (15, 20),
            'WORST_FRAMES': (10, 6)
        }
    },
    'FAILURE_HANDLING': {                              # Options for handling failed runs  
        'ANALYZE_FAILED_RUNS': False,                  # Analyse failed runs in the same way as successful runs
        'EXCLUDE_FAILED_RUNS_FROM_WORST_FRAMES': False # Exclude failed runs from worst frames analysis
    },
    'WORST_FRAMES': {
        'ENABLED': True,  # Toggle to False to skip picking worst frames entirely
        'JSON_FILE': 'worst_frames.json',  
        'XYZ_FILE': 'worst_frames.xyz',             #None to skip saving XYZ file 
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
                'n_frames': 10,
                'frame_range': (100, None),
                'within_window': 50
            },
            {
                'metric': 'force',
                'stat_type': 'max',
                'measure': 'uncertainty',
                'species': 'Zn',
                'n_frames': 40,
                'frame_range': (100, None),
                'within_window': 25
            },
            {
                'metric': 'force',
                'stat_type': 'mean',
                'measure': 'uncertainty',
                'species': 'Zn',
                'n_frames': 20,
                'frame_range': (100, None),
                'within_window': 50
            }
        ],
        'WINDOWS': {
            'WITHIN_CRITERION': 5,
            'GLOBAL': 2
        }
    }
}

# Type Aliases
PathLike = Union[str, Path]
NDArray = np.ndarray
RunData = Dict[str, Any]
AggregatedData = Dict[str, Any]
FrameData = Dict[str, Any]
FrameTuple = Tuple[int, int, float]  # (run_idx, frame_idx, value)
CriteriaDict = Dict[str, Any]
Species = str

# Criterion type definition
VALID_METRICS = {'force', 'energy', 'stress'}
VALID_STAT_TYPES = {'max', 'mean'}
VALID_MEASURES = {'value', 'uncertainty'}

def setup_logging(output_dir: PathLike) -> logging.Logger:
    """Configure logging with file and console output.
    
    Parameters
    ----------
    output_dir : PathLike
        Directory where log files will be stored
        
    Returns
    -------
    logging.Logger
        Configured logger instance
        
    Notes
    -----
    Creates timestamped log files in output_dir/logs/
    Logs to both file and console with INFO level
    """
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'analysis_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def collect_frames_from_json(json_file: str, base_path: str = None) -> List[Dict]:
    """
    Read the JSON file and collect all frames into a list of ASE atoms objects.

    Parameters
    ----------
    json_file : str
        Path to the JSON file containing frame information
    base_path : str, optional
        Base path to prepend to run directories if they're relative paths

    Returns
    -------
    List[ase.Atoms]
        List of ASE Atoms objects for all frames
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    collected_frames = []

    # Process each criterion in the JSON
    for criterion_name, frames in data.items():
        print(f"\nProcessing criterion: {criterion_name}")

        for frame_info in frames:
            run_dir = frame_info['run_dir']
            if base_path:
                run_dir = os.path.join(base_path, run_dir)

            # Construct path to trajectory file
            traj_path = os.path.join(run_dir, 'npt.extxyz')

            if not os.path.exists(traj_path):
                print(f"Warning: Trajectory file not found at {traj_path}")
                continue

            try:
                # Read the specific frame from trajectory
                atoms = read(traj_path, index=frame_info['frame_idx'])

                # Add extra info to atoms object for reference
                atoms.info['original_run'] = frame_info['run_idx']
                atoms.info['original_frame'] = frame_info['frame_idx']
                atoms.info['criterion'] = criterion_name
                atoms.info['value'] = frame_info['value']
                atoms.info['stat_type'] = frame_info['stat_type']
                atoms.info['metric'] = frame_info['metric']
                atoms.info['measure'] = frame_info['measure']
                if frame_info.get('species'):
                    atoms.info['species'] = frame_info['species']

                collected_frames.append(atoms)
                print(f"Added frame {frame_info['frame_idx']} from run {frame_info['run_idx']}")
            except Exception as e:
                print(f"Error reading frame {frame_info['frame_idx']} from {traj_path}: {e}")

    return collected_frames

def validate_criterion(criterion: CriteriaDict) -> None:
    """Validate a criterion against allowed values.
    
    Parameters
    ----------
    criterion : CriteriaDict
        Dictionary containing metric, stat_type, and measure
        
    Raises
    ------
    ValueError
        If any values invalid

    """
    if criterion['metric'] not in VALID_METRICS:
        raise ValueError(f"Invalid metric: {criterion['metric']}. Must be one of {VALID_METRICS}")
    
    if criterion['stat_type'] not in VALID_STAT_TYPES:
        raise ValueError(f"Invalid stat_type: {criterion['stat_type']}. Must be one of {VALID_STAT_TYPES}")
        
    if criterion.get('measure', 'value') not in VALID_MEASURES:
        raise ValueError(f"Invalid measure: {criterion.get('measure')}. Must be one of {VALID_MEASURES}")


def load_run_data(run_dir: PathLike) -> RunData:
    """Load simulation data from directory.
    
    Parameters
    ----------
    run_dir : PathLike
        Directory containing core_stats.npz and atom_data.npz
        
    Returns
    -------
    RunData
        Dictionary with:
        - core_stats: Dict[str, NDArray], Global statistics
        - atom_data: Dict[str, NDArray], Per-atom data
        
    Raises
    ------
    ValueError
        If files missing or corrupted
        
    Notes
    -----
    Expected file structure:
    run_dir/
        ├── core_stats.npz
        └── atom_data.npz
    """
    try:
        core_stats = np.load(run_dir / 'core_stats.npz')
        atom_data = np.load(run_dir / 'atom_data.npz', allow_pickle=True)

        return {
            'core_stats': {key: core_stats[key] for key in core_stats.files},
            'atom_data': {key: atom_data[key] for key in atom_data.files}
        }
    except Exception as e:
        raise ValueError(f"Failed to load data from {run_dir}: {str(e)}")


def aggregate_runs(base_dir: Path) -> AggregatedData:
    """Combine data from multiple simulation runs.
    
    Parameters
    ----------
    base_dir : Path
        Directory containing run_* subdirectories
        
    Returns
    -------
    AggregatedData
        Dictionary with:
        - max_force_uncertainty: List[NDArray]
        - mean_force_uncertainty: List[NDArray]
        - energy_uncertainty: List[NDArray]
        - stress_uncertainty: List[NDArray]
        - max_forces: List[NDArray]
        - mean_forces: List[NDArray]
        - per_species_stats: Dict[str, Dict]
        - run_metadata: List[Dict]
        - frame_data: List[Dict]
        
    Notes
    -----
    Processes all run_* directories in base_dir
    Combines statistics and frame data
    Converts lists to numpy arrays
    """

    run_dirs = sorted(base_dir.glob('run_*'))

    aggregated = {
        'max_force_uncertainty': [],
        'mean_force_uncertainty': [],
        'energy_uncertainty': [],
        'stress_uncertainty': [],
        'max_forces': [],
        'mean_forces': [],
        'per_species_stats': {},
        'run_metadata': [],
        'frame_data': [] 
    }

    run_dirs = sorted(base_dir.glob('run_*'))

    # Attempt to load run status
    run_status_json = base_dir / 'workflow_status.json'
    failed_indices = set()
    if run_status_json.exists():
        try:
            with open(run_status_json, 'r') as f:
                status_data = json.load(f)
            for s in status_data.get('run_status', []):
                # Mark as failed if not all true
                if not (s.get('relaxation_success') and s.get('npt_success') and s.get('analysis_success')):
                    failed_indices.add(s['structure_index'])
        except Exception as e:
            logging.warning(f"Could not parse workflow_status.json: {e}")

    # Prepare aggregator
    aggregated = {
        'max_force_uncertainty': [],
        'mean_force_uncertainty': [],
        'energy_uncertainty': [],
        'stress_uncertainty': [],
        'max_forces': [],
        'mean_forces': [],
        'per_species_stats': {},
        'run_metadata': [],
        'frame_data': []
    }

    # Global flags
    analyze_failed_runs = RUN_CONFIG['FAILURE_HANDLING']['ANALYZE_FAILED_RUNS']

    for run_idx, run_dir in enumerate(tqdm(run_dirs, desc="Processing runs")):
        # Check if run is failed
        is_failed = (run_idx in failed_indices)
        if is_failed and not analyze_failed_runs:
            logging.info(f"Skipping failed run {run_dir}")
            continue

        try:
            data = load_run_data(run_dir)

            for key in data['core_stats']:
                if key in aggregated:
                    aggregated[key].append(data['core_stats'][key])

            n_frames = len(data['atom_data']['symbols'])
            for frame_idx in range(n_frames):
                frame_info = {
                    'run_idx': run_idx,
                    'frame_idx': frame_idx,
                    'species_data': {},
                    'global_stats': {},
                    'failed_run': is_failed
                }

                for key in ['max_forces','mean_forces','max_force_uncertainty',
                            'mean_force_uncertainty','energy_uncertainty','stress_uncertainty']:
                    if key in data['core_stats']:
                        frame_info['global_stats'][key] = data['core_stats'][key][frame_idx]

                sym_list = data['atom_data']['symbols'][frame_idx]
                force_list = data['atom_data']['forces'][frame_idx]
                unc_list = data['atom_data']['uncertainties'][frame_idx]
                for sym, force, unc in zip(sym_list, force_list, unc_list):
                    if sym not in aggregated['per_species_stats']:
                        aggregated['per_species_stats'][sym] = {
                            'forces': [], 'uncertainties': [],
                            'frame_indices': [], 'run_indices': []
                        }
                    if sym not in frame_info['species_data']:
                        frame_info['species_data'][sym] = {'forces': [], 'uncertainties': [] }

                    aggregated['per_species_stats'][sym]['forces'].append(force)
                    aggregated['per_species_stats'][sym]['uncertainties'].append(unc)
                    aggregated['per_species_stats'][sym]['frame_indices'].append(frame_idx)
                    aggregated['per_species_stats'][sym]['run_indices'].append(run_idx)

                    frame_info['species_data'][sym]['forces'].append(force)
                    frame_info['species_data'][sym]['uncertainties'].append(unc)

                aggregated['frame_data'].append(frame_info)

            aggregated['run_metadata'].append({
                'run_dir': str(run_dir),
                'n_frames': n_frames,
                'failed_run': is_failed
            })

        except Exception as e:
            logging.warning(f"Skipping {run_dir}: {e}")
            continue

    for key in ['max_force_uncertainty','mean_force_uncertainty','energy_uncertainty',
                'stress_uncertainty','max_forces','mean_forces']:
        if aggregated[key]:
            aggregated[key] = np.array(aggregated[key])

    for sym in aggregated['per_species_stats']:
        for arrkey in ['forces','uncertainties','frame_indices','run_indices']:
            aggregated['per_species_stats'][sym][arrkey] = np.array(aggregated['per_species_stats'][sym][arrkey])

    return aggregated

def plot_aggregated_results(data: AggregatedData,
                            output_dir: Path,
                            n_skip: int = 0) -> None:
    """Create comprehensive plots of aggregated simulation results.

    Parameters
    ----------
    data : AggregatedData
        Dictionary containing aggregated simulation data with the following structure:
        - max_force_uncertainty: List[List[float]] (n_runs, n_frames) Maximum force uncertainty per frame
        - mean_force_uncertainty: List[List[float]] (n_runs, n_frames) Mean force uncertainty per frame
        - energy_uncertainty: List[List[float]] (n_runs, n_frames) Energy uncertainty per frame
        - stress_uncertainty: List[List[float]] (n_runs, n_frames) Stress uncertainty per frame
        - max_forces: List[List[float]] (n_runs, n_frames) Maximum force per frame
        - mean_forces: List[List[float]] (n_runs, n_frames) Mean force per frame
        - per_species_stats: Dict[str, Dict] Statistics per atomic species containing:
            - forces: List[float] Forces for this species
            - uncertainties: List[float] Uncertainties for this species
        - run_metadata: List[Dict] Metadata for each run
    output_dir : Path
        Directory where plots and statistics will be saved
    n_skip : int, optional
        Number of initial frames to skip in analysis, by default 0

    Creates
    -------
    analysis.png : Figure
        4x2 grid of plots showing:
        1. Mean force box plot
        2. Max force box plot
        3. Force uncertainty distributions
        4. Force-uncertainty correlation heatmap
        5. Energy uncertainty with error bands
        6. Stress uncertainty with error bands
        7. Forces by species boxplot
        8. Uncertainties by species boxplot

    time_series.png : Figure
        3x1 grid showing evolution over frames:
        1. Mean/max forces with uncertainty bands
        2. Energy uncertainty with std bands
        3. Stress uncertainty with std bands

    summary_stats.json : JSON file
        Statistical summary of key metrics
    """
    print(f"Starting analysis (skipping first {n_skip} frames)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Converting data arrays...")
    # Convert lists to numpy arrays and apply n_skip
    max_forces_array = np.array(data['max_forces'])[:, n_skip:]
    mean_forces_array = np.array(data['mean_forces'])[:, n_skip:]
    max_force_unc_array = np.array(data['max_force_uncertainty'])[:, n_skip:]
    mean_force_unc_array = np.array(data['mean_force_uncertainty'])[:, n_skip:]
    energy_unc_array = np.array(data['energy_uncertainty'])[:, n_skip:]
    stress_unc_array = np.array(data['stress_uncertainty'])[:, n_skip:]

    print("Creating main analysis plots...")
    fig, axes = plt.subplots(4, 2, figsize=RUN_CONFIG['ANALYSIS']['FIGURE_SIZES']['MAIN'])

    # 1. Mean Force Distribution (Histogram)
    axes[0,0].hist(mean_forces_array.flatten(), bins=100, 
                   color='lightblue', alpha=0.7)
    axes[0,0].set_title('Mean Force Distribution')
    axes[0,0].set_xlabel('Force (eV/Å)')
    axes[0,0].set_ylabel('Count')

    # 2. Force Uncertainty Distribution (Histogram)
    axes[0,1].hist(mean_force_unc_array.flatten(), bins=100,
                   color='lightcoral', alpha=0.7)
    axes[0,1].set_title('Force Uncertainty Distribution')
    axes[0,1].set_xlabel('Uncertainty (eV/Å)')
    axes[0,1].set_ylabel('Count')

    # 3. Force vs Uncertainty Correlation (Heatmap)
    h = axes[1,0].hist2d(mean_forces_array.flatten(), 
                        mean_force_unc_array.flatten(), 
                        bins=50, cmap='viridis')
    fig.colorbar(h[3], ax=axes[1,0], label='Count')
    axes[1,0].set_xlabel('Force (eV/Å)')
    axes[1,0].set_ylabel('Uncertainty (eV/Å)')
    axes[1,0].set_title('Force vs Uncertainty Correlation')

    # 4. Force Uncertainty Evolution
    frames = np.arange(mean_force_unc_array.shape[1]) + n_skip
    mean_unc = np.mean(mean_force_unc_array, axis=0)
    std_unc = np.std(mean_force_unc_array, axis=0)
    axes[1,1].plot(frames, mean_unc, 'b-', label='Mean')
    axes[1,1].fill_between(frames, mean_unc - std_unc, mean_unc + std_unc,
                          color='b', alpha=0.2, label='±1σ')
    axes[1,1].set_title('Force Uncertainty Evolution')
    axes[1,1].set_xlabel('Frame')
    axes[1,1].set_ylabel('Uncertainty (eV/Å)')
    axes[1,1].legend()

    # 5. Energy Uncertainty Evolution
    print("Plotting energy and stress distributions...")
    mean_energy = np.mean(energy_unc_array, axis=0)
    std_energy = np.std(energy_unc_array, axis=0)

    axes[2,0].plot(frames, mean_energy, 'b-', label='Mean')
    axes[2,0].fill_between(frames, 
                          mean_energy - std_energy,
                          mean_energy + std_energy,
                          color='b', alpha=0.2, label='±1σ')
    axes[2,0].set_title('Energy Uncertainty Evolution')
    axes[2,0].set_xlabel('Frame')
    axes[2,0].set_ylabel('Energy Uncertainty (eV)')
    axes[2,0].legend()

    # 6. Stress Uncertainty Evolution
    mean_stress = np.mean(stress_unc_array, axis=0)
    std_stress = np.std(stress_unc_array, axis=0)

    axes[2,1].plot(frames, mean_stress, 'r-', label='Mean')
    axes[2,1].fill_between(frames,
                          mean_stress - std_stress,
                          mean_stress + std_stress,
                          color='r', alpha=0.2, label='±1σ')
    axes[2,1].set_title('Stress Uncertainty Evolution')
    axes[2,1].set_xlabel('Frame')
    axes[2,1].set_ylabel('Stress Uncertainty (GPa)')
    axes[2,1].legend()

    # 7-8. Species Analysis
    print("Analyzing per-species statistics...")
    if data['per_species_stats']:
        species = sorted(data['per_species_stats'].keys())
        force_data = []
        unc_data = []

        for s in species:
            # Filter data based on frame indices
            frame_mask = data['per_species_stats'][s]['frame_indices'] >= n_skip
            forces = data['per_species_stats'][s]['forces'][frame_mask]
            uncs = data['per_species_stats'][s]['uncertainties'][frame_mask]
            force_data.append(forces)
            unc_data.append(uncs)

        # Forces plot 
        sns.violinplot(data=force_data, ax=axes[3,0], color='#2EC4B6', inner='box', cut=0)
        axes[3,0].set_xticks(range(len(species)))
        axes[3,0].set_xticklabels(species, rotation=45, ha='right')
        axes[3,0].set_title('Forces by Species', pad=15)
        axes[3,0].set_ylabel('Force (eV/Å)')

        # Uncertainties plot
        sns.violinplot(data=unc_data, ax=axes[3,1], color='#FF9F9F', inner='box', cut=0)
        axes[3,1].set_xticks(range(len(species)))
        axes[3,1].set_xticklabels(species, rotation=45, ha='right')
        axes[3,1].set_title('Uncertainties by Species', pad=15)
        axes[3,1].set_ylabel('Uncertainty (eV/Å)')

        # Style adjustments
        for ax in [axes[3,0], axes[3,1]]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    print("Saving analysis plots...")
    plt.savefig(output_dir / 'analysis.png', dpi=RUN_CONFIG['ANALYSIS']['DPI'], bbox_inches='tight')
    plt.close()

    # Time series plots
    print("Creating time series plots...")
    fig, axes = plt.subplots(3, 1, figsize=RUN_CONFIG['ANALYSIS']['FIGURE_SIZES']['TIME_SERIES'])

    # Forces
    mean_force = np.mean(mean_forces_array, axis=0)
    max_force = np.mean(max_forces_array, axis=0)
    std_mean_force = np.std(mean_forces_array, axis=0)
    std_max_force = np.std(max_forces_array, axis=0)

    axes[0].plot(frames, mean_force, 'b-', label='Mean Force')
    axes[0].fill_between(frames, mean_force - std_mean_force, 
                        mean_force + std_mean_force, color='b', alpha=0.2)
    axes[0].plot(frames, max_force, 'r-', label='Max Force')
    axes[0].fill_between(frames, max_force - std_max_force, 
                        max_force + std_max_force, color='r', alpha=0.2)
    axes[0].set_ylabel('Force (eV/Å)')
    axes[0].legend()

    # Energy
    axes[1].plot(frames, mean_energy, 'b-', label='Mean Energy')
    axes[1].fill_between(frames, mean_energy - std_energy,
                        mean_energy + std_energy, color='gray', alpha=0.2)
    axes[1].set_ylabel('Energy (eV)')
    axes[1].legend()

    # Stress
    axes[2].plot(frames, mean_stress, 'b-', label='Mean Stress')
    axes[2].fill_between(frames, mean_stress - std_stress,
                        mean_stress + std_stress, color='gray', alpha=0.2)
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Stress (GPa)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'time_series.png', dpi=RUN_CONFIG['ANALYSIS']['DPI'], bbox_inches='tight')
    plt.close()

    print("Calculating and saving summary statistics...")
    summary_stats = {
        'Mean Force': float(np.mean(mean_forces_array)),
        'Max Force': float(np.mean(max_forces_array)),
        'Mean Force Uncertainty': float(np.mean(mean_force_unc_array)),
        'Max Force Uncertainty': float(np.mean(max_force_unc_array)),
        'Mean Energy Uncertainty': float(np.mean(energy_unc_array)),
        'Mean Stress Uncertainty': float(np.mean(stress_unc_array)),
        'Number of Runs': len(data['run_metadata']),
        'Frames per Run': max_forces_array.shape[1],
        'Skipped Frames': n_skip,
        'Total Original Frames': max_forces_array.shape[1] + n_skip
    }

    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=4)

    print("Analysis complete!")

def identify_worst_frames(data: AggregatedData,
                        criteria: List[CriteriaDict],
                        n_skip: int,
                        within_criterion_window: int = 5,
                        global_window: int = 2) -> Dict[str, List[FrameTuple]]:

    """
    Identify worst frames with two types of exclusion windows:

    1. within_criterion_window: Frames to exclude around a selected frame in the same criterion
    2. global_window: Criterion-agnostic window - once a frame is selected, nearby frames 
       are globally excluded regardless of criteria

    
    Parameters
    ----------
    data : AggregatedData
        Aggregated simulation data
    criteria : List[CriteriaDict]
        List of criteria dictionaries
    n_skip : int
        Initial frames to skip
    within_criterion_window : int, optional
        Window for same criterion, by default 5
    global_window : int, optional
        Global exclusion window, by default 2
        
    Returns
    -------
    Dict[str, List[FrameTuple]]
        Dictionary mapping criteria to worst frames
        
    """

    from collections import defaultdict

    worst_frames = {}
    globally_excluded = set()
    global_frame_pool = set()

    exclude_failed = RUN_CONFIG['FAILURE_HANDLING']['EXCLUDE_FAILED_RUNS_FROM_WORST_FRAMES']

    # Build global pool
    for f in data['frame_data']:
        if f['frame_idx'] < n_skip:
            continue
        if exclude_failed and f['failed_run']:
            continue
        global_frame_pool.add((f['run_idx'], f['frame_idx']))

    def exclude_global_window(run_idx: int, frame_idx: int):
        for offset in range(-global_window, global_window + 1):
            globally_excluded.add((run_idx, frame_idx + offset))
            if (run_idx, frame_idx + offset) in global_frame_pool:
                global_frame_pool.remove((run_idx, frame_idx + offset))

    for criterion in criteria:
        validate_criterion(criterion)
        metric = criterion['metric']
        stat_type = criterion['stat_type']
        measure = criterion.get('measure', 'value')
        n_frames = criterion['n_frames']
        frame_range = criterion.get('frame_range', (n_skip, None))
        species = criterion.get('species', None)
        local_window = criterion.get('within_window', within_criterion_window)

        desc = f"{stat_type}_{metric}_{measure}"
        if species:
            desc += f"_{species}"

        # Local pool
        criterion_pool = set(global_frame_pool)
        heap = []

        for f in data['frame_data']:
            run_idx = f['run_idx']
            frame_idx = f['frame_idx']
            if (run_idx, frame_idx) not in criterion_pool: 
                continue
            if (run_idx, frame_idx) in globally_excluded:
                continue
            if frame_idx < frame_range[0] or (frame_range[1] and frame_idx >= frame_range[1]):
                continue

            if species:
                if species not in f['species_data']:
                    continue
                values = (f['species_data'][species]['forces'] if measure=='value'
                          else f['species_data'][species]['uncertainties'])
                val = np.max(values) if stat_type=='max' else np.mean(values)
            else:
                valname = f'{stat_type}_forces' if measure=='value' else f'{stat_type}_force_uncertainty'
                val = f['global_stats'].get(valname, 0.0)

            heapq.heappush(heap, (-val, run_idx, frame_idx))

        selected = []
        while heap and len(selected) < n_frames:
            neg_val, run_idx, frame_idx = heapq.heappop(heap)
            val = -neg_val
            if ((run_idx, frame_idx) not in criterion_pool 
                or (run_idx, frame_idx) in globally_excluded):
                continue

            selected.append((run_idx, frame_idx, val))
            for offset in range(-local_window, local_window + 1):
                if (run_idx, frame_idx + offset) in criterion_pool:
                    criterion_pool.remove((run_idx, frame_idx + offset))
            exclude_global_window(run_idx, frame_idx)

        worst_frames[desc] = selected

    return worst_frames

def save_worst_frames(data: AggregatedData,
                     worst_frames: Dict[str, List[FrameTuple]],
                     output_dir: Path,
                     save_xyz: bool = True) -> None:
    """Save worst frame analysis results.

    Parameters
    ----------
    data : AggregatedData
        Original simulation data
    worst_frames : Dict[str, List[FrameTuple]]
        Worst frames by criterion
    output_dir : Path
        Output directory for results
    save_xyz : bool, optional
        Whether to save frames to XYZ file (default: True)
    """
    worst_runs_dir = output_dir / 'worst_runs'
    worst_runs_dir.mkdir(parents=True, exist_ok=True)

    # Save summary to JSON
    summary = {}
    for criterion, frames in worst_frames.items():
        parts = criterion.split('_')
        stat_type = parts[0]
        metric = parts[1]
        measure = parts[2]
        species = parts[3] if len(parts) > 3 else None

        summary[criterion] = [
            {
                'run_idx': int(run_idx),
                'frame_idx': int(frame_idx),
                'value': float(value),
                'run_dir': str(data['run_metadata'][run_idx]['run_dir']),
                'stat_type': stat_type,
                'metric': metric,
                'measure': measure,
                'species': species
            }
            for run_idx, frame_idx, value in frames
        ]

    json_path = worst_runs_dir / RUN_CONFIG['WORST_FRAMES']['JSON_FILE']
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)

    # Save XYZ file if enabled
    if save_xyz and RUN_CONFIG['WORST_FRAMES']['XYZ_FILE']:
        try:
            frames = collect_frames_from_json(json_path)
            if frames:
                xyz_path = worst_runs_dir / RUN_CONFIG['WORST_FRAMES']['XYZ_FILE']
                write(xyz_path, frames)
                logging.info(f"Saved worst frames to XYZ file: {xyz_path}")
        except Exception as e:
            logging.error(f"Failed to save XYZ file: {str(e)}")

    # Create plots for each criterion
    for criterion, frames in worst_frames.items():
        # Parse criterion string to get components
        parts = criterion.split('_')
        stat_type = parts[0]
        metric = parts[1]
        measure = parts[2]
        species = parts[3] if len(parts) > 3 else None

        fig, ax = plt.subplots(figsize=RUN_CONFIG['ANALYSIS']['FIGURE_SIZES']['WORST_FRAMES'])
        values = [value for _, _, value in frames]
        indices = range(len(values))

        ax.bar(indices, values)

        # Create more informative title and labels
        title_parts = [
            f"{stat_type.capitalize()} {metric}",
            f"({measure})" if measure == "uncertainty" else "",
            f"for {species}" if species else ""
        ]
        title = " ".join(filter(None, title_parts))

        ax.set_title(f'Worst Frames - {title}')
        ax.set_xlabel('Frame Index (in selection)')

        # More informative y-label based on metric and measure
        ylabel_parts = [
            f"{stat_type.capitalize()} {metric}",
            f"{measure}" if measure == "uncertainty" else "",
            "(eV/Å)" if metric == "force" else "(eV)" if metric == "energy" else ""
        ]
        ylabel = " ".join(filter(None, ylabel_parts))
        ax.set_ylabel(ylabel)

        plt.tight_layout()
        plt.savefig(worst_runs_dir / f'{criterion}_worst_frames.png', dpi=RUN_CONFIG['ANALYSIS']['DPI'])
        plt.close()


def main():
    base_dir = RUN_CONFIG['PATHS']['BASE_DIR']
    output_dir = RUN_CONFIG['PATHS']['OUTPUT_DIR'] or base_dir / 'analysis'
    n_skip = RUN_CONFIG['ANALYSIS']['N_SKIP_FRAMES']

    if not base_dir.exists():
        raise ValueError(f"Directory {base_dir} does not exist")

    logger = setup_logging(output_dir)

    try:
        logger.info(f"Starting analysis (skipping first {n_skip} frames)")
        data = aggregate_runs(base_dir)
        plots_dir = output_dir / 'plots'
        plot_aggregated_results(data, plots_dir, n_skip)

        # Check if worst frames analysis is enabled
        if RUN_CONFIG['WORST_FRAMES']['ENABLED']:
            logger.info("Starting worst frames analysis...")
            worst_frames = identify_worst_frames(
                data,
                RUN_CONFIG['WORST_FRAMES']['CRITERIA'],
                n_skip,
                within_criterion_window=RUN_CONFIG['WORST_FRAMES']['WINDOWS']['WITHIN_CRITERION'],
                global_window=RUN_CONFIG['WORST_FRAMES']['WINDOWS']['GLOBAL']
            )

            save_worst_frames(
                data,
                worst_frames,
                output_dir,
                save_xyz=bool(RUN_CONFIG['WORST_FRAMES']['XYZ_FILE'])
            )
            logger.info("Worst frames analysis complete")
        else:
            logger.info("Worst frames analysis disabled")

        logger.info(f"Analysis complete. Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

