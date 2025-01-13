import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem
import argparse
import logging
from datetime import datetime
import json
from typing import Dict, List, Any, Optional, Tuple, Set
import heapq
import sys
from collections import defaultdict
from tqdm import tqdm

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to file and stdout"""
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

def load_run_data(run_dir: Path) -> Dict[str, Any]:
    """Load data from a single run directory"""
    try:
        core_stats = np.load(run_dir / 'core_stats.npz')
        atom_data = np.load(run_dir / 'atom_data.npz', allow_pickle=True)

        return {
            'core_stats': {key: core_stats[key] for key in core_stats.files},
            'atom_data': {key: atom_data[key] for key in atom_data.files}
        }
    except Exception as e:
        raise ValueError(f"Failed to load data from {run_dir}: {str(e)}")

def aggregate_runs(base_dir: Path) -> Dict[str, Any]:
    """Aggregate data from all run directories with per-frame tracking"""
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
        'frame_data': []  # New: track per-frame data
    }

    for run_idx, run_dir in enumerate(tqdm(run_dirs, desc="Processing runs")):
        try:
            data = load_run_data(run_dir)

            # Aggregate core statistics
            for key in data['core_stats']:
                if key in aggregated:
                    aggregated[key].append(data['core_stats'][key])

            # Process per-frame and per-species data
            n_frames = len(data['atom_data']['symbols'])
            for frame_idx in range(n_frames):
                frame_info = {
                    'run_idx': run_idx,
                    'frame_idx': frame_idx,
                    'species_data': {},
                    'global_stats': {}
                }

                # Add global statistics for this frame
                for key in ['max_forces', 'mean_forces', 'max_force_uncertainty', 
                          'mean_force_uncertainty', 'energy_uncertainty', 'stress_uncertainty']:
                    if key in data['core_stats']:
                        frame_info['global_stats'][key] = data['core_stats'][key][frame_idx]

                # Process atomic data for this frame
                sym_list = data['atom_data']['symbols'][frame_idx]
                force_list = data['atom_data']['forces'][frame_idx]
                unc_list = data['atom_data']['uncertainties'][frame_idx]

                for sym, force, unc in zip(sym_list, force_list, unc_list):
                    if sym not in aggregated['per_species_stats']:
                        aggregated['per_species_stats'][sym] = {
                            'forces': [],
                            'uncertainties': [],
                            'frame_indices': [],
                            'run_indices': []
                        }
                    if sym not in frame_info['species_data']:
                        frame_info['species_data'][sym] = {
                            'forces': [],
                            'uncertainties': []
                        }

                    # Add to per-species aggregated data
                    aggregated['per_species_stats'][sym]['forces'].append(force)
                    aggregated['per_species_stats'][sym]['uncertainties'].append(unc)
                    aggregated['per_species_stats'][sym]['frame_indices'].append(frame_idx)
                    aggregated['per_species_stats'][sym]['run_indices'].append(run_idx)

                    # Add to frame-specific data
                    frame_info['species_data'][sym]['forces'].append(force)
                    frame_info['species_data'][sym]['uncertainties'].append(unc)

                aggregated['frame_data'].append(frame_info)

            aggregated['run_metadata'].append({
                'run_dir': str(run_dir),
                'n_frames': n_frames
            })

        except Exception as e:
            logging.warning(f"Skipping {run_dir}: {e}")
            continue

    # Convert lists to numpy arrays
    for key in ['max_force_uncertainty', 'mean_force_uncertainty', 
                'energy_uncertainty', 'stress_uncertainty', 
                'max_forces', 'mean_forces']:
        if aggregated[key]:
            aggregated[key] = np.array(aggregated[key])

    # Convert per-species data to numpy arrays
    for sym in aggregated['per_species_stats']:
        for key in ['forces', 'uncertainties', 'frame_indices', 'run_indices']:
            aggregated['per_species_stats'][sym][key] = np.array(
                aggregated['per_species_stats'][sym][key]
            )

    return aggregated

def plot_aggregated_results(data: Dict[str, Any], output_dir: Path, n_skip: int = 0) -> None:
    """Create comprehensive plots of aggregated simulation results.

    Parameters
    ----------
    data : Dict[str, Any]
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
    fig, axes = plt.subplots(4, 2, figsize=(15, 25))

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

        def fast_violin(ax, data, positions, color, alpha=0.6, width_scale=0.4):
            bp = ax.boxplot(data, positions=positions, patch_artist=True,
                          showfliers=False)
            plt.setp(bp['boxes'], facecolor=color, alpha=alpha)
            plt.setp(bp['medians'], color='black')

            from scipy.stats import gaussian_kde
            for pos, d in zip(positions, data):
                sample = np.random.choice(d, min(1000, len(d)))
                kernel = gaussian_kde(sample)
                y_eval = np.linspace(min(sample), max(sample), 100)
                x_eval = kernel(y_eval) * width_scale
                ax.fill_betweenx(y_eval, pos-x_eval, pos+x_eval, color=color, alpha=alpha/2)

        # Forces plot
        fast_violin(axes[3,0], force_data, range(1, len(species) + 1), '#2EC4B6')
        axes[3,0].set_xticklabels(species, rotation=45, ha='right')
        axes[3,0].set_title('Forces by Species', pad=15)
        axes[3,0].set_ylabel('Force (eV/Å)')

        # Uncertainties plot
        fast_violin(axes[3,1], unc_data, range(1, len(species) + 1), '#FF9F9F', width_scale=0.01)
        axes[3,1].set_xticklabels(species, rotation=45, ha='right')
        axes[3,1].set_title('Uncertainties by Species', pad=15)
        axes[3,1].set_ylabel('Uncertainty (eV/Å)')
        axes[3,1].set_xlim(0.5, len(species) + 0.5)
        axes[3,1].margins(x=0.1)

        # Style adjustments
        for ax in [axes[3,0], axes[3,1]]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    print("Saving analysis plots...")
    plt.savefig(output_dir / 'analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Time series plots
    print("Creating time series plots...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))

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
    plt.savefig(output_dir / 'time_series.png', dpi=300, bbox_inches='tight')
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

def identify_worst_frames(data: Dict[str, Any], criteria: List[Dict], n_skip: int, 
                         within_criterion_window: int = 5,
                         global_window: int = 2) -> Dict[str, List[Tuple]]:
    """
    Identify worst frames with two types of exclusion windows:

    1. within_criterion_window: Frames to exclude around a selected frame in the same criterion
    2. global_window: Criterion-agnostic window - once a frame is selected, nearby frames 
       are globally excluded regardless of criteria

    Example:
    Initial selection in Criterion 1 (within=2):
    Frame indices:    0  1  2  3  4  5  6  7  8  9
    Selected frame:         S
    Within excluded:     x  x  S  x  x
    Global excluded:       x  S  x        (global_window=1)

    Next selection in same criterion:
    Available:        √              S        S
                     (respects both windows)
    """
    worst_frames = {}

    # Track globally excluded frames (criterion-agnostic)
    globally_excluded = set()

    # Initialize global frame pool
    global_frame_pool = set((f['run_idx'], f['frame_idx']) 
                           for f in data['frame_data'] 
                           if f['frame_idx'] >= n_skip)

    def exclude_global_window(run_idx: int, frame_idx: int):
        """Helper to exclude frames within global window."""
        for offset in range(-global_window, global_window + 1):
            nearby_frame = (run_idx, frame_idx + offset)
            globally_excluded.add(nearby_frame)
            if nearby_frame in global_frame_pool:
                global_frame_pool.remove(nearby_frame)

    for criterion in criteria:
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

        # Create criterion-specific frame pool from available global frames
        criterion_frame_pool = global_frame_pool.copy()

        heap = []
        # Collect initial candidates into heap
        for frame_info in data['frame_data']:
            run_idx = frame_info['run_idx']
            frame_idx = frame_info['frame_idx']

            # Skip if frame is globally excluded or out of range
            if ((run_idx, frame_idx) not in criterion_frame_pool or
                (run_idx, frame_idx) in globally_excluded or
                frame_idx < frame_range[0] or 
                (frame_range[1] and frame_idx >= frame_range[1])):
                continue

            if species:
                if species not in frame_info['species_data']:
                    continue
                if measure == 'value':
                    values = frame_info['species_data'][species]['forces']
                else:  # uncertainty
                    values = frame_info['species_data'][species]['uncertainties']
                value = max(values) if stat_type == 'max' else np.mean(values)
            else:
                if measure == 'value':
                    value = frame_info['global_stats'][f'{stat_type}_forces']
                else:  # uncertainty
                    value = frame_info['global_stats'][f'{stat_type}_force_uncertainty']

            heapq.heappush(heap, (-value, run_idx, frame_idx))

        selected = []

        while heap and len(selected) < n_frames:
            neg_value, run_idx, frame_idx = heapq.heappop(heap)
            value = -neg_value

            # Skip if frame was excluded
            if ((run_idx, frame_idx) not in criterion_frame_pool or
                (run_idx, frame_idx) in globally_excluded):
                continue

            # Add to selected frames
            selected.append((run_idx, frame_idx, value))

            # Apply within-criterion window
            for offset in range(-local_window, local_window + 1):
                nearby_frame = (run_idx, frame_idx + offset)
                if nearby_frame in criterion_frame_pool:
                    criterion_frame_pool.remove(nearby_frame)

            # Apply global window
            exclude_global_window(run_idx, frame_idx)

        worst_frames[desc] = selected

    return worst_frames

def save_worst_frames(data: Dict[str, Any], worst_frames: Dict[str, List[Tuple]], 
                     output_dir: Path) -> None:
    """Save information about worst frames to a dedicated directory"""
    worst_runs_dir = output_dir / 'worst_runs'
    worst_runs_dir.mkdir(parents=True, exist_ok=True)

    # Save summary to JSON
    summary = {}
    for criterion, frames in worst_frames.items():
        # Parse criterion string to get components
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

    with open(worst_runs_dir / 'worst_frames.json', 'w') as f:
        json.dump(summary, f, indent=4)

    # Create plots for each criterion
    for criterion, frames in worst_frames.items():
        # Parse criterion string to get components
        parts = criterion.split('_')
        stat_type = parts[0]
        metric = parts[1]
        measure = parts[2]
        species = parts[3] if len(parts) > 3 else None

        fig, ax = plt.subplots(figsize=(10, 6))
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
        plt.savefig(worst_runs_dir / f'{criterion}_worst_frames.png', dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze and plot workflow runs')
    parser.add_argument('--base-dir', type=Path, default=Path('workflow_runs'),
                       help='Base directory containing run folders')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for plots (defaults to base-dir/plots)')
    parser.add_argument('--n-skip', type=int, default=100,
                       help='Number of initial frames to skip')

    args = parser.parse_args()

    if not args.base_dir.exists():
        raise ValueError(f"Directory {args.base_dir} does not exist")

    output_dir = args.output_dir or args.base_dir
    logger = setup_logging(output_dir)

    try:
        logger.info(f"Starting analysis of workflow runs (skipping first {args.n_skip} frames)")
        data = aggregate_runs(args.base_dir)
        plots_dir = output_dir / 'plots'
        plot_aggregated_results(data, plots_dir, args.n_skip)
        logger.info(f"Analysis complete. Plots saved to {plots_dir}")

        # Add criteria for worst frames
        criteria = [
        {
            'metric': 'force',
            'stat_type': 'max',
            'measure': 'value',
            'n_frames': 10,
            'frame_range': (args.n_skip, None),
            'within_window': 50
        },
        {
            'metric': 'force',
            'stat_type': 'mean',
            'measure': 'uncertainty',
            'n_frames': 10,
            'frame_range': (args.n_skip, None),
            'within_window': 50
        },
        {
            'metric': 'force',
            'stat_type': 'max',
            'measure': 'uncertainty',
            'species': 'Zn',
            'n_frames': 40,
            'frame_range': (args.n_skip, None),
            'within_window': 25
        },
        {
            'metric': 'force',
            'stat_type': 'mean',
            'measure': 'uncertainty',
            'species': 'Zn',
            'n_frames': 20,
            'frame_range': (args.n_skip, None),
            'within_window': 50
        }
        ]

        # Identify and save worst frames
        worst_frames = identify_worst_frames(data, 
                                             criteria, 
                                             args.n_skip, 
                                             within_criterion_window=5,
                                             global_window=2)

        save_worst_frames(data, worst_frames, output_dir)

        logger.info(f"Analysis complete. Plots saved to {plots_dir}")
        logger.info(f"Worst frames analysis saved to {output_dir/'worst_runs'}")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()