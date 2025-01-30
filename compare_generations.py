#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare multiple workflows, each containing a 'workflow_results' folder with run_* directories
(e.g. run_0000, run_0001, etc.). Aggregates data and creates comprehensive plots showing
distribution changes, per-species force/uncertainty differences, and high-level statistics.

Copyright 2025 Chris Davies
Usage under attached LICENSE
"""

import os
import sys
import json
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Union
from datetime import datetime
from scipy.stats import gaussian_kde

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
COMPARE_CONFIG = {
    # List of top-level workflow directories to compare.
    # Each directory should contain a subfolder named "workflow_results"
    # that itself contains run_* directories (e.g. run_0000, run_0001, etc.).
    "WORKFLOW_FOLDERS": [
        "run_01",
        "run_01_validation",
        "run_02"
    ],

    "RESULTS_SUBFOLDER": "workflow_results",  # Where run_* directories live
    "OUTPUT_DIR": "compare_analysis",         # Master output directory for the comparison
    "N_SKIP_FRAMES": 0,                       # Number of frames to skip
    "DPI": 300,                               # Figure DPI
    "FIGURE_SIZE": (20, 16),                  # Size of the main comparison figure
    "SPECIES_OF_INTEREST": "Zn"               # Target species for detailed analysis
}

# For reference, define the data structure that aggregator returns
AggregatedData = Dict[str, Any]
PathLike = Union[str, Path]

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def load_run_data(run_dir: PathLike) -> Dict[str, Any]:
    """
    Load simulation data from the given run directory.
    Expects:
      run_dir/core_stats.npz
      run_dir/atom_data.npz
    Returns a dictionary of parsed data.
    """
    run_dir = Path(run_dir)
    core_path = run_dir / "core_stats.npz"
    atom_path = run_dir / "atom_data.npz"

    if not core_path.exists() or not atom_path.exists():
        raise FileNotFoundError(f"Missing data files in {run_dir}")

    return {
        "core_stats": dict(np.load(core_path)),
        "atom_data": dict(np.load(atom_path, allow_pickle=True))
    }

def aggregate_runs(base_dir: Path) -> AggregatedData:
    """
    Aggregate data from run_* directories within 'base_dir'.
    Returns a dictionary with shape:
       {
         'max_forces': List[np.array],
         'mean_forces': List[np.array],
         'max_force_uncertainty': List[np.array],
         'mean_force_uncertainty': List[np.array],
         'energy_uncertainty': List[np.array],
         'stress_uncertainty': List[np.array],
         'per_species_stats': {species: {...}},
         'run_metadata': [...],
         'frame_data': [...]
       }
    """
    run_dirs = sorted(base_dir.glob("run_*"))
    aggregated = {
        'max_forces': [],
        'mean_forces': [],
        'max_force_uncertainty': [],
        'mean_force_uncertainty': [],
        'energy_uncertainty': [],
        'stress_uncertainty': [],
        'per_species_stats': {},
        'run_metadata': [],
        'frame_data': []
    }

    for run_idx, run_path in enumerate(run_dirs):
        try:
            data = load_run_data(run_path)
            n_frames = len(data['atom_data']['symbols'])

            # Append global stats arrays
            for arr_key in ['max_forces', 'mean_forces',
                            'max_force_uncertainty', 'mean_force_uncertainty',
                            'energy_uncertainty', 'stress_uncertainty']:
                if arr_key in data['core_stats']:
                    aggregated[arr_key].append(data['core_stats'][arr_key])

            # Create frame-wise data structure
            for frame_idx in range(n_frames):
                frame_info = {
                    'run_idx': run_idx,
                    'frame_idx': frame_idx,
                    'global_stats': {},
                    'species_data': {}
                }
                # Fill in global stats for this frame
                for arr_key in ['max_forces', 'mean_forces',
                                'max_force_uncertainty', 'mean_force_uncertainty',
                                'energy_uncertainty', 'stress_uncertainty']:
                    if arr_key in data['core_stats']:
                        frame_info['global_stats'][arr_key] = data['core_stats'][arr_key][frame_idx]

                # Per-atom info
                symbols = data['atom_data']['symbols'][frame_idx]
                forces = data['atom_data']['forces'][frame_idx]
                uncs = data['atom_data']['uncertainties'][frame_idx]

                for sym, fval, uval in zip(symbols, forces, uncs):
                    if sym not in aggregated['per_species_stats']:
                        aggregated['per_species_stats'][sym] = {
                            'forces': [],
                            'uncertainties': [],
                            'frame_indices': [],
                            'run_indices': []
                        }
                    aggregated['per_species_stats'][sym]['forces'].append(fval)
                    aggregated['per_species_stats'][sym]['uncertainties'].append(uval)
                    aggregated['per_species_stats'][sym]['frame_indices'].append(frame_idx)
                    aggregated['per_species_stats'][sym]['run_indices'].append(run_idx)

                    # Also store in frame_info
                    if sym not in frame_info['species_data']:
                        frame_info['species_data'][sym] = {
                            'forces': [],
                            'uncertainties': []
                        }
                    frame_info['species_data'][sym]['forces'].append(fval)
                    frame_info['species_data'][sym]['uncertainties'].append(uval)

                aggregated['frame_data'].append(frame_info)

            aggregated['run_metadata'].append({
                'run_dir': str(run_path),
                'n_frames': n_frames
            })

        except Exception as e:
            logging.warning(f"Skipping {run_path} due to error: {e}")
            continue

    # Convert run-wise lists into a NumPy array for each key
    for arr_key in ['max_forces', 'mean_forces',
                    'max_force_uncertainty', 'mean_force_uncertainty',
                    'energy_uncertainty', 'stress_uncertainty']:
        if aggregated[arr_key]:
            aggregated[arr_key] = np.array(aggregated[arr_key])  # shape: (n_runs, variable_length)

    # Convert species-level lists to arrays
    for sym, subdict in aggregated['per_species_stats'].items():
        for k in ['forces', 'uncertainties', 'frame_indices', 'run_indices']:
            subdict[k] = np.array(subdict[k])

    return aggregated

# ------------------------------------------------------------------------------
# Main Comparison and Plotting
# ------------------------------------------------------------------------------
def gather_workflow_data(
    workflow_folder: PathLike,
    results_subfolder: str
) -> AggregatedData:
    """
    Gathers aggregated data from a single workflow folder.
    The folder structure is:
       workflow_folder/
         └── results_subfolder/  (e.g. "workflow_results")
             └── run_0000/
                 ├── core_stats.npz
                 └── atom_data.npz
             └── run_0001/
                 ...
    """
    workflow_folder = Path(workflow_folder)
    results_path = workflow_folder / results_subfolder
    if not results_path.exists():
        raise FileNotFoundError(f"Expected subfolder '{results_subfolder}' not found in {workflow_folder}")
    return aggregate_runs(results_path)

def compare_multiple_workflows(
    workflow_folders: List[str],
    results_subfolder: str,
    output_dir: PathLike,
    n_skip: int = 0
) -> None:
    """
    Compare aggregated data across multiple workflow folders, each containing
    a 'results_subfolder' with run_* directories. Create multi-faceted plots that
    illustrate distribution changes, per-species force averages, and uncertainties.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    log_file = output_dir / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting workflow comparison...")

    # --- 1) Load data for each workflow ---
    all_aggregated = []
    for folder in workflow_folders:
        wpath = Path(folder)
        if not wpath.exists():
            logger.warning(f"Workflow folder not found: {folder}")
            continue
        logger.info(f"Gathering aggregated data from {folder} ...")
        try:
            data = gather_workflow_data(wpath, results_subfolder)
            all_aggregated.append((folder, data))
        except FileNotFoundError as e:
            logger.warning(f"Skipping {folder}, folder or data missing: {e}")
        except Exception as e:
            logger.warning(f"Error loading data from {folder}: {e}")

    if not all_aggregated:
        logger.error("No valid workflow data found. Exiting.")
        return

    # --- 2) Prepare figure for multi-panel distribution/time-series plots ---
    fig, axes = plt.subplots(3, 2, figsize=COMPARE_CONFIG["FIGURE_SIZE"])
    axes = axes.flatten()

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(all_aggregated))]

    def trimmed_data(arr_2d):
        flattened = []
        for row in arr_2d:
            if row.shape[0] > n_skip:
                flattened.extend(row[n_skip:])
        return np.array(flattened)

    # --- Plot 0: Distribution of Mean Forces ---
    for idx, (folder, data) in enumerate(all_aggregated):
        if isinstance(data['mean_forces'], np.ndarray) and data['mean_forces'].size > 0:
            arr = trimmed_data(data['mean_forces'])
            if arr.size > 0:
                axes[0].hist(arr, bins=80, alpha=0.3, color=colors[idx], label=folder)
    axes[0].set_title("Distribution of Mean Forces (All Runs, Frame ≥ n_skip)")
    axes[0].set_xlabel("Force (eV/Å)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # --- Plot 1: Distribution of Mean Force Uncertainties ---
    for idx, (folder, data) in enumerate(all_aggregated):
        if isinstance(data['mean_force_uncertainty'], np.ndarray) and data['mean_force_uncertainty'].size > 0:
            arr = trimmed_data(data['mean_force_uncertainty'])
            if arr.size > 0:
                axes[1].hist(arr, bins=80, alpha=0.3, color=colors[idx], label=folder)
    axes[1].set_title("Distribution of Mean Force Uncertainties (All Runs, Frame ≥ n_skip)")
    axes[1].set_xlabel("Uncertainty (eV/Å)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    # --- Plot 2: Time-series of Mean Force ---
    for idx, (folder, data) in enumerate(all_aggregated):
        if isinstance(data['mean_forces'], np.ndarray) and data['mean_forces'].size > 0:
            valid_arrays = [row[n_skip:] for row in data['mean_forces'] if len(row) > n_skip]
            if valid_arrays:
                min_len = min(len(arr) for arr in valid_arrays)
                stacked = np.array([arr[:min_len] for arr in valid_arrays])
                y = stacked.mean(axis=0)
                x = np.arange(min_len) + n_skip
                axes[2].plot(x, y, label=folder, color=colors[idx])
    axes[2].set_title("Time-Series of Mean Force (Averaged over runs)")
    axes[2].set_xlabel("Frame Index (≥ n_skip)")
    axes[2].set_ylabel("Mean Force (eV/Å)")
    axes[2].legend()

    # --- Plot 3: Time-series of Mean Force Uncertainty ---
    for idx, (folder, data) in enumerate(all_aggregated):
        if isinstance(data['mean_force_uncertainty'], np.ndarray) and data['mean_force_uncertainty'].size > 0:
            valid_arrays = [row[n_skip:] for row in data['mean_force_uncertainty'] if len(row) > n_skip]
            if valid_arrays:
                min_len = min(len(arr) for arr in valid_arrays)
                stacked = np.array([arr[:min_len] for arr in valid_arrays])
                y = stacked.mean(axis=0)
                x = np.arange(min_len) + n_skip
                axes[3].plot(x, y, label=folder, color=colors[idx])
    axes[3].set_title("Time-Series of Mean Force Uncertainty (Averaged)")
    axes[3].set_xlabel("Frame Index (≥ n_skip)")
    axes[3].set_ylabel("Uncertainty (eV/Å)")
    axes[3].legend()

    # --- Plot 4 & 5: Per-species violin plots ---
    species = COMPARE_CONFIG["SPECIES_OF_INTEREST"]
    labels = [folder for folder, _ in all_aggregated]
    palette = {folder: (*colors[i][:-1], 0.3) for i, folder in enumerate(labels)}  # alpha=0.3
    has_species = {folder: False for folder in labels}

    # Prepare DataFrames
    forces_data, unc_data = [], []
    for folder, data in all_aggregated:
        if species in data['per_species_stats']:
            has_species[folder] = True
            sp_data = data['per_species_stats'][species]
            mask = sp_data['frame_indices'] >= n_skip
            
            # Add forces data
            for val in sp_data['forces'][mask]:
                forces_data.append({'Workflow': folder, 'Value': val, 'Type': 'Force'})
            
            # Add uncertainties data
            for val in sp_data['uncertainties'][mask]:
                unc_data.append({'Workflow': folder, 'Value': val, 'Type': 'Uncertainty'})

    # Combine into single DataFrame
    df = pd.DataFrame(forces_data + unc_data)

    # Create modified labels with species presence info
    new_labels = [f"{label}\n(No {species})" if not has_species[label] else label 
                 for label in labels]

    # Plot Forces
    forces_plot = sns.violinplot(
        x='Workflow', 
        y='Value', 
        data=df[df['Type'] == 'Force'],
        ax=axes[4],
        hue='Workflow',
        palette=palette,
        order=labels,
        cut=0,
        width=0.8,
        legend=False
    )
    axes[4].set_xticks(range(len(new_labels)))
    axes[4].set_xticklabels(new_labels, rotation=45, ha='right')
    axes[4].set_title(f'{species} Forces Distribution')

    # Plot Uncertainties
    uncertainty_plot = sns.violinplot(
        x='Workflow', 
        y='Value', 
        data=df[df['Type'] == 'Uncertainty'],
        ax=axes[5],
        hue='Workflow',
        palette=palette,
        order=labels,
        cut=0,
        width=0.8,
        legend=False
    )
    axes[5].set_xticks(range(len(new_labels)))
    axes[5].set_xticklabels(new_labels, rotation=45, ha='right')
    axes[5].set_title(f'{species} Uncertainty Distribution')
    plt.setp(axes[4].collections, alpha=0.4)
    plt.setp(axes[5].collections, alpha=0.4)
    logger.info(f"Comparison plots generated. Saving to {output_dir} as comparison_plot.png")

    # Final saving and cleanup
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_plot.png", dpi=300)
    plt.close()

    # --- 3) Summary stats JSON ---
    summary = {}
    for folder, data in all_aggregated:
        info = {}
        # Example: average mean force, average uncertainty, etc.
        # Just show a couple of stats for demonstration:
        if isinstance(data['mean_forces'], np.ndarray) and data['mean_forces'].size > 0:
            arr = trimmed_data(data['mean_forces'])
            if arr.size > 0:
                info["AvgMeanForce"] = float(np.mean(arr))
        if isinstance(data['mean_force_uncertainty'], np.ndarray) and data['mean_force_uncertainty'].size > 0:
            arr = trimmed_data(data['mean_force_uncertainty'])
            if arr.size > 0:
                info["AvgMeanForceUncertainty"] = float(np.mean(arr))
        summary[folder] = info

    summary_path = output_dir / "compare_summary_stats.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Comparison summary stats saved to {summary_path}")


def main():
    wf_folders = COMPARE_CONFIG["WORKFLOW_FOLDERS"]
    results_subfolder = COMPARE_CONFIG["RESULTS_SUBFOLDER"]
    out_dir = COMPARE_CONFIG["OUTPUT_DIR"]
    n_skip = COMPARE_CONFIG["N_SKIP_FRAMES"]

    compare_multiple_workflows(
        workflow_folders=wf_folders,
        results_subfolder=results_subfolder,
        output_dir=out_dir,
        n_skip=n_skip
    )

if __name__ == "__main__":
    main()
