# Copyright Chris Davies 2025 
# Usage under attached LICENSE

import os
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.atoms import Atoms
from ase.optimize import BFGS
from ase import units
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import matplotlib.pyplot as plt
from ase.stress import voigt_6_to_full_3x3_stress
from tqdm import tqdm
import sys
import logging
from datetime import datetime
from dataclasses import dataclass
import json
from typing import List, Dict, Any, Tuple


##############################
#           CONFIG           #
#                            #
#    Anything in bold is     #
#    something you might     #
#    want to change          #
##############################

BASE_DIR = "workflow_results"           # Directory to run this workflow in
INPUT_XYZ = "generated_structures.xyz"  # Path to input XYZ file with structures to run workflow on

# This section needs to define a list of ase calculators
# The first model in the list will be used for relaxation and NPT simulations
# The rest of the models (and the first) will be used for ensemble analysis

# So for you, this MODELS isnt strictly necessary
# But you will need to define the ASE_CALCULATORS list with your models
from nequip.ase import NequIPCalculator

nequip_model_files = ['deployed_model_0.pth',
                      'deployed_model_1.pth',
                      'deployed_model_2.pth']
ASE_CALCULATORS = [NequIPCalculator.from_deployed_model(model_path=model, 
                                                        device="cuda") for model in nequip_model_files] 


# Simulation Parameters

RELAXATION_STEPS = 50   # Number of relaxation steps before running NPT simulation
RELAX_FORCE_CONVERGENCE = 0.01  # Convergence criterion for relaxation

NPT_STEPS = 4000             # Number of NPT simulation steps
NPT_TIME_STEP = 2 * units.fs # Time step for NPT simulation
NPT_PRESSURE = 1.01325 * units.bar # Pressure for NPT simulation
NPT_SAVE_FRAME_EVERY_N = 10  # Save NPT trajectory every N steps
NPT_TEMPERATURE = 600        # Temperature for NPT simulation

##############################
#      Suppress warnings     #
#      for Allegro but       #
#      might be useful for   #
#      other users           #
##############################

import warnings

def suppress_warnings():
    """Suppress common NequIP and PyTorch warnings."""

    # Suppress specific NequIP warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="nequip")

    # Suppress CUDA/PyTorch warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*PyTorch version.*")

    # Set environment variable to suppress NVFUSER warning
    os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"

    warnings.filterwarnings(
        "ignore",
        message="!! PyTorch version *",
        category=UserWarning,
        module="nequip"
    )
##############################


class TrajectoryAnalysis:
    def __init__(self) -> None:
        """Initialize analysis with paths to committee models.
        Keeps models loaded in memory.

        Parameters: None
        Returns: None
        """
        self.calculators = ASE_CALCULATORS

    def calculate_von_mises_stress(self, stress_tensor: np.ndarray) -> float:
        """Calculate von Mises stress from full 3x3 stress tensor.

        Parameters:
        ----------- 
        stress_tensor : np.ndarray
            6-component Voigt or 3x3 full stress tensor
        
        Returns:
        --------
        von_mises : float
            Von Mises stress scalar value
        """
        if len(stress_tensor) == 6:
            stress_tensor = voigt_6_to_full_3x3_stress(stress_tensor)
        pressure = np.trace(stress_tensor) / 3
        deviatoric = stress_tensor - pressure * np.eye(3)
        von_mises = np.sqrt(1.5 * np.sum(deviatoric * deviatoric))
        return von_mises

    def analyse_trajectory(self, 
                           trajectory: List[Atoms]) -> Dict[str, Any]:
        """Analyse trajectory using committee of models.

        Parameters:
        -----------
        trajectory : List[Atoms]
            List of ASE Atoms objects representing MD trajectory

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - max_force_uncertainty: np.ndarray
            - mean_force_uncertainty: np.ndarray 
            - energy_uncertainty: np.ndarray
            - stress_uncertainty: np.ndarray
            - force_uncertainties: List[np.ndarray]
            - max_forces: np.ndarray
            - mean_forces: np.ndarray
            - force_species_data: List[Tuple[List[str], np.ndarray, np.ndarray]]
        """

        n_frames = len(trajectory)
        results = {
            'max_force_uncertainty': np.zeros(n_frames),
            'mean_force_uncertainty': np.zeros(n_frames),
            'energy_uncertainty': np.zeros(n_frames),
            'stress_uncertainty': np.zeros(n_frames),
            'force_uncertainties': [],
            'max_forces': np.zeros(n_frames),
            'mean_forces': np.zeros(n_frames),
            'all_forces': [],
            'force_species_data': []
        }

        for frame_idx, structure in tqdm(enumerate(trajectory), total=n_frames):
            frame_results = {}

            for i, calc in enumerate(self.calculators):
                structure.calc = calc
                frame_results[i] = {
                    'forces': structure.get_forces(),
                    'stress': structure.get_stress(),
                    'energy': structure.get_potential_energy()
                }

            force_std = np.std([frame_results[i]['forces'] for i in range(len(self.calculators))], axis=0)
            force_uncertainty = np.linalg.norm(force_std, axis=1)
            energy_values = [frame_results[i]['energy'] for i in range(len(self.calculators))]
            stress_values = [frame_results[i]['stress'] for i in range(len(self.calculators))]
            von_mises_values = [self.calculate_von_mises_stress(stress) for stress in stress_values]

            results['max_force_uncertainty'][frame_idx] = np.max(force_uncertainty)
            results['mean_force_uncertainty'][frame_idx] = np.mean(force_uncertainty)
            results['energy_uncertainty'][frame_idx] = np.std(energy_values)
            results['stress_uncertainty'][frame_idx] = np.std(von_mises_values)
            results['force_uncertainties'].append(force_uncertainty)

            mean_forces = np.mean([frame_results[i]['forces'] for i in range(len(self.calculators))], axis=0)
            force_magnitudes = np.linalg.norm(mean_forces, axis=1)

            results['max_forces'][frame_idx] = np.max(force_magnitudes)
            results['mean_forces'][frame_idx] = np.mean(force_magnitudes)
            results['all_forces'].append(force_magnitudes)
            results['force_species_data'].append((structure.get_chemical_symbols(), 
                                                force_magnitudes, force_uncertainty))

        return results

@dataclass
class RunStatus:
    '''
    Dataclass to track the status of a single run
    The idea is that this sits in the WorkflowManager and is updated as the workflow progresses
    Then we can save this to a JSON file at the end of the workflow -> this might help later analysis

    Parameters:
    -----------
    structure_index : int
        Index of the structure being processed
    relaxation_success : bool
        Whether relaxation was successful
    npt_success : bool
        Whether NPT simulation was successful
    analysis_success : bool
        Whether analysis was successful
    error_message : str
        Error message if any step failed
    '''
    structure_index: int
    relaxation_success: bool = False
    npt_success: bool = False
    analysis_success: bool = False
    error_message: str = ""

    def mark_failed(self, msg: str):
        """Mark the run as failed and record the error message."""
        self.error_message = msg
        self.relaxation_success = self.relaxation_success or False
        self.npt_success = self.npt_success or False
        self.analysis_success = self.analysis_success or False

    def is_failed(self) -> bool:
        """Check whether any step has failed."""
        return bool(self.error_message) or not (self.relaxation_success and self.npt_success and self.analysis_success)

    def __str__(self) -> str:
        """Format status as human-readable string."""
        status = []
        if not self.relaxation_success:
            status.append("relaxation failed")
        if not self.npt_success:
            status.append("NPT failed")
        if not self.analysis_success:
            status.append("analysis failed")
            
        return (f"Structure {self.structure_index}: "
                f"{' and '.join(status) if status else 'completed successfully'}"
                f"{f' - {self.error_message}' if self.error_message else ''}")

    def to_dict(self):
        return {
            "structure_index": self.structure_index,
            "relaxation_success": self.relaxation_success,
            "npt_success": self.npt_success,
            "analysis_success": self.analysis_success,
            "error_message": self.error_message
        }

class WorkflowManager:
    def __init__(self, 
                 base_dir: str, 
                 input_xyz: str, 
                 relax_steps: int, 
                 npt_steps: int, 
                 npt_temp: float, 
                 logger: logging.Logger) -> None:
        """Initialize workflow manager.

        Parameters:
        -----------
        base_dir : str
            Base directory for workflow output
        input_xyz : str 
            Path to input XYZ file
        relax_steps : int
            Number of relaxation steps
        npt_steps : int
            Number of NPT simulation steps
        npt_temp : float
            NPT temperature in Kelvin
        logger : logging.Logger
            Logger instance
        """

        self.base_dir = Path(base_dir)
        self.input_xyz = Path(input_xyz)
        self.relax_steps = relax_steps
        self.npt_steps = npt_steps
        self.npt_temp = npt_temp
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger

        try:
            self.structures = read(str(self.input_xyz), index=":")
            self.logger.info(f"Loaded {len(self.structures)} structures from {self.input_xyz}")
        except Exception as e:
            self.logger.error(f"Failed to load structures: {str(e)}")
            raise

        self.analyser = TrajectoryAnalysis()
        self.run_status = []
    
    def run_all(self) -> Tuple[int, int]:
        """Run complete workflow for all structures.

        Returns:
        --------
        Tuple[int, int]
            (number of successful runs, number of failed runs)
        """
        self.logger.info("Starting workflow")
        failed_runs = []

        for idx, structure in enumerate(self.structures):
            status = RunStatus(structure_index=idx)
            self.run_status.append(status)

            run_dir = self.setup_run_directory(idx)
            try:
                relaxed = self.relax_structure(
                    atoms=structure.copy(), 
                    run_dir=run_dir, 
                    status=status, 
                    steps=self.relax_steps
                )
                npt_trajectory = self.run_npt(
                    atoms=relaxed, 
                    run_dir=run_dir, 
                    status=status, 
                    temperature=self.npt_temp, 
                    steps=self.npt_steps
                )

                # Only analyse if we haven't failed yet
                if not status.is_failed():
                    self.analyse_run(run_dir, npt_trajectory, status)

            except Exception as e:
                self.logger.error(f"Failure for structure {idx}: {str(e)}")
                failed_runs.append(idx)
                continue

        # Save global status
        status_data = {
            "total_structures": len(self.structures),
            "failed_runs": failed_runs,
            "run_status": [s.to_dict() for s in self.run_status],
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.base_dir / 'workflow_status.json', 'w') as f:
            json.dump(status_data, f, indent=2)

        successful = len(self.structures) - len(failed_runs)
        return successful, len(failed_runs), self.run_status

    def setup_run_directory(self, idx: int) -> Path:
        """Create run directory for given structure index.

        Parameters:
        -----------
        idx : int
            Structure index

        Returns:
        --------
        Path
            Path to created run directory
        """
        try:
            run_dir = self.base_dir / f"run_{idx:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        except Exception as e:
            self.logger.error(f"Failed to create directory for run {idx}: {str(e)}")
            raise

    def relax_structure(self, atoms: Atoms, 
                        run_dir: Path, 
                        status: RunStatus,
                        steps: int = 200) -> Atoms:
        """Perform structure relaxation.

        Parameters:
        -----------
        atoms : Atoms
            ASE Atoms object to relax
        run_dir : Path
            Directory for output files
        status : RunStatus
            Status tracking object
        steps : int, optional
            Maximum optimization steps

        Returns:
        --------
        Atoms
            Relaxed structure
        """
        self.logger.info(f"Starting relaxation for structure {status.structure_index}")

        try:
            calc = self.analyser.calculators[0]
            atoms.calc=calc

            optimizer = BFGS(atoms, trajectory=str(run_dir / 'relaxation.traj'))
            optimizer.run(fmax=RELAX_FORCE_CONVERGENCE, steps=steps)

            write(str(run_dir / 'relaxed.xyz'), atoms)
            status.relaxation_success = True
            self.logger.info(f"Successfully relaxed structure {status.structure_index}")
            return atoms

        except Exception as e:
            error_msg = f"Relaxation failed for structure {status.structure_index}: {str(e)}"
            self.logger.error(error_msg)
            status.error_message = error_msg
            raise

    def run_npt(self, 
                atoms: Atoms, 
                run_dir: Path, 
                status: RunStatus,
                temperature: float = 500,
                steps: int = 1000) -> List[Atoms]:
        """Run NPT molecular dynamics simulation.

        Parameters:
        -----------
        atoms : Atoms
            Initial structure
        run_dir : Path
            Output directory
        status : RunStatus
            Status tracking object
        temperature : float
            Simulation temperature in Kelvin
        steps : int
            Number of MD steps

        Returns:
        --------
        List[Atoms]
            Trajectory frames
        """

        self.logger.info(f"Starting NPT for structure {status.structure_index}")

        try:
            atoms.calc = self.analyser.calculators[0]

            pressure = NPT_PRESSURE
            ttime = 25 * units.fs
            ptime = 100 * units.fs
            timestep = NPT_TIME_STEP

            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

            dyn = NPT(atoms, 
                    timestep=timestep,
                    temperature_K=temperature,
                    externalstress=pressure,
                    ttime=ttime,
                    pfactor=ptime)

            trajectory_file = str(run_dir / 'npt.extxyz')

            def save_frame():
                write(trajectory_file, atoms, append=True)
            
            pbar = tqdm(total=steps, desc='NPT Simulation')
            
            def update_progress():
                pbar.update(10)  # Update by interval size
            
            dyn.attach(save_frame, interval=NPT_SAVE_FRAME_EVERY_N)
            dyn.attach(update_progress, interval=10)
            
            dyn.run(steps)
            pbar.close

            status.npt_success = True
            self.logger.info(f"Successfully completed NPT for structure {status.structure_index}")
            return read(trajectory_file, index=":")

        except Exception as e:
            error_msg = f"NPT failed for structure {status.structure_index}: {str(e)}"
            self.logger.error(error_msg)
            status.error_message = error_msg
            raise

    def format_failed_runs_report(self) -> str:
        """Generate detailed report of failed runs."""
        failed_runs = [status for status in self.run_status if status.is_failed()]
        
        if not failed_runs:
            return "No failed runs"
            
        report = ["Failed Runs Report", "-----------------"]
        for status in failed_runs:
            report.append(str(status))
            
        return "\n".join(report)

    def analyse_run(self, 
                    run_dir: Path,
                    trajectory: List[Atoms], 
                    status: RunStatus) -> Dict[str, Any]:
        """Analyse a single simulation run.

        Parameters:
        -----------
        run_dir : Path
            Run directory
        trajectory : List[Atoms]
            MD trajectory
        status : RunStatus
            Status tracking object

        Returns:
        --------
        Dict[str, Any]
            Analysis results dictionary
        """
        self.logger.info(f"Starting analysis for structure {status.structure_index}")

        try:
            results = self.analyser.analyse_trajectory(trajectory)
            self.save_run_statistics(results, run_dir)
            self.plot_run_statistics(results, run_dir)
            status.analysis_success = True
            self.logger.info(f"Successfully analysed structure {status.structure_index}")
            return results

        except Exception as e:
            error_msg = f"Analysis failed for structure {status.structure_index}: {str(e)}"
            self.logger.error(error_msg)
            status.error_message = error_msg
            raise
            

    def summarise_workflow(self):
        """Print workflow summary"""
        successful = sum(1 for status in self.run_status 
                        if status.relaxation_success and status.npt_success and status.analysis_success)
        failed = len(self.run_status) - successful

        summary = f"""
        Workflow Summary
        ---------------
        Total structures: {len(self.run_status)}
        Successful runs: {successful}
        Failed runs: {failed}

        Failure breakdown:
        - Relaxation failures: {sum(1 for s in self.run_status if not s.relaxation_success)}
        - NPT failures: {sum(1 for s in self.run_status if s.relaxation_success and not s.npt_success)}
        - Analysis failures: {sum(1 for s in self.run_status if s.npt_success and not s.analysis_success)}
        """

        self.logger.info(summary)
        return summary

    def save_run_statistics(self, results, run_dir):
        """Save essential statistics for a single run in an efficient format"""
        run_dir = Path(run_dir)

        # Save core statistics in compressed numpy format
        core_stats = {
            'max_force_uncertainty': results['max_force_uncertainty'],
            'mean_force_uncertainty': results['mean_force_uncertainty'],
            'energy_uncertainty': results['energy_uncertainty'],
            'stress_uncertainty': results['stress_uncertainty'],
            'max_forces': results['max_forces'],
            'mean_forces': results['mean_forces']
        }
        np.savez_compressed(run_dir / 'core_stats.npz', **core_stats)

        # Save per-atom data in compressed format
        atom_data = {
            'symbols': [data[0] for data in results['force_species_data']],
            'forces': [data[1] for data in results['force_species_data']],
            'uncertainties': [data[2] for data in results['force_species_data']]
        }
        np.savez_compressed(run_dir / 'atom_data.npz', **atom_data)


    def plot_run_statistics(self, results, run_dir):
        """Generate and save plots for a single run"""

        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        from matplotlib import colors

        # 1. Force vs Uncertainty Correlation
        all_forces = np.concatenate(results['all_forces'])
        all_uncertainties = np.concatenate(results['force_uncertainties'])
        
        # Create heatmap  of scatter
        heatmap, xedges, yedges = np.histogram2d(all_forces, all_uncertainties, bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        axes[0,0].imshow(heatmap.T, extent=extent, origin='lower', 
                        norm=colors.LogNorm(), aspect='auto', cmap='viridis')

        axes[0,0].set_xlabel('Force Magnitude (eV/Å)')
        axes[0,0].set_ylabel('Force Uncertainty (eV/Å)')
        axes[0,0].set_title('Force vs Uncertainty Correlation')


        # 2. Time evolution of max/mean forces
        axes[0,1].plot(results['max_forces'], label='Max Force')
        axes[0,1].plot(results['mean_forces'], label='Mean Force')
        axes[0,1].set_xlabel('Frame')
        axes[0,1].set_ylabel('Force (eV/Å)')
        axes[0,1].set_title('Force Evolution')
        axes[0,1].legend()


        # 3. Force distribution
        axes[1,0].hist(all_forces, bins=50, alpha=0.5, label='Forces')
        axes[1,0].set_xlabel('Force Magnitude (eV/Å)')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Force Distribution')

        # 4. Force uncertainty distribution
        axes[1,1].hist(all_uncertainties, bins=50, alpha=0.5, label='Uncertainties')
        axes[1,1].set_xlabel('Force Uncertainty (eV/Å)')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Force Uncertainty Distribution')

        # 5. Species-specific force analysis
        unique_species = set()
        for symbols, _, _ in results['force_species_data']:
            unique_species.update(symbols)

        species_forces = {s: [] for s in unique_species}
        species_uncertainties = {s: [] for s in unique_species}

        for symbols, forces, uncertainties in results['force_species_data']:
            for s, f, u in zip(symbols, forces, uncertainties):
                species_forces[s].append(f)
                species_uncertainties[s].append(u)

        axes[2,0].boxplot([species_forces[s] for s in unique_species], tick_labels=list(unique_species))
        axes[2,0].set_ylabel('Force (eV/Å)')
        axes[2,0].set_title('Forces by Species')
        axes[2,0].tick_params(axis='x', rotation=45)

        axes[2,1].boxplot([species_uncertainties[s] for s in unique_species], tick_labels=list(unique_species))
        axes[2,1].set_ylabel('Uncertainty (eV/Å)')
        axes[2,1].set_title('Uncertainties by Species')
        axes[2,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(run_dir / 'run_statistics.png')
        plt.close()

def setup_logging():
    '''
    Setup logging for the workflow

    Returns:
    --------
        logger : logging.Logger
    '''
    log_dir = Path(BASE_DIR) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'workflow_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


if __name__ == "__main__":
    suppress_warnings()
    logger = setup_logging()

    workflow = WorkflowManager(
        base_dir=BASE_DIR,
        input_xyz=INPUT_XYZ,
        relax_steps=RELAXATION_STEPS,
        npt_steps=NPT_STEPS,
        npt_temp=NPT_TEMPERATURE,
        logger=logger
    )
    
    successful, failed, statuses = workflow.run_all()

    print("\nWorkflow Summary:")
    print(f"Successful runs: {successful}")
    print(f"Failed runs: {failed}")
    
    if failed > 0:
        print("\n" + workflow.format_failed_runs_report())