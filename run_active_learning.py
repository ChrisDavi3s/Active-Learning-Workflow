# Copyright Chris Davies 2025 
# Usage under attached LICENSE

import os
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.atoms import Atoms
from ase.optimize import BFGS
from ase.cell import Cell
from ase import units
from ase.md.npt import NPT
from ase.md.langevin import Langevin

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import matplotlib.pyplot as plt
from ase.stress import voigt_6_to_full_3x3_stress
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import sys
import logging
from datetime import datetime
from dataclasses import dataclass
import json
from typing import List, Dict, Any, Tuple, Optional, Union


nequip_model_files = [
    'deployed_model_0.pth',
    'deployed_model_1.pth',
    'deployed_model_2.pth'
]

from nequip.ase import NequIPCalculator
ASE_CALCULATORS = [
    NequIPCalculator.from_deployed_model(model_path=model, device="cpu")
    for model in nequip_model_files
]

WORKFLOW_CONFIG = {
    'PATHS': {
        'BASE_DIR': Path('workflow_results'),
        'INPUT_XYZ': Path('generated_structures.xyz'),
        'LOG_DIR': None  # Will default to BASE_DIR/logs if None
    },
    'RELAXATION': {
        'STEPS': 5,             # Number of relaxation steps.
        'FORCE_CONVERGENCE': 0.01
    },
    'MD': { 
        'ENSEMBLE': 'NPT',  # Can be 'NPT' or 'NVT'
        'STEPS': 20,
        'TIME_STEP': 2 * units.fs,
        'PRESSURE': 1.01325 * units.bar,  # For NPT
        'SAVE_INTERVAL': 10,
        'TEMPERATURE': 600,
        'THERMOSTAT_TIME': 25 * units.fs,  # For NPT
        'BAROSTAT_TIME': 100 * units.fs,   # For NPT
        'FRICTION': 0.002,                 # For NVT
    },
    'ANALYSIS': {
        'DPI': 300,
        'FIGURE_SIZES': {
            'MAIN': (15, 15),
            'SPECIES': (10, 6)
        }
    },
    'CONVERT_UPPER_DIAGONAL_CELL': True,
    'CALCULATORS': ASE_CALCULATORS,
    'RUNTIME': {
        'ANALYZE_FAILED_RUNS': True,    # Analyze failed runs
        'VERBOSE_LOGGING': True,        # Enable verbose logging
        'QUIET': True                  # Suppress all output to terminal
    }
 }

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


# Create a type alias for our objects - this is bad practice but it's useful for the type hints
# (TODO) - use actual OOP to define these classes

PathLike = Union[str, Path]
NDArray = np.ndarray 
AtomsList = List[Atoms]
CalcList = List[Any] 
RunStatus = Dict[str, Any]
WorkflowConfig = Dict[str, Any]


class TrajectoryAnalysis:
    def __init__(self, calculators: CalcList, logger: Optional[logging.Logger] = None) -> None:
        """Initialize analysis with paths to committee models and logger.
        
        Parameters:
        -----------
        logger : logging.Logger, optional
            Logger instance for analysis
        """
        self.calculators = calculators
        self.logger = logger or logging.getLogger(__name__)


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

        try:
            for frame_idx, structure in tqdm(enumerate(trajectory), total=n_frames, desc='Analysing uncertainty with committee'):
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
                results['force_species_data'].append((
                    structure.get_chemical_symbols(),
                    force_magnitudes,
                    force_uncertainty
                ))

        except Exception as e:
            self.logger.error(f"Error in analyse_trajectory: {str(e)}")
            # Return partial results even if some frames weren't processed
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
    md_success : bool
        Whether NPT simulation was successful
    analysis_success : bool
        Whether analysis was successful
    error_message : str
        Error message if any step failed
    '''
    structure_index: int
    relaxation_success: Optional[bool] = None
    md_success: Optional[bool] = None
    analysis_success: Optional[bool] = None
    error_message: str = ""

    def mark_failed(self, msg: str):
        """Mark the run as failed and record the error message."""
        self.error_message = msg
        self.relaxation_success = self.relaxation_success or False
        self.md_success = self.md_success or False
        self.analysis_success = self.analysis_success or False

    def is_failed(self) -> bool:
        """Check whether any step has failed."""
        return bool(self.error_message) or not (self.relaxation_success and self.md_success and self.analysis_success)

    def is_failed_before_analysis(self) -> bool:
        """Check whether relaxation or MD failed."""
        return bool(self.error_message) or not (self.relaxation_success and self.md_success)

    def __str__(self) -> str:
        """Format status as human-readable string."""
        status = []
        if not self.relaxation_success:
            status.append("relaxation failed")
        if not self.md_success:
            status.append("MD failed")
        if not self.analysis_success:
            status.append("analysis failed")
            
        return (f"Structure {self.structure_index}: "
                f"{' and '.join(status) if status else 'completed successfully'}"
                f"{f' - {self.error_message}' if self.error_message else ''}")

    def to_dict(self):
        return {
            "structure_index": self.structure_index,
            "relaxation_success": self.relaxation_success,
            "md_success": self.md_success,
            "analysis_success": self.analysis_success,
            "error_message": self.error_message
        }

class WorkflowManager:
    def __init__(self,
                 config: WorkflowConfig,
                 calculators: CalcList,
                 logger: Optional[logging.Logger] = None) -> None:
        """Initialize workflow manager with configuration and calculators.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        calculators : CalcList
            List of initialized ASE calculators
        logger : Optional[logging.Logger]
            Logger instance
        """
        self.config = config
        self.calculators = calculators
        self.base_dir = Path(config['PATHS']['BASE_DIR'])
        self.input_xyz = Path(config['PATHS']['INPUT_XYZ'])

        # Setup paths
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = logger or self._setup_logging()
        
        # Load structures
        try:
            self.structures = read(str(self.input_xyz), index=":")
            self.logger.info(f"Loaded {len(self.structures)} structures")
        except Exception as e:
            self.logger.error(f"Failed to load structures: {str(e)}")
            raise

        # Initialize analysis
        analysis_logger = self.logger.getChild('analysis')
        self.analyser = TrajectoryAnalysis(self.config['CALCULATORS'], logger=analysis_logger)
        self.run_status: List[RunStatus] = []
    
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
                if self.config['CONVERT_UPPER_DIAGONAL_CELL']:
                    structure = self.make_cell_upper_triangular(structure.copy())

                if self.config['RELAXATION']['STEPS'] > 0:
                    relaxed = self.relax_structure(
                        atoms=structure, 
                        run_dir=run_dir, 
                        status=status, 
                        steps=self.config['RELAXATION']['STEPS']
                    )                
                    
                else:
                    self.logger.info(f"Skipping relaxation for structure {idx} (relax_steps=0)")
                    relaxed = structure
                    status.relaxation_success = True

                md_trajectory = self.run_md(
                    atoms=relaxed,
                    run_dir=run_dir,
                    status=status,
                    temperature=self.config['MD']['TEMPERATURE'],
                    steps=self.config['MD']['STEPS']
                )

                # Only analyse if we haven't failed yet
                if not status.is_failed_before_analysis():
                    self.analyse_run(run_dir, md_trajectory, status)

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
            "relaxation_performed": self.config['RELAXATION']['STEPS'] is not None and self.config['RELAXATION']['STEPS'] > 0
        }
        with open(self.base_dir / 'workflow_status.json', 'w') as f:
            json.dump(status_data, f, indent=2)

        successful = len(self.structures) - len(failed_runs)
        return successful, len(failed_runs), self.run_status

    def make_cell_upper_triangular(self, atoms) -> Atoms:
        """
        Convert a cell matrix to an upper triangular cell using QR decomposition.

        Parameters:

            atoms (Atoms): ASE Atoms object with cell matrix to convert.

        Returns:
            atoms (Atoms): ASE Atoms object with upper triangular cell matrix.
        """
        self.logger.info("Converting cell matrix to upper triangular")
        cellpar = atoms.cell.cellpar()
        atoms.set_cell(cellpar, scale_atoms=True)
        self.logger.info("Successfully converted cell matrix to upper triangular")
        return atoms

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
            optimizer.run(fmax=self.config['RELAXATION']['FORCE_CONVERGENCE'], steps=steps)

            write(str(run_dir / 'relaxed.xyz'), atoms)
            status.relaxation_success = True
            self.logger.info(f"Successfully relaxed structure {status.structure_index}")
            return atoms

        except Exception as e:
            error_msg = f"Relaxation failed for structure {status.structure_index}: {str(e)}"
            self.logger.error(error_msg)
            status.error_message = error_msg
            raise

    def run_md(self, 
           atoms: Atoms, 
           run_dir: Path, 
           status: RunStatus, 
           temperature: float, 
           steps: int) -> List[Atoms]:
        """Run MD simulation (either NPT or NVT ensemble).

        Parameters:
        -----------
        atoms : Atoms
            Initial structure.
        run_dir : Path
            Output directory.
        status : RunStatus
            Status tracking object.
        temperature : float
            Simulation temperature in Kelvin.
        steps : int
            Number of MD steps.

        Returns:
        --------
        List[Atoms]
            Trajectory frames.
        """
        self.logger.info(f"Starting MD simulation for structure {status.structure_index}")
        trajectory_file = str(run_dir / 'md_trajectory.extxyz')
        try:
            atoms.calc = self.analyser.calculators[0]
            md_config = self.config['MD']
            ensemble = md_config['ENSEMBLE']
            timestep = md_config['TIME_STEP']
            save_interval = md_config['SAVE_INTERVAL']

            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

            if ensemble == 'NPT':
                pressure = md_config['PRESSURE']
                ttime = md_config['THERMOSTAT_TIME']
                ptime = md_config['BAROSTAT_TIME']
                dyn = NPT(atoms, timestep=timestep,
                        temperature_K=temperature,
                        externalstress=pressure,
                        ttime=ttime,
                        pfactor=ptime)
            elif ensemble == 'NVT':
                friction = md_config.get('FRICTION', 0.002)
                dyn = Langevin(atoms, timestep=timestep,
                            temperature_K=temperature,
                            friction=friction)
            else:
                raise ValueError(f"Unsupported ensemble type: {ensemble}")

            def save_frame():
                write(trajectory_file, atoms, append=True)
            def update_progress():
                pbar.update(10)

            with logging_redirect_tqdm():
                pbar = tqdm(total=steps, desc=f'{ensemble} Simulation')
                dyn.attach(save_frame, interval=save_interval)
                dyn.attach(update_progress, interval=10)
                dyn.run(steps)
                pbar.close()

            status.md_success = True
            self.logger.info(f"MD ({ensemble}) successful for structure {status.structure_index}")
            return read(trajectory_file, index=":")

        except Exception as e:
            err = f"MD ({ensemble}) failed for structure {status.structure_index}: {str(e)}"
            self.logger.error(err)
            status.error_message = err
            if self.config['RUNTIME'].get('ANALYZE_FAILED_RUNS', False):
                self.logger.info("Attempting partial analysis for failed run.")
                if os.path.exists(trajectory_file):
                    try:
                        partial_frames = read(trajectory_file, index=":")
                        if partial_frames:
                            self.analyse_run(run_dir, partial_frames, status)
                    except Exception as e2:
                        self.logger.error(f"Partial analysis failed: {str(e2)}")
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
            err = f"Analysis failed for structure {status.structure_index}: {str(e)}"
            self.logger.error(err)
            status.analysis_success = False
            status.error_message = err
            # Attempt partial saving if results exist
            partial_results = self.analyser.analyse_trajectory(trajectory)  # runs but only partial if it fails again
            self.save_run_statistics(partial_results, run_dir)
            return partial_results
            
    def summarise_workflow(self):
        """Print workflow summary"""
        successful = sum(1 for status in self.run_status 
                        if status.relaxation_success and status.md_success and status.analysis_success)
        failed = len(self.run_status) - successful

        summary = f"""
        Workflow Summary
        ---------------
        Total structures: {len(self.run_status)}
        Successful runs: {successful}
        Failed runs: {failed}

        Failure breakdown:
        - Relaxation failures: {sum(1 for s in self.run_status if not s.relaxation_success)}
        - MD failures: {sum(1 for s in self.run_status if s.relaxation_success and not s.md_success)}
        - Analysis failures: {sum(1 for s in self.run_status if s.md_success and not s.analysis_success)}
        """

        self.logger.info(summary)
        return summary

    def save_run_statistics(self, results, run_dir):
        """Save essential statistics for a single run"""
        run_dir = Path(run_dir)
        
        try:
            # Ensure directory exists
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save core statistics
            stats_path = run_dir / 'core_stats.npz'
            self.logger.info(f"Saving core statistics to {stats_path}")
            core_stats = {
                'max_force_uncertainty': results['max_force_uncertainty'],
                'mean_force_uncertainty': results['mean_force_uncertainty'],
                'energy_uncertainty': results['energy_uncertainty'],
                'stress_uncertainty': results['stress_uncertainty'],
                'max_forces': results['max_forces'],
                'mean_forces': results['mean_forces']
            }
            np.savez_compressed(stats_path, **core_stats)

            # Save atom data
            atom_path = run_dir / 'atom_data.npz'
            self.logger.info(f"Saving atom data to {atom_path}")
            atom_data = {
                'symbols': [data[0] for data in results['force_species_data']],
                'forces': [data[1] for data in results['force_species_data']],
                'uncertainties': [data[2] for data in results['force_species_data']]
            }
            np.savez_compressed(atom_path, **atom_data)
            self.logger.info("Successfully saved all statistics")
            
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {str(e)}")
            raise


    def plot_run_statistics(self, results, run_dir):
        """Generate and save plots for a single run"""
        run_dir = Path(run_dir)
        plot_path = run_dir / 'run_statistics.png'
        
        try:
            # Create figure and axes
            fig, axes = plt.subplots(3, 2, figsize=self.config['ANALYSIS']['FIGURE_SIZES']['MAIN'])
            from matplotlib import colors

            # Calculate step numbers 
            steps = np.arange(len(results['max_forces'])) * self.config['MD']['SAVE_INTERVAL']


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
            axes[0,1].plot(steps, results['max_forces'], label='Max Force')
            axes[0,1].plot(steps, results['mean_forces'], label='Mean Force')
            axes[0,1].set_xlabel('MD Step')
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
            self.logger.info(f"Saving plot to {plot_path}")
            plt.tight_layout()
            
            # Ensure directory exists
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with high DPI and explicit format
            plt.savefig(plot_path, dpi=self.config['ANALYSIS']['DPI'], format='png', bbox_inches='tight')
            self.logger.info(f"Successfully saved plot to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create/save plot: {str(e)}")
            raise
        finally:
            plt.close('all')  # Ensure figures are closed

def setup_logging(log_dir: PathLike = None, 
                 verbose: bool = True,
                 quiet: bool = False) -> logging.Logger:
    '''
    Setup logging for the workflow with configurable output levels
    
    Parameters:
    -----------
    log_dir : PathLike
        Directory for log files
    verbose : bool
        Enable verbose console output
    quiet : bool
        Suppress console output entirely
        
    Returns:
    --------
    logger : logging.Logger
    '''
    log_dir = Path(log_dir) if log_dir else WORKFLOW_CONFIG['PATHS']['BASE_DIR'] / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'workflow_{timestamp}.log'

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logger and prevent propagation to root logger
    logger = logging.getLogger('workflow')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  
    
    # File handler always logs everything
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Console handler depends on settings
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
        logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    suppress_warnings()

    workflow = WorkflowManager(
        config=WORKFLOW_CONFIG,
        calculators=ASE_CALCULATORS,
        logger=setup_logging(
            WORKFLOW_CONFIG['PATHS']['LOG_DIR'],
            verbose=WORKFLOW_CONFIG['RUNTIME']['VERBOSE_LOGGING'],
            quiet=WORKFLOW_CONFIG['RUNTIME']['QUIET']
        )
    )

    successful, failed, statuses = workflow.run_all()

    print("\nWorkflow Summary:")
    print(f"Successful runs: {successful}")
    print(f"Failed runs: {failed}")

    if failed > 0:
        print("\n" + workflow.format_failed_runs_report())
