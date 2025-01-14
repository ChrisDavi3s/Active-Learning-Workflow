'''
# generate_vasp_single_points.py

## Purpose
This script generates VASP input files for single point runs from an xyz/extxyz file containing multiple structures.

## Usage
```
python generate_training_runs.py --extxyz_file <path> --incar_file <path> [--output_dir <path>] [--structure_name <prefix>]
python generate_training_runs.py --extxyz_file zinc_validation_set.xyz --incar_file INCAR --output_dir structures

```

## Arguments
- `--extxyz_file` (required): Path to the extended XYZ file containing multiple structures.
- `--incar_file` (required): Path to the INCAR file to be used as a template.
- `--output_dir` (optional): Directory where the structure directories will be created. Default is the current directory.
- `--structure_name` (optional): Prefix for naming each structure directory. Default is "structure_".

## Functionality
1. Reads structures from the provided extended XYZ file.
2. For each structure:
   - Creates a new directory in the output directory.
   - Generates a POSCAR file from the structure.
   - Creates a POTCAR file based on the elements in the structure.
   - Copies and modifies the INCAR file, adjusting the LANGEVIN_GAMMA parameter.
3. Handles element-specific POTCAR selections and Langevin damping parameters.
4. Provides logging information for each generated set of VASP input files.

## Notes
- Requires ASE (Atomic Simulation Environment) and Pymatgen libraries.
- Uses a predefined set of POTCAR files and Langevin damping parameters for various elements.
- Sorts atoms in each structure before generating VASP input files.
'''

import os
import argparse
import logging
import shutil
import os

from typing import List, Dict
from ase.io import read
from ase.build.tools import sort
from pymatgen.io.vasp.inputs import Poscar, Potcar
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_POTCAR_FOLDER = "/work/e05/e05/linc5314/potpaw_PBE.54"

DEFAULT_POTCARS = {
  'H':'H', 'He':'He', 'Li':'Li', 'Be':'Be', 'B':'B', 'C':'C', 'N':'N', 'O':'O', 'F':'F', 'Na':'Na_pv',
  'Al':'Al', 'Si':'Si', 'P':'P', 'S':'S', 'Cl':'Cl', 'K':'K_sv', 'Ca':'Ca_sv', 'Sc':'Sc_sv', 'Ti':'Ti_sv',
  'V':'V_sv', 'Cr':'Cr_pv', 'Mn':'Mn_pv', 'Fe':'Fe', 'Co':'Co', 'Ni':'Ni', 'Cu':'Cu', 'Ga':'Ga_d', 'Ge':'Ge_d',
  'Y':'Y_sv', 'Zr':'Zr_sv', 'Mo':'Mo_sv', 'Ta':'Ta_pv', 'Ag':'Ag', 'La':'La', 'W':'W_sv', 'Br':'Br', 'I':'I', 'Zn':'Zn'
}

class VaspNVTInputGenerator:
  def __init__(self, 
               extxyz_file: str, 
               incar_path: str, 
               output_dir: str,
               structure_name: str):
      self.extxyz_file = extxyz_file
      self.incar_path = incar_path
      self.output_dir = output_dir
      self.structure_name = structure_name + "{}"

  def read_poscar_elements(self, poscar_path: str) -> List[str]:
      with open(poscar_path, 'r') as fin:
          for _ in range(5):
              fin.readline()
          elements = fin.readline().strip().split()
          counts = [int(x) for x in fin.readline().strip().split()]
      
      all_elements = []
      for element, count in zip(elements, counts):
          all_elements.extend([element] * count)
      
      return all_elements, elements
  
  def generate_potcar(self, poscar_path: str, directory: str) -> None:
    _, elements = self.read_poscar_elements(poscar_path)
    potcar_path = os.path.join(directory, "POTCAR")
    try:
        with open(potcar_path, 'w') as fout:
            for element in elements:
                potcar_file = os.path.join(DEFAULT_POTCAR_FOLDER, DEFAULT_POTCARS[element], "POTCAR")
                try:
                    with open(potcar_file, 'r') as fin:
                        fout.write(fin.read())
                except IOError as e:
                    logger.error(f"Error reading POTCAR for element {element}: {e}")
                    raise
    except IOError as e:
        logger.error(f"Error writing POTCAR file: {e}")
        raise

  def generate_files(self) -> None:
      structures = read(self.extxyz_file, index=':')
      
      for i, ase_atoms in enumerate(structures):
          directory = os.path.join(self.output_dir, self.structure_name.format(i))
          os.makedirs(directory, exist_ok=True)
          ase_atoms = sort(ase_atoms, tags = ase_atoms.get_atomic_numbers())
          ase_atoms.wrap()

          structure = AseAtomsAdaptor.get_structure(ase_atoms)

          # Generate POSCAR
          poscar = Poscar(structure)
          poscar_path = os.path.join(directory, "POSCAR")
          poscar.write_file(poscar_path)
          
          # Generate POTCAR
          self.generate_potcar(poscar_path, directory)
          
          # Copy and modify INCAR
          try:
            dest_path = os.path.join(directory, 'INCAR')
            shutil.copy(self.incar_path, dest_path)
          except IOError as e:
            print(f"Error copying INCAR file: {e}")

          logger.info(f"Generated VASP input files for NVT run in {directory}")
          logger.info(f"The above file had formula: {ase_atoms.get_chemical_formula()}")


def main():
    parser = argparse.ArgumentParser(description="Generate VASP input files for NVT runs from extxyz file.")
    parser.add_argument("--extxyz_file", type=str, required=True, help="Path to the extxyz file containing multiple structures.")
    parser.add_argument("--incar_file", type=str, required=True, help="Path to the INCAR file to be used as a template.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory where the structure directories will be created. Default is the current directory.")
    parser.add_argument("--structure_name", type=str, default="structure_", help="String which this code will add  +i to for each structure in the extxyz")

    args = parser.parse_args()
    
    generator = VaspNVTInputGenerator(args.extxyz_file, args.incar_file, args.output_dir, args.structure_name)
    generator.generate_files()

if __name__ == "__main__":
    main()