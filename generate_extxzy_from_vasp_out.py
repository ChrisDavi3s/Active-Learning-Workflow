import os
import re
import argparse
import logging
from typing import List
from ase.io import write
from ase.io.vasp import read_vasp_out

# Usage: python generate_extxyz.py --dir ./runs --output single_points.extxyz --prefix structure_

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def natural_sort_key(s):
    """Sort strings containing numbers naturally"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class StructureExtractor:
    def __init__(self, base_dir: str, structure_prefix: str, output_file: str):
        self.base_dir = base_dir
        self.structure_prefix = structure_prefix
        self.output_file = output_file

    def get_sorted_directories(self) -> List[str]:
        """Get numerically sorted list of structure directories"""
        dirs = []
        for d in os.listdir(self.base_dir):
            if d.startswith(self.structure_prefix):
                full_path = os.path.join(self.base_dir, d)
                if os.path.isdir(full_path):
                    dirs.append(d)
        return sorted(dirs, key=natural_sort_key)

    def extract_structures(self) -> None:
        """Extract final structures from OUTCAR files and combine into one XYZ"""
        dirs = self.get_sorted_directories()
        structures = []

        for dir_name in dirs:
            outcar_path = os.path.join(self.base_dir, dir_name, "OUTCAR")
            if os.path.exists(outcar_path):
                try:
                    atoms = read_vasp_out(outcar_path, index=-1)
                    structures.append(atoms)
                    logger.info(f"Successfully read structure from {dir_name}/OUTCAR")
                except Exception as e:
                    logger.error(f"Failed to read {outcar_path}: {e}")

        if structures:
            write(self.output_file, structures)
            logger.info(f"Written {len(structures)} structures to {self.output_file}")
        else:
            logger.error("No structures found to write")

def main():
    parser = argparse.ArgumentParser(description="Extract and combine structures from VASP OUTCAR files")
    parser.add_argument("--dir", type=str, required=True,
                        help="Base directory containing structure folders")
    parser.add_argument("--prefix", type=str, default="structure_",
                        help="Prefix for structure directories (default: structure_)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output XYZ file path")

    args = parser.parse_args()

    extractor = StructureExtractor(
        base_dir=args.dir,
        structure_prefix=args.prefix,
        output_file=args.output
    )
    
    extractor.extract_structures()

if __name__ == "__main__":
    main()