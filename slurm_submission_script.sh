#!/bin/bash

#SBATCH --job-name=SCAN_zn_active
#SBATCH --nodes=4
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --array=0-79%40
#SBATCH --time=4:00:00
#SBATCH --account=e05-bulk-isl
#SBATCH --partition=standard
#SBATCH --qos=taskfarm

# load the required modules
module load vasp

# Set environment variables
export OMP_NUM_THREADS=1

cd {**PATH_TO_YOUR_STRUCTURE_DIR**}/structure_$SLURM_ARRAY_TASK_ID

# echo location
echo "Running in $(pwd)"
echo "Running iteration $SLURM_ARRAY_TASK_ID"
echo "$(date) : Starting VASP 6 for AIMD and writing to vasp.out"
srun --distribution=block:block --hint=nomultithread --unbuffered vasp_std > "vasp.out"
echo "$(date) : VASP 6 finished"

echo "$(date) : Copying files"
if [ -f "vasprun.xml" ]; then
  cp "vasprun.xml" "vasprun_structure_${SLURM_ARRAY_TASK_ID}.xml"
  if [ $? -eq 0 ]; then
    echo "$(date) : File copied successfully"
  else
    echo "$(date) : File copy failed"
  fi
else
  echo "$(date) : vasprun.xml not found"
fi
