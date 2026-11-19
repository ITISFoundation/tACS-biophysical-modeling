#!/bin/bash -l
#SBATCH --partition=normal
#SBATCH --nodes=48
#SBATCH --time=24:00:00
#SBATCH --job-name=simHalfVm

# SPDX-License-Identifier: GPL-3.0-or-later
module use /capstor/scratch/cscs/jtharayi/builds/obi-stim/modules
module load neurodamus-models 

CORES_PER_TASK=1
N_NODES=48
CPUS_PER_NODE=128
TOTAL_CPUS=$((N_NODES*CPUS_PER_NODE))
NTASKS=$((TOTAL_CPUS/CORES_PER_TASK))

srun -N $N_NODES -n $NTASKS -c $CORES_PER_TASK --hint=nomultithread special -mpi -python $NEURODAMUS_PYTHON/init.py --configFile=sim_with_stim_new.json
