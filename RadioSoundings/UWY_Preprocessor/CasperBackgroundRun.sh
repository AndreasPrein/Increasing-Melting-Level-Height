#!/bin/bash -l
#SBATCH -J UW-Soundings
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH -A P66770001
#SBATCH -p dav

module load python/2.7.14
ncar_pylib
srun ./HailParametersFromRadioSounding.py
