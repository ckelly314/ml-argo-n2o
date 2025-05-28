#!/bin/bash
#
#SBATCH --job-name=uncertainties # give a meaningful name.
#SBATCH --error=slurmoutput/uncertainties.err
#SBATCH --out=slurmoutput/uncertainties-1-%j.out # Every job has an out file
#SBATCH -N 1 -n 1 # 36 is the max. number of cores per node on compute
#SBATCH --mem=50G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=colette.kelly@whoi.edu
#SBATCH --time=01:00:00
#

module load miniconda/23.11
. $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate gobgc

python3 flux_uncertainties.py
