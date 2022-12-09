#!/bin/bash
#SBATCH --job-name=mmnullsHaufe
#SBATCH --output="MMnulls.log"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-user=sidhant.chopra@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=psych_week

module load miniconda
cd /home/sc2998/project/MM/analysis/scripts/nulls
conda init bash
conda activate MM_env
python MM_Stacking_Nulls.py $ind 10000
