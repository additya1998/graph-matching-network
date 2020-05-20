#!/bin/bash
#SBATCH -A additya.popli
#SBATCH -n 10
#SBATCH --mem-per-cpu=1024
#SBATCH --time=6:00:00
#SBATCH --mincpus=10
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode56

python main.py 
