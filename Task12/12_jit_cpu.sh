#!/bin/bash
#BSUB -J all_plans
#BSUB -q hpc
#BSUB -W 120
#BSUB -o output/all_plans_%J.out
#BSUB -e output/all_plans_%J.err
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4096MB]"
#BSUB -R "select[model==XeonGold6226R]"

#Initialize python env
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

#Run script
python -u 12_jit_cpu.py all