#!/bin/bash
#BSUB -J timing
#BSUB -q hpc
#BSUB -W 120
#BSUB -o output/task6/timing_%J.out
#BSUB -e output/task6/timing_%J.err
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4096MB]"
#BSUB -R "select[model==XeonGold6226R]"

#Initialize python env
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

#Run script
python -u python/Task6/6_dynamic_scheduling.py 