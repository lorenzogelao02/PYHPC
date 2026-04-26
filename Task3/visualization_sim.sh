#!/bin/bash
#BSUB -J vis
#BSUB -q hpc
#BSUB -W 15
#BSUB -o visOK_%J.out
#BSUB -e visOK_%J.err
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4096MB]"
#BSUB -R "select[model==XeonGold6226R]"

#Initialize python env
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#Run script
python -u Task3/visualization_jacobi.py 10