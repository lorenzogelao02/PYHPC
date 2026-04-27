#!/bin/bash
#BSUB -J cupy_bench
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=shared"
#BSUB -W 15
#BSUB -o Task9/cupy_%J.out
#BSUB -e Task9/cupy_%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=4096MB]"
#BSUB -R "span[hosts=1]"


#Initialize python env
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#Run script
python -u Task9/9_cupy.py
