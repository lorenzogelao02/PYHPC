#!/bin/bash
#BSUB -J cuda_bench
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=shared"
#BSUB -W 15
#BSUB -o Task8/cuda_%J.out
#BSUB -e Task8/cuda_%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=4096MB]"

#Initialize python env
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

#Run script
python -u Task8/8_cuda.py
