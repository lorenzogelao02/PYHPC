#!/bin/bash
#BSUB -J profiling
#BSUB -q hpc
#BSUB -W 15
#BSUB -o output/profiling_%J.out
#BSUB -e output/profiling_%J.err
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4096MB]"
#BSUB -R "select[model==XeonGold6226R]"

#Initialize python env
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#Run script
kernprof -l python/jacobi_profile.py 10
python -m line_profiler -rmt "jacobi_profile.py.lprof"