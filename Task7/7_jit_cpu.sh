#!/bin/bash
#BSUB -J jit_cpu_1
#BSUB -q hpc
#BSUB -W 120
#BSUB -o output/jit_cpu_%J.out
#BSUB -e output/jit_cpu_%J.err
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4096MB]"
#BSUB -R "select[model==XeonGold6226R]"

#Initialize python env
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

#Run script
python -u 7_jit_cpu.py profiling
# python -m cProfile -s cumulative 7_jit_cpu.py _mem_opt
# kernprof -l 7_jit_cpu.py profiling
# python -m line_profiler -rmt r'/zhome/b6/8/228751/py_hpc/mini_project/PYHPC/Task7/7_jit_cpu.py.lprof'