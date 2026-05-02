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
# python -u 7_jit_cpu.py _JIT_timing
# python -u 7_no_jit.py _no_JIT_baseScript_timing
# python -m cProfile -s cumulative 7_jit_cpu.py _mem_opt
# kernprof -l 7_jit_cpu.py profiling
# python -m line_profiler -rmt "7_jit_cpu.py.lprof"
python -u 7_jit_cpu_corrected.py corrected