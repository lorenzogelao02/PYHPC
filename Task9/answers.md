# Task 9: CuPy Accelerated Jacobi Solver

### Run and time the new solution for a small subset of floorplans. How does the performance compare to the reference?
The CuPy vectorized solution processed 2 floorplans in **4.56 seconds**, resulting in an average of **2.28 seconds per floorplan**. It is about **2x slower** than our hand-written Numba CUDA implementation (which averaged 1.03s per floorplan).

### How long would it now take to process all floorplans?
At 2.28 seconds per floorplan, processing all 282,000 floorplans sequentially on the GPU using CuPy would take approximately **178.74 hours**.

### Was anything surprising about the performance?
The most surprising aspect is that despite CuPy utilizing the exact same underlying GPU hardware as Numba, it performed significantly worse (2.28s vs 1.03s). 

This happens because of **Kernel Launch Overhead** and intermediate memory allocations. In CuPy, every mathematical operation (`+`, `*`, `cp.where`) is evaluated as a separate independent CUDA kernel. Evaluating `0.25 * (A + B + C + D)` requires CuPy to launch 5 separate mini-programs from the CPU to the GPU and allocate temporary arrays in GPU memory to hold the intermediate sums. Doing this 20,000 times creates massive overhead bottlenecks across the PCIe bus. 

In contrast, our Numba implementation compiled the entire arithmetic calculation and the mask condition into a **single fused kernel**, entirely eliminating the launch overhead and temporary array allocations. 
