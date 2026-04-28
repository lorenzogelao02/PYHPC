# Task 10: CuPy Optimization Summary

### Profiling Results
Profiling with `nsys` revealed that the CPU was bottlenecked by **Kernel Launch Overhead**, spending 90% of its time (1.4s) launching 280,000 tiny kernels instead of performing math.

### The Fix: @cp.fuse
We applied `@cp.fuse()` to the core Jacobi logic. This merged multiple operations (adds, multiplies, and `where` conditions) into a single optimized CUDA kernel, reducing launches from 280k to 80k and eliminating intermediate GPU memory allocations.

### Performance Comparison
| Implementation | Avg Time / Floorplan |
| :--- | :--- |
| **Vectorized CuPy** | 2.29s |
| **Custom Numba Kernel** | 1.03s |
| **Fused CuPy** | **0.86s** |

**Conclusion:** Kernel fusion allowed high-level CuPy code to outperform low-level hand-written Numba CUDA kernels by minimizing the PCIe communication bottleneck.
