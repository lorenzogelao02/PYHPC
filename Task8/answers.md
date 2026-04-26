# Task 8: Custom CUDA Kernel Implementation

### a) Briefly describe your new solution. How did you structure your kernel and helper function?

Our new solution divides the workload between the CPU (Host) and the GPU (Device) to leverage massive parallelization:

**1. The Kernel (`jacobi_kernel`)**
The kernel executes directly on the GPU. Instead of using `for` loops to iterate through the grid, we mapped the floorplan coordinates directly to a 2D grid of GPU threads using `cuda.grid(2)`. Each thread is responsible for computing exactly one point `(i, j)` per iteration. 
To prevent race conditions (where threads might overwrite data that neighboring threads are still trying to read), we utilize a Ping-Pong buffer approach. The kernel reads the four neighboring temperatures from an input array (`u_old`) and writes the newly averaged temperature to a separate output array (`u_new`). If a thread falls on a wall or outside the room, it simply copies the temperature over to preserve the boundaries.

**2. The Helper Function (`helper_function`)**
The helper function runs on the CPU and acts as the manager. It handles three main responsibilities:
- **Memory Management:** It transfers the initial `u` grid and `interior_mask` from the CPU memory to the GPU memory using `cuda.to_device`, and allocates an empty buffer `d_u_new` using `cuda.device_array_like`.
- **Grid Configuration:** It calculates the required Blocks Per Grid (`bpg`) to cover the entire floorplan using 16x16 Threads Per Block (`tpb`).
- **Synchronization Loop:** Because the kernel only performs a single step, the CPU handles synchronization by running a `for` loop `max_iter` times. Inside the loop, it launches the kernel and then swaps the `u_new` and `u_old` array pointers. Once the loop concludes, it copies the final result back to the CPU.

---

### b) Run and time the new solution for a small subset of floorplans. How does the performance compare to the reference?

The new CUDA solution was benchmarked on a subset of 2 floorplans for 20,000 iterations each on a V100 GPU.
- **Total Time:** 2.06 seconds
- **Average Time per Floorplan:** 1.03 seconds

**Comparison to Reference:**
The GPU implementation is faster than the purely sequential CPU reference implementation. However, the speedup is not as massive as typically expected from a GPU. This is primarily due to **Kernel Launch Overhead**. Because the `helper_function` uses a Python `for` loop to launch the CUDA kernel 20,000 separate times per floorplan, the communication overhead between the CPU and the GPU driver dominates the total execution time. The actual mathematical computation on the GPU takes microseconds, but the overhead of launching it 20,000 times takes roughly 1 second.

---

### c) How long would it now take to process all floorplans?

Based on the benchmark of 1.03 seconds per floorplan:
- **Estimated Time:** `1.03 seconds * 282,000 floorplans = 290,460 seconds`
- **Total Hours:** Approximately **80.67 hours** to process the entire dataset sequentially on the GPU.
