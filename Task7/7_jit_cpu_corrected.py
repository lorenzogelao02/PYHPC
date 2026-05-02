from os.path import join
import sys

import numpy as np
from multiprocessing.pool import Pool
import multiprocessing as mp
from itertools import chain
from numba import jit
  
def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

#18.23s record
@jit(nopython=True, fastmath=True)
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    u_new = np.copy(u)
    n, m = interior_mask.shape
    for _ in range(max_iter):
        max_delta = 0.0
        for i in range(1, n+1):
            for j in range(1, m+1):
                if interior_mask[i-1, j-1]:
                    u_new_val = 0.25*(u[i-1, j]+u[i+1, j]+u[i, j+1]+u[i, j-1])
                    delta = np.abs(u[i, j]-u_new_val)
                    u_new[i, j] = u_new_val
                    if delta > max_delta:
                        max_delta=delta
        u, u_new = u_new, u
        if max_delta < atol:
            break
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

def process_single(floorplan):
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    u0, interior_mask = floorplan
    u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    return u 

if __name__ == '__main__' :
    mp.set_start_method('fork', force=True) #Added to force fork and thus make the workers inherit the compiled jacobi function
    
    #Use the following process to force the compilation of the jacobi function before timing
    dummy_u = np.zeros((10, 10), dtype=np.float64)
    dummy_mask = np.ones((8, 8), dtype=np.bool_)
    jacobi(dummy_u, dummy_mask, 1, 1e-4)

    run_number = sys.argv[1]
    import time
    N = 100
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    floorplans_data = []
    for bid in building_ids:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        floorplans_data.append((u0, interior_mask))

    num_workers = 16
    chunk_sizes = [1]#, 3, 10]
    print(f"Starting chunk size benchmark on {num_workers} workers...")
    
    master_file_path = "output/JITopt_results"+str(run_number)+".txt"
    with open(master_file_path, "w") as out_file:
        out_file.write("--- JIT optimizing Results ---\n")

    times = []
    for c_size in chunk_sizes:
        
        start_time = time.time()
        with Pool(num_workers) as pool:
            parallel_results = pool.imap_unordered(process_single, floorplans_data, chunksize=c_size)
            all_u = list(parallel_results)
        end_time = time.time()
    
    # --- 3. PRINT RESULTS ---
        time_taken = end_time - start_time
        times.append(time_taken)
        with open(master_file_path, "a") as out_file:
            # out_file.write(f"Time taken with chunksize={c_size:02d}: {time_taken:.2f} seconds\n")
            out_file.write(f"Time taken with chunksize={c_size:02d} and {num_workers:02d} workers and JIT: {time_taken:.2f} seconds\n")
            
        print(f"Finished evaluating chunksize={c_size}!")        
        # stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
        # print('building_id, ' + ', '.join(stat_keys))
        # for i, bid in enumerate(building_ids):
        #     # We also need to extract out the interior mask again to print the stats
        #     _, interior_mask = floorplans_data[i] 
        #     stats = summary_stats(all_u[i], interior_mask)
        #     print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

    # --- 4. AUTO-GENERATE GRAPH ---
    import matplotlib.pyplot as plt

    # Calculate Speedup (T_serial / T_parallel)
    t_serial = times[0] # The time for 1 worker
    speedups = [t_serial / t for t in times]

    # plt.figure(figsize=(10, 6))

    # # Plot actual speedup with larger markers
    # plt.plot(chunk_sizes, speedups, marker='o', color='crimson', linewidth=3, markersize=8)

    # # Annotate each point
    # for i, txt in enumerate(speedups):
    #     plt.annotate(f"{txt:.2f}x", 
    #                  (chunk_sizes[i], speedups[i]), 
    #                  textcoords="offset points", 
    #                  xytext=(0, 10), 
    #                  ha='center',
    #                  fontsize=12,
    #                  fontweight='bold')

    # plt.title('Speedup Growth vs Chunk Size (16 Workers)', fontsize=16, fontweight='bold')
    # plt.xlabel('Chunk Size', fontsize=14)
    # plt.ylabel('Speedup Factor', fontsize=14)
    # plt.xticks(chunk_sizes, fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()

    # plt.savefig("output/speedup_plot"+str(run_number)+".png", dpi=600, bbox_inches="tight")
    # print("Plot successfully saved to output/speedup_plot"+str(run_number)+".png!")