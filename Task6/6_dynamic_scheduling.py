from os.path import join
import sys

import numpy as np
from multiprocessing.pool import Pool
from itertools import chain

    
def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
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
    chunk_sizes = [1, 3, 10]
    print(f"Starting chunk size benchmark on {num_workers} workers...")
    
    master_file_path = "output/task6/chunksizes_results.txt"
    with open(master_file_path, "w") as out_file:
        out_file.write("--- Chunk Size Benchmark Results ---\n")

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
            out_file.write(f"Time taken with chunksize={c_size:02d}: {time_taken:.2f} seconds\n")
            
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

    plt.figure(figsize=(10, 6))

    # Plot actual speedup with larger markers
    plt.plot(chunk_sizes, speedups, marker='o', color='crimson', linewidth=3, markersize=8)

    # Annotate each point
    for i, txt in enumerate(speedups):
        plt.annotate(f"{txt:.2f}x", 
                     (chunk_sizes[i], speedups[i]), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center',
                     fontsize=12,
                     fontweight='bold')

    plt.title('Speedup Growth vs Chunk Size (16 Workers)', fontsize=16, fontweight='bold')
    plt.xlabel('Chunk Size', fontsize=14)
    plt.ylabel('Speedup Factor', fontsize=14)
    plt.xticks(chunk_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig('output/task6/speedup_plot.png', dpi=300)
    print("Plot successfully saved to output/task6/3_chunks_speedup_plot.png!")