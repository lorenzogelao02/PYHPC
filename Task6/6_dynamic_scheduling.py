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

    worker_counts = [10, 12, 16]
    chunk_sizes = [1, 3, 10]
    print(f"Starting 2D benchmark on workers={worker_counts} and chunksizes={chunk_sizes}...")
    
    master_file_path = "output/task6/2d_benchmark_results.txt"
    with open(master_file_path, "w") as out_file:
        out_file.write("--- 2D Nested Benchmark Results ---\n")

    results_matrix = {c: [] for c in chunk_sizes}

    for c_size in chunk_sizes:
        for num_workers in worker_counts:
            start_time = time.time()
            with Pool(num_workers) as pool:
                parallel_results = pool.imap_unordered(process_single, floorplans_data, chunksize=c_size)
                all_u = list(parallel_results)
            end_time = time.time()
        
            time_taken = end_time - start_time
            results_matrix[c_size].append(time_taken)
            
            with open(master_file_path, "a") as out_file:
                out_file.write(f"Chunksize={c_size:02d} | Workers={num_workers:02d} | Time: {time_taken:.2f} seconds\n")
            print(f"Finished chunksize={c_size} with {num_workers} workers!")

    # --- 4. AUTO-GENERATE MULTI-LINE GRAPH ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    colors = ['dodgerblue', 'mediumseagreen', 'crimson']

    # Because Speedup gets messy comparing 2 variables, let's just plot Raw Execution Time!
    for idx, c_size in enumerate(chunk_sizes):
        times = results_matrix[c_size]
        plt.plot(worker_counts, times, label=f'Chunksize={c_size}', marker='o', color=colors[idx], linewidth=3, markersize=8)

        # Annotate each point with its time in seconds
        for i, val in enumerate(times):
            plt.annotate(f"{val:.0f}s", 
                         (worker_counts[i], val), 
                         textcoords="offset points", 
                         xytext=(0, 10), 
                         ha='center',
                         fontsize=10,
                         fontweight='bold')

    plt.title('Execution Time vs Workers (Grouped by Chunk Size)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Workers', fontsize=14)
    plt.ylabel('Execution Time (seconds) - Lower is Better!', fontsize=14)
    plt.xticks(worker_counts, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Dynamic Chunksize", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig('output/task6/2d_benchmark_plot.png', dpi=300)
    print("Plot successfully saved to output/task6/2d_benchmark_plot.png!")