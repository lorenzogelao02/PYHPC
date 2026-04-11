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

def process_chunk(chunks):
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    
    result = []

    for u0, interior_mask in chunks:
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        result.append(u)
    return result

if __name__ == '__main__' : 
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    else:
        num_workers = 4
    
    import time
    N = 100
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    floorplans_data = []
    for bid in building_ids:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        floorplans_data.append((u0, interior_mask))
    
    chunk_size = N // num_workers
    chunks = [floorplans_data[i : i + chunk_size] for i in range(0, len(floorplans_data), chunk_size)]
    
    start_time = time.time()
    with Pool(num_workers) as pool:
        parallel_results = pool.map(process_chunk, chunks)
    end_time = time.time()

    all_u = [item for row in parallel_results for item in row]

 # --- 3. PRINT RESULTS ---
    print(f"Time taken with {num_workers} workers: {end_time - start_time:.2f} seconds")
    
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for i, bid in enumerate(building_ids):
        # We also need to extract out the interior mask again to print the stats
        _, interior_mask = floorplans_data[i] 
        stats = summary_stats(all_u[i], interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))