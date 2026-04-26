from os.path import join
import sys

import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@jit(nopython=True)
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    M, N = interior_mask.shape

    for n in range(max_iter):
        delta = 0.0
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if interior_mask[i-1, j-1]:
                    old_val = u[i, j]
                    new_val = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
                    u[i, j] = new_val
                    diff = abs(old_val - new_val)
                    if diff > delta:
                        delta = diff       
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


if __name__ == '__main__':
    # Load data
    LOAD_DIR = r'/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

        #Visualization
        plt.figure()
        im_sim = plt.imshow(u, origin='lower', cmap='viridis', aspect='equal')
        plt.title('Sim_results')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(im_sim, label='Temperature')
        plt.tight_layout()
        name = str('vis'+str(i+1)+'.png')
        plt.savefig(name, dpi = 600, bbox_inches = 'tight')
        # plt.show()

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))