from numba import cuda

@cuda.jit
def jacobi_kernel(u_new, u_old, mask):
    i, j = cuda.grid(2)

    if (0 < i < u_new.shape[0] - 1) and (0 < j < u_old.shape[1] - 1) and (mask[i-1, j-1] == 1):
        new_val = 0.25 * (u_old[i-1, j] + u_old[i+1, j] + u_old[i, j-1] + u_old[i, j+1])
        u_new[i, j] = new_val
    elif i < u_new.shape[0] and j < u_old.shape[1]:
        u_new[i, j] = u_old[i, j]

def helper_function(u, interior_mask, max_iter):
    d_u_old = cuda.to_device(u)
    d_u_new = cuda.device_array_like(u)
    d_mask = cuda.to_device(interior_mask)

    tpb = (16, 16)
    blocks_x = (u.shape[0] + tpb[0] - 1) // tpb[0]
    blocks_y = (u.shape[1] + tpb[1] - 1) // tpb[1]
    bpg = (blocks_x, blocks_y)

    for _ in range(max_iter):
        jacobi_kernel[bpg, tpb](d_u_new, d_u_old, d_mask)
        d_u_old, d_u_new = d_u_new, d_u_old

    u = d_u_old.copy_to_host()
    return u

if __name__ == '__main__':
    import time
    from os.path import join
    import numpy as np

    def load_data(load_dir, bid):
        SIZE = 512
        u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float32)
        u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
        interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
        return np.ascontiguousarray(u), np.ascontiguousarray(interior_mask)

    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        # Run a small subset (2 floorplans)
        building_ids = f.read().splitlines()[:2]

    print("Warming up GPU and compiling kernel...", flush=True)
    u0, mask = load_data(LOAD_DIR, building_ids[0])
    _ = helper_function(u0, mask, 10)

    print("Running CUDA benchmark on 2 floorplans (20,000 iterations each)...", flush=True)
    start = time.time()
    for bid in building_ids:
        u0, mask = load_data(LOAD_DIR, bid)
        u_final = helper_function(u0, mask, 20000)
    end = time.time()
    
    time_taken = end - start
    print(f"Time for 2 floorplans: {time_taken:.2f} seconds", flush=True)
    print(f"Average time per floorplan: {time_taken/2:.2f} seconds", flush=True)
    
    total_floorplans = 282_000 # Approximation based on dataset
    estimated_hours = (time_taken / 2) * total_floorplans / 3600
    print(f"Estimated time to process ALL floorplans sequentially on GPU: {estimated_hours:.2f} hours", flush=True)