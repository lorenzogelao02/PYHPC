import cupy as cp

# The decorator fuses all the math into a single CUDA kernel!
@cp.fuse(kernel_name='compute_iteration')
def compute_iteration(top, bottom, left, right, center, mask):
    u_new = 0.25 * (top + bottom + left + right)
    return cp.where(mask, u_new, center)

def jacobi_cupy(u, interior_mask, max_iter):
    for _ in range(max_iter):
        u[1:-1, 1:-1] = compute_iteration(
            u[:-2, 1:-1], u[2:, 1:-1], u[1:-1, :-2], u[1:-1, 2:], 
            u[1:-1, 1:-1], interior_mask
        )
    return u

if __name__ == '__main__':
    import time
    from os.path import join
    import cupy as cp
    
    def load_data(load_dir, bid):
        SIZE = 512
        import numpy as np
        u = cp.zeros((SIZE + 2, SIZE + 2), dtype=cp.float32)
        # Using np.load then cp.asarray is much faster and safer than cp.load directly
        u[1:-1, 1:-1] = cp.asarray(np.load(join(load_dir, f"{bid}_domain.npy")))
        interior_mask = cp.asarray(np.load(join(load_dir, f"{bid}_interior.npy")))
        return u, interior_mask

    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        # Run a small subset (2 floorplans)
        building_ids = f.read().splitlines()[:2]

    print("Warming up GPU...", flush=True)
    u0, mask = load_data(LOAD_DIR, building_ids[0])
    _ = jacobi_cupy(u0, mask, 10)

    print("Running CuPy benchmark on 2 floorplans (20,000 iterations each)...", flush=True)
    start = time.time()
    for bid in building_ids:
        u0, mask = load_data(LOAD_DIR, bid)
        u_final = jacobi_cupy(u0, mask, 20000)
    end = time.time()
    
    time_taken = end - start
    print(f"Time for 2 floorplans: {time_taken:.2f} seconds", flush=True)
    print(f"Average time per floorplan: {time_taken/2:.2f} seconds", flush=True)
    
    total_floorplans = 282_000 # Approximation based on dataset
    estimated_hours = (time_taken / 2) * total_floorplans / 3600
    print(f"Estimated time to process ALL floorplans sequentially on GPU: {estimated_hours:.2f} hours", flush=True)