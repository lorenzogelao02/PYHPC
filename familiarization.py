import numpy as np
import matplotlib.pyplot as plt

temp = np.load(r"C:\Users\Juri\OneDrive\Studium\Master\Lehrmaterial\Python and High-Performance Computing\Assignments\Mini project\floorplan_data\23_domain.npy")
mask = np.load(r"C:\Users\Juri\OneDrive\Studium\Master\Lehrmaterial\Python and High-Performance Computing\Assignments\Mini project\floorplan_data\23_interior.npy")

# fig, ax = plt.subplots(1, 2)
# im_temp = ax[0].imshow(temp, origin='lower', cmap='viridis', aspect='equal')
# # ax[0].set_colorbar(label='value')
# ax[0].set_title('Initial values')
# ax[0].set_xlabel('x')
# ax[0].set_ylabel('y')
# im_mask = ax[1].imshow(mask, origin='lower', cmap='viridis', aspect='equal')
# ax[1].set_title('Interior map')
# ax[1].set_xlabel('x')
# ax[1].set_ylabel('y')
# fig.colorbar(im_temp, ax=ax[0], label='value')
# fig.colorbar(im_mask, ax=ax[1], label='value')
# plt.tight_layout()
# plt.savefig(r'C:\Users\Juri\OneDrive\Studium\Master\Lehrmaterial\Python and High-Performance Computing\Assignments\Mini project\Images for report\dwelling_23', dpi=600, bbox_inches = 'tight')
# plt.show()

#Visualization
plt.figure()
im_temp = plt.imshow(temp, origin='lower', cmap='viridis', aspect='equal')
# ax[0].set_colorbar(label='value')
plt.title('Initial values')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(im_temp, label='Temperature')
plt.tight_layout()
# plt.savefig(r'C:\Users\Juri\OneDrive\Studium\Master\Lehrmaterial\Python and High-Performance Computing\Assignments\Mini project\report\images\23_initVal.png', dpi=600, bbox_inches = 'tight')
plt.show()

plt.figure()
im_mask = plt.imshow(mask, origin='lower', cmap='viridis', aspect='equal')
# ax[0].set_colorbar(label='value')
plt.title('Interior map')
plt.xlabel('x')
plt.ylabel('y')
# plt.colorbar(im_mask, label='Value')
plt.tight_layout()
# plt.savefig(r'C:\Users\Juri\OneDrive\Studium\Master\Lehrmaterial\Python and High-Performance Computing\Assignments\Mini project\report\images\23_interior.png', dpi=600, bbox_inches = 'tight')
plt.show()

# plt.show()

# from os.path import join
# import sys

# import numpy as np

# def load_data(load_dir, bid):
#     SIZE = 512
#     u = np.zeros((SIZE + 2, SIZE + 2))
#     u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
#     interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
#     return u, interior_mask


# def jacobi(u, interior_mask, max_iter, atol=1e-6):
#     u = np.copy(u)

#     for i in range(max_iter):
#         # Compute average of left, right, up and down neighbors, see eq. (1)
#         u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
#         u_new_interior = u_new[interior_mask]
#         delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
#         u[1:-1, 1:-1][interior_mask] = u_new_interior

#         if delta < atol:
#             break
#     return u


# def summary_stats(u, interior_mask):
#     u_interior = u[1:-1, 1:-1][interior_mask]
#     mean_temp = u_interior.mean()
#     std_temp = u_interior.std()
#     pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
#     pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
#     return {
#         'mean_temp': mean_temp,
#         'std_temp': std_temp,
#         'pct_above_18': pct_above_18,
#         'pct_below_15': pct_below_15,
#     }


# if __name__ == '__main__':
#     # Load data
#     LOAD_DIR = r"C:\Users\juri-\OneDrive\Studium\Master\Lehrmaterial\Python and High-Performance Computing\Assignments\Mini project\floorplan_data"#'/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
#     with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
#         building_ids = f.read().splitlines()

#     if len(sys.argv) < 2:
#         N = 1
#     else:
#         N = int(sys.argv[1])
#     building_ids = building_ids[:N]

#     # Load floor plans
#     all_u0 = np.empty((N, 514, 514))
#     all_interior_mask = np.empty((N, 512, 512), dtype='bool')
#     for i, bid in enumerate(building_ids):
#         u0, interior_mask = load_data(LOAD_DIR, bid)
#         all_u0[i] = u0
#         all_interior_mask[i] = interior_mask

#     # Run jacobi iterations for each floor plan
#     MAX_ITER = 20_000
#     ABS_TOL = 1e-4

#     all_u = np.empty_like(all_u0)
#     for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
#         u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
#         all_u[i] = u

#     # Print summary statistics in CSV format
#     stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
#     print('building_id, ' + ', '.join(stat_keys))  # CSV header
#     for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
#         stats = summary_stats(u, interior_mask)
#         print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))