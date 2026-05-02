from os.path import join
import sys
import pandas as pd
import numpy as np
from multiprocessing.pool import Pool
import multiprocessing as mp
from itertools import chain
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

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
    mp.set_start_method('fork', force=True)
    num_workers = 16
    c_size = 1
    
    dummy_u = np.zeros((10, 10), dtype=np.float64)
    dummy_mask = np.ones((8, 8), dtype=np.bool_)
    jacobi(dummy_u, dummy_mask, 1, 1e-4)

    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    
    if str(sys.argv[1]) == 'all':
        with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
            building_ids = f.read().splitlines()
    else:
        N = int(sys.argv[1])#100
        with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
            building_ids = f.read().splitlines()[:N]

    floorplans_data = []
    for bid in building_ids:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        floorplans_data.append((u0, interior_mask))
    
    with Pool(num_workers) as pool:
            parallel_results = pool.imap_unordered(process_single, floorplans_data, chunksize=c_size)
            all_u = list(parallel_results)
    
    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    all_interior_mask = [fp[1] for fp in floorplans_data]
    # print('building_id, ' + ', '.join(stat_keys))  # CSV header
    rows = []
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        rows.append({'building_id': bid, **stats})
    df = pd.DataFrame(rows)
    df.to_csv(r'output/corr_summary_stats.csv')
    temps = df["mean_temp"]

    #Histogram
    bin_edges = np.histogram_bin_edges(temps, bins="fd")
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2
    fig, ax = plt.subplots()
    ax.hist(temps, bins=bin_edges, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.axvline(temps.mean(), color="r", linestyle="--", linewidth=1.5, label=f"Mean = {temps.mean():.1f} °C")
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f"{c:.1f}" for c in bin_centers], rotation=45, ha="right")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlabel("Mean temperature [°C]")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(r"output/corr_temp_distribution.png", dpi=600, bbox_inches="tight")


    # #Histogram
    # # bin_edges = np.arange(df['mean_temp'].min(), df['mean_temp'].max() + 1, 1)
    # # plt.hist(df['mean_temp'], bins=bin_edges, edgecolor='black', linewidth=1.2)
    # # plt.grid(alpha = 0.5)
    # # plt.xlabel(r'Mean Temperature [$°C$]')
    # # plt.ylabel('Count')
    # # plt.savefig(r'output\temp_distribution.png', dpi = 600, bbox_inches = "tight")
    # temps = df['mean_temp']
    # q25, q75 = np.percentile(temps, [25, 75])
    # iqr = q75 - q25
    # bin_width = 2 * iqr * len(temps) ** (-1 / 3)
    # if bin_width <= 0:
    #     bin_width = 1.0
    # bin_edges = np.arange(np.floor(temps.min()),
    #                     np.ceil(temps.max()) + bin_width,
    #                     bin_width)
    
    # fig, ax = plt.subplots()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.hist(temps, bins=bin_edges, edgecolor="black", linewidth=0.8, alpha=0.85)
    # ax.axvline(temps.mean(), color = "r", linestyle="--", linewidth=1.5, label=f"Mean = {temps.mean():.1f} °C")
    # ax.set_xlabel(r"Mean temperature [°C]")
    # ax.set_ylabel("Count")
    # ax.grid(True, alpha=0.2)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(r"\output\temp_distribution.png", dpi=600, bbox_inches="tight")

    #Average mean temp of the buildings
    av_mean_temp = np.mean(temps)

    #Average temp standartd deviation
    std = np.std(temps)

    #50% above/below 18/15
    above = 0
    below = 0
    for i, row in df.iterrows():
        if row['pct_above_18'] >= 50:
            above+=1
        if row['pct_below_15'] >= 50:
            below+=1
    
    print('The average mean temperature is:', av_mean_temp)
    print('The associated standard deviation is:', std)
    print(r'The number of floorplans with 50% above 18 °C:', above)
    print(r'The number of floorplans with 50% below 15 °C:', below)