import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse
from tqdm import tqdm

def load_sparse_custom(file):
    """Reconstruct sparse matrix from custom npz format"""
    npz = np.load(file)
    return sp.coo_matrix(
        (npz["data"], (npz["row"], npz["col"])),
        shape=tuple(npz["shape"])
    ).tocsr()

def build_adjacency_matrix(data_dir, lookup_dir, output_file):
    # Read taxi zone lookup
    zone_lookup = pd.read_csv(os.path.join(lookup_dir, "taxi_zone_lookup.csv"))
    zones = sorted(zone_lookup["LocationID"].unique())
    zone_to_idx = {z: i for i, z in enumerate(zones)}

    n = len(zones)
    adj = np.zeros((n, n), dtype=np.int32)

    # List all sparse files
    files = [f for f in os.listdir(data_dir) if f.endswith("_sparse.npz")]

    for file in tqdm(files, desc="Processing sparse matrices", unit="file"):
        trip_sparse = load_sparse_custom(os.path.join(data_dir, file))

        # Ensure we are working with zone IDs
        # Trip sparse "col" indices are actually LocationIDs, not 0..264
        npz = np.load(os.path.join(data_dir, file))
        cols = npz["col"]  # raw LocationIDs (not compact indices)

        # Map raw LocationIDs to [0..n-1]
        valid_mask = np.isin(cols, zones)
        mapped_cols = [zone_to_idx[c] for c in cols[valid_mask]]

        # Rebuild trip matrix restricted to known zones
        trip_sparse = sp.coo_matrix(
            (npz["data"][valid_mask], (npz["row"][valid_mask], mapped_cols)),
            shape=(npz["shape"][0], n)
        ).tocsr()

        # Compute co-occurrence
        co_occurrence = (trip_sparse.T @ trip_sparse).toarray()
        co_occurrence[co_occurrence > 0] = 1
        co_occurrence = co_occurrence.astype(np.int32)
        adj += co_occurrence

    np.fill_diagonal(adj, 0)
    adj = (adj > 0).astype(np.int32)

    print(f"\nAdjacency matrix shape: {adj.shape}")
    np.save(output_file, adj)
    print(f"✅ Saved adjacency matrix to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--lookup_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="adj_matrix.npy")
    args = parser.parse_args()

    build_adjacency_matrix(args.data_dir, args.lookup_dir, args.output_file)
