import os
import glob
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm

def preprocess_csvs(input_dir, out_dir, grid_size=0.01, time_freq='1H'):
    os.makedirs(out_dir, exist_ok=True)
    
    # List all CSV files in the directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    
    for path in tqdm(csv_files, desc="Processing CSVs"):
        print(f"Reading {path} ...")
        df = pd.read_csv(path)
        
        # Detect required columns dynamically
        datetime_col = next((c for c in df.columns if 'pickup_datetime' in c.lower()), None)
        pu_col = next((c for c in df.columns if 'pulocationid' in c.lower()), None)
        do_col = next((c for c in df.columns if 'dolocationid' in c.lower()), None)
        
        if not datetime_col or not pu_col or not do_col:
            print(f"Skipping {path}: Required columns not found")
            continue
        
        df = df[[datetime_col, pu_col, do_col]].copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df = df.dropna(subset=[datetime_col, pu_col, do_col])
        
        # Aggregate by time bin and edge
        df['time_bin'] = df[datetime_col].dt.floor(time_freq)
        df['count'] = 1
        df["edge_id"] = df[pu_col].astype(str) + "_" + df[do_col].astype(str)
        
        # Pivot to time x edge matrix
        pivot = df.pivot_table(
            index="time_bin",
            columns="edge_id",
            values="count",
            aggfunc="sum"  # sum counts if duplicates exist
        ).fillna(0).astype(np.float32)
        
        # Convert to sparse matrix
        sparse_mat = coo_matrix(pivot.values)
        
        # Save sparse matrix
        base_name = os.path.basename(path).replace(".csv", "")
        sparse_path = os.path.join(out_dir, f"{base_name}_sparse.npz")
        print(f"Saving sparse matrix to {sparse_path}")
        np.savez_compressed(sparse_path, data=sparse_mat.data, row=sparse_mat.row, col=sparse_mat.col, shape=sparse_mat.shape)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--grid_size", type=float, default=0.01)
    parser.add_argument("--time_freq", type=str, default="1H")
    args = parser.parse_args()
    
    preprocess_csvs(args.input_dir, args.out_dir, args.grid_size, args.time_freq)
