import argparse
import os
import requests
from tqdm import tqdm   # ✅ progress bar

# Example links from DataTalksClub releases
GITHUB_BASE = "https://github.com/DataTalksClub/nyc-tlc-data/releases/download"
EXAMPLE_FILES = [
    "yellow_tripdata_2019-01.csv.gz",
    "yellow_tripdata_2019-02.csv.gz",
    "green_tripdata_2019-01.csv.gz"
]

def download_and_extract_gzip(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(url)
    local_gz = os.path.join(out_dir, fname)
    print("Downloading:", url)

    r = requests.get(url, stream=True)
    r.raise_for_status()
    total_size = int(r.headers.get("content-length", 0))

    # ✅ tqdm progress bar
    with open(local_gz, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=fname
    ) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print("Downloaded gzip:", local_gz)

    # decompress .gz
    import gzip, shutil
    local_csv = local_gz[:-3]  # drop .gz
    with gzip.open(local_gz, "rb") as f_in, open(local_csv, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print("Extracted to:", local_csv)
    return local_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/nyc_raw")
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()

    for fname in EXAMPLE_FILES[:args.count]:
        url = f"{GITHUB_BASE}/yellow/{fname}"
        try:
            download_and_extract_gzip(url, args.out_dir)
        except Exception as e:
            print("Error downloading", url, e)
