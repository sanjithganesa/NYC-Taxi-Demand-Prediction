import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Dataset
# -------------------------
class NYCTaxiDataset(Dataset):
    def __init__(self, data_dir, lookup_file, seq_len=12, pred_len=1):
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Load zones from taxi lookup
        zone_lookup = pd.read_csv(lookup_file)
        self.zones = sorted(zone_lookup["LocationID"].unique())
        self.zone_to_idx = {z: i for i, z in enumerate(self.zones)}
        n = len(self.zones)

        self.data = []

        # Load sparse matrices and align columns
        for file in os.listdir(data_dir):
            if file.endswith("_sparse.npz"):
                npz = np.load(os.path.join(data_dir, file))
                mat = np.zeros((npz["shape"][0], n), dtype=np.float32)

                for d, r, c in zip(npz["data"], npz["row"], npz["col"]):
                    if c < n:
                        mat[r, c] = d
                self.data.append(mat)

        self.data = np.concatenate(self.data, axis=0)  # (time, num_zones)
        print(f"Loaded demand tensor shape: {self.data.shape}")

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len: idx+self.seq_len+self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -------------------------
# Simple Model
# -------------------------
class SimpleFC(nn.Module):
    def __init__(self, num_zones, seq_len, pred_len):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.fc = nn.Sequential(
            nn.Linear(seq_len*num_zones, 512),
            nn.ReLU(),
            nn.Linear(512, pred_len*num_zones)
        )
        self.num_zones = num_zones

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1)
        out = self.fc(x)
        out = out.view(b, self.pred_len, self.num_zones)
        return out

# -------------------------
# Training
# -------------------------
def train(data_dir, lookup_file, adj_file, seq_len=12, pred_len=1, batch_size=64, epochs=5, lr=1e-3):
    dataset = NYCTaxiDataset(data_dir, lookup_file, seq_len, pred_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_zones = len(dataset.zones)
    model = SimpleFC(num_zones, seq_len, pred_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/nyc_model.pt")
    print("✅ Model saved to checkpoints/nyc_model.pt")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--lookup_file", type=str, required=True)
    parser.add_argument("--adj_file", type=str, required=True)  # not used for now, placeholder
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args.data_dir, args.lookup_file, args.adj_file,
          args.seq_len, args.pred_len, args.batch_size, args.epochs, args.lr)
