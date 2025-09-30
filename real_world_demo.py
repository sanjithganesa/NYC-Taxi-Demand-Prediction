import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train import NYCTaxiDataset, SimpleFC

# ------------------------
# Parameters
# ------------------------
SEQ_LEN = 12
PRED_LEN = 1

DATA_DIR = "data/nyc"
LOOKUP_FILE = "data/nyc_raw/taxi_zone_lookup.csv"
MODEL_FILE = "checkpoints/nyc_model.pt"

# ------------------------
# Load dataset
# ------------------------
dataset = NYCTaxiDataset(DATA_DIR, LOOKUP_FILE)
print(f"Loaded demand tensor shape: {dataset.data.shape}")

num_zones = len(dataset.zones)

# ------------------------
# Load model
# ------------------------
model = SimpleFC(num_zones, SEQ_LEN, PRED_LEN)
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

# ------------------------
# Pick a demo sequence safely
# ------------------------
start_idx = len(dataset.data) - SEQ_LEN - 1  # leave 1 step for y_true
x_demo = dataset.data[start_idx:start_idx + SEQ_LEN]  # (seq_len, num_zones)
y_true_demo = dataset.data[start_idx + SEQ_LEN]       # (num_zones,)

x_demo_tensor = torch.tensor(x_demo, dtype=torch.float32).unsqueeze(0)  # add batch dim

# ------------------------
# Model prediction
# ------------------------
with torch.no_grad():
    y_pred_demo = model(x_demo_tensor).squeeze(0).numpy()  # flatten to (num_zones,)

# ------------------------
# Compare first 10 zones
# ------------------------
print("True (first 10 zones):", y_true_demo[:10])
print("Pred (first 10 zones):", y_pred_demo[:10])

# ------------------------
# Visualization
# ------------------------
zone_lookup = pd.read_csv(LOOKUP_FILE)
zone_labels = zone_lookup.set_index("LocationID").loc[dataset.zones]["Zone"].values

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.heatmap(y_true_demo.reshape(1, -1), ax=axes[0], cmap="YlOrRd", cbar=True)
axes[0].set_title("True Demand (all zones)")
axes[0].set_yticks([])
axes[0].set_xticks(np.arange(num_zones))
axes[0].set_xticklabels(zone_labels, rotation=90, fontsize=8)

sns.heatmap(y_pred_demo.reshape(1, -1), ax=axes[1], cmap="YlOrRd", cbar=True)
axes[1].set_title("Predicted Demand (all zones)")
axes[1].set_yticks([])
axes[1].set_xticks(np.arange(num_zones))
axes[1].set_xticklabels(zone_labels, rotation=90, fontsize=8)

plt.tight_layout()
plt.savefig("real_world_demo_heatmaps.png")
print("Heatmaps saved to real_world_demo_heatmaps.png")
