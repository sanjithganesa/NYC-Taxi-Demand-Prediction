import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train import NYCTaxiDataset, SimpleFC

# ------------------------
# Load dataset
# ------------------------
data_dir = "data/nyc"
lookup_file = "data/nyc_raw/taxi_zone_lookup.csv"
dataset = NYCTaxiDataset(data_dir, lookup_file)

# ------------------------
# Load model
# ------------------------
seq_len = 12
pred_len = 1
num_zones = len(dataset.zones)
model = SimpleFC(num_zones, seq_len, pred_len)
model.load_state_dict(torch.load("checkpoints/nyc_model.pt"))
model.eval()

# ------------------------
# Collect predictions and true values
# ------------------------
true_list = []
pred_list = []

with torch.no_grad():
    for i in range(len(dataset)):
        x, y_true = dataset[i]
        x = x.unsqueeze(0)  # batch dim
        y_pred = model(x)
        true_list.append(y_true.numpy().flatten())
        pred_list.append(y_pred.numpy().flatten())

true_array = np.array(true_list)    # shape: (num_sequences, num_zones)
pred_array = np.array(pred_list)

print(f"Loaded demand tensor shape: {true_array.shape}")

# ------------------------
# Visualize first 10 zones
# ------------------------
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

for zone_idx in range(10):   # first 10 zones
    plt.figure(figsize=(8, 3))
    plt.plot(true_array[:, zone_idx], label="True", marker='o')
    plt.plot(pred_array[:, zone_idx], label="Pred", marker='x')
    plt.title(f"Zone {zone_idx} demand prediction")
    plt.xlabel("Sequence index")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"zone_{zone_idx}_lineplot.png"))
    plt.close()

# ------------------------
# Heatmaps
# ------------------------
plt.figure(figsize=(12, 5))
plt.imshow(true_array, aspect='auto', cmap='Blues')
plt.colorbar(label="True demand")
plt.title("True Demand Heatmap")
plt.xlabel("Zone index")
plt.ylabel("Sequence index")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "true_demand_heatmap.png"))
plt.close()

plt.figure(figsize=(12, 5))
plt.imshow(pred_array, aspect='auto', cmap='Reds')
plt.colorbar(label="Predicted demand")
plt.title("Predicted Demand Heatmap")
plt.xlabel("Zone index")
plt.ylabel("Sequence index")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "predicted_demand_heatmap.png"))
plt.close()

print(f"All plots saved to {plot_dir}/")
