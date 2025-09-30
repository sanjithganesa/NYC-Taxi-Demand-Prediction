# NYC Taxi Demand Prediction

This project implements a **spatio-temporal taxi demand prediction system** for New York City using historical trip data. It constructs a **graph-based adjacency matrix** of taxi zones and trains a **simple feedforward neural network** to predict future demand at each zone. The project also includes visualization of predictions for real-world scenarios.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Evaluation](#evaluation)  
- [Real-World Demo](#real-world-demo)  
- [Future Work](#future-work)  
- [References](#references)  

---

## Project Overview

The goal is to predict taxi demand for each zone in NYC based on historical trip data. The workflow includes:

1. **Data preprocessing**: Convert raw taxi trip data into sparse matrices per day.
2. **Adjacency matrix construction**: Build a co-occurrence adjacency matrix of taxi zones based on trips.
3. **Model training**: Train a feedforward neural network (`SimpleFC`) using sequences of historical demand.
4. **Evaluation & visualization**: Compare predictions with true demand and generate heatmaps.
5. **Real-world demo**: Show predictions for a selected time step and visualize them across NYC zones.

---

## Dataset

- **Source**: [NYC Taxi & Limousine Commission (TLC)](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
- **Files used**:  
  - `yellow_tripdata_2019-02.csv` (preprocessed into sparse matrices)  
  - `taxi_zone_lookup.csv` (contains `LocationID` and zone names)  

- **Preprocessed Data**: Sparse matrices per day for efficient storage and adjacency matrix construction.

---

## Project Structure

```

Spatio_Case_study/
│
├─ data.zip/                # unzip before use
│   ├─ nyc/                 # Preprocessed sparse matrices & adjacency matrix
│   └─ nyc_raw/             # Original taxi zone lookup
│
├─ checkpoints/
│   └─ nyc_model.pt         # Trained model
│
├─ build_adj_matrix.py      # Builds adjacency matrix from sparse matrices
├─ plots/                   #plots from evaluate.py
├─ train.py                 # Trains the SimpleFC model
├─ download_stream.py       # To download the required data (ignore)
├─ preprocess_nyc.py        # To preprocess the raw data (ignore)
├─ evaluate.py              # Evaluates model predictions
├─ real_world_demo.py       # Real-world prediction demo and heatmaps
├─ real_world_demo_heatmaps.png #generated image from rel_world_demo.py
└─ README.md

````

---

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd Spatio_Case_study
````

2. Install dependencies:

```bash
pip install numpy pandas scipy torch matplotlib seaborn tqdm
```

---

## Usage

### 1. Build adjacency matrix

```bash
python3 build_adj_matrix.py \
    --data_dir data/nyc \
    --lookup_dir data/nyc_raw \
    --output_file data/nyc/adj_matrix.npy
```

### 2. Train the model

```bash
python3 train.py \
    --data_dir data/nyc \
    --lookup_file data/nyc_raw/taxi_zone_lookup.csv \
    --adj_file data/nyc/adj_matrix.npy \
    --epochs 5 \
    --batch_size 128
```

### 3. Evaluate predictions

```bash
python3 evaluate.py
```

### 4. Real-world demonstration

```bash
python3 real_world_demo.py
```

* Generates a heatmap comparing true vs predicted demand for all zones.
* Heatmaps saved as `real_world_demo_heatmaps.png`.

---

## Evaluation

* Prints true and predicted demand for the first 10 zones.
* Generates heatmaps for all zones.
* Loss decreases over epochs during training, demonstrating learning.

---

## Real-World Demo

The `real_world_demo.py` script simulates a real-world case by:

1. Selecting the latest sequence of taxi demand data.
2. Predicting demand for the next time step.
3. Visualizing predictions and ground truth across all zones.

This can help taxi operators or city planners **anticipate demand** in different NYC neighborhoods.

---

## Future Work

* Extend the model to **multi-step forecasting** for future demand.
* Replace the feedforward network with **graph neural networks** for improved spatial modeling.
* Include **weather, events, or traffic data** to improve prediction accuracy.
* Develop an **interactive dashboard** to visualize predictions in real-time.

---

## References

1. NYC Taxi & Limousine Commission (TLC) Trip Record Data: [link](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
2. PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
3. SciPy Sparse Matrix Utilities: [https://docs.scipy.org/doc/scipy/reference/sparse.html](https://docs.scipy.org/doc/scipy/reference/sparse.html)

---
