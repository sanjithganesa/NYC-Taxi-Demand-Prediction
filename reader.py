import numpy as np
import scipy.sparse as sp

file = "data/nyc/yellow_tripdata_2019-02_sparse.npz"
npz = np.load(file)
print("Keys inside:", list(npz.keys()))
for k in npz.files:
    print(k, npz[k].shape, npz[k].dtype)
