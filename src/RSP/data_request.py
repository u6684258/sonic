import numpy as np
from sklearn.preprocessing import normalize

nir = np.load("data/nir.npy").astype("int")
swir = np.load("data/swir.npy").astype("int")
nir = nir - nir.min()
swir = swir - swir.min()

sample_f = nir[50,:2]
sample_l = swir[50,:2]

np.save("input/inputData/sample_f.npy", sample_f)
np.save("input/inputData/sample_l.npy", sample_l)
print("Data request successful")



