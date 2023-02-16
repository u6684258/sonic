import numpy as np
from sklearn.preprocessing import normalize

nir = np.load("data/nir.npy").astype("int")
swir = np.load("data/swir.npy").astype("int")
nir = nir - nir.min()
swir = swir - swir.min()
size = 8
sample_f = nir[50,:size]
sample_l = swir[50,:size]
a1 = nir[100,:size]
b1 = swir[100,:size]

np.save("input/inputData/sample_f.npy", sample_f)
np.save("input/inputData/sample_l.npy", sample_l)

np.save("input/inputData/a0.npy", sample_f)
np.save("input/inputData/b0.npy", sample_l)
np.save("input/inputData/a1.npy", a1)
np.save("input/inputData/b1.npy", b1)

print("Data request successful")



