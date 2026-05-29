import numpy as np
path = "/home/seokjun/taylor-root-prediction/data/taylor_data_physchem_v4_deg50/taylor_deg50_test.npz"
data = np.load(path)
for key in data.files:
    print(data[key].shape)
print(data["coeffs"].shape[0])