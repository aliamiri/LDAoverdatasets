docs = list()
import pickle

import os

import numpy as np
import pandas as pd

data_dir = os.path.expanduser("cora")
df = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', usecols=[0], header=None)

indexes = np.array(df.values)
cite_data = np.array(pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None).values)

all_dists = np.ones((indexes.shape[0], indexes.shape[0])) * np.inf

for i in range(0, indexes.shape[0]):
    all_dists[i, i] = 0

for data in cite_data:
    try:
        ind0 = np.where(indexes[:, 0] == data[0])
        ind1 = np.where(indexes[:, 0] == data[1])
        all_dists[ind0[0][0], ind1[0][0]] = 1
        all_dists[ind1[0][0], ind0[0][0]] = 1
    except:
        pass

# Floyd Warshall
for k in range(0, indexes.shape[0]):
    print(k)
    for i in range(0, indexes.shape[0]):
        for j in range(0, indexes.shape[0]):
            all_dists[i, j] = min(all_dists[i, j], all_dists[i, k] + all_dists[k, j])

with open('cora_dist.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(all_dists, filehandle)
print("sala,")
