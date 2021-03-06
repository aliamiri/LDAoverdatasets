import os
import pickle

import numpy as np
import pandas as pd


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def calc_list_kl(dict):
    all_kls = np.zeros((indexes.shape[0], indexes.shape[0]))
    keys = dict.keys()
    for i in keys:
        i_index = np.where(indexes[:, 0] == i)
        for j in keys:
            try:
                j_index = np.where(indexes[:, 0] == j)
                kl = KL(dict[i], dict[j])

                all_kls[i_index[0][0]][j_index[0][0]] = kl
                all_kls[j_index[0][0]][i_index[0][0]] = kl
            except:
                print(j)
    return all_kls


data_dir = os.path.expanduser("citeseer")
df = pd.read_csv(os.path.join(data_dir, "citeseer.content"), sep='\t', usecols=[0], header=None, dtype=str)

indexes = np.array(df.values)
#
# with open('deep/citeseer_ntm10_res.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     citeList = pickle.load(filehandle)
#
#     cite_kls = calc_list_kl(citeList)
#     with open('citeseer_ntm10.data', 'wb') as filehandle:
#         # store the data as binary data stream
#         pickle.dump(cite_kls, filehandle)
#
# with open('deep/citeseer_gsm10_res.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     citeList = pickle.load(filehandle)
#
#     cite_kls = calc_list_kl(citeList)
#     with open('citeseer_gsm10.data', 'wb') as filehandle:
#         # store the data as binary data stream
#         pickle.dump(cite_kls, filehandle)
#
# with open('deep/citeseer_nvdm10_res.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     citeList = pickle.load(filehandle)
#
#     cite_kls = calc_list_kl(citeList)
#     with open('citeseer_nvdm10.data', 'wb') as filehandle:
#         # store the data as binary data stream
#         pickle.dump(cite_kls, filehandle)


# with open('deep/citeseer_ntm20_res.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     citeList = pickle.load(filehandle)
#
#     cite_kls = calc_list_kl(citeList)
#     with open('citeseer_ntm20.data', 'wb') as filehandle:
#         # store the data as binary data stream
#         pickle.dump(cite_kls, filehandle)

# with open('deep/cora_gsm20_res.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     citeList = pickle.load(filehandle)
#
#     cite_kls = calc_list_kl(citeList)
#     with open('citeseer_gsm20.data', 'wb') as filehandle:
#         # store the data as binary data stream
#         pickle.dump(cite_kls, filehandle)

with open('deep/citeseer_nvdm20_res.data', 'rb') as filehandle:
    # read the data as binary data stream
    citeList = pickle.load(filehandle)

    cite_kls = calc_list_kl(citeList)
    with open('citeseer_nvdm20.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(cite_kls, filehandle)
