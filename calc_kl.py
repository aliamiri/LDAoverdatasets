import pickle
import numpy as np


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def calc_list_kl(list):
    all_kls = np.zeros((len(list), len(list)))
    for i in range(0, len(list) - 1):
        for j in range(i, len(list)):
            kl = KL(list[i], list[j])
            all_kls[i][j] = kl
            all_kls[j][i] = kl
    return all_kls


# with open('HepTh.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     HepThlist = pickle.load(filehandle)
#
#     hep_kls = calc_list_kl(HepThlist)
#     with open('hep_kls.data', 'wb') as filehandle:
#         # store the data as binary data stream
#         pickle.dump(hep_kls, filehandle)

with open('cora_10.data', 'rb') as filehandle:
    # read the data as binary data stream
    coraList = pickle.load(filehandle)

    cora_kls = calc_list_kl(coraList)
    with open('cora_kls_10.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(cora_kls, filehandle)

with open('cora_20.data', 'rb') as filehandle:
    # read the data as binary data stream
    coraList = pickle.load(filehandle)

    cora_kls = calc_list_kl(coraList)
    with open('cora_kls_20.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(cora_kls, filehandle)

with open('cora_15.data', 'rb') as filehandle:
    # read the data as binary data stream
    coraList = pickle.load(filehandle)

    cora_kls = calc_list_kl(coraList)
    with open('cora_kls_15.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(cora_kls, filehandle)
with open('citeseer_10.data', 'rb') as filehandle:
    # read the data as binary data stream
    citeList = pickle.load(filehandle)

    cite_kls = calc_list_kl(citeList)
    with open('citeseer_kls_10.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(cite_kls, filehandle)


with open('citeseer_15.data', 'rb') as filehandle:
    # read the data as binary data stream
    citeList = pickle.load(filehandle)

    cite_kls = calc_list_kl(citeList)
    with open('citeseer_kls_15.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(cite_kls, filehandle)


with open('citeseer_20.data', 'rb') as filehandle:
    # read the data as binary data stream
    citeList = pickle.load(filehandle)

    cite_kls = calc_list_kl(citeList)
    with open('citeseer_kls_20.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(cite_kls, filehandle)