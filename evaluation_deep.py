import pickle
import numpy as np

with open('cora_dist.data', 'rb') as filehandle:
    # read the data as binary data stream
    dist_list = pickle.load(filehandle)
#
# with open('citeseer_ntm10.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     kl_list = pickle.load(filehandle)
#
# sum_dist = 0
# for i in range(0, dist_list.shape[0]):
#     for j in range(0, dist_list.shape[0]):
#         dist = dist_list[i][j]
#         kl = kl_list[i][j]
#         if dist != 0:
#             if dist != np.inf:
#                 sum_dist += (1 / dist) * kl
#             else:
#                 sum_dist += (1 / 100) * kl
#
# print("citeseer ntm 10")
# print(sum_dist / (dist_list.shape[0] * dist_list.shape[0]))
#

with open('cora_nvdm20.data', 'rb') as filehandle:
    # read the data as binary data stream
    kl_list = pickle.load(filehandle)

sum_dist = 0
for i in range(0, dist_list.shape[0]):
    for j in range(0, dist_list.shape[0]):
        dist = dist_list[i][j]
        kl = kl_list[i][j]
        if dist != 0:
            if dist != np.inf:
                sum_dist += (1 / dist) * kl
            else:
                sum_dist += (1 / 100) * kl

print("citeseer ntm 20")
print(sum_dist / (dist_list.shape[0] * dist_list.shape[0]))
#
#
# with open('citeseer_nvdm10.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     kl_list = pickle.load(filehandle)
#
# sum_dist = 0
# for i in range(0, dist_list.shape[0]):
#     for j in range(0, dist_list.shape[0]):
#         dist = dist_list[i][j]
#         kl = kl_list[i][j]
#         if dist != 0:
#             if dist != np.inf:
#                 sum_dist += (1 / dist) * kl
#             else:
#                 sum_dist += (1 / 100) * kl
#
# print("citeseer nvdm 10")
# print(sum_dist / (dist_list.shape[0] * dist_list.shape[0]))
#
#
#
#
#
# with open('citeseer_ntm15.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     kl_list = pickle.load(filehandle)
#
# sum_dist = 0
# for i in range(0, dist_list.shape[0]):
#     for j in range(0, dist_list.shape[0]):
#         dist = dist_list[i][j]
#         kl = kl_list[i][j]
#         if dist != 0:
#             if dist != np.inf:
#                 sum_dist += (1 / dist) * kl
#             else:
#                 sum_dist += (1 / 100) * kl
#
# print("citeseer ntm 15")
# print(sum_dist / (dist_list.shape[0] * dist_list.shape[0]))
#
#
# with open('citeseer_gsm15.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     kl_list = pickle.load(filehandle)
#
# sum_dist = 0
# for i in range(0, dist_list.shape[0]):
#     for j in range(0, dist_list.shape[0]):
#         dist = dist_list[i][j]
#         kl = kl_list[i][j]
#         if dist != 0:
#             if dist != np.inf:
#                 sum_dist += (1 / dist) * kl
#             else:
#                 sum_dist += (1 / 100) * kl
#
# print("citeseer gsm 15")
# print(sum_dist / (dist_list.shape[0] * dist_list.shape[0]))
#
#
# with open('citeseer_nvdm15.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     kl_list = pickle.load(filehandle)
#
# sum_dist = 0
# for i in range(0, dist_list.shape[0]):
#     for j in range(0, dist_list.shape[0]):
#         dist = dist_list[i][j]
#         kl = kl_list[i][j]
#         if dist != 0:
#             if dist != np.inf:
#                 sum_dist += (1 / dist) * kl
#             else:
#                 sum_dist += (1 / 100) * kl
#
# print("citeseer nvdm 15")
# print(sum_dist / (dist_list.shape[0] * dist_list.shape[0]))
