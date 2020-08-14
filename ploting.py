import seaborn as sns;

sns.set()

import matplotlib.pyplot as plt
import numpy as np

cite_cora_lda = [0.015244, 0.020638, 0.028296]
cite_cora_ntm = [0.000741, 0.000696, 0]
cite_cora_gsm = [0.000005, 0.000006, 0.00001]
cite_cora_nvmd = [0.133828, 0.132662, 0.1258]

cite_citeseer_lda = [0.008418, 0.011122, 0.019936]
cite_citeseer_ntm = [0.000963, 0.000796, 0.000462]
cite_citeseer_gsm = [0.000009, 0.000005, 0]
cite_citeseer_nvmd = [0.081616, 0.078039, 0.070912]

prep_cora_lda = [861.27, 915.07, 907.07]
prep_cora_ntm = []
prep_cora_gsm = []
prep_cora_nvmd = []

prep_citeseer_lda = [746.84, 800.27, 822.85]
prep_citeseer_ntm = []
prep_citeseer_gsm = []
prep_citeseer_nvmd = []

x = np.array([10, 15, 20])

plt.plot(x, cite_citeseer_lda, label="lda")
plt.plot(x, cite_citeseer_ntm, label="ntm")
plt.plot(x, cite_citeseer_gsm, label="gsm")
plt.plot(x, cite_citeseer_nvmd, label="nvdm")

# plt.plot(x, cite_cora_lda, label="lda")
# plt.plot(x, cite_cora_ntm, label="ntm")
# plt.plot(x, cite_cora_gsm, label="gsm")
# plt.plot(x, cite_cora_nvmd, label="nvdm")
plt.legend()
plt.savefig("citeseer_citation.png")
# plt.savefig("cora_citation.png")
