##################This gives the Truncated Systematic Errors Covariance Matrix using the first three principal components of "The Return of the Templates: Revisiting the Galactic Center Excess with Multi-Messenger Observations". Look at Eq. 17.
import numpy as np
corr_cov = np.load("cov_mat_21Dec02.npy")#load the full systematic errors covariance matrix
corr_covsvd = np.linalg.svd(corr_cov)

cov_PCs = np.array([corr_covsvd[1][i]*outer(corr_covsvd[0].T[i], corr_covsvd[2][i]) for i in range(14)])

def approxcovmatrix(p):
    out = np.zeros((14,14))
    if ((type(p)==int) & (p>0)):
        out += np.sum([cov_PCs[i] for i in range(p)], axis=0)
    return out
###################
