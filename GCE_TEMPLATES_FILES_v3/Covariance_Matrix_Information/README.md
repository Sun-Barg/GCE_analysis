File "Truncated_Systematic_Errors_Covariance_Matrix" contains a quick code to evaluate the truncated systematic errors covariance matrix, Σ_ij,mod^trunc. That is Eq. 17 from "The Return of the Templates: Revisiting the Galactic Center Excess with Multi-Messenger Observations", arXiv:2112.09706. 

File "cov_mat_21Dec02.npy" contains the full systematic errors covaiance matrix Σ_ij,mod.

File "GCE_Statistical_errors.dat" contains the statistical errors sigma_i. To calculate the proper  
covariance matrix C_ij used in the fits as Eq. 18 of same paper you need to calculate:
C_ij= σ_i^2*δ_ij +Σij,mod^trunc.
