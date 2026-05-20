# gce_mask

This repository contains the masks used for Galactic Center Gamma-ray Excess (GCE) analysis in [Zhong & Cholis (2024)](https://arxiv.org/abs/2401.02481). 

## Masks

The following masks are included:

- `mask_4FGL-DR1_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR1 + L20 mask
- `mask_4FGL-DR1_large_14_Ebin_20x20window_normal.npy`: Large 4FGLDR1 + L20 mask
- `mask_4FGL-DR1_small_14_Ebin_20x20window_normal.npy`: Small 4FGLDR1 + L20 mask
- `mask_4FGL-DR2_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR2 + L20 mask
- `mask_4FGL-DR2_large_14_Ebin_20x20window_normal.npy`: Large 4FGLDR2 + L20 mask
- `mask_4FGL-DR2_small_14_Ebin_20x20window_normal.npy`: Small 4FGLDR2 + L20 mask
- `mask_4FGL-DR2_L5_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR2 + L5 mask
- `mask_4FGL-DR2_L8_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR2 + L8 mask
- `mask_4FGL-DR2_S2_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR2 + Wavelet S2 mask
- `mask_4FGL-DR2_S3_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR2 + Wavelet S3 mask
- `mask_4FGL-DR2_S4_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR2 + Wavelet S4 mask
- `mask_4FGL-DR3_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR3 + L20 mask
- `mask_4FGL-DR3_large_14_Ebin_20x20window_normal.npy`: Large 4FGLDR3 + L20 mask
- `mask_4FGL-DR3_small_14_Ebin_20x20window_normal.npy`: Small 4FGLDR3 + L20 mask
- `mask_4FGL-DR3_L5_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR3 + L5 mask
- `mask_4FGL-DR3_L8_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR3 + L8 mask
- `mask_4FGL-DR3_S2_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR3 + Wavelet S2 mask
- `mask_4FGL-DR3_S3_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR3 + Wavelet S3 mask
- `mask_4FGL-DR3_S4_14_Ebin_20x20window_normal.npy`: Standard 4FGLDR3 + Wavelet S4 mask

All masks cover the region of interest (ROI) defined as l: [+20, -20] deg and b: [-20, +20] deg with Cartesian pixels of size 0.1 deg x 0.1 deg. Each mask is defined in 14 energy bins as listed in Table 1 of [Zhong & Cholis (2024)](https://arxiv.org/abs/2401.02481). Therefore, their dimensions are (14, 400, 400).

## Usage

Please acknowledge [Zhong & Cholis (2024)](https://arxiv.org/abs/2401.02481) if you used the repository masks.

For questions or issues, please contact Ilias Cholis (cholis@oakland.edu) or Yiming Zhong (yiming.zhong@cityu.edu.hk).
