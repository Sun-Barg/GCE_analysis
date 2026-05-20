Filename "NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat" contains the name convention between the paper, the name of the models in the Maps and the GALDEF filenames

Folder "GALDEF_FILES" contains the 80 galdef files that were used as input in the WebRun simulations of the Milky Way 

Folder "GALACTIC_DIFFUSE_EMISSION_MAPS_0p25deg" contains the Pi0, Bremss(trahlung) and ICS emission maps in 0.25 deg Cartesian pixelization. We used 38 energy bins for the inner 60 deg x 60 deg
The energy bins are logarithmically spaced by a factor of 1.2996566. The first bin starts at 0.0438587 GeV (it is centered at 0.050 GeV). See specific list below.

pi0_XX_Map_flux_E_50-814008_MeV_InnerGalaxy_60x60.fits
bremss_XX_Map_flux_E_50-814008_MeV_InnerGalaxy_60x60.fits
ICS_XX_Map_flux_E_50-814008_MeV_InnerGalaxy_60x60.fits

Units are in GeV cm^-2 s^-1 sr^-1 i.e. E^2*dPhi/dE

Exposure file (in the same folder with the other maps):
"ExposureDEnergyMapsInnerGalaxy_01degBins_60x60_E_50-814009_MeV_Cartesian.fits".  

In units of cm^2 s sr / GeV . The per GeV comes from the given energy bin width for each of the 38 energy bins of the parameterization described above.  
NOTE: the exposure is in 0.1 deg x 0.1 deg bins!

Energy binning Original:
#label, Emin [MeV], Ectr [MeV], Emax [MeV] 
0 43.8587 50.0 57.0013 
1 57.0013 64.98283 74.082 
2 74.082 84.4553638962 96.2812 
3 96.2812 109.762971093 125.133 
4 125.133 142.654169817 162.629 
5 162.629 185.40143332 211.362 
6 211.362 240.958196464 274.698 
7 274.698 313.162910358 357.014 
8 357.014 407.004243322 463.995 
9 463.995 528.965751061 603.034 
10 603.034 687.473829541 783.737 
11 783.737 893.47989989 1018.59 
12 1018.59 1161.21704886 1323.82 
13 1323.82 1509.18340158 1720.51 
14 1720.51 1961.42016848 2236.07 
15 2236.07 2549.17266733 2906.12 
16 2906.12 3313.04908164 3776.96 
17 3776.96 4305.82610508 4908.75 
18 4908.75 5596.09531592 6379.69 
19 6379.69 7273.00221156 8291.4 
20 8291.4 9452.40532607 10776.0 
21 10776.0 12284.8809679 14005.1 
22 14005.1 15966.1266302 18201.8 
23 18201.8 20750.4818513 23656.1 
24 23656.1 26968.5006912 30744.8 
25 30744.8 35049.7899155 39957.6 
26 39957.6 45552.6907923 51931.2 
27 51931.2 59202.8552359 67492.7 
28 67492.7 76943.3815462 87717.4 
29 87717.4 99999.9736528 114002.0 
30 114002.0 129965.625758 148164.0 
31 148164.0 168910.683289 192562.0 
32 192562.0 219525.884347 250265.0 
33 250265.0 285308.264463 325258.0 
34 325258.0 370802.768944 422724.0 
35 422724.0 481916.265956 549396.0 
36 549396.0 626325.655697 714027.0 
37 714027.0 814008.272176 927989.0

#For the paper "The Return of the Templates: Revisiting the Galactic Center Excess with Multi-Messenger Observations", we select then and group 38 ebin into 14 ebin as shown in the PRL 2020 Zhong et al. SM table 1 and in Table III of the "The Return of the Templates: Revisiting the Galactic Center Excess with Multi-Messenger Observations" Cholis, Zhong, McDermott, Surdutovich. 
#label under 14-ebin setup, label under 38-ebin setup, Emin [MeV], Ectr [MeV], Emax[MeV]
0 7 274.698 313.162910358 357.014 
1 8 357.014 407.004243322 463.995 
2 9 463.995 528.965751061 603.034 
3 10 603.034 687.473829541 783.737 
4 11 783.737 893.47989989 1018.59 
5 12 1018.59 1161.21704886 1323.82 
6 13 1323.82 1509.18340158 1720.51 
7 14 1720.51 1961.42016848 2236.07 
8 15 2236.07 2549.17266733 2906.12 
9 16 2906.12 3313.04908164 3776.96 
10 17 3776.96 4305.82610508 4908.75 
11 18,19,20 4908.75 7273.00221156 10776.0 
12 21,22,23 10776.0 15966.1266302 23656.1
13 24,25,26 23656.1 35049.7899155 51931.2 

Folder "Covariance_Matrix_Information" contains files for the calulation of the covariance matrix. Read README file inside 

Filename "GCE_Models_LogLikelihoods_2021_DMprofiles_October_GCE_vs_Background.dat" contains the log-likelihoods (LogL) for alterntive GCE cuspinness i.e. $\gamma$ parameter of our paper, after performing the template fit in the 40x40 deg^2 region. In Figure 13(left) of published paper we show the 2*Delta_LogL . Third column  "LogL(back_only)" provides the log-likelihood without the GCE component.  

Folder "Figures_12_and_14_GCE_Spectra" provides the GCE spectral information from each model that is plotted in Figures 12 and 14. Each file has four columns : E(GeV), E^2 * dPhi/dE (GeV/cm^2/s/sr)_best_fit,  E^2 * dPhi/dE (GeV/cm^2/s/sr)_1sigma_low, E^2 * dPhi/dE (GeV/cm^2/s/sr)_1sigma_high. 1 sigma low and 1 sigmna high values refer to the actuall template fit errors, not the correlated errors not Eq 18 of paper. For Model XLIX these are the statistical errors shown in Figure 19 of published PRD paper.
Files:
GCE_BestFitModel_North_flux_Inner40x40_masked_disk.dat
GCE_BestFitModel_South_flux_Inner40x40_masked_disk.dat
GCE_BestFitModel_flux_Inner40x40_masked_disk.dat

GCE_2ndBestFitModel_North_flux_Inner40x40_masked_disk.dat
GCE_2ndBestFitModel_South_flux_Inner40x40_masked_disk.dat
GCE_2ndBestFitModel_flux_Inner40x40_masked_disk.dat

etc, refer to the models that provide best-fit (2nd best fit etc) in the respective windows (see paper for deatils).
