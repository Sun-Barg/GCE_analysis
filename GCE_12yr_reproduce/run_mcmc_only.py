#!/usr/bin/env python3
"""
Stage 2: emcee MCMC on pre-computed gtmodel output FITS files.

This script runs in a completely fresh Python process that does NOT import
Fermi ScienceTools (GtApp). This avoids the segfault/hang issues observed
when emcee is run in the same process as gtsrcmaps/gtmodel.

Input (per model):
  - ./GC_analysis_sanghwan/GC_{pion,bremss,ics}_model{M}_12yr_front_clean{,_no_convol}.fits
  - ./GC_analysis_sanghwan/GC_{GCE,fermi_bubble,isotropic}_model_12yr_front_clean{,_no_convol}.fits
  - ./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits (data)
  - ./GC_analysis_sanghwan/GC_expcube_center_12yr_front_clean.fits
  - ./GC_analysis_sanghwan/Model/GC_disk_mask_60x60_definitions.npy
  - ./GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2.npy
  - Local isotropic/bubble spectrum .txt files

Output (per model):
  - ./GCE_model_{M}_12yr_cholis.dat (E, flux, err, lower, upper)
  - ./GCE_model_{M}_12yr_cholis_likelihood_value

Usage:
    python run_mcmc_only.py MODEL_NAME
    python run_mcmc_only.py X         # for model X only
    python run_mcmc_only.py all       # for all models in MODEL_LIST below

This script uses NO multiprocessing.Pool (emcee runs serially) to avoid 
fork-related issues.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")

# === Standard scientific stack ONLY. No Fermi tools! ===
import matplotlib
matplotlib.use("Agg")  # No display needed in subprocess
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.integrate import dblquad
import emcee
from chainconsumer import ChainConsumer
from multiprocessing import Pool
import time

# Number of parallel workers for emcee sampling.
# Stage 2 does NOT import Fermi tools, so fork-based Pool is safe here.
# 4 workers × 14 bins (sequential) uses 4 cores.
# Increase if server has spare cores and other users aren't affected.
N_MCMC_WORKERS = 4

MODEL_LIST_DEFAULT = ["X", "XLIX", "I", "IV", "V", "VI", "VII", "IX",
                      "XV", "XLI", "XLVII", "XLVIII", "L", "LII"]

def run_one_model(model):
    """Run full emcee MCMC for one model."""
    print(f"\n{'='*60}", flush=True)
    print(f"==== Stage 2 MCMC for MODEL {model} ====", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Check that required input files exist before starting
    required = [
        f'./GC_analysis_sanghwan/GC_pion_model{model}_12yr_front_clean.fits',
        f'./GC_analysis_sanghwan/GC_bremss_model{model}_12yr_front_clean.fits',
        f'./GC_analysis_sanghwan/GC_ics_model{model}_12yr_front_clean.fits',
        f'./GC_analysis_sanghwan/GC_pion_model{model}_12yr_front_clean_no_convol.fits',
        f'./GC_analysis_sanghwan/GC_bremss_model{model}_12yr_front_clean_no_convol.fits',
        f'./GC_analysis_sanghwan/GC_ics_model{model}_12yr_front_clean_no_convol.fits',
        './GC_analysis_sanghwan/GC_GCE_model_12yr_front_clean.fits',
        './GC_analysis_sanghwan/GC_fermi_bubble_model_12yr_front_clean.fits',
        './GC_analysis_sanghwan/GC_isotropic_model_12yr_front_clean.fits',
        './GC_analysis_sanghwan/GC_GCE_model_12yr_front_clean_no_convol.fits',
        './GC_analysis_sanghwan/GC_fermi_bubble_model_12yr_front_clean_no_convol.fits',
        './GC_analysis_sanghwan/GC_isotropic_model_12yr_front_clean_no_convol.fits',
        './GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits',
        './GC_analysis_sanghwan/GC_expcube_center_12yr_front_clean.fits',
        './GC_analysis_sanghwan/Model/GC_disk_mask_60x60_definitions.npy',
        './GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2.npy',
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"[model={model}] ERROR: missing input files:", flush=True)
        for f in missing:
            print(f"  - {f}", flush=True)
        return False

    # === Emcee body (copied from run_main_loop.py, with `model` as function arg) ===

    ## Emcee running part

    def roi_solid_angle(delta_l_deg, delta_b_deg, b_deg):
        # Convert degrees to radians
        delta_l_rad = np.radians(delta_l_deg)
        delta_b_rad = np.radians(delta_b_deg)
        b_rad = np.radians(b_deg)
    
        # Calculate solid angle in steradians
        solid_angle = delta_l_rad * delta_b_rad * np.cos(b_rad)
        return solid_angle

    raw_map=fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')
    w=WCS(raw_map[0].header).dropaxis(2)
    # Define the dimensions of the numpy array
    width, height = np.shape(raw_map[0].data[0])

    # Create the counts map
    steradian_per_pixel=np.zeros([width, height])

    for i in range(0, height, 1):
        for j in range(0, width, 1):
            l, b = w.wcs_pix2world(j, i, 0) #x-axis array - b, y-axis array - l
            steradian_per_pixel[i, j] = roi_solid_angle(0.1, 0.1, b)

    # Revision :: Aug 11, 2024
    disk_mask=np.load('./GC_analysis_sanghwan/Model/GC_disk_mask_60x60_definitions.npy')[100:500, 100:500]
    #psc_mask=np.load('./GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2.npy')[:, 100:500, 100:500]
    #psc_mask=np.load('./GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2.npy')[:, 100:500, 100:500]

    front='_front'
    convol='_no_convol'

    E_bounds=fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')[1].data


    E=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        E[i] = np.sqrt(E_bounds[i][2]*E_bounds[i][1]*1e-6)*1e-3

    delta_E=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        delta_E[i] = (E_bounds[i][2] - E_bounds[i][1])*1e-6

    exp_cube = fits.open(f'./GC_analysis_sanghwan/GC_expcube_center_12yr_front_clean.fits')[0].data[:, 100:500, 100:500]*steradian_per_pixel[100:500, 100:500]



    file_name=f'./GC_analysis_sanghwan/GC_pion_model{model}_12yr_front_clean_no_convol.fits'
    pion=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        pion[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)

    file_name=f'./GC_analysis_sanghwan/GC_bremss_model{model}_12yr_front_clean_no_convol.fits'
    bremss=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        bremss[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
    
    file_name=f'./GC_analysis_sanghwan/GC_ics_model{model}_12yr_front_clean_no_convol.fits'
    ics=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        ics[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
    
    file_name=f'./GC_analysis_sanghwan/GC_GCE_model_12yr_front_clean_no_convol.fits'
    GCE=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        GCE[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/exp_cube[i]) )/np.sum(disk_mask)


    file_name=f'./GC_analysis_sanghwan/GC_fermi_bubble_model_12yr_front_clean_no_convol.fits'
    bubble=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        bubble[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
    
    file_name=f'./GC_analysis_sanghwan/GC_isotropic_model_12yr_front_clean_no_convol.fits'
    isotropic=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        isotropic[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
    
    counts_per_exp=np.zeros(len(E_bounds))
    i=0
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        counts_per_exp[i]=np.sum( disk_mask*( (fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')[0].data[i][100:500, 100:500]) /a) )/np.sum(disk_mask)

    counts_per_exp_err=np.zeros(len(E_bounds))
    i=0
    for i in range(0, len(E_bounds), 1):
        counts_per_exp_err[i]= np.sqrt( np.sum( ( (np.sqrt(disk_mask*fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')[0].data[i][100:500, 100:500]) /exp_cube[i] )**2)) )/np.sum(disk_mask)
        #counts_per_exp_err[i]=  np.sqrt(np.sum(disk_mask*fits.open(f'./GC_analysis_sanghwan/GC_all_time_60x60_ccube{front}_12yr.fits')[0].data[i][100:500, 100:500]))/np.sum(disk_mask*exp_cube[i]) 

    

    # Use scipy.special.gammaln for vectorized log(n!) — orders of magnitude faster
    from scipy.special import gammaln as _gammaln
    def log_factorial(O):
        return _gammaln(O + 1)  # gammaln(n+1) = log(n!)
    # No np.vectorize needed — gammaln is already vectorized in C
    #Constraints interpolated function
    #Contains constraints for bubble and isotropic as well
    #For isotropic, from https://arxiv.org/pdf/1410.3696.pdf Table 3
    #Correcting bubble template given from https://arxiv.org/pdf/1407.7905, Table 2

    bubble_constraints=np.loadtxt('./GC_analysis_sanghwan/Model/bubble_constraints.txt')
    bubble_constraints_energy=bubble_constraints[0:, 0]
    bubble_constraints_flux=bubble_constraints[0:, 1]
    bubble_constraints_lower_error=bubble_constraints[0:, 2]
    bubble_constraints_upper_error=bubble_constraints[0:, 3]

    bubble_fluxint = interp1d((bubble_constraints_energy), (bubble_constraints_flux), fill_value='extrapolate', kind='quadratic')
    bubble_lower_errint = interp1d((bubble_constraints_energy), (bubble_constraints_lower_error), fill_value='extrapolate', kind='quadratic')
    bubble_upper_errint = interp1d((bubble_constraints_energy), (bubble_constraints_upper_error), fill_value='extrapolate', kind='quadratic')

    bubble_flux_data=bubble_fluxint((E))
    bubble_lower_error_data=bubble_lower_errint((E))
    bubble_upper_error_data=bubble_upper_errint((E))


    iso_constraints=np.loadtxt('./GC_analysis_sanghwan/Model/iso_constraints_full_err.txt')
    #iso_constraints=np.loadtxt('./GC_analysis_sanghwan/Model/egb_constraints_full_err.txt')

    iso_constraints_energy=iso_constraints[0:, 0]
    iso_constraints_flux=iso_constraints[0:, 1]
    iso_constraints_low_err=iso_constraints[0:, 2]
    iso_constraints_upp_err=iso_constraints[0:, 3]

    isotropic_fluxint=interp1d(iso_constraints_energy, iso_constraints_flux, fill_value="extrapolate", kind='quadratic')
    isotropic_lower_errint=interp1d(iso_constraints_energy, iso_constraints_low_err, fill_value="extrapolate", kind='quadratic')    
    isotropic_upper_errint=interp1d(iso_constraints_energy, iso_constraints_upp_err, fill_value="extrapolate", kind='quadratic')  

    isotropic_flux_data=((E)**2)*(isotropic_fluxint((E)))
    isotropic_lower_error_data=((E)**2)*(isotropic_lower_errint((E)))
    isotropic_upper_error_data=((E)**2)*(isotropic_upper_errint((E)))

    front='_front'
    class Likelihood:
        def __init__(self, model, energy_bin):
            import sys
            print(f"  [Likelihood __init__] model={model}, energy_bin={energy_bin}", flush=True)
            self.model=model
            self.energy_bin=energy_bin
            self.data=fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.pion_bremss=fits.open(f'./GC_analysis_sanghwan/GC_pion_model{model}_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500] + fits.open(f'./GC_analysis_sanghwan/GC_bremss_model{model}_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]  
            self.ics=fits.open(f'./GC_analysis_sanghwan/GC_ics_model{model}_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.GCE=fits.open(f'./GC_analysis_sanghwan/GC_GCE_model_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.bubble=fits.open(f'./GC_analysis_sanghwan/GC_fermi_bubble_model_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.iso=fits.open(f'./GC_analysis_sanghwan/GC_isotropic_model_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            E_bounds=fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')[1].data

            E=np.zeros(len(E_bounds))
            for i in range(0, len(E_bounds), 1):
                E[i] = np.sqrt(E_bounds[i][2]*E_bounds[i][1]*1e-6)*1e-3
            self.E = E
            delta_E=np.zeros(len(E_bounds))
            for i in range(0, len(E_bounds), 1):
                delta_E[i] = (E_bounds[i][2] - E_bounds[i][1])*1e-6
            self.delta_E = delta_E
            self.exp_cube = (fits.open(f'./GC_analysis_sanghwan/GC_expcube_center_12yr_front_clean.fits')[0].data[self.energy_bin]*steradian_per_pixel)[100:500, 100:500]

        
            psc_mask=np.load('./GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2.npy')[self.energy_bin, 100:500, 100:500]
    
            # Pre-cache no_convol maps for chi2 constraint terms (avoid fits.open in hot loop)
            self._iso_noconv = fits.open('./GC_analysis_sanghwan/GC_isotropic_model_12yr_front_clean_no_convol.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self._bub_noconv = fits.open('./GC_analysis_sanghwan/GC_fermi_bubble_model_12yr_front_clean_no_convol.fits')[0].data[self.energy_bin, 100:500, 100:500]
        
            #psc_mask=np.load('./GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2_corrected.npy')[self.energy_bin, 100:500, 100:500]

            disk_mask=np.load('./GC_analysis_sanghwan/Model/GC_disk_mask_60x60_definitions.npy')[100:500, 100:500]
            full_mask=psc_mask*disk_mask
            self.disk_mask=disk_mask
            self.full_mask=full_mask
            
            # Pre-compute log_factorial for observed data (doesn't change between evaluations)
            _obs_masked = self.data[self.full_mask == 1]
            self._log_fact_obs = log_factorial(_obs_masked)
        def likelihood_constrained(self, parameter_set):
            #####################################
            pion_bremss_param=parameter_set[0]
            ics_param=parameter_set[1]
            GCE_param=parameter_set[2]
            bubble_param=parameter_set[3]
            isotropic_param=parameter_set[4]
            ######################################
            expected_pixel= (pion_bremss_param)*self.pion_bremss + (ics_param)*self.ics + (GCE_param)*self.GCE + (isotropic_param)*self.iso + (bubble_param)*self.bubble   
            observed_pixel = self.data

            observed_pixel = observed_pixel[self.full_mask == 1]
            expected_pixel = expected_pixel[self.full_mask == 1]

        
            if (expected_pixel < 0).any():
                return np.inf
            
            #expected_pixel[expected_pixel == 0.0] += 1e-20
        
            observed_log_expected=observed_pixel*np.log(expected_pixel)
            #nan_index = np.where(np.isnan(observed_log_expected))
            #observed_log_expected[nan_index] = 0
            lhd=2*( expected_pixel - observed_log_expected + self._log_fact_obs )

        
            # Use pre-cached no_convol map (avoid fits.open in hot loop)
            isotropic = np.sum( self.full_mask*(self._iso_noconv)/self.exp_cube )*isotropic_param/np.sum(self.full_mask)

            isotropic_sed = (self.E[self.energy_bin]**2)*isotropic/(self.delta_E[self.energy_bin])


            # Use pre-cached no_convol map (avoid fits.open in hot loop)
            bubble = np.sum( self.full_mask*(self._bub_noconv)/self.exp_cube )*bubble_param/np.sum(self.full_mask)
            bubble_sed = (self.E[self.energy_bin]**2)*bubble/(self.delta_E[self.energy_bin])

            larger_error=max([bubble_upper_error_data[self.energy_bin], bubble_lower_error_data[self.energy_bin]])
            if bubble_flux_data[self.energy_bin] < bubble_sed:
                chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin])/bubble_upper_error_data[self.energy_bin])**2
            if bubble_flux_data[self.energy_bin] > bubble_sed:
                chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin])/bubble_lower_error_data[self.energy_bin])**2
            if bubble_flux_data[self.energy_bin] == bubble_sed:
                chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin])/larger_error)**2


            isotropic_larger_error=max([isotropic_lower_error_data[self.energy_bin], isotropic_upper_error_data[self.energy_bin]])
            if isotropic_flux_data[self.energy_bin] < isotropic_sed:
                chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed)/isotropic_lower_error_data[self.energy_bin])**2
            if isotropic_flux_data[self.energy_bin] > isotropic_sed:
                chi2_isotropic = ((isotropic_flux_data[i] - isotropic_sed)/isotropic_upper_error_data[self.energy_bin])**2
            if isotropic_flux_data[self.energy_bin] == isotropic_sed:
                chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed)/isotropic_larger_error)**2
            #print(chi2_bubble, chi2_isotropic)
            return (np.sum(lhd)  + chi2_bubble + chi2_isotropic)


    # [REMOVED] Sanghwan's sanity test: Likelihood('I', 0).likelihood_constrained(np.ones(5))

    # Define your likelihood function (with Likelihood object caching)
    _likelihood_cache = {}
    def log_likelihood(params, energy_bin):
        key = (model, energy_bin)
        if key not in _likelihood_cache:
            _likelihood_cache[key] = Likelihood(model, energy_bin)
        return -(1/2)*_likelihood_cache[key].likelihood_constrained(params)  # log likelihood -> Need to maximize

    # Define the prior function with parameter limits
    def log_prior(params):
        limits = [
            (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)
            #(0, 2), (0, 2), (-5, 5), (0.09, 0.11), (0, 5)
            #(-3, 3),  # Bounds for param1
            #(-3, 3),  # Bounds for param2
            #(-5, 5),  # Bounds for param3
            #(-3, 3),  # Bounds for param4
            #(-3, 3)   # Bounds for param5
        ]
    
        for i, (lower, upper) in enumerate(limits):
            if not (lower <= params[i] <= upper):
                return -np.inf  # Return negative infinity if outside bounds
        return 0.0  # Return zero if all parameters are within bounds

    # Define the log probability function
    def log_probability(params, energy_bin):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, energy_bin)  # Maximize this
        #return log_likelihood(params, energy_bin)

    def run_mcmc_for_bin(energy_bin):  
        ndim = 5
        nwalkers = 100
        nsteps = 1000
        burn_in_steps = 400
        start_time=time.time()
        # Memory check before starting Pool
        try:
            import resource
            mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
            print(f"  energy_bin={energy_bin}, parent RSS before Pool: {mem_gb:.2f} GB", flush=True)
        except Exception:
            print(f"  energy_bin={energy_bin}", flush=True)
        # === Run emcee SERIALLY (no Pool) ===
        # multiprocessing.Pool with fork-based workers crashes immediately when 
        # the parent has imported Fermi ScienceTools (GtApp). Running emcee serially
        # avoids fork issues entirely. Slower but reliable.
        if True:  # keep block-level scoping similar to original
            initial_params = np.vstack([
                np.random.uniform(0, 3, [nwalkers]),
                np.random.uniform(0, 3, [nwalkers]),
                np.random.uniform(0, 3, [nwalkers]),
                np.random.uniform(0, 10, [nwalkers]),
                np.random.uniform(0, 10, [nwalkers]),
            ]).T
            pos = initial_params
            from emcee.moves import DEMove, KDEMove
            pool = Pool(processes=N_MCMC_WORKERS) if N_MCMC_WORKERS > 1 else None
            # Serial execution (nested functions can't be pickled for multiprocessing)
            # Speed comes from gammaln optimization + fits.open caching instead
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(energy_bin,))
            #for iteration in range(10):    
            print(f"  About to call sampler.run_mcmc(nsteps={nsteps}, walkers={nwalkers})...", flush=True)
            print(f"  Performing one test log_probability call to verify it works...", flush=True)
            _test_lp = log_probability(initial_params[0], energy_bin)
            print(f"  Test log_probability returned: {_test_lp}", flush=True)
            print("Running production... (chunked progress, no tqdm)", flush=True)
        
            # Run in chunks of 100 steps with explicit progress logging.
            # Avoids tqdm which doesn't flush properly in nohup/non-tty environments.
            chunk_size = 100
            current_pos = pos
            t_chunk_start = time.time()
            for chunk_start in range(0, nsteps, chunk_size):
                chunk_n = min(chunk_size, nsteps - chunk_start)
                try:
                    import resource
                    mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
                except Exception:
                    mem_gb = -1
                t0 = time.time()
                state_obj = sampler.run_mcmc(current_pos, chunk_n, progress=False)
                current_pos = state_obj.coords if hasattr(state_obj, 'coords') else state_obj[0]
                t_elapsed = time.time() - t0
                rate = chunk_n / t_elapsed if t_elapsed > 0 else 0
                cum_steps = chunk_start + chunk_n
                eta_sec = (nsteps - cum_steps) / rate if rate > 0 else 0
                print(f"    [bin={energy_bin}] step {cum_steps:>5}/{nsteps}  "
                      f"chunk_time={t_elapsed:.1f}s  rate={rate:.2f} it/s  "
                      f"ETA={eta_sec/60:.1f} min  RSS={mem_gb:.2f} GB", flush=True)
        
            # Reconstruct (pos, prob, state) for downstream code compatibility
            pos = current_pos
            prob = sampler.get_log_prob()[-1]
            state = state_obj
            

            
            # Clean up Pool
            if pool is not None:
                pool.close()
                pool.join()
            #sampler.run_mcmc(pos, nsteps, progress=True)
        
            max_pos = pos[np.argmax(prob)]
            fitted_param = max_pos
            #max_lhd = np.argmax(prob)

  
        
            log_prob_samples = sampler.get_log_prob(discard=burn_in_steps, flat=True)

            max_prob_index = np.argmax(log_prob_samples)
            max_lhd = log_prob_samples[max_prob_index]

            best_fit_params = sampler.get_chain(discard=burn_in_steps, flat=True)[max_prob_index]        

            fitted_param = best_fit_params
        
            flat_samples = sampler.get_chain(discard=burn_in_steps, flat=True)

            lower_1sigma = np.percentile(flat_samples, 16, axis=0)
            upper_1sigma = np.percentile(flat_samples, 84, axis=0)

            #for i in range(ndim):
            # Calculate the 16th, 50th, and 84th percentiles for the i-th parameter
                #mcmc = np.percentile(flat_samples[:, i], [50])
                #fitted_param[i] = mcmc[0]
                #print(mcmc[0])

            print(max_pos)#, best_fit_params, fitted_param)

        

            #fitted_param = best_fit_params
        
            # Get only the samples from the current iteration
            samples = sampler.get_chain(discard=burn_in_steps, thin=1, flat=False)
            current_samples = samples[-nsteps:]  # Get only the last `nsteps` samples
    
            print("Max position:", fitted_param)
            # Trace Plot for Each Walker and Parameter
            fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            for i in range(ndim):
                ax = axes[i]
                for walker in range(nwalkers):
                    ax.plot(current_samples[:, walker, i], alpha=0.3)  # Plot each walker separately
                ax.set_xlim(0, nsteps)
                ax.set_ylim(-3, 20)
                ax.set_ylabel(f"param{i+1}")
                ax.yaxis.set_label_coords(-0.1, 0.5)
    
            axes[-1].set_xlabel("step number")
            plt.suptitle(f"Trace Plot for Each Walker after iteration")
            plt.show()
    
            # Final Corner Plot with ChainConsumer (wrapped in try/except for API compatibility)
            flat_samples = sampler.get_chain(discard=burn_in_steps, thin=1, flat=True)
            try:
                c = ChainConsumer()
                c.add_chain(flat_samples, parameters=["param1", "param2", "param3", "param4", "param5"])
                fig = c.plotter.plot(figsize=(6, 6))
                plt.savefig(f'./GC_analysis_sanghwan/corner_{model}_bin{energy_bin}.png', dpi=100)
                plt.close()
            except Exception as _e:
                print(f"  [ChainConsumer] skipped corner plot: {_e}", flush=True)
            print("std", np.std(flat_samples, axis=0, ddof=1))
        end_time=time.time()
        #print(f"{np.round((end_time-start_time)/(60*60), 5)}hours")
        print(fitted_param, np.median(flat_samples, axis=0))
        return fitted_param.T, np.median(flat_samples, axis=0).T, np.std(flat_samples, axis=0, ddof=1).T, max_lhd, upper_1sigma, lower_1sigma
        #return np.median(samples, axis=0), np.std(samples, axis=0)

    n=len(E)
    fitted_params=np.ones([n*5])
    fitted_params_median=np.ones([n*5])
    fitted_params_std = np.zeros([n*5])
    max_likelihood = np.zeros([n])

    fitted_params_upper = np.zeros([n*5])
    fitted_params_lower = np.zeros([n*5])
    for i in range(0, n, 1):
        max_value, median_value, std_value, maximum_value, upper_value, lower_value = run_mcmc_for_bin(i)
        fitted_params[n*0:n*1][i] = max_value[0]
        fitted_params[n*1:n*2][i] = max_value[1]
        fitted_params[n*2:n*3][i] = max_value[2]
        fitted_params[n*3:n*4][i] = max_value[3]
        fitted_params[n*4:n*5][i] = max_value[4]

        fitted_params_std[n*0:n*1][i] = std_value[0]
        fitted_params_std[n*1:n*2][i] = std_value[1]
        fitted_params_std[n*2:n*3][i] = std_value[2]
        fitted_params_std[n*3:n*4][i] = std_value[3]
        fitted_params_std[n*4:n*5][i] = std_value[4]

        fitted_params_median[n*0:n*1][i] = median_value[0]
        fitted_params_median[n*1:n*2][i] = median_value[1]
        fitted_params_median[n*2:n*3][i] = median_value[2]
        fitted_params_median[n*3:n*4][i] = median_value[3]
        fitted_params_median[n*4:n*5][i] = median_value[4]

        fitted_params_upper[n*0:n*1][i] = upper_value[0]
        fitted_params_upper[n*1:n*2][i] = upper_value[1]
        fitted_params_upper[n*2:n*3][i] = upper_value[2]
        fitted_params_upper[n*3:n*4][i] = upper_value[3]
        fitted_params_upper[n*4:n*5][i] = upper_value[4]


        fitted_params_lower[n*0:n*1][i] = lower_value[0]
        fitted_params_lower[n*1:n*2][i] = lower_value[1]
        fitted_params_lower[n*2:n*3][i] = lower_value[2]
        fitted_params_lower[n*3:n*4][i] = lower_value[3]
        fitted_params_lower[n*4:n*5][i] = lower_value[4]


        max_likelihood[i] = maximum_value
    
        plt.style.use('default')
        ax=plt.subplot()
    
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('E [GeV]')
        ax.set_ylabel(r'$E^2 \frac{dN}{dE}$[GeV$cm^{-2}$$s^{-1} sr^{-1}$]')
    
        ax.set_ylim(1e-8, 1e-4)
        ax.set_xlim(0.3, 500)
    
        ax.tick_params(axis='y', which='both', direction='in', left=True)
        ax.tick_params(axis='x', which='both', direction='in', bottom=True)
        ax.minorticks_on()
        ax.grid(True, which='Major', linestyle='-', linewidth=0.5)
    
        fitted=fitted_params#fitted_params
        fitted_errors=fitted_params_std
        #sr=0.4288213187542626#0.214411*2
        sr=1
        #sr=0.4387
        #sr=0.4776

        ax.errorbar(E, counts_per_exp*(E**2)/(delta_E*sr) , yerr=counts_per_exp_err*(E**2)/(delta_E*sr), linestyle='dotted', marker='.', elinewidth=2, capsize=4, capthick=2, label='Raw_data')
    
    
    
    
        ax.errorbar(E, fitted[n*0:n*1]*(pion+bremss)*(E**2)/(delta_E*sr), yerr=fitted_errors[n*0:n*1]*(pion+bremss)*(E**2)/(delta_E*sr), linestyle='dotted', marker='.', elinewidth=2, capsize=4, capthick=2, label='pion+bremss', color='red')
        ax.errorbar(E, fitted[n*1:n*2]*(ics)*(E**2)/(delta_E*sr), yerr=fitted_errors[n*1:n*2]*(ics)*(E**2)/(delta_E*sr), linestyle='dashdot', marker='.', elinewidth=2, capsize=4, capthick=2, label='ics', color='blue')
    
        ax.plot(E, (pion+bremss)*(E**2)/(delta_E*sr), linestyle='solid', label='pion+bremss', color='red')
        ax.plot(E, (ics)*(E**2)/(delta_E*sr), linestyle='solid', label='ics', color='blue')
    
    
    
        ax.errorbar(E,  fitted[n*2:n*3]*(GCE)*(E**2)/(delta_E*sr), yerr=np.sqrt((fitted_errors[n*2:n*3]*GCE)**2)*(E**2)/(delta_E*sr), alpha=0.1, linestyle='dashed', marker='.', elinewidth=2, capsize=4, capthick=2, label='GCE', color='black')
    
    
    
        ax.errorbar(E, fitted[n*3:n*4]*(bubble)*(E**2)/(delta_E*sr),yerr=fitted_errors[n*3:n*4]*(bubble)*(E**2)/(delta_E*sr), linestyle='dashed', marker='.', elinewidth=2, capsize=4, capthick=2, label='bubble', color='purple')
    
        ax.errorbar(E, fitted[n*4:n*5]*(isotropic)*(E**2)/(delta_E*sr),yerr=fitted_errors[n*4:n*5]*(isotropic)*(E**2)/(delta_E*sr), linestyle='dashed', marker='.', elinewidth=2, capsize=4, capthick=2, label='isotropic', color='green')
        summed = fitted[n*0:n*1]*(pion+bremss) + fitted[n*1:n*2]*(ics) + fitted[n*2:n*3]*(GCE) + fitted[n*3:n*4]*(bubble) + fitted[n*4:n*5]*(isotropic)

        ax.plot(E, (E**2)*summed/(delta_E*sr), label='summed')
    
    
        plt.show()

        print((fitted[n*2:n*3]*(GCE)*(E**2)/(delta_E*sr))[i])
        print(((fitted_params_upper[n*2:n*3] - fitted[n*2:n*3])*(GCE)*(E**2)/(delta_E*sr))[i])
        print(((fitted[n*2:n*3]- fitted_params_lower[n*2:n*3]  )*(GCE)*(E**2)/(delta_E*sr))[i])

                                                            



    max_likelihood

    np.sum(max_likelihood)

    np.savetxt(f'./GCE_model_{model}_12yr_cholis.dat', np.vstack([E, fitted[n*2:n*3]*(GCE)*(E**2)/(delta_E), (fitted_errors[n*2:n*3]*GCE)*(E**2)/(delta_E), (fitted_params_lower[n*2:n*3])*(GCE)*(E**2)/(delta_E), (fitted_params_upper[n*2:n*3])*(GCE)*(E**2)/(delta_E)]).T)
    np.savetxt(f'./GCE_model_{model}_12yr_cholis_likelihood_value', np.array((max_likelihood))) #Positive of log likelihood
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_mcmc_only.py MODEL_NAME | all", file=sys.stderr)
        sys.exit(1)
    
    arg = sys.argv[1]
    if arg.lower() == 'all':
        targets = MODEL_LIST_DEFAULT
    else:
        targets = [arg]
    
    for model in targets:
        out_dat = f'./GCE_model_{model}_12yr_cholis.dat'
        if os.path.exists(out_dat):
            print(f"[skip] {out_dat} already exists", flush=True)
            continue
        t0 = time.time()
        ok = run_one_model(model)
        elapsed = time.time() - t0
        print(f"\n[model={model}] done in {elapsed/60:.1f} min, success={ok}", flush=True)
