import os
import numpy as np
import pyhepmc as hep  # HEPMC reader
import numpy as np
import os
import pyhepmc as hep
from pyhepmc.view import savefig


def spectra_generation(model_name):
    # -----------------------------------------------------------------------------
    # Build PPPC-style log(x) grid once
    # x = E / M,   log10 x ∈ [-8.9, 0.0]  in steps of 0.05 dex
    # -----------------------------------------------------------------------------
    LOGX_MIN, LOGX_MAX, DLOGX = -8.9, 0.0, 0.05
    X_EDGES = 10 ** np.arange(LOGX_MIN, LOGX_MAX + DLOGX, DLOGX)  # length 181
    
    # -----------------------------------------------------------------------------
    # Helper: convert the fixed x-grid into an *energy* grid for a given M
    # -----------------------------------------------------------------------------
    
    def energy_edges(mass_gev: float) -> np.ndarray:
        """Return the bin edges *in GeV* that correspond to the global X_EDGES."""
        return X_EDGES * mass_gev
    
    
    def spectrum_from_hepmc(hepmc_file: str, mass_gev: float) -> np.ndarray:
        """Return a dN/dE table on the PPPC x-grid for one HEPMC file."""
    
        # 1. Read photon energies -------------------------------------------------
        egamma = []
        n_events = 0
        with hep.open(hepmc_file) as fh:
            for event in fh:
                n_events += 1
                egamma.extend(
                    p.momentum.e for p in event.particles if p.status == 1 and abs(p.pid) == 22
                )
    
        egamma = np.asarray(egamma)
    
        # 2. Build mass-dependent energy bins -------------------------------------
        e_edges = energy_edges(mass_gev)
        delta_e = np.diff(e_edges)
        e_cent = np.sqrt(e_edges[:-1] * e_edges[1:])
    
        # 3. Histogram and convert to differential spectrum -----------------------
        counts, _ = np.histogram(egamma, bins=e_edges)
        dnde = counts / delta_e / n_events  # units: ph GeV^-1 per annihilation
    
        return np.column_stack([e_cent, dnde])
    
    
    # -----------------------------------------------------------------------------
    # Main loop over masses
    # -----------------------------------------------------------------------------
    
    dm_masses = np.loadtxt(
        f"/home/sanghwan/Madgraph_3.5.7/MG5_aMC_v3_5_7/bin/{model_name}/dm_mass.txt"
    )
    
    
    for m in dm_masses:
        base = f"/home/sanghwan/Madgraph_3.5.7/MG5_aMC_v3_5_7/bin/{model_name}/Events/DM{m}GeV"
    
        # find first available HEPMC file ----------------------------------------
        tag = 0
        while True:
            cand = os.path.join(base, f"tag_{tag}_pythia8_events.hepmc")
            if os.path.exists(cand):
                hepmc_file = cand
                print(f"{hepmc_file} is found and will be proccessed")
                break
            cand = os.path.join(base, f"tag_{tag}_pythia8_events.hepmc.gz")
            if os.path.exists(cand):
                hepmc_file = cand
                print(f"{hepmc_file} is found and will be proccessed")
                break
            tag += 1
    
        print(f"Processing {hepmc_file} …")
    
        table = spectrum_from_hepmc(hepmc_file, m)
    
        out_file = os.path.join(base, f"DM{m}GeV_gamma_v1.dat")
        np.savetxt(out_file, table, header="E  dN/dE  [GeV  ph/GeV]", fmt="%e")
        print(f" → saved {out_file}  ({table.shape[0]} bins)")
