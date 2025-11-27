import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator, interp1d

class GammaSpectrumInterpolator:
    """Interpolate PYTHIA dN/dE tables (on PPPC log‑x grid) in *both* DM mass
    and energy.  Usage:

        Es, dNdE = GammaSpectrumInterpolator(15).table()  # mass = 15 GeV
    """

    def __init__(self, target_mass: float,
                channel: str ="effective_4b",
                 base_dir: Path = Path("/home/sanghwan/Madgraph_3.5.7/MG5_aMC_v3_5_7/bin/"),
                 tag: str = "gamma_v1"
                ):
        self.mass = float(target_mass)
        self.base = base_dir
        self.tag  = tag
        self.channel = channel
        # --------------------------------------------------------------
        # 1. Load all available masses and their spectra
        # --------------------------------------------------------------
        dm_masses = np.loadtxt(self.base / self.channel / "dm_mass.txt", dtype=float)
        dm_masses = np.sort(dm_masses)          # RegularGridInterpolator needs ascending grid
        spectra   = []                          # list of (E, dNdE) arrays

        all_E = []
        for m in dm_masses:
            table = np.loadtxt(self._file_of(m))
            spectra.append(table)
            all_E.append(table[:, 0])            # keep energies for common grid

        # --------------------------------------------------------------
        # 2. Build a *common* energy axis ( ∪ of all energies )
        # --------------------------------------------------------------
        self.E = np.sort(np.unique(np.concatenate(all_E)))

        # new_array = np.logspace(np.log10(self.E[0]), np.log10(self.E[-1]), 179) #This array reform the shape of Energy array.
        # self.E = new_array
        # # --------------------------------------------------------------
        # 3. Re‑sample every spectrum onto that axis
        # --------------------------------------------------------------
        flux_grid = np.zeros((len(dm_masses), len(self.E)))
        for i, (m, table) in enumerate(zip(dm_masses, spectra)):
            interp = interp1d(table[:, 0], table[:, 1], bounds_error=False,
                              fill_value=0.0, kind="linear")
            flux_grid[i, :] = interp(self.E)

        # --------------------------------------------------------------
        # 4. 2‑D interpolator in (mass, energy)
        # --------------------------------------------------------------
        self._interp = RegularGridInterpolator((dm_masses, self.E), flux_grid,
                                               bounds_error=False, fill_value=0.0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _file_of(self, m):
        fn = self.base / self.channel/ f"Events/DM{m}GeV/DM{m}GeV_{self.tag}.dat"
        if not fn.exists():
            raise FileNotFoundError(fn)
        return fn

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def interpolated_table(self):
        """Return (E, dN/dE) at the target mass specified in __init__."""
        pts = np.column_stack([np.full_like(self.E, self.mass), self.E])
        return self.E, self._interp(pts)


def exctractcirellitable(DMmass,DMchannel,particle,EWcorr):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    '''
    This function returns the energy spectrum in 1/GeV for the particle production from DM annihilation.
    These results come from the PPPC4DM http://www.marcocirelli.net/PPPC4DMID.html
    Relevant tables should be downloaded and stored locally
    
    DMmass: dark matter mass in GeV
    DMchannel: dark matter annihilation channel with EW correction (no EW correction)
    e 4 (2), mu 7 (3), tau 10 (4), bb 13 (7), tt 14 (8), WW 17 (9), ZZ 20 (10), gamma 22 (12), h 23 (13)
    particle: particle produced from the DM annihilation ('gammas' or 'positrons')
    EWcorr: electroweak corrections ('Yes' or 'No')
    '''

    if EWcorr=='Yes':
        listenergies = 179
        energy_vec = np.arange(-8.9,0.05,0.05)
    elif EWcorr=='No':
        listenergies = 180
        energy_vec = np.arange(-8.95,0.05,0.05)
    else:
        print('Error Wrong value for EWcorr, Yes or No')
    
    energy = np.zeros(listenergies)
    fluxDM = np.zeros(listenergies)
    if EWcorr=='No': 
        table = np.loadtxt('./PPPC4/particle_data/AtProduction%sEW_%s.dat'%(EWcorr,particle), skiprows=1)
    else:
        table = np.loadtxt('./PPPC4/particle_data/AtProduction_%s.dat'%(particle), skiprows=1)
    massvec = []
    for t in range(len(table)):
        if t%listenergies == 0:
            massvec.append(table[t,0])
    massvec = np.array(massvec)
    
    flux = []
    for t in range(len(table)):
        flux.append(table[t,DMchannel])
    
    f = interpolate.interp2d(massvec, energy_vec, flux, kind='linear')
    
    for t in range(len(energy_vec)):
        fluxDM[t] = f(DMmass,energy_vec[t])
    
    return np.power(10.,energy_vec)*DMmass, fluxDM/(np.log(10.)*np.power(10.,energy_vec)*DMmass)

# ----------------------------------------------------------------------
# Example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    Es, dNdE = GammaSpectrumInterpolator(15, channel="effective_4b").interpolated_table()
    print(Es[:5])
    print(dNdE[:5])
