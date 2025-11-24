import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

class Interpolator:
    def __init__(self, mass):
        # Load DM masses
        dm_mass = np.loadtxt('/home/sanghwan/Madgraph_3.5.7/MG5_aMC_v3_5_7/bin/4b_effective_vertex/dm_mass.txt')
        self.mass = mass
        
        # Build unified energy grid from all files
        all_energies = []
        for m in dm_mass:
            data = np.loadtxt(f'/home/sanghwan/Madgraph_3.5.7/MG5_aMC_v3_5_7/bin/4b_effective_vertex/Events/DM{m}GeV/DM{m}GeV_gamma.dat')
            energy = np.atleast_1d(data)[:, 0] if data.ndim > 1 else np.atleast_1d(data)
            all_energies.append(energy)
            #all_energies.extend(data[:, 0])
        
        # Create common energy grid (sorted unique values)
        #self.common_energy = np.unique(np.concatenate(all_energies))
        self.common_energy = np.unique(np.concatenate([np.atleast_1d(e) for e in all_energies]))
        
        # Interpolate all mass datasets to common energy grid
        flux_data = []
        for m in dm_mass:
            data = np.loadtxt(f'/home/sanghwan/Madgraph_3.5.7/MG5_aMC_v3_5_7/bin/4b_effective_vertex/Events/DM{m}GeV/DM{m}GeV_gamma.dat')
            # Create interpolation function for this mass
            interp_fn = interp1d(data[:, 0], data[:, 1], 
                                bounds_error=False, 
                                fill_value=0.0,  # Extrapolate with zeros
                                kind='linear')    # Can change to 'nearest' if needed
            
            # Interpolate to common grid
            flux_common = interp_fn(self.common_energy)
            flux_data.append(flux_common)
        
        # Create 2D interpolator (mass, energy)
        self.interpolator = RegularGridInterpolator(
            (dm_mass, self.common_energy),
            np.array(flux_data),
            bounds_error=False,
            fill_value=0.0
        )
    
    def interpolated_table(self):
        # Create query points array: (mass, energy) pairs
        query_points = np.column_stack((
            np.full(len(self.common_energy), self.mass),
            self.common_energy
        ))
        
        # Perform interpolation
        interpolated_flux = self.interpolator(query_points)
        return self.common_energy, interpolated_flux


#Generated with help of Deepseek R1