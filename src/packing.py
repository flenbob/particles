import math
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from ovito.io import import_file
from itertools import product

from .filename import FileName
from .content_uniformity import Stange
from .table_params import Param
from .particles import Particles

@dataclass
class CoordinatesGenerator:
    """Generates initial coordinates of a polydisperse packing"""
    diameters: np.ndarray
    collection_intervals: np.ndarray
    levels: np.ndarray = None
    coordinates: np.ndarray = None
    width_box: float = None
    h_diams: np.ndarray = None
    cell_grid: dict = field(default_factory=dict)
    ns: list = field(default_factory=list)
    sort_order: np.ndarray = None
    
    def __post_init__(self):
        #Initialize empty Nx3 array of coordinates
        self.coordinates = np.zeros((self.diameters.shape[0], 3))
        
        #Sort diameters and collection intervals descending
        self.sort_order = np.argsort(-self.diameters)
        self.diameters = self.diameters[self.sort_order]
        self.collection_intervals = -np.sort(-self.collection_intervals)
        
        #Corresponding level (collection interval) for each diameter
        self.h_diams = np.argmax(self.diameters[:, None] > \
                                 np.concatenate((self.collection_intervals, [0])), 
                                 axis = 1) - 1
        
        #Level identifiers
        self.levels = np.unique(self.h_diams)
        
        #Set box width
        volume_fraction = 0.05
        volume_particles = np.pi/6*(self.diameters**3).sum()
        volume_box = volume_particles/volume_fraction
        self.width_box = volume_box**(1/3)
        
        #Construct hierarchy of levels and cells
        self.n_cells = [math.floor(self.width_box/ci) for ci in self.collection_intervals]
        self.width_cells = [self.width_box/n_cells_h for n_cells_h in self.n_cells]
        
        #List of tuples of directions to access all 27 cell neighbors (including its own)
        self.ns = [n for n in product([0, 1, -1], repeat=3)]
        
    def _map_to_cell(self, coord: np.ndarray, h: int) -> tuple:
        """Maps input coordinate and hierarchy level to cell"""
        c = (math.floor(coord[0]/self.width_cells[h]), 
             math.floor(coord[1]/self.width_cells[h]), 
             math.floor(coord[2]/self.width_cells[h]),
             h)
        return c
    
    def _insert_cell(self, p: int, c: tuple) -> None:
        """Inserts particle id (p) into cell grid at cell (c)"""
        if not self.cell_grid.get(c):
            self.cell_grid.update({c: [p]})
        else:
            self.cell_grid[c].append(p)
            
    def _get_neighs(self, coord) -> list[int]:
        """Get particle identifiers in neighboring cells of a coordinate in all levels"""
        p_neighs = []
        for h in self.levels:
            c = self._map_to_cell(coord, h) #Cell of coordinate at h:th level
            for n in self.ns:
                #Neighboring cell for each of the 27 directions
                c_neigh = (c[0] + n[0], 
                           c[1] + n[1], 
                           c[2] + n[2],
                           c[3])
                
                #Neighboring particle identifier
                p = self.cell_grid.get(c_neigh) 
                if p is not None:
                    p_neighs.append(p)
                    
        #Flattened list of particle identifiers
        return list(chain.from_iterable(p_neighs))
    
    def _check_contacts(self, diam: float, coord: np.ndarray, p_neighs: list[int]) -> bool:
        """Check if particle (p) with coordinate (coord) is in contact with
            any of the neighboring particles (p_neighs) 
        """
        #Collect neighbor diameters and coordinates
        diam_neighs = self.diameters[p_neighs]
        coord_neighs = self.coordinates[p_neighs]
        
        #Terminate if euclidean distance squared is shorter than sum of the radii squared
        for (diam_neigh, coord_neigh) in zip(diam_neighs, coord_neighs):
                dist_sq = ((coord_neigh - coord)**2).sum()
                sum_radii_sq = (diam_neigh/2 + diam/2)**2
                if dist_sq < sum_radii_sq:
                    return True
        return False
        
    def generate_coordinates(self) -> np.ndarray:
        """Generates non-overlapping coordinates"""
        for p, (h, diam) in enumerate(zip(self.h_diams, self.diameters)):
            while True:
                #Random coordinate
                coord = np.random.uniform(low = diam/2, 
                                          high = self.width_box - diam/2, 
                                          size = (3,))
                
                #Particle ids of neighbors
                p_neighs = self._get_neighs(coord)
                
                #If no neighbors then insert to cellgrid and sample new coord
                if len(p_neighs) == 0:
                    c = self._map_to_cell(coord, h)
                    self._insert_cell(p, c)
                    self.coordinates[p] = coord
                    break
                
                # Otherwise check contacts and insert if none are found
                if not self._check_contacts(diam, coord, p_neighs):
                    c = self._map_to_cell(coord, h)
                    self._insert_cell(p, c)
                    self.coordinates[p] = coord
                    break
                
        #Sort in same order as input diameters to avoid accidental mismatch
        return self.coordinates[self.sort_order]


@dataclass
class CollectionIntervalGenerator:
    diameters: np.ndarray
    initial_volume_fraction: float = 0.05
    box_width: float = field(default_factory=float, init=False)

    def __post_init__(self):
        #Sort diameters
        self.diameters = np.sort(self.diameters)

        # assert np.array_equal(self.diameters, np.sort(self.diameters)), f'Diameters must be sorted ascending to generate collection intervals correctly.'

        #Calculate (orthogonal) box width given initial volume fraction
        self.box_width = (np.pi/(6*self.initial_volume_fraction)*np.sum(self.diameters**3))**(1/3)

    def _particles_per_cell(self, i0: int, i1: int) -> float:
        return self.diameters[i1]**3*(i1-i0)/self.box_width**3

    def generate_collection_intervals(self) -> list[float]:
        """Generates optimal collection intervals given diameters, simulation box width and volume fraction"""

        # Use binary search to find indicies 
        N = self.diameters.shape[0]
        m_max = N*self.diameters[-1]**3/self.box_width**3
        delta_m = m_max/10000

        ms = np.arange(delta_m, m_max, step=delta_m)
        Ls = np.ones(np.shape(ms))
        level_ids_old = []
        c = 0
        for idx, m0 in enumerate(ms):
            level_ids_new = []
            if idx == 0 or len(level_ids_old) == 0:
                i = 0                   # First m
            else:
                # Start at the idth level of the previous m
                i = level_ids_old[0]

            id = 1
            ref = 0
            i_lower = 0                     
            i_upper = N
            while True:
                c += 1
                # Based on the current position, take a step halfways towards either ref or N
                if i >= N-1: 
                    # Endpoint
                    break
                m = self._particles_per_cell(ref, i)      

                if m < m0:
                    i_lower = i
                    step = np.ceil((i_upper-i)/2).astype(int)
                    i += step
                else:
                    i_upper = i
                    step = np.ceil((i-i_lower)/2).astype(int)
                    i -= step

                if self._particles_per_cell(ref, i) > m0 and self._particles_per_cell(ref, i-1) < m0:
                    level_ids_new.append(i)
                    ref = i
                    i += 1
                    i_lower = i
                    i_upper = N
                    Ls[idx] += 1

                    if idx==0:
                        i += 1              # No previous list
                    elif id >= len(level_ids_old)-1:
                        # Accessed last id of previous m, we're at the end
                        i += 1
                    else: 
                        i = level_ids_old[id]
                        
                    id += 1 
            level_ids_old = level_ids_new

        K = 0.3*self.initial_volume_fraction/0.62
        LsU, indicies = np.unique(Ls, return_index  = True)
        MsU = ms[indicies]
        TCD = LsU*(MsU+K)
        TCD = TCD/np.min(TCD)
        idx_min = np.argmin(TCD)
        m_min = MsU[idx_min]

        # Get the indicies for m_min
        collection_intervals = []         # Level-ids for previous m
        i = 1
        ref = 0
        while True:
            # Assign particles to a level til Nh/Nc > m
            if i == N: 
                # Endpoint
                collection_intervals.append(self.diameters[-1])  
                break
            m = self._particles_per_cell(ref, i)
            if m < m_min:
                # Less than required m; add to current index i
                i += 1
            else:
                # Overstepped; switch reference level, increment i and Ls  
                collection_intervals.append(self.diameters[i])  
                i += 1        
                ref = i

        #Final collection intervals
        return np.unique(collection_intervals)


@dataclass
class ParticlesGenerator:
    """Generates particles given input CSV table"""
    
    table_path: Path
    params: dict[np.ndarray] = field(default_factory=dict, init=False)
    
    #Constants
    mass_fraction_error = 1e-2

    def __post_init__(self):
        #Load and check parameters
        self.params = self._load_table_params()
        self._check_params()

    def generate_particles(self) -> Particles:
        """Samples particles (ids, type_ids and diameters)"""

        #Calculate mass required for each comp
        mass_comps = Stange().mass_given_cov_params(self.params)
        mass_comps_max = mass_comps.max()
        
        #Estimate number of particles to sample
        E_D3 = np.exp(3*self.params[Param.mu]+9/2*self.params[Param.sigma]**2)
        N_expected_comp = np.ceil((6*self.params[Param.mass_fraction]*mass_comps_max)/\
                                  (np.pi*E_D3*self.params[Param.density])).astype(int)
        N_expected_total = N_expected_comp.sum()

        #Sample particle diameters
        N_comps = self.params[Param.types_id].shape[0]
        samples = np.random.lognormal(self.params[Param.mu], 
                                      self.params[Param.sigma], 
                                      (2*N_expected_total, N_comps))
        counts = np.ones(N_comps, dtype=int)

        #Sample until solid particle volume is reached, and mass fraction error is low enough
        rho = self.params[Param.density]
        m_current = np.array([np.pi/6*(rho[id]*samples[:counts[id], id]**3).sum() 
                              for id in self.params[Param.types_id]-1])
        m_total = np.sum(m_current)
        while True:
            #Identify component with highest mass fraction error
            mf_curr = m_current/np.sum(m_current)
            mf_error = (self.params[Param.mass_fraction] - mf_curr)/self.params[Param.mass_fraction]
            i = np.argmax(mf_error)

            #Check exit condition (not too often though)
            if counts[i] % 100 == 0:
                if np.all(np.abs(mf_error) < self.mass_fraction_error) and (m_total > mass_comps_max):
                    break

            #Add sample to packing
            counts[i] += 1
            m_sample = np.pi/6*rho[i]*samples[counts[i], i]**3
            m_current[i] += m_sample
            m_total += m_sample

        #Set particle values
        diameters = np.hstack(([samples[:counts[id], id] 
                                for id in self.params[Param.types_id]-1]))
        ids = np.arange(1, diameters.shape[0]+1, dtype=int)
        type_ids = np.hstack(([np.full(counts[id], id+1) 
                               for id in self.params[Param.types_id]-1])).astype(int)

        #Normalize by smallest diameter
        rf = np.min(diameters)
        diameters = diameters/rf
        particles = Particles(ids = ids, 
                              type_ids = type_ids,
                              diameters = diameters, 
                              density_types=rho, 
                              rescale_factor=rf)

        #Check Stange COV given sampled particles
        cov = Stange().cov_given_mass_particles(particles)

        #Return particles (without coordinates)
        print(f"Total mass: {particles.mass_types.sum():.2f} µg\nN particles: {diameters.shape[0]}\nStange COV component-wise prediction(s): {cov.T}")
        return particles

    def _load_table_params(self) -> dict:
        """Loads CSV table and converts to dictionary with parameters"""
        table = np.loadtxt(self.table_path, delimiter=';', dtype=float)
        assert table.shape[1] == 6, f'Provided table in {self.table_path} has too few columns. It requires 6 columns.'

        #Convert to parameter dictionary
        params = {Param.types_id: table[:, 0].astype(int),
                  Param.density: table[:, 1].astype(float),
                  Param.mass_fraction: table[:, 2].astype(float),
                  Param.mean: table[:, 3].astype(float),
                  Param.std: table[:, 4].astype(float),
                  Param.cv: table[:, 5].astype(float)}
        
        #Rescale densities from kg/m^3 to yg/ym^3
        params[Param.density]*= 1e-9

        #Calculate Lognormal distribution params LN(mu, sigma) that satisfy desired mean and std
        mean, std = params[Param.mean], params[Param.std]
        params[Param.mu] = np.log(mean**2/(np.sqrt(mean**2+std**2)))
        params[Param.sigma] = np.sqrt(np.log(1+std**2/mean**2))
        return params
    
    def _check_params(self) -> None:
        """Checks parameter values are correctly formatted"""
        assert np.array_equal(self.params[Param.types_id], np.arange(1, np.max(self.params[Param.types_id])+1)),\
              "Component ID:s should be provided as an array-like of consecutive integers starting from 1."
        assert (self.params[Param.density] > 0).all(), "Bulk densities must be positive."
        assert np.abs(1 - np.sum(self.params[Param.mass_fraction])) < 1e-3, "Mass fractions do not sum up to one (At least three decimal point precision)." 
        assert (self.params[Param.mu] > 0).all(), "Expected diameter must be positive."
        assert (self.params[Param.sigma] > 0).all(), "Standard deviation must be positive."
        assert (self.params[Param.cv] > 0).all(), "Maximum coefficient variation (cv) must be positive."  


@dataclass
class Packing:
    """""Packing consisting of an initial state of single/multi-component spherical particles, given input CSV table."""
    particles: Particles = None
    collection_intervals: list[float] = field(default_factory=list)
    box_width: float = field(default_factory=float, init=False)

    #Constants
    initial_volume_fraction: float = 0.05
    def __post_init__(self) -> None:
        #If particles are provided at init, set box width
        if self.particles is not None:
            self.box_width = (np.pi/(6*self.initial_volume_fraction)*np.sum(self.particles.diameters**3))**(1/3)

    def generate_packing(self, table_path: Path) -> None:
        """Generates packing given input table"""
        
        #Generate particles (ids, types, diameters)
        generator = ParticlesGenerator(table_path)
        self.particles = generator.generate_particles()

        #Set simulation box width
        self.box_width = (np.pi/(6*self.initial_volume_fraction)*np.sum(self.particles.diameters**3))**(1/3)

        #Sort particles ascending for collection intervals
        self.particles.sort_by_diameters(order='ascending')

        #Generate collection intervals
        self.collection_intervals = CollectionIntervalGenerator(self.particles.diameters).generate_collection_intervals()

        #Generate particle coordinates        
        self.particles.coordinates = CoordinatesGenerator(self.particles.diameters,
                                                          self.collection_intervals).generate_coordinates()
    
    def load_packing(self, file_path: Path) -> None:
        """Loads existing LAMMPS input file into packing object"""
        #Read densities and rescale factor from header
        with open(file_path, 'r') as f:
            f.__next__()
            densities = [float(dens) for dens in f.__next__().rstrip()[1:-1].split(', ')]
            rf = float(f.__next__().rstrip()[1:-1])

        #Data reading pipeline
        pipeline = import_file(file_path, multiple_frames=False)
        data = pipeline.compute()

        #Set simulation box width
        self.box_width = data.cell[0][0]

        #Set particles
        self.particles = Particles(ids = np.array(data.particles['Particle Identifier'][:], dtype=int), 
                                   type_ids = np.array(data.particles['Particle Type'][:], dtype=int),
                                   diameters = np.array(data.particles['Radius'][:], dtype=float)*2,
                                   coordinates = np.array(data.particles['Position'][:], dtype=float),
                                   density_types=np.array(densities),
                                   rescale_factor=rf)
        
        #Sort particles ascending for collection intervals
        self.particles.sort_by_diameters(order='ascending')

        #Generate collection intervals
        self.collection_intervals = CollectionIntervalGenerator(self.particles.diameters).generate_collection_intervals()

    def write_packing(self, path: Path) -> None:
        """Write particles in packing to a file that is readable by LAMMPS

        Args:
            path (Path): Relative path to file which will be written.
        """
        densities_print = [f"{density:.10f}, " for density in self.particles.density_types]
        with open(path/FileName.INPUT_FILE.value, 'w') as file:
            file.write(f'# LAMMPS file containing particle data with densities and rescale factor:\n')
            file.write(f'#{"".join(densities_print)} \n')
            file.write(f'#{self.particles.rescale_factor}\n\n')
            file.write(f"{self.particles.diameters.shape[0]} atoms \n")
            file.write(f"{np.max(self.particles.type_ids).astype(int)} atom types \n")
            file.write(f"0 {self.box_width} xlo xhi \n")
            file.write(f"0 {self.box_width} ylo yhi \n")
            file.write(f"0 {self.box_width} zlo zhi \n")
            file.write(f"0 0 0 xy xz yz \n\n")
            file.write(f"Atoms # sphere\n\n")

            # Particles
            for (id, type_id, diameter, coordinate) in \
                zip(self.particles.ids, self.particles.type_ids, self.particles.diameters, self.particles.coordinates):
                file.write(f"{id} {type_id} {diameter} 1 {coordinate[0]} {coordinate[1]} {coordinate[2]} \n")

        print(f'Particles written to file {path}\n')