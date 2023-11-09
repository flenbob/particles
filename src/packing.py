import math
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from ovito.io import import_file

from .filename import FileName
from .content_uniformity import Stange
from .table_params import Param
from .particles import Particles

@dataclass
class CoordinatesGenerator:
    """Samples coordinates of non-overlapping particles inside a cell using multigrid method."""

    diameters: np.ndarray
    collection_intervals: list[float]
    r: int = field(default=10, repr=False)
    initial_volume_fraction: float = 0.05
    box_width: float = field(default_factory=float, init=False)

    #Cell container in for each axis
    CellGrid = list[list[list[int]]]
    xcells_levels: CellGrid = field(repr=False, init=False)
    ycells_levels: CellGrid = field(repr=False, init=False)
    zcells_levels: CellGrid = field(repr=False, init=False)

    #Particle coordinate container for each axis and level
    AxisLevel = list[list[float]]
    xcoords_levels: AxisLevel = field(repr=False, init=False)
    ycoords_levels: AxisLevel = field(repr=False, init=False)
    zcoords_levels: AxisLevel = field(repr=False, init=False)

    #Particle diameters and number of particles per level
    diameters_levels: AxisLevel = field(repr=False, init=False)
    N_levels: list[float] = field(repr=False, init=False)

    #Number of cells and subcell width
    N_cells: list[int] = field(repr=False, init=False)
    L_cells: float = field(repr=False, init=False)

    def __post_init__(self):
        """Initializes class variables keeping track of particles by level, etc."""
        #Checks
        self._check_collection_intervals()
        self._check_diameters()

        self.diameters = self.diameters
        self.collection_intervals = self.collection_intervals

        #Calculate (orthogonal) box width given initial volume fraction
        self.box_width = (np.pi/(6*self.initial_volume_fraction)*np.sum(self.diameters**3))**(1/3)

        #Divide simulation cell into N_cells using some parameter r
        self.N_cells = int(self.box_width/self.r)
        self.L_cells = self.box_width/self.N_cells

        #Arrange particles by levels corresponding to provided collection intervals
        id_levels = self._order_particles_by_level()

        #Initialize particle coordinate containers for each level
        self.xcoords_levels = [[] for _ in id_levels]
        self.ycoords_levels = [[] for _ in id_levels]
        self.zcoords_levels = [[] for _ in id_levels]

        #Initialize coordinate to subcell containers
        self.xcells_levels = [[[] for _ in range(self.N_cells)] for _ in id_levels]       
        self.ycells_levels = [[[] for _ in range(self.N_cells)] for _ in id_levels]
        self.zcells_levels = [[[] for _ in range(self.N_cells)] for _ in id_levels]

        #Initialize particle diameter container
        self.diameters_levels = [[self.diameters[j] for j in range(lo, hi)] for (lo, hi) in id_levels]
        self.N_levels = [hi-lo for (lo, hi) in id_levels]

    def generate_coordinates(self) -> np.ndarray:
        # Using the cell list we want to be able to get coords and diameters
        # |----|----|-----|-----|      # r = 4 : L = B/r
        #      0    1     2     3  
        # No of cells equals r

        # Incrementally add the particles
        k = 0
        for i, N_level in enumerate(self.N_levels):
            # i is the CI that we're picking particles from
            for j in range(N_level):
                # j denotes the particle id in CI[i]
                d = self.diameters_levels[i][j]
                while True:
                    # Box ranges from [0, box_size]; sample uniformly in [d/2, bl-d/2] 
                    c = np.random.uniform(low = d/2, high = self.box_width-d/2, size = (3,))

                    if(self._subdomain_cc(d, c, i) is False):
                        break
                    # Find where to insert idx
                self._insert_cell(c, i, j)
                k += 1
        
        #Flatten coordinates for each axis over all subcells and write to NumPy array
        x_c = list(chain(*self.xcoords_levels))
        y_c = list(chain(*self.ycoords_levels))
        z_c = list(chain(*self.zcoords_levels))
        coordinates = (np.row_stack((np.array(x_c), np.array(y_c), np.array(z_c)))).T
        return coordinates

    def _order_particles_by_level(self) -> list[tuple[int, int]]:
        """Oder particles so indicies i_j indicate the largest index such that diams[i_j] <= CI_j
        """
        N = self.diameters.shape[0]
        inds = []
        for collection_interval in self.collection_intervals:
            i = 0
            while True:
                if(self.diameters[i] <= collection_interval or i == N-1):
                    inds.append(i)
                    break
                i += 1
        inds.append(N)
        id_tuples = [(inds[i-1], j) for i, j in enumerate(inds) if i > 0]
        return id_tuples

    def _insert_cell(self, c: np.ndarray, i_: int, j: int) -> None:
        """Inserts the index j in the correct cell

        Args:
            c (np.ndarray): Center coordinate of the current particle
            i_ (int): The index of the CI (collection interval)
            j (int): The ID of the particle 
        """

        x_no = min(int(math.floor(c[0]/self.L_cells)), self.N_cells - 1)   
        y_no = min(int(math.floor(c[1]/self.L_cells)), self.N_cells - 1)
        z_no = min(int(math.floor(c[2]/self.L_cells)), self.N_cells - 1)

        self.xcells_levels[i_][x_no].append(j)
        self.ycells_levels[i_][y_no].append(j)
        self.zcells_levels[i_][z_no].append(j)
        
        self.xcoords_levels[i_].append(c[0])
        self.ycoords_levels[i_].append(c[1])
        self.zcoords_levels[i_].append(c[2])

    def _retrieve_cc_lists(self, c: np.ndarray, i_: int, d_: float) -> list[float]:
        """Returns the lists of indicies that are possible collisions from the cell lists

        Args:
            c (np.ndarray): Center coordinate of the current particle
            i_ (int): The index of the CI (collection interval)
            d_ (float): Cutoff distance for overlap check
        Returns:
            list: List of lists of indicies where collisions may occur
        """

        # |-----|--*--|-----|--*--| : [1, 2, 3] = range(1,4)
        #       0     1     2     3 
        
        diff = (d_ + self.diameters_levels[i_][0])/2

        x_hi = min(int(math.ceil((c[0] + diff)/self.L_cells)), 
                   self.N_cells)     
        x_lo = int(math.floor((c[0]-diff)/self.L_cells)) - 1              

        y_hi = min(int(math.ceil((c[1] + diff)/self.L_cells)), 
                   self.N_cells)
        y_lo = int(math.floor((c[1]-diff)/self.L_cells)) - 1

        z_hi = min(int(math.ceil((c[2] + diff)/self.L_cells)), 
                   self.N_cells)  
        z_lo = int(math.floor((c[2]-diff)/self.L_cells)) - 1
        return [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]

    def _overlap_cc(self, c: np.ndarray, ids: list, d_: float, i_: int) -> bool:
        """Checks overlap by comparing squared euclidean distances

        Args:
            c (np.ndarray): Center coordinate of the current particle
            ids (list): Particle IDs to check
            d_ (float): Cutoff distance for overlap check
            i_ (int): The index of the CI (collection interval)

        Returns:
            bool: True if overlap, else False
        """
        for id in ids:
            D = self.diameters_levels[i_][id]
            euclidean = (c[0] - self.xcoords_levels[i_][id])**2+\
                        (c[1] - self.ycoords_levels[i_][id])**2+\
                        (c[2] - self.zcoords_levels[i_][id])**2
            
            if((D+d_)**2 > euclidean):
                return True
        return False
                    
    def _subdomain_cc(self, d_: float, c: np.ndarray, i: int) -> bool:
        '''
        Returns the idxs in the CI searched within the ranges specified by c and diff
            Parameters:
                    diff       : (float)   cutoff distance for overlap check
                    c          : (np.ndarray) center coordinate of the current particle
                    i          : (int)     the index of the CI 
            Returns:
                    idxs       : (list)    of indicies in coords[i] where collisions may occur
        '''
        # Get the lists to run CC against:
        for k in range(i + 1):
            cl_ = self._retrieve_cc_lists(c, k, d_)
            x_r = range(cl_[0], cl_[1])
            y_r = range(cl_[2], cl_[3])
            z_r = range(cl_[4], cl_[5])
            x_l = [self.xcells_levels[k][s] for s in x_r]
            y_l = [self.ycells_levels[k][s] for s in y_r]
            z_l = [self.zcells_levels[k][s] for s in z_r]
            x_lf = list(chain.from_iterable(x_l))
            y_lf = list(chain.from_iterable(y_l))
            z_lf = list(chain.from_iterable(z_l))

            xyz = list(set(x_lf).intersection(y_lf, z_lf))
            if not xyz:
                continue

            if(self._overlap_cc(c, xyz, d_, k)):
                return True
        return False

    def _check_diameters(self):
        """Checks that diameters are sorted descending"""
        self.diameters = np.sort(self.diameters)[::-1]
        #assert np.array_equal(self.diameters, np.sort(self.diameters)[::-1]), f'Diameters must be sorted descending to generate coordinates correctly.'
    
    def _check_collection_intervals(self):
        """Checks that collection intervals are ordered descending and are within diameter range"""
        #Sort descending
        self.collection_intervals = np.sort(self.collection_intervals)[::-1]

        #assert np.array_equal(self.collection_intervals, np.sort(self.collection_intervals)[::-1]), f'Collection intervals must be sorted descending to generate coordinates correctly.'
        assert max(self.collection_intervals) - np.max(self.diameters) < 1e-1, f'Upper collection interval does not correspond to largest particle diameter.'
        assert min(self.collection_intervals) > np.min(self.diameters), f'Lower collection interval is smaller than smallest particle diameter.'


@dataclass
class CoordinatesGeneratorParallell:
    """Samples coordinates of non-overlapping particles inside a cell using multigrid method, work being distributed."""

    # Fields that aren't thread specific
    Ndivs = 4
    diameters: np.ndarray
    collection_intervals: list[float]
    r: int = field(default=5, repr=False)
    initial_volume_fraction: float = 0.05
    box_width: float = field(default_factory=float, init=False)
    box_width_subs: float = field(default_factory=float, init=False)
    no_subs : int =  field(default=Ndivs, repr=False) 
    tot_no_subs : int = field(default=Ndivs**3, repr=False)

    #Cell container in for each axis
    CellGrid = list[list[list[int]]]
    xcells_levels: CellGrid = field(repr=False, init=False)
    ycells_levels: CellGrid = field(repr=False, init=False)
    zcells_levels: CellGrid = field(repr=False, init=False)

    #Particle coordinate container for each axis and level and subs
    AxisLevel = list[list[float]]
    xcoords_levels: AxisLevel = field(repr=False, init=False)
    ycoords_levels: AxisLevel = field(repr=False, init=False)
    zcoords_levels: AxisLevel = field(repr=False, init=False)

    #Particle diameters and number of particles per level
    diameters_levels: AxisLevel = field(repr=False, init=False)
    N_levels: list[float] = field(repr=False, init=False)
    diameters_subdivided : list[np.ndarray] = field(repr=False, init = False)

    #Number of cells and subcell width
    N_cells: list[int] = field(repr=False, init=False)
    L_cells: float = field(repr=False, init=False)

    def __post_init__(self):
        """Initializes class variables keeping track of particles by level, etc."""
        #Checks
        self._check_collection_intervals()
        self._check_diameters()

        self.diameters = self.diameters
        self.collection_intervals = self.collection_intervals

        #Calculate (orthogonal) box width given initial volume fraction
        self.box_width = (np.pi/(6*self.initial_volume_fraction)*np.sum(self.diameters**3))**(1/3)
        self.box_width_subs = self.box_width/self.no_subs

        #Divide simulation cell into N_cells using some parameter r
        self.N_cells = int(self.box_width_subs/self.r)
        self.L_cells = self.box_width_subs/self.N_cells


        # Divide particles into equal volumes
        choose_inds = np.random.randint(low = 0, high = self.tot_no_subs, size = np.size(self.diameters),  dtype = int)
        diameters_subdivided = []
        for i in range(self.tot_no_subs):
            # Assign diameters from indeces
            diameters_subdivided.append(self.diameters[choose_inds == i])

        self.diameters_subdivided = diameters_subdivided

    def _CoordinateGen_(self, diameters: np.ndarray):

    
        def _order_particles_by_level() -> list[tuple[int, int]]:
            """Order particles so indicies i_j indicate the largest index such that diams[i_j] <= CI_j
            """
            N = diameters.shape[0]
            inds = []
            for collection_interval in self.collection_intervals:
                i = 0
                while True:
                    if(diameters[i] <= collection_interval or i == N-1):
                        inds.append(i)
                        break
                    i += 1
            inds.append(N)
            id_tuples = [(inds[i-1], j) for i, j in enumerate(inds) if i > 0]
            return id_tuples

        def generate_coordinates() -> np.ndarray:

            def _insert_cell(c: np.ndarray, i_: int, j: int) -> None:
                """Inserts the index j in the correct cell

                Args:
                    c (np.ndarray): Center coordinate of the current particle
                    i_ (int): The index of the CI (collection interval)
                    j (int): The ID of the particle 
                """

                x_no = min(int(math.floor(c[0]/self.L_cells)), self.N_cells - 1)   
                y_no = min(int(math.floor(c[1]/self.L_cells)), self.N_cells - 1)
                z_no = min(int(math.floor(c[2]/self.L_cells)), self.N_cells - 1)

                xcells_levels[i_][x_no].append(j)
                ycells_levels[i_][y_no].append(j)
                zcells_levels[i_][z_no].append(j)
                
                xcoords_levels[i_].append(c[0])
                ycoords_levels[i_].append(c[1])
                zcoords_levels[i_].append(c[2])
                return 

            def _overlap_cc(c: np.ndarray, ids: list, d_: float, i_: int) -> bool:
                """Checks overlap by comparing squared euclidean distances

                Args:
                    c (np.ndarray): Center coordinate of the current particle
                    ids (list): Particle IDs to check
                    d_ (float): Cutoff distance for overlap check
                    i_ (int): The index of the CI (collection interval)

                Returns:
                    bool: True if overlap, else False
                """
                for id in ids:
                    D = diameters_levels[i_][id]
                    euclidean = (c[0] - xcoords_levels[i_][id])**2+\
                                (c[1] - ycoords_levels[i_][id])**2+\
                                (c[2] - zcoords_levels[i_][id])**2
                    
                    if((D+d_)**2 > euclidean):
                        return True
                return False
                            
            def _subdomain_cc(d_: float, c: np.ndarray, i: int) -> bool:
                '''
                Returns the idxs in the CI searched within the ranges specified by c and diff
                    Parameters:
                            diff       : (float)   cutoff distance for overlap check
                            c          : (np.ndarray) center coordinate of the current particle
                            i          : (int)     the index of the CI 
                    Returns:
                            idxs       : (list)    of indicies in coords[i] where collisions may occur
                '''
                # Get the lists to run CC against:
                for k in range(i + 1):
                    # Retrieve contact detection lists
                    if len(diameters_levels[k])==0:
                        continue

                    diff = (d_ + diameters_levels[k][0])/2

                    x_hi = min(int(math.ceil((c[0] + diff)/self.L_cells)), 
                            self.N_cells)     
                    x_lo = int(math.floor((c[0]-diff)/self.L_cells)) - 1              

                    y_hi = min(int(math.ceil((c[1] + diff)/self.L_cells)), 
                            self.N_cells)
                    y_lo = int(math.floor((c[1]-diff)/self.L_cells)) - 1

                    z_hi = min(int(math.ceil((c[2] + diff)/self.L_cells)), 
                            self.N_cells)  
                    z_lo = int(math.floor((c[2]-diff)/self.L_cells)) - 1

                    x_l = [xcells_levels[k][s] for s in range(x_lo, x_hi)]
                    y_l = [ycells_levels[k][s] for s in range(y_lo, y_hi)]
                    z_l = [zcells_levels[k][s] for s in range(z_lo, z_hi)]

                    x_lf = list(chain.from_iterable(x_l))
                    y_lf = list(chain.from_iterable(y_l))
                    z_lf = list(chain.from_iterable(z_l))

                    xyz = list(set(x_lf).intersection(y_lf, z_lf))
                    if not xyz:
                        continue

                    if(_overlap_cc(c, xyz, d_, k)):
                        return True
                return False


            # Using the cell list we want to be able to get coords and diameters
            # |----|----|-----|-----|      # r = 4 : L = B/r
            #      0    1     2     3  
            # No of cells equals r

            # Incrementally add the particles
            k = 0
            for i, N_level in enumerate(N_levels):
                # i is the CI that we're picking particles from
                for j in range(N_level):
                    # j denotes the particle id in CI[i]
                    d = diameters_levels[i][j]
                    while True:
                        # Box ranges from [0, box_size]; sample uniformly in [d/2, bl-d/2] 
                        c = np.random.uniform(low = d/2, high = self.box_width_subs-d/2, size = (3,))

                        if(_subdomain_cc(d, c, i) is False):
                            break
                        # Find where to insert idx
                    _insert_cell(c, i, j)
                    k += 1
            
            #Flatten coordinates for each axis over all subcells and write to NumPy array
            x_c = list(chain(*xcoords_levels))
            y_c = list(chain(*ycoords_levels))
            z_c = list(chain(*zcoords_levels))
            coordinates = (np.row_stack((np.array(x_c), np.array(y_c), np.array(z_c)))).T
            return coordinates

            

        #Arrange particles by levels corresponding to provided collection intervals
        id_levels = _order_particles_by_level()

        #Initialize particle coordinate containers for each level
        xcoords_levels = [[] for _ in id_levels]
        ycoords_levels = [[] for _ in id_levels]
        zcoords_levels = [[] for _ in id_levels]

        #Initialize coordinate to subcell containers
        xcells_levels = [[[] for _ in range(self.N_cells)] for _ in id_levels]       
        ycells_levels = [[[] for _ in range(self.N_cells)] for _ in id_levels]
        zcells_levels = [[[] for _ in range(self.N_cells)] for _ in id_levels]

        #Initialize particle diameter container
        diameters_levels = [[diameters[j] for j in range(lo, hi)] for (lo, hi) in id_levels]
        N_levels = [hi-lo for (lo, hi) in id_levels]

        # Generate the particles 
        coords = generate_coordinates()
        return coords

    def generate_coordinates(self) -> np.ndarray:
        # Assign the particles to a subdivided volume and core, shift returned coordinates and concatenate

        # Gather the raw coordinates
        with Pool(16) as pool:
            coords_raw_list = pool.map(self._CoordinateGen_, self.diameters_subdivided)
        
        # Shift the coordinates by the required amount
        Nelems = [np.shape(coords)[0] for coords in coords_raw_list]
        shift = np.zeros((sum(Nelems), 3), dtype=float)
        ind = 0
        for i in range(self.Ndivs):
            for j in range(self.Ndivs):
                for k in range(self.Ndivs):
                    shift[ind:ind+Nelems[i+j+k], 0] = self.box_width_subs*i
                    shift[ind:ind+Nelems[i+j+k], 1] = self.box_width_subs*j
                    shift[ind:ind+Nelems[i+j+k], 2] = self.box_width_subs*k

        coordinates = np.concatenate(coords_raw_list) + shift        
        return coordinates

    def _check_diameters(self):
        """Checks that diameters are sorted descending"""
        self.diameters = np.sort(self.diameters)[::-1]
        #assert np.array_equal(self.diameters, np.sort(self.diameters)[::-1]), f'Diameters must be sorted descending to generate coordinates correctly.'
    
    def _check_collection_intervals(self):
        """Checks that collection intervals are ordered descending and are within diameter range"""
        #Sort descending
        self.collection_intervals = np.sort(self.collection_intervals)[::-1]

        #assert np.array_equal(self.collection_intervals, np.sort(self.collection_intervals)[::-1]), f'Collection intervals must be sorted descending to generate coordinates correctly.'
        assert max(self.collection_intervals) - np.max(self.diameters) < 1e-1, f'Upper collection interval does not correspond to largest particle diameter.'
        assert min(self.collection_intervals) > np.min(self.diameters), f'Lower collection interval is smaller than smallest particle diameter.'


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
        return list(np.unique(collection_intervals))

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
        
        #Switch order in packing
        self.collection_intervals = self.collection_intervals[::-1]
        self.particles.sort_by_diameters(order='descending')

        #Generate particle coordinates        
        self.particles.coordinates = CoordinatesGenerator(self.particles.diameters, self.collection_intervals).generate_coordinates()
    
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