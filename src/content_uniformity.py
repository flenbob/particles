from dataclasses import dataclass, field
import math 
from multiprocessing import Pool
from itertools import product

import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
@dataclass
class Stange:
    """Stange COV calculation given input PSD and total mass/volume"""
    diameters: np.ndarray
    types: np.ndarray
    flag : str              # Specifies either mass or volume

    type_densities: list[float] = field(default_factory=list)

    #Set by post init
    types_id: list[int] = field(default_factory=list, init=False, repr=False)
    d_types: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    D63: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        #Number of types
        N_types = int(self.types.max())
        self.types_id = list(range(N_types))

        #Diameters by type
        self.d_types = [self.diameters[self.types == i+1] for i in self.types_id]

        #Calculate D63 moment ratio
        self.D63 = np.array([(self.d_types[i]**6).sum()/(self.d_types[i]**3).sum() for i in self.types_id])

    def _calculate_by_volume(self, Vp: np.ndarray) -> np.ndarray:
        """Calculates Stange COV with respect to volume fractions"""
        #Calculate typewise volume fractions tau
        V_types = np.array([(self.d_types[i]**3).sum() for i in self.types_id])
        V_tot = V_types.sum()
        tau = V_types/V_tot

        #Typewise COV for each total Vume input V
        cov = np.zeros((len(self.types_id), Vp.shape[0]), dtype=float)
        for i in self.types_id:
            cov_temp = (1-tau[i])**2/tau[i]*self.D63[i]
            for j in self.types_id:
                if j != i:
                    cov_temp += tau[j]*self.D63[j]
            cov[i, :] = np.sqrt(np.pi*cov_temp/(6*Vp))
        return cov
    
    def _calculate_by_mass(self, G_particles: np.ndarray) -> float:
        """Calculates Stange COV with respect to M fractions"""
        #Check that mass densities are provided
        assert len(self.type_densities) == len(self.types_id), f'M densities {self.type_densities} do not correspond to the number of types {self.types_id}.'
        assert self.type_densities.all() > 0, 'M densities must be positive.'

        #Calculate typewise mass fractions tau
        M_types = np.array([self.type_densities[i]*(self.d_types[i]**3).sum() for i in self.types_id])
        M_tot = M_types.sum()
        tau = M_types/M_tot

        #Typewise COV for each total mass input G
        cov = np.zeros((len(self.types_id), G_particles.shape[0]), dtype=float)
        for i in self.types_id:
            cov_temp = (1-tau[i])**2/tau[i]*self.type_densities[i]*self.D63[i]
            for j in self.types_id:
                if j != i:
                    cov_temp += tau[j]*self.type_densities[j]*self.D63[j]
            cov[i, :] = np.sqrt(np.pi*cov_temp/(6*G_particles))
        return cov
    
    def calc_CU(self, X: np.ndarray) -> np.ndarray:
        match self.flag:
            case "mass":
                cov = self._calculate_by_mass(X)
            case "volume":
                cov = self._calculate_by_volume(X)
            case other:
                raise('Incorrect flag option. Provide either "mass" or "volume')
        return cov
    
@dataclass
class COVPredictor:
    #Continuation scale, i.e. prediction of COV x times beyond last value of X_data 
    diameters: np.ndarray
    type_ids: np.ndarray
    flag: str                  # Specifies mass or volume
    rescale_factor : float = 1

    type_densities: np.ndarray = field(default_factory=list)

    cov_mean_pred: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    cov_mean_pred_contd: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    cov_std_pred: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    cov_std_pred_contd: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    x_pts: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    x_pts_contd: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    
    def predict(self, X: np.ndarray, Y: np.ndarray) -> None: 
        #Estimate COV given data
        estimator = COVEstimator()
        self.cov_mean, self.cov_std, x_pts = estimator.fit_predict(X, Y)

        # Calculate the cell volume/mass
        unique_ids = np.unique(self.type_ids)
        V = np.array([np.pi/6*np.sum(self.diameters[self.type_ids==uq_id]**3) for uq_id in unique_ids])
        match self.flag:
            case "mass":
                Xfinal = np.sum(self.type_densities*V)
            case "volume": 
                Xfinal = np.sum(V)
            case other: 
                raise("Wrong flag; choose either 'mass' or 'volume'!")
        
        # Rescale x_pts and X_final to real units again 
        x_pts = x_pts*self.rescale_factor**3            # ym^3 or mg 
        Xfinal = Xfinal*self.rescale_factor**3          # ym^3 or mg 

        #Create data points up to the size of the entire cell
        self.x_pts = x_pts
        x_pts_contd = np.linspace(x_pts[-1], Xfinal, estimator.Npred)
        self.x_pts_contd = x_pts_contd

        #Predict COV for each component
        for uq_id in unique_ids:
            CF_mean = COVCurveFitter()
            CF_mean.fit(x_pts, self.cov_mean[uq_id-1,:], self.cov_std[uq_id-1, :])
            plt.scatter(np.log(x_pts), np.log(self.cov_mean[uq_id-1,:]), color='g', label='mean')
            plt.scatter(np.log(x_pts), np.log(self.cov_std[uq_id-1,:]), color='r', label='std')
            plt.legend()
            plt.show()
            print(f"Params mean: {CF_mean.a, CF_mean.b}")

            #Curve fit to produce std of COV prediction 
            CF_std = COVCurveFitter()
            CF_std.fit(self.x_pts, self.cov_std[uq_id-1, :], 1)
            print(f"Params std: {CF_std.a, CF_std.b}")

            #Curve predict available points + continued range on both mean and std
            self.cov_mean_pred.append(CF_mean.predict(x_pts))
            self.cov_mean_pred_contd.append(CF_mean.predict(x_pts_contd))
            self.cov_std_pred.append(CF_std.predict(x_pts))
            self.cov_std_pred_contd.append(CF_std.predict(x_pts_contd))
        return 
                  
class COVEstimator:
    """Allows to calculate the COV of mass / volume fractions given sample mass / volume"""

    def __init__(self, K: int=10, Npred: int=500):
        self.K = K
        self.Npred = Npred
        self.h: float
        self.prop = 0.01

        # Data after fitting
        self.CV_mean: np.ndarray
        self.CV_std: np.ndarray
        self.X0: np.ndarray

    def fit_predict(self, X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
        # Get available Cv prediction range
        Xmax = np.min(X[:, -1, :])
        Xmin = np.max(X[:, 0, :])
        X0 = np.linspace(Xmin, Xmax, self.Npred)
        self.X0 = X0

        # Set bandwidth from the average distance between data points
        delta = self._avg_delta(X)
        print(f"Delta : {delta}")
        self.h = self.K*delta/(np.sqrt(-8*np.log(self.prop)))
        print(f"Bandwidth set : {self.h}")
        tau_estimate = self._nw_estimator_block(X, Y)

        # Std and mean across the FCC structure where VTOT is non-zero - i.e there are nothing sampled yet
        S = np.std(tau_estimate, axis = 2, ddof = 1)
        M = np.mean(tau_estimate, axis = 2)
        COV_raw = S/M

        # Calculate the mean and std of the CV across shifts
        COV_mean = np.nanmean(COV_raw, axis = 0)
        COV_std = np.nanstd(COV_raw, axis = 0, ddof = 1)
        
        #return values
        return COV_mean.T, COV_std.T, X0

    def _avg_delta(self, X: np.ndarray) -> float:
        """Calculate the average step in X for the data

        Args:
            X (np.ndarray): Array containing data

        Returns:
            float: Avergae step
        """

        # Shifts, Shellradii, Centers
        I, J, K = np.shape(X)

        DELTA = []
        for i in range(I):
            # Across shifts
            for k in range(K):
                # Across centers
                for j in range(1, J):
                    delta = X[i, j, k] - X[i, j-1, k]
                    DELTA.append(delta)
        DELTA = np.array(DELTA)
        deltas_mean = np.mean(DELTA)
        return deltas_mean
     
    def _nw_inner(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Nadaraya-Watson inner function"""
        # I is no Radii, J is components
        I, J = np.shape(Y)
        N = np.shape(self.X0)[0]
        MX = np.zeros((N, J), dtype = float)
        D = 1/(self.h*math.sqrt(2*math.pi))
        for n in range(N):
            # Over prediction points
            x0 = self.X0[n]
            K = 1/(2*self.h**2)*(X-x0)**2
            Kh = D*np.exp(-K)
            Sx = np.sum(Kh)
            if Sx == 0:
                # All weights are zero: Too far from any training data --> set NaN - ignored in calculation later
                MX[n, :] = np.nan
                continue
            
            for j in range(J):
                # Over components
                Khy = np.multiply(Kh, Y[: , j])
                Sy = np.sum(Khy)
                mx = Sy/Sx
                MX[n, j] = mx
            # Fractions should sum to one
            MX[n, :] = MX[n, :]/np.sum(MX[n, :])
        return MX

    def _nw_estimator_block(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Nadaraya-Watson estimator over X0 for different Shifts, Centers and Components"""
        # Order MXof X
        # Shifts, Shellradii, Centers, Components
        I, J, K, L = Y.shape
        N = self.X0.shape[0]

        tau_estimate = np.zeros((I, N, K, L), dtype = float)
        with Pool(16) as pool:
            for i in range(I):
                # For each shift
                CHUNKS = [(X[i, :, k], Y[i, :, k, :]) for k in range(K)]
                TAU = pool.starmap(self._nw_inner, CHUNKS)
                for k in range(K):
                    tau_estimate[i, :, k, :] = TAU[k]
        return tau_estimate

class COVCurveFitter:
    def __init__(self):
        # Parameters used for the curve fitting
        self.a: float
        self.b: float
        self.c: float
        self.d: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.a/x**(self.b)
    
    def fit(self, X: np.ndarray, Y: np.ndarray, SY: np.ndarray, verbose = False):
        # Generic fit function
        def _loss(p0):
            residuals = (Y - p0[0]/X**p0[1])**2
            weights = X**3/SY
            return (residuals*weights).sum()
        
        # Initial guessses
        b = 1/2
        a = np.sum(X**3*Y/(SY*X**b))/np.sum(X**3/(SY*X**(2*b)))

        # Minize in both
        min_both = minimize(_loss, x0 = [a, b], method='Nelder-Mead', tol=1e-6, options = {'maxiter': 20000}, bounds = ((0, None), (0, None)))
        params = min_both.x
        self.a, self.b =  params[0], params[1]
        print(f"Parameters : {params}")
        if verbose:
            print(min_both.message)
        return






    def calculate_by_mass(self, G_particles: np.ndarray) -> float:
        """Calculates Stange COV with respect to M fractions"""
        #Check that mass densities are provided
        assert len(self.type_densities) == len(self.types_id), f'M densities {self.type_densities} do not correspond to the number of types {self.types_id}.'
        assert self.type_densities.all() > 0, 'M densities must be positive.'

        #Calculate typewise mass fractions tau
        M_types = np.array([self.type_densities[i]*(self.d_types[i]**3).sum() for i in self.types_id])
        M_tot = M_types.sum()
        tau = M_types/M_tot

        #Typewise COV for each total mass input G
        cov = np.zeros((len(self.types_id), G_particles.shape[0]), dtype=float)
        for i in self.types_id:
            cov_temp = (1-tau[i])**2*self.type_densities[i]*self.D63[i]
            for j in self.types_id:
                if j != i:
                    cov_temp += tau[j]*self.type_densities[j]*self.D63[j]
            cov[i, :] = np.sqrt(np.pi*cov_temp/(6*self.type_densities))
        return cov

@dataclass
class CSSMDataGenerator:
    """Generation of data using the concentric spherical shell method (CSSM)."""
    #Init variables
    cell_matrix: np.ndarray
    diameters: np.ndarray
    coordinates: np.ndarray
    type_ids: np.ndarray
    n_workers: int

    type_densities: list[float] = field(default_factory=list)
    dr: float = 0.1         #Concentric spherical shell thickness
    n_fcc: int = 2          #Number of coordinates within the length of one layer of fcc structure
    n_shift: int = 50       #Number of coordinate shifts

    #Post init variables
    r_shells: np.ndarray = field(init=False, repr=False)
    r_max: float = field(init=False, repr=False)
    l_max: float = field(init=False, repr=False)
    n_shells: int = field(init=False, repr=False)
    n_centers: int = field(init=False, repr=False)
    n_types: int = field(init=False, repr=False)
    PV: np.ndarray = field(init=False, repr=False)
    
    def __post_init__(self):
        #Number of types (components) in packing
        self.n_types = int(self.type_ids.max())

        # Maximum particle radius
        self.r_max = 1/2*self.diameters.max()

        # Number of center coordinates in fcc structure
        self.n_centers = self.n_fcc**3+3*(self.n_fcc-1)**2*self.n_fcc

        self.PV = self._calculate_pv_outer(self.n_workers)

    def generate_by_mass(self) -> tuple[np.ndarray, np.ndarray]:
        #Check provided densities
        assert len(self.type_densities) == self.n_types, f'M densities {self.type_densities} do not correspond to the number of types {self.n_types}.'
        assert self.type_densities.all() > 0, 'M densities must be positive.'

        #4D matrices used as data containers. 
        # For each shift, shell, fcc center coordinate and type, calculate:
        # - mass
        # - total mass
        # - mass fractions 
        M = np.zeros((self.n_shift, 
                      self.n_shells, 
                      self.n_centers, 
                      self.n_types), dtype = float)
        MTOT = np.zeros((self.n_shift, 
                         self.n_shells, 
                         self.n_centers), dtype = float)
        MFR = np.zeros((self.n_shift, 
                        self.n_shells, 
                        self.n_centers, 
                        self.n_types), dtype = float)

        #Total mass and mass fractions
        for i in range(self.n_shift):
            # For each shift
            for j in range(self.n_shells):
                # For each shell
                for k in range(self.n_centers):
                    # For each center coordinate
                    # Calculate individual Mes
                    M[i, j, k, :] = np.multiply(self.PV[i, : , k, j], self.type_densities)

                    # Calculate the total M - summing across components
                    MTOT[i, j, k] = np.sum(M[i, j, k, :])

                    # Calculate M fractions
                    if MTOT[i, j, k] == 0:
                        MFR[i, j, k, :] = 1/self.n_types
                    else: 
                        MFR[i, j, k, :] = M[i, j, k, :]/MTOT[i, j, k]

        return MTOT, MFR

    def generate_by_volume(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculates CSSM particle volumes and volume fractions for every shift, shell, fcc coordinate and type

        Returns:
            tuple[np.ndarray, np.ndarray]: VTOT: Total volume for each shift, shell and fcc coord. VFR: Volume fraction for each shift, shell, fcc coord and type
        """
        VTOT = np.zeros((self.n_shift, 
                         self.n_shells, 
                         self.n_centers), dtype = float)
        VFR = np.zeros((self.n_shift, 
                        self.n_shells, 
                        self.n_centers, 
                        self.n_types), dtype = float)

        # Total volume and volume fractions
        for i in range(self.n_shift):
            for j in range(self.n_shells):
                for k in range(self.n_centers):
                    # For each center coordinate
                    # Calculate V fractions
                    VTOT[i, j, k] = np.sum(self.PV[i, :, k, j])
                    if VTOT[i, j, k] == 0:
                        VFR[i, j, k, :] = 1/self.n_types
                    else: 
                        VFR[i, j, k, :] = self.PV[i, :, k, j]/VTOT[i, j, k]

        return VTOT, VFR

    def _generate_fcc_coordinates(self) -> tuple[np.ndarray, float]:
        """Generates the necessary sampling shell structure"""
        #Set length by minimum sidelength of the cell
        l = min([self.cell_matrix[0,0], self.cell_matrix[1, 1], self.cell_matrix[2,2]])

        #Outer shell radius
        l_max = l/(2*np.sqrt(2)*(self.n_fcc-1) + 2)

        #Lattice parameter
        A = 2*math.sqrt(2)*l_max

        #Shift vector
        x0 = np.array([l_max, l_max, l_max])  

        #Initialize coordinate array
        Ntot = self.n_fcc**3 + 3*(self.n_fcc-1)**2*self.n_fcc
        coordinates = np.zeros((Ntot, 3), dtype = float)

        #Keep track of added shell count and generate outer
        ind = 0
        for i in range(self.n_fcc):
            for j in range(self.n_fcc):
                for k in range(self.n_fcc):
                    coordinates[ind, :] = np.array([A*i, A*j, A*k]) + x0
                    ind += 1

        #Generate inner
        for i in range(self.n_fcc-1): # X
            for j in range(self.n_fcc-1): # Y
                for k in range(self.n_fcc-1): # Z
                    coordinates[ind, :] = x0 + np.array([A/2, A/2, 0]) + np.array([A*i, A*j, A*k])
                    ind += 1
                    coordinates[ind, :] = x0 + np.array([A/2, 0, A/2]) + np.array([A*i, A*j, A*k])
                    ind += 1
                    coordinates[ind :] = x0 + np.array([0, A/2, A/2]) + np.array([A*i, A*j, A*k])
                    ind += 1

        #Add last three faces
        dx = x0 + np.array([A*(self.n_fcc-1), 0, 0]) + np.array([0, A/2, A/2])
        dy = x0 + np.array([0, A*(self.n_fcc-1), 0]) + np.array([A/2, 0, A/2])
        dz = x0 + np.array([0, 0, A*(self.n_fcc-1)]) + np.array([A/2, A/2, 0])
        for i in range(self.n_fcc-1):
            for j in range(self.n_fcc-1):
                coordinates[ind, :] = dx + np.array([0, A*i, A*j])
                ind += 1
                coordinates[ind, :] = dy + np.array([A*i, 0, A*j])
                ind += 1
                coordinates[ind, :] = dz + np.array([A*i, A*j, 0])
                ind += 1

        return coordinates, l_max

    def _overlap_volume(self, r_shell: float, r: np.ndarray, d: np.ndarray) -> float:
        """Calculates total overlap volume

        Args:
            r_shell (float): Shall radius
            r (np.ndarray): Particle sphere radii
            d (np.ndarray): Particle sphere distance to shell center

        Returns:
            float: Total overlap volume
        """
        V = np.pi/12*np.sum((r_shell+r-d)**2*(d**2+2*d*(r+r_shell)-3*(r**2+r_shell**2)+6*r*r_shell)/d)
        return V
    
    def _calculate_pv_inner(self, coords: np.ndarray, r: np.ndarray, r_shells: np.ndarray, center_k):
        """Calculates the particle volume for each shell 

        Args:
            coords (np.ndarray): _description_
            r (_type_): _description_
            r_shells (_type_): _description_
            center_k (_type_): _description_

        Returns:
            _type_: _description_
        """
        #Distance to particles from shell center
        d = np.linalg.norm(coords - center_k, axis=1)

        #Distance to outer edge of each particle from shell center
        m = d+r

        # Sort by distance
        sort_IDs = np.argsort(m)
        m = m[sort_IDs]
        r = r[sort_IDs]
        d = d[sort_IDs]

        #Total particle volume entirely inside shell
        V_inside_shell = 0

        #For each shell
        n_shells = r_shells.shape[0]
        PV_shell = np.empty(shape=(n_shells,), dtype=float)
        for l, r_shell in enumerate(r_shells):
            #Catch if shell is entirely within a particle (occurs for small shell radii)
            if (r_shell+d<r).any():
                PV_shell[l] = 4*np.pi/3*r_shell**3
                continue
            
            #Catch particles entirely within the shell
            inside_shell = np.searchsorted(m, r_shell)
            if m[inside_shell-1] > r_shell:
                inside_shell = 0
            V_inside_shell += 4*np.pi/3*(r[:inside_shell]**3).sum()

            # Remove these particles from consideration
            r, d, m = r[inside_shell:], d[inside_shell:], m[inside_shell:]

            # Collect particles that intersect and calculate overlap volume
            isect_shell = d-r < r_shell
            r_isect, d_isect = r[isect_shell], d[isect_shell]
            V_overlap_shell = self._overlap_volume(r_shell, r_isect, d_isect)
            
            #Particle volume l:th shell
            PV_shell[l] = V_inside_shell + V_overlap_shell
        return PV_shell

    def _calculate_pv_outer(self, n_workers: int) -> np.ndarray:
        #Cell translation matrix and origin
        #origin, H = self.cell_matrix[:, -1], self.cell_matrix[0:3, 0:3]
        H = self.cell_matrix

        # Gather the minimum coordinate
        dist_to_origin = np.linalg.norm(self.coordinates, axis = 1)
        min_ind = np.argmin(dist_to_origin)
        origin = self.coordinates[min_ind, :]

        #Set of translation vectors for wrapped coordinates (3, 27)
        n_set = np.array([p for p in product([0, 1, -1], repeat=3)]).T
        H_set = (H.T@n_set).T

        # Get fcc structure coordinates
        print(f"Generating FCC structure with NFCC : {self.n_fcc}, yielding {self.n_centers} central coordinates")
        fcc_coords, self.l_max = self._generate_fcc_coordinates()
        print("Finished FCC structure")
        
        # Radial shells with shell width dr
        self.r_shells = np.arange(self.dr, self.l_max, self.dr)
        self.n_shells = self.r_shells.shape[0]

        # Generate random, uniformly sampled points within the cell
        print("Sampling random vectors")
        T = np.random.uniform(low = 0, high = 1, size = (3, self.n_shift))
        shift_vectors = (H@T).T
        print("Finished sampling shift vectors")

        #Particle volume for each shell center, shell size and type
        PV = np.zeros((self.n_shift, self.n_types, self.n_centers, self.n_shells), dtype = float)

        # Index in order i, j, k, l
        with Pool(n_workers) as pool:
            for i in range(self.n_types):
                #Typewise (component) KD-trees
                print(f"Concentric shells on component {i+1}")
                ids = (self.type_ids==i+1)
                coords = self.coordinates[ids, :]
                diams = self.diameters[ids]

                #Create unwrapped (uw) particles in all 2^3-1 directions around cell
                n_type = coords.shape[0]
                coords_uw = np.resize(coords, (27*n_type, 3))
                H_set_uw = np.repeat(H_set, n_type, axis=0)
                coords_uw = coords_uw + H_set_uw
                
                #Unwrap diameters
                diams_uw = np.resize(diams, (27*n_type, ))
                radii_uw = diams_uw/2

                #Construct KD Tree of coordinates for efficient neighbor-searching
                tree = cKDTree(coords_uw)

                coords_uw_list = self.n_centers*[0]
                r_uw_list = self.n_centers*[0]
                r_shells_list = self.n_centers*[0]
                center_list = self.n_centers*[0]

                for j in range(self.n_shift):
                    # For each shift vector
                    # Shift coordinates
                    coords_K = origin + fcc_coords + shift_vectors[j, :]     # (n_centers, 3)
                    for k, center_k in enumerate(coords_K):
                        center_list[k] = center_k

                        # Get particle ids from KDTree
                        tree_IDs = tree.query_ball_point(center_k, self.l_max+self.r_max, workers=n_workers)
                        tree_IDs = np.array(tree_IDs)

                        #Distribute neighboring coordinates for each center coordinate
                        #print(tree_IDs)
                        coords_uw_list[k] = coords_uw[tree_IDs, :]
                        r_uw_list[k] = radii_uw[tree_IDs]
                        r_shells_list[k] = self.r_shells

                    #Calculate particle volumes over shells given list of neighboring coordinate
                    # for each center coordinate, in parallel
                    PV_list = pool.starmap(self._calculate_pv_inner, 
                                       zip(coords_uw_list, r_uw_list, r_shells_list, center_list))
                    
                    #Unpack values
                    for k in range(self.n_centers):
                        PV[j, i, k, :] = np.array(PV_list[k])
        return PV