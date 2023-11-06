from dataclasses import dataclass, field
import math 
from multiprocessing import Pool
from itertools import product

import numpy as np
from scipy.stats import truncnorm
from scipy.spatial import cKDTree

from .table_params import Param
from .particles import Particles

@dataclass
class Stange:
    """Stange COV calculation given input PSD and total mass/volume"""

    def cov_given_mass_particles(self, particles: Particles, G: float | np.ndarray = None) -> np.ndarray:
        """Estimates the Stange COV given mass of sampled particles

        Args:
            particles (Particles): Particles containing diameter, type ids, rescale factor 
            G (np.ndarray | float, optional): Float or Numpy-array of masses to estimate COV at. If not provided will estimate given total mass of provided particles. Defaults to None.

        Returns:
            np.ndarray: Estimated COV's
        """
        
        #Calculate D63 moment ratio
        D63 = np.array([(d_type**6).sum()/(d_type**3).sum() 
                        for d_type in particles.diameter_types])
        
        #Calculate typewise mass fractions
        mtypes = particles.mass_types
        mtot = mtypes.sum()
        mf = mtypes/mtot
        Gs = np.array([mtot]) if G is None else np.array(G)
        
        cov = np.zeros((particles.types.shape[0], Gs.shape[0]))
        for (i, tau_i, rho_i, D63_i) in zip(particles.types, mf, particles.density_types, D63):
            j = particles.types[particles.types != i]
            a = np.pi/(6*Gs)
            b = (1-tau_i)**2/tau_i * rho_i * D63_i
            c = np.sum(mf[j] * particles.density_types[j] * D63[j])
            cov[i, :] = np.sqrt(a * (b + c))
        return cov

    def cov_given_mass_params(self, params: dict[np.ndarray], G: float) -> np.ndarray:
        """Estimates the Stange COV given input table params and input total mass"""
        rho = params[Param.density]
        mf = params[Param.mass_fraction]
        mu = params[Param.mu]
        sigma = params[Param.sigma]

        #6:th and 3:rd moment of lognormal and moment ratio
        E_D6 = np.exp(6*mu+18*sigma**2)
        E_D3 = np.exp(3*mu+9/2*sigma**2)
        D63 = E_D6/E_D3

        #Component-wise COV
        n = mf.shape[0]
        cov = np.zeros((n,))
        indicies = np.array((range(n)))
        for (i, tau_i, rho_i, D63_i) in zip(indicies, mf, rho, D63):
            j = indicies[indicies != i]
            a = np.pi/(6*G)
            b = (1-tau_i)**2/tau_i * rho_i * D63_i
            c = np.sum(mf[j] * rho[j] * D63[j])
            cov[i] = np.sqrt(a * (b + c))
        return cov

    def mass_given_cov_params(self, params: dict[np.ndarray], cov: float | list | np.ndarray = None) -> np.ndarray:
        """Estimates component-wise masses [µg] required to satisfy input Stange COV"""
        #Unpack input params dict
        rho = params[Param.density]
        mf = params[Param.mass_fraction]
        mu = params[Param.mu]
        sigma = params[Param.sigma]
        
        #Number of components
        n = mf.shape[0]
        
        #Check type of input cov
        if cov is None:
            cov = params[Param.cv]
        if isinstance(cov, float):
            cov = n*[cov]
        assert len(cov) == n, f'Incorrect number of component-wise COV:s provided {len(cov)}, should be {n}.'

        #6:th and 3:rd moment of lognormal and moment ratio
        E_D6 = np.exp(6*mu+18*sigma**2)
        E_D3 = np.exp(3*mu+9/2*sigma**2)
        D63 = E_D6/E_D3

        #Component-wise COV
        Gs = np.zeros((n,))
        indicies = np.array((range(n)))
        for (i, tau_i, rho_i, D63_i, cov_i) in zip(indicies, mf, rho, D63, cov):
            j = indicies[indicies != i]
            a = np.pi/(6*cov_i**2)
            b = (1-tau_i)**2/tau_i * rho_i * D63_i
            c = np.sum(mf[j] * rho[j] * D63[j])
            Gs[i] = a * (b + c)
        return Gs

@dataclass
class COVPredictor:
    """Predicts the COV w.r.t total mass given input CSSM data"""
    particles: Particles
    alpha: float = 0.05

    cov_mean: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    cov_mean_pred: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    cov_mean_pred_contd: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

    #Confidence intervals
    cov_upper_pred: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    cov_lower_pred: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    cov_upper_pred_contd: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    cov_lower_pred_contd: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

    #Standard deviation
    cov_std: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    cov_std_pred: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

    #Total particle mass points
    x_pts: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    x_pts_contd: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    
    def predict(self, X: np.ndarray, Y: np.ndarray) -> None: 
        #Estimate COV given data
        estimator = COVEstimator()
        cov_mean, cov_std, x_pts = estimator.fit_predict(X, Y)
       
        #Remove first point
        cov_mean = cov_mean[:, 1:]
        cov_std = cov_std[:, 1:]
        x_pts = x_pts[1:]
        
        #Total particle mass
        mass = self.particles.mass_types.sum()
        
        #Create data points up to the size of the entire cell
        self.x_pts = x_pts
        x_pts_contd = np.linspace(x_pts[-1], mass, estimator.Npred)
        self.x_pts_contd = x_pts_contd

        #Predict COV for each component
        for cov_m, cov_s in zip(cov_mean, cov_std):
            CF_mean = COVCurveFitter()
            CF_mean.fit(x_pts, cov_m)
            #print(f"Params mean: {CF_mean.a, CF_mean.b}")

            #Curve fit to produce std of COV prediction 
            CF_std = COVCurveFitter()
            CF_std.fit(self.x_pts, cov_s)
            #print(f"Params std: {CF_std.a, CF_std.b}")

            #Curve predict available points + continued range on both mean and std
            CF_mean_pred = CF_mean.predict(x_pts)
            CF_mean_pred_contd = CF_mean.predict(x_pts_contd)

            CF_std_pred = CF_std.predict(x_pts)
            CF_std_pred_contd = CF_std.predict(x_pts_contd)

            # Construct confidence intervals
            clip_a0, clip_b0 = -CF_mean_pred/CF_std_pred, np.full(np.shape(CF_std_pred), np.inf)           # Clips for truncated normal
            clip_ap, clip_bp = -CF_mean_pred_contd/CF_std_pred_contd, np.full(np.shape(CF_std_pred_contd), np.inf)

            N0 = np.size(CF_mean_pred)
            Np = np.size(CF_mean_pred_contd) 

            upper_pred = np.array([truncnorm.ppf(self.alpha/2, clip_a0[j], clip_b0[j], CF_mean_pred[j], CF_std_pred[j]) for j in range(N0)])
            lower_pred = np.array([truncnorm.ppf(1-self.alpha/2, clip_a0[j], clip_b0[j], CF_mean_pred[j], CF_std_pred[j]) for j in range(N0)])

            upper_pred_contd = np.array([truncnorm.ppf(self.alpha/2, clip_ap[j], clip_bp[j], CF_mean_pred_contd[j], CF_std_pred_contd[j]) for j in range(Np)])
            lower_pred_contd = np.array([truncnorm.ppf(1-self.alpha/2, clip_ap[j], clip_bp[j], CF_mean_pred_contd[j], CF_std_pred_contd[j]) for j in range(Np)])

            #Write to class variables
            self.cov_mean.append(cov_m)
            self.cov_std.append(cov_s)
            
            self.cov_mean_pred.append(CF_mean_pred)
            self.cov_std_pred.append(CF_std_pred)
            
            self.cov_upper_pred.append(upper_pred)
            self.cov_lower_pred.append(lower_pred)
            
            self.cov_upper_pred_contd.append(upper_pred_contd)
            self.cov_mean_pred_contd.append(CF_mean_pred_contd)
            self.cov_lower_pred_contd.append(lower_pred_contd)
                              
class COVEstimator:
    """Allows to calculate the COV of mass fractions given sample mass"""
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
        self.h = self.K*delta/(np.sqrt(-8*np.log(self.prop)))    
        tau_estimate = self._nw_estimator_block(X, Y)

        # Std and mean across the FCC structure where VTOT is non-zero - i.e there are nothing sampled yet
        S = np.std(tau_estimate, axis = 2, ddof = 1)
        M = np.mean(tau_estimate, axis = 2)
        COV_raw = S/M

        # Calculate the mean and std of the CV across shifts
        COV_mean = np.nanmean(COV_raw, axis = 0)
        COV_std = np.nanstd(COV_raw, axis = 0, ddof = 1)
        
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
        I, J = Y.shape
        N = self.X0.shape[0]
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
    """Fits power-law function of type f(x) = a*x^b to COV data."""
    #The assumption that the sampled data is power-law distributed is strongly supported
    # by looking at a log-log plot of the datapoints, showing almost perfect linear correlation
    def __init__(self):
        self.a: float
        self.b: float
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.a*x**(self.b)
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        #Fit parameters using least squares
        A = np.vstack([np.log(X), np.ones(len(X))]).T
        b, a = np.linalg.lstsq(A, np.log(Y), rcond=None)[0]
        self.a, self.b = np.exp(a), b
    
@dataclass
class CSSMDataGenerator:
    """Generation of data using the concentric spherical shell method (CSSM)."""
    #Init variables
    cell_matrix: np.ndarray
    particles: Particles
    n_workers: int
    
    dr: float = field(default=0.1, repr=False)      #Concentric spherical shell thickness
    n_fcc: int = field(default=2, repr=False)       #Number of coordinates within the length of one layer of fcc structure
    n_shift: int = field(default=10, repr=False)    #Number of coordinate shifts

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
        self.n_types = self.particles.types.shape[0]
        
        #Check provided densities
        assert len(self.particles.density_types) == self.n_types, f'M densities {self.particles.density_types} do not correspond to the number of types {self.n_types}.'
        assert self.particles.density_types.all() > 0, 'M densities must be positive.'
        
        #Rescale so smallest diameter is 1 in this case
        self.particles.diameters = self.particles.diameters/self.particles.diameters.min()         

        #Maximum particle radius
        self.r_max = 1/2*self.particles.diameters.max()

        #Number of center coordinates in fcc structure
        self.n_centers = self.n_fcc**3+3*(self.n_fcc-1)**2*self.n_fcc

        #Calculate particle volumes for each type, center-coord, shell and shift
        self.PV = self._calculate_pv_outer(self.n_workers)

    def generate_by_mass(self) -> tuple[np.ndarray, np.ndarray]:
        """Generates CSSM data by mass, i.e. obervations of masses and mass fractions
        given different shift vectors, radial shells, fcc center coordinates and component. 

        Returns:
            tuple[np.ndarray, np.ndarray]: CSSM data
        """
        #Mass fraction observations 
        mass_fractions = np.zeros((self.n_shift, 
                                   self.n_shells, 
                                   self.n_centers, 
                                   self.n_types), dtype = float)
        
        #Calculate masses (use rescale factor to get correct units)
        masses = self.particles.rescale_factor**3*self.PV*self.particles.density_types
        
        #Total masses
        mass_total = np.sum(masses, axis=3)
        
        #Calculate mass fractions
        mask0 = mass_total == 0
        mask = mass_total != 0
        mass_fractions[mask0] = 1/self.n_types
        mass_fractions[mask] = masses[mask] / mass_total[mask, np.newaxis]
        return mass_total, mass_fractions
    
    def _generate_fcc_coordinates(self) -> tuple[np.ndarray, float]:
        """Generates the necessary sampling shell structure using fcc crystal structure
        (to maximise covered volume of the non-overlapping concentric shells within the cubic cell.)"""
        #Set length by minimum sidelength of the cell
        l = min([self.cell_matrix[0, 0], self.cell_matrix[1, 1], self.cell_matrix[2, 2]])

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
        # Distance to particles from shell center
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
            # If m is empty there are no particles left to check
            if m.shape[0] == 0:
                return PV_shell
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
        origin, H = self.cell_matrix[:, -1], self.cell_matrix[0:3, 0:3]

        #Set of translation vectors for wrapped coordinates (3, 27)
        n_set = np.array([p for p in product([0, 1, -1], repeat=3)]).T
        H_set = (H.T@n_set).T

        # Get fcc structure coordinates
        print(f"Generating FCC structure with nfcc: {self.n_fcc}, yielding {self.n_centers} central coordinates")
        fcc_coords, self.l_max = self._generate_fcc_coordinates()
        
        # Radial shells with shell width dr
        self.r_shells = np.arange(self.dr, self.l_max, self.dr)
        self.n_shells = self.r_shells.shape[0]
        
        # Generate random, uniformly sampled points within the cell
        T = np.random.uniform(low = 0, high = 1, size = (self.n_shift, 3))
        shift_vectors = T@H

        #Particle volume for each shell center, shell size and type
        PV = np.zeros((self.n_shift, 
                      self.n_shells, 
                      self.n_centers, 
                      self.n_types), dtype = float)

        # Index in order i, j, k, l
        with Pool(n_workers) as pool:
            for l in range(self.n_types):
                #Typewise (component) KD-trees
                print(f"Concentric shells on component {l+1}")
                ids = (self.particles.type_ids==l+1)
                coords = self.particles.coordinates[ids, :]
                diams = self.particles.diameters[ids]

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
                for i, shift_vector in enumerate(shift_vectors):
                    #Generate center coordinates for given particle type
                    coords_K = origin + fcc_coords + shift_vector
                    tree_IDs = tree.query_ball_point(coords_K, self.l_max+self.r_max, workers=n_workers)
                
                    #Distribute neighboring coordinates for each center coordinate
                    coords_uw_K = [coords_uw[tree_IDs[k]] for k in range(self.n_centers)]
                    r_uw_K = [radii_uw[tree_IDs[k]] for k in range(self.n_centers)]
                    r_shells_K = [self.r_shells for _ in range(self.n_centers)]

                    #Calculate particle volumes over shells given list of neighboring coordinate
                    # for each center coordinate, in parallel
                    PV_list = np.array(pool.starmap(self._calculate_pv_inner, 
                                    zip(coords_uw_K, r_uw_K, r_shells_K, coords_K)))
     
                    for k in range(self.n_centers):
                        PV[i, :, k, l] = PV_list[k]
        return PV