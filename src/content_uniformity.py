from dataclasses import dataclass, field
import numpy as np
from itertools import product
from numba import jit
from scipy.spatial import cKDTree
from src import Particles
from enum import Enum, auto


class Attribute(Enum):
    """Attribute names for Coefficient of Variation"""
    mass_fraction = auto()
    volume_fraction = auto()
    concentration = auto()

@dataclass
class Stange:
    particles: Particles
    
    def _calculate_properties(self, attribute: Attribute, comp_densities: list = None):
        comp_ids = [comp_id for comp_id in range(1, np.max(self.particles.types_ids))]

        #Component-wise diameters
        comps_particles_diameters = []

        #Component-wise attribute
        comps_particles_attribute = []
        comps_attribute_std = []
        comps_attribute_mean = []
        comps_attribute = []

        #Component-wise concentration properties
        for comp_id in comp_ids:
            diameters = self.particles.diameters[self.particles.type_ids == comp_id]
            comps_particles_diameters.append(diameters)

            if attribute == Attribute.mass_fraction:
                particles_attribute = [np.pi/6*diam**3 for diam in diameters]
            elif attribute == (Attribute.volume_fraction or Attribute.concentration):
                particles_attribute = [np.pi/6*diam**3 for diam in diameters]


            else:
                raise(f'Provided attribute {attribute} is not valid. Available are: ')
            
            comps_particles_attribute.append(particles_attribute)

            attribute_std = np.std(particles_attribute)
            comps_attribute_std.append(attribute_std)

            attribute_mean = np.mean(particles_attribute)
            comps_attribute_mean.append(attribute_mean)

            attribute_sum = np.sum(particles_attribute)
            comps_attribute.append(attribute_sum)

        comps_attribute = np.array(comps_attribute)
        comps_attribute_fractions = comps_attribute / np.sum(comps_attribute)

        #Stange COV
        comps_cov = []
        for comp_id in comp_ids:
            #Other comp ids
            non_ids = list(set(comp_ids) - set([comp_id]))

            #First term
            term = ((1-comps_attribute_fractions)/comps_attribute_fractions)**2*\
                comps_attribute_fractions*comps_attribute_mean*\
                (1+comps_attribute_std**2/comps_attribute_mean)
            
            #Second sum term
            sum_term = comps_attribute_fractions[non_ids]*comps_attribute_mean[non_ids]*\
                (1+comps_attribute_std[non_ids]**2/comps_attribute_mean[non_ids])
            
            #COV calculation
            comps_cov.append(1/attribute_sum*(term + sum_term))






    def calculate_by_attribute_fraction(self):
        pass
    def calculate_by_volume_fraction(self):
        pass
    def calculate_by_concentration(self):
        pass

@dataclass
class Hilden:
    pass

@dataclass
class ConcentricSphericalShells:
    particles: Particles
    cell_matrix: np.ndarray

    def _generate_FCC_structure(self, N: int) -> tuple[np.ndarray, float]:
        """Generates a face-centered-cubic lattice structure of equally sized spheres

        Args:
            N (int): Number of spheres per row

        Returns:
            tuple[np.ndarray, float]: Sphere center coordinates and sphere radii
        """

        #Translation matrix and cell origin coordinates
        H = self.cell_matrix[:3, :3]

        # Specify sidelength as the minimum span in x, y, z
        X = min([H[0,0], H[1, 1], H[2,2]])

        # Outer shell radius
        r = X/(2*np.sqrt(2)*(N-1) + 2)

        # Lattice parameter
        A = 2*np.sqrt(2)*r

        # Shift vector
        x0 = np.array([r, r, r])  

        # Initialize coordinate array
        Ntot = N**3 + 3*(N-1)**2*N
        coordinates = np.zeros((Ntot, 3), dtype = float)

        # Keep track of added shell count
        ind = 0

        # Generate outer
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    coordinates[ind, :] = np.array([A*i, A*j, A*k]) + x0
                    ind += 1

        # Generate inner
        for i in range(N-1): # X
            for j in range(N-1): # Y
                for k in range(N-1): # Z
                    coordinates[ind, :] = x0 + np.array([A/2, A/2, 0]) + np.array([A*i, A*j, A*k])
                    ind += 1
                    coordinates[ind, :] = x0 + np.array([A/2, 0, A/2]) + np.array([A*i, A*j, A*k])
                    ind += 1
                    coordinates[ind :] = x0 + np.array([0, A/2, A/2]) + np.array([A*i, A*j, A*k])
                    ind += 1

        # Add last three faces
        dx = x0 + np.array([A*(N-1), 0, 0]) + np.array([0, A/2, A/2])
        dy = x0 + np.array([0, A*(N-1), 0]) + np.array([A/2, 0, A/2])
        dz = x0 + np.array([0, 0, A*(N-1)]) + np.array([A/2, A/2, 0])
        for i in range(N-1):
            for j in range(N-1):
                coordinates[ind, :] = dx + np.array([0, A*i, A*j])
                ind += 1
                coordinates[ind, :] = dy + np.array([A*i, 0, A*j])
                ind += 1
                coordinates[ind, :] = dz + np.array([A*i, A*j, 0])
                ind += 1

        return coordinates, r

    @jit(nopython=True)
    def _calculate_overlap_volume(R: float, r: float, d: float) -> float:
        """Volume intersection between spheres - precompiled using numba

        Args:
            R (float): Shell radius
            r (float): Sphere radii
            d (float): Sphere distance to shell center

        Returns:
            float: Overlap volume
        """

        V = np.pi/12*np.sum((R+r-d)**2*(d**2+2*d*(r+R)-3*(r**2+R**2)+6*r*R)/d)
        return V

    def calculate_content_uniformity(self, dr: float=0.1) -> tuple[np.ndarray, np.ndarray]:
        """Calculates Content Uniformity of packing at last frame using CSSM with spheres distributed as FCC.

        Args:
            dr (float, optional): Shell thickness. Defaults to 0.1.

        Returns:
            R_shells, Cv (tuple[np.ndarray, np.ndarray]): Radial mixing scale and corresponding Coefficient of Variation as equally sized NumPy-arrays
        """

        #Translation matrix and cell origin coordinates
        H = self.cell_matrix[:3, :3]
        O = self.cell_matrix[:, -1]

        # Number of comps in packing
        N_comps = np.max(self.particles.type_ids).astype(int)

        # Maximum particle radius
        R_max = np.max(self.particles.diameters)/2

        #Set of translation vectors for wrapped coordinates (3, 27)
        n_set = np.array([p for p in product([0, 1, -1], repeat=3)]).T
        Hn = (H.T@n_set).T

        # Get coordinates from FCC and generate shells
        N = 3
        coordinates, L_max = self._generate_FCC_structure(N) 
        K = np.shape(coordinates)[0]

        # Shift coordinates
        coords_K = O + coordinates       # (K, 3)  
 
        # Radial shells with shell width dr
        R_shells = np.arange(dr, L_max, dr)
        N_shells = R_shells.shape[0]

        # Pre-allocate array for intersection volume for each comp and shell radii
        V_mat = np.zeros((K, N_shells, N_comps))

        # Iterate over each comp
        for k in range(1, N_comps+1):
            # Pick out particles of type k 
            IDs = (self.particles.type_ids==k)
            C_ = self.particles.coordinates[IDs, :]
            diams_ = self.particles.diameters[IDs]

            # Tile the coordinate array
            N = np.shape(C_)[0]
            coords_uw = np.resize(C_, (27*N, 3))
            Hn_uw = np.repeat(Hn, N, axis = 0)
            coords_uw = coords_uw + Hn_uw
            
            # Tile diams
            diams_uw = np.resize(diams_, (27*N, ))
            radii_uw = diams_uw/2

            # Construct KD Tree of coordinates
            tree = cKDTree(coords_uw)

            # Iterate over FCC coordinates
            for i in range(K):
                center_k = coords_K[i, :]

                # Get particle ids from KDTree
                tree_IDs = tree.query_ball_point(center_k, L_max+R_max, workers=-1)
                tree_IDs = np.array(tree_IDs)

                # Get distance to particles from shell center: d
                C_ = coords_uw[tree_IDs, :]
                d = np.linalg.norm(C_ - center_k, axis=1)
                r  = radii_uw[tree_IDs]

                # M denotes the distance to the outer edge of each particle from shell center
                M = d+r

                # Sort 
                sort_IDs = np.argsort(M)
                C_ = C_[sort_IDs, :]
                M = M[sort_IDs]
                r = r[sort_IDs]
                d = d[sort_IDs]
                
                # VB denotes the sum particle volume entirely in the shell
                VB = 0
                for j, R in enumerate(R_shells):

                    # VS is the shell volume
                    VS = 4*np.pi/3*R**3

                    # A indicates whether the shell is entirely in a particle
                    A = R+d<r
                    if np.any(A):
                        VA = 4*np.pi/3*R**3
                        V_mat[i, j, k-1] = VA
                    
                    # Catch particles entirely within the shell and add to VB
                    B = np.searchsorted(M, R)
                    if M[B-1]>R:
                        B = 0
                    VB += 4*np.pi/3*np.sum(r[:B]**3)

                    # Remove these particles from consideration
                    r, d, M = r[B:], d[B:], M[B:]

                    # Collect particles that intersect
                    Cind = d-r<R
                    rC, dC = r[Cind], d[Cind]
                    VC = self._calculate_overlap_volume(R, rC, dC)
                    V = VB + VC
                    V_mat[i, j, k-1] = V
                    if V/VS>1.03:
                        print(f"C: {V/VS}, r : {R}, coord : {center_k}")
                        print(f"xlo : {O}")

        # Calculate volume fractions 
        for i in range(K):
            for j in range(N_shells):
                V_tot = np.sum(V_mat[i, j, :])
                if V_tot == 0:
                    V_mat[i, j, :] = 1/N_comps
                else:
                    V_mat[i, j, :] = V_mat[i, j, :]/V_tot
        
        # Calculate standard deviations and means
        V_std = np.std(V_mat, axis=0, ddof=1)
        V_mean = np.mean(V_mat, axis=0)
        Cv = np.divide(V_std, V_mean)
        return R_shells, Cv
