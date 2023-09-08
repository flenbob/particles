import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
import h5py
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker
import os
from itertools import product

#TODO: Outdated. Is (will be) replaced with "datareader" and "plotter"

@dataclass
class PostProcessing:
    """Data handling and output of statistics using written HDF5 file
    """
    #Required init variables
    fpath_data: str

    #Frame independent, dependent and scalar keys
    frames: list[int] = field(default_factory=list, init=False, repr=False)
    keys_ind: list[str] = field(default_factory=list, init=False, repr=False)
    keys_d: list[str] = field(default_factory=list, init=False, repr=False)
    scalars: list[float] = field(default_factory=list, init=False, repr=False)
   
    def __post_init__(self) -> None:
        #Check datafile
        assert os.path.isfile(self.fpath_data), f"Path to HDF5 datafile in {self.fpath_data} is not valid."

        #Collect keys
        with h5py.File(self.fpath_data, 'r') as file:
            keys = file.keys()
            self.frames = [str(key) for key in sorted([int(key) for key in keys if key.isdigit()])]
            self.keys_ind = list(keys - self.frames)
            self.keys_d = list(file[self.frames[0]].keys())

            #Convert frame keys to int
            self.frames = [int(frame) for frame in self.frames]

            if 'scalars' in self.keys_ind:
                self.keys_ind.remove('scalars')
                self.scalars = list(file['scalars'].keys())
            else:
                print('No scalars found using key "scalars".')

    def read_data(self, keys: list[str], frames: int | list[int] = None) -> int | list:
        """Returns data from HDF5 file

        Args:
            keys (list[str]): File keys for data to read. See class function print_keys() for available keys.
            frame (int | list[int], optional): Selected frame(s). For multiple frames provide list of frame keys as integers. frame = -1 returns the last frame. Defaults to None.

        Returns:
            list: Returned data
        """
        #Single key provided as string
        if isinstance(keys, str):
            keys = [keys]

        #Check that every key is valid
        for key in keys:
            assert key in self.keys_ind+self.keys_d, f'Provided key "{key}" is invalid. See class function print_keys() for available keys.'

        #Selected frame(s) to read if provided
        if frames is not None:
            #Single frame provided as int
            if isinstance(frames, int):
                frames = [frames]

            #Check that every frame is valid
            for frame in frames:
                try:
                    self.frames[frame]
                except:
                    f'Sought frame {frame} is invalid. See class function "print_keys()" for available frame keys.'

            #Convert to list
            frames = [self.frames[frame] for frame in frames]
        else:
            frames = self.frames

        #Read keys from file
        with h5py.File(self.fpath_data, 'r') as file:
            data = []
            for key in keys:
                #Timestep independent
                if key in self.keys_ind:
                    data.append(np.array(file[key]))

                #Frame dependent
                elif key in self.keys_d:
                    #assert frames is not None, f"Need to provide frame key for frame dependent data using key {key}"
                    if len(frames) == 1:
                        data.append(np.array(file[f'{frames[0]}/{key}']))
                    else:
                        datasets = []
                        for frame in frames:
                            datasets.append(np.array(file[f'{frame}/{key}']))
                        data.append(datasets)

        #Return element if single sized list
        if len(data) == 1:
            return data[0]
        return data

    def print_keys(self) -> None:
        print(f'Frames: {self.frames}\nFrame dependent: {self.keys_d}\nFrame independent: {self.keys_ind}\nScalars: {self.scalars}')

    def Z_distribution(self) -> None:
        #Read contact data
        types, contacts = self.read_data(['type', 'contacts'], -1)

        #Componentwise contact lists
        contact_list = [contacts[types == i] for i in range(1, int(max(types))+1)]
        
        #Densities
        for i, contacts in enumerate(contact_list):
            Z_vals = np.arange(min(contacts), max(contacts))
            density = [contacts[contacts == Z_val].shape[0]/contacts.shape[0] for Z_val in Z_vals]
            plt.plot(Z_vals, density, label=i+1)
        
        plt.ylabel('Density')
        plt.xlabel('Coordination number')
        plt.xscale('log')
        plt.legend()
        plt.show()
        
    def Z_matrix(self) -> None:
        #Helper functions
        def fmt(x, pos):
            a, b = '{:.1e}'.format(x).split('e')
            b = int(b)
            return '{:.1f}'.format(10**float(x))

        def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
            """
            Create a heatmap from a numpy array and two lists of labels.

            Parameters
            ----------
            data
                A 2D numpy array of shape (M, N).
            row_labels
                A list or array of length M with the labels for the rows.
            col_labels
                A list or array of length N with the labels for the columns.
            ax
                A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
                not provided, use current axes or create a new one.  Optional.
            cbar_kw
                A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
            cbarlabel
                The label for the colorbar.  Optional.
            **kwargs
                All other arguments are forwarded to `imshow`.
            """

            if ax is None:
                ax = plt.gca()

            if cbar_kw is None:
                cbar_kw = {}

            # Plot the heatmap
            im = ax.imshow(data, **kwargs)

            # Create colorbar
            cbar = ax.figure.colorbar(ax.imshow(np.log10(data), **kwargs), ax=ax, format=ticker.FuncFormatter(fmt), **cbar_kw)
            cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

            # Show all ticks and label them with the respective list entries.
            ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
            ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

            # Let the horizontal axes labeling appear on top.
            ax.tick_params(top=True, bottom=False,
                        labeltop=True, labelbottom=False)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                    rotation_mode="anchor")

            # Turn spines off and create white grid.
            ax.spines[:].set_visible(False)

            ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
            ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
            ax.tick_params(which="minor", bottom=False, left=False)
            return im, cbar

        def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                            textcolors=("black", "white"),
                            threshold=None, **textkw):
            """
            A function to annotate a heatmap.

            Parameters
            ----------
            im
                The AxesImage to be labeled.
            data
                Data used to annotate.  If None, the image's data is used.  Optional.
            valfmt
                The format of the annotations inside the heatmap.  This should either
                use the string format method, e.g. "$ {x:.2f}", or be a
                `matplotlib.ticker.Formatter`.  Optional.
            textcolors
                A pair of colors.  The first is used for values below a threshold,
                the second for those above.  Optional.
            threshold
                Value in data units according to which the colors from textcolors are
                applied.  If None (the default) uses the middle of the colormap as
                separation.  Optional.
            **kwargs
                All other arguments are forwarded to each call to `text` used to create
                the text labels.
            """

            if not isinstance(data, (list, np.ndarray)):
                data = im.get_array()

            # Normalize the threshold to the images color range.
            if threshold is not None:
                threshold = im.norm(threshold)
            else:
                threshold = im.norm(data.max())/2.

            # Set default alignment to center, but allow it to be
            # overwritten by textkw.
            kw = dict(horizontalalignment="center",
                    verticalalignment="center")
            kw.update(textkw)

            # Get the formatter in case a string is supplied
            if isinstance(valfmt, str):
                valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

            # Loop over the data and create a `Text` for each "pixel".
            # Change the text's color depending on the data.
            texts = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                    text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                    texts.append(text)
            return texts
        
        #Read data at last frame
        contacts, types, contact_pairs, ID = self.read_data(['contacts', 'type', 'contact_pairs', 'ID'], -1)

        #Componentwise counts
        _, types_cnt = np.unique(types, return_counts=True)
        
        #Componentwise contact occurences
        N_types = int(max(types))
        contact_matrix = np.zeros((N_types, N_types))
        for contact_pair in contact_pairs[0]:
            I, J = int(types[contact_pair[0]-1]), int(types[contact_pair[1]-1])
            contact_matrix[I-1, J-1] += 1
            contact_matrix[J-1, I-1] += 1

        #Component-wise coordination number ()
        component_Z_matrix = contact_matrix/types_cnt

        #Add color scale to matrix
        fig, ax = plt.subplots()
        im, cbar = heatmap(component_Z_matrix, np.arange(N_types)+1, np.arange(N_types)+1, ax=ax,
                        cmap="rainbow", cbarlabel="Coordination number")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")

        fig.tight_layout()
        plt.show()

        #Pairwise coordination number()
        cnt_matrix = types_cnt[np.newaxis, :] + types_cnt[:, np.newaxis]
        pair_Z_matrix = contact_matrix/cnt_matrix
        pair_Z_matrix[np.diag_indices_from(pair_Z_matrix)]*=2
        
        fig, ax = plt.subplots()
        im, cbar = heatmap(pair_Z_matrix, np.arange(N_types)+1, np.arange(N_types)+1, ax=ax,
                        cmap="rainbow", cbarlabel="Coordination number")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")

        fig.tight_layout()
        plt.show()

    def _generate_FCC_structure(self, N: int) -> tuple[np.ndarray, float]:
        """Generates a face-centered-cubic lattice structure of equally sized spheres

        Args:
            N (int): Number of spheres per row

        Returns:
            tuple[np.ndarray, float]: Sphere center coordinates and sphere radii
        """

        # Read data at last frame
        H = self.read_data('H', -1)

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

    def content_uniformity(self, dr: float=0.1) -> tuple[np.ndarray, np.ndarray]:
        """Calculates Content Uniformity of packing at last frame using CSSM with spheres distributed as FCC.

        Args:
            dr (float, optional): Shell thickness. Defaults to 0.1.

        Returns:
            R_shells, Cv (tuple[np.ndarray, np.ndarray]): Radial mixing scale and corresponding Coefficient of Variation as equally sized NumPy-arrays
        """
        
        @jit(nopython=True)
        def V_ov(R: float, r: float, d: float) -> float:
            """Volume intersection between spheres - precompiled

            Args:
                R (float): Shell radius
                r (float): Sphere radii
                d (float): Sphere distance to shell center

            Returns:
                float: Overlap volume
            """

            V = np.pi/12*np.sum((R+r-d)**2*(d**2+2*d*(r+R)-3*(r**2+R**2)+6*r*R)/d)
            return V

        #Read data at last frame
        H, C_lo, C, diams, types = self.read_data(['H', 'origin', 'position', 'diameter', 'type'], -1)

        # Number of components in packing
        N_comps = np.max(types).astype(int)

        # Maximum particle radius
        R_max = np.max(diams)/2

        #Set of translation vectors for wrapped coordinates (3, 27)
        n_set = np.array([p for p in product([0, 1, -1], repeat=3)]).T
        Hn = (H.T@n_set).T

        # Get coordinates from FCC and generate shells
        N = 3
        coordinates, L_max = self._generate_FCC_structure(N) 
        K = np.shape(coordinates)[0]
      
        # # Temporary 
        # # Write LAMMPS data file
        # with open('crystalline.data','w') as fdata:
        #     # First line is a comment line 
        #     fdata.write('FCC crystal structure\n\n')

        #     #--- Header ---#
        #     # Specify number of atoms and atom types 
        #     fdata.write('{} atoms\n'.format(K))
        #     fdata.write('{} atom types\n'.format(1))

        #     # Specify box dimensions
        #     fdata.write('{} {} xlo xhi\n'.format(C_lo[0], C_lo[0] + H[0, 0]))
        #     fdata.write('{} {} ylo yhi\n'.format(C_lo[1], C_lo[1] + H[1, 1]))
        #     fdata.write('{} {} zlo zhi\n'.format(C_lo[2], C_lo[2] + H[2, 2]))
        #     fdata.write('{} {} {} xy xz yz\n'.format(H[0, 1], H[0, 2], H[1, 2]))
        #     fdata.write('\n')

        #     # Atoms section
        #     fdata.write('Atoms\n\n')

        #     # Write each position 
        #     for i in range(K):
        #         fdata.write('{} 1 {} 1 {} {} {}\n'.format(i+1, 2*L_max,  C_lo[0] + coordinates[i, 0], C_lo[1] +  coordinates[i, 1], C_lo[2] + coordinates[i, 2]))
        # return

        # Shift coordinates
        coords_K = C_lo + coordinates       # (K, 3)  
 
        # Radial shells with shell width dr
        R_shells = np.arange(dr, L_max, dr)
        N_shells = R_shells.shape[0]

        # Pre-allocate array for intersection volume for each component and shell radii
        V_mat = np.zeros((K, N_shells, N_comps))

        # Iterate over each component
        for k in range(1, N_comps+1):
            # Pick out particles of type k 
            IDs = (types==k)
            C_ = C[IDs, :]
            diams_ = diams[IDs]

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
                    VC = V_ov(R, rC, dC)
                    V = VB + VC
                    V_mat[i, j, k-1] = V
                    if V/VS>1.03:
                        print(f"C: {V/VS}, r : {R}, coord : {center_k}")
                        print(f"xlo : {C_lo}")

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
