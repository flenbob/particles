from enum import Enum

class FrameKey(str, Enum):
    """Names of frame keys"""
    cell_matrix = 'cell_matrix'
    Z_g = 'Z_g'
    Z_nr = 'Z_nr'
    contact_pairs = 'contact_pairs'
    distance_pairs = 'distance_pairs'
    particles = 'particles'
    particle_contacts = 'particle_contacts'
    particle_coordinates = 'particle_coordinates'
    packing_fraction = 'packing_fraction'
    volume = 'volume'

    def __str__(self):
        return str.__str__(self)

class CommonKey(str, Enum):
    """Names of common keys"""
    particle_ids = 'particle_ids'
    particle_types = 'particle_types'
    particle_diameters = 'particle_diameters'
    scalars = 'scalars'
    polydispersity = 'polydispersity' 
    density_types = 'density_types'
    rescale_factor = 'rescale_factor' 

    def __str__(self):
        return str.__str__(self)