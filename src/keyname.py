from enum import Enum, auto

class FrameKey(Enum):
    """Names of frame keys"""
    cell_matrix = auto()
    Z_g = auto()
    Z_nr = auto()
    contact_pairs = auto()
    distance_pairs = ()
    particle_contacts = auto()
    particle_coordinates = auto()
    packing_fraction = auto()
    volume = auto()

    def __str__(self):
        return self.name
    
class CommonKey(Enum):
    """Names of common keys"""
    particle_ids = auto()
    particle_types = auto()
    particle_diameters = auto()
    scalars = auto()
    polydispersity = auto()

    def __str__(self):
        return self.name