from dataclasses import dataclass

import numpy as np
@dataclass
class Particles:
    """Packing particles data"""
    ids: np.ndarray
    type_ids: np.ndarray
    diameters: np.ndarray
    coordinates: np.ndarray = None
    density_types: np.ndarray = None 
    rescale_factor: float = 1

    def sort_by_diameters(self, order: str) -> None:
        match order:
            case 'ascending':
                sorted_indicies = np.argsort(self.diameters)
            case 'descending':
                sorted_indicies = np.argsort(self.diameters)[::-1]
            case _:
                raise ValueError(f'Invalid sort order: {order}. Available are: "ascending", "descending".')
            
        #Sort
        self.ids = self.ids[sorted_indicies]
        self.type_ids = self.type_ids[sorted_indicies]
        self.diameters = self.diameters[sorted_indicies]
        if self.coordinates is not None:
            self.coordinates = self.coordinates[sorted_indicies]