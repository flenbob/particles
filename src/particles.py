from dataclasses import dataclass

import numpy as np
@dataclass
class Particles:
    """Packing particles data"""
    ids: np.ndarray
    type_ids: np.ndarray
    diameters: np.ndarray
    density_types: np.ndarray
    rescale_factor: float
    coordinates: np.ndarray = None
    
    @property 
    def types(self) -> np.ndarray:
        return np.unique(self.type_ids).astype(int)-1
    
    @property
    def diameter_types(self) -> list[np.ndarray]:
        rf = self.rescale_factor if (self.diameters.min() < 1.1) else 1
        return [rf*self.diameters[self.type_ids == t+1] for t in self.types] 
    
    @property
    def volume_types(self) -> np.ndarray:
        return np.pi/6*np.array([(d_type**3).sum() for d_type in self.diameter_types])
        
    @property
    def mass_types(self) -> np.ndarray:
        return np.pi/6*self.density_types*np.array([(d_type**3).sum() for d_type in self.diameter_types])
        
    def sort_by_diameters(self, order: str) -> None:
        """Sort particles by diameters in ascending/descending order"""
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