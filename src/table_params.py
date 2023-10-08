from enum import Enum, auto

class Param(Enum):
    """CSV-table parameter names"""
    types_id = 'types_id'
    density = 'density'
    mass_fraction = 'mass_fraction'
    mean = 'mean'
    std = 'std'
    mu = 'mu'
    sigma = 'sigma'
    cv = 'cv'

    def __str__(self):
        return self.value