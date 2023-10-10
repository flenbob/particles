from enum import Enum

class FileName(str, Enum):
    """Sets the name of various files"""
    INPUT_FILE = 'input.txt'
    DATA_FILE = 'data.hdf5'

    # LAMMPS dumpfiles and log
    GLOBAL_FILE = 'out_global.txt'
    LOCAL_FILE = 'out_local.txt'
    SCALAR_FILE = 'out_scalar.txt'
    LOG_FILE = 'log.lammps'

    def __str__(self):
        return str.__str__(self)