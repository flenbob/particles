import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import h5py

from .datawriter import DataWriter
from .filename import FileName
from .packing import Packing


class LAMMPSScript(Enum):
    """Names of selectable LAMMPS scripts"""
    compression: str = 'in.compression.lmp'
    two_phase: str = 'in.two_phase.lmp'

@dataclass
class Runner:
    """Handles running of a simulation"""
    script_path: Path = None
    folder_path: Path = None

    def run_new(self, script: str, table: str, n_tasks: int, folder_path: Path = None, screen: bool = False) -> None:
        """Runs a complete simulation on existing LAMMPS formatted packing file

        Args:
            script (str): String identifier for simulation type
            table (str): String identifier to input CSV-table
            n_tasks (int): Number of MPI-tasks to use in simulation
            folder_path (Path, optional): Path identifier to write LAMMPS files to. If not provided, it is given its own folder in ./simulations/data/ID. Defaults to None.
            screen (bool, optional): Suppresses LAMMPS output to screen and only write to log-file. Defaults to False.
        """
        #Input table path
        table_path = Path(Path.cwd())/'input_tables'/table
        assert table_path.exists(), f"Path to input table {table_path} is not valid."

        #Set destination folder path
        self._set_destination_folder(folder_path)

        #Set simulation script type
        self._set_script(script) 

        #Create packing
        packing = Packing()
        packing.generate_packing(table_path)
        packing.write_packing(self.folder_path)

        #Convert collection intervals to LAMMPS readable format
        collection_intervals = [len(packing.collection_intervals)] + packing.collection_intervals[::-1]
        collection_intervals = " ".join(str(item) for item in collection_intervals)

        #Initialize LAMMPS simulation
        self._initialize_lammps_script(collection_intervals, n_tasks, screen)

        #Convert LAMMPS dumpfiles to HDF5 file
        DataWriter(self.folder_path).write_hdf5()

        #Also write rescale factor and densities of each type to hdf5 file
        with h5py.File(self.folder_path, 'a') as file:
            file.create_dataset(f'rescale_factor', data=float(packing.rescale_factor))
            file.create_dataset(f'types_density', data=float(packing.types_density))
            
    def run_existing(self, script: str, packing_path: Path, n_tasks: int, screen: bool = False) -> None:
        """Runs a complete simulation on existing LAMMPS formatted packing file

        Args:
            script (str): String identifier for simulation type
            packing_path (Path): Path identifier to LAMMPS packing file
            n_tasks (int): Number of MPI-tasks to use in simulation
            screen (bool, optional): Shows LAMMPS output to screen and only write to log-file. Defaults to False.
        """

        assert packing_path.exists(), f'Cannot find packing in {packing_path}'

        #Set destination folder to same path as packing
        self.folder_path = packing_path

        #Set simulation script type
        self._set_script(script)

        #Load packing to generate collection intervals
        packing = Packing()
        packing.load_packing(packing_path/FileName.INPUT_FILE.value)
        collection_intervals = [len(packing.collection_intervals)] + packing.collection_intervals
        collection_intervals[-1] += 1e-3 #Ensure last interval contains largest particle
        collection_intervals = " ".join(str(item) for item in collection_intervals)

        #Initialize LAMMPS simulation
        self._initialize_lammps_script(collection_intervals, n_tasks, screen)

        #Convert LAMMPS dumpfiles to HDF5 file
        DataWriter(self.folder_path).write_hdf5()    

    def _set_script(self, script: str) -> None:
        """Selects type of LAMMPS script to run

        Args:
            script (str): Selected script type

        Returns:
            Path: Path to selected script type
        """
        # Path to simulation scripts
        match script:
            case LAMMPSScript.compression.name:
                script = LAMMPSScript.compression.value
            case LAMMPSScript.two_phase.name:
                script = LAMMPSScript.two_phase.value
            case _:
                raise Exception(f"Option '{script}' is not valid. Avaliable are: {[script.name for script in LAMMPSScript]}.")
            
        self.script_path = Path(Path.cwd())/f'simulations/scripts/{script}'
    
    def _set_destination_folder(self, folder_path) -> None:
        """Sets destination folder for simulation files"""
        #New folder generated at standard path
        if folder_path is None:
            ID = 0
            while (Path.cwd()/f'simulations/data/{ID}').exists():
                ID += 1
            folder_path = Path.cwd()/f'simulations/data/{ID}'
            try:
                folder_path.mkdir()
                self.folder_path = folder_path
            except FileExistsError:
                self.folder_path = folder_path
            except Exception as e:
                raise Exception(f'Could create destination folder in {folder_path}\n{e}')
            
        #Provided folder generated if it does not exist
        else:
            folder_path = Path(folder_path)
            assert folder_path.exists(), f'Cannot find destination folder in {folder_path}'
            self.folder_path = folder_path
     
    def _initialize_lammps_script(self, collection_intervals: list[int | float], n_tasks: int, screen: bool = False) -> None:
        print(f'Running LAMMPS script.{f" Screen output is suppressed, view progress in logfile: {self.folder_path}/log.lammps" if not screen else ".."}')
        os.system(f'mpirun -np {n_tasks} lmp '\
              f'-v input_data "{self.folder_path/FileName.INPUT_FILE.value}" '\
              f'-v dump_global "{self.folder_path/FileName.GLOBAL_FILE.value}" '\
              f'-v dump_local "{self.folder_path/FileName.LOCAL_FILE.value}" '\
              f'-v dump_scalar "{self.folder_path/FileName.SCALAR_FILE.value}" '\
              f'-v CI "{collection_intervals}" '\
              f'-v self "{self.script_path}" '\
              f'-log "{self.folder_path}/log.lammps" '\
              f'{"-screen none" if not screen else ""} '\
              f'-in {self.script_path}')