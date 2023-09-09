import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from ovito.io import import_file

from .filename import FileName
from .keyname import CommonKey, FrameKey


@dataclass
class DataWriter:
    """Handles reading of LAMMPS dumpfiles and writing them to a structured HDF5 file"""
    #Folder path to write/read
    folder_path: Path
    data_path: Path = None

    def __post_init__(self):
        self._check_folder_path()
        self._set_data_path()

    def write_hdf5(self) -> None:
        """Writes LAMMPS data to hdf5 file"""
        #Path to dumpfiles
        global_path = self.folder_path/str(FileName.GLOBAL_FILE)
        local_path = self.folder_path/str(FileName.LOCAL_FILE)
        scalar_path = self.folder_path/str(FileName.SCALAR_FILE)

        #Write to data-file if dumpfile is found
        self._write_global(global_path) if global_path.exists() else print(f'Could not find global file in: {global_path}')
        self._write_local(local_path) if local_path.exists() else print(f'Could not find local file in: {local_path}')
        self._write_scalar(scalar_path) if scalar_path.exists() else print(f'Could not find scalar file in: {scalar_path}')

    def _check_folder_path(self) -> None:
        """Check that selected folder is valid"""
        assert isinstance(self.folder_path, Path), f'Selected folder {self.folder_path} is not of type <Path>'
        assert self.folder_path.exists(), f'Selected folder {self.folder_path} is not valid.'

    def _set_data_path(self) -> None:
        """Sets the data path"""
        self.data_path = self.folder_path/FileName.DATA_FILE.value
        try:
            assert not self.data_path.exists()
        except AssertionError:
            while True:
                decision = input(f'A HDF5 file already exists in: {self.data_path}. Overwrite it (y) or (n)?\n').lower()
                match decision:
                    case 'n':
                        sys.exit('Program exited.')
                    case 'y':
                        os.remove(self.data_path)
                        break
                    case _:
                        print('Invalid key. Use (y) or (n)')
                    
    def _write_global(self, global_path: Path) -> None:
        try:
            #Global file pipeline
            pipeline_global = import_file(global_path, multiple_frames=True, columns=
                            ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z', 'Diameter', 'Contacts'])

            #Write to HDF5 for each frame (timestep)
            with h5py.File(self.data_path, 'a') as file:
                #Read frame independent attributes from 0:th frame
                data_global = pipeline_global.compute(0)
                diameters = data_global.particles['Diameter'][:]
                particle_volume = np.pi/6*np.sum(diameters**3)
                polydispersity = np.max(diameters)/np.min(diameters)
                file.create_dataset(CommonKey.polydispersity, data=polydispersity)
                attrs = np.array([data_global.particles['Particle Identifier'][:], 
                                data_global.particles['Particle Type'][:],
                                diameters]).T
                
                #Sort by Particle Identifier (ID) and write
                attrs = attrs[attrs[:, 0].argsort(kind='stable')]
                file.create_dataset(CommonKey.particle_ids, data=attrs[:, 0])
                file.create_dataset(CommonKey.particle_types, data=attrs[:, 1])
                file.create_dataset(CommonKey.particle_diameters, data=attrs[:, 2])

                #Frame dependent attributes
                for frame in range(pipeline_global.source.num_frames):
                    data_global = pipeline_global.compute(frame)

                    #Global and non-rattler coordination number
                    Z = data_global.particles['Contacts'][:]
                    no_contacts = np.all(Z == 0)
                    no_nonrattlers = np.all(Z < 4)
                    Z_g = 0 if no_contacts else np.mean(Z)
                    Z_nr = 0 if no_nonrattlers else np.mean(Z[Z >= 4])
                    file.create_dataset(f'{frame}/{FrameKey.Z_g}', data=float(Z_g))
                    file.create_dataset(f'{frame}/{FrameKey.Z_nr}', data=float(Z_nr))
                                
                    attrs = np.array([data_global.particles['Particle Identifier'][:],
                                    data_global.particles['Position'][:, 0],
                                    data_global.particles['Position'][:, 1],
                                    data_global.particles['Position'][:, 2],
                                    Z
                                    ]).T

                    #Sort by ID and write
                    attrs = attrs[attrs[:, 0].argsort(kind='stable')]
                    file.create_dataset(f'{frame}/{FrameKey.particle_coordinates}', data=attrs[:, 1:4])
                    file.create_dataset(f'{frame}/{FrameKey.particle_contacts}', data=attrs[:, 4])

                    #Cell attributes
                    cell_matrix = data_global.cell[:3, :3]
                    cell_origin = data_global.cell[:, -1]
                    cell_volume = data_global.cell.volume
                    cell_density = particle_volume/cell_volume
                    file.create_dataset(f'{frame}/{FrameKey.cell_matrix}', data=cell_matrix)
                    file.create_dataset(f'{frame}/{FrameKey.cell_origin}', data=cell_origin)
                    file.create_dataset(f'{frame}/{FrameKey.volume}', data=cell_volume)
                    file.create_dataset(f'{frame}/{FrameKey.packing_fraction}', data=cell_density)

            print('Global file written to HDF5 file.')
        except Exception:
            traceback.print_exc(file=sys.stdout)	
            print('Could not write global file.')

    def _write_local(self, local_path: Path) -> None:
        try:
            #Local file pipeline
            pipeline_local = import_file(local_path, multiple_frames=True, columns=['Particle Identifiers.1', 'Particle Identifiers.2', 'Distance'])

            #Write to HDF5 for each frame (timestep)
            with h5py.File(self.data_path, 'a') as file:
                #Iterate through each frame
                for frame in range(pipeline_local.source.num_frames):
                    data_local = pipeline_local.compute(frame)
                    file.create_dataset(f'{frame}/{FrameKey.distance_pairs}', data=data_local.particles.bonds['Distance'][:])
                    file.create_dataset(f'{frame}/{FrameKey.contact_pairs}', data=data_local.particles.bonds['Particle Identifiers'][:])

            print('Local file written to HDF5 file.')
        except Exception:
            traceback.print_exc(file=sys.stdout)
            print('Could not write local file.')
            
    def _write_scalar(self, scalar_path: Path) -> None:
        try:
            #Read as comma-separated file
            header = open(scalar_path, 'r').readline().strip().split(', ')
            scalars = np.loadtxt(scalar_path, delimiter=",", skiprows=1)
            
            #Write scalars to HDF5
            with h5py.File(self.data_path, 'a') as file:
                for label, scalar in zip(header, scalars.T):
                    file.create_dataset(f'{CommonKey.scalars}/{label}', data=scalar)

            print('Scalar file written to HDF5 file.')
        except Exception:
            traceback.print_exc(file=sys.stdout)
            print('Could not write scalar file.')