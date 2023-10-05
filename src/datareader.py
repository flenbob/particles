from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


@dataclass
class DataReader:
    """Handles data reading of generated HDF5 file."""
    data_path: Path

    #Common and framekeys
    frames: list[int] = field(default_factory=list, init=False)
    frame_keys: list[str] = field(default_factory=list, init=False)
    common_keys: list[str] = field(default_factory=list, init=False)

    def get_data(self, selected_keys: str | list[str], selected_frames: Optional[int | list[int]] = None) -> list[float | int | np.ndarray]:
        """Reads data given selected key(s) at selected frame(s) if required.

        Args:
            selected_keys (str | list[str]): String identifier of key(s)
            selected_frames (Optional[int | list[int]], optional): String identifier of frame(s) if any of the sought keys are frame dependent. Defaults to None.

        Returns:
            list[float | int | np.ndarray]: List of data in same order as keys were provided
        """

        #Convert single key and/or frame to list
        if isinstance(selected_keys, str):
            selected_keys = [selected_keys]
        if isinstance(selected_frames, int):
            selected_frames = [selected_frames]

        #Check if frames are valid and sufficient
        if selected_frames is None:
            selected_frame_keys = [key for key in selected_keys if key in self.frame_keys]
            assert len(selected_frame_keys) == 0, f'Sought key(s) "{selected_frame_keys} require that frames are provided.'
        else:
            selected_frame_ids = self._get_frame_ids(selected_frames)

        #Check that keys are valid
        self._check_keys(selected_keys)

        #Read each key and append data
        data = []
        with h5py.File(self.data_path, 'r') as file:
            for key in selected_keys:
                if key in self.frame_keys:
                    data_frames = [np.array(file[f'{frame}/{key}']) for frame in selected_frame_ids]
                    data.append(data_frames) if len(data_frames) > 1 else data.append(data_frames[0])
                else:
                    data.append(np.array(file[f'{key}']))

        return data if len(data) > 1 else data[0]
    
    def print_keys(self) -> None:
        """Prints datakeys in HDF5 file"""
        #Reformat Format frames output if they are too many
        if len(self.frames) > 5:
            frames = [str(i) for i in range(self.frames[0], self.frames[0] + 3)] + ['...', str(len(self.frames))]
            frames_reformatted = '[' + ', '.join(frames) + ']'
            print(f'Frames: {frames_reformatted}\nFrame keys: {self.frame_keys}\nCommon keys: {self.common_keys}')
        else:
            print(f'Frames: {self.frames}\nFrame keys: {self.frame_keys}\nCommon keys: {self.common_keys}')

    def __post_init__(self) -> None:
        """Sets the frames, keys and scalars class variables"""
        self._check_data_path()
        self._read_metadata()

    def _check_data_path(self) -> None:
        """Checks that provided path to HDF5-file is valid."""
        assert Path.exists(self.data_path), f"Path to HDF5 datafile in {self.data_path} is not valid."

    def _read_metadata(self) -> None:
        """Read HDF5 metadata and set class variables."""
        with h5py.File(self.data_path, 'r') as file:
            keys = file.keys()
            self.frames = [str(key) for key in sorted([int(key) for key in keys if key.isdigit()])]
            self.common_keys = list(keys - self.frames)
            self.frame_keys = list(file[self.frames[0]].keys())
            self.frames = [int(frame) for frame in self.frames]

    def _check_keys(self, keys: str | list[str]) -> None:
        """Check if the provided keys are valid."""
        valid_keys = self.common_keys + self.frame_keys
        for key in keys:
            assert key in valid_keys, f'Sought key "{key}" is invalid. See class function print_keys() for readable keys.'

    def _get_frame_ids(self, frames: list[int]) -> list[int]:
        """Get frame IDs from frame numbers."""
        frame_ids = []
        for frame in frames:
            try:
                frame_ids.append(self.frames[frame])
            except IndexError:
                raise ValueError(f'Sought frame {frame} is invalid. See class function "print_keys()" for available frame keys.')
        return frame_ids
