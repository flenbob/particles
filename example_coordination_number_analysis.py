from pathlib import Path

from src import Plotter, DataReader, FrameKey, CommonKey

#Some code displaying how to analyze coordination number statistics of a given packing 
def coordination_number_matrix():
    """Example of how to generate Z-matrix"""
    
    #Relative or absolute path to the HDF5 datafile
    path = Path('/home/fivan/particles/simulations/data/0/data.hdf5')

    #Create datareader object and read keys of interest.
    # Default keynames are listed in Enums 'CommonKey' and 'FrameKey'.
    # Second argument specifies to read from the last frame (last timestep) of the simulation.
    reader = DataReader(path)
    
    data = reader.get_data(selected_keys = [CommonKey.particle_types, FrameKey.contact_pairs], 
                           selected_frames = -1)
    
    #Use plotter object to plot the Z matrix
    Plotter().plot_Z_matrix(data[CommonKey.particle_types], 
                            data[FrameKey.contact_pairs])

def coordination_number_distributions():
    """Example of how to generate Z-distributions"""

    #Relative or absolute path to the HDF5 datafile
    #'/path/to/hdf5/file/here'
    path = Path('/home/fivan/particles/simulations/data/0/data.hdf5')

    #Create datareader object and read keys of interest.
    # Default keynames are listed in Enums 'CommonKey' and 'FrameKey'.
    # Second argument specifies to read from the last frame (last timestep) of the simulation.
    reader = DataReader(path)
    data = reader.get_data(selected_keys = [CommonKey.particle_types, FrameKey.particle_contacts], 
                           selected_frames = -1)
    
    #Use plotter object to plot the Z distributions
    Plotter().plot_Z_distribution(data[CommonKey.particle_types], 
                                  data[FrameKey.particle_contacts])

if __name__ == "__main__":
    print('Coordination number matrix example...')
    coordination_number_matrix()

    print('Coordination number distributions matrix example...')
    coordination_number_distributions()