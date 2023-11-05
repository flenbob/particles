from pathlib import Path
from src import ParticlesGenerator, Stange, CSSMDataGenerator, \
                COVPredictor, Plotter, FrameKey, DataReader

def main():
    print('Content uniformity using Stange example...')
    content_uniformity_stange()

    print('Content uniformity summary example...')
    content_uniformity_summary()
 
def content_uniformity_stange():
    """Example of how to use Stange"""
    #Stange (COV wrt mass) can be calculated in two ways:
    # - Using the sampled particles, which is dependent of a particles object.
    # - Using the analytical expression, which is just dependent of an input table.

    #Given sampled particles
    path = Path('/home/fivan/particles/input_tables/packing_2.csv')
    pg = ParticlesGenerator(path)
    particles = pg.generate_particles()
    cov_particles = Stange().cov_given_mass_particles(particles = particles) 
    
    #Analytical expression given table parameters and some mass G.
    # For this example, set mass to total mass sampled to compare the 
    # two methods
    cov_params = Stange().cov_given_mass_params(params = pg.params,
                                                G = particles.mass_types.sum())
    
    #For large packings the difference in COV between both methods is minimal. But 
    # using the sampled particles is obviously more "correct".
    print('Stange component-wise COV:s given ' + \
          '(1): sampled particles, (2): table params' + \
          f'\n 1: {cov_particles.T[0]}\n 2: {cov_params}\n')
    
    #For any arbitrary total mass in micrograms the Stange COV is given by
    total_mass = 10000
    cov_total_mass = Stange().cov_given_mass_params(params = pg.params,
                                                    G = total_mass)
    print(f'Stange COV given {total_mass} μg: {cov_total_mass}\n')
    
    #Or it can be used the other way around (which is used to determine number of 
    # particles required to sample). Given a desired maximum COV of any component,
    # how much total mass is required?
    cov = 0.05 #Can also be of list-type if you want component-wise requirements
    mass_cov = Stange().mass_given_cov_params(params = pg.params,
                                              cov = cov)
    print(f'Mass {mass_cov.sum():.3f} μg required to satisfy maximum COV of {cov}')
    
def content_uniformity_summary():
    """Example of how to generate summary CU statistics of entire packing."""
    #Relative or absolute path to the HDF5 datafile
    path = Path('/home/fivan/particles/simulations/data/8/data.hdf5')
    
    #Create datareader object and read keys of interest.
    # Default keynames are listed in Enums 'CommonKey' and 'FrameKey'.
    # Second argument specifies to read from the last frame (last timestep) of the simulation.
    reader = DataReader(path)
    data = reader.get_data(selected_keys = [FrameKey.cell_matrix, FrameKey.particles], 
                           selected_frames =-1)
    particles = data[FrameKey.particles]
    
    #Generate content uniformity data using CSSM
    # - "n_workers" sets the number of parallel processes to employ 
    #   (limited by number of cores of the system)
    # - "n_shift" sets the number of random shifts of the FCC structure to sample from.
    #    A higher value reduces the sensitivity of data to random fluctuations, but at a higher
    #    computational cost.
    #NOTE: Too small packings (particles fewer than ~10K) can sometimes cause unexpected behavior.
    cssm = CSSMDataGenerator(data[FrameKey.cell_matrix], 
                             particles, 
                             n_workers=16,
                             n_shift=10)
    
    #Resulting data is 4D matrices containing relationship between
    # X_data: Total masses
    # Y_data: Mass fractions 
    #, for every combination of the parameters 
    #   - shift vector
    #   - concentric shell radius
    #   - FCC center coordinate
    #   - Particle type (component)
    X_data, Y_data = cssm.generate_by_mass()

    #Using the CSSM data and particles the COV and its 95% confidence intervals 
    # can be predicted over the full mass range given that the COV is power-law
    # distributed with respect to total mass (which it is).
    predictor = COVPredictor(particles)
    predictor.predict(X_data, Y_data)

    #Use fitted predictor and particles to plot summary statistics of the content uniformity.
    #Each component is given its own plot window with 3 subplots:
    # 1: COV of CSSM data and curvefit. Dashed lines represents upper and lower CI:s.
    # 2: STD of COV of CSSM data and curvefit, i.e. a curvefit for the CI:s.
    # 3: COV of CSSM data and curvefit + CI:s over entire range of masses. The range is limited
    #   , to the total mass of the entire packing. Dashed lines reprensets upper and lower 95% CI:s.
    Plotter().plot_content_uniformity(predictor, particles)
    
if __name__ =="__main__":
    main()
    #test()