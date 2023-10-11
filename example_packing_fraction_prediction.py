from pathlib import Path

from src import Plotter, DataReader, FrameKey, CommonKey, Packing, ParticlesGenerator, LinearMixturePackingModel

#An example showing how the jammed packing fraction 
# (for frictionless spherical polydisperse particles)
# can be estimated using LMPM. 

# An alternative method, which we denoted "Master Curve" based on 
# https://pubs.rsc.org/en/content/articlelanding/2020/sm/d0sm00110d
# was not included in this repo.

def packing_fraction_given_table():
    #Path to table
    path = Path('/home/fivan/particles/input_tables/packing_2.csv')
    
    #Generate particles given table params
    generator = ParticlesGenerator(table_path=path)
    particles = generator.generate_particles()

    #Initialize LMPM object and predict packing fraction on particles
    lmpm = LinearMixturePackingModel(particles.diameters, 
                                     particles.type_ids)
    pf = lmpm.predict_packing_fraction()
    print(f'Predicted packing fraction given table: {100*pf:.5f}%')


def packing_fraction_given_file():
    #Path to packing file, same file as the one used as input to LAMMPS script
    path = Path('/home/fivan/particles/simulations/data/0/input.txt')

    #Load particles from LAMMPS input file to the packing object
    packing = Packing()
    packing.load_packing(path)

    #Initialize LMPM object and predict packing fraction on particles
    lmpm = LinearMixturePackingModel(packing.particles.diameters, 
                                     packing.particles.type_ids)
    pf = lmpm.predict_packing_fraction()
    print(f'Predicted packing fraction given file: {100*pf:.5f}%')
    
if __name__ == "__main__":
    packing_fraction_given_table()
    packing_fraction_given_file()