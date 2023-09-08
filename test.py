from packing import *
from datawriter import *
from packing import ParticlesGenerator, CoordinatesGenerator, CollectionIntervalGenerator, Packing, Particles
from pathlib import Path

def test_packing():
    table_path = Path('../../input_tables/input_5pd.csv')
    packing = Packing(Particles())

def test_datawriter():    
    folder_path = Path('../../simulations/data/5')
    writer = DataWriter(folder_path)

def test_particle_generator():
    table_path = Path('../../input_tables/input_5pd.csv')
    particles = ParticlesGenerator(table_path).generate_particles()
    print(particles)

def test_collection_interval_generator():
    table_path = Path('../../input_tables/input_5pd.csv')
    particles = ParticlesGenerator(table_path).generate_particles()

    collection_intervals = CollectionIntervalGenerator(particles.diameters).generate_collection_intervals()
    print(collection_intervals)

def test_coordinate_generator():
    table_path = Path('../../input_tables/input_5pd.csv')

    #Generate particles and collection intervals
    particles = ParticlesGenerator(table_path).generate_particles()
    collection_intervals = CollectionIntervalGenerator(particles.diameters).generate_collection_intervals()

    #Generate coordinates
    particles.coordinates = CoordinatesGenerator(particles.diameters, collection_intervals).generate_coordinates()

def test_packing_generator():
    #Write packing
    table_path = Path('../../input_tables/input_5pd.csv')
    write_path = Path('./')
    packing = Packing()
    packing.generate_packing(table_path)
    packing.write_packing(write_path)

def test_packing_loader():
    #create packing object by loading file
    file_path = Path('input.txt')
    packing = Packing()
    packing.load_packing(file_path)
    
def test_other():
    Path('/home/fivan/exjobb/classes/pp/doesntexist').mkdir()

if __name__ == "__main__":
    #test_packing()
    #test_datawriter()
    #test_particle_generator()
    #test_collection_interval_generator()
    #test_coordinate_generator()
    #test_packing_generator()
    test_packing_loader()
    #test_other()