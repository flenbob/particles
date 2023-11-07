from pathlib import Path 

from src import CoordinatesGenerator, ParticlesGenerator, CollectionIntervalGenerator, Packing

import time
import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    #Generate particles
    path = Path('/home/fivan/particles/input_tables/packing_2.csv')
    particles = ParticlesGenerator(table_path=path).generate_particles()

    #Generate collection intervals
    cinters = CollectionIntervalGenerator(particles.diameters).\
        generate_collection_intervals()


    #Benchmark over different cell parameters r
    r = np.array([1])
    repeats = 1
    benchlist = np.zeros((repeats, r.shape[0]))

    for i in range(repeats):
        print(f'---------{i}--------')
        for j, n_subcells in enumerate(r):
            time_start = time.perf_counter()
            #Generate coordinates
            coords = CoordinatesGenerator(diameters = particles.diameters,
                                        collection_intervals = cinters,
                                        N_subcells = n_subcells).generate_coordinates()
            benchlist[i, j] = time.perf_counter() - time_start

    #Plot bench-scores
    print(benchlist)
    
    # particles.coordinates = coords
    # packing = Packing(particles=particles)
    # packing.write_packing(Path('/home/fivan/particles'))

if __name__ == "__main__":
    main()