import pathlib
import os
os.environ['OVITO_GUI_MODE'] = '1'
import ovito

from src.content_uniformity import *
from src.datareader import *
from src.plotter import Plotter
from src import CommonKey, FrameKey

def test_on_file():
    path = Path('/home/fivan/particles/simulations/data/0/data.hdf5')
    reader = DataReader(path)

    #Read data
    reader.print_keys()
    keys = [FrameKey.cell_matrix, FrameKey.particles]
    data = reader.get_data(keys, -1)
    cell_matrix, particles = data.values()

    #CSSM data
    cssm = CSSMDataGenerator(cell_matrix, 
                             particles, 
                             n_workers=16,
                             n_shift=10)
    X_data, Y_data = cssm.generate_by_mass()

    #COV predictor
    predictor = COVPredictor(particles)
    predictor.predict(X_data, Y_data)

    #Make plots
    Plotter().plot_content_uniformity(predictor, particles)
    
if __name__ =="__main__":
    test_on_file()
    #main()
    #read()