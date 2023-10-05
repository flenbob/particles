import pathlib
import os
os.environ['OVITO_GUI_MODE'] = '1'
import ovito

from src.content_uniformity import *
from src.datareader import *
from src.plotter import Plotter

def read():
    path = Path('/home/fivan/particles/simulations/data/hdf5/300k_pack.hdf5')
    reader = DataReader(path)
    reader.print_keys()

def test_on_file():
    path = Path('/home/fivan/particles/simulations/data/hdf5/300k_pack.hdf5')
    reader = DataReader(path)

    #Read data
    reader.print_keys()
    H, diams, coords, type_ids = reader.get_data(['H_matrix', 
                                                  'diams', 
                                                  'coords', 
                                                  'ID_types'], -1)
    #This data will otherwise be written to hdf5 file aswell
    #densities = 1e-12*np.array([1000, 2000, 3000, 1000, 2000])     # Transform to yg/ym^3
    densities = np.array([1, 1, 1, 1, 1])

    rescale_factor = 1.5        # ym            

    #CSSM data
    cssm = CSSMDataGenerator(H, diams, coords, type_ids, n_workers=16, type_densities=densities)

    #Make plots
    ##Plotter().plot_content_uniformity(predictor, stange, "blabla")

    X_data, Y_data = cssm.generate_by_mass()

    #COV predictor
    predictor = COVPredictor(diams, type_ids, 'mass', type_densities=densities)
    predictor.predict(X_data, Y_data)

    #Stange
    stange = Stange(diams, type_ids, 'mass', densities)

    #Make plots
    Plotter().plot_content_uniformity(predictor, stange, "yg")
    


if __name__ =="__main__":
    test_on_file()
    #main()
    #read()