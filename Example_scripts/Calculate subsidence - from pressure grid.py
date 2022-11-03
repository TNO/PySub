"""Pilot script for modeling workflow for subsidence, uses an arbitrary grid for the pressure
"""
#%%
from PySub.SubsidenceModelGas import *
from PySub import plot_utils
from PySub.memory import *
import numpy as np
import matplotlib.pyplot as plt

def plot_pressures(pressures):
    plot_pressures = pressures.copy()
    plot_pressures[pressures == 0] = None
    plot_pressures = np.nansum(plot_pressures, axis = 2)
    mapped = plt.contourf(plot_pressures[:,:,-1])
    plt.title('Pressure (bar) in reservoirs')
    plt.axis('off')
    plt.colorbar(mapped)
    plt.show()

if __name__ == '__main__':
    pressure_files = [r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Example with grids\Model from grid Norg pressures.csv",
                      r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Example with grids\Model from grid Allardsoog pressures.csv",
                      r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Example with grids\Model from grid Een pressures.csv"]
    
    import_path = r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Input example - from pressure grid.xlsx"
    project_folder = r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe output"
    
    Model = build_model(import_path, name = 'Model from grid')
    
    Model.set_pressures(pressure_files) # Replaced pressures from the excel import with csv raster files
    plot_utils.plot_subsidence(Model, variable = 'pressures', unit = 'm', contour_steps = 10, plot_reservoir_shapes = True)
    
    Model.calculate_compaction()
    Model.calculate_subsidence()
    Model.calculate_subsidence_at_points()
    Model.calculate_subsidence_at_observations()
    
    
    plot_utils.plot_reservoirs(Model)
    plot_utils.plot_subsidence(Model, time = 10)
    plot_utils.plot_subsidence(Model, time = '2010')
    
    line = ((210950, 568010), (220001, 568300.1), (225001, 575300.1))
    plot_utils.plot_cross_section(Model, line,
                                  reservoir = None, 
                                  time = ['1995', '2000', '2005', '2010'], 
                                  service = 'opentopo')
    plot_utils.plot_points_on_map(Model, show_data = True, plot_reservoir_shapes = True, 
                                  annotate_reservoirs = False)
    plot_utils.plot_subsidence_points(Model, points = None, reservoir = None, 
                                      plot_kwargs = {'c': ['r', 'b']})
#%% Code saved for grid conversion
# from PySub.shape_utils import save_raster
# Model1.convert_to_grid('pressures')
# export_grids = Model1.grid['pressures']
# folder_path = r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input"
# save_path = os.path.join(folder_path, 'pressures.tif')
# save_raster(export_grids.values, Model1.x, Model1.y, Model1.dx, Model1.dy, 28992, fname = save_path)
# for r in range(Model1.number_of_reservoirs):
#     save_path = os.path.join(folder_path, Model1.reservoirs[r] + '_pressures.tif')
#     save_raster(export_grids.isel(reservoir = r).values, Model1.x, Model1.y, Model1.dx, Model1.dy, 28992, fname = save_path)
    
   