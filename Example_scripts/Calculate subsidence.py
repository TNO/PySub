"""Pilot script for modeling workflow for subsidence"""
#%% import packages used in this example
from PySub import plot_utils as plot
from PySub.memory import build_model
from PySub.SubsidenceSuite import ModelSuite

if __name__ == '__main__':
    models = []
    
    #%% Make the linear model
    import_path = "Input example linear.json"
    name = 'linear'
    project_folder = None
    Model = build_model(import_path, name = name, 
                        project_folder = project_folder)
    
    Model.calculate_compaction()
    Model.calculate_subsidence()
    
    # plot with a different background map
    plot.plot_subsidence(Model,
                         service = 'https://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi',
                         layer = 'VIIRS_CityLights_2012',
                         epsg = 28992,
                         )
    models.append(Model)
    
    #%% Make the time-decay model
    import_path = "Input example time-decay.xlsx"
    name = 'time-decay'
    Model = build_model(import_path, name = name, 
                        project_folder = project_folder)
    
    Model.calculate_compaction()
    Model.calculate_subsidence()
    
    plot.plot_subsidence(Model)
    models.append(Model)
    
    #%% Make the ratetype model
    import_path = "Input example ratetype.xlsx"
    name = 'ratetype'
    Model = build_model(import_path, name = name, 
                        project_folder = project_folder)
    
    Model.calculate_compaction()
    Model.calculate_subsidence()
    
    plot.plot_subsidence(Model)
    models.append(Model)
    
    Suite = ModelSuite("compare", project_folder = project_folder)
    Suite.set_models(models)
    plot.plot_timeseries(Suite, mode = "max")
