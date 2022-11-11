"""Pilot script for modeling workflow for subsidence"""
#%% import packages used in this example
from PySub import plot_utils as plot
from PySub.memory import build_model, export_tiff
from PySub.SubsidenceSuite import ModelSuite

if __name__ == '__main__':
    models = []
    
    #%% Make the linear model
    import_path = "Input example linear.json"
    name = 'linear'
    project_folder = None
    Model = build_model(import_path, name = name, 
                        project_folder = project_folder)
    
    Model.grid['gridded_thickness'] = Model.thickness*Model.reservoir_mask
    fname = r"C:\Users\davidsb\OneDrive - TNO\Documents\PySub\Example_scripts\Model from grid\Allardsoog_thickness.tif"
    export_tiff(Model, variable = 'gridded_thickness', time = -1, reservoir = 'Allardsoog', fname = fname, epsg = 28992)
    Model.calculate_compaction()
    Model.calculate_subsidence()
    
    plot.plot_subsidence(Model)
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
