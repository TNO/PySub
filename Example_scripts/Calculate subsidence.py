"""Pilot script for modeling workflow for subsidence"""
#%% import packages used in this example
from PySub import plot_utils as plot
from PySub.memory import build_model

line = {'A': (211484.4334172095, 565558.8829436488), 
        'B': (228764.33469374882, 572297.3772638438)}

if __name__ == '__main__':
    #%% Make a new model
    import_path = r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Input example linear.xlsx"
    name = 'linear'
    project_folder = r'\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output'
    Model = build_model(import_path, name = name, 
                        project_folder = project_folder)
    
    Model.calculate_compaction()
    Model.calculate_subsidence()
    
    plot.plot_reservoirs(Model)
    plot.plot_subsidence(Model)
    plot.plot_cross_section(Model, line)
    # Model.calculate_slope()  
    # plot.plot_subsidence(Model, 
    #                      variable = 'slope', 
    #                      contour_steps = 1e-6,
    #                     )
    
    # Model.calculate_concavity() 
    # plot.plot_subsidence(Model, 
    #                      variable = 'concavity', 
    #                      contour_steps = 1e-9)

    # Model.calculate_subsidence_rate()  
    # plot.plot_subsidence(Model, variable = 'subsidence_rate', time = 1995,
    #                      contour_steps = 0.0005)
    #%%
    # Export a raster and csv or .asc with the subsidence data of the final (-1) timestep
    # export_tiff(Model, time = -1)
    # export_csv(Model) # or export_ascii(Model)
    
    # Export all available information to a map (rasters, csv's, shapefiles and 
    # a project file to rule them all). This can take a while.
    # export_all(Model) 
    
    # A short, quick report with basic overview can also be exported:
    # Model.report(figures = False)
    # save(Model)

    
    
    
