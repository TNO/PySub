# -*- coding: utf-8 -*-
"""Pilotscript for calculating and comparing multiple subsidence model scenarios
using the ModelSuite feature from PySub.
"""
from PySub import plot_utils as plot
from PySub.memory import build_suite

line = ((210950, 568010), (220001, 568300.1), (225001, 575300.1))
#%% Determin min. mid and max subsidence based on variablilty in parameters
if __name__ == '__main__':
     
    Suite = build_suite(
        import_path = [r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Suite\Min.xlsx",
                        r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Suite\Mid.xlsx",
                        r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Suite\Max.xlsx"],
        name = 'Suite',
        project_folder = r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output",
        bounds = None, 
        all_same_bounds = True) # They do not have to be the same grids, but you can force it with setting this parameter to True
    
    Suite.assign_point_parameters()
    Suite.assign_observation_parameters()
    
    Suite.calculate_compaction()
    Suite.calculate_subsidence()
    plot.plot_overlap_cross_section(Suite, line)
    contour_levels = Suite.get_contour_levels() 
    plot.plot_cross_section(Suite, line, time = '2010', 
                                  reservoir = None, 
                                  y_axis_exageration_factor = 2,
                                  horizontal_line = -0.06, raster_kwargs = {'cmap': 'winter'})
     
    
    Suite.calculate_subsidence_at_points()
    Suite.calculate_subsidence_at_observations()
    
    print('Mean absolute error (in m) per model: \n', [round(e, 4) for e in Suite.error(method = 'mae').values()])
    
    plot.plot_min_mean_max(Suite, point = (220001., 568300.1))
    plot.plot_timeseries(Suite, mode = 'max')
    plot.plot_subsidence_observations(Suite)
    plot.plot_reservoirs(Suite, figsize = (8,8), raster_kwargs = {'cmap': 'winter_r', 'alpha': 0.25})
    plot.plot_subsidence(Suite, time = -1, reservoir = None, figsize = (8,8), contour_levels = contour_levels)
    plot.plot_subsidence_points(Suite)
