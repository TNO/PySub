# -*- coding: utf-8 -*-
"""Pilot and example script for determining subsidence due to volume change in caverns.
"""
from PySub.memory import build_cavern_model
from PySub.SubsidenceSuite import ModelSuite
from PySub import plot_utils as plot

LINE = ((214000,573250),(220000,573800))
if __name__ == '__main__':
    import_path_salt = r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Input example salt.xlsx"
    import_path_MRB = r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Input example salt MRB.xlsx"
    
    salt_model = build_cavern_model(import_path_salt, name = 'withouth MRB',
                                               project_folder = r'\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output')
    MRB_model = build_cavern_model(import_path_MRB, name = 'with MRB',
                                               project_folder = r'\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output')   
    salt_model.calculate_subsidence()
    plot.plot_subsidence(salt_model)
       
    MRB_model.calculate_subsidence()
    plot.plot_subsidence(MRB_model)
    
    Suite = ModelSuite('compare salt models', r'\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output')
    Suite.set_models((salt_model, MRB_model))
    
    plot.plot_cross_section(Suite, lines = LINE, time = -1, y_axis_exageration_factor= 1 , figsize = (12,12))
    plot.plot_timeseries(Suite, mode = 'max')
