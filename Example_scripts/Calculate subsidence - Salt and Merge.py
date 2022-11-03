# -*- coding: utf-8 -*-
"""Pilot and example script for determining subsidence due to volume change in caverns.
"""
from PySub.MergedModel import merge
from PySub.memory import build_cavern_model, build_model
from PySub import plot_utils as plot


import_path_salt = r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Input example salt.xlsx"

salt_model = build_cavern_model(import_path_salt, name = 'salt',
                                           project_folder = r'\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output')
   
salt_model.calculate_subsidence()

plot.plot_subsidence(salt_model)

salt_model.calculate_subsidence_at_observations()

# No observations in the neighbourhood, extract timeserie for point in model
specified_point = (217000, 573500)
plot.plot_timeseries(salt_model, points = specified_point)
plot.plot_points_on_map(salt_model, points = specified_point, scatter_kwargs = dict(c = 'r', ec = 'k', zorder = 100))

import_path_gas = r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Input example linear.xlsx"
gas_model = build_model(import_path_gas,
                                   name = 'linear', 
                                   project_folder = r'\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output')

gas_model.calculate_compaction()
gas_model.calculate_subsidence()
plot.plot_subsidence(gas_model)


#%%
merged = merge([salt_model, gas_model], variables = [], dx = 100)
merged.calculate_slope()

plot.plot_subsidence(merged)

plot.plot_subsidence(merged, variable = 'slope', contour_steps = 1e-5)

