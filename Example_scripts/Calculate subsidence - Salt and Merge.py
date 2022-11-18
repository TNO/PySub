# -*- coding: utf-8 -*-
"""Pilot and example script for determining subsidence due to volume change in caverns.
"""
from PySub.MergedModel import merge
from PySub.memory import build_cavern_model, build_model
from PySub import plot_utils as plot


import_path_salt = r"Input example salt.xlsx"

salt_model = build_cavern_model(import_path_salt,
                                name = 'salt',
                                project_folder = None)
   
salt_model.calculate_subsidence()

plot.plot_subsidence(salt_model)

salt_model.calculate_subsidence_at_observations()

# No observations in the neighbourhood, extract timeserie for point in model
specified_point = (217000, 573500)
plot.plot_timeseries(salt_model, points = specified_point)
plot.plot_points_on_map(salt_model, points = specified_point, scatter_kwargs = dict(c = 'r', ec = 'k'))

import_path_gas = r"Input example linear.json"
gas_model = build_model(import_path_gas,
                        name = 'linear', 
                        project_folder = None)

gas_model.calculate_compaction()
gas_model.calculate_subsidence()
plot.plot_subsidence(gas_model)


#%%
merged = merge([salt_model, gas_model], variables = [], dx = 100)

plot.plot_subsidence(merged)

