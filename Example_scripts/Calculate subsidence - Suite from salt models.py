# -*- coding: utf-8 -*-
"""Pilot script calculating subsidence due to volume change in salt caverns in
ModelSuite.
"""
from PySub.memory import build_cavern_model
from PySub import plot_utils as plot
from PySub.SubsidenceSuite import ModelSuite

import_path_salt = r"Input example salt.xlsx"
import_paths = [import_path_salt, import_path_salt]


salt_models = [build_cavern_model(import_path_salt, name = 'salt1'),
               build_cavern_model(import_path_salt, name = 'salt2')]

Suite = ModelSuite('compare compaction models', project_folder=None)
Suite.set_models(salt_models)
Suite.calculate_subsidence()
plot.plot_subsidence(Suite)
