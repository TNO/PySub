# -*- coding: utf-8 -*-
"""Pilot script calculating subsidence due to volume change in salt caverns in
ModelSuite.
"""
from PySub.SubsidenceModelCavern import SubsidenceModel
from PySub.MergedModel import MergedModel, merge
from PySub.memory import build_cavern_model_from_excel, build_model_from_excel
from PySub import plot_utils as plot
from PySub import utils
from PySub import memory
from PySub import grid_utils
import pandas as pd 
import numpy as np
from PySub.SubsidenceSuite import ModelSuite

import_path_salt = r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Input example salt.xlsx"
import_paths = [import_path_salt, import_path_salt]


salt_models = [build_cavern_model_from_excel(import_path_salt, name = 'salt1'),
               build_cavern_model_from_excel(import_path_salt, name = 'salt2')]

Suite = ModelSuite('compare compaction models', r'\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output')
Suite.set_models(salt_models)
Suite.calculate_subsidence()
plot.plot_subsidence(Suite)
