# -*- coding: utf-8 -*-
"""Pilot script for making ModelSuite objects using existing SubsidenceModel objects.
"""
import os
from PySub.SubsidenceSuite import ModelSuite
from PySub import plot_utils as plot
from PySub import utils
from PySub.memory import save, load

from matplotlib import cm
import time

line = ((210950, 568010), (220001, 568300.1), (225001, 575300.1))

if __name__ == '__main__':
    Model1 = load(r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output\linear\save\linear.smf")
    Model2 = load(r'\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output\ratetype\save\ratetype.smf')
    Model3 = load(r'\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output\time-decay\save\time-decay.smf')
    
    Suite = ModelSuite('compare compaction models', r'\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Example output')
    Suite.set_models((Model1, Model2, Model3))
    
    #%% Compare
    contour_levels = Suite.get_contour_levels(start = -0.05, end = 0.01, contour_steps = 0.01) 
    plot.plot_min_mean_max(Suite, mode = 'max')
    plot.plot_reservoirs(Suite)
    plot.plot_subsidence(Suite, contour_levels = contour_levels)
    plot.plot_subsidence_observations(Suite, figsize = (8,8))
