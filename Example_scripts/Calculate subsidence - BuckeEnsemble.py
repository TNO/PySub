# -*- coding: utf-8 -*-
"""Pilot script for making probabilistic analyses using bucket ensembles.
"""
from PySub import plot_utils as plot
from PySub import utils
from PySub.BucketEnsemble import BucketEnsemble
from PySub.memory import build_bucket_ensemble
from PySub.Points import load_observation_points_from_excel

import numpy as np

line = ((210950, 568010), (220001, 568300.1), (225001, 575300.1))

if __name__ == '__main__':
    import_paths = [r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Bucket\Allardsoog.xlsx",
                    r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Bucket\Een.xlsx"]
    
    model = build_bucket_ensemble(import_paths, name = 'test Vermillion', 
                                  project_folder = r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe output")
    
    observation_points = load_observation_points_from_excel(r"\\tsn.tno.nl\Data\sv\sv-069554\Kluis\Lop_Proj_2021_EZK\B12_KEM16\WP4\Exampe input\Bucket\Allardsoog.xlsx")
    
    model.set_observation_points(observation_points)
    
    results, error = model.calculate_samples(number_of_samples = 10, all_timesteps = True) 
    _, (p90, p50, p10), (model_index_p90, model_index_p50, model_index_p10) = utils.probability_distribution(results)
    plot.plot_probability_distribution(model, results, c = 'k')
    model.calculate_from_sampled(model_index_p10)
    plot.plot_subsidence(model, title = 'Susbsidence (cm) in final timestep - p10')
    model.calculate_from_sampled(np.where(error == np.min(error))[0][0])
    plot.plot_subsidence(model, title = 'Susbsidence (cm) in final timestep - lowest error')
    plot.plot_subsidence_observations(model)
