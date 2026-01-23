"""Example script for making probabilistic analyses using bucket ensembles."""

from PySub import plot_utils as plot
from PySub import utils
from PySub.memory import build_bucket_ensemble, load_observation_points

import os
import numpy as np

line = ((210950, 568010), (220001, 568300.1), (225001, 575300.1))

if __name__ == "__main__":
    if not os.getcwd().endswith("Example_scripts"):
        os.chdir("Example_scripts")
    number_of_samples = 3

    import_paths = [
        r".\BucketEnsemble\Allardsoog.json",  # rateype compaction model
        r".\BucketEnsemble\Een.json",  # Linear compaction model
    ]

    # Project folder is required for running bucket ensemble
    project_folder = os.getcwd()

    model = build_bucket_ensemble(
        import_paths, name="BucketEnsemble_test", project_folder=project_folder
    )

    observation_points = load_observation_points(r".\BucketEnsemble\Allardsoog.json")

    model.set_observation_points(observation_points)

    results, error = model.calculate_samples(
        number_of_samples=number_of_samples, all_timesteps=True, seed=None
    )
    # results, error = model.calculate_deterministic(all_timesteps=True)
    (
        _,
        (p90, p50, p10),
        (model_index_p90, model_index_p50, model_index_p10),
    ) = utils.probability_distribution(results)
    plot.plot_probability_distribution(model, results, c="k")
    model.calculate_from_sampled(model_index_p10)
    plot.plot_subsidence(model, title="Susbsidence (cm) in final timestep - p10")
    model.calculate_from_sampled(np.where(error == np.min(error))[0][0])
    plot.plot_subsidence(
        model, title="Susbsidence (cm) in final timestep - lowest error"
    )
    plot.plot_subsidence_observations(model)
