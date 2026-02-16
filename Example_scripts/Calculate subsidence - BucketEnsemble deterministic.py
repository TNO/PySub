"""
This workflow calculates subsidence by evaluating all possible combinations
of input variables stored in a bucket. Each bucket contains parameters
with associated values and probabilities.

Bucket Structure:
    {"parameter": {"value": [1000, 1250, 1900],"probabilities": [0.1, 0.5, 0.4]}}

The script systematically realizes every combination of variable values across all buckets.

Note: The number of combinations can grow rapidly as more parameters or options are added.

Each combination is assigned a probability of occurrence, computed as
the product of the probabilities of its individual parameter values.
"""

from PySub import plot_utils as plot
from PySub import utils
from PySub.memory import build_bucket_ensemble, load_observation_points

import os
import numpy as np

line = ((210950, 568010), (220001, 568300.1), (225001, 575300.1))

if __name__ == "__main__":
    if not os.getcwd().endswith("Example_scripts"):
        os.chdir("Example_scripts")
    import_paths = [
        r".\BucketEnsemble\Een.json",  # Linear compaction model
    ]

    # Project folder is required for running bucket ensemble
    project_folder = os.getcwd()

    model = build_bucket_ensemble(
        import_paths, name="BucketEnsemble_test", project_folder=project_folder
    )

    observation_points = load_observation_points(r".\BucketEnsemble\Een.json")

    model.set_observation_points(observation_points)

    results, probabilities, error = model.calculate_deterministic(
        iterations=None, all_timesteps=True
    )
    (
        _,
        (p90, p50, p10),
        (model_index_p90, model_index_p50, model_index_p10),
    ) = utils.probability_distribution(
        results,
        probabilities=probabilities,  # very important to add the probabilities!
    )
    plot.plot_probability_distribution(
        model,
        results,
        probabilities=probabilities,  # And here as well
        c="k",
    )
    model.calculate_from_sampled(model_index_p10)
    plot.plot_subsidence(model, title="Susbsidence (cm) in final timestep - p10")
    model.calculate_from_sampled(np.where(error == np.min(error))[0][0])
    plot.plot_subsidence(
        model, title="Susbsidence (cm) in final timestep - lowest error"
    )
    plot.plot_subsidence_observations(model)
