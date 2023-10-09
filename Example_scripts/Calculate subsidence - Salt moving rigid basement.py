"""Example and example script for determining subsidence due to volume change in caverns.
"""
from PySub.memory import build_cavern_model
from PySub.SubsidenceSuite import ModelSuite
from PySub import plot_utils as plot

LINE = ((214000, 573250), (220000, 573800))
if __name__ == "__main__":
    import_path_salt = r"Input example salt.xlsx"
    import_path_MRB = r"Input example salt MRB.xlsx"

    salt_model = build_cavern_model(
        import_path_salt, name="withouth MRB", project_folder=None
    )
    MRB_model = build_cavern_model(
        import_path_MRB, name="with MRB", project_folder=None
    )
    salt_model.calculate_subsidence()
    plot.plot_subsidence(salt_model)

    MRB_model.calculate_subsidence()
    plot.plot_subsidence(MRB_model)

    Suite = ModelSuite("compare salt models", None)
    Suite.set_models((salt_model, MRB_model))

    plot.plot_cross_section(
        Suite,
        lines=LINE,
        time=-1,
        y_axis_exageration_factor=1,
        figsize=(12, 12),
    )
    plot.plot_timeseries(Suite, mode="max")
