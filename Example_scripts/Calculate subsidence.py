"""Example script for modeling workflow for subsidence. Compares available
compaction methods."""

#%% import packages used in this example
from PySub import plot_utils as plot
from PySub.memory import build_model
from PySub.SubsidenceSuite import ModelSuite

line = ((210950, 568010), (220001, 568300.1), (225001, 575300.1))
if __name__ == "__main__":
    models = []

    #%% Make the linear model
    import_path = "Input example linear.json"
    name = "linear"
    project_folder = None
    Model = build_model(import_path, name=name, project_folder=project_folder)

    Model.calculate_compaction()
    Model.calculate_subsidence()

    # plot with a different background map
    plot.plot_subsidence(Model)
    models.append(Model)

    Model.calculate_subsidence_at_points()
    Model.calculate_subsidence_at_observations()
    plot.plot_subsidence_points(Model, points=["S146"])
    plot.plot_subsidence_observations(Model, observations=["00000001"])

    plot.plot_overlap_cross_section(Model, line, mode="individual")
    plot.plot_overlap_cross_section(Model, line, mode="cumulative")

    #%% Make the time-decay model
    import_path = "Input example time-decay.xlsx"
    name = "time-decay"
    Model = build_model(import_path, name=name, project_folder=project_folder)

    Model.calculate_compaction()
    Model.calculate_subsidence()

    plot.plot_subsidence(Model)
    models.append(Model)

    #%% Make the ratetype model
    import_path = "Input example ratetype.xlsx"
    name = "ratetype"
    Model = build_model(import_path, name=name, project_folder=project_folder)

    Model.calculate_compaction()
    Model.calculate_subsidence()

    plot.plot_subsidence(Model)
    models.append(Model)

    Suite = ModelSuite("compare", project_folder=project_folder)
    Suite.set_models(models)
    plot.plot_timeseries(Suite, mode="max")

    Suite.calculate_subsidence_at_observations()
    Suite.calculate_subsidence_at_points()

    plot.plot_overlap_cross_section(Suite, line, mode="individual")
    plot.plot_overlap_cross_section(Suite, line, mode="cumulative")
