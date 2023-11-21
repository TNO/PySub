# -*- coding: utf-8 -*-
"""Running and displaying multiple SubsidenceModel objects
"""
import os
from PySub import SubsidenceModelGas as _SubsidenceModelGas
from PySub import Points as _Points
from PySub import utils as _utils
from PySub import plot_utils as _plot_utils
from PySub import ProjectFolder as _ProjectFolder
import numpy as np
import pandas as pd
import xarray as xr
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from warnings import warn
from tqdm import tqdm


class ModelSuite:
    """Object to contain subsidence modeling data and functionalities for multiple
    models.

    This object creates multiple SubsidenceModel objects.

    Parameters need to be added to define the SubsidenceModel objects,
    the reservoirs, the timesteps and, optionally, points on which the
    subsidence will be determined. Each with their own dimensionality.
    """

    def __init__(self, name, project_folder=None):
        """Inititalize the ModelSuite object.

        Attributes
        ----------
        models : dict
            Dictionary storing the SubsidenceModel objects. The keys are the names
            of the models.
        number_of_models : int
        number_of_reservoirs : list
            list with the number of reservoirs in each model.
        number_of_steps : list
            list with the number of timesteps in each model.
        number_of_points : list
            list with the number of points at which the subsidence is being calculated
            in each model.
        number_of_observations : list
            list with the number of observation points in each model.
        points : PointCollection
            PointCollection object storing names, x and y data of points.
        observation_points : ObservationCollection
            ObservationCollection object storing the names, location, observations and
            errors of observations. Some functionality regarding observation analysis.
        """
        if isinstance(name, str):
            self.suite_name = name
        else:
            raise Exception(
                f"variable name must be a string, is: {type(name)}"
            )

        self.set_project_folder(project_folder, self.suite_name)
        self._models = None
        self._bounds = None
        self.number_of_models = None
        self._variable_that_set_number_of_models = None
        self._bounds = None
        self._contour_levels = None

        # Defaults
        self._contourf_defaults = {
            "cmap": "winter_r",
            "alpha": 0.5,
            "extend": "both",
        }
        self._contour_defaults = {"cmap": "winter_r"}
        self._clabel_defaults = {"colors": "k", "inline": True, "fontsize": 10}
        self._colorbar_defaults = {
            "cmap": "winter_r",
            "spacing": "proportional",
        }
        self._plot_defaults = {"cmap": "winter_r"}
        self._shape_defaults = {
            "facecolor": "green",
            "edgecolor": "k",
            "alpha": 0.3,
        }
        self._annotation_defaults = {}
        self._scatter_defaults = {}
        self._errorbar_defaults = {
            "cmap": "winter_r",
            "fmt": "o",
            "linestyle": "none",
            "markersize": 2,
            "linewidth": 1,
        }
        self._fill_between_defaults = {"facecolor": "grey", "alpha": 0.5}
        self._raster_defaults = {"cmap": "winter_r", "alpha": 0.8}

    def __repr__(self):
        return repr(self._models)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._models[item]
        if isinstance(item, str):
            return self.models[item]

    def __len__(self):
        if self.hasattr("number_of_models"):
            return self.number_of_models
        else:
            return 0

    def hasattr(self, var):  # redefenition for convenient syntax
        _var = f"_{var}"
        if _var in self.__dict__:
            if _utils.is_iterable(type(self.__dict__[_var])):
                if len(self.__dict__[_var]) > 0:
                    return True
                else:
                    return False
            elif self.__dict__[_var] is not None:
                return True

            else:
                return False
        elif var in self.__dict__:
            if _utils.is_iterable(type(self.__dict__[var])):
                if len(self.__dict__[var]) > 0:
                    return True
                else:
                    return False
            elif self.__dict__[var] is not None:
                return True
            else:
                return False
        else:
            try:
                attr = getattr(self, var)
                if attr is not None:
                    return True
                else:
                    return False
            except:
                return False

    # Properties
    @property
    def number_of_reservoirs(self):
        """Property : return the number of reservoirs for each model.

        Returns
        -------
        dict
            Dictionary with the number of reservoirs for each model. The model
            name is the key of the dictionary.

        """
        return self.dict_of_vars("number_of_reservoirs")

    @property
    def number_of_timesteps(self):
        """Property : return the number of timesteps for each model.

        Returns
        -------
        dict
            Dictionary with the number of timesteps for each model. The model
            name is the key of the dictionary.

        """
        return self.dict_of_vars("number_of_timesteps")

    @property
    def number_of_observation_points(self):
        """Property : return the number of observation points for each model.

        Returns
        -------
        dict
            Dictionary with the number of observation_points for each model. The model
            name is the key of the dictionary.

        """
        return self.dict_of_vars("number_of_observation_points")

    @property
    def number_of_points(self):
        """Property : return the number of points at which the subsidence will
        be calculated for each model.

        Returns
        -------
        dict
            Dictionary with the number of points for each model. The model
            name is the key of the dictionary.

        """
        return self.dict_of_vars("number_of_points")

    @property
    def contourf_defaults(self):
        """Property: Represents default settings for the plotting of filled
        contours. Can be adjusted with ModelSuite.set_contourf_defaults()
        function.

        Returns:
        -------
        dict:
            keyword arguments for the matplotlib.pyplot.contourf function.
        """
        return self._contourf_defaults

    @property
    def contour_defaults(self):
        """Property: Represents default settings for the plotting of
        contours. Can be adjusted with ModelSuite.set_contour_defaults()
        function.

        Returns:
        -------
        dict:
            keyword arguments for the matplotlib.pyplot.contour function.
        """
        return self._contour_defaults

    @property
    def clabel_defaults(self):
        """Property: Represents default settings for the plotting of contour
        labels. Can be adjusted with ModelSuite.set_clabel_defaults()
        function.

        Returns:
        -------
        dict:
            keyword arguments for the matplotlib.pyplot.clabel function.
        """
        return self._clabel_defaults

    @property
    def colorbar_defaults(self):
        """Property: Represents default settings for the plotting of colorbars.
        Can be adjusted with ModelSuite.set_colorbar_defaults()
        function.

        Returns:
        -------
        dict:
            keyword arguments for the matplotlib.pyplot.colorbar function.
        """
        return self._colorbar_defaults

    @property
    def plot_defaults(self):
        """Property: Represents default settings for the plotting of lines.
        Can be adjusted with ModelSuite.set_plot_defaults()
        function.

        Returns:
        -------
        dict:
            keyword arguments for the matplotlib.pyplot.plot function.
        """
        return self._plot_defaults

    @property
    def shape_defaults(self):
        """Property: Represents default settings for the plotting of shapes.
        Can be adjusted with ModelSuite.set_shape_defaults()
        function.

        Returns:
        -------
        dict:
            keyword arguments for the matplotlib.patches.polygonfunction.
        """
        return self._shape_defaults

    @property
    def annotation_defaults(self):
        """Property: Represents default settings for the plotting of labels
        in a graph. Can be adjusted with ModelSuite.set_annotation_defaults()
        function.

        Returns:
        -------
        dict:
            keyword arguments for the matplotlib.pyplot.annotate function.
        """
        return self._annotation_defaults

    @property
    def scatter_defaults(self):
        """Property: Represents default settings for the plotting of points
        in graphs. Can be adjusted with ModelSuite.set_scatter_defaults()
        function.

        Returns:
        -------
        dict:
            keyword arguments for the matplotlib.pyplot.scatter function.
        """
        return self._scatter_defaults

    @property
    def errorbar_defaults(self):
        """Property: Represents default settings for the plotting of errorbars.
        Can be adjusted with ModelSuite.set_errorbar_defaults() function.

        Returns:
        -------
        dict:
            keyword arguments for the matplotlib.pyplot.errorbar function.
        """
        return self._errorbar_defaults

    @property
    def fill_between_defaults(self):
        """Property: Represents default settings for the plotting of filled areas.
        Can be adjusted with ModelSuite.set_fill_between_defaults() function.

        Returns:
        -------
        dict:
            keyword arguments for the matplotlib.pyplot.fill_between function.
        """
        return self._fill_between_defaults

    @property
    def raster_defaults(self):
        """Property: Represents default settings for the plotting of rasters.
        Can be adjusted with ModelSuite.set_raster_defaults() function.

        Returns:
        -------
        dict:
            keyword arguments for the matplotlib.pyplot.imshow function.
        """
        return self._raster_defaults

    @property
    def models(self):
        """Property : return a dictionary with the SubsidenceModel objects stored
        in this ModelSuite.

        Returns
        -------
        model_dict : dict
            A dictionary with the SubsidenceModel objects. The keys are the names
            of the models.

        """
        if self.hasattr("models"):
            model_dict = {}
            for model in self._models:
                model_dict[model.name] = model
            return model_dict

    @property
    def name(self):
        """Property : Returns a list with the names of the models.

        Returns
        -------
        list, str
            List with the names of the models.

        """
        if self.hasattr("models"):
            return [m.name for m in self._models]

    @property
    def reservoirs(self):
        """Property : Return a dictionary with the reservoir names in each SubsidenceModel
        object.

        Returns
        -------
        dict,
            A dictionary where for each model a list with
            the reservoir names of that model is stored. The keys of this
            dictionary are the names of the models.

        """
        return self.dict_of_vars("reservoirs")

    @property
    def bounds(self):
        """Property : returns the bounds of this project as set with the
        ModelSuite.set_bounds() method. The bounds of this ModelSuite cover
        all the bounds in its models.

        Returns
        -------
        np.ndarray
           List with 4 values representing the extend/bounds
           of the model grid:
           [0] lower x
           [1] lower y
           [2] upper x
           [3] upper y

        """

        return self._bounds

    @property
    def dx(self):
        """Property : return a dictionary with the distance between grid nodes
        along the x-axis for each SubsidenceModel object.

        Returns
        -------
        dict,
            A dictionary where for each model where a float value is stored for
            the distance between grid nodes aling the x-axis.

        """
        return self.dict_of_vars("dx")

    @property
    def dy(self):
        """Property : return a dictionary with the distance between grid nodes
        along the y-axis for each SubsidenceModel object.

        Returns
        -------
        dict,
            A dictionary where for each model where a float value is stored for
            the distance between grid nodes aling the y-axis.

        """
        return self.dict_of_vars("dy")

    @property
    def influence_radius(self):
        """Property : return a dictionary with the influence radius for each SubsidenceModel
        object. The influence radius determines the extend of the SubsidenceModel
        grid together with the area covered by the reservoir shape. Distance from
        which the subsidence is set to 0 in m.

        Returns
        -------
        dict,
            A dictionary where for each model where a float value is stored for
            the distance between grid nodes aling the x-axis.

        """
        return self.dict_of_vars("influence_radius")

    @property
    def compaction_model(self):
        """Property: The type of compaction model used for each model and reservoir.
        Can be adjusted with ModelSuite.set_compaction_model_parameters() function.

        Returns:
        -------
        dict
            returns a dictionary with for each model a list of PySub.CompactionModels
            objects. The keys of the dictionary are the names of the models.
            The types of compaction models as defined in PySub.CompactionModels
            for each reservoir.
            Available compaction models: # TODO: keep updated with added methods
            - linear (LineraCompaction object)
            - time-decay (TimeDecayCompaction object)
            - ratetype (RateTypeCompaction object)
            The returned list has the same length as the number of reservoirs in
            the model.
        """
        return self.dict_of_vars("compaction_model")

    @property
    def subsidence_model(self):
        """Property: The method used for calculating the subsidence based on
        compaction. Can be adjusted with ModelSuite.set_subsidence_model_parameters()
        function.

        Returns:
        -------
        dict
            A dictionary with a string value for the method of subsidence
            calculation for each model. The keys of the dictionary are the names
            of the models.
            Method of subsidence of the model. Currently available: # TODO: keep updated with added methods
            - Nucleus of strain, Van Opstal 1974
            - Knothe, Stroka et al. 2011
        """
        return self.dict_of_vars("subsidence_model")

    @property
    def shapefiles(self):
        """Property: The location of the path files on which the extend of the
        reservoirs are based.
        Can be adjusted with ModelSuite.set_shapefiles() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with paths to the locations of shape- or raster files for each reservoir.

            The list in the dictionary has the same length as the number of reservoirs in
            the model.
        """
        return self.dict_of_vars("shapefiles")

    @property
    def depths(self):
        """Property: The depth (m) of the top of the reservoirs.
        Can be adjusted with ModelSuite.set_depths() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with the depths (float) of the top of each reservoir.

            The list in the dictionary has the same length as the number of reservoirs in
            the model.
        """
        return self.dict_of_vars("depths")

    @property
    def depth_to_basements(self):
        """Property: The depth  (m) of the rigid basement for each reservoir.
        Can be adjusted with ModelSuite.set_depth_to_basements() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with the depth to the rigid basement (float) for each reservoir.

            The list in the dictionary has the same length as the number of reservoirs in
            the model.
        """
        return self.dict_of_vars("depth_to_basements")

    @property
    def compaction_coefficients(self):
        """Property: The direct compaction coefficient (1/bar) for each reservoirs.
        Can be adjusted with ModelSuite.set_compaction_coefficients() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with the direct compaction coefficients (float) for each reservoir.

            The list in the dictionary has the same length as the number of reservoirs in
            the model.
        """
        return self.dict_of_vars("compaction_coefficients")

    @property
    def thickness(self):
        """Property: The thickness (m) of each reservoirs.
        Can be adjusted with ModelSuite.set_thickness() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with the thickness (float) of each reservoir.

            The list in the dictionary has the same length as the number of reservoirs in
            the model.
        """
        return self.dict_of_vars("thickness")

    @property
    def tau(self):
        """Property: The time decay coefficient (s) for each reservoirs.
        Can be adjusted with ModelSuite.set_compaction_model_parameters() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with the tau values (float) for each reservoir.

            The list in the dictionary has the same length as the number of reservoirs in
            the model.
        """
        return self.dict_of_vars("tau")

    @property
    def compaction_type(self):
        """Property: The direct compaction coefficient for each reservoirs.
        Can be adjusted with ModelSuite.set_compaction_coefficients() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are strings
            with the name of each model's compaction model.

        """
        return self.dict_of_vars("compaction_type")

    @property
    def reference_stress_rates(self):
        """Property: The reference stress rate for each reservoirs.
        Can be adjusted with ModelSuite.set_compaction_coefficients() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with the direct reference stress rates (float) for each reservoir.

            The list in the dictionary has the same length as the number of reservoirs in
            the model.
        """
        return self.dict_of_vars("reference_stress_rates")

    @property
    def b(self):
        """Property: The creep coefficient for the ratetype compaction model
        for each reservoir.
        Can be adjusted with ModelSuite.set_compaction_coefficients() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with the b coefficient (float) for each reservoir.

            The list in the dictionary has the same length as the number of reservoirs in
            the model.
        """
        return self.dict_of_vars("b")

    @property
    def cmref(self):
        """Property: The reference compaction coefficient for each reservoirs.
        Can be adjusted with ModelSuite.set_compaction_coefficients() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with the reference compaction coefficients (float) for each reservoir.

            The list in the dictionary has the same length as the number of reservoirs in
            the model.
        """
        return self.dict_of_vars("cmref")

    @property
    def density(self):
        """Property: The density of the rock above each reservoir.
        Can be adjusted with ModelSuite.set_compaction_coefficients() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with the density (float) for each reservoir.

            The list in the dictionary has the same length as the number of reservoirs in
            the model.
        """
        return self.dict_of_vars("density")

    @property
    def timesteps(self):
        """Property: The timesteps in the models.
        Can be adjusted with ModelSuite.set_timesteps() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with the timesteps (datetime objects).

            The list in the dictionary has the same length as the number of timesteps in
            the model.
        """
        return self.dict_of_vars("timesteps")

    @property
    def pressures(self):
        """Property: For each reservoir, the development of the reservoir
        pressure over time.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are
            2D or 4D lists with the shape of (y, x,) m, n, where m is the number of reservoirs, and
            n is the number of timesteps. Contains the pressure development over
            time for eacht reservoir in bar.
        """
        return self.dict_of_vars("pressures")

    @property
    def shape(self):
        """Property: Dimensions of the grid the calculations are taken over.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are
            lists with a length of 4 (the number of dimensions, minus the dimensions
                                      for the number of points and observations!):
            [0] SubsidenceModel.ny, the number of cells over the y-axis
            [1] SubsidenceModel.nx, the number of cells over the x-axis
            [2] SubsidenceModel.number_of_reservoirs
            [3] SubsidenceModel.number_of_steps
            Returns None for each value that has not been set. Nx and ny are set
            with the set_bounds() function. Number_of_reservoirs and number_of_steps
            are each set after parameters are set that are dependent on the number
            of those dimensions
        """
        return self.dict_of_vars("shape")

    @property
    def x(self):
        """Property: X-coordinates of the grid nodes. Determined by the bounds of
        the grid and the grid cell size. Set after the grid is built.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with he coordinates of the grid nodes along the x-axis.
            The returned list has the same length as nx, the number of grid nodes
            in the model.
        """
        return self.dict_of_vars("x")

    @property
    def y(self):
        """Property: Y-coordinates of the grid nodes. Determined by the bounds of
        the grid and the grid cell size. Set after the grid is built.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are lists
            with he coordinates of the grid nodes along the y-axis.
            The returned list has the same length as ny, the number of grid nodes
            in the model.
        """
        return self.dict_of_vars("y")

    @property
    def compaction(self):
        """Property: Compaction in meters as determined by the
        calculate_compaction() function.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are xarray DataArrays
            with the compaction in meters for each reservoir per
            timestep over the entirety of the model grid. Shape =
            (y, x, reservoir, time).
        """
        return self.dict_of_vars("compaction")

    @property
    def subsidence(self):
        """Property: subsidence over time in m for each Model grid node as
        determined by the calculate_subsidence() method. Seperate for each
        reservoir.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are xarray DataArrays
            with the subsidence in m for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).
        """
        return self.dict_of_vars("subsidence")

    @property
    def slope(self):
        """Property : gradient of the subsidence bowl in m/m for each grid node,
        reservoir and timestep as determined by the calculate_slope() method.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are xarray DataArrays
            with the gradient of the subsidence bowl in m/m for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).

        """
        return self.dict_of_vars("slope")

    @property
    def concavity(self):
        """Property : concavity of the subsidence bowl in m/m² for each grid node,
        reservoir and timestep as determined by the calculate_concavity() method.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are xarray DataArrays
            with the concavity of the subsidence bowl in m/m² for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).

        """
        return self.dict_of_vars("concavity")

    @property
    def subsidence_rate(self):
        """Property : rate of the subsidence bowl in m/y for each grid node,
        reservoir and timestep as determined by the calculate_subsidence_rate() method.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are xarray DataArrays
            with the rate of the subsidence bowl in m/y for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).

        """
        return self.dict_of_vars("subsidence_rate")

    @property
    def volume(self):
        """Property : volume of the subsidence bowl in m³ for each grid node,
        reservoir and timestep as determined by the calculate_volume() method.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are xarray DataArrays
            with the volume of the subsidence bowl in m³ for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).

        """
        return self.dict_of_vars("volume")

    @property
    def point_subsidence(self):
        """Property : Subsidence at the stored point objects in the model (Model.points)
        for each reservoir and timestep as determined by the calculate_subsidence_at_points()
        method.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are xarray DataArrays
            with the subsidence in m for each reservoir per timestep
            over the entirety of the grid. Shape = (points, reservoir, time).

        """
        return self.dict_of_vars("points_subsidence")

    @property
    def observation_subsidence(self):
        """Property : Subsidence at the stored point objects in the model (Model.observation_points)
        for each reservoir and timestep as determined by the calculate_subsidence_at_observations()
        method.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are xarray DataArrays
            with the subsidence in m for each reservoir per timestep
            over the entirety of the grid. Shape = (observations, reservoir, time).

        """
        return self.dict_of_vars("observation_subsidence")

    @property
    def total_subsidence(self):
        """Property : The subsidence taking into consideration all reservoirs in the model,
        for each timestep as determined by the calculate_subsidence() method.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are xarray DataArrays
            with the subsidence (m) per timestep over the entirety of the grid.
            Shape = (y, x, time).

        """
        return self.dict_of_vars("total_subsidence")

    @property
    def points(self):
        """Property: The PointCollection object storing the coordinates of any points of
        intereset in the model.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are PointCollection
            objects storing the coordinates of the points of interest in the models.

        """
        return self.dict_of_vars("points")

    @property
    def observation_points(self):
        """Property: The ObservationCollection object storing the coordinates and
        observed values at which times and their errors.

        Returns:
        -------
        dict
            A dictionary with keys for each model. The keys of the doctionary are
            the names of the models. The values stored in the model are ObservationCollection
            objects storing the coordinates, time and error of the observationponts.

        """
        return self.dict_of_vars("observation_points")

    def unique_observations(self, observations=None, model=None):
        """Returns a ObservationCollection object with the unique observations in
        all the models in the Suite.

        Parameters
        ----------
        observations : int/str or list of int/str, optional
            The label of an observation (as a name, or the index as an integer) to
            indicate a specific observation. A list of these labels can also be used.
            The default is None, when None, all the observations will be retrieved.
            NB: When using integers, there can be differences in ordering between
            models.
        model : str or list of str, optional
            The label for a specific model in the Suite to retrieve the unique
            observations from. The label is de name of the model (Subsdencemodel.name)
            or a list of names of the models in the Suite. The default is None,
            when None, the observations for all models will be retrieved.

        Returns
        -------
        ObservationCollection
            ObservationCollection object with the unique observations in
            all the models in the Suite.

        """

        self._unique_observations = self.get_unique_observations(
            observations=observations, model=model
        )
        return self._unique_observations

    def get_unique_observations(self, observations=None, model=None):
        """Returns a ObservationCollection object with the unique observations in
        all the models in the Suite.

        Parameters
        ----------
        observations : int/str or list of int/str, optional
            The label of an observation (as a name, or the index as an integer) to
            indicate a specific observation. A list of these labels can also be used.
            The default is None, when None, all the observations will be retrieved.
            NB: When using integers, there can be differences in ordering between
            models.
        model : str or list of str, optional
            The label for a specific model in the Suite to retrieve the unique
            observations from. The label is de name of the model (Subsdencemodel.name)
            or a list of names of the models in the Suite. The default is None,
            when None, the observations for all models will be retrieved.

        Returns
        -------
        ObservationCollection
            ObservationCollection object with the unique observations in
            all the models in the Suite.

        """
        if observations is None:
            if self.hasattr("observation_points"):
                observations = np.unique(
                    [
                        col.names
                        for model_name, col in self.observation_points.items()
                    ]
                )
            else:
                return
        else:
            _observations = [
                col[observations]
                for model_name, col in self.observation_points.items()
            ]
            observations = [[i.name for i in j] for j in _observations]
            observations = _utils.flatten_ragged_lists2D(observations)
            observations = np.unique(observations)

        # check for inconsistensies
        double_named = {
            o: [m for m in self.name if o in self.observation_points[m].names]
            for o in observations
        }

        dfs = {
            o: {
                m: self.observation_points[m]
                .as_df.loc[[o]]
                .sort_values("Time")
                for m in models
            }
            for o, models in double_named.items()
        }

        check_dfs = {
            o: np.array(
                [
                    [not o1.equals(o2) for o2 in df.values()]
                    for o1 in df.values()
                ]
            ).any(axis=0)
            for o, df in dfs.items()
        }

        doubles = {
            o: np.array(double_named[o])[check]
            for o, check in check_dfs.items()
        }

        exception = []
        for double_observation, different_models in doubles.items():
            if len(different_models) > 1:
                print_models = ", ".join(different_models)
                exception.append(
                    f'The observation "{double_observation}" has entries in the models {print_models} with the same name, but different entries.'
                )
        if len(exception) > 1:
            raise Exception("\n".join(exception))

        unique_observations = [
            self.observation_points[m[0]][o] for o, m in double_named.items()
        ]

        return _Points.ObservationCollection(unique_observations)

    def unique_relative(self, observation_name=None, model=None):
        """Returns a ObservationCollection object with the unique observations
        relative to the first observation in the series of observation for all
        the models in the Suite.

        Parameters
        ----------
        observations : int/str or list of int/str, optional
            The label of an observation (as a name, or the index as an integer) to
            indicate a specific observation. A list of these labels can also be used.
            The default is None, when None, all the observations will be retrieved.
            NB: When using integers, there can be differences in ordering between
            models.
        model : str or list of str, optional
            The label for a specific model in the Suite to retrieve the unique
            observations from. The label is de name of the model (Subsdencemodel.name)
            or a list of names of the models in the Suite. The default is None,
            when None, the observations for all models will be retrieved.

        Returns
        -------
        ObsevrationCollection
            ObservationCollection object with the unique observations relative to
            the first observation in the series for all the models in the Suite.

        """
        if observation_name is None:
            if self.hasattr("observation_points"):
                observation_name = np.unique(
                    [
                        col.names
                        for model_name, col in self.observation_points.items()
                    ]
                )
            else:
                return
        else:
            if (
                type(observation_name) == list
                or type(observation_name) == np.ndarray
            ):
                pass
            elif type(observation_name) == str:
                observation_name = [observation_name]
            else:
                raise Exception(
                    f"Data type {type(observation_name)} is invalid."
                )

        if model is None:
            model_index = list(range(self.number_of_models))
        else:
            model_index = self.model_label_to_index(model)

        names = []
        x = []
        y = []
        time = []
        observation = []
        lower_error = []
        upper_error = []
        for m, (_, observation_points) in enumerate(
            self.observation_points.items()
        ):
            if m in model_index:
                for i, name in enumerate(observation_points.names):
                    if name in observation_name:
                        integer_timestamps = observation_points.time[i].astype(
                            int
                        )
                        for j, observations in enumerate(
                            observation_points.relative[i]
                        ):
                            names.append(name)
                            x.append(observation_points.x[i])
                            y.append(observation_points.y[i])
                            time.append(integer_timestamps[j])
                            observation.append(
                                observation_points.relative[i][j]
                            )
                            lower_error.append(
                                observation_points.lower_errors[i][j]
                            )
                            upper_error.append(
                                observation_points.upper_errors[i][j]
                            )
        all_observations = np.array(
            [names, x, y, time, observation, lower_error, upper_error]
        )
        unique_observations = np.unique(all_observations, axis=1)
        (
            names,
            x,
            y,
            time,
            observation,
            lower_error,
            upper_error,
        ) = unique_observations
        time = pd.to_datetime([int(t) for t in time])
        points = []
        for i, name in enumerate(observation_name):
            indeces = np.where(names == name)
            p = _Points.ObservationPoint(
                name, x.astype(float)[indeces], y.astype(float)[indeces]
            )
            p.set_time(time[indeces])
            p.set_observations(observation.astype(float)[indeces])
            p.set_lower_error(lower_error.astype(float)[indeces])
            p.set_upper_error(upper_error.astype(float)[indeces])
            points.append(p)
        return _Points.ObservationCollection(points)

    def unique_reservoirs(self):
        """Get a list of all the unique reservoirs in the models of the Suite.

        Returns
        -------
        unique : list
            A list with all the unique reservoirs in the Suite.

        """
        if self.hasattr("models"):
            unique = []
            for m in self._models:
                for r in m.reservoirs:
                    if r not in unique:
                        unique.append(r)
            return unique

    def reservoir_label_to_int(self, val):
        """Returns the index of the reservoir from unique reservoirs in Suite based
        on the entry val.

        Parameters
        ----------
        val : int/str
            Representation of the reservoir by index or name.
            Examples by index: 0, 1, -1, -2.
            Examples by reservoir name: 'Groningen'.

        Returns
        -------
        int
            Index of the reservoir.
        """
        index = 0
        unique_reservoirs = np.array(self.unique_reservoirs())
        if isinstance(val, str):
            if val in self.unique_reservoirs():
                index = np.where(unique_reservoirs == val)[0][0]
            else:
                warn(
                    f"Warning: The key {val} does not correspond with an available reservoir"
                )
                return None
        elif isinstance(val, int):
            if val in list(range(len(unique_reservoirs))):
                index = val
            elif val in list(range(-len(unique_reservoirs), 0)):
                index = val
            else:
                raise Exception(
                    f"The key {val} does not correspond with an available reservoir"
                )
        else:
            raise Exception(
                f"Type {type(val)} not supported, use string or integer to index."
            )
        return int(index)

    def reservoir_dict(self):
        unique_reservoirs = np.array(self.unique_reservoirs())
        return {u: i for i, u in enumerate(unique_reservoirs)}

    def unique_timesteps(self):
        if self.hasattr("models"):
            unique = []
            for m in self._models:
                for r in m.timesteps:
                    if r not in unique:
                        unique.append(r)
            return unique

    # Check
    def _dim_exception(self, name, dims_equal, dims, set_var):
        raise Exception(
            f"Variable {name} should have the same number of entries as {dims_equal}, which are currently set to {dims} by variable: {set_var}."
        )

    def _check_dim1D(self, name, var, dims_equal="models", _set=True):
        try:
            var[0]
        except:
            raise Exception("Variable {name} must be array-like.")
        if dims_equal == "models":
            if self.number_of_models == None:
                if _set:
                    self.number_of_models = len(var)
                    self._variable_that_set_number_of_models = name
                else:
                    raise Exception(
                        "Number of models has not been determined, set with appropriate variable or explicitly."
                    )
            elif len(var) != self.number_of_models:
                self._dim_exception(
                    name,
                    dims_equal,
                    self.number_of_models,
                    self._variable_that_set_number_of_models,
                )

    # Set plotting defaults
    def set_contourf_defaults(self, kwargs={}):
        """Set the standard keyword arguments for matplotlib.pyplot.contourf for
        all models in the Suite.
        These will influence the presentation of filled contours in figures.

        This will set the keyword arguments for the relevant functions for the
        individual models as well.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.contourf
            function. The dictionary should be built like:
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.contourf. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
        """
        self._contourf_defaults = _plot_utils.set_defaults(
            kwargs, defaults=self._contourf_defaults
        )

        if self.hasattr("models"):
            for model in self.models:
                model.set_contourf_defaults(self._contourf_defaults)

    def set_contour_defaults(self, kwargs={}):
        """Set the standard keyword arguments for matplotlib.pyplot.contourf for
        all models in the Suite.
        These will influence the presentation of contours in figures.

        This will set the keyword arguments for the relevant functions for the
        individual models as well.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.contour
            function. The dictionary should be built like:
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.contour. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html
        """
        self._contour_defaults = _plot_utils.set_defaults(
            kwargs, defaults=self._contour_defaults
        )
        if self.hasattr("models"):
            for model in self.models:
                model.set_contour_defaults(self.contour_defaults)

    def set_clabel_defaults(self, kwargs={}):
        """Set the standard keyword arguments for matplotlib.pyplot.clabel for
        all models in the Suite.
        These will influence the presentation of contour labels in figures.

        This will set the keyword arguments for the relevant functions for the
        individual models as well.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.clabel
            function. The dictionary should be built like:
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.clabel. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.clabel.html
        """
        self._clabel_defaults = _plot_utils.set_defaults(
            kwargs, defaults=self._clabel_defaults
        )
        if self.hasattr("models"):
            for model in self.models:
                model.set_clabel_defaults(self.clabel_defaults)

    def set_colorbar_defaults(self, kwargs={}):
        """Set the standard keyword arguments for matplotlib.pyplot.colorbar for
        all models in the Suite.
        These will influence the presentation of colorbars in figures.

        This will set the keyword arguments for the relevant functions for the
        individual models as well.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.colorbar
            function. The dictionary should be built like:
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.colorbar. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
        """
        self._colorbar_defaults = _plot_utils.set_defaults(
            kwargs, defaults=self._colorbar_defaults
        )
        if self.hasattr("models"):
            for model in self.models:
                model.set_colorbar_defaults(self.colorbar_defaults)

    def set_plot_defaults(self, kwargs={}):
        """Set the standard keyword arguments for matplotlib.pyplot.plot for
        all models in the Suite.
        These will influence the presentation of plotted lines in figures.

        This will set the keyword arguments for the relevant functions for the
        individual models as well.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.plot
            function. The dictionary should be built like:
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.plot. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
        """
        self._plot_defaults = _plot_utils.set_defaults(
            kwargs, defaults=self._plot_defaults
        )
        if self.hasattr("models"):
            for model in self.models:
                model.set_plot_defaults(self.plot_defaults)

    def set_shape_defaults(self, kwargs={}):
        """Set the standard keyword arguments for matplotlib.patches.Polygon for
        all models in the Suite.
        These will influence the presentation of filled shapes in figures.

        This will set the keyword arguments for the relevant functions for the
        individual models as well.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.patches.Polygon
            function. The dictionary should be built like:
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.patches.Polygon. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html
        """
        self._shape_defaults = _plot_utils.set_defaults(
            kwargs, defaults=self._shape_defaults
        )
        if self.hasattr("models"):
            for model in self.models:
                model.set_shape_defaults(self.shape_defaults)

    def set_raster_defaults(self, kwargs={}):
        """Set the standard keyword arguments for matplotlib.patches.Polygon for
        all models in the Suite.
        These will influence the presentation of filled shapes in figures.

        This will set the keyword arguments for the relevant functions for the
        individual models as well.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.patches.Polygon
            function. The dictionary should be built like:
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.patches.Polygon. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html
        """
        self._raster_defaults = _plot_utils.set_defaults(
            kwargs, defaults=self._raster_defaults
        )
        if self.hasattr("models"):
            for model in self.models:
                model.set_raster_defaults(self.raster_defaults)

    def set_annotation_defaults(self, kwargs={}):
        """Set the standard keyword arguments for matplotlib.pyplot.annotation for
        all models in the Suite.
        These will influence the presentation of annotation in figures.

        This will set the keyword arguments for the relevant functions for the
        individual models as well.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.annotation
            function. The dictionary should be built like:
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.annotation. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotation.html
        """
        self._annotation_defaults = _plot_utils.set_defaults(
            kwargs, defaults=self._annotation_defaults
        )
        if self.hasattr("models"):
            for model in self.models:
                model.set_annotation_defaults(self.annotation_defaults)

    def set_scatter_defaults(self, kwargs={}):
        """Set the standard keyword arguments for matplotlib.pyplot.scatter for
        all models in the Suite.
        These will influence the presentation of points in figures.

        This will set the keyword arguments for the relevant functions for the
        individual models as well.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.scatter
            function. The dictionary should be built like:
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.scatter. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
        """
        self._scatter_defaults = _plot_utils.set_defaults(
            kwargs, defaults=self._scatter_defaults
        )
        if self.hasattr("models"):
            for model in self.models:
                model.set_scatter_defaults(self.scatter_defaults)

    def set_errorbar_defaults(self, kwargs={}):
        """Set the standard keyword arguments for matplotlib.pyplot.errorbars for
        all models in the Suite.
        These will influence the presentation of errorbars in figures.

        This will set the keyword arguments for the relevant functions for the
        individual models as well.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.errorbars
            function. The dictionary should be built like:
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.errorbars. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbars.html
        """
        self._errorbar_defaults = _plot_utils.set_defaults(
            kwargs, self._errorbar_defaults
        )
        if self.hasattr("models"):
            for model in self.models:
                model.set_errorbar_defaults(self.errorbar_defaults)

    def set_fill_between_defaults(self, kwargs={}):
        """Set the standard keyword arguments for matplotlib.pyplot.fill_between for
        this Suite.
        These will influence the presentation of colored areas between lines in
        figures.

        This will set the keyword arguments for the relevant functions for the
        individual models as well.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.fill_between
            function. The dictionary should be built like:
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.fill_between. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html
        """
        self._fill_between_defaults = _plot_utils.set_defaults(
            kwargs, self._fill_between_defaults
        )
        if self.hasattr("models"):
            for model in self._models:
                model.set_fill_between_defaults(self.fill_between_defaults)

    def get_contour_levels(
        self,
        variable="subsidence",
        levels=None,
        start=None,
        end=None,
        contour_steps=0.01,
        drop_value=0,
    ):
        """
        Set the contour levels which are to be plotted for all the models in the Suite.

        Parameters
        ----------
        levels : list floats, optional
            A list with values for the contour levels that are to be presented in m.
            The default is None, when None, the Suite will base the contour levels of the
            subsidence plots based on the values already determined using calculate_subsidence()
            or using the start and end value.
            When levels is None, no subsidence has been calculated and no start and
            end value have been entered, an Exception will occur.
        start : float, optional
            The lowest value (in m) in the to be plotted range. The default is None, which will
            raise en Exception when levels is None and no subsidence has been calculated.
        end : float, optional
            The highest value (in m) in the to be plotted range. The default is None, which will
            raise en Exception when levels is None and no subsidence has been calculated.
        contour_steps : float, optional
            The spacing between contour levels in m. The default is 0.01.
        drop_value : float, optional
            A single value can be removed from the plotted range. The default is 0.

        Sets
        -------
        Suite.contour_levels.

        """
        if levels is None:
            if start is not None and end is not None:
                levels = _utils.stepped_space(start, end, contour_steps)
            else:
                if self.hasattr("models"):
                    model_contour_levels = [
                        m.get_contour_levels() for m in self
                    ]
                    min_values, max_values = list(
                        zip(
                            *[
                                (np.min(c), np.max(c))
                                for c in model_contour_levels
                            ]
                        )
                    )
                    min_value, max_value = np.min(min_values), np.max(
                        max_values
                    )
                    levels = _utils.stepped_space(
                        min_value, max_value, contour_steps
                    )
                    levels = _plot_utils.set_contour_levels(
                        contour_levels=levels, contour_steps=contour_steps
                    )
                    return levels
            levels = _plot_utils.set_contour_levels(
                contour_levels=levels,
                contour_steps=contour_steps,
                drop_value=drop_value,
            )
        elif not _utils.is_itterable(levels):
            warn(
                'Warning: Not enough information to set contour levels. Assign a list to "levels", or assign values to "start", "end" and "contour_steps"'
            )
        return levels

    # Set parameters
    def set_project_folder(self, folder=None, name=None):
        if name is None:
            if self.hasattr("name"):
                name = self.suite_name
            else:
                name = "unnamed_subsidence_suite"
        if folder is not None:
            project_folder = os.path.join(folder, name)
        else:
            project_folder = None

        self.project_folder = _ProjectFolder.ProjectFolder(project_folder)
        if self.hasattr("models"):
            for m in self._models:
                model_project_folder = os.path.join(
                    project_folder, "input", m.name
                )
                m.set_project_folder(model_project_folder)

    def add_model(self, model):
        """Add an existing SubsidenceModel to the Suite.

        Parameters
        ----------
        model : SubsidenceModel object.

        Raises
        ------
        Exception
            When an object is entered which is not a SubsidenceModel object.

        """
        if _utils.isSubsidenceModel(model):
            if self._models is None:
                self._models = []
            if model.name in self.name:
                warn(
                    f'Warning: Model with the name "{model.name}" already in Suite. skipped. Try adding with different name if not the same model.'
                )
            else:
                model.set_project_folder(self.project_folder.project_folder)
                self._models.append(model)
        else:
            raise Exception(
                "Models part of ModelSuite object need to be a SubsidenceModel object."
            )

    def initiate_models(self, names):
        """Create empty SubsidenceModels using their names. Also sets the SubsidenceModel
        object's figure settings the same as the Suite's.

        Parameters
        ----------
        names : list, str
            A list with unique names for each model.

        Raises
        ------
        Exception
            When the names are not unique, or an invalid format is being used.

        Sets
        -------
        Suite.model_names and Suite.models.

        """
        if not _utils.is_list_of_strings(names):
            raise Exception(
                "Suite.inititate_models takes a list of strings representing model names."
            )

        _utils.except_if_not_unique("name", names)

        self._check_dim1D("Model names", names, dims_equal="models")

        valid_types = np.array([_utils.is_str_or_int(val) for val in names])
        if np.logical_not(valid_types).any():
            raise Exception(
                "Model names need to be a list or 1D np.ndarray strings or integers"
            )

        self.model_names = names

        for i in range(len(names)):
            model_project_folder = os.path.join(
                self.project_folder.project_folder, names[i]
            )
            model = _SubsidenceModelGas.SubsidenceModel(
                names[i], project_folder=model_project_folder
            )
            model.set_contourf_defaults(self.contourf_defaults)
            model.set_contour_defaults(self.contour_defaults)
            model.set_clabel_defaults(self.clabel_defaults)
            model.set_colorbar_defaults(self.colorbar_defaults)
            model.set_plot_defaults(self.plot_defaults)
            model.set_shape_defaults(self.shape_defaults)
            model.set_annotation_defaults(self.annotation_defaults)
            model.set_scatter_defaults(self.scatter_defaults)
            model.set_errorbar_defaults(self.errorbar_defaults)

            self.add_model(model)

    def set_models(self, list_of_models):
        """Make a model using exisitng SubsidenceModel objects.

        Parameters
        ----------
        list_of_models : list, SubsidenceModel objects
            A list with .

        Returns
        -------
        None.

        """
        self.number_of_models = len(list_of_models)
        for i, _model in enumerate(list_of_models):
            model = _model.copy()
            if _utils.isSubsidenceModel(model):
                model.set_contourf_defaults(self.contourf_defaults)
                model.set_contour_defaults(self.contour_defaults)
                model.set_clabel_defaults(self.clabel_defaults)
                model.set_colorbar_defaults(self.colorbar_defaults)
                model.set_plot_defaults(self.plot_defaults)
                model.set_shape_defaults(self.shape_defaults)
                model.set_annotation_defaults(self.annotation_defaults)
                model.set_scatter_defaults(self.scatter_defaults)
                model.set_errorbar_defaults(self.errorbar_defaults)
                if not model.name:
                    model.name = "Model " + str(i + 1)
                    warn(
                        f"Warning: Model added without name, name set as: {model.name}."
                    )
                self.add_model(model)
            else:
                raise Exception(
                    "Invalid model type to add to ModelSuite. Add SubsidenceModel object types only."
                )
        self.set_bounds(bounds=None, all_same_bounds=False)

    def _check_for_models(self, function):
        if self.hasattr("models"):
            return True
        else:
            warn(f"Warning: No models in Suite, {function}() failed!")
            return False

    def set_observation_points(self, observation_points):
        if self._check_for_models("set_observation_points"):
            for model in self._models:
                model.set_observation_points(observation_points)

    def set_points(self, points):
        if self._check_for_models("set_points"):
            for model in self._models:
                model.set_points(points)

    def set_dx(self, multi_dx):
        if self._check_for_models("set_dx"):
            self._check_dim1D("dx", multi_dx, dims_equal="models")
            for i, model in enumerate(self._models):
                model.set_dx(multi_dx[i])

    def set_dy(self, multi_dy=None):
        if self._check_for_models("set_dy"):
            if multi_dy is not None:
                self._check_dim1D("dy", multi_dy, dims_equal="models")
                for i, model in enumerate(self._models):
                    model.set_dy(dy=multi_dy[i])
            else:
                for i, model in enumerate(self._models):
                    if hasattr(model, "dx"):
                        model.set_dy(dy=model.dx)
                    else:
                        raise Exception(
                            "When setting dy for models, supply a dy. When dy is None, have previously set dx."
                        )

    def set_influence_radius(self, multi_influence_radius):
        if self._check_for_models("set_influence_radius"):
            self._check_dim1D(
                "dy", multi_influence_radius, dims_equal="models"
            )
            for i, model in enumerate(self._models):
                model.set_influence_radius(multi_influence_radius[i])

    def set_compaction_model_parameters(
        self,
        multi_compaction_model,
        multi_compaction_coefficients,
        multi_tau,
        multi_reference_stress_rates,
        multi_density,
        multi_cmref,
        multi_b,
    ):
        if self._check_for_models("set_compaction_model"):
            self.set_compaction_coefficients(multi_compaction_coefficients)
            self._check_dim1D(
                "compaction_model", multi_compaction_model, dims_equal="models"
            )
            if multi_tau is None:
                multi_tau = [None] * self.number_of_models
            self._check_dim1D("tau", multi_tau, dims_equal="models")
            if multi_reference_stress_rates is None:
                multi_reference_stress_rates = [None] * self.number_of_models
            self._check_dim1D(
                "reference_stress_rates",
                multi_reference_stress_rates,
                dims_equal="models",
            )
            if multi_density is None:
                multi_density = [None] * self.number_of_models
            self._check_dim1D("density", multi_density, dims_equal="models")

            if multi_cmref is None:
                multi_cmref = [None] * self.number_of_models
            self._check_dim1D("cmref", multi_cmref, dims_equal="models")
            if multi_b is None:
                multi_b = [None] * self.number_of_models
            self._check_dim1D("b", multi_b, dims_equal="models")
            for i, model in enumerate(self._models):
                model.set_compaction_model_parameters(
                    multi_compaction_model[i],
                    multi_tau[i],
                    multi_reference_stress_rates[i],
                    multi_density[i],
                    multi_cmref[i],
                    multi_b[i],
                )

    def set_subsidence_model_parameters(
        self,
        multi_subsidence_model,
        knothe_angles=None,
        depth_to_basements=None,
    ):
        if self._check_for_models("set_subsidence_model"):
            self._check_dim1D(
                "subsidence_model", multi_subsidence_model, dims_equal="models"
            )
            if knothe_angles is None:
                knothe_angles = [None] * self.number_of_models
            else:
                self._check_dim1D(
                    "knothe_angles", knothe_angles, dims_equal="models"
                )

            if depth_to_basements is None:
                depth_to_basements = [None] * self.number_of_models
            else:
                self._check_dim1D(
                    "depth_to_basements",
                    depth_to_basements,
                    dims_equal="models",
                )

            for i, model in enumerate(self._models):
                model.set_subsidence_model_parameters(
                    multi_subsidence_model[i],
                    knothe_angles=knothe_angles[i],
                    depth_to_basements=depth_to_basements[i],
                )

    def set_reservoirs(self, multi_reservoir_names):
        if self._check_for_models("set_reservoirs"):
            self._check_dim1D(
                "reservoirs", multi_reservoir_names, dims_equal="models"
            )
            for i, model in enumerate(self._models):
                model.set_reservoirs(multi_reservoir_names[i])

    def set_shapefiles(self, multi_shapefile_paths):
        if self._check_for_models("set_shapefiles"):
            if multi_shapefile_paths is None:
                multi_shapefile_paths = [None] * self.number_of_models
            self._check_dim1D(
                "shapefiles", multi_shapefile_paths, dims_equal="models"
            )
            for i, model in enumerate(self._models):
                model.set_shapefiles(multi_shapefile_paths[i])

    def set_depths(self, multi_depths):
        if self._check_for_models("set_depths"):
            self._check_dim1D("depths", multi_depths, dims_equal="models")
            for i, model in enumerate(self._models):
                model.set_depths(multi_depths[i])

    def set_depth_to_basements(self, multi_depth_to_basements):
        if self._check_for_models("set_depth_to_basements"):
            self._check_dim1D(
                "depths", multi_depth_to_basements, dims_equal="models"
            )
            for i, model in enumerate(self._models):
                model.set_depth_to_basements(multi_depth_to_basements[i])

    def set_compaction_coefficients(self, multi_compaction_coefficients):
        if self._check_for_models("set_compaction_coefficients"):
            self._check_dim1D(
                "compaction_coefficients",
                multi_compaction_coefficients,
                dims_equal="models",
            )
            for i, model in enumerate(self._models):
                model.set_compaction_coefficients(
                    multi_compaction_coefficients[i]
                )

    def set_thickness(self, multi_thickness):
        if self._check_for_models("set_thickness"):
            self._check_dim1D(
                "thickness", multi_thickness, dims_equal="models"
            )
            for i, model in enumerate(self._models):
                model.set_thickness(multi_thickness[i])

    def set_timesteps(self, multi_timesteps):
        if self._check_for_models("set_timesteps"):
            self._check_dim1D(
                "timesteps", multi_timesteps, dims_equal="models"
            )
            for i, model in enumerate(self._models):
                model.set_timesteps(multi_timesteps[i])

    def set_bounds(self, bounds=None, all_same_bounds=True):
        if bounds is None:
            if self._check_for_models("set_bounds"):
                shapes = []
                model_bounds = []
                for model in self._models:
                    if model.hasattr("bounds"):
                        model_bounds.append(model.bounds)
                    elif model.hasattr("shapes"):
                        shapes.append(model.shapes)
                    else:
                        raise Exception(
                            "Not enough information to base bounds on. Set bounds of this function, of the models in this Suite or set the shapes of the models in this Suite."
                        )
                if len(shapes) > 0 and len(model_bounds) == 0:
                    shapes_per = _utils.flatten_ragged_lists3D(shapes)
                    np_shapes = shapes_per[0]
                    for p in shapes_per[1:]:
                        np_shapes = np.vstack((np_shapes, p))

                    bounds = np.array(_utils.bounds_from_collection(np_shapes))
                elif len(model_bounds) > 0 and len(shapes) > 0:
                    combined_bounds = np.array(
                        [model_bounds, np.arrray(shapes)]
                    )
                    bounds = np.zeros(4)
                    bounds[0] = np.min(combined_bounds[:, 0])
                    bounds[1] = np.min(combined_bounds[:, 1])
                    bounds[2] = np.max(combined_bounds[:, 2])
                    bounds[3] = np.max(combined_bounds[:, 3])
                elif len(model_bounds) > 0 and len(shapes) == 0:
                    model_bounds = np.array(model_bounds)
                    bounds = np.zeros(4)
                    bounds[0] = np.min(model_bounds[:, 0])
                    bounds[1] = np.min(model_bounds[:, 1])
                    bounds[2] = np.max(model_bounds[:, 2])
                    bounds[3] = np.max(model_bounds[:, 3])
                else:
                    raise Exception(
                        "Not enough information to base bounds on. Set bounds of this function, of the models in this Suite or set the shapes of the models in this Suite."
                    )

                self._bounds = bounds
                if all_same_bounds:
                    for model in self._models:
                        if not model.hasattr("bounds"):
                            model.set_bounds(bounds=self._bounds)
                else:
                    for model in self._models:
                        if not model.hasattr("bounds"):
                            model.set_bounds(bounds=None)
            else:
                return
        else:
            bounds = [s for s in bounds if _utils.is_number(s)]
            if len(bounds) == 4:
                self._bounds = bounds
                for model in self._models:
                    if not model.hasattr("bounds"):
                        model.set_bounds(bounds=self._bounds)

    def set_pressures(self, multi_pressures):
        if self._check_for_models("set_pressures"):
            self._check_dim1D("pressure", multi_pressures, dims_equal="models")
            for i, model in enumerate(self._models):
                model.set_pressures(multi_pressures[i])

    # Assign
    def assign_compaction_parameters(self):
        if self._check_for_models("assign_compaction_parameters"):
            for model in self._models:
                model.assign_compaction_parameters()

    def assign_point_parameters(self):
        if self._check_for_models("assign_point_parameters"):
            for model in self._models:
                model.assign_point_parameters()

    def assign_observation_parameters(self):
        if self._check_for_models("assign_observation_parameters"):
            for model in self._models:
                model.assign_observation_parameters()

    # Build
    def build_grid(self):
        if self._check_for_models("build_grid"):
            for i, model in enumerate(self._models):
                model.build_grid()

    def mask_reservoirs(self):
        if self._check_for_models("mask_reservoirs"):
            for i, model in enumerate(self._models):
                model.mask_reservoirs()

    def mask_from_grid(self, multi_grid):
        if self._check_for_models("mask_from_grid"):
            self._check_dim1D(
                "grid", multi_grid, dims_equal="models", _set=False
            )
            for i, model in enumerate(self._models):
                model.mask_from_grid(multi_grid[i])

    # Tools
    def dict_of_vars(self, var):
        """Suite util : create a dictionary with the variable asked for in var,
        and the keys will be the model names.

        Parameters
        ----------
        Suite : PySub.SubsidenceSuite.ModelSuite object
        var : str
            A variable stored in the models. If the variable does not exist, this
            function raises an exception.

        Raises
        ------
        Exception
            Specified model doesn't have the requested attribute.

        Returns
        -------
        dict_var : dict
            Dictionary which stores the variables of the models in the Suite, where
            where the keys are the model names. NB: Doesn't check if model names are
            unique! Will overwrite variables of models with the same name.

        """
        if self.hasattr("models"):
            dict_var = {}
            for model in self._models:
                if model.hasattr(var):
                    dict_var[model.name] = getattr(model, var)
                else:
                    raise Exception(
                        f"Model {model.name} of type {type(model)} has no attribute: {var}."
                    )
            return dict_var

    def model_name_to_index(self, name):
        """Returns the index of the model based on its name.

        Parameters
        ----------
        name : str
            Name of a model.

        Returns
        -------
        int
            Index of the model.

        """
        if self.hasattr("models"):
            return np.where(np.array(self.name) == name)[0][0]

    def model_label_to_index(self, label):
        """Return the index of the model based on any type of model label.

        Parameters
        ----------
        label : int, str, list
            Index or name of model, or a list with multiple indeces and names of models..

        Raises
        ------
        Exception
            When an invalid input format is used.

        Returns
        -------
        list
            A list of indeces.

        """
        if self.hasattr("models"):
            if label is None:
                return list(range(self.number_of_models))
            elif type(label) == list or type(label) == np.ndarray:
                return_labels = []
                for l in label:
                    if type(l) == int:
                        return_labels.append(l)
                    elif type(l) == str:
                        return_labels.append(self.model_name_to_index(l))
                return return_labels
            elif type(label) == int:
                return [label]
            elif type(label) == str:
                return [self.model_name_to_index(label)]
            else:
                raise Exception(
                    f"Invalid model label type: {type(label)}. Enter integer or string or list of integers or strings."
                )

    # Calculate
    def calculate_compaction(self):
        """For each model, calulcate the compaction.

        Returns
        -------
        result : dict
            A dictionary with a key for each model. Each value in the dictionary is
            an xarray.DataArray object with the compaction in m with the shape
            (y, x, reservoirs, time)

        """
        if self.hasattr("models"):
            print("Calculating compaction.")
            result = {}
            for model in tqdm(self._models):
                if hasattr(model, "calculate_compaction"):
                    value = model.calculate_compaction(_print=False)
                    result[model.name] = value
                else:
                    result[model.name] = np.nan
            print("Calculated compaction.")
            return result

    def calculate_subsidence_at_points(self):
        """For each point stored in each model, calulcate the subsidence.

        Returns
        -------
        result : dict
            A dictionary with a key for each model. Each value in the dictionary is
            an xarray.DataArray object with the subsidence in m with the shape
            (point, reservoirs, time)

        """
        if self.hasattr("models"):
            print("Calculating subsidence at points")
            result = {}
            for model in tqdm(self._models):
                value = model.calculate_subsidence_at_points(_print=False)
                result[model.name] = value
            print("Calulated subsidence at points")
            return result

    def calculate_subsidence_at_observations(self):
        """For each observation point in each model, calulcate the subsidence.

        Returns
        -------
        result : dict
            A dictionary with a key for each model. Each value in the dictionary is
            an xarray.DataArray object with the subsidence in m with the shape
            (observations, reservoirs, time)

        """
        if self.hasattr("models"):
            print("Calulating subsidence at observations")
            result = {}
            for model in tqdm(self._models):
                value = model.calculate_subsidence_at_observations(
                    _print=False
                )
                result[model.name] = value
            print("Calculated subsidence at observations")
            return result

    def calculate_subsidence(self):
        """For each model, calulcate the subsidence.

        Returns
        -------
        result : dict
            A dictionary with a key for each model. Each value in the dictionary is
            an xarray.DataArray object with the subsidence in m with the shape
            (y, x, reservoirs, time)

        """
        if self.hasattr("models"):
            print("Calculating subsidence")
            result = {}
            for model in tqdm(self._models):
                value = model.calculate_subsidence(_print=False)
                result[model.name] = value
            print("Calculated subsidence")
            return result

    def calculate_slope(self):
        """For each model, calulcate the slope of the subsidence bowl.

        Returns
        -------
        result : dict
            A dictionary with a key for each model. Each value in the dictionary is
            an xarray.DataArray object with the subsidence slope in m/m with the shape
            (y, x, reservoirs, time)

        """
        if self.hasattr("models"):
            print("Calculating slope")
            result = {}
            for model in tqdm(self._models):
                value = model.calculate_slope(_print=False)
                result[model.name] = value
            print("Calculated slope")
            return result

    def calculate_concavity(self):
        """For each model, calulcate the concavity of the subsidence bowl.

        Returns
        -------
        result : dict
            A dictionary with a key for each model. Each value in the dictionary is
            an xarray.DataArray object with the concavity of the subsidence bowl
            in m/² with the shape (y, x, reservoirs, time)

        """
        if self.hasattr("models"):
            print("Calculating concavity")
            result = {}
            for model in tqdm(self._models):
                value = model.calculate_concavity(_print=False)
                result[model.name] = value
            print("Calculated concavity")
            return result

    def calculate_subsidence_rate(self):
        """For each model, calulcate the rate subsidence.

        Returns
        -------
        result : dict
            A dictionary with a key for each model. Each value in the dictionary is
            an xarray.DataArray object with the subsidence rate in m/year with the shape
            (y, x, reservoirs, time)

        """
        if self.hasattr("models"):
            result = {}
            for model in self._models:
                value = model.calculate_subsidence_rate()
                result[model.name] = value
            return result

    def error(self, method="mse"):
        """Return the error between model results and subsidence observations for each model.

        Parameters
        ----------
        method : str, optional
            The method the devation is being calculated. The default is 'mae'.
            Available methods are:
                - Mean absolute error : 'mae'
                - Mean squared error : 'mse'

        Returns
        -------
        result : dict
            A dictionary with a key for each model. Each value in the dictionary is
            a float with the model deviation from observations.

        """
        if self.hasattr("models"):
            result = {}
            for model in self._models:
                value = model.error(method=method)
                result[model.name] = value
            return result

    # Get from results
    def get_timeseries(
        self, x, y, variable="subsidence", reservoir=None, model=None
    ):
        """Get the timeserie subsidence for all or a single specified reservoir and/or model.
        The location can be entered as grid index or coordinate. When the coordinate is not
        present in the grid, it will be interpolated.

        Parameters
        ----------
        x : int/float
            An integer or float representing the location of the grid node over
            which the subsidence timeseries will be returned. When an integer,
            the index of the grid node will be retrieved. When the variable is
            a float, it will be interpreted as a coordinate value.
        y : int/float
            An integer or float representing the location of the grid node over
            which the subsidence timeseries will be returned. When an integer,
            the index of the grid node will be retrieved. When the variable is
            a float, it will be interpreted as a coordinate value.
        variable : str
            A variable whith time as a dimension and currently in the Model.grid
            object.
        reservoir : int/str, optional
            Name or index of the reservoir of which the relevant data is being
            asked for. The default is None. With None, the total subsidence for all
            reservoirs will be returned.
        model : int, str, list
            Either an index of the model, the name of a model as a string, or a
            list with indeces or names. The default is None. With None, the
            timeseries for all models will be returned.


        Returns
        -------
        timeseries : dict
            A dictionary with a key for each model. Each value in the dictionary is
            an xarray.DataArray object with the subsidence values (m) for specified
            location and reservoir input over time.
        """
        if self.hasattr("models"):
            model = self.model_label_to_index(model)
            timeseries = {}
            for i, m in enumerate(self._models):
                if i in model:
                    timeseries[m.name] = m.get_timeseries(
                        x=x, y=y, variable=variable, reservoir=reservoir
                    )
            return timeseries

    def get_subsidence_spread(self, x, y, reservoir=None, model=None):
        """Get the minimum, mean and maximum subsidence as a timeseries
        at a specified location.

        Parameters
        ----------
        x : int/float
            An integer or float representing the location of the grid node over
            which the subsidence timeseries will be returned. When an integer,
            the index of the grid node will be retrieved. When the variable is
            a float, it will be interpreted as a coordinate value.
        y : int/float
            An integer or float representing the location of the grid node over
            which the subsidence timeseries will be returned. When an integer,
            the index of the grid node will be retrieved. When the variable is
            a float, it will be interpreted as a coordinate value.
        variable : str
            A variable whith time as a dimension and currently in the Model.grid
            object.
        reservoir : int/str, optional
            Name or index of the reservoir of which the relevant data is being
            asked for. The default is None. With None, the total subsidence for all
            reservoirs will be returned.
        model : int, str, list
            Either an index of the model, the name of a model as a string, or a
            list with indeces or names. The default is None. With None, the
            timeseries for all models will be returned.


        Returns
        -------
        serie_min : dict
            A dictionary with a key for each model. Each value in the dictionary is
            an xarray.DataArray object with the minimum subsidence values (m) for specified
            location and reservoir input over time.
        serie_mean : dict
            A dictionary with a key for each model. Each value in the dictionary is
            an xarray.DataArray object with the mean subsidence values (m) for specified
            location and reservoir input over time.
        serie_max : dict
            A dictionary with a key for each model. Each value in the dictionary is
            an xarray.DataArray object with the maximum subsidence values (m) for specified
            location and reservoir input over time.
        """
        if self.hasattr("models"):
            timeseries = self.get_timeseries(
                x, y, reservoir=reservoir, model=model
            )
            timesteps = self.unique_timesteps()
            timeseries = [
                t.interp(time=timesteps, method="linear")
                for t in timeseries.values()
            ]
            timeseries_xr = xr.DataArray(
                timeseries, coords={"model": self.name, "time": timesteps}
            )
            serie_max = timeseries_xr.min(dim="model")
            serie_min = timeseries_xr.max(dim="model")
            serie_mean = timeseries_xr.mean(dim="model")
            return serie_min, serie_mean, serie_max

    def get_max_subsidence_timeseries(self, reservoir=None, model=None):
        if self.hasattr("models"):
            model = self.model_label_to_index(model)
            timeseries = {}
            for i, m in enumerate(self._models):
                if i in model:
                    timeseries[m.name] = m.get_max_subsidence_timeseries(
                        reservoir=reservoir
                    )
            return timeseries

    def get_max_subsidence(self, time=None, reservoir=None):
        """Get the minimum, mean and maximum subsidence as a timeseries
        at a specified location.

        Parameters
        ----------
        reservoir : int/str, optional
            Name or index of the reservoir of which the relevant data is being
            asked for. The default is None. With None, the total subsidence for all
            reservoirs will be returned.
        time : int, str, optional
            The index or name of the timestep you want to know the maximum
            subsidence of. If it is a list, an Exception will occur. The default
            is -1, the final timestep.


        Returns
        -------
        max_subsidence : float
            The maximum subsidence.
        model_max : str
            Name of the model with the maximum subsidence.
        xmax, ymax : float
            The x- and y-coordinates of the location with the most subsidence.
        """
        if self.hasattr("models"):
            smax = []
            xs = []
            ys = []
            which_model = []
            for m in self._models:
                s, (x, y) = m.get_max_subsidence(
                    time=time, reservoir=reservoir
                )
                smax.append(s)
                xs.append(x)
                ys.append(y)
                which_model.append(m)
            imax = np.argmin(smax)
            max_subsidence = smax[imax]
            model_max = which_model[imax].name
            xmax, ymax = (xs[imax], ys[imax])
            return max_subsidence, model_max, (xmax, ymax)

    def get_min_subsidence(self, time=None, reservoir=None):
        if self.hasattr("models"):
            smin, xs, ys = [], [], []
            for m in self._models:
                s, (x, y) = m.get_min_subsidence(
                    time=time, reservoir=reservoir
                )
                smin.append(s)
                xs.append(x)
                ys.append(y)
            imin = np.argmax(smin)
            return smin[imin], (xs[imin], ys[imin])

    def report(self):
        if self.hasattr("models"):
            for model in self._models:
                model.report()

    def get_subsidence_overview(self):
        overview = pd.DataFrame()
        for i, Model in enumerate(self._models):
            model_df = Model.get_subsidence_overview()
            overview = overview.append(model_df.T)
        return overview
