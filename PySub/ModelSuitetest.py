import os
import sys

environment_location = os.path.split(sys.path[0])[0]
os.environ['PROJ_LIB'] = os.path.join(environment_location, 'Library\share\proj')
os.environ['GDAL_DATA'] = os.path.join(environment_location, 'Library\share')

from PySub import SubsidenceModelGas as _SubsidenceModelGas
from PySub import Points as _Points
from PySub import utils as _utils
from PySub import plot_utils as _plot_utils
from PySub import memory as _memory
from PySub import ProjectFolder as _ProjectFolder
import numpy as np
import pandas as pd
import xarray as xr
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
import datetime

class ModelSuite:
    """Object to contain subsidence modeling data and functionalities for multiple 
    models.
    
    This object creates multiple SubsidenceModel objects.
    
    Parameters need to be added to define the SubsidenceModel objects,
    the reservoirs, the timesteps and, optionally, points on which the 
    subsidence will be determined. Each with their own dimensionality.
    """
    def __init__(self, name, project_folder = None):
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
            raise Exception(f'variable name must be a string, is: {type(name)}')
        
        self.set_project_folder(project_folder, self.suite_name)
        self._models = None
        self._bounds = None
        self.number_of_models = None
        self._variable_that_set_number_of_models = None
        self._bounds = None
        self._contour_levels = None
        
        # Defaults
        self._contourf_defaults = {'cmap': 'winter_r', 
                                   'alpha': 0.5, 
                                   'extend': 'both'}
        self._contour_defaults = {'cmap': 'winter_r'}
        self._clabel_defaults = {'colors': 'k', 
                                 'inline': True, 
                                 'fontsize': 10}
        self._colorbar_defaults = {'cmap': 'winter_r', 
                                   'spacing': 'proportional'}
        self._plot_defaults = {'cmap' : 'winter_r'}
        self._shape_defaults = {'facecolor': 'green', 
                                'edgecolor': 'k', 
                                'alpha': 0.3}
        self._annotation_defaults = {}
        self._scatter_defaults = {}
        self._errorbar_defaults = {'cmap': 'winter_r', 
                                   'fmt': 'o', 
                                   'linestyle': 'none', 
                                   'markersize': 2, 
                                   'linewidth': 1}
        self._fill_between_defaults = {'facecolor': 'grey', 
                                       'alpha' : 0.5}
        self._raster_defaults = {'cmap': 'winter_r', 
                                       'alpha' : 0.8}
    
    def __repr__(self):
        return repr(self._models)
    
    def __getitem__(self, item):
         if isinstance(item, int):
             return self._models[item]
         if isinstance(item, str):
             return self.models[item]
     
    def __len__(self):
         if self.hasattr('number_of_models'):
             return self.number_of_models
         else: 
             return 0
    
    def hasattr(self, var): # redefenition for convenient syntax
        _var = f"_{var}"   
        if _var in self.__dict__:
            if _utils.is_iterable(type(self.__dict__[_var])):
                if len(self.__dict__[_var]) > 0:
                    return True
                else:
                    return False
            elif self.__dict__[_var] is not None:
                return True
            
            else: return False
        elif var in self.__dict__:
            if _utils.is_iterable(type(self.__dict__[var])):
                if len(self.__dict__[var]) > 0:
                    return True
                else:
                    return False
            elif self.__dict__[var] is not None:
                return True
            else: return False 
        else:
            try:
                attr = getattr(self, var)
                if attr is not None:
                    return True
                else:
                    return False
            except:
                return False
    
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
        if self.hasattr('models'):
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
        if self.hasattr('models'):
            return [m.name for m in self._models]
    
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
        if self.hasattr('models'):
            dict_var = {}
            for model in self._models:
                if model.hasattr(var):
                    dict_var[model.name] = getattr(model, var)
                else:
                    raise Exception(f'Model {model.name} of type {type(model)} has no attribute: {var}.')
            return dict_var
    
    # Set parameters
    def set_project_folder(self, folder = None, name = None):
        if name is None:
            if self.hasattr('name'):
                name = self.suite_name
            else:
                name = 'unnamed_subsidence_suite'
        if folder is not None:
            project_folder = os.path.join(folder, name)
        else:
            project_folder = None

        self.project_folder = _ProjectFolder.ProjectFolder(project_folder)
        if self.hasattr('models'):
            for m in self._models:
                model_project_folder = os.path.join(project_folder, 'input', m.name)
                m.set_project_folder(model_project_folder)
        
        
    def set_property(self, property_name):
        def property_func(self):
            return self.dict_of_vars(property_name)
        setattr(self, property_func, property)
        
    def set_method(self, method_name):
        def method_func(self, *args, **kwargs):
            result = {}
            for m in self._models:
                result[m.name] = getattr(m, method_name)(*args, **kwargs)  
            return result
        setattr(self, method_func)
    
    def set_attributes(self, model):
        model_attributes = dir(model)
        non_dunder = [attr for attr in model_attributes if not attr.startswith('__')]
        properties = [attr for attr in non_dunder if isinstance(model[attr], property)]
        method = [attr for attr in non_dunder if attr not in properties]
        
        for p in properties:
            if p not in dir(self):
                self.set_property(p)
        for m in method:
            if m not in dir(self):
                self.set_method(m)
        
    
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
                    model.name = 'Model ' + str(i + 1)
                    print(f'Warning: Model added without name, name set as: {model.name}.')
                self.add_model(model)
            else:
                raise Exception('Invalid model type to add to ModelSuite. Add SubsidenceModel object types only.')
        self.set_bounds(bounds = None, all_same_bounds = False)