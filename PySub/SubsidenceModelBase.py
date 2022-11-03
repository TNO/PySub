"""Module storing the SubsidenceModel class all other SubsidenceModel classes inherit from.
"""
import os
import sys
import copy
from pathlib import Path
for candidate in sys.path:
    if 'envs' in candidate:
        p = Path(candidate)
        environment_location = os.path.join(*p.parts[:p.parts.index('envs') + 2])
        break
    
os.environ['PROJ_LIB'] = os.path.join(environment_location, 'Library\share\proj')
os.environ['GDAL_DATA'] = os.path.join(environment_location, 'Library\share')
# External imports
import xarray as xr
import numpy as np
import pandas as pd
# internal imports
from PySub.grid_utils import generate_grid_from_bounds
from PySub import SubsidenceKernel as _SubsidenceKernel
from PySub import HorizontalDisplacementKernel as _HKernel
from PySub import SubsidencePointKernel as _SubsidencePointKernel
from PySub import plot_utils as _plot_utils
from PySub import shape_utils as _shape_utils
from PySub import utils as _utils
from PySub import Points as _Points
from PySub import memory as _memory
from PySub import Geometries as _Geometries
from PySub import ProjectFolder as _ProjectFolder

import warnings
from warnings import warn
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = UserWarning)

EPSILON = 1e-10

class SubsidenceModel:
    """Object to contain subsidence modeling data and functionalities.
    
    This object creates an xarray grid (with x and y coördinates) for each 
    reservoir and timestep. Parameters need to be added to define the grid,
    the reservoirs, the timesteps and, optionally, points on which the 
    subsidence will be determined. Each with their own dimensionality.
    
    This class is used to inherit common functionality between different
    types of SubsidenceModels (like SubsidenceModelGas.SubsidenceModel and
    SubsidenceModelCavern.SubsidenceModel).
    
    """
    def __init__(self, name, project_folder = None):
        """Inititalize the SubsidenceModel object and construct an empty grid.
        

        Parameters
        ----------
        name : str
            name of the model.
        project_folder : str, optional
            Path to valid folder where all model data is stored. The default is 
            None.

        Attributes
        ----------
        # Grid attributes, where only 1 entry is needed
        grid : xarray.DataArray
            The grid with 6 dimensions:
                - y, float/int : an ny number of entries with the y-values of the 
                  grid nodes in crs.
                - x, float/int : an nx number of entries with the x-values of the 
                  grid nodes in crs.
                - reservoir, str : the names of each reservoir
                - time, str : the timestamp of the timestep.
                - points, str : the names of the points on which the subsidence 
                  is determined.
                - observations, str : the names of the points on which the subsidence 
                  is observed and will be determined.
        name : str
            The name of the model.
        built : bool
            True when the grid has been succesfully built, False if not.        
        nx : int
            number of grid cells along the x-axis.
        ny : int
            number of grid cells along the x-axis.
        number_of_reservoirs : int
        number_of_steps : int
            Number of timesteps
        number_of_points : int
        number_of_observations : int
        
        # Additional fuctionality with locations
        points : PointCollection
            PointCollection object storing names, x and y data of points.
        observation_points : ObservationCollection
            ObservationCollection object storing the names, location, observations and
            errors of observations. Some functionality regarding observation analysis.
        """
        if isinstance(name, str):
            self.name = name
        else:
            raise Exception(f'Variable name is not a string: {name}')
        
        self.set_project_folder(project_folder, name)
        
        self.built = False # past perfect of build
        self.grid = xr.DataArray()
        self.nx = None
        self.ny = None
        self.number_of_reservoirs = None
        self.number_of_steps = None
        self.number_of_points = None
        self.number_of_observation_points = None
        self._variable_that_set_number_of_reservoirs = None
        self._variable_that_set_number_of_steps = None
        self._variable_that_set_number_of_points = None
        self._projection = None
        
        self._contour_levels = None
        
        self._dx = None
        self._dy = None
        self._influence_radius = None
   
        self._subsidence_model = None
        self._subsidence_model_type = None
        self._reservoirs = None 
        self._depths = None 
        self._depth_to_basements = None 
    
        self._timesteps = None 
        self._bounds = None
        self._point_kernels = None
        self._kernels = None
        self._observation_points = None
                
        # Defaults
        self.CMAP = 'bwr'
        self._contourf_defaults = {'cmap': self.CMAP, 
                                   'alpha': 0.5, 
                                   'extend': 'both'}
        self._contour_defaults = {'cmap': self.CMAP}
        self._clabel_defaults = {'colors': 'k', 
                                 'inline': True, 
                                 'fontsize': 10}
        self._colorbar_defaults = {'cmap': self.CMAP, 
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
        """The representation of the object as a string.

        Returns
        -------
        str
            Overview of completeness of model and available methods.

        """
        return str(self)
    
    @property
    def calc_vars(self):
        return ['subsidence', 'volume', 'slope', 'concavity', 'subsidence_rate']
    
    @property
    def vars_to_calculate(self):
        return ['subsidence_model_type', 'depth_to_basements', 'depths', 'knothe_angles']
         
    @property
    def vars_to_build(self):
        return ['reservoirs', 'timesteps', 'points', 'observation_points',
                'dx', 'dy', 'influence_radius', 'shapes', 'bounds']
       
    def __str__(self):
        """The representation of the object as a string.

        Returns
        -------
        str
            Overview of completeness of model and available methods.

        """
        columns = ['Is set:', 'Set method:']
        
        build_df = pd.DataFrame(columns = columns)
        build_df.loc['name'] = self.name, 'Model.name ='
        for v in self.vars_to_build:
            setter =  f'set_{v}'
            build_df.loc[v] = self.hasattr(v), setter
        build_df.loc['number_of_reservoirs'] = self.hasattr('number_of_reservoirs'), 'set_reservoirs'
        build_df.loc['number_of_steps'] = self.hasattr('number_of_steps'), 'set_timesteps'
        
        calc_df = pd.DataFrame(columns = columns)
        calc_df.loc['grid'] = self.built, 'build_grid'
        calc_df.loc['nx'] = self.built, 'build_grid'
        calc_df.loc['ny'] = self.built, 'build_grid'
        
        
        for v in self.vars_to_calculate:
            setter =  f'set_{v}'
            calc_df.loc[v] = self.hasattr(v), setter
        
        calced_df = pd.DataFrame(columns = columns)
        for v in self.calc_vars:
            setter =  f'calculate_{v}'
            calced_df.loc[v] = self.hasattr(v), setter
        
        point_df = pd.DataFrame(columns = columns)
        point_df.loc['points'] = self.hasattr('points'), 'set_points'
        point_df.loc['number_of_points'] = self.hasattr('number_of_points'), 'set_points'
        point_df.loc['point_subsidence'] = self.hasattr('point_subsidence'), 'calculate_subsidence_at_points'
        
        observation_df = pd.DataFrame(columns = columns)
        observation_df.loc['observation_points'] = self.hasattr('observation_points'), 'set_obersvation_points'
        observation_df.loc['number_of_observation_points'] = self.hasattr('number_of_observation_points'), 'set_obersvation_points'
        observation_df.loc['observation_subsidence'] = self.hasattr('observation_subsidence'), 'calculate_subsidence_at_observations'
        
        build_df.index.name = 'Property:'
        calc_df.index.name = 'Property:'
        calced_df.index.name = 'Property:'
        point_df.index.name = 'Property:'
        observation_df.index.name = 'Property:'
        
        build = build_df.to_string(col_space = [15, 37])
        calc = calc_df.to_string(col_space = [12, 37])
        calced = calced_df.to_string(col_space = [20, 37])
        point = point_df.to_string(col_space = [19, 37])
        observation = observation_df.to_string(col_space = [0, 37])
        
        statement = ('Model variables', '\n',
                     ' ' * 18, '### Required to build ***', '\n',
                     build, '\n',
                     '\n',
                     ' ' * 16, '### Required to calculate ###', '\n',
                     calc, '\n',
                     '\n',
                     ' ' * 17, '### After calculation ###', '\n',
                     calced, '\n',
                     '\n',
                     ' ' * 24, '### Point ###', '\n',
                     point, '\n',
                     '\n',
                     ' ' * 21, '### Observations ###', '\n',
                     observation, '\n')
        statement = ' '.join(statement)
        return statement
    
    def __getitem__(self, item):
        try: return getattr(self.grid, item)
        except: 
            try: return getattr(self, item)
            except: raise AttributeError(f"""'{self.__class__.__name__}' has no attribute '{item}'""" )
    
    def _fetch(self, item):
        if hasattr(self.grid, item):
            return getattr(self.grid, item)
        else:
            return getattr(self, f'_{item}')
    
    # Properties
    @property
    def contourf_defaults(self):
        """Property: Represents default settings for the plotting of filled 
        contours. Can be adjusted with SubsidenceModel.set_contourf_defaults()
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
        contours. Can be adjusted with SubsidenceModel.set_contour_defaults()
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
        labels. Can be adjusted with SubsidenceModel.set_clabel_defaults() 
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
        Can be adjusted with SubsidenceModel.set_colorbar_defaults() 
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
        Can be adjusted with SubsidenceModel.set_plot_defaults() 
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
        Can be adjusted with SubsidenceModel.set_shape_defaults() 
        function.
        
        Returns:
        -------
        dict: 
            keyword arguments for the matplotlib.patches.polygon function.
        """
        return self._shape_defaults 
    
    @property
    def annotation_defaults(self):
        """Property: Represents default settings for the plotting of labels 
        in a graph. Can be adjusted with SubsidenceModel.set_annotation_defaults() 
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
        in graphs. Can be adjusted with SubsidenceModel.set_scatter_defaults() 
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
        Can be adjusted with SubsidenceModel.set_errorbar_defaults() function.
        
        Returns:
        -------
        dict: 
            keyword arguments for the matplotlib.pyplot.errorbar function.
        """
        return self._errorbar_defaults
    
    @property
    def fill_between_defaults(self):
        """Property: Represents default settings for the plotting of filled areas. 
        Can be adjusted with SubsidenceModel.set_fill_between_defaults() function.
        
        Returns:
        -------
        dict: 
            keyword arguments for the matplotlib.pyplot.fill_between function.
        """
        return self._fill_between_defaults 
    
    @property
    def raster_defaults(self):
        """Property: Represents default settings for the plotting of rasters. 
        Can be adjusted with SubsidenceModel.set_raster_defaults() function.
        
        Returns:
        -------
        dict: 
            keyword arguments for the matplotlib.pyplot.imshow function.
        """
        return self._raster_defaults 
    
    @property
    def contour_levels(self):
        """Property : Contour levels of the subsidence plots. Set with 
        SubsidenceModel.set_contour_levels().

        Returns
        -------
        list, floats:
            The values of the contour levels in a list.

        """
        return self._contour_levels
    
    @property
    def dx(self):
        """Property: Represents the size of the grid cells aling the x-axis. 
        Can be adjusted with SubsidenceModel.set_dx() function.
        
        Returns:
        -------
        float/int
            Distance between grid nodes along the x-axis in m.
        """
        return self._fetch('dx')
    
    @property
    def dy(self):
        """Property: Represents the size of the grid cells aling the y-axis. 
        Can be adjusted with SubsidenceModel.set_dy() function.
        
        Returns:
        -------
        float/int
            Distance between grid nodes along the y-axis in m.
        """
        return self._fetch('dy')
    
    @property
    def influence_radius(self):
        """Property: The distance in m after which the subsidence caused by
        compaction in a grid cell is deemed insignificant and is set to 0.
        Can be adjusted with SubsidenceModel.set_influence_radius() 
        function.
        
        Returns:
        -------
        float/int
            Distance from compacted cell after which subsidence is 
            set to 0.
        """
        return self._fetch('influence_radius')
    
    @property
    def reservoirs(self):
        """Property: The names of the reservoirs in model.
        Can be adjusted with SubsidenceModel.set_reservoirs() function.
        
        Returns:
        -------
        list, str 
            The name of each of the reservoirs.
            The returned list has the same length as the number of reservoirs in 
            the model.
        """
        return self._reservoirs
    
    @property
    def shapes(self):
        """Property: The spatial distribution of the reservoirs as Geometry objects.
        Can be adjusted with SubsidenceModel.set_shapes() function.
            
        Returns:
        -------    
        list, str
            The Geometry objects for each reservoir.
        
            The returned list has the same length as the number of reservoirs in 
            the model.
        """
        return self._shapes
    
    @property
    def depths(self):
        """Property: The depth of the top of the reservoir in m.
        Can be adjusted with SubsidenceModel.set_depths() function.
            
        Returns:
        -------    
        xarray DataSet if grid is built, else list, float/int
            Depths to the top of each reservoirs in m.
            The returned list has the same length as the number of reservoirs in 
            the model.
        """
        return self._fetch('depths')
    
    @property
    def depth_to_basements(self):
        """Property: The depth of the top of the reservoir in m.
        Can be adjusted with SubsidenceModel.set_depths() function.
            
        Returns:
        -------  
        xarray DataSet if grid is built, else list, float/int
            Depth to the rigid basement for the van Opstal 
            nucleus of strain method in m. Values for each reservoir. 
            The returned list has the same length as the number of reservoirs in 
            the model.
        """
        return self._fetch('depth_to_basements')
    
    @property
    def poissons_ratios(self):
        """Property: The poisson's ratio of each reservoir.
        Can be adjusted with SubsidenceModel.set_depths() function.
            
        Returns:
        -------  
        xarray DataSet if grid is built, else list, float/int
            Poisson's ratio for the van Opstal 
            nucleus of strain method in m. Values for each reservoir. 
            The returned list has the same length as the number of reservoirs in 
            the model.
        """
        return self._fetch('poissons_ratios')
    
    @property
    def bounds(self):
        """Property: The upper and lower bounds of the model in x- and y-coordinates.
        
        Returns:
        -------  
        list, int/float
            List with 4 values representing the extend/bounds
            of the model grid:
            [0] lower x
            [1] lower y
            [2] upper x
            [3] upper y
        """  
        return self._bounds  
    
    @property
    def shape(self):
        """Property: Dimensions of the grid the calculations are taken over.
        
        Returns:
        ------- 
        list, int 
            Dimensionality of the model with a length of 4 (the number of 
            dimensions, minus the dimensions for the number of points and observations!):
            [0] SubsidenceModel.ny, the number of cells over the y-axis
            [1] SubsidenceModel.nx, the number of cells over the x-axis
            [2] SubsidenceModel.number_of_reservoirs
            [3] SubsidenceModel.number_of_steps
            Returns None for each value that has not been set. Nx and ny are set
            with the set_bounds() function. Number_of_reservoirs and number_of_steps 
            are each set after parameters are set that are dependent on the number 
            of those dimensions
        """
        return(self.ny, self.nx, self.number_of_reservoirs, self.number_of_steps)
    
    @property
    def x(self):
        """Property: X-coordinates of the grid nodes. Determined by the bounds of
        the grid and the grid cell size. Set after the grid is built.
        
        Returns:
        ------- 
        xarray Dataset, float
            The coordinates of the grid nodes along the x-axis.
            The returned list has the same length as nx, the number of grid nodes
            in the model.
        """
        return self.grid.x.values
    
    @property
    def y(self):
        """Property: Y-coordinates of the grid nodes. Determined by the bounds of
        the grid and the grid cell size. Set after the grid is built.
        
        Returns:
        ------- 
        xarray Dataset, float
            The coordinates of the grid nodes along the y-axis.
            The returned list has the same length as ny, the number of grid nodes
            in the model.
        """
        return self.grid.y.values
    
    @property
    def reservoir_mask(self):
        """Property: Returns an xarray DataSet with for each reservoir its extend 
        in the grid. If a value in the grid is 1, it is inside the reservoir. If 
        it is 0, the cell is outside the reservoir. Is set with SubsidenceModel.mask_reservoirs().

        Returns
        -------
        xarray Dataset, int
            DataSet of the reservoir masks. The xarray has the shape (ny, nx, number_of_reservoirs).

        """
        return self.grid.reservoir_mask
    
    @property
    def compaction(self):
        """Property: Compaction in cubic meters as determined by the 
        calculate_compaction() or set_compaction(grid) function.
        
        Returns:
        -------
        xarray.DataArray, float
            Compaction in cubic meters for each reservoir per 
            timestep over the entirety of the model grid. Shape =  
            (y, x, reservoir, time).
        """
        try: return self.grid.compaction
        except: raise Exception('Compaction has not been set/calculated yet.')  
    
    @property
    def subsidence(self):
        """Property: subsidence over time in m for each Model grid node as 
        determined by the calculate_subsidence() method. Seperate for each 
        reservoir. 
        
        Returns:
        -------
        xarray.DataArray, float
            Subsidence in m for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).
        """
        try: return self.grid.subsidence
        except: raise Exception('Subsidence has not been calculated yet.')
    
    @property
    def slope(self):
        """Property : gradient magnitude of the subsidence bowl in m/m for each grid node,
        reservoir and timestep as determined by the calculate_slope() method.

        Returns
        -------
        xarray.DataArray, float
            Gradient magnitude of the subsidence bowl in m/m for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).

        """
        try: return self.grid.slope
        except: raise Exception('Calculate susbsidence before calculating subsidence gradient.')
    
    @property
    def subsidence_rate(self):
        """Property : rate of the subsidence bowl in m/j for each grid node,
        reservoir and timestep as determined by the calculate_subsidence_rate() method.

        Returns
        -------
        xarray.DataArray, float
            Rate of the subsidence bowl in m/j for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).

        """
        try: return self.grid.subsidence_rate
        except: raise Exception('Volume has not been calculated yet.')
    
    @property
    def concavity(self):
        """Property : concavity magnitude of the subsidence bowl in m/m² for each grid node,
        reservoir and timestep as determined by the calculate_concavity() method.

        Returns
        -------
        xarray.DataArray, float
            Concavity magnitude of the subsidence bowl in m/m² for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).

        """
        try: return self.grid.concavity_x
        except: raise Exception('Concavity has not been calculated yet.')    
    
    @property
    def concavity_x(self):
        """Property : concavity of the subsidence bowl over x-axis, in m/m² for each grid node,
        reservoir and timestep as determined by the calculate_concavity() method.

        Returns
        -------
        xarray.DataArray, float
            Concavity of the subsidence bowl in m/m² for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).

        """
        try: return self.grid.concavity_x
        except: raise Exception('Concavity has not been calculated yet.')
    
    @property
    def concavity_y(self):
        """Property : concavity of the subsidence bowl over y-axis, in m/m² for each grid node,
        reservoir and timestep as determined by the calculate_concavity() method.

        Returns
        -------
        xarray.DataArray, float
            Concavity of the subsidence bowl in m/m² for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).

        """
        try: return self.grid.concavity_y
        except: raise Exception('Concavity has not been calculated yet.')
    
    @property
    def volume(self):
        """Property : volume of the subsidence bowl in m³ for each grid node,
        reservoir and timestep as determined by the calculate_volume() method.

        Returns
        -------
        xarray.DataArray, float
            Volume of the subsidence bowl in m³ for each reservoir per timestep
            over the entirety of the grid. Shape = (y, x, reservoir, time).

        """
        try: return self.grid.volume
        except: raise Exception('Volume has not been calculated yet.')
    
    @property
    def point_subsidence(self):
        """Property: subsidence over time in m on each Model point as 
        determined by the calculate_subsidence_at_points() method. Seperate for 
        each reservoir
        
        Returns:
        -------
        xarray.DataArray, float
            Subsidence in m for each reservoir per timestep
            at each point. Shape = (points, reservoir, time).
        """
        try: return self.grid.point_subsidence
        except: raise Exception('Point subsidence has not been calculated yet.')
        
    @property
    def observation_subsidence(self):
        """Property: subsidence over time in m on each observation added to the model
        as determined by the calculate_subsidence_at_observations() method. Seperate 
        for each reservoir
        
        Returns:
        -------
        xarray.DataArray, float
            Subsidence in m for each reservoir per timestep
            at each observation point. Shape = (observations, reservoir, time).
        """
        try: return self.grid.observation_subsidence
        except: raise Exception('Observation subsidence has not been calculated yet.')
    
    @property
    def total_subsidence(self):
        """Property: subsidence in m on each Model cell node as determined by 
        the calculate_subsidence() function. Summed over all reservoirs.
        
        Returns:
        -------
        xarray.DataArray, float
            Total subsidence in m per timestep
            over the entirety of the grid. Shape = (y, x, time).
        """
        try: return self.grid.subsidence.sum(dim = 'reservoir')
        except: raise Exception('Calculate susbsidence before calculating subsidence total.')
        
    @property
    def points(self):
        """Property : returns the set points as a PointCollection object.

        Returns
        -------
        PointCollection.
        """
        return self._points
    
    @property
    def observation_points(self):
        """Property : returns the set onbesrvations points as a ObservationCollection
        object.

        Returns
        -------
        ObservationCollection.

        """
        return self._observation_points
    
    @property
    def subsidence_model_type(self):
        """Property: The method used for calculating the subsidence based on 
        compaction. Can be adjusted with SubsidenceModel.set_subsidence_model_parameters() 
        function.
        
        Returns:
        -------
        str 
            Method of subsidence of the model. Currently available: # TODO: keep updated with added methods
            - Nucleus of strain, Van Opstal 1974
            - Knothe, Stroka et al. 2011
        """    
        return self._subsidence_model_type
    
    @property
    def timesteps(self):
        """Property: The labels for each timestep in np.datetime64 objects.
        The notation is Year - Month - Day.
        Can be adjusted with SubsidenceModel.set_timesteps() function.
            
        Returns:
        -------  
        list, datetime64[ns] : 
            The labels for each timestep.
            The returned list has the same length as the number of steps in 
            the model.
        """
        return self._timesteps
    
    @property
    def knothe_angles(self):
        """Property: Returns the Knothe angles for each reservoir as a list or xarray 
        Dataset. 

        Returns
        -------
        xarray DataSet if grid is built, else list, float/int
            The knothe_angles for each reservoirs. 
            Can be a 1D or 3D list/DataSet. When 1D, the length is equal to 
            the number of reservoirs in the model and the values of this parameter are 
            distributed uniformly over each reservoir. When 3D the shape of the returned 
            variable is (ny, nx, number_of_reservoirs).
        """
        return self._fetch('knothe_angles')
    
    # Exception handling
    def _exception(self, exception): # XXX Houdt logboek van fouten bij
        raise(Exception(f'{exception}'))
        
    def _nonetype_exception(self, name):
        raise Exception(f'Variable is {name} is None.')
        
    def _dim_exception(self, name, dims_equal, dims, set_var):
        raise Exception(f'Variable {name} should have the same number of entries as {dims_equal}, which are currently set to {dims} by variable: {set_var}.')
    
    # Checks
    def _check_projection(self, projection, file_name):
        if projection != self._projection:
            raise Exception(f'Two different projections have been encountered: {projection} and {self.projection}.')
    def _check_none(self, name, var):
        if var is None:
            self._nonetype_exception(f'{name}')
            
    def _check_dim1D(self, name, var, dims_equal = 'reservoirs'):
        if dims_equal == 'reservoirs':
            if self.number_of_reservoirs is None:
                self.number_of_reservoirs = len(var)
                self._variable_that_set_number_of_reservoirs = name
            elif len(var) != self.number_of_reservoirs:
                self._dim_exception(name, dims_equal, self.number_of_reservoirs, self._variable_that_set_number_of_reservoirs)
        elif dims_equal == 'timesteps':
            if self.number_of_steps is None:
                self.number_of_steps = len(var)
                self._variable_that_set_number_of_steps = name
            elif len(var) != self.number_of_steps:
                self._dim_exception(name, dims_equal, self.number_of_steps, self._variable_that_set_number_of_steps)
        elif dims_equal == 'point_ids':
            if self.number_of_points is None:
                self.number_of_points = len(var)
                self._variable_that_set_number_of_points = name
            elif len(var) != self.number_of_points:
                self._dim_exception(name, dims_equal, self.number_of_points, self._variable_that_set_number_of_points)
        elif dims_equal == 'x':
            if len(var) != self.nx:
                 self._dim_exception(name, dims_equal, self.nx, self._variable_that_set_number_of_points)
        elif dims_equal == 'y':
            if len(var) != self.ny:
                 self._dim_exception(name, dims_equal, self.ny, self._variable_that_set_number_of_points)
        else:
            raise Exception(f'{dims_equal} not supported')
    
    def _check_dimND(self, name, var, dims_equal = ('reservoirs', 'timesteps')):
        shape = var.shape
        if len(shape) != len(dims_equal):
            raise Exception(f'Variable {name} is not {len(dims_equal)}D.')
        for axis in range(len(shape)):
            sum_axis = [i for i in range(len(shape)) if i != axis]
            self._check_dim1D(f'{name}, axis {axis}', var.sum(axis = tuple(sum_axis)), dims_equal[axis])
    
    def _check_for_model_attributes(self, var, action = 'perform this action'):
        if not isinstance(var, str): 
            raise Exception(f'When checking attribute existance, attribute name {var} needs to be a string')
        if not self.hasattr(var):
            raise Exception(f'SubsidenceModel object has no attribute {var}, this is required to {action} of the model. Use the method SubsidenceModel.set_{var} to proceed.')
    
    def _check_for_grid_attributes(self, var, action = 'perform this action'):
        if not isinstance(var, str): 
            raise Exception(f'When checking attribute existance, attribute name {var} needs to be a string')
        if not hasattr(self.grid, var):
            raise Exception(f'SubsidenceModel object has no attribute {var}, this is required to {action} of the model. Use the method SubsidenceModel.set_{var} to proceed.')
        
    
    def _check_build_paramaters(self):
        action = 'build the model'
        
        self._check_for_model_attributes('dx', action = action)
        self._check_for_model_attributes('timesteps', action = action)
        self._check_for_model_attributes('reservoirs', action = action)
        self._check_for_model_attributes('influence_radius', action = action)
        self._check_for_model_attributes('bounds', action = action)

    def hasattr(self, var): # redefenition for convenient syntax and lack of editable dunder method
        """Check if the SubsidenceModel object has a certain attribute.

        Redefinition of python native hasattr function.

        Parameters
        ----------
        var : str
            Name of the potential attribute.

        Returns
        -------
        bool
            True if the SubsidenceModel object does have that attribute or if that 
            variable has been set, False if neither.

        """    
        _var = f"_{var}"   
        grid_var = False
        if hasattr(self, 'grid'):
            grid_var = hasattr(self.grid, var)
        if not grid_var:
            
            if _var in self.__dict__:
                if _utils.is_iterable(self.__dict__[_var]):
                    if len(self.__dict__[_var]) > 0:
                        return True
                    else:
                        return False
                elif self.__dict__[_var] is not None:
                    return True
                
                else: return False
            elif var in self.__dict__:
                if _utils.is_iterable(self.__dict__[var]):
                    if len(self.__dict__[var]) > 0:
                        return True
                    else:
                        return False
                elif self.__dict__[var] is not None:
                    return True
                else: return False 
            else:
                try:
                    if self[var] is None:
                        return False
                    return True
                except:
                    return False
        else:
            return grid_var
    
    def _check_crs(self, crs):
        if not self.hasattr('crs'):
            self._crs = crs
        else:
            if crs is None:
                return
            if crs != self._crs:
                raise Exception(f'Sources use different coordinate reference systems: {crs} and {self._crs}')
    
    def check_subsidence_model(self, name_model):
        """Raise an exception if the entry given is not in the correct format to 
        select a specified subsidence model with, or if that entry is not in the list of
        available subsidence models.

        Parameters
        ----------
        name_model : str
            Name of a subsidence model.

        Raises
        ------
        Exception
            When the entry is not a valid label of a subsidence model.
        """
        if isinstance(name_model, str):
            if name_model.lower() not in ['knothe', 'nucleus of strain']:
                raise Exception(f"{name_model} not recognized as valid subsidence model. Choose from: 'knothe', 'nucleus of strain''")
        else:
            raise Exception("set the subsidence model using a string with either 'knothe', 'nucleus of strain'")

    def check_for_grid(self):
        """Raise an exception if the model grid has not been built yet. To return 
        a boolean without exception of the status of the grid, use the property:
            SubsidenceModel.built
            
        Raises
        ------
        Exception
            When the grid has not been built.
        """
        if not self.built:
            raise Exception(f'No grid has been build for this model: {self.name}.')
    
    def check_influence_radius(self):
        """Asks for feedback (y/n) when the influence radius is not high enough
        to cover half of the largest reservoir.
        """
        if not hasattr(self.grid, 'reservoir_mask'):
            raise Exception('Reservoir mask layer not set, run mask_from_shapefiles before checking influence radius')
        for reservoir in range(self.number_of_reservoirs):
            lx,ly,hx,hy = self.get_id_bounds_reservoir(reservoir_layer = reservoir)
            if ((hx - lx) // 2 > self._influence_radius // self._dx or
                (hy - ly) // 2 > self._influence_radius // self._dy):
                if (hx - lx) // 2 >= (hy - ly) // 2:
                    max_distance = np.round(((hx - lx) // 2) * self._dx)
                else:
                    max_distance = np.round(((hy - ly) // 2) * self._dy)
                input_is_valid = False
                while not input_is_valid:
                    check = input(f'The influence radius {self._influence_radius} needs to be higher than half of the largest reservoir: {max_distance}. Do you want to continue? y/n: ')
                    if check.lower() == 'y':
                        input_is_valid = True
                        return
                    elif check.lower() == 'n':
                        raise Exception(f'The influence radius {self._influence_radius} needs to be higher than half of the largest reservoir: {max_distance}.')
    
    # add
    def add_kernels(self):
        """Creates an SubsidenceKernel.InfluenceKernel object for each 
        reservoir which represents the subsidence caused by compaction in a
        grid cell. The kernel has the size of twice the influence radius by 
        twice the influence radius.
        
        Sets
        -------
        "kernels" if the kernels have not been set yet, else it adds to the list of
        kernels.

        """
        self._kernels = [_SubsidenceKernel.InfluenceKernel(self._influence_radius, self._dx) for i in range(self.number_of_reservoirs)]
        if self._depth_to_basements is None:
            depth_to_basements = [False] * self.number_of_reservoirs
        else:
            depth_to_basements = self._depth_to_basements
        for i, kernel in enumerate(self._kernels):
            if self._subsidence_model_type.lower().startswith('nucleus'):
                kernel.nucleus(self._depths[i], depth_to_basements[i])
            elif self._subsidence_model_type == 'knothe':
                kernel.knothe(self._depths[i], self.knothe_angles[i])
       
    
    def add_point_kernels(self, points = None):
        """Creates an SubsidencePointKernel.InfluencePoint object for each 
        reservoir which represents the subsidence caused by compaction on a
        specific location. The kernel has the size of twice the influence radius by 
        twice the influence radius.
        
        Sets
        -------
        "point_kernels" if the kernels have not been set yet, else it adds to the list of
        point_kernels.

        """
        object_points = False
        if points is None:
            points = self.points.coordinates
            object_points = True
        number_of_points = len(points)
        point_kernels = []
        for point in points:
            point_kernels_per_point = [_SubsidencePointKernel.InfluencePoint(self.grid, point, self._influence_radius) for r in range(self.number_of_reservoirs)]
            for reservoir in range(self.number_of_reservoirs):
                if self._subsidence_model_type.lower().startswith('nucleus'):
                    point_kernels_per_point[reservoir].nucleus(self._depths[reservoir], self._depth_to_basements[reservoir])
                elif self._subsidence_model_type == 'knothe':
                    point_kernels_per_point[reservoir].knothe(self._depths[reservoir], self.knothe_angles[reservoir])
            point_kernels.append(point_kernels_per_point)
        if object_points:
            self._point_kernels = point_kernels
        return point_kernels, number_of_points
    
    # assign
    def assign_new_coordinates(self, name, coordinates):
        """Assign a new coordinate to the SubsidenceModel.grid xarray.Dataset 
        object.

        Parameters
        ----------
        name : str
            Name of the dimension.
        coordinates : list/np.ndarray
            The values of these coordinates.
        """
        if hasattr(self.grid, name):
            self.grid = self.grid.assign_coords({name: coordinates})
        else:
            self.grid[name] = coordinates

    def assign_attribute(self, name, var):
        """Assign variable to the SubsidenceModel.grid xarray.Dataset object.
        An attribute is not dependent on the dimensions of the grid.

        Parameters
        ----------
        name : str
            name of the attribute.
        var : float/int/str/list/object
            Attribute to be set.

        """
        self.grid.attrs[str(name)] = var
        
    def assign_data_variable(self, name, dimensions, var):
        """Assign variable to the SubsidenceModel.grid xarray.Dataset object.
        A variable is dependent on one or more of the dimensions of the 
        grid.

        Parameters
        ----------
        name : str
            name of the attribute.
        coordinates : tuple, str
            Tuple of strings which explicitly state the dimensions the variable 
            should be assigned to.
        var : list/np.array/pd.DataFrame
            Variable to be set. Must have the dimensions of the coordinates.
            If for instance, the coordinates are x and time, the shape of var must
            be the nx (the number of x-coordinates) and number_of_steps (the
            number of timesteps).

        """
        if var is not None:
            self.grid[str(name)] = (dimensions, var)
            if hasattr(self.grid, 'reservoir_mask') and 'x' in dimensions:
                self.grid[str(name)] = xr.where(self.grid['reservoir_mask'], self.grid[str(name)], 0)
        
    def assign_point_parameters(self):
        """Assign the parameters important for determining subsidence at the 
        points in the SubsidenceModel.grid xarray.Dataset object.
        """
        if self.hasattr('points'):
            self.assign_new_coordinates('points', self.points.names)
        
    def assign_observation_parameters(self):
        """Assign the parameters important for determining subsidence at the 
        observation points in the SubsidenceModel.grid xarray.Dataset object.
        """
        if self.hasattr('observation_points'):
            self.assign_new_coordinates('observations', self.observation_points.names)
     
    def assign_1D_or_2D(self, var, name):
        var = np.array(var)
        var_shape = var.shape
        if len(var_shape) == 3:
            self.assign_data_variable(name, ('y', 'x', 'reservoir'), var)
        if len(var_shape) == 2:
            try: self.assign_data_variable(name, ('reservoir', 'time'), var)
            except: self.assign_data_variable(name, ('y', 'x'), var) 
        if len(var_shape) == 1:
            self.assign_data_variable(name, ('reservoir'), var)
            
    def drop_from_grid(self, var, name):
        if name in list(self.grid.attrs.keys()):
            del self.grid.attrs[name]
        elif hasattr(self.grid, name):
            self.grid = self.grid.drop(name)

        # reassign
        self.assign_1D_or_2D(var, name)
        
    # Set plottig defaults
    def set_contourf_defaults(self, kwargs = {}):
        """Set the standard keyword arguments for matplotlib.pyplot.contourf.
        These will influence the presentation of filled contours in figures.
        
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
        self._contourf_defaults = _plot_utils.set_defaults(kwargs, defaults = self._contourf_defaults)
        
    def set_contour_defaults(self, kwargs = {}):
        """Set the standard keyword arguments for matplotlib.pyplot.contour.
        These will influence the presentation of contours in figures.
        
        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.contour.
            function. The dictionary should be built like: 
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.contour. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.contour.html
        """
        self._contour_defaults = _plot_utils.set_defaults(kwargs, defaults = self._contour_defaults)
        
    def set_clabel_defaults(self, kwargs = {}):
        """Set the standard keyword arguments for matplotlib.pyplot.clabel.
        These will influence the presentation of contours in figures.
        
        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.clabel.
            function. The dictionary should be built like: 
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.clabel. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.clabel.html
        """
        self._clabel_defaults = _plot_utils.set_defaults(kwargs, defaults = self._clabel_defaults)
        
    def set_colorbar_defaults(self, kwargs = {}):
        """Set the standard keyword arguments for matplotlib.pyplot.colorbar.
        These will influence the presentation of colorbars in figures.
        
        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.colorbar
            function. The dictionary should be built like: 
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.colorbar. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.colorbar.html
        """
        self._colorbar_defaults = _plot_utils.set_defaults(kwargs, defaults = self._colorbar_defaults)
    
    def set_plot_defaults(self, kwargs = {}):
        """Set the standard keyword arguments for matplotlib.pyplot.plot.
        These will influence the presentation of line plots in figures.
        
        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.plot
            function. The dictionary should be built like: 
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.plot. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
        """
        self._plot_defaults = _plot_utils.set_defaults(kwargs, defaults = self._plot_defaults)

    def set_shape_defaults(self, kwargs = {}):
        """Set the standard keyword arguments for matplotlib.patches.Polygon.
        These will influence the presentation of polygons in figures.
        
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
        self._shape_defaults = _plot_utils.set_defaults(kwargs, defaults = self._shape_defaults)
    
    def set_annotation_defaults(self, kwargs = {}):
        """Set the standard keyword arguments for matplotlib.pyplot.annotate.
        These will influence the presentation of line plots in figures.
        
        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.annotate
            function. The dictionary should be built like: 
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.annotate. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.annotate.html
        """
        self._annotation_defaults = _plot_utils.set_defaults(kwargs, defaults = self._annotation_defaults)
    
    def set_scatter_defaults(self, kwargs = {}):
        """Set the standard keyword arguments for matplotlib.pyplot.scatter.
        These will influence the presentation of line plots in figures.
        
        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.scatter
            function. The dictionary should be built like: 
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.scatter. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html
        """
        self._scatter_defaults = _plot_utils.set_defaults(kwargs, defaults = self._scatter_defaults) 
        
    def set_errorbar_defaults(self, kwargs = {}):
        """Set the standard keyword arguments for matplotlib.pyplot.errorbar.
        These will influence the presentation of errorbars in figures.
        
        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.errorbar
            function. The dictionary should be built like: 
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.errorbar. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html
        """
        self._errorbar_defaults = _plot_utils.set_defaults(kwargs, self._errorbar_defaults)
    
    def set_fill_between_defaults(self, kwargs = {}):
        """Set the standard keyword arguments for matplotlib.pyplot.fill_between.
        These will influence the presentation of colored areas between lines in 
        figures.
        
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
        self._fill_between_defaults = _plot_utils.set_defaults(kwargs, self._fill_between_defaults)
    
    def set_raster_defaults(self, kwargs = {}):
        """Set the standard keyword arguments for matplotlib.pyplot.imshow.
        These will influence the presentation of rasters in figures.
        
        Parameters
        ----------
        kwargs : dict
            A dictionary with the keyword arguments for the matplotlib.pyplot.imshow
            function. The dictionary should be built like: 
                {'keyword': value, 'other_keyword': other_value}.
            The keyword argument must be a string that is one of the kwargs (other parameters)
            of matplotlib.pyplot.imshow. Which keyword arguments are available
            and which values fit with it, are noted here:
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.imshow.html
        """ 
        self._raster_defaults = _plot_utils.set_defaults(kwargs, self._raster_defaults)
    
    def get_contour_levels(self, variable = 'subsidence', levels = None, start = None, end = None, contour_steps = 0.01, drop_value = 0):
        """Get the contour levels for a specific variable based on its minimum and 
        maximum values. The contour levels can be controlle further with this function's 
        arguments.

        Parameters
        ----------
        variable : str, optional
            Name of a model variable. The default is 'subsidence'.
        levels : list, float, optional
            If you want a specific set of contour levels they can be entered here. 
            The default is None. If not None, this option will be taken instead 
            of based on variable values or start-end values.
        start : float, optional
            If you want the contour levels to be set between certain value, you can 
            set the lowest value here. The default is None. If this parameter or 
            the parameter "end" is None, the preference will be the value set with 
            levels first, and then the values of the variable itself. 
        end : float, optional
            If you want the contour levels to be set between certain value, you can 
            set the highest value here. The default is None. If this parameter or 
            the parameter "start" is None, the preference will be the value set with 
            levels first, and then the values of the variable itself. 
        contour_steps : float, optional
            The step size between the contours. The contour levels set with the 
            start and end paramters can be further controlled with this parameter. 
            The default is 0.01.
        drop_value : float, optional
            Drop a value from the range of contour files. The default is 0.

        Raises
        ------
        Exception
            With invalid input.

        Returns
        -------
        levels : list, floats
            List with contour values.

        """
        if not self.hasattr(variable):
            raise Exception(f'Model {self.name} has no attribute "{variable}"')
        if levels is not None:
            if not _utils.is_iterable(levels):
                raise Exception(f'Invalid type ({type(levels)}) for setting contour levels. Use list.')
        elif (start is not None and
              end is not None and
              contour_steps is not None):
            levels = _utils.stepped_space(start, end, contour_steps)
        elif levels is None:
            as_array = self[variable]
            if 'reservoir' in list(as_array.coords):
                min_value, max_value = (
                    np.min((np.min(self[variable].sum(dim = 'reservoir').values), -contour_steps)), 
                    np.max((np.max(self[variable].sum(dim = 'reservoir').values), contour_steps))
                    )
            else:
                min_value, max_value = (np.min((np.min(self[variable].values), -contour_steps)), 
                                        np.max((np.max(self[variable].values), contour_steps))
                                       )
            levels = _utils.stepped_space(min_value, max_value, contour_steps)
            levels = _plot_utils.set_contour_levels(contour_levels = levels, contour_steps = contour_steps, drop_value = drop_value)
        else:
            warn('Warning: Not enough information to set contour levels. Assign a list to "levels", or assign values to "start", "end" and "contour_steps"')
        levels = _plot_utils.set_contour_levels(contour_levels = levels, contour_steps = contour_steps, drop_value = drop_value)
        return  levels
     
    def get_colors_grouped(self, cmap = 'brg', argument = {'similar_first' : 3}, group_colors = None):
        """Get unique colors for each reservoir based on grouping arguments.

        Parameters
        ----------
        cmap : str, optional
            A valid name for matplotlib cmaps. The default is 'brg'. If the 
            group colors argument is not None, this argument is ignored.
        argument : dict, optional
            A dictionary with the basis on which you seperate the groups. 
            The first key is the method on which it is grouped, and value the 
            argument around which this method is organised. 
            Available grouping methods:
                similar_first : Where the reservoirs have the same first number (as dictated by the value) 
                    letters in their name, they will be grouped.
                similar_end: Where the reservoirs have the same last number (as dictated by the value) 
                    letters in their name, they will be grouped.
                similar_name: Where the reservoirs share the same part of text 
                    as indicated by the values in their names. The value for this
                    argument type should be a list with the length of the group.
                    If a reservoir doesn't match any of the groups, it is added
                    as its own group. 
            The default is {'similar_first' : 3}.
        group_colors : list, optional
            A list of tuples with RGB or RGBA colors with the length of the 
            resulting group. Each group will have a base color, on which different 
            occurences in that method have a different brightness.
            The default is None. When None, the colors are based 
            on the cmap argument. 

        Returns
        -------
        list
            A list of tuples with RGB(A) values.

        """
        if isinstance(argument, dict):
            argument_type = list(argument.keys())[0]
            argument_value = list(argument.values())[0]
            if argument_type == 'similar_first':
                if not isinstance(argument_value, int): raise Exception('The argument value for the argument type "similar_first" must be an integer, which indicates the number of letters in the text that should be the same from the start.')
                sort_by = [occurence[:argument_value] for occurence in self.reservoirs]
            elif argument_type == 'similar_end':
                if not isinstance(argument_value, int): raise Exception('The argument value for the argument type "similar_end" must be an integer, which indicates the last few number of letters in the text that should be the same from the start.')
                sort_by = [occurence[:argument_value] for occurence in self.reservoirs]
            elif argument_type == 'similar_name':
                if not isinstance(argument_value, list): raise Exception('The argument value for the argument type "similar_name" must be a list, where each entry is a segement of text present in the isolated groups.')
                sort_by = []
                for occurence in self.reservoirs:
                    corresponding_values = [v in occurence for v in argument_value]
                    if any(corresponding_values):
                        values_that_contain_that_segment = np.array(argument_value)[corresponding_values]
                        longest_value_index = np.argmax(values_that_contain_that_segment)
                        sort_by.append(values_that_contain_that_segment[longest_value_index])
                    else:
                        sort_by.append(occurence)
        else:
            raise Exception('Additional arguments should be of a dict type with the shape: dict(argument_type = argument_value)')
        
        return _plot_utils.get_colors_grouped(self.reservoirs, sort_by, cmap = cmap, group_colors = group_colors)
    
    # Set parameters
    def set_project_folder(self, folder = None, name = None):
        """Set the folder in which model input, save and output files are stored.

        Parameters
        ----------
        folder : str, optional
            Path to a folder. If the folder does not exist, an attempt is made to 
            make it. The default is None. When None, no files are saved.
        name : str, optional
            Name of the model to name the folder with. The default is None. If 
            this parameter is None, the Model name is taken. If this is not present 
            too, it defaults to "unnamed_subsidence_model".
            
        Sets
        ----
        project_folder

        """
        if name is None:
            if self.hasattr('name'):
                name = self.name
            else:
                name = 'unnamed_subsidence_model'
       
        if self.hasattr('project_folder') and folder is not None:
            self.project_folder.project_folder = os.path.join(folder, name)
                    
        if folder is not None:
            project_folder = os.path.join(folder, name)
        else:
            project_folder = None
        self.project_folder = _ProjectFolder.ProjectFolder(project_folder)
    
    def set_points(self, points):
        """Set the points on which the model will calculate the subsidence
        as PointCollection object.

        Parameters
        ----------
        points : PySub.Points.PointCollection.
            
        Sets
        ----
        points:
            PointCollection object storing point location and 
        """

        if points is None:
            self._points = None
            return
        if not isinstance(points, _Points.PointCollection):
            if isinstance(points, _Points.Point):
                points = _Points.PointCollection(points)
            else:
                raise Exception('Points set into model need to be PointCollection or Points objects from the Point module')
        self._check_dim1D('point', points.names, dims_equal = 'point_ids')      
        self._points = points
        self.assign_point_parameters()
    
    def set_observation_points(self, observation_points):
        """Set the observation points as part of the model.
        
        Parameters
        ----------
        observation_points : PySub.Points.ObservationCollection
            ObservationCollection object containing the data of all the observation 
            data.
            
        Sets
        ----
        ObservationCollection.
            Object storing the cartesian coordinates and label/name of a location as
            a point. Additionally it stores the observation s and its errors over
            time.
        """
        if _utils.is_iterable(observation_points):
            if str(type(observation_points)) == "<class 'PySub.Points.ObservationCollection'>":
                pass
            else:
                for i in observation_points:
                    if not str(type(i)) == "<class 'PySub.Points.ObservationPoint'>":
                        raise Exception('Entries should be a list or numpy array with ObservationPoint objects from the Points module.')
                observation_points = _Points.ObservationCollection(observation_points)
        else:
            raise Exception('Entries should be a list or numpy array with ObservationPoint objects from the Points module.')
        
        self._observation_points = observation_points 
        self.number_of_observation_points = self._observation_points.number_of_observation_points
        self.assign_observation_parameters()
    
    def load_from_raster(self, fname, layer = None):
        """Load a raster file and convert its data to fit the model.

        Parameters
        ----------
        fname : str
            Path to valid .tif raster file.
        layer : int, optional
            Index of the layer to be extracted from the .tif file. Starts with 1. 
            The default is None. When None, all layers will be extracted.

        Raises
        ------
        Exception
            When the grid is not yet built.

        Returns
        -------
        3D np.ndarray, floats
            The loaded raster with the values integrated in the already built grid.
            has the shape (y, x, number of bands in the tiff file)
        """
        
        print(f'Attempting to load raster file from {fname}')
        loaded_data, x, y, crs = _shape_utils.load_raster(fname, layer = layer)
        self._check_crs(crs)
        number_of_bands = loaded_data.shape[0]
        if not (self.hasattr('number_of_steps') and self.hasattr('number_of_reservoirs') and self.built):
            raise Exception('To set variable from a raster file, built the grid first.')
        reshaped_data = np.transpose(loaded_data, axes = (1, 2, 0))
        data_xr = xr.DataArray(reshaped_data, (('y', y[:, 0]), ('x', x[0, :]), ('band', np.arange(number_of_bands))))
        interpolated_data = data_xr.interp_like(self.grid, method = 'linear', kwargs = {'fill_value': 0})
        return interpolated_data.values
    
    def load_from_csv(self, fname, delimiter = ';', header = 0, decimal = ','):
        loaded_data, x, y = _shape_utils.load_raster_from_csv(fname, delimiter = delimiter, header = header, decimal = decimal)
        
        
        number_of_bands = loaded_data.shape[-1]
        if not (self.hasattr('number_of_steps') and self.hasattr('number_of_reservoirs') and self.built):
            raise Exception('To set variable from a raster file, built the grid first.')
        data_xr = xr.DataArray(loaded_data, (('y', y[:, 0]), ('x', x[0, :]), ('band', np.arange(number_of_bands))))
        interpolated_data = data_xr.interp_like(self.grid, method = 'linear', kwargs = {'fill_value': 0})
        return interpolated_data.values
    
    def set_1D_or_2D(self, name, var, layer = None):
        if not _utils.is_iterable(var):
            if isinstance(var, str):
                if os.path.isfile(var): 
                    self.project_folder.write_to_input(var)
                else:
                    raise Exception(f'Invalid file: {var}')
                if var.endswith('tif'):
                    loaded_var = self.load_from_raster(var, layer = layer)
                    
                elif var.endswith(('.txt', '.csv')):
                    loaded_var = self.load_from_csv(var, delimiter = ';', header = 0)
                else:
                    raise Exception('Invalid file type encountered. Supported file types: .tif, .txt, .csv.')
                number_of_bands = loaded_var.shape[-1]
                if number_of_bands != self.number_of_reservoirs:
                    raise Exception(f'Number of bands {number_of_bands} disequals number of reservoirs: {self.number_of_reservoirs}.')
                
                var = loaded_var
            else:
                raise Exception(f'Set {name} with an iterable or a link to a file. Invalid type encountered {type(var)}.')
        else:      
            
            self._check_dim1D(name, var)    
            
            if _utils.is_list_of_strings(list(var)):
                list_of_paths = var # redefinition for clarity
                list_var = []
                for path in list_of_paths:
                    if os.path.isfile(path): 
                        self.project_folder.write_to_input(path)
                    else:
                        raise Exception(f'Invalid file: {var}')
                    if path.endswith('.tif'):
                        loaded_var = self.load_from_raster(path, layer = layer)
                    elif path.endswith(('.txt', 'csv')):
                        loaded_var = self.load_from_csv(path, delimiter = ';', header = 0)
                    else:
                        raise Exception(f'Invalid path to raster file: {path}')
                    number_of_bands = loaded_var.shape[-1]
                    if number_of_bands == 1:
                        transposed_var = loaded_var.reshape((self.ny, self.nx))
                        list_var.append(transposed_var)
                    else:
                        raise Exception(f'Invalid number of bands in .tif file: {number_of_bands}. Enter as a string for a .tif file with all the reservoirs in it, as a list for tif files with each one reservoir.')
                    
                var = np.array(list_var)
            elif _utils.is_list_of_numbers(var):
                pass
            else:
                raise Exception(f'Invalid type {type(var)} for setting variable.')
        self.drop_from_grid(var, name)
    
    def set_shapes(self, shapes):
        """Set the reservoir extent from different shape formats as Geometry subjects.

        Parameters
        ----------
        shapes : list, str or tuple, float
            List of paths to .csv, .tif or .shp file for raster and polygon files.
            If the shapes is a list of tuples with float values, it is interpreted 
            as a list of point coordinates.
        
        Sets
        ----
        shapes
        """
        if shapes is not None:
            if not _utils.is_iterable(shapes):
                raise Exception('Set shapes with lists of the available datatypes (str for .shp, .csv or .tif files or xy-coordinates.)')
            self._check_dim1D('shapes', shapes)    
            self._shapes = _Geometries.fetch(shapes, 
                                             scatter_kwargs = self.scatter_defaults,
                                             shape_kwargs = self.shape_defaults,
                                             raster_kwargs = self.raster_defaults)
            
            if self.hasattr('bounds'):
                for shape in self._shapes:
                    shape.in_bound(self.bounds, self.dx)
                            
    def set_depth_to_basements(self, depth_to_basements):
        """Sets the depths of the basements as a part of the model, perform checks.
        
        depth_to_basements : list, float/int
            Depth to the rigid basement for the van Opstal nucleus of strain 
            method in m. The list must have the same length as the number of reservoirs in 
            the model. The default is None. Raises Exception when None.
        
        Sets
        ----
        depth_to_basements
        """
        if _utils.isnan(depth_to_basements):
            depth_to_basements = None
        if not depth_to_basements is None:
            _utils._check_low_high(depth_to_basements, 'depth_to_basements', EPSILON, 100000)
            self._check_dim1D('depth to basements', depth_to_basements)
            self.set_1D_or_2D('depth_to_basements', depth_to_basements)
            
    def set_poissons_ratios(self, poissons_ratios):
        if _utils.isnan(np.array(poissons_ratios)) or poissons_ratios is None:
            raise Exception(f"Poisson's ratio must be entered. Current values {poissons_ratios}")
        if not poissons_ratios is None:
            _utils._check_low_high(poissons_ratios, 'poisson_ratios', EPSILON, 0.45)
            self._check_dim1D('poissons ratios', poissons_ratios)
            self._poissons_ratios = poissons_ratios
            self.set_1D_or_2D('poissons_ratios', poissons_ratios)
    
    def set_depths(self, depths, layer = None):
        """Sets the depths as a part of the model and performs checks.
        depths : list float/int, optional
            Depths to the top of each reservoirs in m. The list must have 
            the same length as the number of reservoirs in 
            the model. The default is None. Raises Exception when None.
        
        Sets
        ----
        depths
        """
        self.set_1D_or_2D('depths', depths, layer = layer)
        
    def set_timesteps(self, timesteps):
        """Sets the times for eacht timestep as a part of the model and performs checks.
        timesteps : list, np.datetime64, optional
            The timestamps of each step in time. These need to be of equal 
            step length. Per year would be ['1990', '1991', etc.]. The default 
            is None. Raises Exception when None.
        
        Sets
        ----
        timesteps
        """
        self._check_dim1D('timesteps', timesteps, dims_equal = 'timesteps')
        self._timesteps = _utils.convert_to_datetime(timesteps)
        
        if self.number_of_steps < 2:
            raise Exception(f'Not enough timesteps to calculate compaction/subsidence. At least 2 timesteps are required. \n Current number of timesteps: {self.number_of_steps}')
    
    def set_reservoirs(self, reservoirs):
        """Sets the names of the reservoirs for as a part of the model and performs checks.
        reservoirs : list, str, optional
            The names of each reservoir. The default is None. The list must have 
            the same length as the number of reservoirs in the model.

        Sets
        ----
        reservoirs
        """
        self._check_dim1D('reservoirs', reservoirs)
        reservoirs = [str(r) for r in reservoirs]
        self._reservoirs = np.array(reservoirs)
        
    def set_bounds(self, bounds = None):
        """Set the bounds of the model based on entry or set shapefiles.

        Parameters
        ----------
        bounds : array-like, int/float, optional
            An array-like object with 4 values.
            [0] lower x, [1] lower y, [2] upper x, [3] upper y.
            Default is None. When None, it will check for any set shapes.
            If there are no shapes, returns an exception.

        Sets
        ----
        bounds

        """
        if bounds is None:
            if self.shapes is not None:   
                bound_collection = np.array([shape.bounds for shape in self.shapes])
                self._bounds = _utils.bounds_from_bounds_collection(bound_collection)
            else:
                raise Exception('The function set_bounds requires 4 numeric entries or requires SubsidenceModel object to have shape- or gridfiles set.')
        else:
            bounds = [s for s in bounds if _utils.is_number(s)]
            if len(bounds) == 4:                
                if self.hasattr('shapes'):   
                    shapes_per = _utils.flatten_ragged_lists2D(self.shapes)
                    np_shapes = np.array(shapes_per[0])
                    if (bounds[2] < np.min(np_shapes[:,0]) or 
                        bounds[0] > np.max(np_shapes[:,0]) or
                        bounds[3] < np.min(np_shapes[:,1]) or
                        bounds[1] > np.max(np_shapes[:,1])):
                        raise Exception(f'Bounds {bounds} do not overlap with set shapes.')
                self._bounds = bounds
                
            else:
                raise Exception('The function set_bounds requires 4 numeric entries or requires SubsidenceModel object to have shape- or gridfiles set.')
        return self._bounds
    
    def set_dx(self, dx):
        """Sets the size of the cells along the x-axis as a part of the model and performs checks.
        
        dx : float
            Distance between grid nodes along the x-axis in m. The default is 
            None. Raises exception when None.
        
        Sets
        ----
        dx 
        dy
        """
        _utils._check_low_high(dx, 'dx', 0, 10000)
        self._dx = dx
        self._dy = dx
        self.drop_from_grid(dx, 'dx')
    
    def set_dy(self, dy = None):
        """Sets the size of the cells along the x-axis as a part of the model and performs checks.
        
        dy : float/int, optional
            Distance between grid nodes along the y-axis in m. The default is 
            None. Defaults to dx if None. If dx is None, raises Exception.
            
        Sets
        ----
        dy
        """
        if dy is None:
            if self._dx != None:
                self._dy = self._dx
            else:
                raise Exception('Not able to set dy with None. Set dx or enter numerical value above 0.')
        else:
            _utils._check_low_high(dy, 'dy', 0, 10000)
            self._dy = dy
    
    def set_influence_radius(self, influence_radius):
        """Sets the influence radius as a part of the model and performs checks.
        
        influence_radius : float
            Distance from which the subsidence is set to 0 in m. The default 
            is 0.
        """
        _utils._check_low_high(influence_radius, 'influence radius', 0, 100000)
        self._influence_radius = influence_radius
    
    def set_dxyradius(self, dx = None, dy = None, influence_radius = 0):
        """Sets the dx, dy and influence parameters as a part of the model.
        dx : float, optional
            Distance between grid nodes along the x-axis in m for each reservoir. 
            The default is None. Raises exception when None.
        dy : float, optional
            Distance between grid nodes along the y-axis in m. The default is 
            None. Defaults to dx if None, if dx is None, raises Exception.
        influence_radius : float, optional
            Distance from which the subsidence is set to 0 in m. The default 
            is 0. Raises exception when None.
        
        Sets
        ----
        dx
        dy
        influence_radius        
        """
        self.set_dx(dx)
        self.set_dy(dy)
        self.set_influence_radius(influence_radius)
        
    def set_subsidence_model_type(self, subsidence_model):
        """Set the type of the subsidence model with which the model will calculate 
        the subsidence based on the compaction.

        Parameters
        ----------
        subsidence_model : str, optional
            Method of subsidence of the model. Currently available:
            - "nucleus of strain", Van Opstal 1974
            - "knothe", Stroka et al. 2011. 
            
        Sets
        ----
        Exception
            When the model type or format is invalid
        """
        self.check_subsidence_model(subsidence_model)        
        self._subsidence_model_type = subsidence_model.lower()
        
    def set_knothe_angles(self, knothe_angles, layer = None):
        """Sets the knothe angles for each reservoir.
        
        Parameters
        ----------
        knothe_angles : list, str, float
            The knothe angles for the Knothe subsidence model. The list must have 
            the same length as the number of reservoirs in the model.
            When a specific value for the entire reservoir, must be a float. 
            When a string it is assumed to be a path to .tif or .csv raster file.
        layer : int, optional
            Integer value higher than 0 indicating the index of the layer storing 
            the specific data in a .tif raster file. 
            The default is None. Then the first layer is chosen.

        Raises
        ------
        Exception
            When invalid values are encountered.

        Sets
        ----
        Knothe angles

        """
        if knothe_angles is not None:
            invalid_values = np.where(np.array(knothe_angles) % 90 == 0)[0]
            if len(invalid_values) > 0:
                raise Exception(f'Invalid values for knothe angles encountered: {knothe_angles[invalid_values]}')
            self.set_1D_or_2D('knothe_angles', knothe_angles, layer = layer)
    
    def set_subsidence_model_parameters(self, model, knothe_angles = None, 
                             depth_to_basements = None, poissons_ratios = None, layer = None):
        """Set the subsidence model and any relevant variables.

        Parameters
        ----------
        model : str, optional
            Method of subsidence of the model. Currently available:
            - "nucleus of strain", Van Opstal 1974
            - "knothe", Stroka et al. 2011. 
        knothe_angles : list, str, float
            The knothe angles for the Knothe subsidence model. The list must have 
            the same length as the number of reservoirs in the model.
            When a specific value for the entire reservoir, must be a float. 
            When a string it is assumed to be a path to .tif or .csv raster file.
            Raises Exception when None and model = 'knothe'.
        depth_to_basements : list, float/int
            Depth to the rigid basement for the van Opstal nucleus of strain 
            method in m. The list must have the same length as the number of reservoirs in 
            the model. The default is None. Raises Exception when None and model = 'nucleus of strain'.
        layer : int, optional
            Integer value higher than 0 indicating the index of the layer storing 
            the specific data in a .tif raster file. 
            The default is None. Then the first layer is chosen.

        Raises
        ------
        Exception
            When combination model and variable do not match.

        """
        self.set_subsidence_model_type(model)
        
        if model == 'knothe':
            if knothe_angles is not None:
                self.knothe_angles(knothe_angles, layer = layer)
            else:
                raise Exception('Knothe angles need to be defined for each reservoir when subsidence model type is Knothe.')
        
        if model == 'nucleus of strain':
            if poissons_ratios is not None:
                self.set_poissons_ratios(poissons_ratios)
                if depth_to_basements is None:
                    self.set_depth_to_basements([None for _ in self.reservoirs])
                else:
                    self.set_depth_to_basements(depth_to_basements)
            else:
                raise Exception("""Poisson's ratios' need to be defined for each reservoir when subsidence model type is "nucleus of strain".""")
    
    def set_compaction(self, compaction):
        """Set the compaction in m³ that is used to compute the subsidence from.

        Parameters
        ----------
        compaction : list or numpy array.
            The compaction volume in m³ with the shape (ny, nx, number of reservoirs, 
            number_of_steps).

        Sets
        ----
        compaction
        """
        if not _utils.is_iterable(compaction):
            raise Exception(f'Compaction must be an iterable with the same shape as the model: {self.shape}. wrong type: {type(compaction)}.')
        compaction = np.array(compaction)
        
        if not self.built:
            raise Exception('Build model before setting compaction')
        
        if not compaction.shape == self.shape:
            raise Exception(f'Compaction must be an iterable with the same shape as the model: {self.shape}, {list(self.grid.coords)}. Wrong shape: {compaction.shape}.')
    
        self.assign_data_variable('compaction', ['y', 'x', 'reservoir', 'time'], compaction)
        
    # Tools
    def copy(self):
        """Make a deep copy of the model.

        Returns
        -------
        SubsidenceModel
            A model with the same variables that is independent of the model
            it has been copied from.

        """
        return copy.deepcopy(self)
    
    def get_id_bounds_reservoir(self, reservoir_layer, grid = None):
        """Get the bounds of the reservoirs in grid x and y indeces.

        Parameters
        ----------
        reservoir_layer : int
            The index of the reservoir dimension of the grid, which stores the 
            names of the reservoirs.

        Returns
        -------
        lower_x : int
        lower_y : int
        higher_x : int
        higher_y : int

        """
        ys, xs = np.where(self.grid.reservoir_mask.isel(reservoir = reservoir_layer) == 1)
        if len(xs) == 0:
            raise Exception(f'Reservoir {self.reservoirs[reservoir_layer]} extend not in grid!')
        lower_x, higher_x = np.min(xs), np.max(xs)
        lower_y, higher_y = np.min(ys), np.max(ys)
        return lower_x, lower_y, higher_x, higher_y
    
    def mask_reservoirs(self):
        """Set a reservoir in the grid object based on the added shapefiles
        stored in the SubsidenceModel.shapefiles property.
        
        Sets
        -------
        reservoir_mask
        """
        self.check_for_grid()
        model_grid = np.zeros(shape = (self.ny, self.nx, self.number_of_reservoirs))
        for i, reservoir in enumerate(self._shapes):
            model_grid[:,:,i] = reservoir.mask(self.grid)
        self.assign_data_variable('reservoir_mask', ('y', 'x', 'reservoir'), model_grid)
        
    def convert_to_grid(self, name):
        """Convert data variable that has been indexed by number of reservoirs 
        and number of timesteps and time, to a spatial occurence (y, x, reservoir, 
        time) in the grid. The data variable in the grid will be named 'grid_name'.
        """
        if 'x' in list(self.grid[name].coords):
            return
        
        self.grid[name] = self.grid[name] * self.grid.reservoir_mask
    
    def reservoir_label_to_int(self, val):
        """Returns the index of the reservoir based on the entry val.

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
        if isinstance(val, str):
            if val in self.grid.reservoir.values:
                index = np.where(self.grid.reservoir.values == val)[0][0]
            else:
                warn(f'Warning: The key {val} does not correspond with an available reservoir')
                return None
        elif isinstance(val, int):
            if val in list(range(self.number_of_reservoirs)):
                index = val
            elif val in list(range(-self.number_of_reservoirs, 0)):
                index = val
            else:
                raise Exception(f'The key {val} does not correspond with an available reservoir')
        else:
            raise Exception(f'Type {type(val)} not supported, use string or integer to index.')
        return int(index)
            
    def time_label_to_int(self, val):
        """Returns the index of the timestep based on the entry val.

        Parameters
        ----------
        val : int/str
            Representation of the timestep by index year number or name.
            Examples by index: 0, 1, -1, -2.
            Examples by year: 1990
            Examples by name: '1990', '1990-01-01'
            When a year is entered in integer it will find the first entry of that year.
            When an index overlaps with a year, it will interpret the entered
            integer as an index.
        Returns
        -------
        int
            Index of the timestep.

        """
        index = 0
        if self.timesteps.dtype == np.int64:
            if isinstance(val, str):
                try:
                    if int(val) in self.grid.time.values:
                        index = np.where(self.grid.time.values == int(val))[0][0]
                except:
                    raise Exception(f'The key {val} is not an available timestep')
            elif isinstance(val, int):
                if int(val) in list(range(self.number_of_steps)):
                    index = int(val)
                elif int(val) in list(range(-self.number_of_steps, 0)):
                    index = int(val)
                elif int(val)  in self.grid.time.values:
                    index = np.where(self.grid.time.values == int(val))[0][0]
                else:
                    raise Exception(f'The key {val} is not an available timestep')
            else:
                raise Exception(f'Type {type(val)} not supported, use string or integer to index.')
        elif self.timesteps.dtype == 'datetime64[ns]':
            if isinstance(val, int):
                times_as_years = np.datetime_as_string(self.timesteps, unit = 'Y').astype(int)
                if val in times_as_years:
                    index = np.where(times_as_years == val)[0][0]
                elif int(val) in list(range(self.number_of_steps)):
                    index = int(val)
                elif int(val) in list(range(-self.number_of_steps, 0)):
                    index = int(val)
                else:
                    raise Exception(f'The key {val} is not an available timestep')
            elif isinstance(val, str):
                times_yy_mm_dd = np.datetime_as_string(self.timesteps, unit = 'D')
                times_as_years = np.datetime_as_string(self.timesteps, unit = 'Y')
                t = pd.Series(self.timesteps)
                times_dd_mm_yy = t.dt.strftime('%d-%m-%Y').values
                if val in times_yy_mm_dd:
                    index = np.where(times_yy_mm_dd == val)[0][0]
                elif val in times_as_years:
                    index = np.where(times_as_years == val)[0][0]
                elif val in times_dd_mm_yy:
                    index = np.where(times_dd_mm_yy == val)[0][0]
            elif isinstance(val, pd.Timestamp):
                index = np.where(self.grid.time.values == val)[0][0]
        return int(index)
    
    def point_label_to_int(self, val):
        """Returns the index of the point based on the entry val.

        Parameters
        ----------
        val : int/str
            Representation of the point by index or name.
            Examples by index: 0, 1, -1, -2.
            Examples by reservoir name: 'B100'.

        Returns
        -------
        int
            Index of the point.

        """
        index = 0
        if isinstance(val, str):
            if str(val) in self.grid.points.values:
                index = np.where(self.grid.points.values == str(val))[0][0]
            else:
                raise Exception(f'The key {val} is not an available point')
        elif isinstance(val, int):
                index = val
        else:
            raise Exception(f'Type {type(val)} not supported, use string or integer to index.')
        return int(index)
    
    def observation_label_to_int(self, val):
        """Return the index of the observation point based on entry parameter val.

        Parameters
        ----------
        val :int/str
            Representation of the observation point by index or name.
            Examples by index: 0, 1, -1, -2.
            Examples by reservoir name: 'B100-1'.

        Returns
        -------
        int
            Index of the observation point.
        """
        index = 0
        if isinstance(val, str):
            if str(val) in self.grid.observations.values:
                index = np.where(self.grid.observations.values == str(val))[0][0]
            else:
                raise Exception(f'The key {val} is not an available observation point')
        elif isinstance(val, int):
                index = val
        else:
            raise Exception(f'Type {type(val)} not supported, use string or integer to index.')
        return int(index)

    def get_subsidence_with_ids(self, reservoir = None, time = None):
        """Get the subsidence data for a specific reservoir and timestep

        Parameters
        ----------
        reservoir : int/str, optional
            Name or index of the reservoir of which the relevant data is being
            asked for. The default is None. With None, the total subsidence for all 
            reservoirs will be returned. When a list is entered, only the first 
            entry of that list will be used.
        time : int/str, optional
            Name, year or index of the timestep of which the relevant data is being
            asked for. The default is None. With None, the data for all timesteps
            will be returned. When a list is entered, only the first entry of 
            that list will be used.
            Examples by index: 0, 1, -1, -2.
            Examples by year: 1990
            Examples by name: '1990', '1990-01-01'
        Returns
        -------
        np.ndarray
            3D numpy array containing subsidence in m in a grid for all timesteps 
            (shape = (ny, nx, number_of_steps)) or for 1 timestep (shape = 
            (ny, nx, 1)).
        """
        if reservoir is None:
            if time is None:
                data = np.zeros((self.ny, self.nx, self.number_of_steps))
                for step in range(self.number_of_steps):
                    time_i = self.time_label_to_int(str(self._timesteps[step]))
                    data[:, :, step] = self.total_subsidence.isel(time = time_i)
            else:
                data = np.zeros((self.ny, self.nx, 1))
                time_i = self.time_label_to_int(time)
                data[:, :, 0] = self.total_subsidence.isel(time = time_i)
        else:
            reservoir = self.reservoir_label_to_int(reservoir)
            if time is None:
                data = np.zeros((self.ny, self.nx, self.number_of_steps))
                for step in range(self.number_of_steps):
                    data[:, :, step] = self.subsidence.isel(reservoir = reservoir, time = step)
            else:
                data = np.zeros((self.ny, self.nx, 1))
                time_i = self.time_label_to_int(time)
                data[:, :, 0] = self.subsidence.isel(reservoir = reservoir, time = time_i)               
        return data
    
    def get_timeseries_coords(self, variable = 'subsidence', reservoir = None, x = None, y = None):
        if not hasattr(self, variable):
            raise Exception(f'SubsidenceModel {self.name} has no attribute {variable}.')
        reservoir_index = _plot_utils.reservoir_entry_to_index(self, reservoir)
        
        variable_values = getattr(self, variable).isel(reservoir = reservoir_index).sum(dim = 'reservoir')
        
        timeseries = variable_values.interp(x = x, y = y)
        timeseries = timeseries.fillna(0)
        return timeseries
    
    def get_timeseries(self, x, y, variable = 'subsidence', reservoir = None):
        """Get the timeserie subsidence for all or a single specified reservoir.
        The location can be entered as grid index or coordinate. When the coordinate is not
        present in the grid, it will be interpolated.

        Parameters
        ----------
        x : int/float
            An integer or float representing the x-location of the grid node over 
            which the subsidence timeseries will be returned. The variable is 
            interpreted as a coordinate value.
        y : int/float
            An integer or float representing the y-location of the grid node over 
            which the subsidence timeseries will be returned. The variable is 
            interpreted as a coordinate value.
        variable : str
            A variable whith time as a dimension and currently in the Model.grid
            object.
        reservoir : int/str, optional
            Name or index of the reservoir of which the relevant data is being
            asked for. The default is None. With None, the total subsidence for all 
            reservoirs will be returned. 
        
            
        Returns
        -------
        timeseries : xr.Dataset
            Subsidence values (m) for specified location and reservoir input over time.
        """
        
        
        reservoir = _plot_utils.reservoir_entry_to_index(self, reservoir)
        
        if not self.bounds[0] < x < self.bounds[2]:
            warn(f'X-coordinate {x} does not fall within the range of x-coordinates of the model ({self.bounds[0]}, {self.bounds[2]}).')
        if not self.bounds[1] < y < self.bounds[3]:
            warn(f'Y-coordinate {y} does not fall within the range of y-coordinates of the model ({self.bounds[1]}, {self.bounds[3]}).')
            
        timeseries = self.get_timeseries_coords(variable = variable, reservoir = reservoir, x = x, y = y)
        return timeseries
    
    def get_max_subsidence(self, time = None, reservoir = None):
        """Return the subsidence of maximum subsidence in the model at a specified time
        and from specified reservoirs.

        Parameters
        ----------
        time : int, stroptional
            Index or label (SubsidenceModel.timesteps) for the chosen timstep. 
            The default is None. When None, the last timestep is chosen.
            If a list with, no error will occur, but the first timestep will be 
            chosen.
        reservoir : int, str or list of int or str, optional
            The index or name of the reservoirs you want to know the maximum 
            subsidence of. If it is a list, all the reservoirs in that list 
            will be displayed. The default is None. When None, all reservoirs 
            will be displayed.

        Returns
        -------
        maximum_subsidence, (x, y) : float, (float, float)
            A tuple with the value of the maximum subsidence and a tuple
            with the x- and y-coordinates of the location with the most
            subsidence.
        """
        _reservoir = _plot_utils.reservoir_entry_to_index(self, reservoir)
        data = np.zeros(shape = (self.ny, self.nx))
        if time is None:
            time = -1
        else:
            time = _plot_utils.time_entry_to_index(self, time)[0]
        for r in _reservoir:
            data3D = self.get_subsidence_with_ids(reservoir = r, time = time)
            data += data3D.reshape((data.shape[0], data.shape[1]))
        maximum_subsidence = np.min(data)
        yi, xi = np.unravel_index(np.argmin(data, axis=None), data.shape)
        x = float(self.grid.x.isel(x = xi).values)
        y = float(self.grid.y.isel(y = yi).values)
        return maximum_subsidence, (x, y)
    
    def get_subsidence_overview(self, group = None):
        """Makes a pandas DataFrame with the subsidence caused by each reservoir
        with an option to group several reservoirs.

        Parameters
        ----------
        group : list of lists, optional
            a list with the length of each group. Each entry into this list is 
            another list with strings that are the names of the reservoirs in 
            that group. The default is None. When None, no grouping is implied.

        Returns
        -------
        subsidence_df : pandas.DataFrame
            A pandas DataFrame with the subsidence caused by each reservoir and 
            group.

        """
        if group is not None:
            if not all([isinstance(subgroup, list) for subgroup in group]):
                raise Exception('The format for groups should be a list of lists representing groups.')
            flattened_grouping = [j for sub in group for j in sub]
            if len(flattened_grouping) > self.number_of_reservoirs:
                raise Exception('Too many reservoirs. Grouping needs to be some, or all of the reservoirs.')
            filtered_group = []
            for subgroup in group:
                filtered_group.append([r for r in subgroup if r in self.reservoirs])
            check_if_unique, doubles = _utils.check_if_unique(flattened_grouping)
            if not check_if_unique:
                raise Exception(f'The reservoirs: {doubles} are referenced more then once, which conflicts with the grouping.')
            
            
        subsidence_dict = {}
        subsidence_at_deepest_point = {r: 
             self.get_max_subsidence(reservoir = r)[0] 
             for r in self.reservoirs}
        subsidence_dict[self.name] = subsidence_at_deepest_point
        
        subsidence_df = pd.DataFrame(subsidence_dict)
        subsidence_df = pd.concat([subsidence_df, pd.DataFrame([[self.get_max_subsidence()[0]]], index = ['Total'], columns = [self.name])])
        subsidence_df['Cumulative'] = subsidence_df[self.name].values
        
        if group is not None:
            for subgroup in filtered_group:
                group_subsidence = self.get_max_subsidence(reservoir = subgroup)[0]
                subsidence_df.loc[subgroup, 'Cumulative'] = group_subsidence
        return subsidence_df
    
    def get_min_subsidence(self, time = None, reservoir = None):
        """Return the subsidence of minimum subsidence in the model at a specified time
        and from specified reservoirs.

        Parameters
        ----------
        time : int, stroptional
            Index or label (SubsidenceModel.timesteps) for the chosen timstep. 
            The default is None. When None, the last timestep is chosen.
            If a list with, no error will occur, but the first timestep will be 
            chosen.
        reservoir : int, str or list of int or str, optional
            The index or name of the reservoirs you want to know the minimum 
            subsidence of. If it is a list, all the reservoirs in that list 
            will be displayed. The default is None. When None, all reservoirs 
            will be displayed.

        Returns
        -------
        minimum_subsidence, (x, y) : float, (float, float)
            A tuple with the value of the minimum subsidence and a tuple
            with the x- and y-coordinates of the location with the most
            subsidence.
        """
        _reservoir = _plot_utils.reservoir_entry_to_index(self, reservoir)
        data = np.zeros(shape = (self.ny, self.nx))
        if time is None:
            time = -1
        else:
            time = _plot_utils.time_entry_to_index(self, time)[0]
        for r in _reservoir:
            data3D = self.get_subsidence_with_ids(reservoir = r, time = time)
            data += data3D.reshape((data.shape[0], data.shape[1]))
        minimum_subsidence = np.max(data)
        xi, yi = np.unravel_index(np.argmax(data, axis=None), data.shape)
        if 0 < xi < self.nx:
            x = float(self.grid.x.isel(x = xi).values)
        else: 
            x = 0
        if 0 < yi < self.ny: 
            y = float(self.grid.y.isel(y = yi).values)
        else:
            y = 0
        return minimum_subsidence, (x, y)
    
    def get_max_subsidence_timeseries(self, reservoir = None):
        """Get the timeseries of the location with the largest amount of subsidence.
        
        The maximum subsidence is determined at the final timestep in the model.

        Parameters
        ----------
        reservoir : int/str, optional
            Name or index of the reservoir of which the relevant data is being
            asked for. The default is None. With None, the total subsidence for all 
            reservoirs will be returned. 
            
        Returns
        -------
        timeseries : xr.Dataset
            Subsidence values for the location with the largest amount of subsidence 
            and reservoir input over time.
        """
        _, (x, y) = self.get_max_subsidence(time = -1, reservoir = reservoir)
        timeseries = self.get_subsidence_timeseries(x, y, reservoir = reservoir)
        return timeseries
    
    # build grid
    def build_grid(self):
        """Generates an xarray Dataset, grid, which stores the subsidence and other 
        data in a geographic and temporal representative format.
        
        It is generated using already set parameters in the SubsidenceModel object.
        When these values are not set, an error will occur and indicate which parameters 
        are missing and how to set them.
        
        Additional dimensions can be added when assigning points or observations.
        
        Sets
        -------
        grid : xarray Dataset
        """ 
        self._check_build_paramaters()
        self.grid = generate_grid_from_bounds(self._bounds, 
                                              self._dx, self._dy, 
                                              timesteps = self._timesteps, 
                                              reservoir_layers = self._reservoirs, 
                                              influence_radius = self._influence_radius,
                                              include_mask = True)
        self.nx = self.grid.dims['x']
        self.ny = self.grid.dims['y']
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.assign_attribute('influence_radius', self._influence_radius)
            
        self.built = True
        
    def build_like(self, grid):
        """Generates an xarray Dataset, grid, which stores the subsidence and other 
        data in a geographic and temporal representative format.
        
        It is generated using already set parameters in the argument grid.
                
        Additional dimensions can be added when assigning points or observations.
        
        Parameters
        ----------
        grid : xr.Dataset
            xarray grid with at least the coordinates x, y, time and reservoir and
            the attributes bounds, dx, dy and influence_radius.
        
        Sets
        -------
        grid : xarray Dataset
        """ 
        if all(hasattr(grid, attr) for attr in ['bounds', 'dx', 'dy', 'influence_radius']):
            self.grid = generate_grid_from_bounds(grid.bounds, 
                                                  grid.dx, grid.dy, 
                                                  timesteps = grid.time, 
                                                  reservoir_layers = grid.reservoir, 
                                                  influence_radius = grid.influence_radius,
                                                  include_mask = True)
            self.nx = self.grid.dims['x']
            self.ny = self.grid.dims['y']
            self.X, self.Y = np.meshgrid(self.x, self.y)
            self.built = True
        else:
            warn('Warning: grid has not enough information to build grid from.')

    def build_from_bounds(self, bounds, dx, dy = None, 
                                  influence_radius = 0):
        """Generates an xarray Dataset, grid, which stores the subsidence data in a geographic and temporal representative format.
        It is generated using lower and upper limits of the bounds as input.

        Parameters
        ----------
        bounds : array-like, int/float
            An array-like object with 4 values.
            [0] lower x, [1] lower y, [2] upper x, [3] upper y
        dx : float/int, optional
            Distance between grid nodes along the x-axis in m for each reservoir. 
            The default is None. Raises exception when None.
        dy : float/int, optional
            Distance between grid nodes along the y-axis in m. The default is 
            None. Defaults to dx if None, if dx is None, raises Exception.
        influence_radius : float/int, optional
            Distance from which the subsidence is set to 0 in m. The default 
            is None. Raises exception when None.

        Sets
        -------
        grid : xarray Dataset
        """
        self.set_bounds(bounds = bounds)
        self.set_dxyradius(self, dx, dy = dy, influence_radius = influence_radius)
        
        self.build_grid()

    # Calculate
    
    def calculate_subsidence(self, _print = True):
        """Determine subsidence per reservoir over the grid and store results 
        in the grid xarray object (y, x, reservoir, time) as data variable
        'subsidence'. Return result from SubsidenceModel object using
        SubsidenceModel.subsidence.
        
        Returns
        -------
        xr.DataSet
            xarray dataset with the coordinates of the grid (dimensions: y, x, reservoir, time) 
            and the subsidence (in m) represented in that grid.
        """
        if _print: print(f'Calculating subsidence, model: {self.name}')
        if not hasattr(self.grid, 'reservoir_mask'):
            raise Exception('Reservoir mask not set, run mask_reservoirs before calculating')
        if not self.hasattr('compaction'):
            raise Exception('No compaction has been set/calculated, determine compaction before calculating subsidence.')
        if not self.hasattr('depths'):
            raise Exception('Reservoir depth has not been set. Use set_depth before calculating subsidence.')
        
        self.check_influence_radius()
        
        # self.add_kernels()
        self.kernel = _SubsidenceKernel.InfluenceKernel(self._influence_radius, self._dx)
        
        if self._subsidence_model_type.lower().startswith('nucleus'):
            self.kernel.nucleus(self.depths, self.depth_to_basements, v = self.poissons_ratios)
        elif self._subsidence_model_type.lower() == 'knothe':
            self.kernel.knothe(self.depths, self.knothe_angles)
            
        self.grid['subsidence'] = -xr.apply_ufunc(
            _utils.convolve,
            self.grid.compaction,
            self.kernel.ds.uz,
            input_core_dims = [['x', 'y'],['kx', 'ky']],
            exclude_dims = set(('kx', 'ky')),
            output_core_dims = [['x', 'y']],
            vectorize = True,
            ).transpose('y', 'x', 'reservoir', 'time', ...
            )
        
        if _print: print(f'Calculated subsidence, model: {self.name}')
        return self.grid['subsidence']
    
    def calculate_volume(self):
        """Determine volume of the subsidence bowl per reservoir over the grid 
        and store results in the grid xarray object (y, x, reservoir, time) as 
        data variable 'volume'. Return result from SubsidenceModel object using
        SubsidenceModel.volume.
        
        Returns
        -------
        xr.DataSet
            xarray dataset with the coordinates of the grid (dimensions: y, x, reservoir, time) 
            and the volume (in m³) represented in that grid. Volume per grid node.
        """
        if self.hasattr('subsidence'):
            surface_area_node = self.dx * self.dy
            self.grid['volume'] = surface_area_node * -self.subsidence
            return self.grid['volume']
        else:
            raise Exception('Calculate susbsidence before calculating volume.')
    
    def calculate_subsidence_at_points(self, points = None, interpolate = True, _print = True):
        """Determine subsidence per reservoir on set points and store results 
        in the grid xarray object (points, reservoir, time) as the data variable 
        'point_subsidence'. Return result from SubsidenceModel object using
        SubsidenceModel.point_subsidence.
        
        Parameters
        ----------
        interpolate : boolean, optional
            When True, interpolates between previously subsidence over grid.
            When False, it will determine the subsidence on those points
            using the analytical method, which takes approximately 10 times
            longer.
            When True and no subsidence has been calculated in the model, an 
            Exception will occur.
        
        Returns
        -------
        xr.DataSet
            xarray dataset with the coordinates of the timeseries (dimensions: points, reservoir, time) 
            and the subsidence (in m) represented at those points.
        """
        self.assign_point_parameters()
        if _print: print(f'Calculating subsidence at points, model: {self.name}')
       
        if not self.hasattr('points') and points is None:
            warn('Warning: No point objects have been defined in the model, or set as function parameters. No subsidence at points has been calculated.')
            return
        if interpolate and not self.hasattr('subsidence'):
            interpolate = False
            warn('Warning: No subsidence has been calculated to interpolate over, calculating subsidence for points.')
        
        if interpolate:
            if points is None:
                interpolate_points = self.points.coordinates
            elif _utils.is_iterable(points):
                try: 
                    interpolate_points = np.array(points)
                except: 
                    raise Exception(f'Invalid type of points: {type(points)}. Use an iterable with shape mx2, where m is the amount of points and 2 the x- and y-coordinate.')
                if len(interpolate_points.shape) == 1:
                    interpolate_points = interpolate_points[None, :]
                if len(interpolate_points.shape) != 2 or interpolate_points.shape[1] != 2:
                    raise Exception('Invalid indication of points. Use an iterable with shape mx2, where m is the amount of points and 2 the x- and y-coordinate.')
            else:
                warn('Warning: No point objects have been defined in the model, or set as function parameters.')
                return
            x, y = zip(*interpolate_points)
            point_subsidence = -self.grid['subsidence'].interp(
                x = ('z', np.array(x)), 
                y = ('z', np.array(y)))
        else:
            # Not nice code, prone to errors
            if not self.hasattr('compaction'):
                raise Exception('No compaction has been calculated, determine compaction before calculating subsidence.')
            if not hasattr(self.grid, 'reservoir_mask'):
                 raise Exception('Reservoir mask not set, run mask_reservoirs before calculating')
            point_kernels, number_of_points = self.add_point_kernels(points)
            point_subsidence = np.zeros(shape = (number_of_points, self.number_of_reservoirs, self.number_of_steps)) 
            for i in range(number_of_points):
                subsidence = np.zeros(shape = self.shape)
                for reservoir in range(self.number_of_reservoirs): 
                    compaction = self.grid.compaction.isel(reservoir = reservoir)
                    s = point_kernels[i][reservoir].ds.u * compaction
                    subsidence[:, :, reservoir, :] = s.transpose('y','x','time')
                point_subsidence[i, :, :] += subsidence.sum(axis = (0,1))
            
        if _print: print(f'Calculated subsidence at points, model: {self.name}')

        
        if points is None:
            self.grid['point_subsidence'] =  -point_subsidence.rename({'z': 'points'})
            return self.grid['point_subsidence'] 
        else: 
            return point_subsidence
        
    def calculate_subsidence_at_observations(self, interpolate = True, _print = True):
        """Determine subsidence at set observation points and store results 
        in the grid xarray object (observations, reservoir, time) as the
        data variable 'observation_subsidence'. Return result from SubsidenceModel 
        object using SubsidenceModel.observation_subsidence.
        
        Parameters
        ----------
        interpolate : boolean, optional
            When True, interpolates between previously subsidence over grid.
            When False, it will determine the subsidence on those points
            using the analytical method, which takes approximately 10 times
            longer.
            When True and no subsidence has been calculated in the model, an 
            Exception will occur.
        
        Returns
        -------
        xr.DataSet
            xarray dataset with the coordinates of the timeseries (dimensions: observations, reservoir, time) 
            and the subsidence (in m) at the observations.
        """
        self.assign_observation_parameters()
        if _print: print(f'Calculating subsidence at observation, model: {self.name}')

        if not hasattr(self.grid, 'reservoir_mask'):
            raise Exception('Reservoir reservoir layer not set, run mask_from_shapefiles before calculating')
        if not self.hasattr('compaction'):
            raise Exception('No compaction has been calculated, determine compaction before calculating subsidence.')
        if not self.hasattr('observation_points'):
            warn('Warning: No observation points have been set in this model. Subsidence at the location of observations have not been calculated.')
            return
        points = self.observation_points.coordinates
        point_subsidence = self.calculate_subsidence_at_points(points = points, interpolate = interpolate, _print = False)
        point_subsidence = point_subsidence.rename({'z': 'observations'})
        self.grid['observation_subsidence']  = -point_subsidence
        if _print: print(f'Calculated subsidence at observation, model: {self.name}')
        return self.grid['observation_subsidence'] 
    
    def compare_observations(self):
        if not self.hasattr('observation_subsidence'):
            self.calculate_subsidence_at_observations()
            
        if self.hasattr('observation_points'):
            differences = []
            for p in self.observation_points:
                model_value = self.grid['observation_subsidence'].loc[p.name].interp(
                    time = p.time).sum(dim = 'reservoir')
                observation_value = -p.relative
                difference = model_value - observation_value
                differences.append(list(difference.values))
            return differences
        else:
            warn('Warning: No observation points have been set.')
    
    def mse(self, differences):
        return np.mean(np.power(differences, 2))
    
    def mae(self, differences):
        return np.mean(np.abs(differences))
    
    def error(self, method = 'mae'):
        """Get the error based on specified methods (default is mean absolute error)
        and the set observations.

        Parameters
        ----------
        method : str, optional
            Either 'mse' for mean squared error 'mae'for meand absolute error. 
            The default is 'mae'.

        Raises
        ------
        Exception
            When invalid method is entered.

        Returns
        -------
        np.ndarray
            The error.

        """
        differences = self.compare_observations()
        differences = _utils.flatten_ragged_lists2D(differences)
        if method.lower() == 'mse':
            return self.mse(differences)
        elif method.lower() == 'mae':
            return self.mae(differences)
        else: 
            raise Exception(f'Invalid method {method}.')
           
    def calculate_slope(self, reservoir = None, numeric = False, _print = True):
        """Returns the spatial gradient of the subsidence in m/m.

        Parameters
        ----------
        reservoir : int, str or list of int or str, optional
            The index or name of the reservoirs you want to know the slope of. 
            If it is a list, the cumulative slope of all the reservoirs in that 
            list will be determined. The default is None. When None, all reservoirs 
            will be accumulated.
        numeric : bool, optional
            If True, the calculations will be determined numerically 
            (second order central difference scheme). The default is False. When
            False the slope will be determined analytically.
        
        Returns
        -------
        xr.DataSet
            The gradient of the subsidence in m/m for each coordinate, reservoir and timestep.
            The dimentions of the dataset are (y, x, reservoir, time).
            
            The gradient magnitude (slope), the gradient over the 
            x-axis (slope_x) and the y-axis (slope_y)
            will be a part of the xarray data array.
            
        Sets
        ----
        SubsidenceModel.slope
        SubsidenceModel.slope_x
        SubsidenceModel.slope_y
        """
        if _print: print(f'Calculating subsidence gradient, model: {self.name}')
        if numeric:
            if not self.hasattr('subsidence'):
                raise Exception(f'No subsidence has been designed for ModelSubsidence object {self.name}. Run calculate_subsidence and try again.')
            self.grid['slope'] = xr.apply_ufunc(
                _utils.gradient_magnitude,
                self.subsidence,
                self.dx, self.dy,
                input_core_dims = [['y', 'x'], [], []],
                output_core_dims = [['y', 'x']],
                vectorize = True,
                )
        else:
            _reservoir = _plot_utils.reservoir_entry_to_index(self, reservoir)
            self.kernel = _HKernel.InfluenceKernel(self._influence_radius, self._dx)
            self.kernel.nucleus_slope(self.depths, self.depth_to_basements, self.poissons_ratios)
            self.grid['slope_x'] = -xr.apply_ufunc(
                _utils.convolve,
                self.grid.compaction,
                self.kernel.ds.slope_x,
                input_core_dims = [['x', 'y'],['kx', 'ky']],
                exclude_dims = set(('kx', 'ky')),
                output_core_dims = [['x', 'y']],
                vectorize = True,
                ).isel(reservoir = _reservoir
                ).transpose('y', 'x', 'time', ...
                )
                            
            self.grid['slope_y'] = -xr.apply_ufunc(
                _utils.convolve,
                self.grid.compaction,
                self.kernel.ds.slope_y,
                input_core_dims = [['x', 'y'],['kx', 'ky']],
                exclude_dims = set(('kx', 'ky')),
                output_core_dims = [['x', 'y']],
                vectorize = True,
                ).isel(reservoir = _reservoir
                ).transpose('y', 'x', 'time', ...
                )
        
            self.grid['slope'] = np.sqrt(
                self.grid['slope_x']**2 + self.grid['slope_y']**2
                )
        if _print: print(f'Calculated subsidence gradient, model: {self.name}')

        return self.grid['slope']
    
    def calculate_concavity(self, reservoir = None, numeric = False, _print = True):
        """Returns the second order derivative of the subsidence in m²/m.

        Parameters
        ----------
        reservoir : int, str or list of int or str, optional
            The index or name of the reservoirs you want to know the concavity of. 
            If it is a list, the cumulative concavity of all the reservoirs in that 
            list will be determined. The default is None. When None, all reservoirs 
            will be accumulated.
        numeric : bool, optional
            If True, the calculations will be determined numerically 
            (second order central difference scheme). The default is False. When
            False the concavity will be determined analytically.
        
        
        Returns
        -------
        xr.DataSet
            The concavity of the subsidence in m²/m for each coordinate, reservoir and timestep.
            The dimentions of the dataset are (y, x, reservoir, time).
            
            The concavity over the x-axis (concavity_x) and the y-axis (concavity_y)
            will be a part of the xarray data array.
            
        Sets
        ----
        SubsidenceModel.concavity
        SubsidenceModel.concavity_xx
        SubsidenceModel.concavity_xy
        SubsidenceModel.concavity_yx
        SubsidenceModel.concavity_yy
        """
        if _print: print(f'Calculating subsidence concavity, model: {self.name}')
        _reservoir = _plot_utils.reservoir_entry_to_index(self, reservoir)
        if numeric:
            if not self.hasattr('subsidence'):
                raise Exception(f'No subsidence has been designed for ModelSubsidence object {self.name}. Run calculate_subsidence and try again.')
            
            self.grid['concavity'] = xr.apply_ufunc(
                _utils.concavity_magnitude,
                self.subsidence,
                self.dx, self.dy,
                input_core_dims = [['y', 'x'], [], []],
                output_core_dims = [['y', 'x']],
                vectorize = True,
                ).isel(reservoir = _reservoir
                ).sum('reservoir')
        else:
            # if not self.hasattr('kernel'):
            
            self.kernel = _HKernel.InfluenceKernel(self._influence_radius, self._dx)
            self.kernel.nucleus_concavity(self.depths, self.depth_to_basements, v = self.poissons_ratios)
            
            self.grid['concavity_xx'] = -xr.apply_ufunc(
                _utils.convolve,
                self.grid.compaction,
                self.kernel.ds.concavity_xx,
                input_core_dims = [['x', 'y'],['kx', 'ky']],
                exclude_dims = set(('kx', 'ky')),
                output_core_dims = [['x', 'y']],
                vectorize = True,
                ).isel(reservoir = _reservoir
                ).transpose('y', 'x', 'time', ...
                )
                            
            self.grid['concavity_xy'] = -xr.apply_ufunc(
                _utils.convolve,
                self.grid.compaction,
                self.kernel.ds.concavity_xy,
                input_core_dims = [['x', 'y'],['kx', 'ky']],
                exclude_dims = set(('kx', 'ky')),
                output_core_dims = [['x', 'y']],
                vectorize = True,
                ).isel(reservoir = _reservoir
                ).transpose('y', 'x', 'time', ...
                )
            
            self.grid['concavity_yx'] = -xr.apply_ufunc(
                _utils.convolve,
                self.grid.compaction,
                self.kernel.ds.concavity_yx,
                input_core_dims = [['x', 'y'],['kx', 'ky']],
                exclude_dims = set(('kx', 'ky')),
                output_core_dims = [['x', 'y']],
                vectorize = True,
                ).isel(reservoir = _reservoir
                ).transpose('y', 'x', 'time', ...
                )
                            
            self.grid['concavity_yy'] = -xr.apply_ufunc(
                _utils.convolve,
                self.grid.compaction,
                self.kernel.ds.concavity_yy,
                input_core_dims = [['x', 'y'],['kx', 'ky']],
                exclude_dims = set(('kx', 'ky')),
                output_core_dims = [['x', 'y']],
                vectorize = True,
                ).isel(reservoir = _reservoir
                ).transpose('y', 'x', 'time', ...
                )
                            
            self.grid['concavity'] = np.sqrt( 
                self.grid['concavity_xx']**2 + 
                self.grid['concavity_xy']**2 + 
                self.grid['concavity_yx']**2 + 
                self.grid['concavity_yy']**2
                )
        
        if _print: print(f'Calculated subsidence concavity, model: {self.name}')

        return self.grid['concavity']
        
    
    def calculate_subsidence_rate(self, _print = True):
        """Returns the temporal gradient of the subsidence in m/y. 

        Returns
        -------
        xr.DataSet
            The rate of subsidence in m/year for each coordinate, reservoir and timestep.
            The dimentions of the dataset are (y, x, reservoir, time).
        """
        if _print: print(f'Calculating subsidence rate, model: {self.name}')

        if not self.hasattr('subsidence'):
            raise Exception(f'No subsidence has been designed for ModelSubsidence object {self.name}. Run calculate_subsidence and try again.')
        
        timesteps_in_seconds = xr.DataArray(self.timesteps.astype(int)/1e9,
                                          coords = {'time': self.timesteps},
                                          )
        du__dr_per_second = xr.apply_ufunc(
            np.gradient,
            self.subsidence,
            timesteps_in_seconds,
            kwargs = {'axis': -1},
            input_core_dims = [['y', 'x', 'time'], ['time']],
            output_core_dims = [['y', 'x', 'time']],
            vectorize = True,
            dask = 'parallelized',
            output_dtypes = [float],
            )
        
        seconds_in_steps = xr.DataArray(
            np.diff(
                timesteps_in_seconds, 
                prepend = timesteps_in_seconds[0]
                ),
            coords = {'time': self.timesteps},
            )
        
        
        self.grid['subsidence_rate'] = (du__dr_per_second * seconds_in_steps).compute()
        if _print: print(f'Calculated subsidence rate, model: {self.name}')
        return self.grid['subsidence_rate']
        
    def report(self, figures = True):
        """Make a short report with the input variables and subsidence results.
        The report will be saved as a .txt file.

        Parameters
        ----------
        fname : str
            Path to a file. The report will be saved as a .txt file, no mather 
            the file extension put in the fname. Default is None, when None
            the file will be saved in the output folder and the name of the file
            will be report.txt.
            
            Any other paths can be entered to have it stored somewhere else and/or with another name.

        """
        _memory.report(self)
        if figures:
            contour_levels = self.get_contour_levels()
            for step in range(self.number_of_steps):
                if self.timesteps.dtype == np.int64:
                    title = f"Subsidence (cm) - year {self.timesteps[step]}"
                elif np.issubdtype(self.timesteps.dtype, np.datetime64):
                    title = f"Subsidence (cm) - {np.datetime_as_string(self.timesteps, unit = 'D')[step]}"
                _plot_utils.plot_subsidence(self, unit = 'cm', variable = 'subsidence', time = step, title = title, fname = title, contour_levels = contour_levels)
    

    
    