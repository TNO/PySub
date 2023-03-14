"""Stores the SubsidenceModel class, which stores the properties and 
    attributes of the model, together with the functionalities to run
    and plot the model results. User interfaces with this model.
"""
import numpy as np
from PySub import SubsidenceModelBase as _SubsidenceModelBase

from PySub import CompactionModels as _CompactionModels
from PySub import utils as _utils

import warnings
from warnings import warn
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

EPSILON = 1e-10


class SubsidenceModel(_SubsidenceModelBase.SubsidenceModel):
    """Object to contain subsidence modeling data and functionalities.
    
    This object creates an xarray grid (with x and y coördinates) for each 
    reservoir and timestep. Parameters need to be added to define the grid,
    the reservoirs, the timesteps and, optionally, points on which the 
    subsidence will be determined. Each with their own dimensionality.
    
    This class is used inherits common functionality from SubsidenceModelBase.SubsidenceModel.
    In addition to subsidence calculates the compaction.
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
        super().__init__(name, project_folder = project_folder)
        self._compaction_model = None
        self._compaction_model_type = None
        
    @property
    def calc_vars(self):
        return ['compaction', 'subsidence', 'volume', 'slope', 'concavity', 'subsidence_rate']
    
    @property
    def vars_to_calculate(self):
        return ['subsidence_model_type', 'compaction_model_type', 
                'depth_to_basements', 'depths', 'thickness', 'compaction_coefficients',
                'knothe_angles', 'tau', 'reference_stress_rates', 
                'cmref', 'b', 'density', 'pressures']
 
    @property
    def vars_to_build(self):
        return ['reservoirs', 'timesteps', 'points', 'observation_points',
                'dx', 'dy', 'influence_radius', 'shapes', 'bounds']
    
    @property
    def compaction_model(self):
        """Property: The CompactionModel object used for each reservoir.
        Can be adjusted with SubsidenceModel.set_compaction_model() function.
        
        Returns:
        -------
        list, PySub.CompactionModels objects
            The types of compaction models as defined in PySub.CompactionModels 
            for each reservoir.
            Available compaction models: # TODO: keep updated with added methods
            - linear (LineraCompaction object)
            - time-decay (TimeDecayCompaction object)
            - ratetype (RateTypeCompaction object)
            The returned list has the same length as the number of reservoirs in 
            the model.
        """
        return self._compaction_model
    
    @property
    def compaction_coefficients(self):
        """Property: Uniaxial compaction coefficient (Cm) in bar for each 
        reservoir. Used for calculating subsidence. Not to be confused with 
        the Cmd property of the RateTypeCompactionModel object, which is just used
        for calculating the compaction.
        
        Can be adjusted with SubsidenceModel.set_compaction_coefficients() 
        function.
            
        Returns:
        -------  
        1D or 3D list, float
            Uniaxial compaction coefficient (Cm) in bar for each reservoir.
            The returned list has the same length as the number of reservoirs in 
            the model. When entered as a grid, there will be a second and third 
            dimension returned. These dimensions have the order, y, x and reservoirs.
        """
        return self._fetch('compaction_coefficients') 
    
    @property
    def thickness(self):
        """Property: The vertical thickness of each reservoir in m.
        Can be adjusted with SubsidenceModel.set_thickness() function.
            
        Returns:
        -------  
        1D or 3D list, float /int
            The thickness of each reservoir in m.
            The returned list has the same length as the number of reservoirs in 
            the model. When entered as a grid, there will be a second and third 
            dimension returned. These dimensions have the order, y, x and reservoirs.
        """
        return self._fetch('thickness') 
    
    @property
    def pressures(self):
        """Property: For each reservoir, the development of the reservoir 
        pressure over time.
        
        Returns:
        -------  
        np.ndarray, float/int:
            2D or 4D list with the shape of (y, x,) m, n, where m is the number of reservoirs, and
            n is the number of timesteps. Contains the pressure development over
            time for eacht reservoir in bar.
        """  
        return self._fetch('pressures')  
    
    @property
    def compaction_model_type(self):
        """Property: The compaction model for this model.
        Can be adjusted with SubsidenceModel.set_compaction_model_type() function.
            
        Returns:
        -------  
        str
            The model type, either linear, time-decay or ratetype.
        """
        return self._compaction_model_type
    
    @property
    def tau(self):
        """Property : returns the tau values (in s) of the time-decay model. 

        Returns
        -------
        1D or 3D list, float
            tau value in bar for each reservoir. Used in the time-decay model.
            The returned list has the same length as the number of reservoirs in 
            the model. When entered as a grid, there will be a second and third 
            dimension returned. These dimensions have the order, y, x and reservoirs.

        """
        return self._fetch('tau') 
    
    @property
    def reference_stress_rates(self):
        """Property : returns the set reference stress rates in bar/s for each reservoir
    
        Returns
        -------
        1D or 3D list, float
            Reference stress rate in bar/s for each reservoir. Used in the ratetype compaction model.
            The returned list has the same length as the number of reservoirs in 
            the model. When entered as a grid, there will be a second and third 
            dimension returned. These dimensions have the order, y, x and reservoirs.

        """
        return self._fetch('reference_stress_rates') 
    
    @property
    def b(self):
        """Property : returns the set dimensionless b creep coefficient for each reservoir.
        
        Returns
        -------
        1D or 3D list, float
            Creep coefficient for each reservoir. Used in the ratetyp ecompaction model.
            The returned list has the same length as the number of reservoirs in 
            the model. When entered as a grid, there will be a second and third 
            dimension returned. These dimensions have the order, y, x and reservoirs.

        """
        return self._fetch('b') 
    
    @property
    def cmref(self):
        """Property : returns the set reference compaction coefficients in 1/bar for each reservoir
        

        Returns
        -------
        1D or 3D list, float
            Reference compaction coefficients for each reservoir. Used in the ratetype compaction model.
            The returned list has the same length as the number of reservoirs in 
            the model. When entered as a grid, there will be a second and third 
            dimension returned. These dimensions have the order, y, x and reservoirs.

        """
        return self._fetch('cmref') 
    
    @property
    def density(self):
        """Property : returns the set bulk rock density kg/m³ for each reservoir.
                

        Returns
        -------
        1D or 3D list, float
            Density of the bulk rock above the reservoir, for each reservoir. 
            Used in the ratetype compaction model.
            The returned list has the same length as the number of reservoirs in 
            the model. When entered as a grid, there will be a second and third 
            dimension returned. These dimensions have the order, y, x and reservoirs.

        """
        return self._fetch('density') 
    
    def _check_compaction_paramaters(self):
        action = 'assign compaction parameters to grid'
        self._check_for_grid_attributes('pressures', action = action)
        self._check_for_grid_attributes('depths', action = action)
        self._check_for_grid_attributes('thickness', action = action)
        self._check_for_grid_attributes('compaction_coefficients', action = action)
    
    def _check_compaction_model(self, name_model):
        if isinstance(name_model, str):
            if name_model not in ['linear', 'ratetype', 'time decay']:
                raise Exception(f"{name_model} not recognized as valid compaction model. Choose from: 'linear', 'ratetype', 'time decay'")
        else:
            raise Exception("set the compaction model using a string with either 'linear', 'ratetype', 'time decay'")
    
    def set_parameters(self, dx = None, dy = None, influence_radius = None, 
                       compaction_model = None, subsidence_model = 'nucleus of strain', 
                       knothe_angles = None, tau = None, reservoirs = None, 
                       shapefile_paths = None, depths = None, 
                       depth_to_basements = None, poissons_ratios = None,
                       reference_stress_rates = None, 
                       density = None, 
                       cmref = None, 
                       b = None,
                       compaction_coefficients = None, thickness = None,
                       timesteps = None, pressures = None,
                       bounds = None,):
        """Set the parameters necesary to build and run a model.

        Parameters
        ----------
        dx : float/int, optional
            Distance between grid nodes along the x-axis in m. The default is 
            None. Raises exception when None.
        dy : float/int, optional
            Distance between grid nodes along the y-axis in m. The default is 
            None. When None, defaults to dx.
        influence_radius : float/int, optional
            Distance from which the subsidence is set to 0 in m. The default 
            is None. Raises exception when None.
        compaction_model : list, str, optional
            Can ba a strin for the model name to be used for all reservoirs, or
            a list of string with the model type to be used for each reservoir.
            The list must have the same length as the number of reservoirs in 
            the model.
            
            The types of compaction models as defined in 
            PySub.CompactionModels for each reservoir: # TODO: keep updated with added methods
                - linear
                - time-decay
                - ratetype
            The default is None. Raises Exception when None.
        subsidence_model : str, optional
            Method of subsidence of the model. Currently available:
            - Nucleus of strain, Van Opstal 1974
            - Knothe, Stroka et al. 2011. 
            The default is 'nucleus of strain'.
        tau : list, float/int, optional
            The time-decay constant for the time-decay method for each 
            reservoir in seconds of delay. The list must have the same length 
            as the number of reservoirs in the model. The default is None.
        knothe_angles : list, float/int, optional
            The influence angle in degrees for the knoth subsidence method for each 
            reservoir. The default is None.The list must have the same length 
            as the number of reservoirs in the model.
        reservoirs : list, str, optional
            The names of each reservoir. The default is None. The list must have 
            the same length as the number of reservoirs in the model.
        shapefile_paths : list, str, optional
            The location to the shapefiles for each reservoir. 
            The list must have the same length as the number of reservoirs in 
            the model.The default is None.
        depths : list float/int, optional
            Depths to the top of each reservoirs in m. The list must have 
            the same length as the number of reservoirs in 
            the model. The default is None. Raises Exception when None.
        depth_to_basements : list, float/int, optional
            Depth to the rigid basement for the van Opstal nucleus of strain 
            method in m. The list must have the same length as the number of reservoirs in 
            the model. The default is None. Raises Exception when None.
        reference_stress_rates : list, float/int, optional
            Reference stress rates in bar/year. The list must have the 
            same length as the number of reservoirs in the model. Raises Exception 
            when None and a ratetype compaction model is used.
        density : list, float/int, optional
            Bulk density of the ground above the reservoir in kg/m³. 
            The list must have the same length as the number of reservoirs in 
            the model. Raises Exception when None and a ratetype compaction model 
            is used.
        cmd : list, float/int, optional
            Direct compaction coefficient in 1/bar used for the ratetype compaction model. 
            The list must have the same length as the number of reservoirs in 
            the model. Raises Exception when None and a ratetype compaction model is used.
        cmref : list, float/int, optional
            Reference compaction coefficient in 1/bar used for the ratetype compaction model. 
            The list must have the same length as the number of reservoirs in 
            the model. Raises Exception when None and a ratetype compaction model is used.
        b : list, float/int, optional
            Dimensionless constant for the stiffer reaction of sandstone over a 
            specific loading rate. The list must have the same length as the number 
            of reservoirs in the model. Raises Exception when None and a ratetype 
            compaction model is used.
        compaction_coefficients : list, floats, optional
            Uniaxial compaction coefficient (Cm) in 1/bar. The list must have 
            the same length as the number of reservoirs in the model. The default 
            is None. Raises Exception when None.
        thickness : list, float/int, optional
            Thickness of each reservoir in m. The list must have the same length 
            as the number of reservoirs in the model.The default is None. Raises 
            Exception when None.
        timesteps : list, np.datetime64, optional
            The timestamps of each step in time. These need to be of equal 
            step length. Per year would be ['1990', '1991', etc.]. The default 
            is None. Raises Exception when None.
        pressures : np.ndarray or pandas.DataFrame, float/int, optional
            Can be a 2D pandas DataFrame.    
            Can be a 2D, 3D or 4D numpy array. 
            2D: array with the shape of (m, n), where m is the number of reservoirs, 
                (SubsidenceModel.number_of_reservoirs) and n is the number of timesteps 
                (SubsidenceModel.number_of_steps). Contains the pressure development 
                over time for eacht reservoir in bar. The pressures will be 
                homogonous over each reservoir.
            3D: array with the shape (i, j, n), where i is the number of nodes
                of the grid along the y-axis (SubsidenceModel.ny), j is the number
                of nodes of the grid along the x-axis (SubsidenceModel.nx) and n
                is the number of timesteps (SubsidenceModel.number_of_timesteps).
                It is assumed there is only 1 reservoir in this case.
            4D: array with the shape (i, j, m, n), where i is the number of nodes
                of the grid along the y-axis (SubsidenceModel.ny), j is the number
                of nodes of the grid along the x-axis (SubsidenceModel.nx), m is
                the number of reservoirs (SubsidenceModel.number_of_reservoirs) and n
                is the number of timesteps (SubsidenceModel.number_of_steps).
            The default is None. Raises Exception when None.
        bounds : array-like, int/float, optional
            An array-like object with 4 values representing the corners of the 
            model. [0] lower x, [1] lower y, [2] upper x, [3] upper y.
            Default is None. When None, it will check for any set shapes.
            If there are no shapes, returns an exception.
            
        """
        
        self.set_reservoirs(reservoirs)
        self.set_dxyradius(dx, dy = dy, influence_radius = influence_radius)
        self.set_shapes(shapefile_paths)
        self.set_bounds(bounds = bounds)
        self.set_timesteps(timesteps)
        self.build_grid()
        self.set_compaction_model_parameters(compaction_model, 
                                  compaction_coefficients = compaction_coefficients,
                                  tau = tau,
                                  reference_stress_rates = reference_stress_rates, 
                                  density = density, 
                                  cmref = cmref, 
                                  b = b)
        self.set_subsidence_model_parameters(subsidence_model, knothe_angles = knothe_angles,
                                  depth_to_basements = depth_to_basements, poissons_ratios = poissons_ratios)
        self.set_depths(depths)
        self.set_thickness(thickness)
        
        self.set_pressures(pressures)
    
    def set_pressures(self, pressures, layer = None):
        """Sets the pressures as a part of the model, perform checks.
        
        Parameters
        ----------
        pressures : np.ndarray or pandas.DataFrame, float/int, optional
            Can be a 2D pandas DataFrame.    
            Can be a 1D, 2D, 3D or 4D numpy array. 
            1D: array with same length as the number of timesteps. It is assumed
                to be used for 1 reservoir only. 
            2D: array with the shape of (m, n), where m is the number of reservoirs, 
                (SubsidenceModel.number_of_reservoirs) and n is the number of timesteps 
                (SubsidenceModel.number_of_steps). Contains the pressure development 
                over time for eacht reservoir in bar. The pressures will be 
                homogonous over each reservoir.
            3D: array with the shape (i, j, n), where i is the number of nodes
                of the grid along the y-axis (SubsidenceModel.ny), j is the number
                of nodes of the grid along the x-axis (SubsidenceModel.nx) and n
                is the number of timesteps (SubsidenceModel.number_of_timesteps).
                It is assumed there is only 1 reservoir in this case.
            4D: array with the shape (i, j, m, n), where i is the number of nodes
                of the grid along the y-axis (SubsidenceModel.ny), j is the number
                of nodes of the grid along the x-axis (SubsidenceModel.nx), m is
                the number of reservoirs (SubsidenceModel.number_of_reservoirs) and n
                is the number of timesteps (SubsidenceModel.number_of_steps).
            list, str: A path to a raster file with the pressures over time. Can only be 3D
                (per reservoir a path).
                The default is None. Raises Exception when None.
        layer : int
            If "pressures" is a path or list of paths to a .tif raster file, 
            layers is a (set of) integer value(s) indicating the index of the layer 
            storing the specific data in the .tif raster file. 
            The default is None.
            """
        
        if isinstance(pressures, str): # pressures as a single file
            fname = pressures
            self.project_folder.write_to_input(fname, rename = 'pressures')
            if fname.endswith('tif'):
                loaded_pressures = self.load_from_raster(fname, layer = layer)
            elif fname.endswith(('.txt', '.csv')):
                loaded_pressures = self.load_from_csv(fname)
            number_of_bands = loaded_pressures.shape[-1]
            if (number_of_bands == self.number_of_steps * self.number_of_reservoirs): # the bands are for each reservoir*timestep, works also with just 1 reservoir.
                transposed_pressures = np.zeros(shape = (self.ny, self.nx, self.number_of_reservoirs, self.number_of_steps))
                for i in range(self.number_of_reservoirs):
                    for j in range(self.number_of_steps):
                        old_index = i * self.number_of_steps + j
                        transposed_pressures[:, :, i, j] = loaded_pressures[:, :, old_index]          
                pressures = transposed_pressures
            else:
                raise Exception('The number of bands in the .tif file or the number of dimensions in the .csv file are not the right shape. The bands of the rasterfile should be equal to the amount of timesteps (1 reservoir) or equal to the number of reservoirs times the number of timestep (where each index is: reservoir*number_of_steps + step)')
        
        elif _utils.is_list_of_strings(pressures): # pressures as multiple files
            fname = pressures
            number_of_reservoirs = len(fname)
            pressures = np.zeros(self.shape)
            for i, f, r in zip(range(number_of_reservoirs), fname, self.reservoirs):
                if f.endswith('.csv'):
                    loaded_pressures = self.load_from_csv(f)
                    pressures[:, :, i, :] = loaded_pressures
                
                    self.project_folder.write_to_input(f, rename = f'{r} pressures')
         
        elif _utils.is_iterable(pressures): # pressures as a list or numpy array
            if not _utils.is_list_of_strings(pressures):
                try:
                    pressures = np.array(pressures)
                except:
                    raise Exception('Pressures should be a 2, 3- or 4-dimensional, non-ragged array-like object.')
        
        if (pressures < -EPSILON).any():
            raise Exception('Invalid values (negative values) encountered in pressures.')
        
        # Ensure correct dimensions 
        if len(pressures.shape) == 1: 
            pressures = pressures.reshape((1, len(pressures)))
            dim = ('reservoir', 'time')
            self._check_dimND('pressure', pressures, dims_equal = ('reservoirs', 'timesteps'))
        elif len(pressures.shape) == 2:
            dim = ('reservoir', 'time')
            self._check_dimND('pressure', pressures, dims_equal = ('reservoirs', 'timesteps'))
        elif len(pressures.shape) == 3: # Only 1 reservoir
            dim = ('y', 'x', 'reservoir', 'time')
            self._check_dimND('pressure', pressures[:, :, None, :], dims_equal = ('y', 'x', 'reservoirs', 'timesteps'))
        elif len(pressures.shape) == 4:
            dim = ('y', 'x', 'reservoir', 'time')
            self._check_dimND('pressure', pressures, dims_equal = ('y', 'x', 'reservoirs', 'timesteps'))
        else:
            raise Exception(f'Entry pressures has invalid number of dimensions {len(pressures.shape)}.')
        
        self.assign_data_variable('pressures', dim, pressures)
    
    def _check_thickness(self):
        if self.hasattr('depths') and self.hasattr('depth_to_basements') and self.hasattr('thickness'):
            check = self.depths + self.thickness > self.depth_to_basements
            if check.any():
                wrong_reservoirs = np.where(check)[0]
                if self.hasattr('reservoirs'):
                    wrong_reservoirs = np.array(self.reservoirs)[wrong_reservoirs]
                warn(f'\nWarning: The reservoirs {wrong_reservoirs} have a depth and thickness that exceed the depth of the rigid basement.')
    
    def set_thickness(self, thickness, layer = None):
        """Sets the thickness as a part of the model, perform checks.
        thickness : list, float/str
            Thickness of each reservoir in m. The list must have the same length 
            as the number of reservoirs in the model. The values in that list must be 
            the values for each reservoir (and are distributed uniformly over each 
            reservoir), or a path to a file containing information on the spatial 
            distribution of this value (.tif raster file).
            The default is None. Raises Exception when None.
        layers : int, optional
            If "thickness" is a path to a .tif raster file, layers is a (set of) integer 
            value(s) indicating the index of the layer storing the specific
            data in the .tif raster file. 
            The default is None.
        """
        _utils._check_low_high(thickness, 'thickness', EPSILON, 10000)
        self._check_thickness()
        self.set_1D_or_2D('thickness', thickness, layer = layer)
        
    
    def set_compaction_coefficients(self, compaction_coefficients, layer = None):
        """Sets the compaction coefficients as a part of the model, perform checks.
        compaction_coefficients : list, floats/str
            Uniaxial compaction coefficient (Cm) in 1/bar. The list must have the same length 
            as the number of reservoirs in the model. The values in that list must be 
            the values for each reservoir (and are distributed uniformly over each 
            reservoir), or a path to a file containing information on the spatial 
            distribution of this value (.tif raster file).
            The default is None. Raises Exception when None.
        layers : int, optional
            If "compaction_coefficients" is a path to a .tif raster file, layers is 
            a (set of) integer value(s) indicating the index of the layer storing 
            the specific data in the .tif raster file. 
            The default is None.
        """
        _utils._check_low_high(compaction_coefficients, 'compaction_coefficients', EPSILON, 1)
        self.set_1D_or_2D('compaction_coefficients', compaction_coefficients, layer = layer)
    
    def set_depth_to_basements(self, depth_to_basements):
        """Sets the depths of the basements as a part of the model, perform checks.
        depth_to_basements : list, float
            Depth to the rigid basement for the van Opstal nucleus of strain 
            method in m. The list must have the same length as the number of reservoirs in 
            the model. The default is None.
        """
        _utils._check_low_high(depth_to_basements, 'depth_to_basements', EPSILON, 100000)
        if _utils.isnan(depth_to_basements):
            depth_to_basements = None
            self.depth_to_basements = depth_to_basements
        else:
            self._check_dim1D('depth to basements', depth_to_basements)
            self.set_1D_or_2D('depth_to_basements', depth_to_basements)
            self._check_thickness()
            self.drop_from_grid(depth_to_basements, 'depth_to_basements')
    
    def set_depths(self, depths, layer = None):
        """Sets the depths as a part of the model and performs checks.
        depths : list, float, optional
            Depths to the top of each reservoirs in m. The list must have 
            the same length as the number of reservoirs in 
            the model. The default is None. Raises Exception when None.
        layers : int, optional
            If "depths" is a path to a .tif raster file, layers is 
            a (set of) integer value(s) indicating the index of the layer storing 
            the specific data in the .tif raster file. 
            The default is None.
        """
        self.set_1D_or_2D('depths', depths, layer = layer)
        self._check_thickness()
    
    def _set_linear_compaction_model(self):
        self._compaction_model = _CompactionModels.LinearCompaction()
                
    def _set_ratetype_compaction_model(self):
        for reservoir in range(self.number_of_reservoirs):
            for var in [self.reference_stress_rates[reservoir], 
                        self.density[reservoir], 
                        self.compaction_coefficients[reservoir], 
                        self.cmref[reservoir], 
                        self.b[reservoir]]:
                if not _utils.not_None_or_empty(var):
                    raise Exception('Compaction parameters for the Ratetype Compaction Model cannot be None.')
        self._compaction_model = _CompactionModels.RateTypeCompaction()
    
    def _set_timedecay_compaction_model(self):
        if (self.tau is None or 
            not _utils.is_iterable(self.tau) or
            np.isnan(np.array(self.tau)).any()):
            raise Exception(f'When the time-decay method is chosen, variable tau needs to be an array-like object with entries equal to the amount of reservoirs in the model. Current entry: {self.tau}')
        
        if len(self.tau.shape) == 1:
            if (self.tau <= -EPSILON).any():
                warn('Warning: Time decay model parameter Tau of <=0 encountered. Calculating compaction with linear compaction model.')
                self._compaction_model = _CompactionModels.LinearCompaction()
            else:
                self._compaction_model = _CompactionModels.TimeDecayCompaction()
        else:
            if (np.max(self.tau, axis = (0,1)) <= -EPSILON).any():
                warn('Warning: Time decay model parameter Tau of <=0 encountered. Calculating compaction with linear compaction model.')
                self._compaction_model = _CompactionModels.LinearCompaction()
            else:
                self._compaction_model = _CompactionModels.TimeDecayCompaction()
    
    def set_tau(self, tau, layers = None):
        """Set the compaction variable tau used in the time-decay compaction model.

        Parameters
        ----------
        tau : list, float/int/str, optional
            The time-decay constant for the time-decay method for each 
            reservoir in seconds of delay. The list must have the same length 
            as the number of reservoirs in the model. The values in that list must be 
            the values for each reservoir (and are distributed uniformly over each 
            reservoir), or a path to a file containing information on the spatial 
            distribution of this value (.tif raster file).
            Raises Exception when None and name_model = 'time-decay'.
        layers : int, optional
            If the input argument is a path to a .tif raster file, layers is 
            a (set of) integer value(s) indicating the index of the layer storing 
            the specific data in the .tif raster file. 
            The default is None.
        """
        if tau is not None:    
            _utils._check_low_high(tau, 'tau', -EPSILON, 10000 * 365*24*3600)
            self.set_1D_or_2D('tau', tau, layer = None)
    
    def set_reference_stress_rates(self, reference_stress_rates, layers = None):
        """Set the compaction variable reference_stress_rates used in the ratetype
        compaction model.

        Parameters
        ----------
        reference_stress_rates: list, float
            Reference stress rates in bar/year.The list must have the same length 
            as the number of reservoirs in the model. The values in that list must be 
            the values for each reservoir (and are distributed uniformly over each 
            reservoir), or a path to a file containing information on the spatial 
            distribution of this value (.tif raster file).
            Raises Exception when None and name_model = 'ratetype'.
        layers : int, optional
            If the input argument is a path to a .tif raster file, layers is 
            a (set of) integer value(s) indicating the index of the layer storing 
            the specific data in the .tif raster file. 
            The default is None.
        """
        if reference_stress_rates is not None:
            _utils._check_low_high(reference_stress_rates, 'reference_stress_rates', -EPSILON, 10)
            self.set_1D_or_2D('reference_stress_rates', reference_stress_rates, layer = None)
            
    def set_density(self, density, layers = None):
        """Set the compaction variable density used in the ratetype
        compaction model.

        Parameters
        ----------
        density : list, int/float
            Bulk density of the ground above the reservoir in kg/m³. 
            The list must have the same length 
            as the number of reservoirs in the model. The values in that list must be 
            the values for each reservoir (and are distributed uniformly over each 
            reservoir), or a path to a file containing information on the spatial 
            distribution of this value (.tif raster file).
            Raises Exception when None and name_model = 'ratetype'.
        layers : int, optional
            If the input argument is a path to a .tif raster file, layers is 
            a (set of) integer value(s) indicating the index of the layer storing 
            the specific data in the .tif raster file. 
            The default is None.
        """
        if density is not None:
            _utils._check_low_high(density, 'density', EPSILON, 10000)
            self.set_1D_or_2D('density', density, layer = None)
            
    def set_cmref(self, cmref, layers = None):
        """Set the compaction variable reference compaction coefficient used in 
        the ratetype compaction model.

        Parameters
        ----------
        cmref : list, float/int, optional
            Reference compaction coefficient in 1/bar used for the ratetype compaction model. 
            The list must have the same length 
            as the number of reservoirs in the model. The values in that list must be 
            the values for each reservoir (and are distributed uniformly over each 
            reservoir), or a path to a file containing information on the spatial 
            distribution of this value (.tif raster file).
            Raises Exception when None and name_model = 'ratetype'.
        layers : int, optional
            layers : int, optional
            If the input argument is a path to a .tif raster file, layers is 
            a (set of) integer value(s) indicating the index of the layer storing 
            the specific data in the .tif raster file. 
            The default is None.
        """
        if cmref is not None:
            _utils._check_low_high(cmref, 'cmref', EPSILON, 1)
            self.set_1D_or_2D('cmref', cmref, layer = None)
            
    def set_b(self, b, layers = None):
        """Set the compaction creep coefficient used in the ratetype compaction model.

        Parameters
        ----------
        b : list, float/int, optional
            Dimensionless constant for the stiffer reaction of sandstone over a 
            specific loading rate. The list must have the same length 
            as the number of reservoirs in the model. The values in that list must be 
            the values for each reservoir (and are distributed uniformly over each 
            reservoir), or a path to a file containing information on the spatial 
            distribution of this value (.tif raster file). Raises Exception 
            when None and a ratetype compaction model is used.
        layers : int, optional
            layers : int, optional
            If the input argument is a path to a .tif raster file, layers is 
            a (set of) integer value(s) indicating the index of the layer storing 
            the specific data in the .tif raster file. 
            The default is None.
        """
        if b is not None:
            _utils._check_low_high(b, 'b', EPSILON, 1)
            self.set_1D_or_2D('b', b, layer = None)
            
    def set_compaction_model_type(self, compaction_model):
        self._check_compaction_model(compaction_model)
        if type(compaction_model) == str: 
            self._compaction_model_type = compaction_model
        
    def set_compaction_model_parameters(self, name_model, 
                             compaction_coefficients = None,
                             tau = None,
                             reference_stress_rates = None, 
                             density = None, 
                             cmref = None, 
                             b = None):
        """Sets the CompactionModel objects as a part of the model and performs checks.
        
        Parameters
        ----------
        name_model : list, str, optional
            List of names with the type of model used for determining 
            compaction for each reservoir. The default is None. Raises 
            Exception when None.
            The types of compaction models as defined in 
            PySub.CompactionModels for each reservoir: # TODO: keep updated with added methods
                - linear
                - time-decay
                - ratetype
        compaction_coefficients : list, floats, optional
            Uniaxial compaction coefficient (Cm) in 1/bar. The list must have 
            the same length as the number of reservoirs in the model. The default 
            is None. Raises Exception when None.
        tau : list, float/int, optional
            The time-decay constant for the time-decay method for each 
            reservoir in seconds of delay. The list must have the same length 
            as the number of reservoirs in the model. The default is None.
            Raises Exception when None and name_model = 'time-decay'.
        reference_stress_rates: list, float
            Reference stress rates in bar/year. The list must have the 
            same length as the number of reservoirs in the model. Raises Exception 
            when None and a ratetype compaction model is used.
        density : list, int/float
             Bulk density of the ground above the reservoir in kg/m³. 
            The list must have the same length as the number of reservoirs in 
            the model. Raises Exception when None and a ratetype compaction model 
            is used.
        cmref : list, float/int, optional
            Reference compaction coefficient in 1/bar used for the ratetype compaction model. 
            The list must have the same length as the number of reservoirs in 
            the model. Raises Exception when None and a ratetype compaction model is used.
        b : list, float/int, optional
            Dimensionless constant for the stiffer reaction of sandstone over a 
            specific loading rate. The list must have the same length as the number 
            of reservoirs in the model. Raises Exception when None and a ratetype 
            compaction model is used.
        """
        
        self.set_compaction_model_type(name_model)
        self.set_compaction_coefficients(compaction_coefficients)
        self.set_tau(tau)
        self.set_reference_stress_rates(reference_stress_rates)
        self.set_density(density)
        self.set_cmref(cmref)
        self.set_b(b)
        self.set_compaction_model()
    
    def set_compaction_model(self):
        """Sets the CompactionModel objects as self.compaction_model"""
        
        if self.hasattr('compaction_model_type'):
            name_model = self.compaction_model_type
            self._compaction_model = []
            if name_model == 'linear':
                self._set_linear_compaction_model()
                    
            elif name_model == 'ratetype':
                self._set_ratetype_compaction_model()
                    
            elif name_model == 'time decay':
                self._set_timedecay_compaction_model()
            else: 
                raise Exception(f'Compaction model {name_model} is not supported.')
                
    # Calculate
    def calculate_compaction(self, _print = True, time = None):
        """Determine compaction per reservoir and distribute over grid using the reservoir reservoirs.
        Compaction is stored in the grid xarray object ((reservoir, time) and (y, x, reservoir, time)).
        
        The result is stored in the SubsidenceModel properties .compaction_reservoir
        and .compaction_grid
        
        Returns
        -------
        xr.DataSet
            xarray dataset with the coordinates of the grid (dimensions: y, x, reservoir, time) 
            and the compaction (in m3) represented in that grid.
        """
        
        if _print: print(f'Calculating compaction, model: {self.name}')
        self._check_compaction_paramaters()
        if not hasattr(self.grid, 'reservoir_mask'):
            raise Exception('Reservoir reservoir mask not set, run mask_reservoirs before calculating')
        
        compaction = self._compaction_model.compute(self.grid)
        self.grid['compaction'] = compaction
        self.convert_to_grid('compaction')
        if _print: print(f'Calculated compaction, model: {self.name}')
        return self.grid['compaction']
