# -*- coding: utf-8 -*-
"""Moldule storing the MergedModel class, to store the information and functionality 
of different classes of SubsidenceModels.
"""
from PySub import SubsidenceModelBase as _SubsidenceModelBase
from PySub import utils as _utils
from PySub import Points as _Points
from PySub import grid_utils as _grid_utils
from PySub import memory as _memory
import numpy as np
import xarray as xr

def MergedModelSetError(*args, **kwargs):
    raise Exception("MergedModel object doesn't allow setting of attributes.")

class MergedModel(_SubsidenceModelBase.SubsidenceModel):
    """SubsedinceModel type object which is immutable and stores the input of multiple 
    models, disregarding their secundary type (Gas, BucketEnsemble or Cavern).
    """
    def __init__(self, name, project_folder = None):
        super().__init__(name, project_folder = project_folder)
        for attr in list(dir(self)):
            if attr.startswith('set'):
                setattr(self, attr, MergedModelSetError)
    
    def __getattr__(self, input):
        if isinstance(input, str):
            if input.startswith('set_'):
                raise AttributeError('MergedModel object cannot set attributes.')
            else:
                return getattr(self, input)
    
    @property
    def calc_vars(self):
        return ['volume', 'slope', 'concavity', 'subsidence_rate']
    
    @property
    def vars_to_calculate(self):
        return []
         
    @property
    def vars_to_build(self):
        return []
    
    def __str__(self):
        return self.name
    
    
    def calculate_subsidence(self):
        raise Exception('MergedModel objects cannot calculate subsidence.')
    
    def build_grid(self):
        raise Exception('Cannot build a MergedModel object.')
    def mask_reservoirs(self):
        raise Exception('MergedModel reservoir mask is set during merging.')
        
    def calculate_subsidence_at_points(self, points, _print = True):
        if _print: print(f'Calculating subsidence at points, model: {self.name}')
        if not self.hasattr('subsidence'): 
            print('Warning: No subsidence has been set or calucalted. MergedModel cannot calculate subsidence.')
            return
        if not self.hasattr('points') and points is None:
            print('Warning: No point objects have been defined in the model, or set as function parameters. No subsidence at points has been calculated.')
            return
    
        if points is None:
            self.assign_point_parameters()
            interpolate_points = self.points.coordinates
        elif _utils.is_iterable(points):
            try: 
                interpolate_points = np.array(points)
            except: 
                raise Exception(f'Invalid type of points: {type(points)}. Use an iterable with shape mx2, where m is the amount of points and 2 the x- and y-coordinate.')
            if len(interpolate_points.shape) != 2:
                if interpolate_points.shape != (2,):
                    raise Exception('Invalid indication of points. Use an iterable with shape mx2, where m is the amount of points and 2 the x- and y-coordinate.')
                else:
                    interpolate_points = interpolate_points[np.newaxis, :]
            if interpolate_points.shape[-1] != 2:
                raise Exception('Invalid indication of points. Use an iterable with shape mx2, where m is the amount of points and 2 the x- and y-coordinate.')
                
        else:
            print('Warning: No point objects have been defined in the model, or set as function parameters.')
            return
        number_of_points = len(interpolate_points)
        x, y = zip(*interpolate_points)
        point_subsidence = []
        for i, p in enumerate(interpolate_points):
            point_subsidence.append(np.array(self.grid['subsidence'].interp(x = x[i], y = y[i])))
        point_subsidence = -np.array(point_subsidence)    
            
        if _print: print(f'Calculated subsidence at points, model: {self.name}')

        
        if points is None:
            self.assign_data_variable('point_subsidence', ('points', 'reservoir', 'time'), -point_subsidence)
            return self.grid['point_subsidence'] 
        else: 
            return point_subsidence
        
    def calculate_subsidence_at_observations(self, _print = True):
        self.assign_observation_parameters()
        if _print: print(f'Calculating subsidence at observation, model: {self.name}')

        if not self.hasattr('observation_points'):
            print('Warning: No observation points have been set in this model. Subsidence at the location of observations have not been calculated.')
            return
        points = self.observation_points.coordinates
        point_subsidence = self.calculate_subsidence_at_points(points = points, _print = False)
        self.assign_data_variable('observation_subsidence', ('observations', 'reservoir', 'time'), -point_subsidence)
        if _print: print(f'Calculated subsidence at observation, model: {self.name}')
        return self.grid['observation_subsidence'] 


def merge(list_of_models, variables = [], dx = 50, project_folder = None):
    """Merge models as a list into one model.

    Parameters
    ----------
    list_of_models : list, SubsidenceModel type objects
        A list with the odels that will be merged into 1.
    variables : list, str, optional
        A list with the variable names you want to merge. The variables must be 
        common between all models. The default is [], which indicates all common 
        variables will be merged.  
    dx : float, optional
        The cell size of the merged model. The default is 50.

    Raises
    ------
    Exception
        When invalid input is given.

    Returns
    -------
    MergedModel
        SubsidenceModel type object which is immutable.

    """
    if _utils.is_iterable(list_of_models):
        if len(list_of_models) == 1:
            return list_of_models[0]
        elif len(list_of_models) == 0:
            raise Exception('Empty list.')
        else:
            
            for i in range(len(list_of_models) - 1):
                if i == 0:
                    merged = _merge(list_of_models[i], list_of_models[i+1], variables = variables, dx = dx)
                else:
                    merged = _merge(merged, list_of_models[i+1], variables = variables, dx = dx)
        merged.set_project_folder(project_folder)
        return merged
    else:
        raise Exception(f'Invalid input type {type(list_of_models)}')

def _merge(model1, model2, variables = [],
          dx = 50):
    if _utils.is_number(dx):
       if dx <= 0:
           raise Exception('dx must be a number higher than 0.')
    else:
        raise Exception('dx must be a number higher than 0.')
     
    if _utils.isSubsidenceModel(model1):
        _model1 = model1.grid
    elif isinstance(model1, xr.Dataset):
        _model1 = model1
    else:
        raise Exception('Merge only supports SubsidenceModel types and xarrays with at least the variables x and y') 
    
    
    if _utils.isSubsidenceModel(model2):
        _model2 = model2.grid
    elif isinstance(model2, xr.Dataset):
        _model2 = model2
    else:
        raise Exception('Merge only supports SubsidenceModel types and xarrays with at least the variables x and y')

    
    if isinstance(variables, str):
        variables = [variables]
    elif _utils.is_list_of_strings(variables):
        pass
    elif _utils.is_iterable(variables):
        if len(variables) == 0:
            variables = [var for var in list(_model2.variables.keys()) if var in  list(_model1.variables.keys())]
            variables = [var for var in variables if var not in list(_model2.coords)]
    else:
        raise Exception('Variables can only be entered as a string or list of strings')
    
    doubles = [reservoir for reservoir in list(_model2.reservoir.values) if reservoir in list(_model1.reservoir.values)]
    if len(doubles) > 0:
        print(f'Warning: Reservoir {doubles} occurs in both models. Reservoir from second model argument is chosen.')
    
    timesteps = np.unique(list(_model2.time.values) + list(_model1.time.values))
    unique_x, unique_y = (np.unique(list(_model2.x.values) + list(_model1.x.values)),
                          np.unique(list(_model2.y.values) + list(_model1.y.values)))
    min_x, max_x = np.min(unique_x), np.max(unique_x)
    min_y, max_y = np.min(unique_y), np.max(unique_y)
    x = _utils.stepped_space(min_x, max_x, dx)
    y = _utils.stepped_space(min_y, max_y, dx)
    
    merged_coords = {
            'time': timesteps,
            'x': x,
            'y': y,
            }
    
    interpolated1 = _model1.interp(
        coords = merged_coords,
        kwargs = {'fill_value': 0},
        )
    
    interpolated2 = _model2.interp(
        coords = merged_coords,
        kwargs = {'fill_value': 0},
        )
    
    interpolated1 = xr.where(
        interpolated1.time > _model1.time[-1],
        interpolated1.sel(time = _model1.time[-1]),
        interpolated1,
        )
    
    interpolated2 = xr.where(
        interpolated2.time > _model2.time[-1],
        interpolated2.sel(time = _model2.time[-1]),
        interpolated2,
        )
    
    
    to_merge = [interpolated1, interpolated2]
    
    unique_variables = np.unique([list(_model1.var())+ 
                                  list(_model2.var())])
    
    for var in unique_variables:
        for i, m in enumerate(to_merge):
            if not hasattr(m, var):
                zeros = xr.zeros_like(to_merge[abs(i - 1)][var])
                nans = zeros.where(zeros != 0)
                m[var] = nans
    
    merged = xr.concat(
        to_merge,
        dim = 'reservoir',        
        fill_value=np.nan
    ).compute()
    
    merged.attrs['bounds'] = _utils.bounds_from_xy(x, y)
    merged.attrs['influence_radius'] = np.max(
        (_model1.influence_radius, _model2.influence_radius)
        )
    merged.attrs['dx'] = dx
    merged.attrs['dy'] = dx
    merged['grid_mask'] = (['y', 'x'], np.zeros((len(y), len(x))))
    merged_model = _memory.build_model_from_xarray(merged, 'merged')
    _merged_model = MergedModel('merged')
    
    merged_model.built = True
    merged_model._shapes = list(model2.shapes) + list(model1.shapes)
    merged_model.mask_reservoirs()
    points = []
    observations = []
    for i, model in enumerate([model1, model2]):
        if model.hasattr('points'):
            points = points + model.points.collection
        if model.hasattr('observation_points'):
            observations = observations + model.observation_points.collection
    if len(points) > 0:
        merged_model.set_points(_Points.PointCollection(points))
        merged_model.calculate_subsidence_at_points(_print = False)
    if len(observations) > 0:
        merged_model.set_observation_points(_Points.ObservationCollection(observations))
        merged_model.calculate_subsidence_at_observations(_print = False)
    merged_model.set_dx(dx)
    _merged_model.__dict__ = merged_model.__dict__
    merged_model = _merged_model
    return merged_model