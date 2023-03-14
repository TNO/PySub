# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 08:47:55 2022

@author: davidsb
"""
import numpy as np
import pandas as pd
from PySub import utils as _utils

class Point:
    """Object to store coordinate data of a single point
    """ 
    def __init__(self, name, x, y):
        """Initialize the point object.

        Parameters
        ----------
        name : str
            The name of the point, used as identification.
        x : int/float
            x-coordinate of the point.
        y : int/float
            y-coordinate of the point.

        Returns
        -------
        Point object.

        """
        self._name = str(name)
        self._x, self._y = x, y
    
    @property
    def x(self):
        """Property : Return the x-coordinate of the point.

        Returns
        -------
        int/float
            x-coordinate of the Point.

        """
        return self._x
    
    @property
    def y(self):
        """Property : Return the y-coordinate of the point.

        Returns
        -------
        int/float
            y-coordinate of the Point.

        """
        return self._y
    
    @property 
    def coordinates(self):
        """Property : Return the x- and y-coordinates of the point.

        Returns
        -------
        tuple, int/float
            (x, y)-coordinates of the Point.

        """
        return self._x, self._y
    
    @property
    def name(self):
        """Property : Rturn the name of the point.

        Returns
        -------
        str
            Identifiaction of the point.

        """
        return self._name
    
    def __repr__(self):
        return str(self.coordinates)
    
class PointCollection:
    """Object to store the information of multiple Point objects.
    """
    def __init__(self, list_of_Points):
        """Initialize the PointCollection object using a list of point objects.
        
        Parameters
        ----------
        list_of_Points : list, Pooints.Point object
            A list with Point objects.

        Raises
        ------
        Exception
            When no list has been entered, or an invalif Point object is in that
            list.

        Returns
        -------
        PointCollection object.

        """
        type_entry = type(list_of_Points) 
        if not (type_entry is list or type_entry is np.ndarray):
            if isinstance(list_of_Points, Point):
                list_of_Points = [list_of_Points]
            else:
                raise Exception('Entries should be a list of Point objects or a single Point object.')
        types = np.array([isinstance(p, Point) for p in list_of_Points])
        if np.logical_not(types).any():
            raise Exception('Entries should be a list of ObservationPoint objects or a single ObservationPoint.')
        
        self.collection = list_of_Points
        self.number_of_points = len(list_of_Points)
        self._names = [p.name for p in self.collection]
        self._x = [p.x for p in self.collection]
        self._y = [p.y for p in self.collection]
        self._coordinates = [p.coordinates for p in self.collection]
    
    def __getitem__(self, item):
        """Return a point object based on the input variable.

        Parameters
        ----------
        item : int or str
            Identification of the point object. When it is an integer, it will 
            look the point up by its location in the list. When it is a string,
            the point object will be found using it's name.

        Raises
        ------
        Exception
            When an invalid data type is entered as input variable.

        Returns
        -------
        Point object
            
        """
        if (isinstance(item, int) or
            isinstance(item, slice)):
            return self.collection[item]
        elif _utils.is_list_of_numbers(item):
            return [self.collection[i] for i in item]
        elif isinstance(item, str):
            if item in self.names:
                return self.collection[np.where(np.array(self.names) == item)[0][0]]
            else:
                raise Exception(f'{item} not available in point collection')
        elif _utils.is_list_of_strings(item):
            return [self.collection[np.where(np.array(self.names) == i)[0][0]] for i in item if i in self.names]
        else: 
            raise Exception(f'Invalid index type: {type(item)}.')
    
    def __len__(self):
        return len(self.collection)
    
    def __repr__(self):
        return str(self._names)
    
    @property
    def names(self):
        """Returns a list with the names of all the points in the Collection.

        Returns
        -------
        list, str
            A List with all the names of the points in the collection.

        """
        return self._names
    
    @property
    def x(self):
        """Returns the x-coordinates of all the points in the collection as a list.

        Returns
        -------
        list, int/float
            A list with the x-coordinates of all the points in the collection.

        """
        return self._x
    
    @property
    def y(self):
        """Returns the y-coordinates of all the points in the collection as a list.

        Returns
        -------
        list, int/float
            A list with the y-coordinates of all the points in the collection.

        """
        return self._y
    
    @property
    def coordinates(self):
        """Returns the x- and y-coordinates of all the points in the collection as a list.

        Returns
        -------
        list, tuples, int/float
            A list with tuples of the x- and y-coordinates of all the points in the collection.

        """
        return self._coordinates
    
    @property
    def as_df(self):
        """Return a pandas DataFrame of all the points in the collection.

        Returns
        -------
        df : pd.DataFrame

        """
        df = pd.DataFrame(self.coordinates, index = self.names, columns = ['X', 'Y'])
        df.index.name = 'Point ID'
        return df

class ObservationPoint:
    """Object to store any temporal observations linked to a specific point. The ObservatinPoint
    object stores information about observations over time. The information includes:
        - The location in x- and y-coordinates
        - Values for observations
        - The The timstamps at which each observation is taken
        - A lower and upper error bound.
    """
    def __init__(self, name, x, y, time = None, observation = None, lower_error = None, upper_error = None):
        """Initialize the CollectionPoint object.

        Parameters
        ----------
        name : str
            Identification of the point object.
        x : int/float
            The x-coordinate of the point.
        y : int/float
            The y-coordinate of the point.
        time : list, int/float/datetime objects, optional
            A list with a representation of the time of the observation. The type 
            of the objects stored in the list is very flexible, any object, int, float, str or
            datetime object can be stored in it. It is up to the vuilder of this object 
            to determin which type is best for the goal of storing this data. 
            The default is None.
        observation : list, float, optional
            The observation value for each of the times. The default is None.
        lower_error : float, optional
            The deviation of the values below the observation value. The default 
            is None.
        upper_error : float, optional
            The deviation of the values above the observation value. The default 
            is None.

        Returns
        -------
        ObservationPoint object.

        """
        self._name = str(name)
        self._x, self._y = x, y
        self._time = None
        self._observations = None
        self._error = None
        self._number_of_entries = None
        self._variable_that_set_number_of_entries = None
        
        self._lower_error = lower_error,
        self._upper_error = upper_error
    
    @property
    def number_of_entries(self):
        """Return the number of observations in the ObservationPoint. The number of observations 
        is equal to the amount of times and errors.

        Returns
        -------
        int
            Number of observations in the Observationpoint.

        """
        return self._number_of_entries
    
    @property
    def x(self):
        """Property : Return the x-coordinate of the point.

        Returns
        -------
        int/float
            x-coordinate of the Point.

        """
        return self._x
    
    @property
    def y(self):
        """Property : Return the y-coordinate of the point.

        Returns
        -------
        int/float
            y-coordinate of the Point.

        """
        return self._y
    
    @property 
    def coordinates(self):
        """Property : Return the x- and y-coordinates of the point.

        Returns
        -------
        tuple, int/float
            (x, y)-coordinates of the Point.

        """
        return self._x, self._y
    
    @property
    def name(self):
        """Property : Rturn the name of the point.

        Returns
        -------
        str
            Identifiaction of the point.

        """
        return self._name
    
    @property
    def time(self):
        """Return the labels for the time of each observation.

        Returns
        -------
        list, datetime objects
             A list with the labels for the time of each observation.
        """
        return self._time
    
    @property
    def observations(self):
        """Return the observation values.

        Returns
        -------
        list, float
            List with the values of observations.

        """
        return self._observations
    
    @property
    def relative(self):
        """The relative difference between each observation and the very
        first observation.

        Returns
        -------
        list, float
            List with the difference between values of observations and the
            very first observation.

        """
        index = np.where(self.time == min(self.time))[0][0]
        return self._observations - self.observations[index]
    
    @property
    def lower_error(self):
        """Returns the deviation of the values below the observation value. 

        Returns
        -------
        list, float
            List with the deviation of the values below the observation value.

        """
        return self._lower_error
    
    @property
    def upper_error(self):
        """Returns the deviation of the values above the observation value. 

        Returns
        -------
        list, float
            List with the deviation of the values above the observation value.

        """
        return self._upper_error
    
    def __repr__(self):
        return str(self.name)
    
    def hasattr(self, var): # redefenition for convenient syntax
        try:
            result = getattr(self, var)
            if result is None:
                return False
            else:
                return True
        except:
            return False
        
    def check_length(self, var, name):
        if not _utils.is_iterable(var):
            raise Exception('Input must be a list, numpy array or pandas Series object.')
        var = np.array(var)
        if len(var.shape) != 1:
            raise Exception('Input must be 1D.')
            
        if self.hasattr('number_of_entries'):
            if self._number_of_entries !=  len(var):
                raise Exception(f'Number of entries, set by {self._variable_that_set_number_of_entries}, do not match with length of {name}. No. of entries: {self._number_of_entries}')
        else:
            self._number_of_entries = len(var)
            self._variable_that_set_number_of_entries = name
        return var
          
    def set_time(self, time):
        """Set the time attribute for this Point object.
        
        Parameters
        ----------
        time : list
            List with objects (int/str/datetime64, etc.) denoting the time at
            which each observation is taken.

        """
        time = self.check_length(time, 'time')
        time = _utils.convert_to_datetime(time)
        self._time = time
    
    def set_observations(self, observations):
        """Set the observation values for this point object.

        Parameters
        ----------
        observations : list, floats
            The observation values in a list.

        """
        observations = self.check_length(observations, 'observations')
        _utils._check_low_high(observations, 'observations', -1000, 1000)
        self._observations = observations
    
    def set_error(self, error):  
        # Bonus funtion for the lazy.
        self.set_lower_error(error)
        self.set_upper_error(error)
    
    def set_lower_error(self, lower_error):
        """Set the lower error function attribute for this point object.

        Parameters
        ----------
        lower_error : list, float
            A list with the error (below the observation values) for each observation 
            value.

        """
        if lower_error is None:
            if self.hasattr('number_of_entries'):
                lower_error = np.zeros(self._number_of_entries)
            else:
                self._lower_error = None
        else:
            lower_error = self.check_length(lower_error, 'lower error')
            if (lower_error < 0).any():
                raise Exception('All values for error should be positive.')
            self._lower_error = lower_error
    
    def set_upper_error(self, upper_error):
        """Set the upper error function attribute for this point object.

        Parameters
        ----------
        upper_error : list, float
            A list with the error (above the observation values) for each observation 
            value.

        """
        if upper_error is None:
            if self.hasattr('number_of_entries'):
                upper_error = np.zeros(self._number_of_entries)
            else:
                self._upper_error = None
        else:
            upper_error = self.check_length(upper_error, 'upper error')
            if (upper_error < 0).any():
                raise Exception('All values for error should be positive.')
            self._upper_error = upper_error
    
    def set_parameters(self, time, observations, lower_error, upper_error):
        """Set all relevant parameters for this ObservationPoint object.

        Parameters
        ----------
        time : list
            List with objects (int/str/datetime64, etc.) denoting the time at
            which each observation is taken.
        observations : list, floats
            The observation values in a list.
        lower_error : list, float
            A list with the error (below the observation values) for each observation 
            value.
        upper_error : list, float
            A list with the error (above the observation values) for each observation 
            value.

        Returns
        -------
        None.

        """
        self.set_time(time)
        self.set_observations(observations)
        self.set_lower_error(lower_error)
        self.set_upper_error(upper_error)
    
    

class ObservationCollection:
    """Object to store multiple ObservatinPoint objects in. The ObservatinPoint
    object stores information about observations over time. The information includes:
        - The location in x- and y-coordinates
        - Values for observations
        - The The timstamps at which each observation is taken
        - A lower and upper error bound.
    """
    def __init__(self, list_of_ObservationPoints):
        """Initialize the ObservationCollection object.

        Parameters
        ----------
        list_of_ObservationPoints : list, ObservationPoint

        Returns
        -------
        ObservationCollection object.

        """
        
        if not _utils.is_iterable(list_of_ObservationPoints):
            if isinstance(list_of_ObservationPoints, ObservationPoint):
                list_of_ObservationPoints = [list_of_ObservationPoints]
            else:
                raise Exception('Entries should be a list of ObservationPoint objects or a single ObservationPoint.')
        types = np.array([isinstance(p, ObservationPoint) for p in list_of_ObservationPoints])
        if np.logical_not(types).any():
            raise Exception('Entries should be a list of ObservationPoint objects or a single ObservationPoint.')
        self.collection = list_of_ObservationPoints
        self.number_of_observation_points = len(self.collection)
        self._names = [p.name for p in self.collection]
        self._x = [p.x for p in self.collection]
        self._y = [p.y for p in self.collection]
        self._coordinates = [p.coordinates for p in self.collection]
        self._time = [p.time for p in self.collection]
        self._observations = [p.observations for p in self.collection]
        self._relative = [p.relative for p in self.collection]
        self._number_of_entries = [p.number_of_entries for p in self.collection]
        self._lower_errors = [p.lower_error for p in self.collection]
        self._upper_errors = [p.upper_error for p in self.collection]
        
    
    def __getitem__(self, item):
        """Return a point object based on the input variable.

        Parameters
        ----------
        item : int or str
            Identification of the point object. When it is an integer, it will 
            look the point up by its location in the list. When it is a string,
            the point object will be found using it's name.

        Raises
        ------
        Exception
            When an invalid data type is entered as input variable.

        Returns
        -------
        Point object
            
        """
        if (isinstance(item, int) or
            isinstance(item, slice)):
            return self.collection[item]
        elif _utils.is_list_of_numbers(item):
            return [self.collection[i] for i in item]
        elif isinstance(item, str):
            if item in self.names:
                return self.collection[np.where(np.array(self.names) == item)[0][0]]
            else:
                raise Exception(f'{item} not available in point collection')
        elif _utils.is_list_of_strings(item):
            return [self.collection[np.where(np.array(self.names) == i)[0][0]] for i in item if i in self.names]
        else: 
            raise Exception(f'Invalid index type: {type(item)}.')
    
    def __len__(self):
        return len(self.collection)
    
    def __repr__(self):
        return str(self._names)
    
    @property
    def names(self):
        """Returns a list with the names of all the points in the Collection.

        Returns
        -------
        list, str
            A List with all the names of the points in the collection.

        """
        return self._names
    
    @property
    def x(self):
        """Returns the x-coordinates of all the points in the collection as a list.

        Returns
        -------
        list, int/float
            A list with the x-coordinates of all the points in the collection.

        """
        return self._x
    
    @property
    def y(self):
        """Returns the y-coordinates of all the points in the collection as a list.

        Returns
        -------
        list, int/float
            A list with the y-coordinates of all the points in the collection.

        """
        return self._y
    
    @property
    def coordinates(self):
        """Returns the x- and y-coordinates of all the points in the collection as a list.

        Returns
        -------
        list, tuples, int/float
            A list with tuples of the x- and y-coordinates of all the points in the collection.

        """
        return self._coordinates
    
    @property
    def time(self):
        """Returns the times for each Observationpoint in the Collection.

        Returns
        -------
        List of lists
            Returns a list with the length of the number of ObservationPoints, containing
            lists with the indications of the time in each Observationpoint object.
            The type of objects that represent the time can be anything, but are the same
            as the input values defined during setting.

        """
        return self._time
    
    @property
    def observations(self):
        """Returns the observations for each Observationpoint in the Collection

        Returns
        -------
        List of lists, float
            Returns a list with the length of the number of ObservationPoints, containing
            lists with the observations in each Observationpoint object.

        """
        return self._observations
    
    @property
    def relative(self):
        """Returns the difference between the first observation for each observation, 
        for each Observationpoint in the Collection

        Returns
        -------
        List of lists, float
            Returns a list with the length of the number of ObservationPoints, containing
            lists with the relative attribute from each Observationpoint object.

        """
        return self._relative
    
    @property
    def number_of_entries(self):
        """Return the number of observations for each ObservationPoint in the Collection. 
        The number of observations is equal to the amount of times and errors.

        Returns
        -------
        int
            Number of observations for each point in the Collection.

        """
        return self._number_of_entries
    
    @property
    def lower_errors(self):
        """Return the lower errors for each Observationpoint in the Collection.

        Returns
        -------
        list of lists
            Returns a list with the length of the number of ObservationPoints, containing
            lists with the lower error in each Observationpoint object.
        """
        return self._lower_errors
    
    @property
    def upper_errors(self):
        """Return the upper errors for each Observationpoint in the Collection.

        Returns
        -------
        list of lists
            Returns a list with the length of the number of ObservationPoints, containing
            lists with the upper error in each Observationpoint object.
        """
        return self._upper_errors
    
    @property
    def as_df(self):
        """Return the collection of observations as a pandas dataframe.

        Returns
        -------
        df : pf.DataFrame

        """
        IDs = [[o.name for _ in self.observations[i]] for i, o in enumerate(self.collection)]
        IDs = _utils.flatten_ragged_lists2D(IDs)
        columns = ['Observation ID', 'Time', 'X', 'Y', 'Subsidence (m)', 'Lower error (m)', 'Upper error (m)']
        df = pd.DataFrame(columns = columns)
        df['Observation ID'] = IDs
        df['Time'] = _utils.flatten_ragged_lists2D(self.time)
        X = [[o.x for _ in self.observations[i]] for i, o in enumerate(self.collection)]
        Y = [[o.y for _ in self.observations[i]] for i, o in enumerate(self.collection)]
        df['X'] = _utils.flatten_ragged_lists2D(X)
        df['Y'] = _utils.flatten_ragged_lists2D(Y)
        df['Subsidence (m)'] = _utils.flatten_ragged_lists2D(self.observations)
        try: df['Lower error (m)'] = _utils.flatten_ragged_lists2D(self.lower_errors)
        except: pass
        try: df['Upper error (m)'] = _utils.flatten_ragged_lists2D(self.upper_errors)
        except: pass
        return df
    
    @property
    def most_subsidence(self):
        last_value_of_each_observations_point = [i[-1] for i in self.relative]
        observation_point_with_most_observed_subsidence = np.where(last_value_of_each_observations_point == np.max(last_value_of_each_observations_point))[0][0]
        i = int(observation_point_with_most_observed_subsidence)
        return self[i]
    
def load_points_from_df(point_df, 
                        x_column = 'X', y_column = 'Y'):
    """Returns a PointCollection object based on a pd.DataFrame.

    Parameters
    ----------
    point_df : pd.DataFrame
        A pandas Dataframe with the names of the points on the index and at least
        two columns with the x and the y values.
    x_column : str or int, optional
        The name or index of the column containing the x-coordinates. The default is 'X'.
    y_column : str or int, optional
        The name or index of the column containing the y-coordinates. The default is 'Y'.

    Returns
    -------
    PointCollection object.

    """
    column_indicators = [x_column, y_column]
    check_columns = [val for val in column_indicators]
    for val in check_columns:
        if not _utils.is_str_or_int(val):
            raise Exception(f'Invalid input {val}, enter string for column name or integer for column index')
            
    missing_columns = _utils.a_missing_from_b(check_columns, point_df.columns.values)
    if len(missing_columns) > 0:
        raise Exception(f'The columns {missing_columns} are not in the Excel sheet.')
    point_ids = point_df.index.values.astype(str)
    point_unique_ids = np.unique(point_ids)
    points = []
    for name in point_unique_ids:
        x = point_df[x_column][name]
        y = point_df[y_column][name]
        
        point = Point(name, x, y)
        
        points.append(point)
        
    return PointCollection(points)
            
def load_observation_points_from_df(observation_df, observation_column = "Observations",
                                    x_column = 'X', y_column = 'Y', time_column = 'Time',
                                    lower_error_column = None, upper_error_column = None):
    """Returns a ObservationCollection object based on a pd.DataFrame.

    Parameters
    ----------
    observation_df : pd.DataFrame
        A pandas Dataframe with the names of the observation names on the index and at least
        the columns with the x and the y values, observation values and indicators for the time.
    x_column : str or int, optional
        The name or index of the column containing the x-coordinates. The default is 'X'.
    y_column : str or int, optional
        The name or index of the column containing the y-coordinates. The default is 'Y'.
    time_column : str or int, optional
        The name or index of the column containing the data for time. The default is 'Time'.
    lower_error_column : str or int, optional
        The name or index of the column containing the data for lower errors. The default is None.
        When None, no lower error will be loaded into the Collection object.
    upper_error_column : str or int, optional
        The name or index of the column containing the data for upper errors. The default is None.
        When None, no upper error will be loaded into the Collection object.

    Returns
    -------
    ObservationCollection object.

    """
    
    column_indicators = [observation_column, x_column, y_column, time_column,
                                lower_error_column, upper_error_column]
    check_columns = [val for val in column_indicators if val is not None]
    for val in check_columns:
        if not _utils.is_str_or_int(val):
            raise Exception(f'Invalid input {val}, enter string for column name or integer for column index')
            
    missing_columns = _utils.a_missing_from_b(check_columns, observation_df.columns.values)
    if len(missing_columns) > 0:
        raise Exception(f'The columns {missing_columns} are not in the Excel sheet.')
    observation_ids = observation_df.index.values.astype(str)
    observation_df.index = observation_ids
    observation_unique_ids = np.unique(observation_ids)
    observation_points = []
    for observation, name in enumerate(observation_unique_ids):
        x = observation_df[x_column][name]
        if _utils.is_iterable(x):
            x = x.unique()
        else:
            x = [x]
        y = observation_df[y_column][name]
        if _utils.is_iterable(y):
            y = y.unique()
        else:
            y = [y]
        if len(x) != 1 and len(y) != 1:
            raise Exception('Observations cannot change location over time, check x and y cordinates and secure uniform x- and y-coordinates.')
        x, y = x[0], y[0]
        observation_point = ObservationPoint(name, x, y)
        
        time = observation_df[time_column][name]
        if _utils.is_iterable(time):
            time = time.values
        else:
            time = _utils.convert_to_datetime(time)
        
        obs = observation_df[observation_column][name]
        if _utils.is_iterable(obs):
            obs = obs.values
        else:
            obs = [obs]
            
        if lower_error_column is not None:
            le = observation_df[lower_error_column][name]
            if _utils.is_iterable(le):
                le = le.values
            else:
                le = [le]
        else: le = None
        if upper_error_column is not None:
            ue = observation_df[upper_error_column][name]
            if _utils.is_iterable(ue):
                ue = ue.values
            else:
                ue = [ue]
        else: ue = None
        
        observation_point.set_parameters(time,
                                         obs,
                                         le, ue)
        observation_points.append(observation_point)
        
    return ObservationCollection(observation_points)


    
    
    
    
    
    
    
    