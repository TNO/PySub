import os
import math
import numpy as np
from scipy.fft import fft2, ifft2
from scipy import ndimage
from scipy import interpolate
import pandas as pd
from matplotlib import path
import datetime
import xarray as xr
from warnings import warn

def fill_dict_with_none(dict):
    """If the values of a dictionary are a list, fill that list
    to the same length as the longest list.
    """
    _dict = {}
    max_length = max(len(entry) for entry in dict.values())
    for v in dict.keys():
        _dict[v] = dict[v] + [None] * (max_length - len(dict[v]))
    return _dict

def gradient2d(values, dx, dy):
    """Get the 2D gradient of a 2D array.
    """
    du__dx, du__dy = np.gradient(values, dx, dy, axis = (0,1))
    return np.array([du__dx, du__dy])
    
def gradient_magnitude(values, dx, dy):
    """Get the 2D gradient magnitude of a 2D array.
    """
    return np.linalg.norm(gradient2d(values, dx, dy), axis = 0)

def concavity_magnitude(values, dx, dy):
    """Get the 2D concavity magnitude of a 2D array.
    """
    du__dx, du__dy = np.gradient(values, dx, dy, axis = (0,1))
    ddu_dxy, ddu_dxx = np.gradient(du__dx, dx, dy, axis = (0,1))
    ddu_dyy, ddu_dyx = np.gradient(du__dy, dx, dy, axis = (0,1))
    
    magnitude = np.linalg.norm(np.array([ddu_dxx, 
                                         ddu_dxy, 
                                         ddu_dyx, 
                                         ddu_dyy]), 
                               axis = 0)
    return magnitude

def get_chunked(ds, chunk_dimensions = ['time', 'reservoir']):  
    """Chunk an xarray dataset using the standard chunk dimension and size, 
    so each cunk is the size of (ny, nx)
    """
    if chunk_dimensions == 'all':
        return ds.chunk({dim: 1 for dim in ds.dims})
    else:    
        return ds.chunk({dim: (1 if dim in chunk_dimensions else -1) for dim in ds.dims})

def pick_from_kwargs(kwargs, index, number_of_entries):
    """Sometimes in PySub, it might be desirable to enter a list of parameter 
    values to the keyword arguments (kwargs), where the resulting function only 
    takes the one value (notorioulsy, matploltib.patches.Polygon). This function 
    selects the keyword argument using index.
    

    Parameters
    ----------
    kwargs : dict
        Dictionary where the keys are the argument kewords and the values are 
        the arguments.
    index : int
        The index of the argument in a list.
    number_of_entries : int
        The number of entries in a list. This must be entered to determine of 
        the list is relevant to the keyword argument.

    Returns
    -------
    adjusted_kwargs : dict
        Dictionary where the keys are the argument kewords and the values are 
        the arguments. Selected for a specific entry.

    """
    adjusted_kwargs = {(kw):
                       (arg[index] if (
                           False if not is_iterable(arg) else (len(arg) == number_of_entries)
                                           ) 
                        else kwargs[kw]) 
                    for kw, arg in kwargs.items()}
    return adjusted_kwargs

def poly_func(v, A, B, C, D, E, F, G, H, I, J, K):
    """A 10th order polynomial dependant on the variable v and coefficients
    A to M.

    Parameters
    ----------
    v : float
        Variable.
    A to M : float
        POlynomial coefficients.
    
    Returns
    -------
    float
        Result of the polynomial.

    """
    return (A + B*v + C*v**2 + 
            D*v**3 + E*v**4 + 
            F*v**5 + G*v**6 + 
            H*v**7 + I*v**8 + 
            J*v**9 + K*v**10 
            )

def get_van_opstal_coefficients(v):
    """Get the Van Opstal coefficients a12, a21, a22, b11, b21 (a11 is 1) 
    determined by the Poisson's ratio.
    
    Parameters
    ----------
    v : float
        Value of the poisson's ratio. Must be between zero and 0.45.

    Returns
    -------
    coefficients : TYPE
        DESCRIPTION.

    """
    try: assert all((0 < v) & (v < 0.45)), f"Poisson's ratio must be between 0 and 0.45. Current value: {v}."
    except: assert 0 < v < 0.45, f"Poisson's ratio must be between 0 and 0.45. Current value: {v}."
    coefficient_directory_name = os.path.join(os.path.dirname(__file__), 'coefficients', 'van_opstal_terms.csv')
    parameter_df = pd.read_csv(coefficient_directory_name, index_col=0)
    parameter_dict = dict(parameter_df )
    coefficients = {
        p: poly_func(v, *c) for p, c in parameter_dict.items()
        }    
    return coefficients

def non_zero_prod(l):
    """Return the product of all the values in a list, ignoring zeros.

    Parameters
    ----------
    l : iterable, float or int

    Returns
    -------
    same type as l
        product of the list, ignoring zeros.

    """
    return np.prod([var for var in l if var != 0])



def is_excel_file(file):
    """Reutn boolean to indicate if the entry is a a path to an exel file.

    Parameters
    ----------
    file : str
        Any string.

    Returns
    -------
    bool
        True if the string is a path to an excel file, False if not.

    """
    _, ext = os.path.splitext(file)
    return ext in ['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']

def make_array(var):
    """Cast securely into an array.
    """
    if not is_iterable(var):
        return_var = np.array([var])
    else:
        return_var = np.array(var)
    return return_var

def probability_distribution(values, probabilities = None):
    """Returns a dataframe with a sorted distribution and 2 tuples with the p10,
    p50 and p90 values and indices of p.

    Parameters
    ----------
    values : list, floats
        A list or array with values.
    probabilities : list, floats
        a list or array with the probabilities of each value, if the probility 
        is None (which would be valid if the sampling process already accounts
        for probability), the probability of the sample is equal for each value. 

    Returns
    -------
    p_df : pd.DataFrame
        A pandas dataframe where the values of p are sorted (descending) in the column
        'values'. In the column 'p' are the probabilities of any value to be below that
        number. The numbers on the index correspond to the indices of p. 
    p_values : tuple, floats
        The values for the p10, p50 and p90 probabilities.
    p_indices : tuple, int
        The indices of p of which values correspond to the p10, p50, p90.

    """
    if is_iterable(values):
        if len(values) <2:
            print('Warning: Not enough samples to create distribution.')
            return None, None, None
    
    if probabilities is None:
        number_of_samples = len(values)
        probabilities = [1/number_of_samples]*number_of_samples
    p_df = pd.DataFrame()
    p_df['probabilities'] = np.array(probabilities)*100
    p_df['values'] = values
    sorted_p_df = p_df.sort_values('values', ascending = False)
    sorted_p_df['cumsum'] = np.cumsum(sorted_p_df['probabilities'])  
    p90, p50, p10 = np.searchsorted(sorted_p_df['cumsum'].values, [10, 50, 90])
    p_values = tuple(sorted_p_df['values'].iloc[[p90, p50, p10]])
    p_indices = tuple(sorted_p_df.index[[p90, p50, p10]])
    return sorted_p_df, p_values, p_indices

def normalize_to_range(to_normalize, range_to_normalize_to):
    """Normalize a range between the minimum and maximum values of another range.

    Parameters
    ----------
    to_normalize : iterable, floats
        Range to normalize.
    range_to_normalize_to : iterable, floats
        Range between which the other range will be normalized.

    Returns
    -------
    normalized : np.ndarray, floats
        Normalized range.

    """
    min0 = range_to_normalize_to[0]
    max0 = range_to_normalize_to[-1]
    min1 = to_normalize[0]
    set_to_min = to_normalize - (min1-min0)
    normalized = (set_to_min/max(set_to_min))*max0
    return normalized

def normalize(data, axis = None):
    if axis is None:
        return (data - np.min(data))/(np.max(data) - np.min(data))
    else:
        return ((data - np.expand_dims(np.min(data, axis = axis), axis)) /
                (np.expand_dims(np.max(data, axis = axis), axis) - 
                 np.expand_dims(np.min(data, axis = axis), axis)))
    
def datetime_to_years_as_float(time):
    """List of datetime type objects to convert to floats, or when they are 
    floats already, return the amount of time that has passed from the first entry.
    Examples:
        - [1990, 2000] -> [0, 10]
        - [1.3, 2,6] -> [0, 1.3]
        - [01-01-1990, 07-03-1990] -> [0. , 0.501]

    Parameters
    ----------
    time : iterable, datetime/floats
        Datetime objects or floats to be converted to floats.

    Returns
    -------
    np.ndarray, floats
        Array of timesteps with the first value being zero, and the subsequent 
        values the amount of time passed since in years.

    """
    time = np.array(time)
    if is_list_of_datetime(time):
        return ((pd.to_datetime(time) - pd.to_datetime(time[0])).astype(int) * 3.16887646 * 1e-17).values
    elif is_list_of_numbers(time):
        return time - time[0]
    else:
        raise Exception('Invalid datetype, enter lists only.')

def interpolate_over_time(timesteps, values, times_to_interpolate):
    """Interpolate over datetime objects. When values fall outside of the range 
    the values will be filled with the first values in de timeseries when before
    the first time in the dataset. The last value will fill all the values after 
    the last value in the timeseries.

    Parameters
    ----------
    timesteps : iterable with datetime objects
        The timesteps between which will be interpolated. Must have the same length 
        as values.
    values : iterable, floats
        values between which will be interpolated. Must have the same length as 
        timesteps
    times_to_interpolate : iterable with datetime objects
        The timesteps for the values to be interpolated.

    Returns
    -------
    ndarray, floats
        Inteprolated values.

    """
    _timesteps = (pd.to_datetime(timesteps) - timesteps[0]).astype(int)
    _times_to_interpolate = (pd.to_datetime(times_to_interpolate) - timesteps[0]).astype(int)
    return np.interp(_times_to_interpolate, _timesteps, values, left = values[0], right = values[-1])

def get_order_of_magnitude(some_float):
    """Get the order of magnitude

    Parameters
    ----------
    some_float : float or list of floats
        number(s) of which to determin (each of) the order of magnitude of.

    Returns
    -------
    int or list of int
        The order of magnitude for the input parameter(s).

    """
    if is_iterable(some_float) or is_number(some_float):
        if some_float == 0:
            return 0
        return np.floor(np.log10(abs(some_float))).astype(int)
    else:
        raise Exception('Entry is not a number or list of numbers.')
        

def add_years(d, years):
    """Return a date that's 'years' years after the date (or datetime)
    object 'd'. Return the same calendar date (month and day) in the
    destination year, if it exists, otherwise use the following day
    (thus changing February 29 to March 1).
    https://stackoverflow.com/a/15743908
    """
    d = pd.Timestamp(d)
    try:
        return d.replace(year = d.year + years)
    except ValueError:
        return d + (datetime.date(d.year + years, 1, 1) - datetime.date(d.year, 1, 1))

def pad_series(timesteps, values, start_time, end_time, start_value, end_value):
    """Pad timeseries with specified values before the start of the timeseries and/or 
    after the timeseries between the start and end of the padding. Pads with a 1 
    year interval.

    Parameters
    ----------
    timesteps : np.array, np.datetime64
        The timesteps of the timeseries which will be padded.
    values : np.array, float
        The values of the timeseries which will be padded.
    start_time : datetime format
        The start of the resulting timeseries. If this time is before the earliest
        timestep in de timeseries, the values in between will be filled with the
        variable start_value.
    end_time : The end of the resulting timeseries. If this time is after the latest
        timestep in de timeseries, the values in between will be filled with the
        variable end_value.
    start_value : float
        Value to fill any data between timeserie start and start_time if start_time
        is before the earliest date in timesteps.
    end_value : float
        Value to fill any data between timeserie end and end_time if end_time
        is after the latest date in timesteps.

    Returns
    -------
    timeseries : np.array, floats
        values of the padded timeseries
    years : np.array, np.datetime64
        The dates with which the timeseries has been padded.

    """
    
    start_time = convert_to_datetime([start_time])[0]
    
    end_time = convert_to_datetime(end_time)[0]
    timesteps = convert_to_datetime(timesteps)
    years = []
    timeseries = []
    current_time = start_time
    while current_time < timesteps[0]:
        years.append(current_time)
        timeseries.append(start_value)
        current_time = add_years(current_time, 1)
    for current_time, value in zip(timesteps, values):
        years.append(current_time)
        timeseries.append(value)
    current_time = add_years(current_time, 1)
    while current_time <= end_time:
        years.append(current_time)
        timeseries.append(end_value)
        current_time = add_years(current_time, 1)
    if years[-1] <= end_time:
        years.append(end_time)
        timeseries.append(end_value)
    return timeseries, years

def profile(start_value, end_value, start_time, end_time,
            profile = [1.        , 0.76207729, 0.45634921, 0.26915114, 0.15562457,
                       0.08661146, 0.04451346, 0.01846101, 0.00293306, 0.        ],
            sub_profile = [1.        , 0.88888889, 0.77777778, 0.66666667, 0.55555556,
                           0.44444444, 0.33333333, 0.22222222, 0.11111111, 0.        ],
            beta_max = 10, beta_min = 1):
    """Make a profile based on the start and end time and start and end value based
    on normalized profiles. based on the time between start and end time, the profile 
    will be more like the main profile or the sub profile.
    
    start_value, end_value, start_time, end_time must be lists with the same length.
    
    The resulting profiles will have the shape based on:
        beta * profile + (1 - beta) * sub_profile
    where beta is the ratio of delta_t (the number of years between start_time and end_time)
    and delta_beta (the number of years between beta_max and beta_min) as per:
        beta = delta_t/delta_beta
            if beta > 1: beta = 1
            if beta < 0: beta = 0
    
    Raises an exception if delta_t is less than 1.

    Parameters
    ----------
    start_value : list, float
        The values at the start of the profile, each value must be higher than their
        counterpart in end_value. Must have same length as end_value.
    end_value : list, float
        The values at the end of the profile, each value must be lower than their 
        counterpart in start_value. Must have same length as start_value.
    start_time : list, datetime objects
        A list of dates which must be less than their corresponding end_time.
    end_time : list, datetime objects
        A list of dates which must be higher than their corresponding start_time.
    profile : list, float
        The main profile, which will be more dominant towards the beta max. The 
        default is a profile similar to a complimentary errorfunction.
    sub_profile : list, optional
        The sub profile, which will be more dominant towards the beta min. The 
        default is a straight line.
    beta_max : float, optional
        The time in number of years at which the resulting profile will be more 
        like the main profile. The default is 10.
    beta_min : float, optional
        The time in number of years at which the resulting profile will be more 
        like the sub profile. The default is 1.

    Returns
    -------
    interpolated : list, floats
        The values of the profile.
    timesteps : list, datetime objects
        The timesteps between which the profile acts. With intervals of 1 year.
        Can have irregular intervals.

    """
    start_value = np.atleast_1d(start_value)
    end_value = np.atleast_1d(end_value)
    start_time = np.atleast_1d(start_time)
    end_time = np.atleast_1d(end_time)
    if (end_value >= start_value).any():
        raise Exception('End value must be lower than start value. More complex pressure profiles than a start and end pressure should be entered manually in the template.')            
    _start_time = convert_to_datetime(start_time)
    _end_time = convert_to_datetime(end_time)
           
    if (end_time <= start_time).any():
        raise Exception('Start time is later than end time.')
    
    if not is_iterable(profile):
        raise Exception('The profile must be a list, tuple or numpy array of numbers.')
    else:
        if not is_number(profile).any():
            raise Exception('The profile must be a list, tuple or numpy array of numbers.')
    
    number_of_reservoirs = len(start_value)
    
    profile = np.array(profile)
    if len(profile.shape) == 1:
        profile = np.repeat(np.expand_dims(profile, 0), number_of_reservoirs, axis = 0)
    elif len(profile.shape) == 2:
        if profile.shape[0] != number_of_reservoirs:
            raise Exception(f'The number of profiles ({profile.shape[0]}) does noet equal the number of reservoirs ({number_of_reservoirs}).')
    
    sub_profile = np.array(sub_profile)
    if len(sub_profile.shape) == 1:
        sub_profile = np.repeat(np.expand_dims(sub_profile, 0), number_of_reservoirs, axis = 0)
    elif len(sub_profile.shape) == 2:
        if sub_profile.shape[0] != number_of_reservoirs:
            raise Exception(f'The number of sub profiles ({sub_profile.shape[0]}) does noet equal the number of reservoirs ({number_of_reservoirs}).')
    
    timesteps = []
    
    interpolated = []
    for i in range(number_of_reservoirs):
        years = []
        current_time = _start_time[i]
        while current_time <= _end_time[i]:
            years.append(current_time)
            current_time = add_years(current_time, 1)
        if years[-1] != _end_time[i]:
            years.append(_end_time[i])

        years = convert_to_datetime(years)
        timesteps.append(years)
        
        number_of_years = len(years)
        if number_of_years < 2:
            raise Exception('Too little time between start and time to establish a proper profile.')
            
        beta = min(1, (number_of_years - beta_min)/beta_max)
        
        interpolated_profile = np.array(beta * profile[i] + (1 - beta) * sub_profile[i])
        transformed_profile = end_value[i] + interpolated_profile * (start_value[i] - end_value[i])
        coords = np.array(list(range(len(transformed_profile))))
        f = interpolate.interp1d(coords, transformed_profile, fill_value = 'extrapolate')
        location_time = np.where((years <= end_time[i]) & (years >= start_time[i]))[0]
        location_end = np.where(years > end_time[i])[0]
        
        
        
        _years = np.zeros(len(years))
        years_float = datetime_to_years_as_float(years)
        _years[location_time] = years_float
        _years[location_end] = years_float[-1]
        normalized_years = normalize_to_range(_years, coords)
        interpolated.append(f(normalized_years))
    return interpolated, timesteps


def interpolate_grid_from_points(points, values, XY, method = 'linear'):
    x, y = XY
    grid_z = interpolate.griddata(points = points, values = values, xi = (x, y), method = method, fill_value = 0)
    return grid_z

def point_or_points(points):
    """Determin if a variable is a point or a list of points.

    Parameters
    ----------
    points : Any type
        Variable to be determined the validity of. Validity is determined by it 
        being a point or list of points. A point is a list or tuple with two numerical 
        values representing an x- and y-coordinate. Multiple points can be represented 
        by a list of such combination of x- and y-coordinates.

    Raises
    ------
    Exception
        When not a point or list of points.

    Returns
    -------
    _points : list of list of tuples
        The point or points as a list of tuples with x- and y-coordinates.
    x : list of floats
        A list with the x-coordinates of each point.
    y : list of floats
        A list with the y-coordinates of each point.

    """
    if is_iterable(points):
        if not is_iterable(points[0]):
            points = [list(points)]
    _points = np.array(points)
    x, y = _points.T 
    x_are_numbers = np.array([is_number(i) for i in x]).all()
    y_are_numbers = np.array([is_number(i) for i in y]).all()
    if not (x_are_numbers and y_are_numbers):
        raise Exception('Point indicators can only be: an integer for indexing, floats for coordinates or a list of integers or floats.')
    return _points, x, y

def check_shapefiles(shapefile_paths):
    is_file = [os.path.isfile(shp) and shp.endswith('.shp') for shp in shapefile_paths]
    check_file = np.where(np.logical_not(is_file))[0]
    if len(check_file) > 0:
        raise Exception(f"Invalid shapefiles: {np.array(shapefile_paths)[check_file]}")

def bounds_from_xy(x, y):
    min_x, min_y = np.min(x), np.min(y)
    max_x, max_y = np.max(x), np.max(y)
    return [min_x, min_y, max_x, max_y]

def bounds_to_polygon(bounds, buffer = 0):
    polygon = [[bounds[0] - buffer, bounds[1] - buffer],
               [bounds[2] + buffer, bounds[1] - buffer],
               [bounds[2] + buffer, bounds[3] + buffer],
               [bounds[0] - buffer, bounds[3] + buffer],
               [bounds[0] - buffer, bounds[1] - buffer]]
    return polygon
        

def is_point_in(polygon, point, radius = 1):
    """Return a list of booleans for each point, which is True for each point
    inside the polygon and False for every point which is not.

    Parameters
    ----------
    polygon : list
        The coordinates of a single polygon with shape m x 2, where m is the 
        amount of points and 2 is the x- and y-coordinates.
    point : list
        The coordinates of a collection of points with shape m x 2, where m is 
        the amount of points and 2 is the x- and y-coordinates.

    Returns
    -------
    list.
        True for each point inside the polygon and False for every point which is not.
    
    """
    _polygon = path.Path(polygon)

    return _polygon.contains_points(point, radius = radius)
    
def _check_low_high(var, var_name, low, high):        
    """Raises an exception if the value entered is None or not within a range
    spcified by a high and a low value.

    Parameters
    ----------
    var : float, list of floats
        Any real number or list of real numbers that can fall in a range.
    var_name : str
        Variable name.
    low : float
        Value the variable(s) cannot be lower than or equal to.
    high : float
        Value the variable(s) cannot be higher than.

    Raises
    ------
    Exception
        When an invalid value has been encountered.

    """
    if var is None:
        raise Exception(f'Variable {var_name} cannot be None.')
    elif isinstance(var, str):
        return
    elif is_iterable(var):
        if is_list_of_strings(list(var)):
            return
        else:
            
            _var = np.array(var)
            wrong = np.where(np.logical_or((_var <= low), (_var > high)))[0]
            if len(wrong) > 0:
                # raise(Exception(f"Invalid input parameter '{var_name}': {_var[wrong]}"))
                test = input(f"Invalid input parameter '{var_name}': {_var[wrong]}. Are you sure? Type: y/n\n")
                valid_input = False
                while valid_input is False:
                    if test.lower() == 'y':
                        valid_input = True
                        return
                    elif test.lower() == 'n':
                        raise Exception('Invalid value(s) encounterd')
                        valid_input = True
                    else:
                        test = input(f"Invalid input parameter '{var_name}': {_var[wrong]}. Are you sure? Type: y/n\n")
                    
    elif var <= low or var > high:
        raise(Exception(f"Invalid input parameter '{var_name}': {var}"))

def extent_from_bounds(bounds, buffer = 0):
    """Reorder a bounds object (lowest x, lowest y, highest x, highest y) to
    a figure extent (lowest x, highest x, lowest y, highest y).

    Parameters
    ----------
    bounds : list, float
        Array-like object with 4 entries for lowest x, lowest y, highest x, 
        highest y.
    buffer : float, optional
        Additional space around the bounds. The default is 0.

    Returns
    -------
    ext : list, floats
        liat with 4 entries for lowest x, highest x, lowest y, highest y.

    """
    ext =  (bounds[0] - buffer, 
            bounds[2] + buffer, 
            bounds[1] - buffer, 
            bounds[3] + buffer)
    return ext

def make_folder(path):
    """Check if the folder to the file at the location indicated by path exists 
    and if not, creates that file.

    Parameters
    ----------
     path : str
        Path to a file.

    Returns
    -------
    None.

    """
    if os.path.isfile(path):
        folder = os.path.dirname(path)
    else:
        folder = path
    if not os.path.isdir(folder):
        os.makedirs(folder)
        
def check_file_extension(path, extension = ['.csv', '.txt']):
    """Raises an exception if any of the valid datatype are encountered in the extension.
    
    Parameters
    ----------
    path : str
        path to a file.
    extension : str or list of str, optional
        The extension the file is allowed to have. The default is ['.csv', '.txt'].
        Use None if nu check for extension is desired.

    Raises
    ------
    Exception
        When file extension is not recognized.

    """
    if is_iterable(extension):
        if not is_list_of_strings(extension):
            raise Exception(f'Invalid data type encountered in list of data extension type: {extension}. Expected strings.')
            
        list_of_valid_extensions = extension
    elif extension is None:
        return
    elif isinstance(extension, str):
        list_of_valid_extensions = [extension]
        
    file_location, file_extension = os.path.splitext(path)

    if not file_extension:
        file_extension = list_of_valid_extensions[0]
    elif file_extension not in list_of_valid_extensions:
        raise Exception(f'Unknown file extension {file_extension} detected, use {list_of_valid_extensions}. Use no extension to default to {list_of_valid_extensions[0]}.')

def _check_df_any_nans(df):
    return df.isna().any(axis = 0).values

def _check_df_all_nan(df):
    return df.isna().all(axis = 0).values

def _check_df_all_filled_or_all_nan(df):
    check = np.logical_not(np.logical_and(_check_df_any_nans(df),
                                          np.logical_not(_check_df_all_nan(df))))
    wrong_columns = list(df.columns[np.logical_not(check)])
    if df.index.isna().any() and not df.index.isna().all():
        wrong_columns.append(df.index.name)
    if len(wrong_columns) > 0:
        raise Exception(f'The columns {wrong_columns} should either have all values filled or all values emtpy (in the case of optional values). Empty and filled values haven been encountered.')

def check_df(df, sheet_name, exclude = [], columns = [], must_haves = [], no_empty = True):
    if len(columns) != 0:
        missing = a_missing_from_b(columns, df.columns.values)
        if len(missing) > 0:
               raise Exception(f'The worksheet {sheet_name} is missing the columns {missing}.')
    df.replace(r'^\s*$', np.nan, regex = True, inplace = True)
    df.replace({pd.NaT: np.NaN}, inplace = True)
    df.dropna(axis = 0, how = 'all', inplace = True)
    if df.shape[0] != 0:
        if no_empty:
            empty = np.unique((df.columns[np.where(df.isnull().values)[1]]))
        else:
            empty = np.unique(np.array(must_haves)[np.where(df.isnull()[must_haves].values)[1]])
        missing = a_missing_from_b(empty, exclude)
        if len(missing) > 0:    
            raise Exception(f'From the worksheet "{sheet_name}" the variable(s): {missing} are missing.')
            
    else:
        raise Exception(f'No entries in worksheet {sheet_name}.')
        
def export_df(df, path, extension = ['.csv', '.txt']):
    """Export a pandas DataFrame to a csv file. If the folder of the file 
    is not yet created, it will be in this function.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
    path : str
        Path to a file to which the df will be stored. Can be a .csv or .txt
        extension. When no extension is assigned, it will be assigned a .csv.

    Raises
    ------
    PermissionError
        When file is being used.

    Returns
    -------
    None.

    """
    if not isinstance(path, str):
        raise Exception(f'File {path} must be a string with the path to the save file location.')
    else:
        check_file_extension(path, extension = extension)
    
    try: df.to_csv(path, index = False)
    except PermissionError: raise PermissionError(f'Permission denied: {path}\n Wait a second if the file is saving. If the file is open, close it before exporting. ')


def convert_SI(val, unit_in, unit_out):
    """Convert the value val from te unit it is in (unit_in) to the requested
    unit (unit_out).

    Parameters
    ----------
    val : numpy array or int/float
    unit_in : str
        Unit the value val is in. Choose: 'mm', 'cm', 'm', 'km'.
    unit_out : str
        Unit the value val will be converted to. Choose: 'mm', 'cm', 'm', 'km'.

    Returns
    -------
    same type as val
        Converted values.

    """
    SI = {'mm':0.001, 'cm':0.01, 'm':1.0, 'km':1000.}
    units = list(SI.keys())
    in_units = True
    if unit_in not in units:
        warn("Input unit {unit_in} not available. Available units: {units}")
        in_units = False
    if unit_out not in units:
        warn("Output unit {unit_out} not available. Available units: {units}")
        in_units = False
    if in_units:
        return val*SI[unit_in]/SI[unit_out]
    else:
        return val

def isSubsidenceModel(Model):
    """SubsdenceModel util : Returns a boolean based an object type. Returns True if the object is of 
    a SubsidenceModel type. Returns False if not.

    Parameters
    ----------
    Model : object
        Any object.

    Returns
    -------
    bool
        Returns True if the object is a SubsidenceModel. Returns False if not.

    """
    
    return (str(type(Model)).endswith(".SubsidenceModel'>") or
            str(type(Model)) == "<class 'PySub.MergedModel.MergedModel'>" or
            str(type(Model)) == "<class 'PySub.BucketEnsemble.BucketEnsemble'>")
    
def isSubsidenceSuite(Model):
    """ModelSuite util : Returns a boolean based an object type. Returns True if the object is of 
    a ModelSuite type. Returns False if not.

    Parameters
    ----------
    Model : object
        Any object.

    Returns
    -------
    bool
        Returns True if the object is a ModelSuite. Returns False if not.

    """
    return str(type(Model)) == "<class 'PySub.SubsidenceSuite.ModelSuite'>"
    
def check_Name_df(df):
    """Returns the unique reservoir and model names entered in a Suite Excel 
    entry. Prints warnings when reservoirs are missing in some of the models,
    but are present in others.

    Parameters
    ----------
    df : pandas.DataFrame
        pandas DataFrame with the reservoir names on the index and at least the
        column 'Name' for the model name.

    Returns
    -------
    reservoir_names : np.ndarray, str
        An array with unique names of the reservoirs thourghout all of the entered
        models.
    model_names : np.ndarray, str
        An array woth the unique names of all of the models.

    """
    reservoir_names = np.unique(df.index)
    model_names = df['Name']
    unique_model_names = np.unique(model_names)
    for i, r in enumerate(reservoir_names):
        model_name = model_names[r].values
        except_if_not_unique('Model name(s)', model_name)
        reservoir_model_missing = a_missing_from_b(model_name, unique_model_names)
        if len(reservoir_model_missing) > 0:
            print(f'Warning: Model name(s) {reservoir_model_missing} have been defined, but miss from reservoir {r}')
        model_missing_reservoir = a_missing_from_b(unique_model_names, model_name)
        if len(model_missing_reservoir) > 0:
            print(f'Warning: Reservoir {r} has no data defined for model(s) {model_missing_reservoir}')
    return reservoir_names, model_names
        
def check_Name(model_parameters, reservoir_parameters, pressure_df):
    """Function to check if the model_parameter, reservoir_parameter and 
    pressure DataFrames are consisitent in naming the models.

    Parameters
    ----------
    model_parameters : pd.DataFrame
        pandas DataFrame with at least the model names in the index.
    reservoir_parameters : pd.DataFrame
        pandas DataFrame with the reservoirs in the index and at least the column
        'Name' for the model name.
    pressure_df : pd.DataFrame
        pandas DataFrame with the reservoirs in the index and at least the column
        'Name' for the model name.

    Raises
    ------
    Exception
        When the model name and/or reservoir names are not the same throughout 
        the dataframes.

    Returns
    -------
    np.array
        An array with the unique reservoir names throughout all of the models.

    """
    reservoir_names_reservoir, model_names_reservoirs = check_Name_df(reservoir_parameters)
    reservoirs_names_pressure, model_names_pressure= check_Name_df(pressure_df)
    model_names_model = model_parameters.index.values
    except_if_not_unique('Model name(s)', model_names_model)
    
    missing = check_if_all_uniques_are_same(model_names_reservoirs.values, model_names_pressure.values, model_names_model)
    
    if len(missing) > 0:
        raise(Exception(f'The reservoirs {missing} are missing from one of the entered data sheets.'))
    
    reservoir_names = reservoir_names_reservoir # if above exception is not triggered, it means the dataframe of reservoir parameters and pressure have the same reservoirs
    
    number_of_reservoirs = len(reservoir_names)
    for r in range(number_of_reservoirs):
        if model_names_reservoirs[r] != model_names_pressure[r]:
            raise(Exception(f'The data for reservoir and pressure show different model names for reservoir {reservoir_names[r]}'))
    
    return np.unique(model_names_reservoirs)

def order_df1_like_df2(df1, df2, column_name):
    """Reorder df1 over a column it shares with df2 so the order is the same.

    Parameters
    ----------
    df1 : pd.DataFrame
    df2 : pd.DataFrame
    column_name : str
        Column header both dataframes share.

    Returns
    -------
    df1 : pd.DataFrame
        reordered df1.

    """
    df1_original_index = df1.index
    df1 = df1.set_index(column_name)
    df1.reindex(df2[column_name])
    original_column_reordered = df1.index.copy()
    df1 = df1.set_index(df1_original_index)
    df1[column_name] = original_column_reordered
    return df1

def reorder_pressure_to_reservoir_parameters(reservoir_parameters, pressure_df):
    """Makes sure the reservoir_parameters and pressure pandas DataFrames have
    the same order in model names and reservoir names for easier comparison and 
    entry.

    Parameters
    ----------
    reservoir_parameters : pd.DataFrame
        pandas DataFrame with the reservoirs in the index and at least the column
        'Name' for the model name.
    pressure_df : pd.DataFrame
        pandas DataFrame with the reservoirs in the index and at least the column
        'Name' for the model name.

    Returns
    -------
    None.
        Sets the order in place, after calling this function the order of any 
        variables entered will have been made the same.

    """   
    reservoir_parameters["Name and Name type"] = reservoir_parameters.index + reservoir_parameters["Name"]
    pressure_df["Name and Name type"] = pressure_df.index + pressure_df["Name"]
    order_df1_like_df2(pressure_df, reservoir_parameters, "Name and Name type")
    
    reservoir_parameters.drop(columns = "Name and Name type", inplace = True)
    pressure_df.drop(columns = "Name and Name type", inplace = True)


def check_if_unique(values):
    """Check if all the values in the list or np.ndarray are unique.
    Returns a tuple with a boolean and a list. The boolean is True when all the 
    values in the list are unique and False when not. The list contains the values 
    that are not unique.
    

    Parameters
    ----------
    values : list/np.dnarray

    Raises
    ------
    Exception
        When not an iterable variable has been entered in values. Strings
        are not counted as iterables in this case.

    Returns
    -------
    all_unique : bool
        True when all the values in the list are unique and False when not.
    doubles : np.ndarray
        A single entry for each value that has been entered multiple times
        in the values array.

    """
    if not is_iterable(values):
        raise Exception(f'Cannot determine if the list {values} contains only unique values, for it is not iterable.')
    all_unique = True
    
    uniques, counts = np.unique(values, return_counts = True)
    doubles = uniques[counts > 1]
    if len(doubles) > 0:
        all_unique = False
    return all_unique, doubles

def not_None_or_empty(var):
    """Returns True if variable var is not None or an empty list. False if it is 
    either None or an empty list.

    Parameters
    ----------
    var : any object
    
    Returns
    -------
    valid : bool
        True if variable var is not None or an empty list. False if it is 
        either None or an empty list.

    """
    valid = True
    if var is None:
        valid = False
    elif is_iterable(var):
        if len(var) == 0:
            valid = False
    
    return valid

def check_if_all_uniques_are_same(*args):
    """Arguments are multiple lists. The entries of these lists are compared
    to return a list of values missing from each list.

    Parameters
    ----------
    *args : any number of lists

    Returns
    -------
    unique_missing : np.ndarray
        The missing values between all the lists.

    """
    missing = []
    for x in args:
        for y in args:
            missing.append([i for i in x if i not in y])
    unique_missing = np.unique(np.array(flatten_ragged_lists2D(missing)))
    return unique_missing

def except_if_not_unique(name, values):
    """Raise an exception if the entries in values are not unique.

    Parameters
    ----------
    name : str
        Parameter decription.
    values : list or np.ndarray

    Raises
    ------
    Exception
        When the entries in variable "values" are not unique.

    
    """
    check_unique = check_if_unique(values)
    if not check_unique[0]:
        raise Exception(f'{name} {check_unique[1]} are not unique.')

def isnan(value):
    value = np.array(value)
    if is_iterable(value ):
        return (value != value).any()
    else:
        return value != value

def is_iterable(var):
    """Returns True when the input parameter is any iterable and not a string.
    False if the input parameter is not an iterable or a is a string.
    """
    if isinstance(var, str):
        return False
    elif isSubsidenceModel(var) or isSubsidenceSuite(var):
        return False
    try:
        iter(var)
        return True
    except TypeError:
        return False

@np.vectorize
def is_number(s):
    """Returns True when the input parameter is numerical, False if not.
    """
    try:
        float(s)
        return True
    except:
        return False

def is_int_or_str(val):
    """Returns True when the input parameter is either a string or an integer, False if neither.
    """
    return is_str_or_int(val)

def is_str_or_int(val):
    """Returns True when the input parameter is either a string or an integer, False if neither.
    """
    if not isinstance(val, str):
        if not isinstance(val, int):
            return False
    return True

def is_list_of_strings(lst):
    """Returns True when the input parameter is a list of strings, False if not.
    """
    if is_iterable(lst):
        lst = list(lst)
        return bool(lst) and not isinstance(lst, str) and all(isinstance(elem, str) for elem in lst)
    else:
        return False

def is_list_of_numbers(lst):
    """Returns True when the input parameter is a list of numerical values, False if not.
    """
    if is_iterable(lst):
        if is_list_of_datetime(lst):
            return False
        lst = list(lst)
        return bool(lst) and all(is_number(lst))
    else:
        return False
    
def is_datetime(val):
    """Check if a variable is a datetime object.
    """
    return isinstance(val, (np.datetime64, datetime.datetime))

def is_list_of_datetime(lst):
    """Check if a list contains only datetime objects.
    """
    if is_iterable(lst):
        return all([is_datetime(val) for val in lst])
    else:
        return False

def a_missing_from_b(a, b):
    """Return the values that are not in list a, but do occur in list b.

    Parameters
    ----------
    a : list or np.ndarray
        The list on which will be checked if the values are in list b.
    b : list or np.ndarray
        The values of list a will be checked against this list to determin if they 
        are missing.

    Returns
    -------
    missing : np.ndarray
        An array of values that are in list b, but are missing in list a.

    """
    missing = np.array([i for i in a if i not in b])
    return missing

def pad_array(data):
    data = np.array(data)
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


def sort_df(df1, df2, sort_column = None):
    """Sort df1 over a specified column to give df1 the same order as df2.
    Occurs inplace: after running this functions df1 will have been sorted
    based on the column indicated by sort_column.
    
    Parameters
    ----------
    df1 : pd.DataFrame
        pandas DataFrame which will be reordered (reindexed) based on the 
        column of another dataframe, df2. Which column is indicated by sort_column.
    df2 : pd.DataFrame
        pandas DataFrame on which df1 will be reordered (reindexed). Df1 will be 
        reorder based on the column of this daaframe. Which column is indicated 
        by sort_column.
    sort_column : str, optional
        The name of the column on which df1 will be sorted by, based on the order
        of that same column name in df2. Default is None, when None, the columns over
        which df1 will be sorted is the index column.

    """
    if sort_column is None:       
        df1.reindex(index = df2.index)
    elif type(sort_column) == str:
        initial_index = df1.index
        initial_index_name = df1.index.name
        df1 = df1.set_index(sort_column)
        df1[initial_index_name] = initial_index
        df1.reindex(index = df2[sort_column])
        df1.set_index(initial_index_name)

def convert_to_datetime(timesteps):
    """Convert a list of integers to the corresponding integers in years as datetime
    objects. When not an integer, it must be another datatime format, from numpy or 
    pandas: np.datetime64 or pd.Timestamp.

    Parameters
    ----------
    timesteps : list or np.ndarray
        A list of values that can (somehow) be interpreted as a date.

    Raises
    ------
    Exception
        When not able to convert to a date, or the order of year/month/day is unclear.

    Returns
    -------
    timesteps : np.ndarray
        An array with the times as datetime objects.

    """
    if not is_iterable(timesteps):
        timesteps = [timesteps]
    try: timesteps = np.array(timesteps)
    except: raise Exception('Timesteps should be a np.ndarray with entries in integers for years, np.datetime64 or a pd.Timestamp object.')

    if np.issubdtype(timesteps.dtype, np.integer):
        
        check_for_year = np.min(timesteps)
        if check_for_year > 1700:                    
            timesteps = pd.to_datetime(timesteps, format = '%Y')
        else:
            timesteps = timesteps
    else:
        try: 
            timesteps = pd.to_datetime(timesteps)
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            raise Exception('The date or ont of the dates entered is too early (or late) for us computers to understand. Consider entering time as an integer to consider times before 1700.')
        except:
            raise Exception('Timesteps should be a np.ndarray with entries in integers for years or np.datetime64 or a pd.Timestamp object.')
    return timesteps

def round_up(x, dx):
    return dx * math.ceil(x / dx)

def round_down(x, dx):
    return dx * math.floor(x / dx)

def round_to_base(x, base=5):
    """Round value to the nearest multitude of the base.

    Parameters
    ----------
    x : int/float
        The value to be rounded towards the base.
    base : int/float, optional
        The value to which the value x will be rounded towards when of its multitudes. 
        The default is 5.

    Returns
    -------
    int/float
        nearest multitude of base from value x.

    """
    return base * round(x/base)

def stepped_space(start, end, step_size):
    """Generate a list of numerical values that increase from start, and increases its 
    next value by the amount indicated by step_size until the value is end or higher.
        
    Parameters
    ----------
    start : int/float
        Lowest value in the resulting list.
    end : int/float
        Highest value in the resulting list.
    step_size : int/float
        The increment size of the values between start and end in the result list.

    Returns
    -------
    result : list
        List containing the values from start, ascending with step size to end.
        The final value is higher than end if end - start is not a multitude of
        step_size.

    """
    result = []
    val = round_to_base(start, step_size)
    while val < end:
        result.append(val)
        val += step_size
    return result

def bounds_from_collection(shapes):
    """Get the highest and lowest values of x and y in a collection of coordinates.

    Parameters
    ----------
    shapes : list or np.ndarray
        list or array with the shape (m, 2) where m is the number of points and
        2 indicates the x coordinate first, and the y-coordinate second.

    Returns
    -------
    lowest_x : int/float
        The lowest x in the collection of points.
    lowest_y : int/float
        The lowest y in the collection of points.
    highest_x : int/float
        The highest x in the collection of points.
    highest_y : int/float
        The highest y in the collection of points.

    """
    lowest_x, lowest_y = shapes.min(axis = 0)
    highest_x, highest_y = shapes.max(axis = 0)
    return (lowest_x, lowest_y, highest_x, highest_y)

def bounds_from_bounds_collection(bound_collection):
    lowest_x, highest_x = np.min(bound_collection[:, 0]), np.max(bound_collection[:, 2])
    lowest_y, highest_y = np.min(bound_collection[:, 1]), np.max(bound_collection[:, 3])
    return [lowest_x, lowest_y, highest_x, highest_y]

def get_values_cross_section(A, B, data, num = 1000):
    """Sample points on a line between points A and B from the the values in data.
    When the sampled points do not fall on indeces or columns of data, the sampled values
    are interpolated.

    Parameters
    ----------
    A : tuple, int/float
        One of the points between which will be sampled. A tuple of column and index 
        values of data.
    B : tuple, int/float
        One of the points between which will be sampled. A tuple of column and index 
        values of data.
    data : 2D np.ndarray, int/float
        The data from which will be sampled..
    num : int, optional
        The amount of points that will be sampled on a line between points A and 
        B. The default is 1000.

    Returns
    -------
    values : np.ndarray, floats
        The (possibly interpolated) values of data, on a line between points A and B.

    """
    row = (A[0], B[0])
    col = (A[1], B[1])
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]
    values = ndimage.map_coordinates(data, np.vstack((row, col)), order = 1)
    return values

def flatten_ragged_lists2D(lists):
    """Sometimes lists contain lists (2D) lists that are of different size (are ragged) 
    and cannot be converted with numpy to arrays in a way that they can be flattened.
    This function will flatten 2D lists and numpy arrays that have ragged nesting.

    Parameters
    ----------
    lists : list/np.ndarray
        Lists to be flattened into one list or numpy array with 1 dimension.

    Returns
    -------
    result : list
        Flattened list.

    """
    if is_iterable(lists):
        result = []
        for l in lists:
            if is_iterable(l):
                for ll in l:
                    result.append(ll)
        return result

def flatten_ragged_lists3D(lists):
    """Sometimes lists contain lists (2D) lists that are of different size (are ragged) 
    and cannot be converted with numpy to arrays in a way that they can be flattened.
    This function will flatten 3D lists and numpy arrays that have ragged nesting.

    Parameters
    ----------
    lists : list/np.ndarray
        Lists to be flattened into one list or numpy array with 1 dimension.

    Returns
    -------
    result : list
        Flattened list.

    """
    if not is_iterable(lists):
        raise Exception('Must be list or numpy array.')
    result = []
    for l in lists:
        result = result + flatten_ragged_lists2D(l)
    return result

def effify(non_f_str: str):
    """ Turns a non-formatted string with {} to a formatted string.
    """
    return eval(f'f"""{non_f_str}"""')

def diff_xarray(values, prepend = None):
    """Apply the np.diff function over an xarray.
    """
    return np.diff(values, prepend = prepend)

def get_time_decay(delta_t, tau):
    """Apply the inverted exponential decay function.

    Parameters
    ----------
    delta_t : np.ndarray, floats
        The amount of time in a specified unit of time between set timesteps.
    tau : float
        Delay factor in the same unit of time as dt.

    Returns
    -------
    np.ndarray, float
        Values between 0 and 1 for the time decay mapping of the input dt.

    """
   
    time_decay = 1 - np.exp(-delta_t/tau)
    time_decay[np.isnan(time_decay)] = 0
    return time_decay
    
def moving_rigid_basement(start_basement, end_basement, reservoir_depth, time, viscous_tau):
    """Determines the rigid basement depth over time from the start basement,
    end basement and an inverse exponential decay function.

    Parameters
    ----------
    start_basement : float
        The start position of the rigid basement in meters below surface.
    end_basement : float
        The end position of the rigid basement in meters below surface.
    reservoir_depth : float
        The depth of the reservoir in meter below surface.
    time : np.ndarray, float
        The time in float (any unit the viscous_tau parameter is also in), over
        which the 
    viscous_tau : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    dt = (time - time[0]) 
    
    time_decay = get_time_decay(dt, viscous_tau) 
    
    moving_basement = (reservoir_depth/start_basement + 
                        (reservoir_depth/end_basement - reservoir_depth/start_basement) *
                        time_decay)
    
    return reservoir_depth/moving_basement

def fft_convolve2d(image, kernel):
    """2D convolution, using FFT
    https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy
    
    Convolve over an array using the kernel.
    https://en.wikipedia.org/wiki/Convolution

    Parameters
    ----------
    image : np.ndarray
        Array to be convolved over. (..., x, y)
    kernel : np.ndarray
        Kernel to convolve with. Must have the same shape as input array.
        (..., x, y)

    Returns
    -------
    result : np.ndarray
        Convoluted array.

    """
    fr = fft2(image).astype(np.complex64)
    fr2 = fft2(kernel).astype(np.complex64)
    m,n = fr.shape[-2:]
    cc = np.real(ifft2(fr*fr2).astype(np.complex64))
    cc = np.roll(cc, -m//2+1,axis=-2)
    result = np.roll(cc, -n//2+1,axis=-1)
    return result

def pad_kernel(array, kernel):
    """Pad the kernel with zeros to have the same size as the array. The kernel 
    will be in the middle of the padded array.

    Parameters
    ----------
    array : 2D np.ndarray, floats
        Array with the dimensions to which the kernel will be padded.
        (..., x, y)
    kernel : 2D np.ndarray, floats
        Kernel to be padded.
        (..., x, y)

    Returns
    -------
    padded : 2D np.ndarray, floats
        Padded kernel with the same dimensions as the input array, located in the 
        middle.

    """
    pad_shape = array.shape
    padded = np.zeros(pad_shape)
    kernel_shape = kernel.shape
    min_x = (pad_shape[-2] - kernel_shape[-2])//2
    max_x = min_x + kernel_shape[-2]
    min_y = (pad_shape[-1] - kernel_shape[-1])//2
    max_y = min_y + kernel_shape[-1]
    padded[min_x:max_x,min_y:max_y] = kernel
    return padded

def convolve(array, kernel):
    """Convolve a kernel over an array

    Parameters
    ----------
    array : 2D np.ndarray, floats
        Array to be convolved the time must be the final axes.
    kernel : 2D np.ndarray, floats
        Kernel to convolve the array with.
    
    Returns
    -------
    result : 2D np.ndarray, floats
        Convolved grid.

    """
    _kernel = pad_kernel(array, kernel)
    result = fft_convolve2d(array, _kernel)
         
    return result

def convolve_xarrays(array, kernel):
    """Convolve a kernel over an xarray

    Parameters
    ----------
    array : >2D xarray.DataArray, floats
        Must have the dimensions "x" and "y".
    kernel : 2D xarray.DataArray, floats
        Kernel to convolve the array with. Must have the dimensions "kx" and "ky"
    
    Returns
    -------
    result : 2D np.ndarray, floats
        Convolved grid.

    """
    return xr.apply_ufunc(
        convolve,
        array,
        kernel,
        input_core_dims = [['x', 'y'],['kx', 'ky']],
        exclude_dims = set(('kx', 'ky')),
        output_core_dims = [['x', 'y']],
        vectorize = True,
        )