import pandas as pd
import numpy as np
import os
import shutil
import json
import _pickle as pickle
from PySub import utils as _utils
from PySub import plot_utils as _plot_utils
from PySub import shape_utils as _shape_utils
from PySub import SubsidenceModelGas as _SubsidenceModelGas
from PySub import SubsidenceModelCavern as _SubsidenceModelCavern
from PySub import SubsidenceSuite as _SubsidenceSuite
from PySub import BucketEnsemble as _BucketEnsemble
from PySub import Points as _Points
from matplotlib import pyplot as plt

import sys
from pathlib import Path
for candidate in sys.path:
    if 'envs' in candidate:
        p = Path(candidate)
        environment_location = os.path.join(*p.parts[:p.parts.index('envs') + 2])
        break

os.environ['PROJ_LIB'] = os.path.join(environment_location, 'Library\share\proj')
os.environ['GDAL_DATA'] = os.path.join(environment_location, 'Library\share')

GRID_VARIABLES = ['dx', 'influence_radius', 'reservoirs', 'shapes', 'timesteps']

## Gas
RESERVOIR_VARIABLES = ['depth_to_basements', 'poissons_ratios', 'depths', 'thickness', 'compaction_coefficients',
                       'knothe_angles', 'tau', 'reference_stress_rates', 
                       'cmref', 'b', 'density', 'shapes']

COLUMN_NAMES = ["Depth to basement (m)", "Poisson's ratio (-)", "Depth to reservoir (m)", "Reservoir thickness (m)", "Compaction coefficient (1/bar)",
                "Knothe angle (°)", "Tau (s)", "Reference stress rate (Pa/year)", 
                "Reference compaction coefficient (1/bar)", "b", "Average density above reservoir (kg/m³)", "Shapefile location", ]

PARAMETER_TRANSLATOR = {par: text for par, text in zip(RESERVOIR_VARIABLES, COLUMN_NAMES)}
COLUMN_TRANSLATOR = {text: par for par, text in zip(RESERVOIR_VARIABLES, COLUMN_NAMES)}

MODEL_VARIABLES = RESERVOIR_VARIABLES + ['pressures']

## project_folder functions
def _write_variable(Model, var, file):
    if Model.hasattr(var):
        variable = Model[var]
        times = [Model.timesteps[t].strftime('%Y-%m-%d') for t in range(Model.number_of_steps)]
        if len(variable.shape) == 2:
            variable = np.array(variable)
            df = pd.DataFrame(variable, index = Model.reservoirs, columns = times)
            df.index.name = f'Uniform {var}'
        if len(variable.shape) == 4:
            if variable.max() > 0:
                max_indices = variable.isel(time = -1).argmax(dim = ('x', 'y'))
            else:
                max_indices = variable.isel(time = -1).argmin(dim = ('x', 'y'))
            max_value = []
            for r in range(Model.number_of_reservoirs):
                max_value.append(variable.isel(reservoir = r, x = max_indices['x'].isel(reservoir = r).values, y = max_indices['y'].isel(reservoir = r).values).values)
            max_value = np.array(max_value)
            df = pd.DataFrame(max_value, index = Model.reservoirs, columns = times)
            df.index.name = f'Max {var}'
        file.write(df.to_string())
        file.write('\n\n')

def _move_tree(root_src_dir, root_dst_dir):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                # in case of the src and dst are the same file
                if os.path.samefile(src_file, dst_file):
                    continue
                # os.remove(dst_file)
            shutil.copy(src_file, dst_dir)

## Export results
def export_contours(model, variable = 'subsidence', reservoir = None, time = -1, contour_levels = None, epsg = None):  
    """Save contours as a shapefile in project folder of .

    Parameters
    ----------
    model : SUbsidenceModel
    variable : str, optional
        model.grid attribute with at least the dimensions (y, x, reservoir, time). The default is 'subsidence'.
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a 
        list, multiple reservoirs will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    time : int, str, optional
        The index or name of the timestep you want to plot. If it is a 
        list, an Exception will occur. The default is -1, the final 
        timestep.
    contour_levels : list, float/int, optional
        The data vlaues to show the contours of. The default is None.
        When None, the contour levels will be based on the data and the 
        contour_steps parameter.

    """     
    if epsg is None: 
        raise Exception('Explicitly define the epsg parameter when exporting contour files.')
    if model.project_folder.project_folder is not None:
        time_index = _plot_utils.time_entry_to_index(model, time)
        reservoir_index = _plot_utils.reservoir_entry_to_index(model, reservoir)
        time_labels = model.timesteps[time_index]
        time_labels = _plot_utils.time_to_legend(time_labels, model)
        
        if contour_levels is None:
            levels = model.get_contour_levels(variable = variable)
        else:
            levels = contour_levels
        
        for it, t in enumerate(time_index):
            data = model[variable].isel(reservoir = reservoir_index, time = t).sum(dim = 'reservoir')
            contours = plt.contour(model.X, model.Y, data, levels = levels)
            file = model.project_folder.output_file(f'contours {time_labels[it]}.shp')
            geom = []
            levels = []
            for level, col in zip(contours.levels, contours.collections):
                # Loop through all polygons that have the same intensity level
                for contour_path in col.get_paths(): 
                    # Create the polygon for this intensity level
                    # The first polygon in the path is the main one, the following ones are "holes"
                    for cp in contour_path.to_polygons():
                        geom.append(cp)
                        levels.append(level)
            _shape_utils.save_polygon(geom, file, epsg, fields = levels)
            plt.close()
    else:
        print('Warning: Model has no project folder assigned. Contours have not been saved.')

def report(Model):
    """Save a short report as a text file in the output folde rof the project folder 
    of the model. The short report contains information about the input and some 
    selected information on the results.

    Parameters
    ----------
    Model : SubsidenceModel object

    Raises
    ------
    Exception
        When not enough information is available to make the report.

    """
    if not _utils.isSubsidenceModel(Model):
        raise Exception(f'First argument must be a SubsidenceModel, variable type is: {type(Model)}.')
    if (not Model.hasattr('timesteps') or 
        not Model.hasattr('reservoirs')):
        raise Exception('Not enough information to make report.')
    
    if Model.project_folder.project_folder is not None:
        fname = Model.project_folder.output_file('report.txt')
    else:
        print(f'Warning, no output file defined for Model {Model.name}, report not saved.')
        return
    
    with open(fname, 'w') as file:
        export_dfs = []
        export_titles = []
        
        # Model parameters
        model_parameters = ['name', 'number_of_reservoirs', 'number_of_steps', 'number_of_points', 
                            'number_of_observation_points', 'dx', 'dy', 'nx', 'ny', 'influence_radius', 
                            'subsidence_model', 'compaction_type']
        project_df = pd.Series()
        for project_var in model_parameters:
            label = project_var.replace('_', ' ')
            try: var = Model[project_var]
            except: var = ''
            project_df[label] = var
        project_df.index.name = 'Project parameters'
        
        export_titles.append(f'## Report {Model.name} ##')
        export_dfs.append(project_df)
        
        'reservoirs', 'timesteps', 'bounds',
        bounds_df = pd.Series(Model.bounds, index = ['Lowest x', 'lowest y', 'highest x', 'highest y'])
        bounds_df.index.name = 'Model bounds'
        
        reservoir_df = pd.DataFrame(index = Model.reservoirs)
        reservoir_df.index.name = 'Reservoir names'
        reservoir_vars = ['depth_to_basements', 'poissons_ratios', 'depths', 'thickness', 'compaction_coefficients',
                          'knothe_angles', 'tau', 'reference_stress_rates', 
                          'cmref', 'b', 'density']
        
        for var in reservoir_vars:
            try: reservoir_df[var] = Model[var]
            except: reservoir_df[var] = ''
        
        timestep_df = pd.Series(Model.timesteps)
        timestep_df.index.name = 'Timesteps'
        
        export_titles.append('')
        export_dfs.append(bounds_df)
        export_titles.append('')
        export_dfs.append(reservoir_df)
        export_titles.append('')
        export_dfs.append(timestep_df)
        
        if Model.hasattr('subsidence'):
            subsidence = Model['subsidence']
            max_indices = subsidence.isel(time = -1).argmin(dim = ('x', 'y'))
            max_value = []
            xs = []
            ys = []
            for r in range(Model.number_of_reservoirs):
                ix = max_indices['x'].isel(reservoir = r).values
                iy = max_indices['y'].isel(reservoir = r).values
                x = Model.x[ix]
                y = Model.y[iy]
                max_value.append(subsidence.isel(reservoir = r, x = ix, y = iy).values)
                xs.append(x)
                ys.append(y)
            max_value = np.array(max_value)
            times = [Model.timesteps[t].strftime('%Y-%m-%d') for t in range(Model.number_of_steps)]
            df = pd.DataFrame(max_value, index = Model.reservoirs, columns = times)
            df['X'] = xs
            df['Y'] = ys
            tupled = [Model.get_max_subsidence(time = t) for t in range(Model.number_of_steps)] 
            max_subsidence, x, y = np.array([(s, coord[0], coord[1]) for s, coord in tupled]).T
            max_subsidence = np.hstack((max_subsidence, x[-1], y[-1]))
            df.loc['Total'] = max_subsidence
            df.index.name = 'Max subsidence'
            
            export_titles.append('Subsidence (m)')
            export_dfs.append(df)
            
        if Model.hasattr('volume'):
            volume = Model['volume'].sum(dim = ('x', 'y'))
            volume = np.array(volume)
            volume = np.vstack((volume, np.sum(volume, axis = 0)))
            indices = Model.reservoirs + ['Total']
            times = [Model.timesteps[t].strftime('%Y-%m-%d') for t in range(Model.number_of_steps)]
            df = pd.DataFrame(volume, index = indices, columns = times)
            export_titles.append('Volume of subsidence bowl (m³)')
            export_dfs.append(df)
        
        df_to_text_file(file, export_dfs, titles = export_titles)
        
        summary_vars = ['compaction', 'pressures']
        labels = ['Compaction (m)', 'Pressures (bar)']
        for i, var in enumerate(summary_vars):
            if Model.hasattr(var):
                file.write(labels[i])
                _write_variable(Model, var, file)

def df_to_text_file(file, list_of_df, titles = None):
    """Create a text file from a pandas DataFrame or list of pandas DataFrames.

    Parameters
    ----------
    file : str
        Path to a valid location.
    list_of_df : list, pd.DataFrame or pd.Series
        A list of pandas DtafFrames to be exported.
    titles : list, str
        The titles will be presented above the dataframes. The default is None.
        When not None, the list must have the same length as list_of_df.

    """
    if not _utils.is_iterable(list_of_df):
        raise Exception(f'Invalid type {type(list_of_df)}. Must be an iterable with pandas DataFrames or Series.')
    elif not all([isinstance(df, (pd.DataFrame, pd.Series)) for df in list_of_df]):
        raise Exception(f'Invalid type {type(list_of_df)}. Must be an iterable with pandas DataFrames or Series.')
    
    if titles is not None:
        try:
            assert len(titles) == len(list_of_df)
        except:
            raise Exception('Titles must be a list of strings, or None, equal to the number of entered DataFrames.')
    else:
        titles = [None] * len(list_of_df)
    
    for df, title in zip(list_of_df, titles):
        if title is not None: file.write(f'{title} \n')
        file.write(df.to_string())
        file.write('\n\n')
        
def export_grid_to_ascii(x, y, cell_size, values, fname):
    """Export a grid with x, y and (2D) values to an ESRI ASCII file.
    
    Parameters
    ----------
    x : iterable, floats
        A 2D list or np.ndarray with x-coordinates. must have the same shape as 
        Y. Can have the same shape as values, unless values is 3D, then it needs 
        to correspond to the shape of the first two dimensions.
    y : iterable, floats
        A 2D list or np.ndarray with y-coordinates.
    values : iterable, floats
        A 2D or 3D list or np.ndarray with values corresponding
        to the x- and y-coordinates.
    cell_size : float.
        The grid cell size.
    fname : str
        Path to the file location.

    """
    file_name, file_extension = os.path.splitext(fname)
    asc_fname = file_name + '.asc'
    
    nrows, ncols = np.array(x).shape
    xllcorner = np.min(x)
    yllcorner = np.min(y)
    
    with open(asc_fname, 'w') as f:
        f.write(f'ncols     {ncols}\n')
        f.write(f'nrows     {nrows}\n')
        f.write(f"xllcorner     {xllcorner}\n")
        f.write(f"yllcorner     {yllcorner}\n")
        f.write(f"cellsize      {cell_size}\n")
        f.write("NODATA_value  0\n")
        for row_values in values:
            f.write(str(row_values).replace('[', '').replace(']', ''))
        
def export_grid_to_csv(x, y, values, columns, fname):
    """Export a grid with x, y and (2D or 3D) values to an xy csv file.
    
    Parameters
    ----------
    x : iterable, floats
        A 2D list or np.ndarray with x-coordinates. must have the same shape as 
        Y. Can have the same shape as values, unless values is 3D, then it needs 
        to correspond to the shape of the first two dimensions.
    y : iterable, floats
        A 2D list or np.ndarray with y-coordinates.
    values : iterable, floats
        A 2D or 3D list or np.ndarray with values corresponding
        to the x- and y-coordinates.
    columns : The title(s) of the column(s) corresponding to the values. If values 
        has 2 dimensions, only one column name is needed. If values has 3 dimensions,
        columns needs to have the same length as the 3rd dimension.
    fname : str
        Path to the file location.

    """
    df = pd.DataFrame(columns = ['X', 'Y'] + list(columns))
    df['X'] = np.array(x).flatten()
    df['Y'] = np.array(y).flatten()
    if len(values.shape) == 2:
        if isinstance(columns, str):
            pass
        elif _utils.is_iterable(columns):
            if len(columns) != 1:
                raise Exception(f'Incompatible number of columns: {len(columns)}. Must have same length as the 3rd number of dimensions as the values to export.: 1.')
        else:
            raise Exception(f'Invalid parameter columns: {columns}. Use a string for 2D variables or a list of strings for 2D or 3D variables.')
        df[columns] = np.array(values).flatten()
    elif len(values.shape) == 3:
        if _utils.is_iterable(columns):
            if len(columns) != values.shape[-1]:
                raise Exception(f'Incompatible number of columns: {len(columns)}. Must have same length as the 3rd number of dimensions as the values to export.: {values.shape[-1]}.')
        df[columns] = values.reshape((values.shape[0] * values.shape[1], values.shape[2]))
    else:
        raise Exception(f'Invalid number of dimension for the variable: {len(values.shape)}. Use 2 or 3 dimensions.')
    _utils.export_df(df, fname)


def attach_metadata(ds, crs):
    """Attach metadata for QGIS, 
    """
    if "dx" in ds.coords:
        ds = ds.drop_vars(["dx", "dy"])
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds.coords["x"].attrs = dict(
        long_name="x coordinate of projection",
        standard_name="projection_x_coordinate",
        axis="X",
        units="m",
    )
    ds.coords["y"].attrs = dict(
        long_name="y coordinate of projection",
        standard_name="projection_y_coordinate",
        axis="Y",
        units="m",
    )
    return ds

def to_netcdf(da, filename, epsg = 28992):
    """Export a netcdf file of your xarray data array.

    Parameters
    ----------
    da : xr.DataArray
        xarary data array with your variables. must have coordiantes x and y.
        Tiem optional. other layers are not recommended.
    filename : str
        Path to the file you want the netcdf to be saved at. Must end with .nc.
    epsg : int, optional
        EPSG code. The default is 28992.

    """
    if 'reservoir' in da.coords:
        # Layers other than time, x and y don't do well
        da = da.sum('reservoir', drop = True)
    crs = f'epsg:{epsg}'
    ds_coords = [i for i in ['time', 'y', 'x'] if i in da.coords]
    ds_coords = ds_coords + [i for i in da.coords if i not in ds_coords]
    ds = da.transpose(
        *ds_coords
        ).sortby(
            ds_coords
            ).to_dataset(name = da.name)
    ds = attach_metadata(ds, crs)        
    ds.to_netcdf(filename)
    
def export_netcdf(model, epsg = 28992):
    """Export a SubsidenceModel object to a netcdf. Each variable stored
    in the model grid will be exported as a seperate file.

    Parameters
    ----------
    model : PySub SubdinceModel object.
    epsg : int, optional
        EPSG code. The default is 28992.

    Returns
    -------
    None.

    """
    if _utils.isSubsidenceModel(model):
        grid = model.grid
        for name, var in grid.data_vars.items():
            f = os.path.join(
                model.project_folder.output_file(
                    f'{name}.nc')
                )
            to_netcdf(var, f, epsg = epsg)

def export_csv(model, variable = 'subsidence', reservoirs = None, time = None):
    """Export a variable from a SubsidenceModel object to a csv file.

    Parameters
    ----------
    model : SubsidenceModel object
        The model from which the variable will be exported.
    variable : str, optional
        The name of the variable that wil be exported. The default is 'subsidence'.
        The variable must be in the model object, if it is not, nothing will be 
        exported.
    reservoirs : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a 
        list, multiple reservoirs will be displayed. The default is None.
        When None, all reservoirs will be exported.
    time : int, str, optional
        The index or name of the timestep you want to plot. If it is a 
        list, the file will be exported for multiple timesteps. The default is 
        None, which indicates all timesteps will be exported.

    """
    if _utils.isSubsidenceModel(model):
        if not isinstance(variable, str):
            raise Exception('Variable must be given as a string variable.')
        else:
            if not model.hasattr(variable):
                print(f'Warning: Model has no attribute: {variable}. Not exported!')
                return
        reservoir_index = _plot_utils.reservoir_entry_to_index(model, reservoirs)
        time_index = _plot_utils.time_entry_to_index(model, time)
        
        values = np.array(
            model[variable].isel(
                reservoir = reservoir_index, time = time_index))
        columns = _plot_utils.time_to_legend(model.timesteps[time_index], model)
        if model.project_folder.project_folder is not None:
            for r in reservoir_index:
                fname = model.project_folder.output_file(f'{model.name} {model.reservoirs[r]} {variable}.csv')
                export_grid_to_csv(model.X, model.Y, values[:, :, r], columns, fname)
        else:
            print(f'Warning: Model {model.name} has no project folder set. Nothing exported.')
    else:
        print(f'Warning: {model} is no SubsidenceModel object. Nothing exported.')
        
def export_ascii(model, variable = 'subsidence', reservoirs = None, time = None):
    """Export a variable from a SubsidenceModel object to an Esri ascii grid file.

    Parameters
    ----------
    model : SubsidenceModel object
        The model from which the variable will be exported.
    variable : str, optional
        The name of the variable that wil be exported. The default is 'subsidence'.
        The variable must be in the model object, if it is not, nothing will be 
        exported.
    reservoirs : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a 
        list, multiple reservoirs will be displayed. The default is None.
        When None, all reservoirs will be exported.
    time : int, str, optional
        The index or name of the timestep you want to plot. If it is a 
        list, the file will be exported for multiple timesteps. The default is 
        None, which indicates all timesteps will be exported.

    """
    if _utils.isSubsidenceModel(model):
        if not isinstance(variable, str):
            raise Exception('Variable must be given as a string variable.')
        else:
            if not model.hasattr(variable):
                print(f'Warning: Model has no attribute: {variable}. Not exported!')
                return
        reservoir_index = _plot_utils.reservoir_entry_to_index(model, reservoirs)
        time_index = _plot_utils.time_entry_to_index(model, time)
        if len(time_index) > 1:
            print(f'Warning: Only 1 time index is allowed. Exporting final timestep in entries at: {model.timesteps[time_index[-1]]}')
        
        values = np.array(
            model[variable].isel(
                reservoir = reservoir_index, time = time_index[-1]))
        if model.project_folder.project_folder is not None:
            for r in reservoir_index:
                fname = model.project_folder.output_file(f'{model.name} {model.reservoirs[r]} {variable}.csv')
                export_grid_to_ascii(model.X, model.Y, values[:, :, r], model.dx, fname)
        else:
            print(f'Warning: Model {model.name} has no project folder set. Nothing exported.')
    else:
        print(f'Warning: {model} is no SubsidenceModel object. Nothing exported.')

def save(model, path = None):  
    """Save a SubsidenceModel object as a binary .smf file.

    Parameters
    ----------
    model : SubsidenceModel object
        The object to be saved.
    path : str, optional
        The path to where the file will be saved. The default is None. When None, 
        the file will be saved in the current working directory with the model name.

    Raises
    ------
    Exception
        When mode has no name attribute and path is None.

    """
    if _utils.isSubsidenceModel(model):
        models = [model]
    if _utils.isSubsidenceSuite(model):
        models = model
    for model in models:
        if model.project_folder.project_folder is not None:
            save_file_name = model.project_folder.save_file(model.name)
    
            save_file = save_file_name + '.smf'
        
            with open(save_file, 'wb') as save_file:
                model_dict = {key: value for key, value in model.__dict__.items() if key != 'csv_writer'}
                pickle.dump(model_dict, save_file)
        
            print(f'Model {model.name} saved at {save_file.name}')
        else:
            print(f'Warning: Model {model.name} has no project folder set. Model not saved.')

def export_tif(model, variable = 'subsidence', time = -1, reservoir = None, fname = None, epsg = 28992):
    """See export_tiff"""
    export_tiff(model, variable = variable, time = time, reservoir = reservoir, fname = fname, epsg = epsg)

def export_tiff(model, variable = 'subsidence', time = -1, reservoir = None, fname = None, epsg = 28992):
    """Export a variable from the SubsidenceModel object "model" to a .tif raster file.

    Parameters
    ----------
    model : SubsidenceModel object
    variable : str
        The name of a variable of the model that is dependant on x- and y-coordinates.
        If it has nu x- and y-coordinates, nothing will be exported.
    time : str/int/datetime object
        Specified time of which the data will be extracted from.
    reservoir : int/str or list of int/str, optional
        Index or name of the reservoirs that are taken into the extracted data. 
        The default is None, when None, all of the reservoirs in the model will 
        be taken into the exported file.
    fname : str, optional
        Path to the file where it will be saved. The default is None.
    epsg : int, optional
        The numeric integer value of the ESPG projection index. The default is 28992.

    Raises
    ------
    Exception
        When invalid input for timestep only 1 timestep is allowed, no list.

    """
    if not _utils.isSubsidenceModel(model):
        raise Exception(f'First variable must be of SubsidenceModel type. Is type {type(model)}.')
   
    try:
        model.grid[variable]
    except:
        raise Exception(f'Variable {variable} not in model.')
    
    file_basename = variable if fname is None else os.path.splitext(fname)[0]
    
    if 'time' in list(model[variable].coords.keys()):     
        steps = _plot_utils.time_entry_to_index(model, time)
        for step in steps:
            if model.timesteps.dtype == np.int64:
                time_label = f"{model.timesteps[step]}"
            elif np.issubdtype(model.timesteps.dtype, np.datetime64):
                time_label = f"{np.datetime_as_string(model.timesteps, unit = 'D')[step]}"
            _export_tiff(model, variable, step, reservoir = reservoir, fname = f'{file_basename} {time_label}.tif', epsg = epsg)
    else:
        _export_tiff(model, variable, 0, reservoir = reservoir, fname = f'{file_basename}.tif', epsg = epsg)

def _export_tiff(model, variable, time, reservoir = None, fname = None, epsg = 28992):
    output_file = model.project_folder.output_file(fname)
    if output_file is not None:
        coordinates = list(model.grid[variable].coords.keys())
        
        if not 'x' in coordinates or not model.built:
            print(f'Warning: No griddable parameter {variable}.')
            return
        
        if 'reservoir' in coordinates and 'time' in coordinates: 
            reservoir_index = _plot_utils.reservoir_entry_to_index(model, reservoir)
            time_index = _plot_utils.time_entry_to_index(model, time)
            if len(time_index) > 1:
                raise Exception(f'Enter only one timestep for the variable time. Current entry: {time}.')
            data = model.grid[variable].isel(reservoir = reservoir_index, time = time_index)
            if len(reservoir_index) > 1:
                data = data.sum(dim = 'reservoir')
            data = data.transpose('x', 'y', ...)
        elif 'reservoir' in coordinates:
            reservoir_index = _plot_utils.reservoir_entry_to_index(model, reservoir)
            data = model.grid[variable].isel(reservoir = reservoir_index)
            if len(reservoir_index) > 1:
                data = data.sum(dim = 'reservoir')
            data = data.transpose('x', 'y', ...)
        elif 'time' in coordinates: 
            time_index = _plot_utils.time_entry_to_index(model, time)
            if len(time_index) > 1:
                raise Exception(f'Enter only one timestep for the variable time. Current entry: {time}.')
            data = model.grid[variable].isel(time = time_index)
            data = data.transpose('x', 'y', ...)
        else:
            data = model.grid[variable]
            data = data.transpose('x', 'y')
        data = np.flip(np.rot90(np.array(data)), axis = 0)
        
        
        
        _shape_utils.save_raster(data, model.x, model.y, model.dx, model.dy, epsg, output_file)


def export_all(model, name = None, epsg = 28992):  
    """Save all the data in a project file, csv (when 2D and not dependant on x and y),
    .tif raster files and shapefiles.

    Parameters
    ----------
    model : SubsidenceModel
    save_dir : str, optional
        Path to a folder. The default is None. WHen None, uses the current working 
        directory.
    name : str, optional
        Name of the model if it has no name, will be the name of the .prj file and
        the dufolder storing all other data. The default is None.
    epsg : int, optional
        The epsg number of the coordinate system the data is in. The default is 
        28992.

    Raises
    ------
    Exception
        When using an invalid path.

    """
    if model.project_folder.project_folder is not None:
        output_folder = os.path.dirname(model.project_folder.output_file('dummy'))
        
        location_smf = os.path.join(output_folder, model.name + '.smf')
        save(model, path = location_smf)
        
        reservoir_df = pd.DataFrame(index = model._reservoirs)
        reservoir_df.index.name = 'Reservoir name'
        
        reservoir_vars = ['depth_to_basements', 'poissons_ratios', 'depths', 'thickness', 'compaction_coefficients',
                          'knothe_angles', 'tau', 'reference_stress_rate', 
                          'cmref', 'b', 'density']
        
        # Fill reservoir dataframe with all relevant values
        for var in reservoir_vars:
            try: reservoir_df[var] = model[var]
            except: reservoir_df[var] = ''
        
        try:
            for i, r in enumerate(model.reservoirs):
                shape_file_location = os.path.join(output_folder, str(r) + '.shp')
                _shape_utils.save_polygon(model.shapes[i], shape_file_location, epsg)
                reservoir_df.loc[r, 'Shapefile location'] = shape_file_location
        except:
            reservoir_df['Shapefile location'] = ''
        
        # Fill project file with all the relevant values
        project_vars = ['name', 'number_of_reservoirs', 'number_of_steps', 'number_of_points', 
                        'number_of_observation_points', 'dx', 'dy', 'influence_radius', 
                        'timesteps', 'bounds', 'subsidence_model', 'contourf_defaults',
                        'contour_defaults', 'clabel_defaults', 'colorbar_defaults',
                        'plot_defaults', 'shape_defaults', 'annotation_defaults',
                        'scatter_defaults', 'errorbar_defaults']
        
        project_df = pd.Series()
        for project_var in project_vars:
            project_df[project_var] = model[project_var]
        project_df.index.name = 'project parameters'
        try:
            project_df['compaction_model'] = model.compaction_model[0].type
        except:
            project_df['compaction_model'] = ''
            
        project_df['smf'] = location_smf
        
        # Save df's and set link in project file
        reservoir_csv = os.path.join(output_folder, 'reservoirs.csv')
        reservoir_df.to_csv(reservoir_csv)
        project_df['reservoirs'] = reservoir_csv    
    
        # Resolve point vars
        if model.hasattr('points'): 
            points_df = model.points.as_df
            points_csv = os.path.join(output_folder, 'points.csv')
            _utils.export_df(points_df, points_csv)
            project_df['points'] = points_csv
        if model.hasattr('observation_points'): 
            observations_df = model.observation_points.as_df
            observations_csv = os.path.join(output_folder, 'observations.csv')
            _utils.export_df(observations_df, observations_csv)
            project_df['observations'] = observations_csv
        
        if model.built:
            x = model.x
            y = model.y
            dx = model.dx
            dy = model.dy
            
            variables = list(dict(model.grid.variables).keys()) 
            coordinates = list(dict(model.grid.coords).keys()) 
            data_variables = [v for v in variables if v not in coordinates]
            for data_var in data_variables:
                coordinates = model.grid[data_var].dims 
                if 'x' in coordinates:
                    data = model.grid[data_var].values
                    
                    if 'reservoir' in coordinates:
                        total_data = model.grid[data_var].sum(dim = 'reservoir').values
                        total_fname = os.path.join(output_folder, 'total_' + data_var + '.tif')
                        _shape_utils.save_raster(total_data, x, y, dx, dy, epsg, total_fname)
                        for i, reservoir in enumerate(model.reservoirs):
                            fname = os.path.join(output_folder, reservoir + '_' + data_var + '.tif')
                            _shape_utils.save_raster(data[:, :, i], x, y, dx, dy, epsg, fname) 
                    else:
                        fname = os.path.join(output_folder, data_var + '.tif')
                        _shape_utils.save_raster(data, x, y, dx, dy, epsg, fname)
                        project_df[f'{data_var}'] = fname
                elif len(coordinates) == 2: # and there is no x in them (pressures and compaction)
                    data_csv = model.grid[data_var].values
                    data_df = pd.DataFrame(data_csv, index = model.reservoirs, columns = model.timesteps)
                    fname = os.path.join(output_folder, f'{data_var}.csv')
                    _utils.export_df(data_df, fname)
                    project_df[f'{data_var}'] = fname   
                elif len(coordinates) == 3: 
                    label_coordinate = [c for c in coordinates if c not in ['reservoir', 'time']][0]
                    labels = model.grid[label_coordinate].values
                    index = [[l] * model.number_of_reservoirs for l in labels]
                    index = _utils.flatten_ragged_lists2D(index)
                    reservoirs = [model.grid.reservoir.values] * model.number_of_points
                    reservoirs = _utils.flatten_ragged_lists2D(reservoirs)
                    data = []
                    for i in range(len(index)):
                        data.append(model.grid[data_var].loc[index[i], reservoirs[i]])
                    data = np.array(data)
                    data_df = pd.DataFrame(data, index = index, columns = model.timesteps)
                    data_df.insert(0, 'Reservoir name', reservoirs)
                    fname = os.path.join(output_folder, f'{data_var}.csv')
                    _utils.export_df(data_df, fname)
                    project_df[f'{data_var}'] = fname
        _utils.export_df(project_df, model.project_folder.save_file(f'{model.name}.prj'), extension = '.prj')

def _load_from_smf(file_name, model = _SubsidenceModelGas.SubsidenceModel):
    if not isinstance(file_name, str):
        raise Exception('Path to saved model location must be a string.')
    save_file_name, save_file_extension = os.path.splitext(file_name)
    if save_file_extension ==  '.smf':
        with open(save_file_name + '.smf', 'rb') as load_file:
            model = model(os.path.basename(file_name))
            model.__dict__ = pickle.load(load_file)
        return model
    else:
        raise Exception('Not a valid .smf file to load model from.')
        
def load(file_name, model = _SubsidenceModelGas.SubsidenceModel):
    """Load a model from a .smf or .prj file.

    Parameters
    ----------
    file_name : str
        path to the .smf or .prj file storing the model.

    Raises
    ------
    Exception 
        When no model is stored at the location or the extension of the file is 
        invalid (not .prj or .smf).

    Returns
    -------
    model : SubsidenceModel

    """
    if not isinstance(file_name, str):
        raise Exception('Path to saved model location must be a string.')
    save_file_name, save_file_extension = os.path.splitext(file_name)
    if save_file_extension ==  '.smf':
        model = _load_from_smf(file_name, model = model)
        return model
    elif save_file_extension == '.prj':
        df = pd.read_csv(file_name, index_col = 0)
        location_smf = df.loc['smf'][0]
        model = _load_from_smf(location_smf, model = model)
        return model
    else:
        raise Exception('Not a valid .smf or .prj file to load model from.')

## convert input formats
def excel_to_json(file, sheet_names = []):
    file_name, file_ext = os.path.splitext(file)
    json_file = "".join((file_name, ".json"))
    data = {}
    
    if len(sheet_names) == 0:
        xls = pd.ExcelFile(file)
        sheet_names = xls.sheet_names
    
    for sheet in sheet_names:
        try: df = pd.read_excel(file, sheet_name = sheet, index_col=0)
        except: df = pd.DataFrame()
        if len(df.index) != len(np.unique(df.index)):
            df = pd.read_excel(file, sheet_name = sheet, index_col=None)
        json_str = df.to_dict()
        # json_str = json_str.replace(',', ',\n')
        data[sheet] = json_str
    json_string = json.dumps(data, indent=4, default=str)
    with open(json_file, 'w') as f:
        f.write(json_string)
        
def json_to_df(json_file, time_columns = {}):
    with open(json_file, encoding = "utf8") as f:
        data = json.load(f)
    dataframes = {}
        
    for sheet in data.keys():
        df = pd.DataFrame(data[sheet])
        if sheet in list(time_columns.keys()):
            convert_columns = [c for c in time_columns[sheet] if c in df.columns]
            for col in convert_columns:
                df[col] = pd.to_datetime(df[col])
        dataframes[sheet] = df
    return dataframes

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
## import tools
def _fetch(df, col, index = None):
    try: 
        return_values = df[col].values
        if index is not None:
            return return_values[index]
        else:
            return return_values
    except:
        return None

## Build BucketEnsemble
def import_bucket_ensemble(import_paths):
    """Import the values for a bucket ensemble analysis where the buckets represent
    a variable which can be sampled with a certain percentage.

    Parameters
    ----------
    import_paths : list, str
        a list of paths where the paths are to valid excel files.

    Returns
    -------
    buckets : dict
        A dictionary where the keys are the number of reservoirs and the value
        is a VariableBucket object.
    timesteps : list
        The timesteps of the model, indicating when a reservoir undergoes the given 
        (change in) pressure.
    global_dx : float
        The grid size.
    global_influence_radius : float
        The distance in m after which the subsidence caused by
        compaction in a grid cell is deemed insignificant and is set to 0.
    global_compaction_model : str
        The types of compaction models as defined in 
        PySub.CompactionModels for each reservoir: # TODO: keep updated with added methods
            - linear
            - time-decay
            - ratetype
    global_subsidence_model : str
        Method of subsidence of the model. Currently available:
        - Nucleus of strain, Van Opstal 1974
        - Knothe, Stroka et al. 2011. 

    """
    
    
    if isinstance(import_paths, str):
        import_paths = [import_paths]
    elif not _utils.is_list_of_strings(import_paths):
        raise Exception(f'Invalid format for importing files from: {type(import_paths)}.')
    buckets = {}
    reservoir_names = {}
    for i, import_path in enumerate(import_paths):
        (model_parameters,
         reservoir_parameters,
         reservoir_probabilities,
         pressure_development_df,
         pressure_start_end_df,
         pressure_probability_df) = import_reservoir_bucket(import_path)
        (reservoir_name, reservoir_dict, dx, influence_radius, compaction_model, 
         subsidence_model, start, end, timesteps,
         ) = import_reservoir_bucket_from_dfs(model_parameters,
                                             reservoir_parameters,
                                             reservoir_probabilities,
                                             pressure_development_df,
                                             pressure_start_end_df,
                                             pressure_probability_df)
        
        
        if reservoir_name in reservoir_names.keys():
            raise Exception(f'{reservoir_name} is entered more then once. These two files are describing the same reservoir: \n {import_path} \n {reservoir_names[reservoir_name]}')
        
        reservoir_names[reservoir_name] = import_path
        buckets[reservoir_name] = reservoir_dict
        if i == 0:
            global_dx = dx
            global_influence_radius = influence_radius
            global_compaction_model = compaction_model
            global_subsidence_model = subsidence_model
            global_start = start
            global_end = end
            global_timesteps = timesteps
            
        else:
            if dx != global_dx: raise Exception('Incompatible cell sizes: {dx} and {global_dx}')
            if influence_radius != global_influence_radius: raise Exception('Incompatible influence radii: {influence_radius} and {global_influence_radius}')
            if compaction_model != global_compaction_model: raise Exception('Incompatible compaction models: {compaction_model} and {global_compaction_model}')
            if subsidence_model != global_subsidence_model: raise Exception('Incompatible subsidence models: {subsidence_model} and {global_subsidence_model}')
            if start != global_start: raise Exception('Incompatible start times: {start} and {global_start}')
            if end != global_end: raise Exception('Incompatible end times: {end} and {global_end}')
            if (timesteps != global_timesteps).any(): raise Exception('Incompatible end times: {timesteps} and {global_timesteps}')
                        
    for reservoir in buckets:
        buckets[reservoir].check_probabilities(additional_message = f'{reservoir}:')
    
    return buckets, timesteps, global_dx, global_influence_radius, global_compaction_model, global_subsidence_model

def import_reservoir_bucket(import_path):
    _, ext = os.path.splitext(import_path)
    if _utils.is_excel_file(import_path):
        return import_reservoir_bucket_from_excel(import_path)
    elif ext == '.json':
        return import_reservoir_bucket_from_json(import_path)
    else:
        raise Exception(f'Files with extension {ext} are not supported.')
        
def import_reservoir_bucket_from_dfs(model_parameters,
                                    reservoir_parameters,
                                    reservoir_probabilities,
                                    pressure_development_df,
                                    pressure_start_end_df,
                                    pressure_probability_df):
    probability_columns = ["Depth to basement probability (factor)",
               "Depth to reservoir probability (factor)",
               "Poisson's ratio probability (factor)",
               "Reservoir thickness probability (factor)",
               "Compaction coefficient probability (factor)",
               "Knothe angle probability (factor)",
               "Tau probability (factor)",
               "Reference stress rate probability (factor)",
               "Reference compaction coefficient probability (factor)",
               "b probability (factor)",
               "Average density above reservoir probability (factor)",
               "Shapefile location probability (factor)"]
    parameter_columns = ["Depth to basement (m)",
                         "Poisson's ratio (-)",
                         "Depth to reservoir (m)",
                         "Reservoir thickness (m)",
                         "Compaction coefficient (1/bar)",
                         "Knothe angle (°)",
                         "Tau (s)",
                         "Reference stress rate (Pa/year)",
                         "Reference compaction coefficient (1/bar)",
                         "b",
                         "Average density above reservoir (kg/m³)",
                         "Shapefile location"]
    (dx, influence_radius, compaction_model, subsidence_model, 
     start, end, reservoir_name) = get_model_parameters_from_df(model_parameters, bucket = True)
    
    reservoir_dict = _BucketEnsemble.VariableBuckets(variables = MODEL_VARIABLES)
    
    
    
    _utils.sort_df(reservoir_parameters, reservoir_probabilities)
    _utils.check_df(reservoir_probabilities, 'Reservoir parameter probability',
                columns = probability_columns,
                must_haves = ["Depth to reservoir probability (factor)",
                              "Poisson's ratio probability (factor)",
                              "Reservoir thickness probability (factor)",
                              "Compaction coefficient probability (factor)", 
                              "Shapefile location probability (factor)"],
                no_empty = False)
    _utils._check_df_all_filled_or_all_nan(reservoir_probabilities)
    
    _utils.check_df(reservoir_parameters, 'Reservoir parameters',
                columns = parameter_columns,
                must_haves = ['Shapefile location', 
                              'Depth to reservoir (m)', 
                              'Reservoir thickness (m)', 
                              'Compaction coefficient (1/bar)'],
                no_empty = False)
    _utils._check_df_all_filled_or_all_nan(reservoir_parameters)
    
    reservoir_versions = reservoir_parameters.index.values
    
    pressures_dict, timesteps = get_pressures_from_df(pressure_development_df,
                                                      pressure_start_end_df,
                                                      start, end)

    missing_reservoirs = list(_utils.a_missing_from_b(pressure_probability_df.index.values, pressures_dict.keys()))
    missing_pressures = list(_utils.a_missing_from_b(pressures_dict.keys(), pressure_probability_df.index.values))
    missing = missing_reservoirs + missing_pressures
    if len(missing) > 0:
        raise Exception(f'The version(s) {missing} is missing from either the pressure or pressure probability worksheet.')
            
    for v in reservoir_versions:
        for var, prob_col, par_col in zip(RESERVOIR_VARIABLES, probability_columns, parameter_columns):
            value = reservoir_parameters.loc[v][par_col]
            probability = reservoir_probabilities.loc[v][prob_col]
            if (_utils.isnan(value) or
                _utils.isnan(probability) or
                (probability == 0)):
                pass
            else:
                reservoir_dict[var]['Probabilities'].append(probability)
                reservoir_dict[var]['Values'].append(value)
    
    pressure_versions = pressures_dict.keys()
    for v in pressure_versions:
        reservoir_dict['pressures']['Values'].append(pressures_dict[v])
        reservoir_dict['pressures']['Probabilities'].append(pressure_probability_df.loc[v]['Probability'])
    return reservoir_name, reservoir_dict, dx, influence_radius, compaction_model, subsidence_model, start, end, timesteps

def import_reservoir_bucket_from_excel(import_path):
    model_parameters = pd.read_excel(import_path, sheet_name = "Model parameters", index_col = 0, header = 0)
    
    reservoir_parameters = pd.read_excel(import_path, sheet_name = "Reservoir parameters", index_col = 0, header = 0)
    
    try:
        pressure_development_df = pd.read_excel(import_path, sheet_name = "Pressure development", index_col = 0, header = 0)
    except:
        pressure_development_df = pd.DataFrame()
        
    try:
        pressure_start_end_df = pd.read_excel(import_path, sheet_name = "Pressure start-end", index_col = 0, header = 0)
    except:
        pressure_start_end_df = pd.DataFrame()
    pressure_probability_df = pd.read_excel(import_path, sheet_name = 'Pressure profile probability', index_col = 0, header = 0)
    reservoir_probabilities = pd.read_excel(import_path, sheet_name = "Reservoir parameter probability", index_col = 0, header = 0)
    return (model_parameters,
            reservoir_parameters,
            reservoir_probabilities,
            pressure_development_df,
            pressure_start_end_df,
            pressure_probability_df)

def import_reservoir_bucket_from_json(import_path):
    dataframes = json_to_df(import_path, time_columns = {'Model parameters': ['Start time', 'End time'],
                                                         'Reservoir parameters': [],
                                                         'Pressure start-end': ['Start time', 'End time'],
                                                         'Pressure development': []})
    model_parameters = dataframes["Model parameters"]
    reservoir_parameters = dataframes["Reservoir parameters"]
    reservoir_probabilities = dataframes["Reservoir parameter probability"]
    pressure_development_df = dataframes["Pressure development"]
    pressure_start_end_df = dataframes["Pressure start-end"]
    pressure_probability_df = dataframes['Pressure profile probability']
    return (model_parameters,
            reservoir_parameters,
            reservoir_probabilities,
            pressure_development_df,
            pressure_start_end_df,
            pressure_probability_df)

def build_bucket_ensemble(import_paths, name, project_folder, bounds = None):
    """Build a bucket ensemble from a collection of Excel templates representing a 
    reservoir, its parameters and probability of parameters.

    Parameters
    ----------
    import_paths : list, str
        a list of paths where the paths are to valid excel files.
    name : str
        Name of the model.
    project_folder : str, optional
        Path to a directory for the model input and results to be saved in. The default 
        is None. If the project_folder parameter is None, nothing will be saved.
    bounds : array-like, int/float, optional
        An array-like object with 4 values representing the corners of the 
        model. [0] lower x, [1] lower y, [2] upper x, [3] upper y.

    Returns
    -------
    BucketEnsemble Model.

    """
    
    (buckets, timesteps, dx, influence_radius, compaction_model, subsidence_model
     ) = import_bucket_ensemble(import_paths)
    
    model = _BucketEnsemble.BucketEnsemble(name, project_folder = project_folder)
    model.set_parameters(
        buckets, timesteps, dx, influence_radius, compaction_model, 
        subsidence_model, bounds = bounds
        )
    return model

## Build SubsidenceModelGas
def import_model(import_path):
    """Import model parameters from Excel ('.xls', '.xlsx', '.xlsm', '.xlsb', 
    '.odf', '.ods', '.odt') and JSON ('.json') files.
    
    Read the documentation on how to fil in these files correctly.
            

    Parameters
    ----------
    import_path : str
        Path to file.

    Returns
    -------
    dx : float/int
        Distance between grid nodes along the x-axis in m. The default is 
        None. Raises exception when None.
    dy : float/int
        Distance between grid nodes along the y-axis in m. The default is 
        None. When None, defaults to dx.
    influence_radius : float/int
        Distance from which the subsidence is set to 0 in m. The default 
        is None. Raises exception when None.
    compaction_model : list, str
        Can be a string for the compaction model type name to be used for all reservoirs, or
        a list of string with the model type to be used for each reservoir.
        The list must have the same length as the number of reservoirs in 
        the model.
        
        The types of compaction models as defined in 
        PySub.CompactionModels for each reservoir: # TODO: keep updated with added methods
            - linear
            - time-decay
            - ratetype
    subsidence_model : str
        Method of subsidence of the model. Currently available:
        - Nucleus of strain, Van Opstal 1974
        - Knothe, Stroka et al. 2011. 
    tau : list, float/int/str
        The time-decay constant for the time-decay method for each 
        reservoir in seconds of delay. The list must have the same length 
        as the number of reservoirs in the model. The default is None.
        When a string, it must be the path to a .tif raster file with 1 band.
    knothe_angles : list, float/int
        The influence angle in degrees for the knoth subsidence method for each 
        reservoir. The default is None.The list must have the same length 
        as the number of reservoirs in the model.
    reservoirs : list, str
        The names of each reservoir. The default is None. The list must have 
        the same length as the number of reservoirs in the model.
    shapefile_paths : list, str
        The location to the shapefiles or .tif raster files for each reservoir. 
        The list must have the same length as the number of reservoirs in 
        the model.The default is None. 
    depths : list float/int
        Depths to the top of each reservoirs in m. The list must have 
        the same length as the number of reservoirs in 
        the model. The default is None. Raises Exception when None.
    depth_to_basements : list, float/int
        Depth to the rigid basement for the van Opstal nucleus of strain 
        method in m. The list must have the same length as the number of reservoirs in 
        the model. The default is None. If None, the Geertsama solution will be used.
    poissons_ratios : list, float/int
        THe Poissons ratio for each reservoir for the van Opstal nucleus of strain 
        method in m. The list must have the same length as the number of reservoirs in 
        the model. The default is None. If None, an error will occur.
    reference_stress_rates : list, float/int/str
        Reference stress rates in bar/year. The list must have the 
        same length as the number of reservoirs in the model. Raises Exception 
        when None and a ratetype compaction model is used. When a string, it must 
        be the path to a .tif raster file with 1 band.
    density : list, float/int/str
        Bulk density of the ground above the reservoir in kg/m³. 
        The list must have the same length as the number of reservoirs in 
        the model. Raises Exception when None and a ratetype compaction model 
        is used. When a string, it must be the path to a .tif raster file with 1 band.
    cmref : list, float/int/str
        Reference compaction coefficient in 1/bar used for the ratetype compaction model. 
        The list must have the same length as the number of reservoirs in 
        the model. Raises Exception when None and a ratetype compaction model is used.
        When a string, it must be the path to a .tif raster file with 1 band.
    b : list, float/int/str
        Dimensionless constant for the stiffer reaction of sandstone over a 
        specific loading rate. The list must have the same length as the number 
        of reservoirs in the model. Raises Exception when None and a ratetype 
        compaction model is used. When a string, it must be the path to a .tif 
        raster file with 1 band.
    compaction_coefficients : list, floats/str
        Uniaxial compaction coefficient (Cm) in 1/bar. The list must have 
        the same length as the number of reservoirs in the model. The default 
        is None. Raises Exception when None. When a string, it must be the path 
        to a .tif raster file with 1 band.
    thickness : list, float/int/str
        Thickness of each reservoir in m. The list must have the same length 
        as the number of reservoirs in the model.The default is None. Raises 
        Exception when None. When a string, it must be the path to a .tif raster 
        file with 1 band.
    timesteps : list, np.datetime64
        The timestamps of each step in time. These need to be of equal 
        step length. Per year would be ['1990', '1991', etc.]. The default 
        is None. Raises Exception when None.
    pressures : np.ndarray, float/int
        a 2D array with the shape of (m, n), where m is the number of reservoirs, 
        (SubsidenceModel.number_of_reservoirs) and n is the number of timesteps 
        (SubsidenceModel.number_of_steps). Contains the pressure development 
        over time for eacht reservoir in bar. The pressures will be 
        homogonous over each reservoir.
    """
    file_name, ext = os.path.splitext(import_path)
    if _utils.is_excel_file(import_path):
        return import_model_from_excel(import_path)
    elif ext == '.json':
        return import_model_from_json(import_path)
    else:
        raise Exception(f'Files with extension {ext} are not supported.')
 
def import_model_from_json(import_path, _print = True):
    dataframes = json_to_df(import_path, time_columns = {'Model parameters': ['Start time', 'End time'],
                                                         'Reservoir parameters': [],
                                                         'Pressure start-end': ['Start time', 'End time'],
                                                         'Pressure development': [],
                                                         'Points': [],
                                                         'Observations': ['Time']})
    model_parameters = dataframes['Model parameters']
    reservoir_parameters = dataframes['Reservoir parameters'] 
    pressure_start_end_df = dataframes['Pressure start-end']
    pressure_development_df = dataframes['Pressure development'] 
    point_df = dataframes['Points']
    observations_df = dataframes['Observations']
    
    return import_model_from_dataframes(model_parameters, reservoir_parameters, pressure_start_end_df, 
                                        pressure_development_df, point_df, observations_df)


def import_model_from_excel(import_path, _print = True):
    
    if _print: print(f'Loading SubsidenceModel from {import_path}')
    
    model_parameters = pd.read_excel(import_path, sheet_name = "Model parameters", index_col = 0, header = 0)
    
    reservoir_parameters = pd.read_excel(import_path, sheet_name = "Reservoir parameters", index_col = 0, header = 0)
    
    try:
        pressure_development_df = pd.read_excel(import_path, sheet_name = "Pressure development", index_col = 0, header = 0)
    except:
        pressure_development_df = pd.DataFrame()
        
    try:
        pressure_start_end_df = pd.read_excel(import_path, sheet_name = "Pressure start-end", index_col = 0, header = 0)
    except:
        pressure_start_end_df = pd.DataFrame()
    
    point_df = pd.read_excel(import_path, sheet_name = "Points", index_col = 0, header = 0)
    observations_df = pd.read_excel(import_path, sheet_name = "Observations", index_col = 0, header = 0)
    
    return import_model_from_dataframes(model_parameters, reservoir_parameters, pressure_start_end_df, 
                                        pressure_development_df, point_df, observations_df)
        
def import_model_from_dataframes(model_parameters, reservoir_parameters, pressure_start_end_df, 
                                 pressure_development_df, point_df, observations_df):
    dx, influence_radius, compaction_model, subsidence_model, global_start, global_end = get_model_parameters_from_df(model_parameters)
    
    (tau, knothe_angles, reservoir_names, 
            shapefile_paths, depths, depth_to_basements, poissons_ratios,
            reference_stress_rates, density, cmref, b,
            compaction_coefficients, thickness) = get_reservoir_parameters_from_df(reservoir_parameters)
    
    pressures_dict, timesteps = get_pressures_from_df(pressure_development_df, pressure_start_end_df, global_start, global_end)
    
    pressures = [pressures_dict[reservoir] for reservoir in reservoir_names if reservoir in pressures_dict.keys()]
    
    compare_reservoir_names(reservoir_names, pressures_dict.keys())
    
   
    points = _Points.load_points_from_df(point_df)
    observation_points = _Points.load_observation_points_from_df(observations_df, 
                                                                 observation_column = 'Subsidence (m)',
                                                                 lower_error_column = 'Lower error (m)', 
                                                                 upper_error_column = 'Upper error (m)')
    
    
    return (dx, influence_radius, compaction_model, subsidence_model, tau, knothe_angles, reservoir_names, 
            shapefile_paths, depths, depth_to_basements, poissons_ratios,
            reference_stress_rates, density, cmref, b,
            compaction_coefficients, thickness, timesteps, pressures,
            points, observation_points)

def get_model_parameters_from_df(df, columns = ['Cell size (m)', 'Influence radius (m)', 'Compaction model', 'Subsidence model', 'Start time', 'End time'], bucket = False):
    _utils.check_df(df, 'Model parameters', exclude = ['Start time', 'End time'],
                    columns = columns)
    
    dx = _fetch(df, "Cell size (m)", index = 0)
    influence_radius = _fetch(df, "Influence radius (m)", index = 0)
    compaction_model = _fetch(df, "Compaction model", index = 0)
    subsidence_model = _fetch(df, "Subsidence model", index = 0)
    try: global_start = df["Start time"].values[0]
    except: global_start = None
    try: global_end = df["End time"].values[0]
    except: global_end = None
    if global_start is not np.nan: 
        global_start = _utils.convert_to_datetime(global_start)[0]
    else:
        global_start = None
    if global_end is not np.nan: 
        global_end =  _utils.convert_to_datetime(global_end)[0]
    else:
        global_end = None
    if bucket:
        reservoir_name = df.index.values[0]
        if not _utils.is_str_or_int(reservoir_name):
            raise Exception(f'{reservoir_name} is not a valid reservoir name.')
        return dx, influence_radius, compaction_model, subsidence_model, global_start, global_end, str(reservoir_name)
    else:
        return dx, influence_radius, compaction_model, subsidence_model, global_start, global_end

def _values_from_start_end(pressure_start_end_df, columns = [], exclude = []):
    pressure_df = pressure_start_end_df
    _utils.check_df(pressure_df, 'Pressure start-end',
                    columns = columns,
                    exclude = exclude)
    
    start_value = _fetch(pressure_df, 'Start pressure (bar)')
    end_value = _fetch(pressure_df, 'End pressure (bar)')
    start_time = _fetch(pressure_df, 'Start time')
    end_time = _fetch(pressure_df, 'End time')
    return start_value, end_value, start_time, end_time

def _pressure_from_start_end(pressure_start_end_df, columns = [], exclude = []):
    start_value, end_value, start_time, end_time = _values_from_start_end(pressure_start_end_df,
                                                                          columns = columns,
                                                                          exclude = exclude)
    
    pressures, timesteps = _utils.profile(start_value, 
                                               end_value, 
                                               _utils.convert_to_datetime(start_time), 
                                               _utils.convert_to_datetime(end_time))
    return pressures, timesteps

def _pressures_from_development(pressure_development_df):
    pressure_df = pressure_development_df
    _utils.check_df(pressure_df, 'Pressure development')
    
    pressures = np.array(pressure_df)
    timesteps = [_utils.convert_to_datetime(pressure_df.columns.values) for r in range(len(pressures))]
    return pressures, timesteps

def convert_pressure_start_end_to_profile(pressure_start_end_df):
    pressures, timesteps = _pressure_from_start_end(pressure_start_end_df, columns = [], exclude = [])
    reservoir_names = pressure_start_end_df['Reservoir name']
    global_start = min([min(i) for i in timesteps])
    global_end = max([max(i) for i in timesteps])
    filled_values = []
    filled_timesteps = []
    for i in range(len(pressures)):
        _pressures, _timesteps = _utils.pad_series(timesteps[i], pressures[i], global_start, global_end, pressures[i][0], pressures[i][-1])
        filled_values.append(_pressures)
        filled_timesteps.append(_timesteps)
    
    unique_timesteps = np.unique(_utils.flatten_ragged_lists2D(filled_timesteps))
    
    pressures = []
    timesteps = unique_timesteps
    for i in range(len(filled_values)):
        pressures.append(_utils.interpolate_over_time(filled_timesteps[i], filled_values[i], timesteps))
    pressures = {r:p for (r,p) in zip(reservoir_names, pressures)}
    
    pressure_df = pd.DataFrame(pressures, index = timesteps).T
    return pressure_df

def convert_pressure(import_path, sheet_name,
                     export_excel = False
                     ):
    pressure_start_end_df = pd.read_excel(import_path, sheet_name = sheet_name, index_col = None)
    
    pressure_df = convert_pressure_start_end_to_profile(pressure_start_end_df)
    if export_excel:
        write_file = os.path.splitext(import_path)[0] + '_pressure_profile.xlsx'
        pressure_start_end_df.to_excel(write_file, sheet_name = 'Pressure start-end',
                                       index = False)
        pressure_df.to_excel(write_file, sheet_name = 'Pressure_profile',
                             )
    return pressure_df

def get_pressures_from_df(pressure_development_df, pressure_start_end_df, global_start, global_end):
    pressure_development_df = pressure_development_df.dropna()
    if pressure_start_end_df.shape[0] == 0: # sheet pressure start-end has not been filled in, use pressure development
        if pressure_development_df.shape[0] == 0: # None have been filled in, raise error
            raise Exception('No pressure data has been entered. Fill either the worksheets "Pressure start-end" or "Pressure development".')
        pressures, timesteps = _pressures_from_development(pressure_development_df)
        reservoir_names = pressure_development_df.index.values
    elif pressure_development_df.shape[0] == 0: # sheet pressure development has not been filled in, use pressure start-end
        
        pressures, timesteps = _pressure_from_start_end(pressure_start_end_df,
                                                        columns = ['Start pressure (bar)', 'End pressure (bar)', 'Start time', 'End time'])
        reservoir_names = pressure_start_end_df.index.values
    else: # Both have been filled, use the profile in development to determine the profile for start-end
        start_value, end_value, start_time, end_time = _values_from_start_end(pressure_start_end_df,
                                                                              columns = ['Start pressure (bar)', 'End pressure (bar)'],
                                                                              exclude = ['Start time', 'End time'])
        compare_reservoir_names(pressure_development_df.index.values, pressure_start_end_df.index.values,
                                    name_df = "Pressure development", pressure_name_df = "Pressure start-end")
        if not np.isnan(start_time).any() and not np.isnan(end_time).any():
            print("""Warning: Values for "Start time" and "End time" are filled in, not using pressure development as profile!""")
            pressures, timesteps = _pressure_from_start_end(pressure_start_end_df)
            reservoir_names = pressure_start_end_df.index.values
        else:
            factor, timesteps = _pressures_from_development(pressure_development_df)
            factor = _utils.normalize(factor, axis = 1)
            nan_factor = np.sum(np.isnan(factor))
            if (nan_factor > 0).any(): 
                raise Exception(f'The reservoirs {list(pressure_development_df.index[np.where(nan_factor > 0)])} show no pressure change.')
            pressures = factor * (start_value - end_value)[:, None] + start_value[:, None]
            reservoir_names = pressure_start_end_df.index.values
    
    if global_start is None: global_start = min([min(i) for i in timesteps])
    if global_end is None: global_end = max([max(i) for i in timesteps])
    filled_values = []
    filled_timesteps = []
    for i in range(len(pressures)):
        _pressures, _timesteps = _utils.pad_series(timesteps[i], pressures[i], global_start, global_end, pressures[i][0], pressures[i][-1])
        filled_values.append(_pressures)
        filled_timesteps.append(_timesteps)
    
    unique_timesteps = np.unique(_utils.flatten_ragged_lists2D(filled_timesteps))
    
    pressures = []
    timesteps = unique_timesteps
    for i in range(len(filled_values)):
        pressures.append(_utils.interpolate_over_time(filled_timesteps[i], filled_values[i], timesteps))
    pressures = {r:p for (r,p) in zip(reservoir_names, pressures)}
    
    
    return pressures, timesteps

def get_reservoir_parameters_from_df(reservoir_parameters):
    _utils.check_df(reservoir_parameters, 'Reservoir parameters',
                    columns = ['Shapefile location', 'Depth to basement (m)', "Poisson's ratio (-)", 'Depth to reservoir (m)', 'Reservoir thickness (m)', 'Compaction coefficient (1/bar)', 'Tau (s)', 'Knothe angle (°)', 'Reference stress rate (Pa/year)', 'Average density above reservoir (kg/m³)', 'Reference compaction coefficient (1/bar)', 'b'],
                    must_haves = ['Shapefile location', 'Depth to reservoir (m)', 'Reservoir thickness (m)', 'Compaction coefficient (1/bar)'],
                    no_empty = False)
    _utils._check_df_all_filled_or_all_nan(reservoir_parameters)
    reservoir_names = reservoir_parameters.index.values
    shapefile_paths = _fetch(reservoir_parameters, "Shapefile location")
    depths = _fetch(reservoir_parameters, "Depth to reservoir (m)")
    depth_to_basements = _fetch(reservoir_parameters, "Depth to basement (m)")
    poissons_ratio = _fetch(reservoir_parameters, "Poisson's ratio (-)")
    compaction_coefficients = _fetch(reservoir_parameters, "Compaction coefficient (1/bar)")
    thickness = _fetch(reservoir_parameters, "Reservoir thickness (m)")
    tau = _fetch(reservoir_parameters, "Tau (s)")
    knothe_angles = _fetch(reservoir_parameters, "Knothe angle (°)")
    reference_stress_rates = _fetch(reservoir_parameters, "Reference stress rate (Pa/year)")
    density = _fetch(reservoir_parameters, "Average density above reservoir (kg/m³)")
    cmref = _fetch(reservoir_parameters, "Reference compaction coefficient (1/bar)")
    b = _fetch(reservoir_parameters, "b")
    return (tau, knothe_angles, reservoir_names, 
            shapefile_paths, depths, depth_to_basements, poissons_ratio,
            reference_stress_rates, density, cmref, b,
            compaction_coefficients, thickness)
            
def compare_reservoir_names(
        reservoir_names, pressure_reservoir_names,
        name_df = "reservoir parameter", pressure_name_df = "pressure"):
    missing_reservoirs = list(_utils.a_missing_from_b(reservoir_names, pressure_reservoir_names))
    missing_pressures = list(_utils.a_missing_from_b(pressure_reservoir_names, reservoir_names))
    missing = missing_reservoirs + missing_pressures
    if len(missing) > 0:
        raise Exception(f'The reservoirs {missing} is missing from either the {pressure_name_df} or {name_df} worksheet.')

def build_model(import_path, name = None, project_folder = None, bounds = None):
    """Build a model from the Excel template.

    Parameters
    ----------
    import_path : str
        Path to an input file.
    name : str
        Name of the model.
    project_folder : str, optional
        Path to a directory for the model input and results to be saved in. The 
        default is None. If the project_folder parameter is None, nothing will 
        be saved.
    bounds : array-like, int/float, optional
        An array-like object with 4 values representing the corners of the 
        model. [0] lower x, [1] lower y, [2] upper x, [3] upper y.
        
    Returns
    -------
    model : SubsidenceModel
        
    """
    (dx, influence_radius, compaction_model, subsidence_model, tau, knothe_angles, reservoir_names, 
            shapefile_paths, depths, depth_to_basements, poissons_ratios,
            reference_stress_rates, density, cmref, b,
            compaction_coefficients, thickness, timesteps, pressures,
            points, observation_points) = import_model(import_path)
    
    print('Building model...')
    if name is None: 
        name = os.path.basename(os.path.splitext(import_path)[0])
    else:
        if not isinstance(name, str):
            raise Exception(f'Invalid entry for Model name {name}. Use a string.')
    pf = project_folder 
    model = _SubsidenceModelGas.SubsidenceModel(name, pf)
    model.set_parameters(dx = dx, 
                        influence_radius = influence_radius, 
                        compaction_model = compaction_model, 
                        subsidence_model = subsidence_model,
                        knothe_angles = knothe_angles, tau = tau, 
                        reservoirs = reservoir_names, 
                        shapefile_paths = shapefile_paths, 
                        depths = depths, 
                        depth_to_basements = depth_to_basements, 
                        poissons_ratios = poissons_ratios,
                        reference_stress_rates = reference_stress_rates, 
                        density = density, 
                        cmref = cmref, 
                        b = b,
                        compaction_coefficients = compaction_coefficients, 
                        thickness = thickness,
                        timesteps = timesteps, 
                        pressures = pressures,
                        bounds = bounds)
    model.project_folder.write_to_input(import_path)
    model.set_points(points)
    model.set_observation_points(observation_points)
    
    model.mask_reservoirs()
    
    print('Model built!')
    return model


## Build SubsidenceModelCavern 
def import_cavern_model(import_path, cumulative_volume = False):
    _, ext = os.path.splitext(import_path)
    if _utils.is_excel_file(import_path): 
        return import_cavern_model_from_excel(import_path, cumulative_volume=cumulative_volume)
    elif ext == 'json':
        return import_cavern_model_from_json(import_path, cumulative_volume=cumulative_volume)
    else:
        raise Exception(f'Files with extension {ext} are not supported.')
        
def import_cavern_model_from_excel(import_path, cumulative_volume = False):
    model_parameters = pd.read_excel(import_path, sheet_name = "Model parameters", index_col = 0, header = 0)
    
    
    cavern_parameters = pd.read_excel(import_path, sheet_name = "Cavern parameters", index_col = 0, header = 0)
    
    
    squeeze_df = pd.read_excel(import_path, sheet_name = "Volume change development", index_col = 0, header = 0)
    point_df = pd.read_excel(import_path, sheet_name = "Points", index_col = 0, header = 0)
    observations_df = pd.read_excel(import_path, sheet_name = "Observations", index_col = 0, header = 0)
    
    return import_cavern_model_from_dfs(model_parameters, cavern_parameters, squeeze_df, 
                                        point_df, observations_df, cumulative_volume = cumulative_volume)

def import_cavern_model_from_json(import_path, cumulative_volume = False):
    dataframes = json_to_df(import_path, 
                            time_columns = {'Observations': 'Time'})
    model_parameters = dataframes["Model parameters"]
    cavern_parameters = dataframes["Cavern parameters"]
    squeeze_df = dataframes["Volume change development"]
    point_df = dataframes["Points"]
    observations_df = dataframes["Observations"]
    return import_cavern_model_from_dfs(model_parameters, cavern_parameters, squeeze_df, 
                                        point_df, observations_df, cumulative_volume = cumulative_volume)

def import_cavern_parameters_from_df(cavern_parameters):
    _utils.check_df(cavern_parameters, 'Cavern parameters',
                    columns = ['X', 'Y', 'Depth (m)', 'Length (m)', 'Depth to basement (m)', "Poisson's ratio (-)", 'Knothe angle (°)', 'Depth to basement moved (m)', 'Viscous tau (s)'],
                    must_haves = ['X', 'Y', 'Depth (m)', 'Length (m)'],
                    no_empty = False)
    _utils._check_df_all_filled_or_all_nan(cavern_parameters)
    reservoir_names = cavern_parameters.index.values
    coordinates = cavern_parameters[['X', 'Y']].values
    
    # Necesary
    depths, lengths, depth_to_basements, poissons_ratios, knothe_angles = cavern_parameters[['Depth (m)', 
                                                                               'Length (m)', 
                                                                               'Depth to basement (m)', 
                                                                               "Poisson's ratio (-)",
                                                                               'Knothe angle (°)']].values.T
    
    # Optional
    # Moving rigid basement
    depth_to_basements_moved, viscous_tau = cavern_parameters[['Depth to basement moved (m)', 
                                                                  'Viscous tau (s)']].values.T
    
    if np.isnan(depth_to_basements_moved).any() and not np.isnan(viscous_tau).any():
        print('Warning: Column "Depth to basement moved (m)" has not been filled in but is required when the column "Viscous tau (s)" is.')
        print('Warning: No moving rigid basement is used.')
        viscous_tau = None
    elif not np.isnan(depth_to_basements_moved).any() and np.isnan(viscous_tau).any():
        print('Warning: Column "Viscous tau (s)" has not been filled in but is required when the column "Depth to basement moved (m)" is.')    
        print('Warning: No moving rigid basement is used.')
        depth_to_basements_moved = None
    elif np.isnan(depth_to_basements_moved).any() and np.isnan(viscous_tau).any():
        depth_to_basements_moved = None
        viscous_tau = None
        
    return (reservoir_names, 
            coordinates, depths, depth_to_basements, poissons_ratios,
            depth_to_basements_moved, viscous_tau,
            knothe_angles, lengths)

def import_squeeze_volumes_from_df(squeeze_df, cumulative_volume = False):
    timesteps = squeeze_df.columns.values
    squeeze = []
    for cavern in squeeze_df.index:
        squeeze.append(squeeze_df.loc[cavern].values)
    if not cumulative_volume:
        cum_squeeze = np.cumsum(np.array(squeeze), axis = 1)
    else:
        cum_squeeze = np.array(squeeze)
    return timesteps, cum_squeeze

def import_cavern_model_from_dfs(model_parameters, cavern_parameters, squeeze_df, 
                                 point_df, observations_df, cumulative_volume = False):
    (dx, influence_radius, _, subsidence_model, start_time, 
     end_time) = get_model_parameters_from_df(model_parameters, 
                                              columns = ['Cell size (m)', 
                                                         'Influence radius (m)', 
                                                         'Subsidence model'])
    (reservoir_names, 
     coordinates, depths, depth_to_basements, poissons_ratios,
     depth_to_basements_moved, viscous_tau,
     knothe_angles, lengths) = import_cavern_parameters_from_df(cavern_parameters)     
    
    compare_reservoir_names(reservoir_names, squeeze_df.index)
    timesteps, cum_squeeze = import_squeeze_volumes_from_df(squeeze_df, cumulative_volume = cumulative_volume)    

    points = _Points.load_points_from_df(point_df)
    observation_points = _Points.load_observation_points_from_df(observations_df, 
                                                                 observation_column = 'Subsidence (m)',
                                                                 lower_error_column = 'Lower error (m)', 
                                                                 upper_error_column = 'Upper error (m)')   
    return (dx, influence_radius, subsidence_model, reservoir_names, 
            coordinates, depths, depth_to_basements, poissons_ratios,
            depth_to_basements_moved, viscous_tau,
            knothe_angles,
            lengths, timesteps, cum_squeeze,
            points, observation_points)                  

def build_cavern_model(import_path, 
                       name = None,
                       project_folder = None,
                       cumulative_volume = False,
                       bounds = None):
    """Build a SubsidenceModelCavern.SubsidenceModel from the relevant template.

    Parameters
    ----------
    import_path : str
        Path to the Excel template file.
    name : str
        Name of the model.
    project_folder : str, optional
        Path to a directory for the model input and results to be saved in. The default 
        is None. If the project_folder parameter is None, nothing will be saved.
    cumulative_volume : bool, optional
        The squeeze volume can be entered as a volume change per timestep, or the
        volume change up until that timestep (cumulative volume change). When this
        variable is set to True, the entries will be treated as the volume change that 
        has occurred untill that point. If False, the entries will be treated as
        the volume change in that timestep. The default is False.
    bounds : array-like, int/float, optional
        An array-like object with 4 values representing the corners of the 
        model. [0] lower x, [1] lower y, [2] upper x, [3] upper y.

    Returns
    -------
    model : SubsidenceModel
        SubsidenceModelCavern.SubsidenceModel object.

    """
    (dx, influence_radius, subsidence_model, reservoir_names, 
    coordinates, depths, depth_to_basements, poissons_ratios,
    depth_to_basements_moved, viscous_tau, 
    knothe_angles,
    lengths, timesteps, input_cumulative_volume,
    points, observation_points) = import_cavern_model_from_excel(import_path, cumulative_volume = cumulative_volume)
    
    print('Building model...')
    if name is None: 
        name = os.path.basename(os.path.splitext(import_path)[0])
    else:
        if not isinstance(name, str):
            raise Exception(f'Invalid entry for Model name {name}. Use a string.')
            
    model = _SubsidenceModelCavern.SubsidenceModel(name, project_folder = project_folder)
    model.set_dx(dx)
    
    model.set_influence_radius(influence_radius)
    model.set_timesteps(timesteps)
    model.set_reservoirs(reservoir_names)
    model.set_shapes(coordinates)
    
    model.set_bounds(bounds)
    model.build_grid()
    model.set_depths(depths)
    model.set_lengths(lengths)
    model.set_depth_to_basements_moved(depth_to_basements_moved)
    model.set_viscous_tau(viscous_tau)
    model.set_subsidence_model_parameters(subsidence_model, knothe_angles = knothe_angles, 
                                 depth_to_basements = depth_to_basements, 
                                 poissons_ratios = poissons_ratios)
    model.set_points(points)
    model.set_observation_points(observation_points)
    model.set_volume_change(input_cumulative_volume)
    
    model.mask_reservoirs()
    model.project_compaction()
    print('Model built!')
    return model
    
## ModelSuite
def build_suite(import_path, name = None, project_folder = None, model_names = None, bounds = None, all_same_bounds = True):
    """Build a model from the Excel template.

    Parameters
    ----------
    import_path : str or list of str
        Path to the Suite Excel template file, or multiple Model Excel template.
    name : str
        Name of the Suite.
    project_folder : str, optional
        Path to a directory for the model input and results to be saved in. The default 
        is None. If the project_folder parameter is None, nothing will be saved.
    model_names : list, str, optional
        A list with the same length as the number of models. In this list must be 
        the names the models are going to have. The default is None, when None,
        the model names are going to be the Excel file names.
    bounds : array-like, int/float, optional
        An array-like object with 4 values.
        [0] lower x, [1] lower y, [2] upper x, [3] upper y.
        Default is None, when None, the bounds of the models will be determined
        on the sizes of the shapes in all of the models. Based on all_same_bounds
        it will be determined if the bounds will all have the same bounds.
    all_same_bounds : bool, optional
        The default is True. When True, the shapes of all the models will be
        analized to determin the minimum and maximum x and y values to set
        the bound for each model such that it will fit all the models. If False, 
        each model will build their own bounds, without being aware of
        the other models. THis can result in models with different shapes in
        the same ModelSuite.
        
    Returns
    -------
    suite : ModelSuite
        
    """
    if isinstance(import_path, str):
        suite_name = name if name != None else os.path.basename(os.path.dirname(import_path))
        pf = project_folder if project_folder is not None else os.path.dirname(import_path)
        suite = _SubsidenceSuite.ModelSuite(suite_name, project_folder = pf)
   
        (names, multi_dx, multi_influence_radius, multi_compaction_model, multi_subsidence_model, multi_tau, 
        multi_knothe_angles, reservoir_names, multi_shapefile_paths, multi_depths, 
        multi_depth_to_basements, multi_reference_stress_rates, multi_density, 
        multi_cmref, multi_b, multi_compaction_coefficients, multi_thickness, 
        multi_timesteps, multi_pressures, points, observation_points) = import_suite(import_path)
        
        suite.initiate_models(names)
        suite.set_observation_points(observation_points)
        suite.set_points(points)
        suite.set_reservoirs(reservoir_names)
        suite.set_timesteps(multi_timesteps)
        suite.set_dx(multi_dx)
        suite.set_dy(multi_dx)
        suite.set_influence_radius(multi_influence_radius)
        suite.set_compaction_model_parameters(multi_compaction_model, multi_compaction_coefficients, 
                                   multi_tau, multi_reference_stress_rates, multi_density, 
                                   multi_cmref, multi_b)
        suite.set_subsidence_model_parameters(multi_subsidence_model, knothe_angles = multi_knothe_angles, 
                                   depth_to_basements = multi_depth_to_basements )
        
        suite.set_shapefiles(multi_shapefile_paths)
        suite.set_depths(multi_depths)
        suite.set_thickness(multi_thickness)
        
        suite.set_bounds(bounds = bounds, all_same_bounds = all_same_bounds) # or: Model.set_bounds(bounds = None) to build from shapefiles, if set
        suite.set_pressure(multi_pressures)
        
        print('Build ModelSuite...')
        suite.build_grid()
        suite.mask_reservoirs()
        suite.assign_compaction_parameters()
    elif isinstance(import_path, list):
        suite_name = name if name is not None else os.path.basename(os.path.dirname(import_path[0]))
        pf = project_folder # if project_folder is not None else os.path.dirname(import_path[0])
        suite = _SubsidenceSuite.ModelSuite(suite_name, project_folder = pf)
        if model_names is None: 
            model_names = [os.path.splitext(os.path.basename(path))[0] for path in import_path]
        elif not _utils.is_list_of_strings(model_names):
            raise Exception('Enter a list of names for each model or enter None to have the model take on the name of the imported Excel file.')
        elif len(model_names) < 1:
            raise Exception('The list with model names requires at least one entry.')
        else:
            pass
        list_of_models = []
        print('Building ModelSuite...')
        for model_name, model_path in zip(model_names, import_path):
            model = build_model(model_path, name = model_name, project_folder = suite.project_folder.project_folder)
            
            list_of_models.append(model)
        suite.set_models(list_of_models)
    print('Built ModelSuite!')
    return suite

def import_suite(import_path):
    """Import Suite parameters from Excel or JSON files.
    
    See the documentation on how to make these files.
            

    Parameters
    ----------
    import_path : str or list of str
        Path to the Suite Excel template file, or multiple Model Excel template.

    Returns
    -------
    dx : list, float/int
        Distance between grid nodes along the x-axis in m. The default is 
        None. Raises exception when None. The list has the length of the number of models.
    dy : list, float/int
        Distance between grid nodes along the y-axis in m. The default is 
        None. When None, defaults to dx. The list has the length of the number of models.
    influence_radius : list, float/int
        Distance from which the subsidence is set to 0 in m. The default 
        is None. Raises exception when None. The list has the length of the number 
        of models.
    compaction_model : list, str
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).
        Can ba a string for the compaction model type name to be used for all reservoirs, or
        a list of string with the model type to be used for each reservoir, in a list for
        each model. 
        
        The types of compaction models as defined in 
        PySub.CompactionModels for each reservoir: # TODO: keep updated with added methods
            - linear
            - time-decay
            - ratetype
        The default is None. Raises Exception when None.
    subsidence_model : list, str
        List with a length of the number of models.    
        Method of subsidence of the model. Currently available:
        - Nucleus of strain, Van Opstal 1974
        - Knothe, Stroka et al. 2011. 
        The default is 'nucleus of strain'.
    tau : list, float/int/str
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).    
        The time-decay constant for the time-decay method for each 
        reservoir in seconds of delay. The default is None.
        When a string, it must be the path to a .tif raster file with 1 band.
    knothe_angles : list, float/int
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged). The influence angle in 
        degrees for the knoth subsidence method. The default is None.
    reservoirs : list, str
        The names of each reservoir. The default is None. A list with the 
        length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).
    shapefile_paths : list, str
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).    
        The location to the shapefiles or .tif raster files for each reservoir. 
        The default is None. 
    depths : list float/int
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).
        Depths to the top of each reservoirs in m. The list must have 
        the same length as the number of reservoirs in 
        the model. The default is None. Raises Exception when None.
    depth_to_basements : list, float/int
    A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).    
        Depth to the rigid basement for the van Opstal nucleus of strain 
        method in m. The default is None. Raises Exception when None.
    reference_stress_rates : list, float/int/str
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).    
        Reference stress rates in bar/year. Raises Exception 
        when None and a ratetype compaction model is used. When a string, it must 
        be the path to a .tif raster file with 1 band.
    density : list, float/int/str
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).
        Bulk density of the ground above the reservoir in kg/m³. 
        Raises Exception when None and a ratetype compaction model 
        is used. When a string, it must be the path to a .tif raster file with 1 band.
    cmref : list, float/int/str
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).
        Reference compaction coefficient in 1/bar used for the ratetype compaction model. 
        Raises Exception when None and a ratetype compaction model is used.
        When a string, it must be the path to a .tif raster file with 1 band.
    b : list, float/int/str
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).
        Dimensionless constant for the stiffer reaction of sandstone over a 
        specific loading rate. Raises Exception when None and a ratetype 
        compaction model is used. When a string, it must be the path to a .tif 
        raster file with 1 band.
    compaction_coefficients : list, floats/str
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).
        Uniaxial compaction coefficient (Cm) in 1/bar. The default 
        is None. Raises Exception when None. When a string, it must be the path 
        to a .tif raster file with 1 band.
    thickness : list, float/int/str
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can be ragged).
        Thickness of each reservoir in m. The default is None. Raises 
        Exception when None. When a string, it must be the path to a .tif raster 
        file with 1 band.
    timesteps : list, np.datetime64
        A list with the length of the number of models, containing lists with the 
        number of reservoirs in each model (can not be ragged).
        The timestamps of each step in time. These do not need to be of equal 
        step length. Per year would be ['1990', '1991', etc.]. The default 
        is None. Raises Exception when None.
    pressures : np.ndarray, float/int
        a 3D array with the shape of (l, m, n), where l is the number of models, 
        m is the number of reservoirs, (SubsidenceModel.number_of_reservoirs) 
        and n is the number of timesteps (SubsidenceModel.number_of_steps). 
        The pressures in bar will be homogonous over each reservoir.
    """
    print(f'Loading ModelSuite object from {import_path}')
    if not _utils.is_iterable(import_path):
        if isinstance(import_path,str):
            import_path = [import_path]
        else:
            raise Exception(f'Invalid data type: {type(import_path)}')
    for i, model_path in enumerate(import_path):
        if i == 0:
            data = [[os.path.splitext(os.path.basename(path))[0] for path in import_path]]
            model_data = import_model(model_path)
            for j in range(len(model_data)):
                data.append([model_data[j]])
        else:
            model_data = import_model(model_path)
            for j in range(len(model_data)):
                data[j + 1].append(model_data[j])           
    return data
        

def seperate_models_from_df(model_parameters, reservoir_parameters, pressure_df, point_df, observations_df):
    """When a ModelSuite is loaded from pandas dataframes need they to be ordered 
    and checked, which is done in this function. 
    
    It returns the model parameters in lists (of lists, where applicable) with 
    the length of the number of models in the Excel template.

    Parameters
    ----------
    model_parameters : pd.DataFrame
        Pandas dataframe with all the necesary parameters to build a model Suite.
    reservoir_parameters : pd.DataFrame
        Pandas dataframe with all the necesary parameters to build a model Suite.
    pressure_df : pd.DataFrame
        Pandas dataframe with all the necesary parameters to build a model Suite.
    point_df : pd.DataFrame
        Pandas dataframe with all the necesary parameters to build a model Suite.
    observations_df : pd.DataFrame
        Pandas dataframe with all the necesary parameters to build a model Suite.

    Returns
    -------
    model_names : TYPE
        DESCRIPTION.
    dx : float/int
        Distance between grid nodes along the x-axis in m. The default is 
        None. Raises exception when None.
    dy : float/int
        Distance between grid nodes along the y-axis in m. The default is 
        None. When None, defaults to dx.
    influence_radius : float/int
        Distance from which the subsidence is set to 0 in m. The default 
        is None. Raises exception when None.
    compaction_model : list, str
        Can ba a string for the compaction model type name to be used for all reservoirs, or
        a list of string with the model type to be used for each reservoir.
        The list must have the same length as the number of reservoirs in 
        the model.
        
        The types of compaction models as defined in 
        PySub.CompactionModels for each reservoir: # TODO: keep updated with added methods
            - linear
            - time-decay
            - ratetype
        The default is None. Raises Exception when None.
    subsidence_model : str
        Method of subsidence of the model. Currently available:
        - Nucleus of strain, Van Opstal 1974
        - Knothe, Stroka et al. 2011. 
        The default is 'nucleus of strain'.
    tau : list, float/int/str
        The time-decay constant for the time-decay method for each 
        reservoir in seconds of delay. The list must have the same length 
        as the number of reservoirs in the model. The default is None.
        When a string, it must be the path to a .tif raster file with 1 band.
    knothe_angles : list, float/int
        The influence angle in degrees for the knoth subsidence method for each 
        reservoir. The default is None.The list must have the same length 
        as the number of reservoirs in the model.
    reservoirs : list, str
        The names of each reservoir. The default is None. The list must have 
        the same length as the number of reservoirs in the model.
    shapefile_paths : list, str
        The location to the shapefiles or .tif raster files for each reservoir. 
        The list must have the same length as the number of reservoirs in 
        the model.The default is None. 
    depths : list float/int
        Depths to the top of each reservoirs in m. The list must have 
        the same length as the number of reservoirs in 
        the model. The default is None. Raises Exception when None.
    depth_to_basements : list, float/int
        Depth to the rigid basement for the van Opstal nucleus of strain 
        method in m. The list must have the same length as the number of reservoirs in 
        the model. The default is None. Raises Exception when None.
    reference_stress_rates : list, float/int/str
        Reference stress rates in bar/year. The list must have the 
        same length as the number of reservoirs in the model. Raises Exception 
        when None and a ratetype compaction model is used. When a string, it must 
        be the path to a .tif raster file with 1 band.
    density : list, float/int/str
        Bulk density of the ground above the reservoir in kg/m³. 
        The list must have the same length as the number of reservoirs in 
        the model. Raises Exception when None and a ratetype compaction model 
        is used. When a string, it must be the path to a .tif raster file with 1 band.
    cmref : list, float/int/str
        Reference compaction coefficient in 1/bar used for the ratetype compaction model. 
        The list must have the same length as the number of reservoirs in 
        the model. Raises Exception when None and a ratetype compaction model is used.
        When a string, it must be the path to a .tif raster file with 1 band.
    b : list, float/int/str
        Dimensionless constant for the stiffer reaction of sandstone over a 
        specific loading rate. The list must have the same length as the number 
        of reservoirs in the model. Raises Exception when None and a ratetype 
        compaction model is used. When a string, it must be the path to a .tif 
        raster file with 1 band.
    compaction_coefficients : list, floats/str
        Uniaxial compaction coefficient (Cm) in 1/bar. The list must have 
        the same length as the number of reservoirs in the model. The default 
        is None. Raises Exception when None. When a string, it must be the path 
        to a .tif raster file with 1 band.
    thickness : list, float/int/str
        Thickness of each reservoir in m. The list must have the same length 
        as the number of reservoirs in the model.The default is None. Raises 
        Exception when None. When a string, it must be the path to a .tif raster 
        file with 1 band.
    timesteps : list, np.datetime64
        The timestamps of each step in time. These need to be of equal 
        step length. Per year would be ['1990', '1991', etc.]. The default 
        is None. Raises Exception when None.
    pressures : np.ndarray, float/int
        a 2D array with the shape of (m, n), where m is the number of reservoirs, 
        (SubsidenceModel.number_of_reservoirs) and n is the number of timesteps 
        (SubsidenceModel.number_of_steps). Contains the pressure development 
        over time for eacht reservoir in bar. The pressures will be 
        homogonous over each reservoir.
    points : PointCollection object
    observation_points : ObservationCollection object
    """
    model_names = model_parameters.index
    dx = []
    influence_radius = []
    compaction_model = []
    subsidence_model = []
    reservoir_names = []
    shapefile_paths = []
    depths = []
    depth_to_basements = []
    poissons_ratios = []
    compaction_coefficients = []
    thickness = []
    tau = []
    knothe_angles = []
    reference_stress_rates = []
    density = []
    cmref = []
    b = []
    timesteps = []
    pressures = []
    
    for model in model_parameters.index:
        dx.append(model_parameters["Cell size (m)"][model])
        influence_radius.append(model_parameters["Influence radius (m)"][model])
        compaction_model.append(model_parameters["Compaction model"][model])
        subsidence_model.append(model_parameters["Subsidence model"][model])
        
        model_reservoir_parameters = reservoir_parameters[reservoir_parameters['Name'] == model]
        model_pressure_df = pressure_df[pressure_df['Name'] == model]
        model_pressure_df = model_pressure_df.drop('Name', axis = 1)
        
        reservoir_names.append(model_reservoir_parameters.index.values)
        try: shapefile_paths.append(model_reservoir_parameters["Shapefile location"].values)
        except: shapefile_paths = None
        depths.append(model_reservoir_parameters["Depth to reservoir (m)"].values)
        depth_to_basements.append(model_reservoir_parameters["Depth to basement (m)"].values)
        poissons_ratios.append(model_reservoir_parameters["Poisson's ratio (-)"].values)
        compaction_coefficients.append(model_reservoir_parameters["Compaction coefficient (1/bar)"].values)
        thickness.append(model_reservoir_parameters["Reservoir thickness (m)"].values)
        try: tau.append(model_reservoir_parameters["Tau (s)"].values)
        except: tau = None
        try: knothe_angles.append(model_reservoir_parameters["Knothe angle (°)"].values)
        except : knothe_angles = None
        try: reference_stress_rates.append(model_reservoir_parameters["Reference stress rate (Pa/year)"].values)
        except: reference_stress_rates  = None
        try: density.append(model_reservoir_parameters["Average density above reservoir (kg/³)"].values)
        except: density = None
        try: cmref.append(model_reservoir_parameters["Reference compaction coefficient (1/bar)"].values )
        except: cmref = None
        try: b.append(model_reservoir_parameters["b"].values)
        except: b = None
        
        timesteps.append(model_pressure_df.columns.values )
        pressures.append(np.array(model_pressure_df))
    
    points = _Points.load_points_from_df(point_df)
    
    observation_points = _Points.load_observation_points_from_df(observations_df, 
                                   observation_column = 'Subsidence (m)',
                                   lower_error_column = 'Lower error (m)', 
                                   upper_error_column = 'Upper error (m)')
    
    return (model_names, dx, influence_radius, 
            compaction_model, 
            subsidence_model, 
            tau, 
            knothe_angles, 
            reservoir_names, 
            shapefile_paths, 
            depths, 
            depth_to_basements, poissons_ratios,
            reference_stress_rates, 
            density, 
            cmref, 
            b,
            compaction_coefficients, 
            thickness, timesteps, pressures,
            points, observation_points)

def build_model_from_xarray(xarray, name, project_folder = None):
    """Build a model from a pre-exisiting xarray object.

    Parameters
    ----------
    xarray : xr.DataSet
        xarray dataset with at least the coordinates ('x', 'y', 'reservoir', 'time').
    name : str
        Name of the model.
    project_folder : str, optional
        Path to a directory for the model input and results to be saved in. The default 
        is None. If the project_folder parameter is None, nothing will be saved. 

    Returns
    -------
    model : SubsidenceModel
        SubsidenceModel.

    """
    _xarray = xarray.copy()
    model = _SubsidenceModelGas.SubsidenceModel(name, project_folder = project_folder)
    model.grid = _xarray
    model.nx = _xarray.dims['x']
    model.ny = _xarray.dims['y']
    model.number_of_timesteps = _xarray.dims['time']
    model.number_of_reservoirs = _xarray.dims['reservoir']
    model.set_reservoirs(_xarray.reservoir.values)
    model.set_timesteps(_xarray.time.values)
    model.set_influence_radius(_xarray.influence_radius)
    
    min_x, max_x = np.min(model.x), np.max(model.x)
    min_y, max_y = np.min(model.y), np.max(model.y)
    min_x -= model.influence_radius
    max_x += model.influence_radius
    min_y -= model.influence_radius
    max_y += model.influence_radius
    model._bounds = (min_x, min_y, max_x, max_y)
    
    return model

def load_observation_points(import_path, sheet_name = 'Observations', 
                                       index_column = 'Observation ID', 
                                       observation_column = 'Subsidence (m)',
                                       lower_error_column = 'Lower error (m)', 
                                       upper_error_column = 'Upper error (m)',
                                       x_column = 'X', y_column = 'Y', time_column = 'Time'):
    """Import an ObservationCollection object from an Excel file. The ObservationCollection
    object stores information about observations over time. The information includes:
        - The location in x- and y-coordinates
        - Values for observations
        - The The timstamps at which each observation is taken
        - A lower and upper error bound.

    Parameters
    ----------
    import_path : str
        Path to Excel file.
    sheet_name : int/str, optional
        If a string, the name of the worksheet in the excel file. When an integer,
        the integer indicates n, where n means the the nth sheet in the Excel workbook.
        The default is 'Observations'.
    index_column : int/str, optional
        An integer for the index of the columns where the observation names are 
        stored or a string for its title. 
        The default is 'Observation ID'.
    observation_column : int/str, optional
        An integer for the index of the columns where the observations are 
        stored or a string for its title. The default is 'Subsidence (m)'.
    lower_error_column : int/str, optional
        An integer for the index of the columns where the observation errors are 
        stored or a string for its title. The default is 'Lower error (m)'.
    upper_error_column : int/str, optional
        An integer for the index of the columns where the observation errors are 
        stored or a string for its title. The default is 'Upper error (m)'.
    x_column : int/str, optional
        An integer for the index of the columns where the observation locations' x-coordinates are 
        stored or a string for its title. The default is 'X'.
    y_column : TYPE, optional
        An integer for the index of the columns where the observation locations' y-coordinates are 
        stored or a string for its title. The default is 'Y'.
    time_column : TYPE, optional
        An integer for the index of the columns where the observations' timestamps are 
        stored or a string for its title. The default is 'Time'.

    Returns
    -------
    observation_points : ObservationCollection
        An object storing information relating to observations over time.

    """
    
    if not os.path.isfile(import_path):
        raise Exception('The path should be a string that directs to a json or Excel file.')
    elif import_path.endswith(('.xlsx', '.xls')):
        observation_df = pd.read_excel(import_path, sheet_name = sheet_name, index_col = index_column, header = 0)
    elif import_path.endswith('.json'):
        observation_df = json_to_df(import_path,)[sheet_name]
        observation_df = observation_df.set_index(index_column)
    observation_points = _Points.load_observation_points_from_df(observation_df, observation_column = observation_column,
                                    x_column = x_column, y_column = y_column, time_column = time_column,
                                    lower_error_column = lower_error_column, upper_error_column = upper_error_column)
        
    return observation_points
    
def load_points(import_path, sheet_name = 'Points', index_column = 'Point ID',
                           x_column = 'X', y_column = 'Y'):
    if not os.path.isfile(import_path):
        raise Exception('The path should be a string that directs to a json or Excel file.')
    elif import_path.endswith(('.xlsx', '.xls')):
        point_df = pd.read_excel(import_path, sheet_name = sheet_name, index_col = index_column, header = 0)
    elif import_path.endswith('.json'):
        point_df = json_to_df(import_path,)[sheet_name]
    points = _Points.load_points_from_df(point_df, 
                        x_column = x_column, y_column = y_column)   
    return points