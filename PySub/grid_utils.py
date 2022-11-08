import xarray as xr
import shapely.geometry
import numpy as np
from PySub import utils as _utils
from PySub import shape_utils as _shape_utils
from matplotlib import path
import shapefile as shp
from shapely.ops import cascaded_union 




def get_mask_from_shp(xy_list, ax, ay):
    """
    Fill in a grid with boolean values based on occurence of a shapely gemotry object.
    Differs from mask_from_shp, which takes te location of a shapefile and creates multpiple masks based on that shapefile.

    Parameters
    ----------
    xy_list : list
        The coordinates of a single polygon with shape m x 2, where m is the amount of points and
        2 is the x- and y-coordinates.
    ax : 2D numpy array, floats/ints
        The x-coördinates of the grid, as a grid.
    ay : TYPE
        The y-coördinates of the grid, as a grid.

    Returns
    -------
    reservoir : 2D numpy array, boolean
        Grid with boolean values of where the shape overlaps. True where it does overlap and False where it doesn't.

    """
    xy = np.array(xy_list)
    x, y = xy.T
    poly_path = path.Path(xy) # faster then shapely in most cases

    xx, yy = np.meshgrid(ax, ay)
    axy = np.column_stack((xx.ravel(), yy.ravel()))

    reservoir = poly_path.contains_points(axy)
    reservoir = np.reshape(reservoir, (len(ay), len(ax)))
    return reservoir


def generate_grid_from_shape(shapefile_path, dx, dy=None, timesteps = [1], reservoir_layers = [1], buffer=0.0):
    """
    Generates an x-array object, grid, which stores the subsidence data in a geographic and temporal representative format.
    It is generated using a single shapefile as input.

    Parameters
    ----------
    shapefile_path : string
        String with the path to a .shp file.
    dx : int
        Size of the grid cells over the x-axis.
    dy : int, optional
        Size of the grid cells over the y-axis. The default is the same value of dx.
    timesteps : int, optional
        The amount of timesteps. The default is 1.
    reservoir_layers : int, optional
        The amount of reservoir layers. The default is 1.
    buffer : int, float, optional
        The distance around the bounds of the fields of the area to be studied. The default is 0.0.

    Returns
    -------
    grid : xarray Dataset
        # XXX Description.

    """
    shapes, crs = _shape_utils.get_shapely(shapefile_path)
    bounds = cascaded_union(shapes).bounds
    print("nr of polygons:", len(shapes))
    print("bounds:", bounds)
    print("crs:", crs)
    print("buffer:", buffer)
    geom_collection = []
    for s in shapes:
        # print(s['properties'][mask_id])
        geom = s.exterior.xy if hasattr(s, 'exterior') else s.xy
        geom_collection.append(geom)

    grid = generate_grid_from_bounds(bounds, dx, dy, timesteps = timesteps, reservoir_layers = reservoir_layers, influence_radius=buffer)
    grid.attrs['crs'] = crs
    grid.attrs['bounds'] = bounds

    for i, geom in enumerate(geom_collection):
        geom_mask = get_mask_from_shp(geom, grid.x, grid.y)
        grid['grid_mask'] = xr.where(geom_mask, 1, grid['grid_mask'])
        grid['reservoir'] = np.append(grid.reservoir.data, 1)

    grid['reservoir'] = np.unique(grid.reservoir.data)

    return grid

def convert_to_grid(grid, name):
    """Convert data variable that has been indexed by number of reservoirs 
    and number of timesteps and time, to a spatial occurence (y, x, reservoir, 
    time) in the grid. The data variable in the grid will be named 'grid_name'.
    """
    if 'x' in list(grid[name].coords):
        return
    if len(grid[name].shape) == 1:
        griddified = np.zeros(shape=[grid.dims[dim] for dim in ['y', 'x', 'reservoir']])
        for m in range(grid.dims['reservoir']):
            grid_reservoir = np.array(xr.where(grid['reservoir_mask'].isel(reservoir = m) == 1,
                                               grid[name].isel(reservoir = m), grid['grid_mask']))
            griddified[:, :, m] = grid_reservoir
        grid[name] = (['y', 'x', 'reservoir'], griddified)
       
    elif len(grid[name].shape) == 2:
        griddified = np.zeros(shape=[grid.dims[dim] for dim in ['y', 'x', 'reservoir', 'time']])
        for m in range(grid.dims['reservoir']):
            for t in range(grid.dims['time']):
                grid_reservoir_time = np.array(xr.where(grid['reservoir_mask'].isel(reservoir = m) == 1,
                                                   grid[name].isel(reservoir = m, time = t), grid['grid_mask']))
                griddified[:, :, m, t] = grid_reservoir_time
        grid[name] = (['y', 'x', 'reservoir', 'time'], griddified)
    # return grid

def generate_grid_from_bounds(bounds, dx, dy=None, timesteps = [1], reservoir_layers = [1], influence_radius=0, include_mask=True):
    """
    Generates an x-array object, grid, which stores the subsidence data in a geographic and temporal representative format.
    It is generated using lower and upper limits of the bounds as input.

    Parameters
    ----------
    bounds : array-like, int/float
        An array-like object with 4 values.
        [0] lower x
        [1] lower y
        [2] upper x
        [3] upper y
    dx : int
        Size of the grid cells over the x-axis.
    dy : int, optional
        Size of the grid cells over the y-axis. The default is the same value of dx.
    timesteps : array-like, optional
        List, array or dataframe of the times in years AD. The default is 1.
    reservoir_layers : array-like, optional
        list, array or dataframe of the labels of the reservoir layers. The default is 1.
    point_ids : array_like, optional
        list, array or dataframe of the label of the points. the default is a single point with label 1.
    influence_radius : int, float, optional
        The distance around the bounds of the fields of the area to be studied. The default is 0.0.
    include_mask : boolean, optional
        Create a data variable in the grid object for a mask. The default is True.

    Returns
    -------
    grid : xarray Dataset
        # XXX DESCRIPTION.

    """
    if dy is None:
        dy = dx

    ox = _utils.round_down(bounds[0] - influence_radius, dx)
    oy = _utils.round_down(bounds[1] - influence_radius, dy)
    ex = _utils.round_up(bounds[2] + influence_radius, dx)
    ey = _utils.round_up(bounds[3] + influence_radius, dy)
    nx = int((ex - ox) / dx + 1)
    ny = int((ey - oy) / dy + 1)

    x = np.linspace(ox, ex, num=nx)
    y = np.linspace(oy, ey, num=ny)

    if include_mask:
        grid = xr.Dataset(coords={'x': x, 'y': y, 
                                  'reservoir': reservoir_layers,
                                  'time': timesteps})
    else:
        grid = xr.Dataset(coords={'x': x, 'y': y})

    grid['grid_mask'] = (['y', 'x'], np.zeros((ny, nx)))

    grid.attrs['dx'] = dx
    grid.attrs['dy'] = dy
    grid.attrs['bounds'] = (ox, oy, ex, ey)
    grid.attrs['influence_radius'] = influence_radius

    return grid


def check_bounds(bounds1, bounds2):
    """
    Check if bounds1 is within bounds2

    Parameters
    ----------
    bounds1 : array-like
        Array-like object with 4 entries representing lower x, lower y, upper x and upper y.
    bounds2 : TYPE
        Array-like object with 4 entries representing lower x, lower y, upper x and upper y.

    Returns
    -------
    check : boolean
        True if bounds1 is within bounds2, False if not.

    """
    if len(bounds1) != 4 or len(bounds2) != 4:
        raise(Exception('bounds require 4 entries'))
    check = True
    for i in range(len(bounds1)):
        if i < 2:
            if bounds1[i] < bounds2[i]:
                check = False
        else:
            if bounds1[i] > bounds2[i]:
                check = False
    return check
    

