# -*- coding: utf-8 -*-
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
import os
import sys
from pathlib import Path
for candidate in sys.path:
    if 'envs' in candidate:
        p = Path(candidate)
        environment_location = os.path.join(*p.parts[:p.parts.index('envs') + 2])
        break
os.environ['PROJ_LIB'] = os.path.join(environment_location, 'Library\share\proj')
os.environ['GDAL_DATA'] = os.path.join(environment_location, 'Library\share')

# External imports
from PySub import utils as _utils
import fiona
from fiona.crs import from_epsg
from shapely import geometry
import numpy as np
import pandas as pd

def save_polygon(geometries, fname, epsg):
    """Makes a shapefile without any properties from a list of list of coordinates.
    

    Parameters
    ----------
    geometries : list of polygons. Polygons are as a list of coordinates.
        Example:
            polygon1 = [[0, 0], [ 1, 0], [ 1,1], [0, 0]]
            polygon2 = [[0, 0], [-1, 0], [-1,1], [0, 0]]
            geometries = [polygon1, polygon2]
    fname : str
        location of the saved file.
    epsg : int
        EPSG coordinate number.

    Returns
    -------
    None.

    """
    schema = {'geometry' : 'MultiPolygon', 'properties' : {}}
    with fiona.open(fname, 'w', crs = from_epsg(epsg),
                    driver='ESRI Shapefile', schema = schema) as output:
        polygons = []
        for j, pol in enumerate(geometries):
            polygon = geometry.Polygon(pol)
            polygons.append(polygon)
        Multi = geometry.MultiPolygon(polygons)
        output.write({'geometry': geometry.mapping(Multi), 'properties' : {}})

def get_polygon(fname):
    """Get the xy coordinates of the polygons in a shapefile.

    Parameters
    ----------
    fname : str
        Location of a shapefile.

    Returns
    -------
    geometries : list of polygons. Polygons are as a list of coordinates.
        Example:
            polygon1 = [[0, 0], [ 1, 0], [ 1,1], [0, 0]]
            polygon2 = [[0, 0], [-1, 0], [-1,1], [0, 0]]
            geometries = [polygon1, polygon2]

    """

    with fiona.open(fname) as fiona_file:
        # crs = int(fiona_file.crs['init'][5:])    
        crs = fiona_file.crs_wkt
        geometries = []
        for s in fiona_file:
            if s['geometry'] is not None:
                if s['geometry']['type'] == 'MultiPolygon':
                    for i in range (len(s['geometry']['coordinates'])):
                        geom = np.array(s['geometry']['coordinates'][i])[..., :2]
                        geom = geom.reshape(geom.shape[1:])
                        geometries.append(geom.tolist())
                else:    
                    geom = np.array(s['geometry']['coordinates'])[..., :2]
                    if len(geom.shape) == 1:
                        geom = np.vstack(geom)[None, ..., :2]
                    geom = geom.reshape(geom.shape[1:])
                    geometries.append(geom.tolist())
    
    return geometries, crs

def save_raster(data, x, y, dx, dy, epsg, fname):
    """Save 2D, 3D, or 4D data is tiff files. 2D data will be stored as
    a tiff file with 1 band, 3D data as multibanded tiff and 4D data will 
    be stacked to fit multibanded data.

    Parameters
    ----------
    data : np.ndarray, floats
        2D, 3D or 4D numpy array. With first dimension being x, the second being y and 3rd or 4th arbitrary.
        The 3rd dimension will be the bands of the tif-file. When 4 dimensions, the layers will be stacked along the 
        3rd dimension to fit into 3 bands. Not more than 4 dimensions is allowed.
    x : np.ndarray, floats
        2D coordinates over the x-axis.
    y : np.ndarray, floats
        2D coordinates over the y-axis.
    dx : float/int
        The raster grid size.
    dy : float/int
        The raster grid size.
    epsg : int
        EPSG coordinate number.
    fname : str
        Path to the location of the saved file.

    Raises
    ------
    Exception
        Invalid data shape when 1D or more than 4D.

    Returns
    -------
    None.

    """
    try: rasterCrs = CRS.from_epsg(epsg)
    except: 
        print('Warning: Finding the projection failed, the raster has been saved without a specified EPSG coordinate system')
        rasterCrs = None
    transform = Affine.translation(x[0]-dx/2, y[0]-dy/2)*Affine.scale(dx,dy)
    if len(data.shape) == 2:
        transposed_data = data.reshape((1, data.shape[0], data.shape[1]))
    elif len(data.shape) == 3:
        transposed_data = np.transpose(data, axes = (2,0,1))
    elif len(data.shape) == 4:
        transposed_data = np.zeros(shape = (np.prod(data.shape[2:]), data.shape[0], data.shape[1]))
        for i in range(data.shape[2]): # reservoir
            for j in range(data.shape[3]): # timestep
                new_index = (i * data.shape[3]) + j
                transposed_data[new_index] = np.array([data[:, :, i, j]])
    else:
        raise Exception(f'Warning: data with shape length {len(data.shape)} is invalid. Add shape with 1 > length < 5.')
    interpRaster = rasterio.open(fname,
                    'w',
                    driver='GTiff',
                    height=transposed_data.shape[1],
                    width=transposed_data.shape[2],
                    count=transposed_data.shape[0],
                    dtype=transposed_data.dtype,
                    crs=rasterCrs,
                    transform=transform,
                    )
    interpRaster.write(transposed_data) # , data.shape[0])
    interpRaster.close()
    
def load_raster(fname, layer = None):
    """

    Parameters
    ----------
    fname : str
        Path to the location of the file.
    layer : int, optional
        The layer to be loaded from the .tif raster file in bands (starting from 1). 
        The default is None.

    Returns
    -------
    data : 3D np.ndarray, floats
        The data in the raster file with the shape (number of bands, number of x-coordinates, number of y-coordinates).
    x : 2D np.ndarray, floats
        The x-cordinates of all raster nodes.
    y : 2D np.ndarray, floats
        The y-cordinates of all raster nodes.

    """
    src = rasterio.open(fname)
    crs = src.crs.wkt if src.crs is not None else None
    
    band1 = src.read(1)
    height = band1.shape[0]
    width = band1.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    x = np.array(xs)
    y = np.array(ys)
    if layer is None:
        data = src.read()
    else:
        data = src.read(layer)
    return data, x, y, crs

def load_raster_from_csv(fname, delimiter = ';', header = 0, decimal = ',', method = 'linear', nan_values = 0):
    df = pd.read_csv(fname, delimiter = delimiter, header = header, decimal = decimal)
    df = df.replace(',', '.', regex = True)
    try: 
        x = df[df.columns[0]].values.astype(float)
    except:
        try:
            df = pd.read_csv(fname, delimiter = ',', header = header, decimal = decimal)
            x = df[df.columns[0]].values.astype(float)
        except:
            raise Exception('Invalid delimiter encountered, use ";" or ",".')
    y = df[df.columns[1]].values.astype(float)
    values = df[df.columns[2:]].values.astype(float)
    
    ids = np.where(values != nan_values)[0]
    
    xs, ys = np.unique(x[ids]), np.unique(y[ids])
    dx = np.min(np.abs(np.diff(xs)))
    xs = _utils.stepped_space(np.min(xs), np.max(xs), dx)
    ys = _utils.stepped_space(np.min(ys), np.max(ys), dx)
    X, Y = np.meshgrid(xs, ys)
    points = np.array((x, y)).T
    interpolated_data = _utils.interpolate_grid_from_points(points[ids], values[ids], (X, Y), method = method)
    return interpolated_data, X, Y

    
