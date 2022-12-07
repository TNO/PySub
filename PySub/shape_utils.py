# -*- coding: utf-8 -*-

import os

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
# External imports
from PySub import utils as _utils

import shapefile as shp
from shapely import geometry
import pyproj
import numpy as np
import pandas as pd


def ogr_polygon(coords):       
    """Make a polygon of the ogr.Geometry type Polygon.

    Parameters
    ----------
    coords : array-like
        2D array-like object with the shape (2, m), where 2 is for the x- and 
        y-coordinates and m is the number of points that define the polygon.

    Returns
    -------
    ogr.Polygon
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in coords:
        ring.AddPoint(coord[0], coord[1])

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly.ExportToWkt()

def extract_poly_coords(geom):
    """Extract the relevant coordinates of a shapely geometry

    Parameters
    ----------
    geom : shapely.geometry type
        The polygon
    
    Returns
    -------
    exterior_coords: 2D np.ndarray
        an m by 2 shaped numpy array with the cordinates of the points that
        form the outside rim of the polygon.
    interior_coords: list of 2D numpy arrays
        If the polygon has holes in it, it is defined 
        through these coordinates. The list contains a 2D np.ndarray with the
        shape (m, 2) for each hole in the polygon.

    Source
    ------
    https://stackoverflow.com/questions/21824157/how-to-extract-interior-polygon-coordinates-using-shapely
    """
    if geom.type == 'Polygon':
        exterior_coords = geom.exterior.coords[:]
        interior_coords = []
        for interior in geom.interiors:
            interior_coords += interior.coords[:]
    elif geom.type == 'MultiPolygon':
        exterior_coords = []
        interior_coords = []
        for part in geom:
            epc = extract_poly_coords(part)  # Recursive call
            exterior_coords += epc['exterior_coords']
            interior_coords += epc['interior_coords']
    else:
        raise ValueError('Unhandled geometry type: ' + repr(geom.type))
    return (np.array(exterior_coords), 
            [np.array(i) for i in interior_coords])

def save_polygon(geometries, fname, epsg, fields = None):
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
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(fname)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    if fields is not None:
        layer.CreateField(ogr.FieldDefn('value', ogr.OFTReal))

    defn = layer.GetLayerDefn()

    for i, geom in enumerate(geometries):

        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', i)
        if fields is not None:
            feat.SetField('value', fields[i])
    
        # Make a geometry, from Shapely object
        polygon = ogr.CreateGeometryFromWkt(ogr_polygon(geom))
        feat.SetGeometry(polygon)
    
        layer.CreateFeature(feat)
        feat = polygon = None  # destroy these
    
        # Save and close everything
    ds = layer = feat = polygon = None

def _get_projection_file(fname):
    return os.path.splitext(fname)[0]+'.prj'

def _get_projection(fname):
    prj_f = _get_projection_file(fname)
    with open(prj_f) as prj:
        crs = pyproj.CRS.from_wkt(prj.read())
    return crs

def _make_projection(fname, epsg):
    prj_f = _get_projection_file(fname)
    
    wkt_string = pyproj.CRS.from_epsg(epsg).to_wkt()
    with open(prj_f, "w") as f:
        f.write(wkt_string)

def get_shapely(fname):
    shapes = shp.Reader(fname)
    crs = _get_projection(fname)
    geometries = []
    for s in shapes:
        geo_json = s.shape.__geo_interface__
        geometry_type = geo_json['type'] 
        
        shapely_type = getattr(geometry, geometry_type)
        geom = shapely_type(s['geometry']['coordinates'][0])
        geometries.append(geom)
    return geometries, crs

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

    shapes = shp.Reader(fname)
    crs = _get_projection(fname)
    geometries = []
    for s in shapes:
        geo_json = s.shape.__geo_interface__
        shape = geometry.shape(geo_json)
        if geo_json['type'] == 'MultiPolygon':
            geometries = geometries + list(shape)
        else:
            geometries.append(shape)
        
    return geometries, crs

def save_raster(data, x, y, dx, dy, epsg, fname, fileformat = "GTiff"):
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
    n_dims = len(data.shape)
    if n_dims == 2:
        transposed_data = data.reshape((1, data.shape[0], data.shape[1]))
    elif n_dims == 3:
        transposed_data = np.transpose(data, axes = (2,0,1))
    else:
        raise Exception(f'Number of dimensions supported for raster exportation is 2 or 3. Number of dimensions is {len(data.shape)}')
    
    srs = osr.SpatialReference()
    srs.SetFromUserInput(f"EPSG:{epsg}")
    
    driver = gdal.GetDriverByName(fileformat)
    dst_ds = driver.Create(fname, xsize=transposed_data.shape[2], ysize=transposed_data.shape[1],
                    bands=transposed_data.shape[0], eType=gdal.GDT_Byte)
    
    geotransform = (min(x), dx, 0, min(y), 0, dy)
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.SetGeoTransform(geotransform)
    for i, raster in enumerate(transposed_data):
        dst_ds.GetRasterBand(i+1).WriteArray(raster)
    dst_ds = None
    
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
    crs : wkt crs as string
    """
    # src = rasterio.open(fname)
    # crs = src.crs.wkt if src.crs is not None else None
    
    src = gdal.Open(fname)
    if src is None:
        raise Exception(f'Can not open file:\n{fname}')
    crs = src.GetProjection() # wkt
    
    if layer is None:
        data = src.GetRasterBand(1).ReadAsArray()
    else:
        data = src.GetRasterBand(layer).ReadAsArray()
    data = data[None, :, :]
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres + xres/2)
    lry = uly + (src.RasterYSize * yres + yres/2)
    x = np.linspace(ulx, lrx, src.RasterXSize)
    y = np.linspace(uly, lry, src.RasterYSize)
    src = None
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

    
