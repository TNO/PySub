# -*- coding: utf-8 -*-
"""Module storing Geometry classes Point, Grid and Polygon for storing and converting 
to grid data and plot this data.
"""
import os
from descartes import PolygonPatch
import descartes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PySub import Points as _Points
from PySub import plot_utils as _plot_utils
from PySub import utils as _utils
from PySub import grid_utils as _grid_utils
from PySub import shape_utils as _shape_utils
from osgeo import osr
from shapely import geometry
from shapely.ops import cascaded_union

class GeometryPoint(_Points.Point):
    def __init__(self, x, y, kwargs):
        """An object representing a point geometry with standardized 
        interactivity with the PySub package. Must have methods to plot, mask, 
        and others.
        
        Parameters
        ----------
        x : float
            x-coordinate
        y : float
            x-coordinate
        kwargs : dict
            Keyword arguments for the function ax.scatter:
                https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html

        Returns
        -------
        GeometryPoint.

        """
        self.type = 'point'
        self._x = np.array(x).astype(float)
        self._y = np.array(y).astype(float)
        self.kwargs = kwargs
    
    def plot(self, ax = None, kwargs = None):
        """Plots the Point object

        Parameters
        ----------
        ax : matplotlib.pyplot.axes.Axes object, optional
            The axis you want to plot the object in. 
            The default is None. When None a new axis is created.
        kwargs : dict
            Keyword arguments for the function ax.scatter:
                https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html

        """
        if kwargs is None:
            kwargs = self.kwargs
        if ax is None:
            fig, ax = plt.subplot()
            bounds = self.bounds
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
        cur_max_zorder = max([_.zorder for _ in ax.get_children()])
        max_zorder = cur_max_zorder + 1
        kwargs['zorder'] = max_zorder
        ax.scatter(self.x, self.y, **kwargs)
    
    def mask(self, grid):
        """Mask an xarray dataset with this geometry. In the case of points 
        the grid cell will be highlighted.

        Parameters
        ----------
        grid : xarray.Dataset
            Dataset with at least the variables x and y.

        Returns
        -------
        mask : xarray.DataArray
            A mask in the shape of the grid argument with where the geometry 
            overlaps, the value is 1, and where not, is 0.

        """
        ix, iy = np.searchsorted(grid.x, self.x), np.searchsorted(grid.y, self.y)
        mask = np.zeros((grid.dims['y'], grid.dims['x']))
        mask[iy, ix] = 1
        return mask
    
    def in_bound(self, *args):
        bounds = args[0]
        if bounds is not None:
            bounds_as_polygon = _utils.bounds_to_polygon(bounds)
            
            if not _utils.is_point_in(bounds_as_polygon, self.coordinates).all():
                print('Warning: Shape outside of grid bounds!')
    
    def _representative_parameters(self):
        return {self.type: (self._x, self._y)}
    
    @property
    def bounds(self):
        return _utils.bounds_from_xy(self.x, self.y)
    
    @property
    def midpoint(self):
        return self.coordinates

class GeometryRaster():
    def __init__(self, X, Y, values, kwargs):
        """An object representing a raster geometry with standardized 
        interactivity with the PySub package. Must have methods to plot, mask, 
        and others.
        
        Parameters
        ----------
        X : np.array, floats
            x-coordinates
        Y : np.array, floats
            y-coordinates
        values : np.array, floats
            values determining the mask. When a value is 0 or lower, it is not
            considered part of the geometry.
        kwargs : dict
            Keyword arguments for the function ax.imshow:
                https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html

        Returns
        -------
        GeometryRaster.

        """
        self.type = 'raster'
        self.X = np.array(X).astype(float)
        self.Y = np.array(Y).astype(float)
        self.values = np.array(values).astype(float)
        self.kwargs = kwargs
        
        self.x = self.X.flatten()
        self.y = self.Y.flatten()
        
        self.coordinates = np.vstack((self.x, self.y)).T
        self.values[self.values > 0] = 1
        
    def _representative_parameters(self):
        return {self.type: (self.X, self.Y, self.values)}
    
    def plot(self, ax = None, kwargs = None):
        """Plots the Raster object

        Parameters
        ----------
        ax : matplotlib.pyplot.axes.Axes object, optional
            The axis you want to plot the object in. 
            The default is None. When None a new axis is created.
        kwargs : dict
            Keyword arguments for the function ax.imshow:
                https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html

        """
        bounds = _utils.bounds_from_xy(self.X, self.Y)
        extent = np.array(bounds)[[0, 2, 1, 3]]
        masked_raster = np.ma.masked_where(self.values == 0, self.values)
        flipped_raster = np.flip(masked_raster, axis = 0)
        if kwargs is None:
            kwargs = self.kwargs
        if ax is None:
            fig, ax = plt.subplot()
            bounds = self.bounds
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
        ax.imshow(flipped_raster, extent = extent, **kwargs)
        
    def mask(self, grid):
        """Mask an xarray dataset with this geometry.

        Parameters
        ----------
        grid : xarray.Dataset
            Dataset with at least the variables x and y.

        Returns
        -------
        mask : xarray.DataArray
            A mask in the shape of the grid argument with where the geometry 
            overlaps, the value is 1, and where not, is 0.

        """
        
        xr_grid = xr.Dataset(coords = {'x': np.unique(self.X),
                                       'y': np.unique(self.Y)})
        xr_grid['mask'] = (('y', 'x'), self.values)
        _reservoir_mask = xr_grid['mask'].interp_like(grid)
        mask = np.nan_to_num(np.array(_reservoir_mask))
        mask[mask > 0] = 1
        mask[mask < 0] = 0
        return mask
    
    def in_bound(self, bound, dx):
        if bound is not None:
            bounds_as_polygon = _utils.bounds_to_polygon(bound)
            if not _utils.is_point_in(bounds_as_polygon, self.coordinates, radius = dx).all():
                print('Warning: Shape (partially) outside of grid bounds!')
    
    @property
    def bounds(self):
        return _utils.bounds_from_xy(self.x, self.y)
    
    @property
    def midpoint(self):
        xs = self.X[np.where(self.values > 0)]
        ys = self.Y[np.where(self.values > 0)]
        return np.mean(xs), np.mean(ys)



class GeometryPolygon():
    def __init__(self, shapes, kwargs):
        """An object representing a polygon geometry with standardized 
        interactivity with the PySub package. Must have methods to plot, mask, 
        and others.
        
        Parameters
        ----------
        shapes : list, shapely gemoetries
        kwargs : dict
            Keyword arguments for the class Polygon:
                https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html

        Returns
        -------
        GeometryPolygon.

        """
        self.type = 'polygon'
        self.shapes = shapes
        self.kwargs = kwargs
        
    def _representative_parameters(self):
        return {self.type: (self.shapes)}
    
    def plot(self, ax = None, kwargs = None):
        """Plots the Polygon object

        Parameters
        ----------
        ax : matplotlib.pyplot.axes.Axes object, optional
            The axis you want to plot the object in. 
            The default is None. When None a new axis is created.
        kwargs : dict
            Keyword arguments for the class Polygon:
                https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html

        """
        if ax is None:
            fig, ax = plt.subplots()
            bounds = self.bounds
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
        
        for geom in self.shapes:
            if kwargs is None:
                kwargs = self.kwargs
            p = PolygonPatch(geom.buffer(0), **kwargs)
            ax.add_patch(p) 
    
    def mask(self, grid):
        """Mask an xarray dataset with this geometry.

        Parameters
        ----------
        grid : xarray.Dataset
            Dataset with at least the variables x and y.

        Returns
        -------
        mask : xarray.DataArray
            A mask in the shape of the grid argument with where the geometry 
            overlaps, the value is 1, and where not, is 0.

        """
        mask = np.zeros((grid.dims['y'], grid.dims['x']))
        mask = mask > 0
        for shape in self.shapes:
            exterior, interiors = _shape_utils.extract_poly_coords(shape)
            is_in_exteriors = _grid_utils.get_mask_from_shp(exterior, grid.x, grid.y)
            mask = mask | is_in_exteriors 
            for interior in interiors:
                is_in_interior = _grid_utils.get_mask_from_shp(exterior, grid.x, grid.y)
                mask = mask & ~is_in_interior
        return mask 
    
    def in_bound(self, bounds, dx):
        if bounds is not None:
            bounds_as_polygon = _utils.bounds_to_polygon(bounds)
            for s in self.shapes:
                if not _utils.is_point_in(bounds_as_polygon, s, radius = dx).all():
                    print('Warning: Shape (partially) outside of grid bounds!')
    
    @property
    def bounds(self):
        bounds = cascaded_union(self.shapes).bounds
        return bounds
    
    @property
    def midpoint(self):
        min_x, max_x = np.array(self.bounds)[[0, 2]]
        min_y, max_y = np.array(self.bounds)[[1, 3]]
        mid_x = min_x + (max_x - min_x)/2
        mid_y = min_y + (max_y - min_y)/2
        return mid_x, mid_y
    
class _crs():
    def __init__(self):
        self.crs = None
        
    def check(self, crs):
        if self.crs is None:
            self.crs = crs
        else:
            if not self.crs is None:
                if self.crs != crs:
                    raise Exception(f'Sources use different coordinate reference systems: {crs} and {self.crs}')

def from_representative_values(representative_parameters,
                               scatter_kwargs = {},
                               shape_kwargs = {},
                               raster_kwargs = {}): 
    geometry_type = list(representative_parameters.keys())[0]
    if geometry_type == 'point':
        geometry = GeometryPoint(representative_parameters[geometry_type][0],
                                 representative_parameters[geometry_type][1], 
                                 kwargs = scatter_kwargs)
    elif geometry_type == 'raster':
        geometry = GeometryRaster(representative_parameters[geometry_type][0],
                                  representative_parameters[geometry_type][1],
                                  representative_parameters[geometry_type][2],
                                  kwargs = raster_kwargs)
    elif geometry_type == 'polygon':
        geometry = GeometryPolygon(representative_parameters[geometry_type], kwargs = shape_kwargs)
    else:
        raise Exception(f'Invalid geometry type: {geometry_type}.')
    return geometry

def fetch(files, scatter_kwargs = {},
                 shape_kwargs = {},
                 raster_kwargs = {}):
    """Load a Geometry object or multiple objects.

    Parameters
    ----------
    files : list or string
        A string with the path to the object or a list with the paths to the
        objects.
    scatter_kwargs : dict
        Keyword arguments for the function ax.scatter:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
    shape_kwargs : dict
        Keyword arguments for the class Polygon:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html
    raster_kwargs : dict
        Keyword arguments for the function ax.imshow:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html

    Raises
    ------
    Exception
        DESCRIPTION.
    TypeError
        DESCRIPTION.

    Returns
    -------
    geometries : TYPE
        DESCRIPTION.

    """
    if not _utils.is_iterable(files):
        files = [files]
    geometries = []
    crs = _crs()
    for f in files:
        if isinstance(f, str):
            if f.endswith('.shp'):
                _utils.check_shapefiles([f])
                shape, file_crs = _shape_utils.get_polygon(f)
                
                crs.check(file_crs)
                geom = GeometryPolygon(shape, shape_kwargs)
                geometries.append(geom)
            elif f.endswith('.tif'):
                values, X, Y, file_crs = _shape_utils.load_raster(f)
                values = values.reshape(values.shape[1:])
              
                crs.check(file_crs)
                
                geometries.append(GeometryRaster(X, Y, values, raster_kwargs))
            elif f.endswith('.csv'):
                df = pd.read_csv(f, delimiter = ';', header = 0, decimal = ',')
                try: 
                    x = df[df.columns[0]].values.astype(float)
                except:
                    try:
                        df = pd.read_csv(f, delimiter = ',', header = 0, decimal = '.')
                        x = df[df.columns[0]].values.astype(float)
                    except:
                        raise Exception('Invalid delimiter encountered in {f}, use ";" or ",".')
                if len(df.columns) == 3 and len(df[df.columns[2]].unique()) != 1:
                    values, X, Y = _shape_utils.load_raster_from_csv(f)
                    values = values.reshape(values.shape[:-1])
                  
                    geometries.append(GeometryRaster(X, Y, values, raster_kwargs))
                elif len(df.columns) == 2 or (
                        len(df.columns) == 3 and len(df[df.columns[2]].unique()) == 1):
                    shape = geometry.Polygon(df.values[...,:2].astype(float))
                    geometries.append(GeometryPolygon([shape], shape_kwargs))
                else:
                    raise Exception(f'Input csv {f} has invalid number of rows: {len(df.columns)}. Use 3 (x, y, mask) for a raster or 2 (x, y) for a polygon.')
            else:
                raise Exception(f'Invalid file type: {os.path.splitext(f)[1]}')
        elif _utils.is_iterable(f):
            if len(np.array(f)) == 2:
                geometries.append(GeometryPoint(f[0], f[1], scatter_kwargs))
        else:
            raise TypeError(f'Invalid type {type(f)}')
    return geometries