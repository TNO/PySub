# -*- coding: utf-8 -*-
"""Point based method for solving the nucleus of strain and knothe method
"""

import numpy as np
from PySub import grid_utils
import xarray as xr

class InfluencePoint:
    def __init__(self, grid, point, influence_radius = 0):
        """Generate a grid that stores the distances from its center. This will be used for 
        multiplication over a grid where compaction has been modelled and will translate that 
        compaction to surface deformation along the z-axis.

        Parameters
        ----------
        grid : xr.DataArray
            The grid over which the distances from the point as entered in 
            the parameter point. The grid must have the following attributes defined:
                x : list, float
                    The x-coordinates for each node in the grid.
                y : list, float
                    The y-coordinates for each node in the grid. 
                dx : float
                    The distance between each node along the x-axis. Must be uniform 
                    over the entire grid.
                dy : float
                    The distance between each node along the y-axis. Must be 
                    uniform over the entire grid. The default is None, which will 
                    set the dy the same is dx.
                bounds : list
                    A list of 4 values indicating: [0] lower x, [1] lower y, 
                    [2] upper x, [3] upper y.
        point : tuple
            A tuple or list with the x- and y-coordinates, indication the location 
            from which the distance in the grid will be measured.
        influence_radius : int/float
            The distance from the center that will be considered significant.
            The size of the grid is dependant on this factor, as it is half the 
            length of a side of the grid. Beyond this influence radius the influence 
            on subsidence from compaction at the grid center is considered 0.

        Returns
        -------
        InfluenceKernel object.

        """
        self.ds = grid_utils.generate_grid_from_bounds(grid.bounds, grid.dx, grid.dy, influence_radius = influence_radius, include_mask=False)
        
        r = np.sqrt((grid['x'] - point[0]) ** 2 + (grid['y'] - point[1]) ** 2)
        if type(influence_radius) == int or type(influence_radius) == float:
            r = xr.where(r > influence_radius, np.nan, r)
        self.ds = self.ds.assign(r = r)
        
        self.point = point
        self.influence_radius = influence_radius
       
    def nucleus(self, depth, basement_depth = None, v = 0.25):
        """Solve the Geertsema or Van Opstal solution for subsidence due to compaction
        for every grid cell.

        Parameters
        ----------
        depth : int/float
            The distance from the surface to the top of the reservoir in m.
        basement_depth : int/float, optional
            The distance from the surface to a rigid basement in m. The default 
            is None. When None, the Geertsema solution will be used, when a (real)
            number, the Van Opstal solution will be used.
        v : int/float, optional
            Dimensionless Poisson's ratio. The default is 0.25.
            # XXX Only 0.25 is valid.

        Sets
        -------
        self.ds.u.
            Sets the parameter u of the grid object as the subsidence in m that
            is caused by the compaction at the centre of the grid.

        """
        u = ((1 - v) / (2 * np.pi)) * ((2 * depth) / ((self.ds.r ** 2 + depth ** 2) ** (3 / 2)))

        if basement_depth:
            # Van Opstal(1974) correction 
            if v != 0.25:
                print("Warning: Opstal correction only valid for Poisson's ratio of 0.25")
                
            # variables for a Poisson's ratio of 0.25
            a11 = 1.0
            a12 = -0.778
            a21 = 2.0
            a22 = 2.80
            b11 = -0.2
            b21 = 4.0

            a1 = (a11 * (a21 * basement_depth - depth) - 2 * a11 * basement_depth)
            a2 = 2 * a11 * basement_depth - (3 - 4 * v) ** 2 * a11 * (a21 * basement_depth + depth)
            a3 = (3 - 4 * v) * a11 * (a21 * basement_depth + 2 * basement_depth + depth)
            a4 = (3 - 4 * v) * a11 * (a21 * basement_depth + 2 * basement_depth - depth)
            a5 = 6 * a11 * basement_depth * (a21 * basement_depth - depth) ** 2
            a6 = 6 * a11 * basement_depth * (a21 * basement_depth + depth) * (
                        6 * basement_depth - (a21 * basement_depth + depth))
            a7 = 60 * a11 * basement_depth ** 2 * (a21 * basement_depth + depth) ** 3
            a8 = a12 * (a22 * basement_depth - depth) - 2 * a12 * basement_depth
            a9 = 2 * a12 * basement_depth - (3 - 4 * v) ** 2 * a12 * (a22 * basement_depth + depth)
            a10 = (3 - 4 * v) * a12 * (a22 * basement_depth + 2 * basement_depth + depth)
            aa11 = (3 - 4 * v) * a12 * (a22 * basement_depth + 2 * basement_depth - depth)
            aa12 = 6 * a12 * basement_depth * (a22 * basement_depth - depth) ** 2
            a13 = 6 * a12 * basement_depth * (a22 * basement_depth + depth) * (
                        6 * basement_depth - (a22 * basement_depth + depth))
            a14 = 60 * a12 * basement_depth ** 2 * (a22 * basement_depth + depth) ** 3
            
            
            b1 = b11 * basement_depth
            b2 = (3 - 4 * v) ** 2 * b11 * basement_depth
            b3 = (3 - 4 * v) * b11 * basement_depth
            b4 = 3 * b11 * basement_depth * (b21 * basement_depth - depth) * (
                        6 * basement_depth - (b21 * basement_depth - depth))
            b5 = 3 * b11 * basement_depth * (b21 * basement_depth + depth) * (6 * basement_depth - (3 - 4 * v) ** 2 * (
                        b21 * basement_depth + depth)) - 36 * b11 * basement_depth ** 3
            b6 = 3 * (3 - 4 * v) * b11 * basement_depth * (b21 * basement_depth + 2 * basement_depth - depth) ** 2
            b7 = 3 * (3 - 4 * v) * b11 * basement_depth * (b21 * basement_depth + 2 * basement_depth + depth) ** 2
            b8 = 30 * b11 * basement_depth ** 2 * (b21 * basement_depth - depth) ** 3
            b9 = 30 * b11 * basement_depth ** 2 * (b21 * basement_depth + depth) ** 2 * (
                        12 * basement_depth - (b21 * basement_depth + depth))
            b10 = 420 * b11 * basement_depth ** 3 * (b21 * basement_depth + depth) ** 4

            # depth dependent terms
            n1 = (self.ds.r ** 2 + (a21 * basement_depth - depth) ** 2)
            n2 = (self.ds.r ** 2 + (a21 * basement_depth + depth) ** 2)
            terma1 = +(a1) / (n1 ** (3 / 2))
            terma2 = +(a2) / (n2 ** (3 / 2))
            terma3 = -(a3) / ((self.ds.r ** 2 + (a21 * basement_depth + 2 * basement_depth + depth) ** 2) ** (3 / 2))
            terma4 = +(a4) / ((self.ds.r ** 2 + (a21 * basement_depth + 2 * basement_depth - depth) ** 2) ** (3 / 2))
            terma5 = +(a5) / (n1 ** (5 / 2))
            terma6 = +(a6) / (n2 ** (5 / 2))
            terma7 = -(a7) / (n2 ** (7 / 2))
            n1 = (self.ds.r ** 2 + (a22 * basement_depth - depth) ** 2)
            n2 = (self.ds.r ** 2 + (a22 * basement_depth + depth) ** 2)
            terma8 = +(a8) / (n1 ** (3 / 2))
            terma9 = +(a9) / (n2 ** (3 / 2))
            terma10 = -(a10) / ((self.ds.r ** 2 + (a22 * basement_depth + 2 * basement_depth + depth) ** 2) ** (3 / 2))
            terma11 = +(aa11) / ((self.ds.r ** 2 + (a22 * basement_depth + 2 * basement_depth - depth) ** 2) ** (3 / 2))
            terma12 = +(aa12) / (n1 ** (5 / 2))
            terma13 = +(a13) / (n2 ** (5 / 2))
            terma14 = -(a14) / (n2 ** (7 / 2))
            sum1a = terma1 + terma2 + terma3 + terma4 + terma5 + terma6 + terma7
            sum1b = +terma8 + terma9 + terma10 + terma11 + terma12 + terma13 + terma14
            sum1 = sum1a + sum1b

            n1 = (self.ds.r ** 2 + (b21 * basement_depth - depth) ** 2)
            n2 = (self.ds.r ** 2 + (b21 * basement_depth + depth) ** 2)
            n3 = (self.ds.r ** 2 + (b21 * basement_depth + 2 * basement_depth - depth) ** 2)
            n4 = (self.ds.r ** 2 + (b21 * basement_depth + 2 * basement_depth + depth) ** 2)
            termb1 = -(b1) / (n1 ** (3 / 2))
            termb2 = +(b2) / (n2 ** (3 / 2))
            termb3 = -(b3) / (n3 ** (3 / 2))
            termb4 = +(b3) / (n4 ** (3 / 2))
            termb5 = -(b4) / (n1 ** (5 / 2))
            termb6 = +(b5) / (n2 ** (5 / 2))
            termb7 = +(b6) / (n3 ** (5 / 2))
            termb8 = -(b7) / (n4 ** (5 / 2))
            termb9 = +(b8) / (n1 ** (7 / 2))
            termb10 = +(b9) / (n2 ** (7 / 2))
            termb11 = -(b10) / (n2 ** (9 / 2))
            sum2 = termb1 + termb2 + termb3 + termb4 + termb5 + termb6 + termb7 + termb8 + termb9 + termb10 + termb11

            # total correction
            du = (1 - v) / (2 * np.pi) * (sum1 + sum2)  # subsidence is defined positive!
            u += du

        self.ds = self.ds.assign(u = u)
    
    def knothe(self, depth, inf_angle):
        """Solve the Knothe solution for subsidence due to compaction
        for every grid cell.

        Parameters
        ----------
        depth : int/float
            The distance from the surface to the top of the reservoir in m.
        knothe_angle : int/float
            The Knothe angle in degrees.

        Sets
        -------
        self.ds.u.
            Sets the parameter u of the grid object as the subsidence in m that
            is caused by the compaction at the centre of the grid.

        """
        R = depth * np.tan((inf_angle / 180) * np.pi)

        u = (1 / R ** 2) * np.exp(((-np.pi) * (self.ds.r ** 2)) / R ** 2)
        self.ds = self.ds.assign(u = u)
        
    def __repr__(self):
        return repr(self.ds.r)

