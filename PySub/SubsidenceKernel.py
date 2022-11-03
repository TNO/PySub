"""Kernel based method for solving the nucleus of strain and knothe method
"""
import xarray as xr
import numpy as np
from PySub import grid_utils as _grid_utils
from PySub import utils as _utils
import matplotlib.pyplot as plt

class InfluenceKernel:
    def __init__(self, influence_radius, dx, dy=None, bounds = (0, 0, 0, 0)):
        """Generate a grid that stores the distances from its center. This will be used for 
        convolution over a grid where compaction has been modelled and will translate that 
        compaction to surface deformation along the z-axis.

        Parameters
        ----------
        influence_radius : int/float
            The distance from the center that will be considered significant.
            The size of the grid is dependant on this factor, as it is half the 
            length of a side of the grid. Beyond this influence radius the influence 
            on subsidence from compaction at the grid center is considered 0.
        dx : int/float
            The distance between grid nodes, must be uniform and equal to dy.
        dy : TYPE, optional
            The distance between grid nodes, must be uniform and equal to dx. The 
            default is None, which will set the dy the same is dx.

        Returns
        -------
        InfluenceKernel object.

        """
        self.ds = _grid_utils.generate_grid_from_bounds((0, 0, 0, 0), dx, dy, 
                                                        influence_radius = influence_radius, 
                                                        include_mask=False
                                                        ).rename({'x':'kx', 'y':'ky'}
                                                        )

        r = np.sqrt(self.ds['kx'] ** 2 + self.ds['ky'] ** 2)
        self.ds = self.ds.assign(r=r)

    def nucleus(self, depth, basement_depth = None, v = 0.25):
        """Solve the Geertsma or Van Opstal solution for subsidence due to compaction
        for every grid cell.

        Parameters
        ----------
        depth : int/float
            The distance from the surface to the top of the reservoir in m.
        basement_depth : int/float, optional
            The distance from the surface to a rigid basement in m. The default 
            is None. When None, the Geertsma solution will be used, when a (real)
            number, the Van Opstal solution will be used.
        v : int/float, optional
            Dimensionless Poisson's ratio. The default is 0.25.

        Sets
        -------
        self.ds.uz.
            Sets the parameter uz of the grid object as the subsidence in m that
            is caused by the compaction at the centre of the grid.

        """
        # Geertsma nucleus solution
        u =  Geertsma(self.ds.r, depth, v = v)
        
        if basement_depth is not None:
            # Van Opstal(1974) correction 
            terms = _utils.get_van_opstal_coefficients(v)
            du = xr.apply_ufunc(Van_Opstal,
                                self.ds.r, 
                                depth, 
                                basement_depth, 
                                v, *terms.values())
            u = u + du.astype(float)
      
        self.ds['uz'] = u
        
    def nucleus_elongated(self, depth, basement_depth = None, v = 0.25): # XXX depricated
        """Solve the Geertsma or Van Opstal solution for subsidence due to compaction
        for every grid cell with additional vertical factor (vertical length is larger 
        than horizontal extend).

        Parameters
        ----------
        depth : 1D np.array, float
            The distance from the surface to the top of the reservoir in m.
        basement_depth : float, optional
            The distance from the surface to a rigid basement in m. The default 
            is None. When None, the Geertsma solution will be used, when a (real)
            number, the Van Opstal solution will be used.
        v : int/float, optional
            Dimensionless Poisson's ratio. The default is 0.25.

        Sets
        -------
        self.ds.uz.
            Sets the parameter uz of the grid object as the subsidence in m that
            is caused by the compaction at the centre of the grid.
        """
        self.ds = self.ds.assign_coords({'depth': depth})
        
        # Geertsma nucleus solution
        u = ((1 - v) / (2 * np.pi)) * ((2 * depth) / ((self.ds.r ** 2 + self.ds.depth ** 2) ** (3 / 2)))
        
        if basement_depth:
            terms = _utils.get_van_opstal_coefficients(v)
            du = xr.apply_ufunc(Van_Opstal,
                                self.ds.r, 
                                self.ds.depth, 
                                basement_depth, 
                                v, *terms.values())
            u = u + du.astype(float)
        self.ds['uz'] = u
    
    def nucleus_moving_basement(self, depth, basement_depth, v = 0.25): # XXX depricated
        """Solve the Geertsma or Van Opstal solution for subsidence due to compaction
        for every grid cell with additional vertical factor (vertical length is larger 
        than horizontal extend) and moving rigid basement.

        Parameters
        ----------
        depth : 1D np.array, float
            The distance from the surface to the top of the reservoir in m.
        basement_depth : 1D np.array, float
            The distance from the surface to a rigid basement in m. The default 
            is None. When None, the Geertsma solution will be used, when a (real)
            number, the Van Opstal solution will be used.
        v : int/float, optional
            Dimensionless Poisson's ratio. The default is 0.25.

        Sets
        -------
        self.ds.uz.
            Sets the parameter uz of the grid object as the subsidence in m that
            is caused by the compaction at the centre of the grid.

        """
        self.ds = self.ds.assign_coords({'depth': depth, 'time': np.arange(len(basement_depth))})
        self.ds['basement_depth'] = (('time'), basement_depth)
        # Geertsma nucleus solution
        u = ((1 - v) / (2 * np.pi)) * ((2 * depth) / ((self.ds.r ** 2 + self.ds.depth ** 2) ** (3 / 2)))
        du = xr.apply_ufunc(Van_Opstal,
                            self.ds.r, 
                            self.ds.depth, 
                            self.ds.basement_depth, 
                            v) 
        u = u + du
        self.ds['uz'] = u #(('y', 'x', 'depth', 'time'), u)
        
    def knothe(self, depth, knothe_angle):
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
        self.ds.uz.
            Sets the parameter uz of the grid object as the subsidence in m that
            is caused by the compaction at the centre of the grid.

        """
        u = Knothe(self.ds.r, depth, knothe_angle)
        self.ds = self.ds.assign(uz=u)

    def as_array(self):
        """Return the grid as a numpy array.

        Returns
        -------
        2D np.ndarray
            The subsidence caused by compaction at the center of the grid in m,
            in a numpy friendly format.

        """
        if 'uz' in self.ds:
            return self.ds['uz'].values
        else:
            print("Kernel not populated with values")
            return None

    def __repr__(self):
        return repr(self.ds)

    def plot2d(self):
        if 'uz' in self.ds:
            self.ds['uz'].plot.contour(add_colorbar=True)
            plt.axis('equal')
            plt.show()
        else:
            print("Nothing to plot")

    def plot(self):
        if 'uz' in self.ds:
            self.ds['uz'].isel(kx=int(self.ds.dims['kx']))[int(self.ds.dims['ky']):].plot()
        else:
            print("Nothing to plot")

def Geertsma(radius, depth, v = 0.25):
    du = ((1 - v) / (2 * np.pi)) * ((2 * depth) / ((radius ** 2 + depth ** 2) ** (3 / 2)))
    return du

def Knothe(radius, depth, knothe_angle):
    R = depth * np.tan((knothe_angle / 180) * np.pi)

    du = (1 / R ** 2) * np.exp(((-np.pi) * (radius ** 2)) / R ** 2)
    return du
    
def Van_Opstal(radius, depth, basement_depth, v = 0.25, 
               a12 = -0.778, a21 = 2.0,
               a22 = 2.80,
               b11 = -0.2,
               b21 = 4.0):
    # Van Opstal(1974) correction 
    a11 = 1.0
    # a12 = -0.778
    # a21 = 2.0
    # a22 = 2.80
    # b11 = -0.2
    # b21 = 4.0
    
    # i = 1
    a1 = (a11 * (a21 * basement_depth - depth) - 2 * a11 * basement_depth)
    a2 = 2 * a11 * basement_depth - (3 - 4 * v) ** 2 * a11 * (a21 * basement_depth + depth)
    a3 = (3 - 4 * v) * a11 * (a21 * basement_depth + 2 * basement_depth + depth)
    a4 = (3 - 4 * v) * a11 * (a21 * basement_depth + 2 * basement_depth - depth)
    a5 = 6 * a11 * basement_depth * (a21 * basement_depth - depth) ** 2
    a6 = 6 * a11 * basement_depth * (a21 * basement_depth + depth) * (
                6 * basement_depth - (a21 * basement_depth + depth))
    a7 = 60 * a11 * basement_depth ** 2 * (a21 * basement_depth + depth) ** 3
    
    # i = 2
    a8 = a12 * (a22 * basement_depth - depth) - 2 * a12 * basement_depth
    a9 = 2 * a12 * basement_depth - (3 - 4 * v) ** 2 * a12 * (a22 * basement_depth + depth)
    a10 = (3 - 4 * v) * a12 * (a22 * basement_depth + 2 * basement_depth + depth)
    aa11 = (3 - 4 * v) * a12 * (a22 * basement_depth + 2 * basement_depth - depth)
    aa12 = 6 * a12 * basement_depth * (a22 * basement_depth - depth) ** 2
    a13 = 6 * a12 * basement_depth * (a22 * basement_depth + depth) * (
                6 * basement_depth - (a22 * basement_depth + depth))
    a14 = 60 * a12 * basement_depth ** 2 * (a22 * basement_depth + depth) ** 3
    
    # j = 1
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

    # radius dependent terms
    n1 = (radius ** 2 + (a21 * basement_depth - depth) ** 2)
    n2 = (radius ** 2 + (a21 * basement_depth + depth) ** 2)
    terma1 = +(a1) / (n1 ** (3 / 2))
    terma2 = +(a2) / (n2 ** (3 / 2))
    terma3 = -(a3) / ((radius ** 2 + (a21 * basement_depth + 2 * basement_depth + depth) ** 2) ** (3 / 2))
    terma4 = +(a4) / ((radius ** 2 + (a21 * basement_depth + 2 * basement_depth - depth) ** 2) ** (3 / 2))
    terma5 = +(a5) / (n1 ** (5 / 2))
    terma6 = +(a6) / (n2 ** (5 / 2))
    terma7 = -(a7) / (n2 ** (7 / 2))
    n1 = (radius ** 2 + (a22 * basement_depth - depth) ** 2)
    n2 = (radius ** 2 + (a22 * basement_depth + depth) ** 2)
    terma8 = +(a8) / (n1 ** (3 / 2))
    terma9 = +(a9) / (n2 ** (3 / 2))
    terma10 = -(a10) / ((radius ** 2 + (a22 * basement_depth + 2 * basement_depth + depth) ** 2) ** (3 / 2))
    terma11 = +(aa11) / ((radius ** 2 + (a22 * basement_depth + 2 * basement_depth - depth) ** 2) ** (3 / 2))
    terma12 = +(aa12) / (n1 ** (5 / 2))
    terma13 = +(a13) / (n2 ** (5 / 2))
    terma14 = -(a14) / (n2 ** (7 / 2))
    sum1a = terma1 + terma2 + terma3 + terma4 + terma5 + terma6 + terma7
    sum1b = +terma8 + terma9 + terma10 + terma11 + terma12 + terma13 + terma14
    sum1 = sum1a + sum1b

    n1 = (radius ** 2 + (b21 * basement_depth - depth) ** 2)
    n2 = (radius ** 2 + (b21 * basement_depth + depth) ** 2)
    n3 = (radius ** 2 + (b21 * basement_depth + 2 * basement_depth - depth) ** 2)
    n4 = (radius ** 2 + (b21 * basement_depth + 2 * basement_depth + depth) ** 2)
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
    return du