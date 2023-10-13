"""Kernel based method for solving the nucleus of strain and knothe method
"""
import xarray as xr
import numpy as np
from PySub import grid_utils as _grid_utils
from PySub import utils as _utils
import matplotlib.pyplot as plt
import numba as nb

a11 = 1


class InfluenceKernel:
    def __init__(self, influence_radius, dx, dy=None, bounds=(0, 0, 0, 0)):
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
        dy : int/float, optional
            The distance between grid nodes, must be uniform and equal to dx. The
            default is None, which will set the dy the same is dx.

        Returns
        -------
        InfluenceKernel object.

        """
        bounds = (
            -influence_radius,
            -influence_radius,
            influence_radius,
            influence_radius,
        )
        self.ds = _grid_utils.generate_grid_from_bounds(
            bounds, dx, dy, include_mask=False
        ).rename({"x": "kx", "y": "ky"})

        r = np.sqrt(self.ds["kx"] ** 2 + self.ds["ky"] ** 2)
        self.ds = self.ds.assign(r=r)

    def nucleus(self, depth, basement_depth=None, v=0.25):
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
        uz = Geertsma_uz(self.ds.r, depth, v=v)

        if basement_depth is not None:
            # Van Opstal(1974) correction
            terms = _utils.get_van_opstal_coefficients(v)
            duz = xr.apply_ufunc(
                vertical_displacement,
                self.ds.r,
                depth,
                basement_depth,
                v,
                *terms.values(),
                vectorize=True,
            )

            uz = uz + duz.astype(float)

        self.ds["uz"] = uz

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

    def nucleus_slope(self, depth, basement_depth=None, v=0.25):
        dz__dx = Geertsma_slope(self.ds.kx, self.ds.ky, depth, v=v)
        dz__dy = Geertsma_slope(self.ds.ky, self.ds.kx, depth, v=v)

        if basement_depth is not None:
            # Van Opstal(1974) correction
            terms = _utils.get_van_opstal_coefficients(v)
            dz__dx_corr = xr.apply_ufunc(
                slope,
                self.ds.kx,
                self.ds.ky,
                depth,
                basement_depth,
                v,
                *terms.values(),
                vectorize=True,
            )

            dz__dy_corr = xr.apply_ufunc(
                slope,
                self.ds.ky,
                self.ds.kx,
                depth,
                basement_depth,
                v,
                *terms.values(),
                vectorize=True,
            )

            dz__dx = dz__dx + dz__dx_corr.astype(float)
            dz__dy = dz__dy + dz__dy_corr.astype(float)

        self.ds["slope_x"] = dz__dx
        self.ds["slope_y"] = dz__dy

    def nucleus_concavity(self, depth, basement_depth=None, v=0.25):
        dzdz__dxdx = Geertsma_concavity_xx(self.ds.kx, self.ds.ky, depth, v=v)
        dzdz__dxdy = Geertsma_concavity_xy(self.ds.kx, self.ds.ky, depth, v=v)
        dzdz__dydx = Geertsma_concavity_xy(self.ds.ky, self.ds.kx, depth, v=v)
        dzdz__dydy = Geertsma_concavity_xx(self.ds.ky, self.ds.kx, depth, v=v)

        if basement_depth is not None:
            # Van Opstal(1974) correction
            terms = _utils.get_van_opstal_coefficients(v)
            dzdz__dxdx_corr = xr.apply_ufunc(
                concavity_xx,
                self.ds.kx,
                self.ds.ky,
                depth,
                basement_depth,
                v,
                *terms.values(),
                vectorize=True,
            )

            dzdz__dxdy_corr = xr.apply_ufunc(
                concavity_xy,
                self.ds.kx,
                self.ds.ky,
                depth,
                basement_depth,
                v,
                *terms.values(),
                vectorize=True,
            )

            dzdz__dydx_corr = xr.apply_ufunc(
                concavity_xy,
                self.ds.ky,
                self.ds.kx,
                depth,
                basement_depth,
                v,
                *terms.values(),
                vectorize=True,
            )

            dzdz__dydy_corr = xr.apply_ufunc(
                concavity_xx,
                self.ds.ky,
                self.ds.kx,
                depth,
                basement_depth,
                v,
                *terms.values(),
                vectorize=True,
            )

            dzdz__dxdx = dzdz__dxdx + dzdz__dxdx_corr.astype(float)
            dzdz__dxdy = dzdz__dxdy + dzdz__dxdy_corr.astype(float)
            dzdz__dydx = dzdz__dydx + dzdz__dydx_corr.astype(float)
            dzdz__dydy = dzdz__dydy + dzdz__dydy_corr.astype(float)

        self.ds["concavity_xx"] = dzdz__dxdx
        self.ds["concavity_xy"] = dzdz__dxdy
        self.ds["concavity_yy"] = dzdz__dydy
        self.ds["concavity_yx"] = dzdz__dydy

    def as_array(self):
        """Return the grid as a numpy array.

        Returns
        -------
        2D np.ndarray
            The subsidence caused by compaction at the center of the grid in m,
            in a numpy friendly format.

        """
        if "uz" in self.ds:
            return self.ds["uz"].values
        else:
            print("Kernel not populated with values")
            return None

    def __repr__(self):
        return repr(self.ds)

    def plot2d(self):
        if "uz" in self.ds:
            self.ds["uz"].plot.contour(add_colorbar=True)
            plt.axis("equal")
            plt.show()
        else:
            print("Kernel not populated with values")

    def plot(self):
        if "uz" in self.ds:
            self.ds["uz"].isel(kx=int(self.ds.dims["kx"]))[
                int(self.ds.dims["ky"]) :
            ].plot()
        else:
            print("Kernel not populated with values")


def Geertsma_uz(radius, depth, v=0.25):
    duz = ((1 - v) / np.pi) * (depth / ((radius**2 + depth**2) ** (3 / 2)))
    return duz


def Geertsma_slope(x, y, c, v=0.25):
    return (
        (1 - v)
        / (2 * np.pi)
        * (-6 * c * x / (c**2 + x**2 + y**2) ** 2.5)
    )


def Geertsma_concavity_xx(x, y, c, v=0.25):
    return (
        (1 - v)
        / (2 * np.pi)
        * (
            +30 * c * x**2 / (c**2 + x**2 + y**2) ** 3.5
            - 6 * c / (c**2 + x**2 + y**2) ** 2.5
        )
    )


def Geertsma_concavity_xy(x, y, c, v=0.25):
    return (
        (1 - v)
        / (2 * np.pi)
        * (+30 * c * x * y / (c**2 + x**2 + y**2) ** 3.5)
    )


def Knothe(radius, depth, knothe_angle):
    R = depth * np.tan((knothe_angle / 180) * np.pi)

    u = (1 / R**2) * np.exp(((-np.pi) * (radius**2)) / R**2)
    return u


def vertical_displacement(
    r, c, k, v=0.25, a12=-0.778, a21=2.0, a22=2.80, b11=-0.2, b21=4.0
):
    # Van Opstal(1974) correction

    uz = (
        (1 - v)
        / (2 * np.pi)
        * (
            # + (2*c / ((r**2 + c**2)**(3 / 2))) #  Geertsma
            +(a11 * (a21 * k - c) - 2 * a11 * k)
            / ((r**2 + (a21 * k - c) ** 2) ** (3 / 2))
            + 2 * a11 * k
            - (3 - 4 * v) ** 2
            * a11
            * (a21 * k + c)
            / ((r**2 + (a21 * k + c) ** 2) ** (3 / 2))
            + (3 - 4 * v)
            * a11
            * (a21 * k + 2 * k + c)
            / ((r**2 + (a21 * k + 2 * k + c) ** 2) ** (3 / 2))
            + (3 - 4 * v)
            * a11
            * (a21 * k + 2 * k - c)
            / ((r**2 + (a21 * k + 2 * k - c) ** 2) ** (3 / 2))
            + 6
            * a11
            * k
            * (a21 * k - c) ** 2
            / ((r**2 + (a21 * k - c) ** 2) ** (5 / 2))
            + 6
            * a11
            * k
            * (a21 * k + c)
            * (6 * k - (a21 * k + c))
            / ((r**2 + (a21 * k + c) ** 2) ** (5 / 2))
            + 60
            * a11
            * k**2
            * (a21 * k + c) ** 3
            / ((r**2 + (a21 * k + c) ** 2) ** (7 / 2))
            + (a12 * (a22 * k - c) - 2 * a12 * k)
            / ((r**2 + (a22 * k - c) ** 2) ** (3 / 2))
            + 2 * a12 * k
            - (3 - 4 * v) ** 2
            * a12
            * (a22 * k + c)
            / ((r**2 + (a22 * k + c) ** 2) ** (3 / 2))
            + (3 - 4 * v)
            * a12
            * (a22 * k + 2 * k + c)
            / ((r**2 + (a22 * k + 2 * k + c) ** 2) ** (3 / 2))
            + (3 - 4 * v)
            * a12
            * (a22 * k + 2 * k - c)
            / ((r**2 + (a22 * k + 2 * k - c) ** 2) ** (3 / 2))
            + 6
            * a12
            * k
            * (a22 * k - c) ** 2
            / ((r**2 + (a22 * k - c) ** 2) ** (5 / 2))
            + 6
            * a12
            * k
            * (a22 * k + c)
            * (6 * k - (a22 * k + c))
            / ((r**2 + (a22 * k + c) ** 2) ** (5 / 2))
            + 60
            * a12
            * k**2
            * (a22 * k + c) ** 3
            / ((r**2 + (a22 * k + c) ** 2) ** (7 / 2))
            - b11 * k / ((r**2 + (b21 * k - c) ** 2) ** (3 / 2))
            + (3 - 4 * v) ** 2
            * b11
            * k
            / ((r**2 + (b21 * k + c) ** 2) ** (3 / 2))
            - (3 - 4 * v)
            * b11
            * k
            / ((r**2 + (b21 * k + 2 * k - c) ** 2) ** (3 / 2))
            + (3 - 4 * v)
            * b11
            * k
            / ((r**2 + (b21 * k + 2 * k + c) ** 2) ** (3 / 2))
            - 3
            * b11
            * k
            * (b21 * k - c)
            * (6 * k - (b21 * k - c))
            / ((r**2 + (b21 * k - c) ** 2) ** (5 / 2))
            + (
                3
                * b11
                * k
                * (b21 * k + c)
                * (6 * k - (3 - 4 * v) ** 2 * (b21 * k + c))
                - 36 * b11 * k**3
            )
            / ((r**2 + (b21 * k + c) ** 2) ** (5 / 2))
            + 3
            * (3 - 4 * v)
            * b11
            * k
            * (b21 * k + 2 * k - c) ** 2
            / ((r**2 + (b21 * k + 2 * k - c) ** 2) ** (5 / 2))
            - 3
            * (3 - 4 * v)
            * b11
            * k
            * (b21 * k + 2 * k + c) ** 2
            / ((r**2 + (b21 * k + 2 * k + c) ** 2) ** (5 / 2))
            + 30
            * b11
            * k**2
            * (b21 * k - c) ** 3
            / ((r**2 + (b21 * k - c) ** 2) ** (7 / 2))
            + 30
            * b11
            * k**2
            * (b21 * k + c) ** 2
            * (12 * k - (b21 * k + c))
            / ((r**2 + (b21 * k + c) ** 2) ** (7 / 2))
            - 420
            * b11
            * k**3
            * (b21 * k + c) ** 4
            / ((r**2 + (b21 * k + c) ** 2) ** (9 / 2))
        )
    )
    return uz


@nb.njit(fastmath=True)
def slope(
    x, y, c, k, v=0.25, a12=-0.778, a21=2.0, a22=2.80, b11=-0.2, b21=4.0
):
    duz__dt = (
        (1 - v)
        / (2 * np.pi)
        * (
            # - 6*c*x/(c**2 + x**2 + y**2)**2.5 #  Geertsma
            -420
            * a11
            * k**2
            * x
            * (a21 * k + c) ** 3
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 4.5
            - 30
            * a11
            * k
            * x
            * (a21 * k - c) ** 2
            / (x**2 + y**2 + (a21 * k - c) ** 2) ** 3.5
            + 30
            * a11
            * k
            * x
            * (a21 * k + c)
            * (a21 * k + c - 6 * k)
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 3.5
            + 3
            * a11
            * x
            * (4 * v - 3) ** 2
            * (a21 * k + c)
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 2.5
            + 3
            * a11
            * x
            * (4 * v - 3)
            * (a21 * k - c + 2 * k)
            / (x**2 + y**2 + (a21 * k - c + 2 * k) ** 2) ** 2.5
            + 3
            * a11
            * x
            * (4 * v - 3)
            * (a21 * k + c + 2 * k)
            / (x**2 + y**2 + (a21 * k + c + 2 * k) ** 2) ** 2.5
            + 3
            * a11
            * x
            * (-a21 * k + c + 2 * k)
            / (x**2 + y**2 + (a21 * k - c) ** 2) ** 2.5
            - 420
            * a12
            * k**2
            * x
            * (a22 * k + c) ** 3
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 4.5
            - 30
            * a12
            * k
            * x
            * (a22 * k - c) ** 2
            / (x**2 + y**2 + (a22 * k - c) ** 2) ** 3.5
            + 30
            * a12
            * k
            * x
            * (a22 * k + c)
            * (a22 * k + c - 6 * k)
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 3.5
            + 3
            * a12
            * x
            * (4 * v - 3) ** 2
            * (a22 * k + c)
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 2.5
            + 3
            * a12
            * x
            * (4 * v - 3)
            * (a22 * k - c + 2 * k)
            / (x**2 + y**2 + (a22 * k - c + 2 * k) ** 2) ** 2.5
            + 3
            * a12
            * x
            * (4 * v - 3)
            * (a22 * k + c + 2 * k)
            / (x**2 + y**2 + (a22 * k + c + 2 * k) ** 2) ** 2.5
            + 3
            * a12
            * x
            * (-a22 * k + c + 2 * k)
            / (x**2 + y**2 + (a22 * k - c) ** 2) ** 2.5
            + 3780
            * b11
            * k**3
            * x
            * (b21 * k + c) ** 4
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 5.5
            - 210
            * b11
            * k**2
            * x
            * (b21 * k - c) ** 3
            / (x**2 + y**2 + (b21 * k - c) ** 2) ** 4.5
            + 210
            * b11
            * k**2
            * x
            * (b21 * k + c) ** 2
            * (b21 * k + c - 12 * k)
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 4.5
            - 3
            * b11
            * k
            * x
            * (4 * v - 3) ** 2
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 2.5
            - 3
            * b11
            * k
            * x
            * (4 * v - 3)
            / (x**2 + y**2 + (b21 * k - c + 2 * k) ** 2) ** 2.5
            + 3
            * b11
            * k
            * x
            * (4 * v - 3)
            / (x**2 + y**2 + (b21 * k + c + 2 * k) ** 2) ** 2.5
            + 5
            * b11
            * k
            * x
            * (12 * v - 9)
            * (b21 * k - c + 2 * k) ** 2
            / (x**2 + y**2 + (b21 * k - c + 2 * k) ** 2) ** 3.5
            - 5
            * b11
            * k
            * x
            * (12 * v - 9)
            * (b21 * k + c + 2 * k) ** 2
            / (x**2 + y**2 + (b21 * k + c + 2 * k) ** 2) ** 3.5
            + 15
            * b11
            * k
            * x
            * (b21 * k - c)
            * (-b21 * k + c + 6 * k)
            / (x**2 + y**2 + (b21 * k - c) ** 2) ** 3.5
            + 3 * b11 * k * x / (x**2 + y**2 + (b21 * k - c) ** 2) ** 2.5
            + 5
            * b11
            * x
            * (
                36 * k**3
                - 3
                * k
                * (6 * k - (4 * v - 3) ** 2 * (b21 * k + c))
                * (b21 * k + c)
            )
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 3.5
        )
    )
    return duz__dt


def concavity_xx(
    x, y, c, k, v=0.25, a12=-0.778, a21=2.0, a22=2.80, b11=-0.2, b21=4.0
):
    return (
        (1 - v)
        / (2 * np.pi)
        * (
            # + 30*c*x**2/(c**2 + x**2 + y**2)**3.5 - 6*c/(c**2 + x**2 + y**2)**2.5 #  Geertsma
            +3780
            * a11
            * k**2
            * x**2
            * (a21 * k + c) ** 3
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 5.5
            - 420
            * a11
            * k**2
            * (a21 * k + c) ** 3
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 4.5
            + 210
            * a11
            * k
            * x**2
            * (a21 * k - c) ** 2
            / (x**2 + y**2 + (a21 * k - c) ** 2) ** 4.5
            - 210
            * a11
            * k
            * x**2
            * (a21 * k + c)
            * (a21 * k + c - 6 * k)
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 4.5
            - 30
            * a11
            * k
            * (a21 * k - c) ** 2
            / (x**2 + y**2 + (a21 * k - c) ** 2) ** 3.5
            + 30
            * a11
            * k
            * (a21 * k + c)
            * (a21 * k + c - 6 * k)
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 3.5
            - 15
            * a11
            * x**2
            * (4 * v - 3) ** 2
            * (a21 * k + c)
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 3.5
            - 15
            * a11
            * x**2
            * (4 * v - 3)
            * (a21 * k - c + 2 * k)
            / (x**2 + y**2 + (a21 * k - c + 2 * k) ** 2) ** 3.5
            - 15
            * a11
            * x**2
            * (4 * v - 3)
            * (a21 * k + c + 2 * k)
            / (x**2 + y**2 + (a21 * k + c + 2 * k) ** 2) ** 3.5
            - 15
            * a11
            * x**2
            * (-a21 * k + c + 2 * k)
            / (x**2 + y**2 + (a21 * k - c) ** 2) ** 3.5
            + 3
            * a11
            * (4 * v - 3) ** 2
            * (a21 * k + c)
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 2.5
            + 3
            * a11
            * (4 * v - 3)
            * (a21 * k - c + 2 * k)
            / (x**2 + y**2 + (a21 * k - c + 2 * k) ** 2) ** 2.5
            + 3
            * a11
            * (4 * v - 3)
            * (a21 * k + c + 2 * k)
            / (x**2 + y**2 + (a21 * k + c + 2 * k) ** 2) ** 2.5
            + 3
            * a11
            * (-a21 * k + c + 2 * k)
            / (x**2 + y**2 + (a21 * k - c) ** 2) ** 2.5
            + 3780
            * a12
            * k**2
            * x**2
            * (a22 * k + c) ** 3
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 5.5
            - 420
            * a12
            * k**2
            * (a22 * k + c) ** 3
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 4.5
            + 210
            * a12
            * k
            * x**2
            * (a22 * k - c) ** 2
            / (x**2 + y**2 + (a22 * k - c) ** 2) ** 4.5
            - 210
            * a12
            * k
            * x**2
            * (a22 * k + c)
            * (a22 * k + c - 6 * k)
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 4.5
            - 30
            * a12
            * k
            * (a22 * k - c) ** 2
            / (x**2 + y**2 + (a22 * k - c) ** 2) ** 3.5
            + 30
            * a12
            * k
            * (a22 * k + c)
            * (a22 * k + c - 6 * k)
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 3.5
            - 15
            * a12
            * x**2
            * (4 * v - 3) ** 2
            * (a22 * k + c)
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 3.5
            - 15
            * a12
            * x**2
            * (4 * v - 3)
            * (a22 * k - c + 2 * k)
            / (x**2 + y**2 + (a22 * k - c + 2 * k) ** 2) ** 3.5
            - 15
            * a12
            * x**2
            * (4 * v - 3)
            * (a22 * k + c + 2 * k)
            / (x**2 + y**2 + (a22 * k + c + 2 * k) ** 2) ** 3.5
            - 15
            * a12
            * x**2
            * (-a22 * k + c + 2 * k)
            / (x**2 + y**2 + (a22 * k - c) ** 2) ** 3.5
            + 3
            * a12
            * (4 * v - 3) ** 2
            * (a22 * k + c)
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 2.5
            + 3
            * a12
            * (4 * v - 3)
            * (a22 * k - c + 2 * k)
            / (x**2 + y**2 + (a22 * k - c + 2 * k) ** 2) ** 2.5
            + 3
            * a12
            * (4 * v - 3)
            * (a22 * k + c + 2 * k)
            / (x**2 + y**2 + (a22 * k + c + 2 * k) ** 2) ** 2.5
            + 3
            * a12
            * (-a22 * k + c + 2 * k)
            / (x**2 + y**2 + (a22 * k - c) ** 2) ** 2.5
            - 41580
            * b11
            * k**3
            * x**2
            * (b21 * k + c) ** 4
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 6.5
            + 3780
            * b11
            * k**3
            * (b21 * k + c) ** 4
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 5.5
            + 1890
            * b11
            * k**2
            * x**2
            * (b21 * k - c) ** 3
            / (x**2 + y**2 + (b21 * k - c) ** 2) ** 5.5
            - 1890
            * b11
            * k**2
            * x**2
            * (b21 * k + c) ** 2
            * (b21 * k + c - 12 * k)
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 5.5
            - 210
            * b11
            * k**2
            * (b21 * k - c) ** 3
            / (x**2 + y**2 + (b21 * k - c) ** 2) ** 4.5
            + 210
            * b11
            * k**2
            * (b21 * k + c) ** 2
            * (b21 * k + c - 12 * k)
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 4.5
            + 15
            * b11
            * k
            * x**2
            * (4 * v - 3) ** 2
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 3.5
            + 15
            * b11
            * k
            * x**2
            * (4 * v - 3)
            / (x**2 + y**2 + (b21 * k - c + 2 * k) ** 2) ** 3.5
            - 15
            * b11
            * k
            * x**2
            * (4 * v - 3)
            / (x**2 + y**2 + (b21 * k + c + 2 * k) ** 2) ** 3.5
            - 35
            * b11
            * k
            * x**2
            * (12 * v - 9)
            * (b21 * k - c + 2 * k) ** 2
            / (x**2 + y**2 + (b21 * k - c + 2 * k) ** 2) ** 4.5
            + 35
            * b11
            * k
            * x**2
            * (12 * v - 9)
            * (b21 * k + c + 2 * k) ** 2
            / (x**2 + y**2 + (b21 * k + c + 2 * k) ** 2) ** 4.5
            - 105
            * b11
            * k
            * x**2
            * (b21 * k - c)
            * (-b21 * k + c + 6 * k)
            / (x**2 + y**2 + (b21 * k - c) ** 2) ** 4.5
            - 15
            * b11
            * k
            * x**2
            / (x**2 + y**2 + (b21 * k - c) ** 2) ** 3.5
            - 3
            * b11
            * k
            * (4 * v - 3) ** 2
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 2.5
            - 3
            * b11
            * k
            * (4 * v - 3)
            / (x**2 + y**2 + (b21 * k - c + 2 * k) ** 2) ** 2.5
            + 3
            * b11
            * k
            * (4 * v - 3)
            / (x**2 + y**2 + (b21 * k + c + 2 * k) ** 2) ** 2.5
            + 5
            * b11
            * k
            * (12 * v - 9)
            * (b21 * k - c + 2 * k) ** 2
            / (x**2 + y**2 + (b21 * k - c + 2 * k) ** 2) ** 3.5
            - 5
            * b11
            * k
            * (12 * v - 9)
            * (b21 * k + c + 2 * k) ** 2
            / (x**2 + y**2 + (b21 * k + c + 2 * k) ** 2) ** 3.5
            + 15
            * b11
            * k
            * (b21 * k - c)
            * (-b21 * k + c + 6 * k)
            / (x**2 + y**2 + (b21 * k - c) ** 2) ** 3.5
            + 3 * b11 * k / (x**2 + y**2 + (b21 * k - c) ** 2) ** 2.5
            - 35
            * b11
            * x**2
            * (
                36 * k**3
                - 3
                * k
                * (6 * k - (4 * v - 3) ** 2 * (b21 * k + c))
                * (b21 * k + c)
            )
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 4.5
            + 5
            * b11
            * (
                36 * k**3
                - 3
                * k
                * (6 * k - (4 * v - 3) ** 2 * (b21 * k + c))
                * (b21 * k + c)
            )
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 3.5
        )
    )


def concavity_xy(
    x, y, c, k, v=0.25, a12=-0.778, a21=2.0, a22=2.80, b11=-0.2, b21=4.0
):
    return (
        (1 - v)
        / (2 * np.pi)
        * (
            # + 30*c*x*y/(c**2 + x**2 + y**2)**3.5 #  Geertsma
            +3780
            * a11
            * k**2
            * x
            * y
            * (a21 * k + c) ** 3
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 5.5
            + 210
            * a11
            * k
            * x
            * y
            * (a21 * k - c) ** 2
            / (x**2 + y**2 + (a21 * k - c) ** 2) ** 4.5
            - 210
            * a11
            * k
            * x
            * y
            * (a21 * k + c)
            * (a21 * k + c - 6 * k)
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 4.5
            - 15
            * a11
            * x
            * y
            * (4 * v - 3) ** 2
            * (a21 * k + c)
            / (x**2 + y**2 + (a21 * k + c) ** 2) ** 3.5
            - 15
            * a11
            * x
            * y
            * (4 * v - 3)
            * (a21 * k - c + 2 * k)
            / (x**2 + y**2 + (a21 * k - c + 2 * k) ** 2) ** 3.5
            - 15
            * a11
            * x
            * y
            * (4 * v - 3)
            * (a21 * k + c + 2 * k)
            / (x**2 + y**2 + (a21 * k + c + 2 * k) ** 2) ** 3.5
            - 15
            * a11
            * x
            * y
            * (-a21 * k + c + 2 * k)
            / (x**2 + y**2 + (a21 * k - c) ** 2) ** 3.5
            + 3780
            * a12
            * k**2
            * x
            * y
            * (a22 * k + c) ** 3
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 5.5
            + 210
            * a12
            * k
            * x
            * y
            * (a22 * k - c) ** 2
            / (x**2 + y**2 + (a22 * k - c) ** 2) ** 4.5
            - 210
            * a12
            * k
            * x
            * y
            * (a22 * k + c)
            * (a22 * k + c - 6 * k)
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 4.5
            - 15
            * a12
            * x
            * y
            * (4 * v - 3) ** 2
            * (a22 * k + c)
            / (x**2 + y**2 + (a22 * k + c) ** 2) ** 3.5
            - 15
            * a12
            * x
            * y
            * (4 * v - 3)
            * (a22 * k - c + 2 * k)
            / (x**2 + y**2 + (a22 * k - c + 2 * k) ** 2) ** 3.5
            - 15
            * a12
            * x
            * y
            * (4 * v - 3)
            * (a22 * k + c + 2 * k)
            / (x**2 + y**2 + (a22 * k + c + 2 * k) ** 2) ** 3.5
            - 15
            * a12
            * x
            * y
            * (-a22 * k + c + 2 * k)
            / (x**2 + y**2 + (a22 * k - c) ** 2) ** 3.5
            - 41580
            * b11
            * k**3
            * x
            * y
            * (b21 * k + c) ** 4
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 6.5
            + 1890
            * b11
            * k**2
            * x
            * y
            * (b21 * k - c) ** 3
            / (x**2 + y**2 + (b21 * k - c) ** 2) ** 5.5
            - 1890
            * b11
            * k**2
            * x
            * y
            * (b21 * k + c) ** 2
            * (b21 * k + c - 12 * k)
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 5.5
            + 15
            * b11
            * k
            * x
            * y
            * (4 * v - 3) ** 2
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 3.5
            + 15
            * b11
            * k
            * x
            * y
            * (4 * v - 3)
            / (x**2 + y**2 + (b21 * k - c + 2 * k) ** 2) ** 3.5
            - 15
            * b11
            * k
            * x
            * y
            * (4 * v - 3)
            / (x**2 + y**2 + (b21 * k + c + 2 * k) ** 2) ** 3.5
            + 35
            * b11
            * k
            * x
            * y
            * (12 * v - 9)
            * (b21 * k - c + 2 * k) ** 2
            / (x**2 + y**2 + (b21 * k - c + 2 * k) ** 2) ** 4.5
            + 35
            * b11
            * k
            * x
            * y
            * (12 * v - 9)
            * (b21 * k + c + 2 * k) ** 2
            / (x**2 + y**2 + (b21 * k + c + 2 * k) ** 2) ** 4.5
            - 105
            * b11
            * k
            * x
            * y
            * (b21 * k - c)
            * (-b21 * k + c + 6 * k)
            / (x**2 + y**2 + (b21 * k - c) ** 2) ** 4.5
            - 15
            * b11
            * k
            * x
            * y
            / (x**2 + y**2 + (b21 * k - c) ** 2) ** 3.5
            - 35
            * b11
            * x
            * y
            * (
                36 * k**3
                - 3
                * k
                * (6 * k - (4 * v - 3) ** 2 * (b21 * k + c))
                * (b21 * k + c)
            )
            / (x**2 + y**2 + (b21 * k + c) ** 2) ** 4.5
        )
    )
