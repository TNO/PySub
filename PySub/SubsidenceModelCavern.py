"""Model storing the SubsidenceModel class for Subsidence due to volume change.
"""
from PySub import SubsidenceModelBase as _SubsidenceModelBase
from PySub.SubsidenceModelBase import EPSILON
from PySub import utils as _utils
import numpy as np
import xarray as xr


class SubsidenceModel(_SubsidenceModelBase.SubsidenceModel):
    def __init__(self, name, project_folder=None):
        super().__init__(name, project_folder=project_folder)

    @property
    def lengths(self):
        return self._fetch("lengths")

    @property
    def volume_change(self):
        return self._fetch("volume_change")

    @property
    def calc_vars(self):
        return [
            "subsidence",
            "volume",
            "slope",
            "concavity",
            "subsidence_rate",
        ]

    @property
    def vars_to_calculate(self):
        return [
            "subsidence_model_type",
            "depth_to_basements",
            "depths",
            "lengths",
            "knothe_angles",
            "volume_change",
            "viscous_tau",
            "depth_to_basements_moved",
        ]

    @property
    def vars_to_build(self):
        return [
            "reservoirs",
            "timesteps",
            "points",
            "observation_points",
            "dx",
            "dy",
            "influence_radius",
            "shapes",
            "bounds",
        ]

    @property
    def depth_to_basements_moved(self):
        return self._fetch("depth_to_basements_moved")

    @property
    def viscous_tau(self):
        return self._fetch("viscous_tau")

    def idxy(self):
        if self.hasattr("shapes") and self.built:
            x = []
            y = []
            for shape in self.shapes:
                x.append(shape.x)
                y.append(shape.y)
            ix, iy = np.searchsorted(self.x, x), np.searchsorted(self.y, y)

            return ix, iy
        else:
            if not self.hasattr("shapes"):
                raise Exception(
                    "No coordinates set. Set coordinates before masking the grid."
                )
            if not self.built:
                raise Exception(
                    "Build grid before masking the grid for caverns."
                )

    def move_basements(self):
        """When the necessary input is detected, this function will set the
        depth_to_basments variable as changing over time between the initital
        depth_to_basement and the depth_to_basement_moved variables. The rate
        of movement of the rigid basement is dictated by the variable
        viscous_tau.

        Sets
        -------
        depth_to_basements

        """
        if (
            self.hasattr("depth_to_basements")
            and self.hasattr("viscous_tau")
            and self.hasattr("depth_to_basements_moved")
        ):  # Has all the ingredients to calculate moving basement
            time = (
                xr.apply_ufunc(
                    _utils.datetime_to_years_as_float, self.grid.time
                )
                * 365.25
                * 24
                * 3600
            )

            moving_rigid_basement = xr.apply_ufunc(
                _utils.moving_rigid_basement,
                self.depth_to_basements,
                self.depth_to_basements_moved,
                self.depths,
                time,
                self.viscous_tau,
                input_core_dims=[[], [], [], ["time"], []],
                output_core_dims=[["time"]],
                vectorize=True,
            )
            self.grid["depth_to_basements"] = moving_rigid_basement

    def seperate_depths(self):
        """Seperate the elongated caverns into multiple vertical segments.
        This adds the new dimension "layers".

        Returns
        -------
        None.

        """
        if self.hasattr("lengths") and self.hasattr("depths"):
            number_of_layers = np.ceil(self.lengths / self.dx).astype(int)

            check_for_layer_thickness = xr.Dataset(
                coords=number_of_layers.coords
            ).expand_dims("layer")

            check_for_layer_thickness["layer"] = (
                np.arange(number_of_layers.max().values) + 1
            )
            check_for_layer_thickness["thickness"] = self.lengths - self.dx * (
                check_for_layer_thickness["layer"] - 1
            )

            layer_thickness = check_for_layer_thickness["thickness"].clip(
                0, self.dx
            )

            self.grid["contribution"] = layer_thickness / self.lengths
            initial_depths = self.depths.expand_dims("layer")
            initial_depths["layer"] = [0]
            layer_depths = self.depths + layer_thickness.cumsum("layer")

            self.grid["depths"] = xr.concat(
                [initial_depths, layer_depths], "layer"
            )

    def set_lengths(self, lengths):
        if not _utils.is_iterable(lengths):
            raise Exception(
                "Lengths set must be iterable objects with the dimensions (m), where m is the number of caverns."
            )

        lengths = np.array(lengths)
        if len(lengths.shape) != 1:
            raise Exception(
                "Lengths set must be iterable objects with the dimensions (m), where m is the number of caverns."
            )

        self._check_dim1D("lengths", lengths, dims_equal="reservoirs")

        self.set_1D_or_2D("lengths", lengths)
        self.seperate_depths()

    def set_depths(self, depths, layer=None):
        """Sets the depths as a part of the model and performs checks.
        depths : list float/int, optional
            Depths to the top of each reservoirs in m. The list must have
            the same length as the number of reservoirs in
            the model. The default is None. Raises Exception when None.

        Sets
        ----
        depths
        """
        self.set_1D_or_2D("depths", depths, layer=layer)
        self.seperate_depths()

    def set_depth_to_basements(self, depth_to_basements):
        """Sets the depths of the basements as a part of the model, perform checks.

        depth_to_basements : list, float/int
            Depth to the rigid basement for the van Opstal nucleus of strain
            method in m. The list must have the same length as the number of reservoirs in
            the model. The default is None. Raises Exception when None.

        Sets
        ----
        depth_to_basements
        """
        if _utils.isnan(depth_to_basements):
            depth_to_basements = None
        if not depth_to_basements is None:
            _utils._check_low_high(
                depth_to_basements, "depth_to_basements", EPSILON, 100000
            )
            self._check_dim1D("depth to basements", depth_to_basements)
            self.set_1D_or_2D("depth_to_basements", depth_to_basements)

            self.move_basements()

    def set_depth_to_basements_moved(self, depth_to_basements_moved):
        if not depth_to_basements_moved is None:
            if not _utils.is_iterable(depth_to_basements_moved):
                raise Exception(
                    "Moved depth to basements set must be iterable objects with the dimensions (m), where m is the number of caverns."
                )
            depth_to_basements_moved = np.array(depth_to_basements_moved)
            _utils._check_low_high(
                depth_to_basements_moved, "depth_to_basements_moved", 0, 100000
            )
            self._check_dim1D(
                "depth_to_basements_moved",
                depth_to_basements_moved,
                dims_equal="reservoirs",
            )
            self.set_1D_or_2D(
                "depth_to_basements_moved", depth_to_basements_moved
            )

            self.move_basements()

    def set_viscous_tau(self, viscous_tau):
        if not viscous_tau is None:
            if not _utils.is_iterable(viscous_tau):
                raise Exception(
                    "Moved depth to basements set must be iterable objects with the dimensions (m), where m is the number of caverns."
                )
            viscous_tau = np.array(viscous_tau)
            _utils._check_low_high(
                viscous_tau,
                "viscous_tau",
                -_SubsidenceModelBase.EPSILON,
                np.inf,
            )
            self._check_dim1D(
                "viscous_tau", viscous_tau, dims_equal="reservoirs"
            )
            self.set_1D_or_2D("viscous_tau", viscous_tau)

            self.move_basements()

    def set_volume_change(self, volume_change):
        try:
            volume_change = np.array(volume_change)
        except:
            raise Exception(
                f"Invalid type for input of volume_change: {type(volume_change)}."
            )
        if len(volume_change.shape) == 2:
            self._check_dimND(
                "volume_change",
                volume_change,
                dims_equal=["reservoirs", "timesteps"],
            )
        else:
            raise Exception(
                f"Invalid number of dimensions of volume_change: {len(volume_change.shape)}. Use 2 (for reservoir and timesteps)."
            )
        self.drop_from_grid(volume_change, "volume_change")

    def project_compaction(self):
        if not hasattr(self.grid, "reservoir_mask"):
            raise Exception(
                f"Model {self.name} has no reservoir mask set. Run model.mask_reservoirs() before projecting compaction."
            )
        compaction = self.grid.reservoir_mask
        compaction = compaction * self.volume_change
        self.set_compaction(compaction)

    def calculate_subsidence(self, _print=True):
        if _print:
            print(f"Calculating subsidence, model: {self.name}")
        super().calculate_subsidence(_print=False)
        self.grid["subsidence"] = (
            self.grid["subsidence"] * self.grid.contribution
        ).sum("layer")

        if _print:
            print(f"Calculated subsidence, model: {self.name}")
        return self.grid["subsidence"]

    def calculate_slope(self, reservoir=None, numeric=False, _print=True):
        if _print:
            print(f"Calculating subsidence gradient, model: {self.name}")
        super().calculate_slope(
            reservoir=reservoir, numeric=numeric, _print=False
        )
        self.grid["slope_x"] = (
            self.grid["slope_x"] * self.grid.contribution
        ).sum("layer")
        self.grid["slope_y"] = (
            self.grid["slope_y"] * self.grid.contribution
        ).sum("layer")
        self.grid["slope"] = (self.grid["slope"] * self.grid.contribution).sum(
            "layer"
        )
        if _print:
            print(f"Calculated subsidence gradient, model: {self.name}")
        return self.grid["slope"]

    def calculate_concavity(self, reservoir=None, numeric=False, _print=True):
        if _print:
            print(f"Calculating subsidence gradient, model: {self.name}")
        super().calculate_concavity(
            reservoir=reservoir, numeric=numeric, _print=False
        )
        self.grid["concavity_xx"] = (
            self.grid["concavity_xx"] * self.grid.contribution
        ).sum("layer")
        self.grid["concavity_xy"] = (
            self.grid["concavity_xy"] * self.grid.contribution
        ).sum("layer")
        self.grid["concavity_yx"] = (
            self.grid["concavity_yx"] * self.grid.contribution
        ).sum("layer")
        self.grid["concavity_yy"] = (
            self.grid["concavity_yy"] * self.grid.contribution
        ).sum("layer")
        self.grid["concavity"] = (
            self.grid["concavity"] * self.grid.contribution
        ).sum("layer")
        if _print:
            print(f"Calculated subsidence gradient, model: {self.name}")
        return self.grid["concavity"]
