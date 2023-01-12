import xarray as xr
import numpy as np
from PySub import utils as _utils
from PySub import grid_utils as _grid_utils
from numba import njit

class LinearCompaction:
    """Caclulates the compaction in volume based on pressure differences using a 
    linear compaction model.
    """
    def __init__(self):
        """Initialize the LinearCompaction object.

        Returns
        -------
        LineraCompaction object.

        """
        self.type = 'linear'
        self.variables = ['thickness', 'compaction_coefficients']

    def __repr__(self):
        return self.type

    def compute(self, grid):
        """Compute the compaction for a uniformly distributed pressure for each reservoir.

        Parameters
        ----------
        grid : xr.Dataset
            xarray dataset with at least the dimensions x, y, reservoir and time,
            the attributes dx, dy, and data variables pressure, thickness and compaction_coefficients.
            The data variables should have only the dimension reservoir, except for
            pressure, which should have the dimensions reservoir and time.

        Returns
        -------
        np.ndarray
            The compaction in m³ in the dimensions (reservoir, time).
        """
        pini = grid.pressures.isel(time = 0)
        compaction = xr.apply_ufunc(linear_strain,
                                    grid.dx, 
                                    grid.dy, 
                                    grid.thickness, 
                                    grid.compaction_coefficients, 
                                    pini, 
                                    grid.pressures)
        return compaction
    
class TimeDecayCompaction:
    """Caclulates the compaction in volume based on pressure differences using a 
    time decay compaction model.
    
    Source
    ------
    Mossop A (2012) An explanation for anomalous time dependent subsidence.
    
    Implementation from:
        https://pdfs.semanticscholar.org/0ae5/682f6abbc55afe31792d2c74f04bf4d99a82.pdf
    """
    def __init__(self):
        """Initialize the TimeDecayCompaction object.

        Returns
        -------
        TimeDecayCompaction object.

        """
        self.type = 'time-decay'
        self.variables = ['thickness', 'compaction_coefficients', 'tau']
    
    def __repr__(self):
        return self.type

    def compute(self, grid):
        """Compute the compaction for a uniformly distributed pressure for each reservoir.

        Parameters
        ----------
        grid : xr.Dataset
            xarray dataset with at least the dimensions x, y, reservoir and time,
            the attributes dx, dy, and data variables pressure, thickness and compaction_coefficients.
            The data variables should have only the dimension reservoir, except for
            pressure, which should have the dimensions reservoir and time.

        Returns
        -------
        np.ndarray
            The compaction in m³ in the dimensions (reservoir, time).
        """
        
        dP = -xr.apply_ufunc(_utils.diff_xarray, 
                             grid.pressures.isel(time = np.arange(1, grid.dims['time'])), 
                             grid.pressures.isel(time = 0)
                             )
        # Superposition
        time = _utils.datetime_to_years_as_float(grid.time.values)
        time_2D = time[None, :]
        dt = time_2D - time_2D.T
        dt = np.where(dt < 0, 0, dt) * 365.25*24*3600 
        ds_dt = xr.DataArray(dt[:-1, 1:], dims = ['time', 'timestep'],
                           coords = dict(time = ('time', grid.time.values[1:]), 
                                         timestep = ('timestep', grid.time.values[1:])))
        
        time_decay = xr.apply_ufunc(_utils.get_time_decay,
                                    ds_dt, 
                                    grid.tau)
        
        strain = xr.apply_ufunc(time_decay_strain,
                                grid.dx, 
                                grid.dy, 
                                grid.thickness,
                                grid.compaction_coefficients,
                                dP,
                                time_decay)
        
        compaction = strain.sum(dim = 'time')
        compaction = compaction.rename({'timestep': 'time'})
        fill = xr.zeros_like(compaction).isel(time = 0)
        fill['time'] = grid.time.values[0]
        return xr.concat([fill, compaction], dim = 'time')

def inner_iterations(values, number_of_inner_iterations):
    interpolated_values = []
    for i in range(len(values) - 1):
        inner = (values[i+1] - values[i]) / number_of_inner_iterations
        for j in range(number_of_inner_iterations):
            interpolated_values.append(values[i] + j*inner)
    return interpolated_values

class RateTypeCompaction:  
    """Caclulates the compaction in volume based on pressure differences using a 
    ratype compaction model.
    
    Source
    ------
    Pruiksma JP (2015) Isotach formulation of the rate type compaction model for sandstone.
    """
    def __init__(self, epsilon = 2.5e-6):
        self.epsilon = epsilon
        self.type = 'ratetype'
        self.variables = ['thickness', 'compaction_coefficients', 'depths', 'cmref', 'b', 'reference_stress_rates']
    
    def __repr__(self):
        return self.type
        
    def compute(self, grid):
        """Compute the compaction for a uniformly distributed pressure for each reservoir.

        Parameters
        ----------
        grid : xr.Dataset
            xarray dataset with at least the dimensions x, y, reservoir and time,
            the attributes dx, dy, and data variables pressure, thickness and compaction_coefficients.
            The data variables should have only the dimension reservoir, except for
            pressure, which should have the dimensions reservoir and time.

        Returns
        -------
        np.ndarray
            The compaction in m³ in the dimensions (reservoir, time).
        """
        var_dims = np.array([len(grid[var].shape) for var in grid.variables if var not in ['reservoir_mask']])
        gridded = (var_dims > 2).any()
        if gridded:
            compaction = self.compute_grid(grid)
        else:
            compaction = self.compute_1D(grid)
        return compaction

    
    def compute_1D(self, grid):
        P = grid.pressures
        sigma_vert_Pascal = grid.density * 9.81 * grid.depths # in Pascal
        sigma_vert_bar = sigma_vert_Pascal * 1e-5 # in bars
        sigma = sigma_vert_bar - P # sigma is stress in bar
        if (sigma < 0).any():
            raise(Exception(f'Pressure ({round(np.max(P.values), 3)}) higher than vertical pressure ({round(np.max(sigma_vert_bar.values),3)})'))
        
        timesteps = _utils.datetime_to_years_as_float(grid.time.values)
        timesteps = xr.DataArray(timesteps, coords = {'time': grid.time})
        
        start_inner_itterations = 16
        strain_prev = xr.apply_ufunc(
            RTiCM,
            timesteps,
            sigma,
            grid.compaction_coefficients,
            grid.cmref, grid.b, grid.reference_stress_rates, 
            start_inner_itterations,
            input_core_dims = [['time'],['time'], [], [], [], [], []],
            output_core_dims = [['time']],
            vectorize = True,
            )
        
        mae = [1]
        diff_mae = 10
        multiplier = 2
        while diff_mae > self.epsilon:
            number_of_inner_iterations = start_inner_itterations * multiplier
            strain_next =xr.apply_ufunc(
                RTiCM,
                timesteps,
                sigma,
                grid.compaction_coefficients,
                grid.cmref, grid.b, grid.reference_stress_rates, 
                number_of_inner_iterations,
                input_core_dims = [['time'],['time'], [], [], [], [], []],
                output_core_dims = [['time']],
                vectorize = True,
                )
            mae.append(np.mean(np.abs(strain_next - strain_prev))) 
            diff_mae = mae[-2] - mae[-1]
            multiplier *=2
            strain_prev = strain_next.copy()
        
        compaction = strain_next * grid.thickness * grid.dx * grid.dy
        
        return grid.reservoir_mask * compaction


    def compute_grid(self, grid):
        """Compute the compaction for a spatially varying pressure or any other
        spatially varying compaction parameter.

        Parameters
        ----------
        grid : xr.Dataset
            xarray dataset with at least the dimensions x, y, reservoir and time,
            the attributes dx, dy, and data variables pressure, thickness and compaction_coefficients.
            The data variables should have the dimensions (y, x, reservoir), except for
            pressure, which should have the additional dimension time.

        Returns
        -------
        np.ndarray
            The compaction in m³ in the dimensions (y, x, reservoir, time).
        """
        grid = grid.transpose('y', 'x', 'reservoir', 'time', ...) 
        P = grid.pressures
        grid = grid.copy()
        sigma_vert_Pascal = grid.density * 9.81 * grid.depths # in Pascal
        sigma_vert_bar = sigma_vert_Pascal * 1e-5 # in bars
        sigma = sigma_vert_bar - P # sigma is stress in bar
        grid['sigma'] = sigma
        _grid_utils.convert_to_grid(grid, 'sigma')
        sigma = grid['sigma']
        if (sigma.max(dim = ('x', 'y')) < -1e-9).any():
            raise(Exception(f'Pressure ({round(np.max(P.values), 3)}) higher than vertical pressure ({round(np.max(sigma_vert_bar.values),3)})'))
        
        timesteps = _utils.datetime_to_years_as_float(grid.time.values)
        timesteps = xr.DataArray(timesteps, coords = {'time': grid.time})
        
        sigma, _, _, _, _ = xr.broadcast(sigma, grid.y, grid.x, grid.reservoir, grid.time)
        compaction_coefficients, _, _ = xr.broadcast(grid.compaction_coefficients, grid.x, grid.y)
        cmref, _, _ = xr.broadcast(
            xr.where(
                grid.cmref <= 0, 1, grid.cmref), 
            grid.x, grid.y)
        b, _, _ = xr.broadcast(
            xr.where(grid.b <= 0, 1, grid.b), 
            grid.x, grid.y)
        refference_stress_rates, _, _ = xr.broadcast(
            xr.where(grid.reference_stress_rates  <= 0,
                      1, grid.reference_stress_rates), 
            grid.x, grid.y)
        
        start_inner_itterations = 100
        
        strain_next = xr.apply_ufunc(
            RTiCM2D,
            timesteps,
            sigma, 
            compaction_coefficients, 
            cmref, 
            b, 
            refference_stress_rates, 
            start_inner_itterations, 
            input_core_dims = [['time'], ['y', 'x', 'time'], ['y', 'x'], ['y', 'x'], ['y', 'x'], ['y', 'x'], []],
            output_core_dims = [['y', 'x', 'time']],
            vectorize = True,
            )
        
        compaction = strain_next * grid.thickness * grid.dx * grid.dy
        return compaction.transpose('y', 'x', 'reservoir', 'time')


def linear_strain(dx, dy, thickness, compaction_coefficients, initial_pressures, pressures):
    return dx*dy*thickness * compaction_coefficients * (initial_pressures - pressures)    

def time_decay_strain(dx, dy, thickness, compaction_coefficients, dP, time_decay):
    return dx * dy * compaction_coefficients * thickness * dP * time_decay
    
@njit(fastmath = True, parallel = True)
def RTiCM(timesteps, sigma, cmd, cmref, b, reference_stress_rates, number_of_inner_iterations):
    """Isotach formulation of the ratetype compaction model (Pruiksma 2015)
    https://www.sciencedirect.com/science/article/pii/S1365160915001525?via%3Dihub#bib18
    
    Notation from Pruiksma '15 in [].

    Parameters
    ----------
    timesteps : 1D np.ndarray, floats
        The time past since sigma ref in years for each timestep in the model. 
    sigma [sigma(t)]: 1D np.ndarray, floats
        The stress differences caused by pressure depletion (positive means loading,
        pressure decrease in the resservoir). One value per timestep in the model.
        Does not have to be in- or decreasing.
    cmd [cmd]: float
        direct compaction coefficient.
    cmref [cmref]: float
        reference compaction coefficient.
    b [b]: float
        Material creep coefficient.
    reference_stress_rates [dot sigma ref]: float
        State parameter, reference stress rate.
    number_of_inner_iterations : int
        Number if iterations between timesteps. Linear interpolation of the sigma
        values occurs. Necessary for stability of the process. Interpolation before
        the algoritm causes overflow in memory more easily.
        

    Returns
    -------
    strain [epsilon]: np.ndarray, floats
        The strain over the loading history for each timestep in timesteps.

    """

    sigma_ref = sigma[0]
    number_of_timesteps = len(timesteps)
    dt = np.zeros(number_of_timesteps)
    dt[1:] = np.diff(timesteps)
    shear_strain = 0 
    direct_strain = 0
    strain = np.zeros((number_of_timesteps))
    strain0 = - cmref * sigma_ref
    
    for i in range(0, number_of_timesteps - 1):
        sig_inner = (sigma[i+1] - sigma[i]) / number_of_inner_iterations
        for j in range(0, number_of_inner_iterations):
            _strain = shear_strain + direct_strain # _strain is current strain, with inner iterations
            
            cm = ((_strain - strain0) / 
                  (sigma[i] + (j-1) * sig_inner))
            shear_strain_rate = reference_stress_rates * (cm - cmd) * (cm / cmref)**(-1/b)
            shear_strain = shear_strain + shear_strain_rate * dt[i+1] / number_of_inner_iterations
            direct_strain = cmd * (sigma[i] + j*sig_inner - sigma_ref)

        strain[i + 1] = direct_strain + shear_strain
    return strain


@njit(parallel = True)
def RTiCM_reservoirs(timesteps, sigma, cmd, cmref, b, reference_stress_rates, number_of_inner_itterations):
    strain = np.zeros(sigma.shape)
    for i in range(len(strain)):
        strain[i] = RTiCM(timesteps, sigma[i], cmd[i], cmref[i], b[i], reference_stress_rates[i], number_of_inner_itterations)
    return strain

# @njit(parallel = True) # Causes errors
def RTiCM2D_reservoirs(timesteps, sigma, cmd, cmref, b, reference_stress_rates, number_of_inner_itterations):
    strain = np.zeros(sigma.shape)
    for i in range(sigma.shape[2]):
        strain[..., i, :] = RTiCM2D(timesteps, sigma[..., i, :], cmd[..., i], cmref[..., i], b[..., i], 
                            reference_stress_rates[..., i], number_of_inner_itterations)
    return strain
     
# @njit(parallel = True, fastmath = True) # Causes errors, but still much faster than apply_ufunc
def RTiCM2D(timesteps, sigma, cmd, cmref, b, reference_stress_rates, number_of_inner_itterations):
    """Isotach formulation of the ratetype compaction model (Pruiksma 2015)
    https://www.sciencedirect.com/science/article/pii/S1365160915001525?via%3Dihub#bib18
    
    Notation from Pruiksma '15 in [].

    Parameters
    ----------
    timesteps : np.ndarray, floats
        The time past since sigma ref in years for each timestep in the model. 
    sigma [sigma(t)]: 3D np.ndarray, floats
        The stress differences caused by pressure depletion (positive means loading,
        pressure decrease in the resservoir). One value per timestep in the model.
        Does not have to be in- or decreasing.
    cmd [cmd]: float
        direct compaction coefficient.
    cmref [cmref]: float
        reference compaction coefficient.
    b [b]: float
        Material creep coefficient.
    reference_stress_rates [dot sigma ref]: float
        State parameter, reference stress rate.
    number_of_inner_itterations : int
        Number if itteration between timesteps. Linear interpolation of the sigma
        values occurs. Necessary for stability of the process. 

    Returns
    -------
    strain [epsilon]: np.ndarray, floats
        The strain over the loading history for each timestep in timesteps.

    """

    sigma_ref = sigma[:, :, 0]
    number_of_timesteps = len(timesteps)
    dt = np.zeros(number_of_timesteps)
    dt[1:] = np.diff(timesteps)
    shear_strain = np.zeros_like(sigma_ref) 
    direct_strain = np.zeros_like(sigma_ref) 
    strain = np.zeros_like(sigma)
    strain0 = - cmref * sigma_ref
    
    for i in range(0, number_of_timesteps - 1):
        sig_inner = (sigma[:, :, i + 1] - sigma[:, :, i]) / number_of_inner_itterations
        for j in range(0, number_of_inner_itterations):
            _strain = shear_strain + direct_strain # _strain is current strain, with inner iterations
            
            cm = ((_strain - strain0) / 
                  (sigma[:, :, i] + (j-1) * sig_inner))
            shear_strain_rate = reference_stress_rates * (cm - cmd) * (cm / cmref)**(-1/b)
            shear_strain = shear_strain + shear_strain_rate * dt[i + 1] / number_of_inner_itterations
            direct_strain = cmd * (sigma[:, :, i] + j*sig_inner - sigma_ref)

        strain[:, :, i + 1] = direct_strain + shear_strain
    return strain

