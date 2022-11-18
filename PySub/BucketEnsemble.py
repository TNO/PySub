# -*- coding: utf-8 -*-
"""Create models sampled from a distribution entered n buckets.
"""
import numpy as np
import xarray as xr
import pandas as pd
import random
from tqdm import tqdm
import csv
import os
from PySub import SubsidenceModelGas as _SubsidenceModelGas
from PySub import SubsidenceKernel as _SubsidenceKernel
from PySub import utils as _utils
from PySub import memory as _memory
from PySub import Geometries as _Geometries



class VariableBuckets(dict):
    """Object to store values of variables and their probability for being sampled.
    Inherits from dictionary.
    
    General structure:
        VariableBucket[variable]['Values']
        VariableBucket[variable]'Probabilities']
    """
    def __init__(self, variables = []):
        for variable in variables:
            self.add_new_variable(variable)
        
    
    @property
    def variables(self):
        """
        Return the variables set. These are the keys to the items in the VariableBucket 
        object.

        Returns
        -------
        list
            A list of variables of which the values and probabilities are stored in
            the VariableBucket object.

        """
        return list(self.keys())
    
    @property
    def values(self):
        """
        Return the values set. These are the values for each variable as a list.

        Returns
        -------
        list
            A (possibly ragged) list of values of any data type.

        """
        return [var['Values'] for k, var in self.items()]
    
    @property
    def probabilities(self):
        """
        Return the values set. These are the values for each variable as a list.

        Returns
        -------
        list
            A (possibly ragged) list of floats representing the probability of 
            the corresponding value being picked by the sample method.

        """
        return [var['Probabilities'] for k, var in self.items()]
    
    def check_probabilities(self, additional_message = ''):
        """Check the validity of all probabilities in the list of variables.
        The probability must be entered as values between 0 and 1 and the total 
        must add up to 1.

        Parameters
        ----------
        additional_message : str, optional
            Additional information to print to the Exception. The default is ''.

        Raises
        ------
        Exception
            When probabilities do not add up to 1 or values not between 0 and 1 
            are in the probabilities.

        """
        for key, item in self.items():
            probabilities = item['Probabilities']
            if not np.isclose(np.sum(probabilities), 1) and len(probabilities) > 0:
                raise Exception(f'{additional_message}     The sum of the probabilities for the parameter {key} is not 1.\n{probabilities}')
            if ((np.array(probabilities) < 0).any() or
                (np.array(probabilities) > 1).any()):
                raise Exception(f'{additional_message}     The sum of the probabilities for the parameter {key} contain invalid values.\n{probabilities}')
    
    def fetch(self, indices, to_fetch = 'values'):
        """fetch a value for each variable.

        Parameters
        ----------
        indices : list, int
            List with the same length as the variables. The list must contain
            integer values for each value.

        Returns
        -------
        out : list
            A list with the same length as the number of variables. The list contains 
            any type of object stored in the VariableBuckt. If an index exceeds the 
            length of the list of variables stored under "Values", the list will 
            contain None for that index.

        """
        _to_fetch = getattr(self, to_fetch)
        out = []
        for i, c in enumerate(indices):
            if c <= len(_to_fetch[i]) and len(_to_fetch[i]) != 0:
                out.append(_to_fetch[i][c])
            else:
                out.append(None)
        return out
    
    def add_new_variable(self, variable):
        """Add a variable with empty lists of Values and Probabilities.
        """
        self[variable] = {'Values': [], 'Probabilities': []}
    
    def as_list(self):
        """Return the dictionary storing the values and probabilities as lists.

        Returns
        -------
        values
            A (possibly ragged) list of variables of any datatype.
        probabilities
            A (possibly ragged) list of floats.

        """
        return self.values, self.probabilities
    
    def sample(self, seed = None):
        """For each variable, sample a value stored in its key "Values" based
        on the probability of it being chosen indicated by its key "Probabilities".

        Returns
        -------
        list
            A list with the same length as the number of variables. The list contains 
            any type of object stored in the VariableBuckt. If an index exceeds the 
            length of the list of variables stored under "Values", the list will 
            contain None for that index.

        """
        random.seed(seed)
        if not hasattr(self, 'cumulative_probabilities'):
            _, probabilities = self.as_list()
            probabilities = _utils.pad_array(probabilities)
            self.cumulative_probabilities = np.cumsum(probabilities, axis = 1)
        chosen = np.array(
            [np.searchsorted(
                self.cumulative_probabilities[i], 
                random.random()) 
            for i in range(len(self.values))])
        chosen_dict = {var:choice for var, choice in zip(self.variables, chosen)}
        return self.fetch(chosen), chosen_dict
    
    def write_lookup_table(self, file_loc):
        values, probabilities = self.as_list()
        max_length = max(len(entry) for entry in values)
        options = np.arange(max_length) + 1
        value_dict = {var:val for var, val in zip(self.variables, values)}
        value_dict['Start pressure'] = [o[0] for o in value_dict['pressures']]
        value_dict['End pressure'] = [o[-1] for o in value_dict['pressures']]
        del value_dict['pressures']
        
        value_dict = _utils.fill_dict_with_none(value_dict)
        value_df = pd.DataFrame(value_dict, index = options)
        value_df.index.name = 'Options parameters'
        
        prob_dict = {var:val for var, val in zip(self.variables, probabilities)}
        prob_dict['Start pressure'] = prob_dict['pressures']
        prob_dict['End pressure'] = prob_dict['pressures']
        del prob_dict['pressures']
        
        prob_dict = _utils.fill_dict_with_none(prob_dict)
        prob_df = pd.DataFrame(prob_dict, index = options)
        prob_df.index.name = 'Option probabilities'
        
        value_df.to_csv(file_loc, mode = 'w', sep = ';')
        prob_df.to_csv(file_loc, mode = 'a', sep = ';')
                
class BucketEnsemble(_SubsidenceModelGas.SubsidenceModel):
    """SubsidenceModel object with additional functionality for sampling from 
    a collection of variables corresponding to a probability distribution.
    
    For each variable, a certain amount of values are presented. Each value can 
    be sampled based on a set of probabilities corresponding to each value.
    Take the example:
        density : {"Values":        [1000, 1200, 1250],
                   "Probabilities": [0.25, 0.50, 0.25]}
    The value 1000 has a 25% change of being sampled, 1200 of 50%, etc.
    Each value represents a bucket that can be picked from, in stead of 
    representing a distribution or a variable setting a distribution.
    """
    def __init__(self, name, project_folder = None):
        """Initiate the object.

        Parameters
        ----------
        name : str
            A name for the model.
        project_folder : str, optional
            A path to a directory where the project data and result will be sotred in. 
            The default is None. When None, the data and results will not be saved.

        Returns
        -------
        BucketEnsemble.

        """
        super().__init__(name, project_folder)
    
    @property
    def calc_vars(self):
        return ['compaction', 'subsidence', 'volume', 'slope', 'concavity', 'subsidence_rate']
    
    @property
    def vars_to_calculate(self):
        return ['buckets']
 
    @property
    def vars_to_build(self):
        return ['reservoirs', 'timesteps', 'points', 'observation_points',
                'dx', 'dy', 'influence_radius']
    
    @property
    def buckets(self):
        """Property: Returns a dictionary with for each reservoir a key.
        In that key is a VariableBucket object stored. In this VariableBucket object 
        variables and probabilities of those variables stored. The structure is:
            buckets[reservoir][variable]["Values"]
            buckets[reservoir][variable]["Probabilities"]
        """
        return self._buckets
    
    @property
    def sampled_parameters(self):
        """Returns a list with variables as having been used during sampling.
        The order is arbitrary as the sampling happens randomly, but corresponds the 
        the results of the calculate_sammples method.

        Returns
        -------
        list, dict
            A list of dictionaries. The dictionaries have the variables as keys
            and the items in those keys are the variables for each reservoir.

        """
        return self._sampled_parameters
    
    def set_buckets(self, buckets):
        """Set the buckets.

        Parameters
        ----------
        buckets : dict,
            Dictionary with for each reservoir a key.
            In that key is a VariableBucket object stored. In this VariableBucket object 
            variables and probabilities of those variables stored. The structure is:
                buckets[reservoir][variable]["Values"]
                buckets[reservoir][variable]["Probabilities"]
            For this model, the buckets must have the following variables:
                 'depth_to_basements',
                 'depths',
                 'thickness',
                 'compaction_coefficients',
                 'knothe_angles',
                 'tau',
                 'reference_stress_rates',
                 'cmref',
                 'b',
                 'density',
                 'shapefiles'
                 'pressures'
        """
        reservoirs = list(buckets.keys())
        
        bounds = []
        for r, bucket in buckets.items():     
            file_name = f'{r} lookup tables.csv'
            file_loc = self.project_folder.output_file(file_name)
            bucket.write_lookup_table(file_loc)
            bucket['shapes']['Values'] = _Geometries.fetch(buckets[r]['shapes']['Values'])
            bounds += [g.bounds for g in buckets[r]['shapes']['Values']] 
        biggest_bounds =  _utils.bounds_from_bounds_collection(np.array(bounds))  
        self.set_bounds(biggest_bounds)
        self._buckets = buckets
        self.number_of_reservoirs = len(self._buckets)
        self._reservoirs = reservoirs
         
    def sample_from_buckets(self, seed = None):
        """Sample values for the variables stored in the bucket, based on the probability 
        of each variable.

        Note: The difference between notation between the two output parameters.
        model_vars_dict is as a SubsidenceModel takes their data (per parameter)
        and parameter_indices is more neatly displayed.

        Returns
        -------
        model_vars_dict : dict
            A dictionary with a key for each variable. Per reservoir the variable parameters
            are stored in lists with the length of the number of reservoirs.
        parameter_indices : dict
            A dictionary with a key for each reservoir. Per reservoir the index of the 
            sampled parameters are stored in a dictionary with for each sample index
            
        """
        model_vars = []
        parameter_indices = {}
        for reservoir in self._reservoirs:
            sampled, indices = self._buckets[reservoir].sample(seed = seed)
            model_vars.append(sampled)
            parameter_indices[reservoir] = indices
        model_vars = np.array(model_vars).T
        model_vars_dict = {}
        for i, var in enumerate(_memory.MODEL_VARIABLES):
            if var != 'pressures': 
                model_vars_dict[var] =  model_vars[i] if not (np.array(model_vars[i]) == None).any() else None
            else:
                pressures = np.array([p for p in model_vars[-1]])
                if not (pressures == None).any():
                    model_vars_dict[var] = pressures
                else:
                    model_vars_dict[var] = None
        return model_vars_dict, parameter_indices
    
    def _set(self, sampled):
        for var, val in sampled.items():
            if var == 'shapes':
                self._shapes = val
            else:
                setter = getattr(self, f'set_{var}')
                setter(val)
        return
    
    def set_from_samples(self, seed = None):
        """Sample from the buckets and set the variables of the model 
        as those samples.
        """
        if not self.hasattr('sampled_parameters'):
            self._sampled_parameters = []
        sampled, indices = self.sample_from_buckets(seed = seed)
        self._set(sampled)
        
        self.set_compaction_model()
        self.mask_reservoirs()
        
        return sampled, indices
     
    def setup_writer(self, write_file, probability = True):
        self.csv_writer = csv.writer(write_file, delimiter= ';', lineterminator = '\n')
        columns = np.unique([[var for var in bucket.variables if len(bucket[var]['Values']) > 0] for bucket in self.buckets.values()])
        columns = [_memory.PARAMETER_TRANSLATOR[var] for var in columns if var != 'pressures']
        if probability:
            self._save_columns = ['Reservoir', 'Iteration'] + columns  + ['Start pressure (bar)', 'End pressure (bar)', 'Max. subsidence (m)', 'Pressure profile', 'Probability']
        else:
            self._save_columns = ['Reservoir', 'Iteration'] + columns  + ['Start pressure (bar)', 'End pressure (bar)', 'Max. subsidence (m)', 'Pressure profile']
        self.csv_writer.writerow(self._save_columns)            
    
    def store_samples(self, i, sampled, indices, max_subsidence, probability = None):
        for r in self.reservoirs:
            r_i = self.reservoir_label_to_int(r)
            store_r = {}
            for var in self._save_columns:
                if var == 'Reservoir':
                    store_r[var] = r
                elif var == 'Iteration':
                    store_r[var] = i
                elif var == 'Start pressure (bar)':
                    store_r[var] = sampled['pressures'][r_i][0]
                elif var == 'End pressure (bar)':
                    store_r[var] = sampled['pressures'][r_i][-1]
                elif var == 'Max. subsidence (m)':
                    store_r[var] = max_subsidence
                elif var == 'Probability' and probability is not None:
                    store_r[var] = probability
                elif var == 'Pressure profile':
                    store_r[var] = indices[r]['pressures'] + 1
                else:
                    translated = _memory.COLUMN_TRANSLATOR[var]
                    
                    if translated == 'shapes':
                        store_r[var] = indices[r][translated] + 1
                    else:
                        if sampled[translated] is not None:
                            store_r[var] = sampled[translated][r_i]
                        else:
                            store_r[var] = None
            
            line_to_write = list(store_r.values())        
                
        
            self.csv_writer.writerow(line_to_write)
            
    def retreive_sample(self, index):
        if self.project_folder.project_folder is None:
            sampled = self._sampled_parameters[index]
        else:
            parameter_file = self.project_folder.output_file('run_parameters.csv')
            parameters = {}
            with open(parameter_file, encoding = "utf8") as f:
                all_sampled = pd.read_csv(f, sep = ';')
            selected = all_sampled[all_sampled['Iteration'] == index]
            selected = selected.set_index('Reservoir')
            read_columns = [col for col in self._save_columns 
                            if col not in ['Reservoir', 
                                           'Iteration', 
                                           'Start pressure (bar)', 
                                           'End pressure (bar)', 
                                           'Max. subsidence (m)',
                                           'Probability']]
            for var in read_columns:
                if var == 'Pressure profile':
                    indices = {reservoir: selected['Pressure profile'][reservoir] for reservoir in self.reservoirs}
                    parameters['pressures'] = [self.buckets[reservoir]['pressures']['Values'][index - 1] for reservoir, index in indices.items()]
                elif var == 'Shapefile location':
                    indices = {reservoir: selected['Shapefile location'][reservoir] for reservoir in self.reservoirs}
                    parameters['shapes'] = [self.buckets[reservoir]['shapes']['Values'][index - 1] for reservoir, index in indices.items()]
                else:
                    translated = _memory.COLUMN_TRANSLATOR[var]
                
                    if selected.shape[0] == 1:
                        parameters[translated] = list(selected[var].values)
                    else:
                        parameters[translated] = [selected[var][r] for r in self.reservoirs]
            sampled = {key:(parameters[key] if key in parameters.keys() else None) for key in _memory.COLUMN_TRANSLATOR.values()}
            sampled['pressures'] = parameters['pressures']
        return sampled
    
    def set_from_sampled(self, index):
        """Set the model variables to values that have been sampled based on the 
        index on the list self.sampled_parameters. This can be useful to revisit
        models run with the calculate_samples method.

        Parameters
        ----------
        index : int
            Index of the list self.sampled_parameters.
        """
        sampled = self.retreive_sample(index)
            
        for var, val in sampled.items():
            if var == 'shapes':
                self._shapes = val
            else:
                setter = getattr(self, f'set_{var}')
                setter(val)
        
        self.set_compaction_model()
        self.mask_reservoirs()
        
    
    def move_through_list(self, i, l):
        amount_of_samples =  _utils.non_zero_prod(l)
        if i < amount_of_samples:
            indexed = []
            for iii in range(len(l)):
                if l[iii] == 0:
                    indexed.append(0)
                elif iii == 0:
                    indexed.append(i % l[iii])
                else:
                    forked_possibilities = _utils.non_zero_prod(l[:iii])
                    indexed.append(np.int(np.floor(i / forked_possibilities) % l[iii]))
            return indexed
            
        else:
            raise Exception(f'The sampled value {i} is not in the list with length {amount_of_samples}.')
        
    
    def get_all_deterministic_options(self): # Not used in process, just for checking
        amount_of_options = {key:
                [len(var) if len(var) != 0 else 0 for var in reservoir.values]
            for key, reservoir in self.buckets.items()}
        
        available_choises = {}
        for r, reservoir_name in enumerate(self.reservoirs):
            parameter_list = []
            for i in range(_utils.non_zero_prod(amount_of_options[reservoir_name])):
                parameter_list.append(self.move_through_list(i, amount_of_options[reservoir_name]))
            available_choises[reservoir_name] = parameter_list
    
    def sample_from_deterministic(self, i):
        values = {}
        probabilities = {}
        choises = {}
        for r, reservoir_name in enumerate(self.reservoirs):
            if self.amount_of_runs_per_reservoir[reservoir_name] == 0:
                raise Exception('Invalid amount of possibilities (0) for reservoir must be at least 1.')
            
            if r == 0:
                choises[reservoir_name] = (
                    self.move_through_list(
                        i % self.amount_of_runs_per_reservoir[reservoir_name], 
                        self.amount_of_options[reservoir_name]))
            else:
                forked_possibilities = _utils.non_zero_prod(list(self.amount_of_runs_per_reservoir.values())[:r])
                choises[reservoir_name] = (
                    self.move_through_list(
                        np.floor(i/ forked_possibilities) % self.amount_of_runs_per_reservoir[reservoir_name], 
                        self.amount_of_options[reservoir_name]))
            
            bucket = self.buckets[reservoir_name]
            probabilities[reservoir_name] = bucket.fetch(choises[reservoir_name], to_fetch = 'probabilities')
            values[reservoir_name] = bucket.fetch(choises[reservoir_name])
        
            choises[reservoir_name] = {var:choice for var, choice in zip(bucket.variables, choises[reservoir_name])}
        
        return values, probabilities, choises
            
    def set_deterministic(self, i):
        i_values, i_probabilities, choises = self.sample_from_deterministic(i)
        probabilities_array = np.array(list(i_probabilities.values()))
        probabilities_array[probabilities_array == None] = 1 
        probabilities = np.prod(probabilities_array)
        run_values = {
            var:[i_values[reservoir_name][val] for reservoir_name in self.reservoirs]
            for val, var in enumerate(_memory.MODEL_VARIABLES)
            }
        
        pruned_run_values = {}
        for key, val in run_values.items():
            pruned = None if any([v is None for v in val]) else val
            pruned_run_values[key] = pruned
        
        self._set(pruned_run_values)
        self.set_compaction_model()
        self.mask_reservoirs()
        return probabilities, pruned_run_values, choises
        
    
    
    def calculate_deterministic(self, iterations = None, error_method = 'mae', all_timesteps = False):
        # Get the amount of runs
        self.amount_of_options = {key:
                [len(var) if len(var) != 0 else 0 for var in reservoir.values]
            for key, reservoir in self.buckets.items()}
        
        self.amount_of_runs_per_reservoir = {key:
                _utils.non_zero_prod(self.amount_of_options[key])
                for key, reservoir in self.amount_of_options.items()}    
    
        total_runs = _utils.non_zero_prod(self.amount_of_runs_per_reservoir.values())
        
            
        probabilities = []
        max_results = []
        error = []
        if iterations is None:
                    
            # Setup and work in parameter tracking file
            self.parameter_file = self.project_folder.output_file('run_parameters.csv')
            with open(self.parameter_file, 'w') as _:
                pass # This creates a new file, but also empties the exisiting
            with open(self.parameter_file, 'a') as write_file:
                self.setup_writer(write_file)
                if all_timesteps:
                    with tqdm(total = total_runs, position = 0, leave = True) as progress_bar:
                        for i in tqdm(range(total_runs)):
                            probability, parameters, choises = self.set_deterministic(i)
                            probabilities.append(probability)
                            self.calculate_compaction(_print = False)
                            if self.hasattr('observation_points'):
                                self.assign_observation_parameters()
                            self.calculate_subsidence(_print = False)
                            maximum_subsidence, (x, y) = self.get_max_subsidence()
                            
                            max_results.append(maximum_subsidence)
                            if self.hasattr('observation_points'):
                                self.calculate_subsidence_at_observations(_print = False)
                                error.append(self.error(method = error_method))
                            self.store_samples(i, parameters, 
                                               choises, 
                                               maximum_subsidence,
                                               probability = probability)
                            progress_bar.update()
                else:
                    with tqdm(total = total_runs, position = 0, leave = True) as progress_bar:
                        for i in tqdm(range(total_runs)):
                            probability, parameters, choises = self.set_deterministic(i)
                            probabilities.append(probability)
                            compaction = self.calc_compaction_final()
                            subsidence = self.calc_subsidence_final(compaction)
                            maximum_subsidence = float(-subsidence.sum('reservoir').max())
                            max_results.append(maximum_subsidence)
                            self.store_samples(i, parameters, 
                                               choises, 
                                               maximum_subsidence, 
                                               probability = probability)
                            progress_bar.update()
        else:
            if not _utils.is_iterable(iterations):
                iterations = [iterations]
            for i in iterations:
                probability = self.set_deterministic(i)
                probabilities.append(probability)
                self.calculate_compaction(_print = False)
                if self.hasattr('observation_points'):
                    self.assign_observation_parameters()
                self.calculate_subsidence(_print = False)
                maximum_subsidence, _ = self.get_max_subsidence()
                max_results.append(maximum_subsidence)
                if self.hasattr('observation_points'):
                    self.calculate_subsidence_at_observations(_print = False)
                    error.append(self.error(method = error_method))
            
        return max_results, probabilities, error
    
    def calculate_samples(self, number_of_samples = 1, error_method = 'mae', all_timesteps = False,
                          seed = None):
        """Calculate 1 or more subsidence models from randomly sampled values.

        Parameters
        ----------
        number_of_samples : int, optional
            The number of models that will be run with randomly sampled 
            variables. The default is 1.
        error_method : str, optional
            The type of measurement of errors. Choose from:
                'mae': The default. Mean absolute error.
                'mse': Mean squared error.
            Returns an empty list if no observations are set or all_timesteps = False.
            Default is 'mae'. Raises an exception if not one of the above options.
        all_timesteps : boolean, optional
            If False, the results will be calculated for all timesteps. In addition 
            to collecting the maximum subsidence, the errors regarding the observations 
            are collected. Default is False, if False, only the final timesteps will 
            be realized and the errors will be returned as an empty list.
            

        Returns
        -------
        max_results : list, floats
            List with the maximum susbsidence in each model realisation.
        error : list, floats
            The error values for each realisation.

        """
        self.parameter_file = (
            'run_parameters.csv'
            if self.project_folder.project_folder is None else 
            self.project_folder.output_file('run_parameters.csv')
            )
        with open(self.parameter_file, 'w') as _:
            pass 
        with open(self.parameter_file, 'a') as write_file:
            self.setup_writer(write_file, probability=False)
            
            max_results = []
            error = []
            if all_timesteps:
                with tqdm(total = number_of_samples, position = 0, leave = True) as progress_bar:
                    for i in tqdm(range(number_of_samples)):
                        self._counter = i
                        parameters, indices = self.set_from_samples(seed = seed)
                        self.calculate_compaction(_print = False)
                        if self.hasattr('observation_points'):
                            self.assign_observation_parameters()
                        self.calculate_subsidence(_print = False)
                        maximum_subsidence, (x, y) = self.get_max_subsidence()
                        max_results.append(maximum_subsidence)
                        if self.hasattr('observation_points'):
                            self.calculate_subsidence_at_observations(_print = False)
                            error.append(self.error(method = error_method))
                        self.store_samples(i, parameters, indices, maximum_subsidence, probability=None)
                        progress_bar.update()
            else:
                with tqdm(total = number_of_samples, position = 0, leave = True) as progress_bar:
                    for i in tqdm(range(number_of_samples)):
                        self._counter = i
                        parameters, indices = self.set_from_samples(seed = seed)
                        compaction = self.calc_compaction_final()
                        subsidence = self.calc_subsidence_final(compaction)
                        maximum_subsidence = -float(subsidence.sum('reservoir').max())
                        max_results.append(maximum_subsidence)
                        self.store_samples(i, parameters, indices, maximum_subsidence, probability=None)
                        progress_bar.update()
                        
        
        return max_results, error
    
    def _mask_parameters(self, mask_array, parameter):
        return xr.where(mask_array == 1,
                        parameter, self.grid['grid_mask'])
    
    def calc_compaction_final(self):
        """Calculate the compaction for the final timestep.
        
        Returns
        -------
        compaction : np.ndarray
            The compaction at the final timestep in m³. With the shape (y, x, reservoir).
        """
        self._check_compaction_paramaters()
        TwoD = len(self._pressures.shape) == 4
        nr_of_dimensions = max([len(self[i].shape) for i in self.compaction_model.variables])
        
        if nr_of_dimensions > 1:
            TwoD = True
            # self.convert_to_grid('pressures')            
        if not TwoD:
            if not hasattr(self.grid, 'reservoir_mask'):
                raise Exception('Reservoir reservoir mask not set, run mask_reservoirs before calculating')
            
            compaction = self._compaction_model.compute(self.grid.isel(time = [0,-1])).astype(float)
            
            mask_grid = self.grid.reservoir_mask
            compaction_grid = mask_grid * compaction
        
            return compaction_grid.isel(time = -1)
        if TwoD:
            raise Exception('2D data detected, invalid input for BucketEnsemble.')    
    
    def calc_subsidence_final(self, compaction):
        """Determine the subsidence in m at the final timestep.

        Parameters
        ----------
        compaction : np.ndarray
            The compaction at the final timestep in m³. With the shape (y, x, reservoir).

        Returns
        -------
        subsidence : np.ndarray
            The subsidence at the final timestep in m. With the shape (y, x, reservoir).
        """

        kernel = _SubsidenceKernel.InfluenceKernel(self._influence_radius, self._dx)
        
        if self._subsidence_model_type.lower().startswith('nucleus'):
            kernel.nucleus(self.depths, self.depth_to_basements)
        elif self._subsidence_model_type.lower() == 'knothe':
            kernel.knothe(self.depths, self.knothe_angles)
            
        subsidence = xr.apply_ufunc(
            _utils.convolve, 
            _utils.get_chunked(compaction), 
            _utils.get_chunked(kernel.ds.uz),
            input_core_dims = [['x', 'y'],['kx', 'ky']],
            exclude_dims = set(('kx', 'ky')),
            output_core_dims = [['x', 'y']],
            vectorize = True,
            dask = 'parallelized',
            ).transpose('y', 'x', 'reservoir'
            ).compute()

        return subsidence 
    
    def calculate_from_sampled(self, index):
        """Calculate subsidence from value kept in the list sampled_parameters and
        store it in the self.subsidence property.

        Parameters
        ----------
        index : int
            Index of the list self.sampled_parameters.
        """
        self.set_from_sampled(index)
        self.calculate_compaction(_print = True)
        if self.hasattr('observation_points'):
            self.assign_observation_parameters()
            self.calculate_subsidence(_print = True)
            self.calculate_subsidence_at_observations(_print = True)
        else:
            self.calculate_subsidence(_print = True)
    
    def set_parameters(self, buckets, 
                       timesteps, 
                       dx, 
                       influence_radius, 
                       compaction_model, 
                       subsidence_model):
        """Set the variables for this type of model. NB: Cannot run calculate_variable
        methods without sampling from the bucket first (using the set_from_samples method).

        Parameters
        ----------
        buckets : dict,
            Dictionary with for each reservoir a key.
            In that key is a VariableBucket object stored. In this VariableBucket object 
            variables and probabilities of those variables stored. The structure is:
                buckets[reservoir][variable]["Values"]
                buckets[reservoir][variable]["Probabilities"]
            For this model, the buckets must have the following variables:
                'depth_to_basements',
                 'depths',
                 'thickness',
                 'compaction_coefficients',
                 'knothe_angles',
                 'tau',
                 'reference_stress_rates',
                 'cmref',
                 'b',
                 'density',
                 'shapefiles'
        timesteps : list, np.datetime64
            The timestamps of each step in time. These need to be of equal 
            step length. Per year would be ['1990', '1991', etc.]. Raises 
            Exception when None.
        dx : float/int
            Distance between grid nodes along the x-axis in m. 
            Raises exception when None.
        influence_radius : float/int
            Distance from which the subsidence is set to 0 in m.
            Raises exception when None.
        compaction_model : list, str
            Can ba a strin for the model name to be used for all reservoirs, or
            a list of string with the model type to be used for each reservoir.
            The list must have the same length as the number of reservoirs in 
            the model.
            
            The types of compaction models as defined in 
            PySub.CompactionModels for each reservoir: # TODO: keep updated with added methods
                - linear
                - time-decay
                - ratetype
            Raises Exception when not one of the above options.
        subsidence_model : str, optional
            Method of subsidence of the model. Currently available:
                - nucleus of strain, Van Opstal 1974
                - knothe, Stroka et al. 2011. 
            Raises Exception when not one of the above options.
        """
        self.set_buckets(buckets)
        self.set_dx(dx)
        self.set_timesteps(timesteps)
        self.set_influence_radius(influence_radius)
        self.build_grid()
        self.set_subsidence_model_type(subsidence_model)
        self.set_compaction_model_type(compaction_model)