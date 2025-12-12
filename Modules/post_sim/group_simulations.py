import sys
sys.path.append("../")
sys.path.append("../Modules/")
from Modules import analysis
import os
import numpy as np

def update_parameters_to_include_morphology_name(simulations_to_parameters): #@DEPRACATING
    # in the near future this will be unnecessary.
    # Warning("Deprecating update_parameters_to_include_morphology_name")
    # for sim_folder, parameters in simulations_to_parameters.items():
    #     replace = False
    #     if not hasattr(parameters, 'morphology_name'):
    #         replace=True
    #     elif getattr(parameters, 'morphology_name') == '' or getattr(parameters, 'morphology_name') == 'no_morphology_name_provided':
    #         replace=True
    #     if replace:
    #         if 'complex' in sim_folder.lower():
    #             best_guess = 'Complex'
    #         elif 'branches' in sim_folder.lower():
    #             best_guess = 'Branches'
    #         elif 'trees' in sim_folder.lower():
    #             best_guess = 'Trees'
    #         else:
    #             Warning("No morphology_name best_guess for {sim_folder}")
    #         parameters.morphology_name = best_guess
    #     else:
    #         print(Warning("Time to fully Deprecate update_parameters_to_include_morphology_name"))
    # return simulations_to_parameters
    return simulations_to_parameters

def get_parameters_for_simulations(simulations_folder, return_full_path_keys=False):
    simulations_to_parameters = {}
    for sim_folder_name in os.listdir(simulations_folder):
        full_path = os.path.join(simulations_folder, sim_folder_name)
        # load parameters
        parameters = analysis.DataReader.load_parameters(full_path)
        if return_full_path_keys:
            simulations_to_parameters[full_path] = parameters
        else:
            simulations_to_parameters[sim_folder_name] = parameters
        simulations_to_parameters = update_parameters_to_include_morphology_name(simulations_to_parameters) #@davidfague can remove this in the future.
    return simulations_to_parameters

def group_simulations_by_parameter(simulations_folder, parameter_name, return_simulations_to_parameters = False):
    simulations_to_parameters = get_parameters_for_simulations(simulations_folder)
    grouped_simulations_by_parameter = {}
    for sim_folder_name in simulations_to_parameters.keys():
        parameter_value = getattr(simulations_to_parameters[sim_folder_name], parameter_name)
        if not parameter_value in grouped_simulations_by_parameter.keys():
            grouped_simulations_by_parameter[str(parameter_value)] = []

        grouped_simulations_by_parameter[str(parameter_value)].append(sim_folder_name)
    if return_simulations_to_parameters:
        return grouped_simulations_by_parameter, simulations_to_parameters
    else:
        return grouped_simulations_by_parameter

def group_simulations_by_morphology_and_seed(simulations_folder, seed_name = 'numpy_random_state'):
    # can also use 'neuron_random_state' and updated to somehow handle both.
    simulations_grouped_by_morphology_and_seed = {}
    simulations_grouped_by_morphology, simulations_to_parameters = group_simulations_by_parameter(simulations_folder, 'morphology_name', return_simulations_to_parameters=True)
    for morphology, simulations in simulations_grouped_by_morphology.items():
        simulations_grouped_by_morphology_and_seed[morphology] = {}
        for sim_folder in simulations:
            simulation_parameters = simulations_to_parameters[sim_folder]
            seed = getattr(simulation_parameters, seed_name)
            simulations_grouped_by_morphology_and_seed[morphology][seed] = sim_folder
