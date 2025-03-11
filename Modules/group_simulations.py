import sys
sys.path.append("../")
sys.path.append("../Modules/")
from Modules import analysis
import os

def update_parameters_to_include_morphology_name(simulations_to_parameters): #@DEPRACATING
    # in the near future this will be unnecessary.
    # Warning("Deprecating update_parameters_to_include_morphology_name")
    for sim_folder, parameters in simulations_to_parameters.items():
        replace = False
        if not hasattr(parameters, 'morphology_name'):
            replace=True
        elif getattr(parameters, 'morphology_name') == '':
            replace=True
        if replace:
            if 'complex' in sim_folder.lower():
                best_guess = 'Complex'
            elif 'branches' in sim_folder.lower():
                best_guess = 'Branches'
            elif 'trees' in sim_folder.lower():
                best_guess = 'Trees'
            else:
                Warning("No morphology_name best_guess for {sim_folder}")
            parameters.morphology_name = best_guess
        else:
            print(Warning("Time to fully Deprecate update_parameters_to_include_morphology_name"))
    return simulations_to_parameters

def get_parameters_for_simulations(simulations_folder):
    simulations_to_parameters = {}
    for sim_folder_name in os.listdir(simulations_folder):
        full_path = os.path.join(simulations_folder, sim_folder_name)
        # load parameters
        parameters = analysis.DataReader.load_parameters(full_path)
        simulations_to_parameters[sim_folder_name] = parameters
    return simulations_to_parameters

def group_simulations_by_parameter(simulations_folder, parameter_name):
    simulations_to_parameters = get_parameters_for_simulations(simulations_folder)
    # print(f"simulations_to_parameters: {simulations_to_parameters}")
    simulations_to_parameters = update_parameters_to_include_morphology_name(simulations_to_parameters) #@MARK can remove this in the near future.
    # print(f"simulations_to_parameters: {simulations_to_parameters}")
    grouped_simulations_by_parameter = {}
    for sim_folder_name in simulations_to_parameters.keys():
        parameter_value = getattr(simulations_to_parameters[sim_folder_name], parameter_name)
        if not hasattr(grouped_simulations_by_parameter, parameter_value):
            grouped_simulations_by_parameter[str(parameter_value)] = []
        grouped_simulations_by_parameter[str(parameter_value)].append(sim_folder_name)
    return grouped_simulations_by_parameter