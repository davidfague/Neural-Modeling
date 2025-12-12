'''
Immediate Use: Get Ra, Rm, Cm from apic, dend, and tufts. You can check the below file for these as well since they should be uniform, Note Rm = 1/g_pas.
"/Neural-Modeling/cells/templates/L5PCbiophys3.hoc"

Specify a length and voltage transfer ratio (VTR) for the dendrite that you want.

Input these parameters in to get a diameter. (output will be a big approximation, and we will have to check the VTR)

Use VTR of 0.1 for apical and then do 0.1 for tufts (resulting in 0.01)

Steady-state voltage transfer ratio can be computed the way that we compute 'electrotonic distance' in CellModel.

For basal dendrites, record the distal VTR in Detailed and approximate length and use that?

Then you have all the parameters you need. Disconnect and delete the old sections and place new sections.

example use:
analyze_parameters(Rm={'value': 1.5}, Cm={'range': (0.1, 0.5)}, Ra={'value': 200}, L={}, Diameter={}, lambda={})

equations:
attenuation constant: alpha = 1/lambda
voltage transfer ratio VTR = exp(-alpha * length) (infinite, unsealed cable)
length = -lambda * ln(VTR)
0.1 is approximately 2.30 units (# of length constants)
0.01 is approximately 4.61 units (# of length constants)

length constant in Neuron
Rm = 1/sec.g_pas
rm = Rm / (pi * diam)
Cm = sec.Cm
Ra = sec.Ra
ra = (4 * Ra) / (pi * d^2)
lambda = sqrt(rm / ra)
solve for diameter: lambda = sqrt([Rm / (pi * diam)] / [(4 * Ra) / (pi * diam^2)])

lambda^2 * [(4 * Ra) ] / Rm = diam







solve for length = 2.30 * lambda

algorithm: choose length, VTR, Rm, Cm, Ra. Find Diameter 

Started to implement parameter ranges, but do not use them for now.

Alternatively, Ziao's stylized may be good for generating stylized geometry. With it, you provide a geometry.csv to build the cell.
'''

import numpy as np
class Parameter:
    def __init__(self, value=None, range=None):
        self.value = value
        self.range = range
    
    @property
    def is_fixed(self):
        return self.value is not None
    
    @property
    def is_range(self):
        return self.range is not None
    
    @property
    def is_unspecified(self):
        return self.value is None and self.range is None

def handle_parameters(**params):
    parameters = {name: Parameter(**value) for name, value in params.items()}
    unspecified = [name for name, param in parameters.items() if param.is_unspecified]
    if len(unspecified) > 1:
        raise ValueError(f"Model is underdefined. Unspecified parameters: {', '.join(unspecified)}")
    return parameters

def compute_unknown(parameters):
    # Identifying the unspecified parameter
    unspecified_params = [name for name, param in parameters.items() if param.is_unspecified]
    if len(unspecified_params) > 1:
        raise ValueError("Error: More than one parameter is unspecified. Cannot determine the system uniquely.")
    if len(unspecified_params) == 0:
        raise ValueError("Error: No parameter is unspecified. System is fully defined.")
    unspecified_param = unspecified_params[0]

    # Compute the value or range for the unspecified parameter
    if unspecified_param == 'Rm':
        calc_Rm(parameters)
    elif unspecified_param == 'Ra':
        calc_Ra(parameters)
    elif unspecified_param == 'Cm':
        calc_Cm(parameters)
    elif unspecified_param == 'diam':
        calc_diam(parameters)
    elif unspecified_param == 'L':
        calc_L(parameters)
    elif unspecified_param == 'VTR':
        calc_VTR(parameters)
    else:
        raise ValueError(f"Error: Unspecified parameter '{unspecified_param}' is not recognized.")
    
    # Finally, check and output the result for the unspecified parameter
    unspecified_param_obj = parameters[unspecified_param]
    print(f"unspecified_param_obj: {unspecified_param_obj.value}")
    if unspecified_param_obj.is_fixed:
        print(f"{unspecified_param}: Fixed Value = {unspecified_param_obj.value}")
    elif unspecified_param_obj.is_range:
        print(f"{unspecified_param}: Range = {unspecified_param_obj.range}")
    else:
        print(f"{unspecified_param}: Calculation did not specify a value or range.")


    
def analyze_parameters(**params):
    try:
        parameters = handle_parameters(**params)
        compute_unknown(parameters)
        
        for name, param in parameters.items():
            if param.is_fixed:
                print(f"{name}: Fixed Value = {param.value}")
            elif param.is_range:
                print(f"{name}: Range = {param.range}")
            else:
                print(f"{name}: Computation unsuccessful or parameter was not properly defined.")

    except ValueError as e:
        print(f"Error: {e}")

def calc_Rm(parameters):
  '''calculate membrane resistance from the other specifications'''
  raise(ImplementationError(f"calc_Rm() not implemented."))
  
def calc_Ra(parameters):
  '''calculate axial resistance (internal resistance) from the other specifications'''
  raise(ImplementationError(f"calc_Ra() not implemented."))
  
def calc_Cm(parameters):
  '''calculate membrane capacitance from the other specifications'''
  raise(ImplementationError(f"calc_Cm() not implemented."))
  
def calc_diam(parameters):
    '''calculate diameter from the other specifications'''
    Ra = parameters['Ra']
    Rm = parameters['Rm']
    L = parameters['L'].value  # Assuming L is always specified as a fixed value for simplicity
    VTR = parameters['VTR'].value  # Assuming VTR is always a fixed value

    if Ra.is_range or Rm.is_range:
        # Calculate min and max diameter based on ranges of Ra and Rm
        min_diam = calc_diameter_min_max(Ra.range[0], Rm.range[1], L, VTR)  # Use min Ra and max Rm for min diameter
        max_diam = calc_diameter_min_max(Ra.range[1], Rm.range[0], L, VTR)  # Use max Ra and min Rm for max diameter
        return (min_diam, max_diam)
    else:
        # Calculate diameter with fixed values
        d = calc_diameter_min_max(Ra.value, Rm.value, L, VTR)
        print(f"DIAMETER: {d*10000}")
        return d

def calc_diameter_min_max(Ra_val, Rm_val, L, VTR):
    # This function calculates the diameter given fixed values of Ra and Rm
    length_constant = -L / np.log(VTR)  # Corrected np.ln to np.log
    diameter = (4 * Ra_val * (length_constant ** 2)) / Rm_val
    return diameter

def calc_L(parameters):
  '''calculate length from the other specifications'''
  raise(ImplementationError(f"calc_length() not implemented."))

def calc_VTR(parameters):
  '''calculate voltage transfer ratio from the other specifications'''
  raise(ImplementationError(f"calc_VTR() not implemented."))
  
if __name__ == "__main__":
    params = {
        'Ra': {'value': 100},  # Example parameter specification
        'Rm': {'value': 1/0.0000338},
        'Cm': {'value': 2},
        'L': {'value': 650/10000},
        'VTR': {'value': 0.1},
        'diam': {}
    }
    analyze_parameters(**params)
    
# Ra = [ohm cm]
# Rm = [ohm cm2] (distributed)
# diam, L =	[um]
# cm =	[uf/cm2]
# lambda = [cm]
# diam = ((ohm cm) * cm^2) / (ohm cm2)
# cm