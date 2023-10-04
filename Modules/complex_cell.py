from neuron import h
import pickle
import pickletools

#################### original loading hay et al model from templates. ###

def build_L5_cell(cell_folder, biophys = 'L5PCbiophys3.hoc', morph = 'cell1.asc', template = 'L5PCtemplate.hoc'):
    '''
    original for building hay et al model & our updates
    '''
    # Load biophysics
    h.load_file(cell_folder + biophys)

    # Load morphology
    h.load_file("import3d.hoc")

    # Load builder
    h.load_file(cell_folder + template)

    # Build complex_cell object
    cell = eval("h." + template.split('.')[0] + '(cell_folder + morph)')
    #complex_cell = h.L5PCtemplate(cell_folder + morph)

    return cell


    
def build_L5_cell_ziao(cell_folder, template='ziao_templates.hoc'):
    '''
    Function for loading reduced Neymotin model from template that Ziao made
    '''
    h.load_file(cell_folder + template)
    cell = h.CP_Cell()
    
    return cell

#### LATEST FUNCTIONS TO LOAD THE LATEST NEYMOTIN MODEL FROM TEMPLATE AND PICKLE ####################

def build_cell_reports_cell(params, cell_folder='../complex_cells/L5PC/M1_CellReports_2023/', template = 'PTcell.hoc'):
  '''
  Builds one of the latest cells from Neymotin group
  '''
  h.load_file(cell_folder + template)
  cell = h.PTcell(3,3,3)
  
  return cell
  

def unpickle_params(cell_folder = '../complex_cells/L5PC/M1_CellReports_2023/', filename = 'PT5B_full_cellParams.pkl'):
  '''
  standard unpickling
  '''
  path = cell_folder + filename
  with open(path, 'rb') as f:
      data = pickle.load(f,encoding='latin1')

  #print(data)
  return data
  
def inspect_pickle(cell_folder = '../complex_cells/L5PC/M1_CellReports_2023/', filename = 'PT5B_full_cellParams.pkl'):
  '''
  uses pickletools to inspect
  '''
  path = cell_folder + filename
  with open(path, 'rb') as f:
      pickletools.dis(f)
      
def assign_parameters_to_section(sec, section_data):
    '''
    assigns parameters to section/segments from data that was stored in pickle
    '''
    # List of common state variables
    state_variables = []#'o_na', 'o_k', 'o_ca', 'm', 'h', 'n', 'i_na', ]

    # Set geometry parameters
    geom = section_data.get('geom', {})
    sec.diam = geom.get('diam', sec.diam)
    sec.L = geom.get('L', sec.L)
    sec.nseg = geom.get('nseg', sec.nseg)
    sec.Ra = geom.get('Ra', sec.Ra)
    sec.cm = geom.get('cm', sec.cm)
    
    # Set ion parameters
    try:
        ions = section_data.get('ions', {})
        for ion, params in ions.items():
            for param, value in params.items():
                if param not in state_variables:
                    main_attr_name = f"{ion}_ion"
                    
                    # Check if parameter ends with 'o', then reverse the naming
                    if param[-1] == 'o':
                        sub_attr_name = f"{ion}{param}"
                    else:
                        sub_attr_name = f"{param}{ion}"
                    try:
                        for seg in sec:
                            ion_obj = getattr(seg, main_attr_name)
                            setattr(ion_obj, sub_attr_name, value)
                    except AttributeError as e:
                        print(f"AttributeError in {sec.name()}: {str(e)}")
                    except ValueError as e:
                        print(f"ValueError in {sec.name()}: {str(e)}")
    except Exception as e:
        print(f"Unhandled error in setting ion params {sec.name()}: {str(e)}")
    
    # Set mechanism parameters
    #try:
    mechs = section_data.get('mechs', {})
    for mech, params in mechs.items():
        #try:
        sec.insert(mech)
            #try:
        for param, value in params.items():
            if param not in state_variables:
                for i,seg in enumerate(sec):
                  if str(type(value)) == "<class 'list'>":
                    try:
                      setattr(seg, f"{param}_{mech}", value[i])
                              #setattr(seg, f"{mech}.{param}", value)
                    except Exception as e:
                      print(f"Warning: Issue setting {mech} {param} in {sec.name()} to {value[i]}. {e} | value type: {type(value[i])} | nseg: {sec.nseg}; len(value): {len(value)}")
                  else:
                    try:
                      setattr(seg, f"{param}_{mech}", value)
                              #setattr(seg, f"{mech}.{param}", value)
                    except Exception as e:
                      print(f"Warning: Issue setting {mech} {param} in {sec.name()} to {value}. {e} value type {type(value)}")
        #except AttributeError:
            #    print(f"Warning: Issue with inserting mechanism {mech} in {sec.name()}.")
    #except Exception as e:
    #    print(f"Unhandled error in setting mechanism params {sec.name()}: {str(e)} {type(e)}")
       
    #TODO add these
    #pt3d = section_data.get('pt3d', {})
    #topol = section_data.get('topol', {}) #{childX: 0.0, parentSec: 'apic_0', parentX: 1.0}
    
def create_cell_from_template_and_pickle():
    '''
    Main Function that uses the above functions related to the latest Neymotin detailed model from Cell Reports 2023
    '''
    # create cell
    complex_cell = build_cell_reports_cell(1.0)
    # get params from pickle
    params = unpickle_params()
    # assign params
    if len(list(complex_cell.all)) != len(params['secs'].keys()):
      print(f"Warning: len(list(complex_cell.all)) != len(params['secs'].keys()): {len(list(complex_cell.all))} {len(params['secs'].keys())}")
    for sec in complex_cell.all:
        cell_section_name = sec.name()
        section_name = sec.name().split(".")[1]  # Remove Cell from name
    
        if "[" in section_name:
            section_type, section_type_index = section_name.split("[")
            section_type_index = section_type_index.strip("]")
            
            # Concatenate with "_"
            section_name_as_stored_in_pickle = f"{section_type}_{section_type_index}"
    
        else:
            section_name_as_stored_in_pickle = section_name  # For sections like soma and axon
    
        if section_name_as_stored_in_pickle in params['secs']:
            #try:
            assign_parameters_to_section(sec, params['secs'][section_name_as_stored_in_pickle])
            #except Exception as e:
            #    print(f"Error in assigning parameters to {section_name_as_stored_in_pickle}: {e}. Parameters: {params['secs'][section_name_as_stored_in_pickle]}")
        else:
            print(f"Warning: No parameters found for {section_name_as_stored_in_pickle}.")
            
    #inspect_pickle()
    # May need to update to use more of the unpickled params
    return complex_cell
      

#################################################################################################################

# functions to make adjustments when we were trying to use the older Neymotin Detailed Soma and axon with our dendrites.
def set_hoc_params():
    h.a0n_kdr = 0.0075
    h.nmax_kdr = 20
    h.nmin_kap = 0.4
    h.lmin_kap = 5
    h.erev_ih = 37.0
    
# old is from Ziao model or BS02.., current is from BS0489
# old axonDiam=1.40966286462, axonL=594.292937602, axon_L_scale=1, somaL=48.4123467666, somaDiam=28.2149102762
def adjust_soma_and_axon_geometry(cell, axonDiam=1.0198477329563544, axonL=549.528226526987, axon_L_scale=1, somaL=28.896601873591436, somaDiam=14.187950175330796):
    '''
    Function for updating soma and axon to M1 model.
    '''
    #ZC uses axon_L_scale = 0.1 for better LFP?
    diam = somaDiam
    orig_soma_diam = cell.soma[0].diam
    orig_soma_L = cell.soma[0].L
    orig_axon_diam = cell.axon[0].diam
    orig_axon_L = cell.axon[0].L
    cell.soma[0].pt3dclear()
    cell.soma[0].pt3dadd(0,0,-somaL/2,diam)
    cell.soma[0].pt3dadd(0,0,somaL/2,diam)
    diam = axonDiam/axon_L_scale
    cell.axon[0].pt3dclear()
    cell.axon[0].pt3dadd(0,0,0,diam)
    cell.axon[0].pt3dadd(0,0,-axonL*axon_L_scale,diam)
    if cell.axon[0].L != orig_axon_L:
      print('axon L updated from',orig_axon_L,'to',cell.axon[0].L)
    if cell.axon[0].diam != orig_axon_diam:
      print('axon diam updated from',orig_axon_diam,'to',cell.axon[0].diam)
    if cell.soma[0].L != orig_soma_L:
      print('soma L updated from',orig_soma_L,'to',cell.soma[0].L)
    if cell.axon[0].diam != orig_soma_diam:
      print('soma diam updated from',orig_soma_diam,'to',cell.soma[0].diam)
