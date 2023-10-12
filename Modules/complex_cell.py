from neuron import h
import pickle
import pickletools
import os

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
      
import os
import csv
import glob

def assign_parameters_to_section(sec, section_data, indicate_soma_and_axon_updates, decrease_axon_Ra_with_update = False):
    '''
    assigns parameters to section/segments from data that was stored in pickle
    #TODO add these
    # set 3d coordinates pt3d clear
    #pt3d = section_data.get('pt3d', {})
    # pt3d_data = geom.get('pt3d', {})
    # set section attributes: 'parentseg', 'psection', 'pt3dadd', 'pt3dchange', 'pt3dclear', 'pt3dinsert' 'x3d', 'y3d', 'z3d']
    #topol = section_data.get('topol', {}) #{childX: 0.0, parentSec: 'apic_0', parentX: 1.0}
    '''
    # List of common state variables
    state_variables = []  # e.g. 'o_na', 'o_k', 'o_ca', 'm', 'h', 'n', 'i_na', ...

    # List to hold the rows for the CSV
    rows = []
    
    # Initialize a dictionary for the section
    section_row = {'Section': sec.name()}
    
    # List to hold the rows for the CSV
    rows = []
    
    # Initialize a dictionary for the section
    section_row = {'Section': sec.name()}
    
    # Set and record geometry parameters
    geom = section_data.get('geom', {})
    for param, value in geom.items():
        if str(param) not in ['pt3d']:
            if (decrease_axon_Ra_with_update) and (str(param) == 'Ra') and ('axon' in sec.name()):
              value = value/2
              print(f"Axon Ra halved")
            setattr(sec, param, value)
            section_row[f"geom.{param}"] = value
            if (('soma' in sec.name()) or ('axon' in sec.name()) or ('apic[0]' in sec.name()) or ('dend[0]' in sec.name())) and (indicate_soma_and_axon_updates): # sanity check
                print(f"Setting {sec.name()} {param} to {value}.")
    
    # Set and record ion parameters
    try:
        ions = section_data.get('ions', {})
        for ion, params in ions.items():
            for param, value in params.items():
                if param not in state_variables:
                    main_attr_name = f"{ion}_ion"
                    if param[-1] == 'o':
                        sub_attr_name = f"{ion}{param}"
                    else:
                        sub_attr_name = f"{param}{ion}"
                    try:
                        for seg in sec:
                            ion_obj = getattr(seg, main_attr_name)
                            setattr(ion_obj, sub_attr_name, value)
                        if (('soma' in sec.name()) or ('axon' in sec.name()) or ('apic[0]' in sec.name()) or ('dend[0]' in sec.name())) and (indicate_soma_and_axon_updates): # sanity check
                            print(f"Setting {sec.name()} {ion_obj} {sub_attr_name}to {value}.")
                    except AttributeError as e:
                        print(f"AttributeError in {sec.name()}: {str(e)}")
                    except ValueError as e:
                        print(f"ValueError in {sec.name()}: {str(e)}")
                    
                    section_row[f"ions.{ion}.{param}"] = value
    except Exception as e:
        print(f"Unhandled error in setting ion params {sec.name()}: {str(e)}")
    
    # Set and record mechanism parameters
    mechs = section_data.get('mechs', {})
    for mech, params in mechs.items():
        sec.insert(mech)
        for param, value in params.items():
            if param not in state_variables:
                if (decrease_axon_Ra_with_update) and (str(mech) == 'pas') and ('soma' in sec.name()) and (str(param) == 'g'):
                    value = value/2
                    print(f"Soma g_pas halved")
                if (('soma' in sec.name()) or ('axon' in sec.name()) or ('apic[0]' in sec.name()) or ('dend[0]' in sec.name())) and (indicate_soma_and_axon_updates): # sanity check
                    print(f"Setting {sec.name()} {mech} {param} to {value}.")
                for i, seg in enumerate(sec):
                    if isinstance(value, list):
                        try:
                            setattr(seg, f"{param}_{mech}", value[i])
                        except Exception as e:
                            print(f"Warning: Issue setting {mech} {param} in {seg} to {value[i]}. {e} | value type: {type(value[i])} | nseg: {sec.nseg}; len(value): {len(value)}")
                    else:
                        try:
                            setattr(seg, f"{param}_{mech}", value)
                        except Exception as e:
                            print(f"Warning: Issue setting {mech} {param} in {sec.name()} to {value}. {e} value type {type(value)}")
    
                section_row[f"mechs.{mech}.{param}"] = value
                
    #except AttributeError:
    #    print(f"Warning: Issue with inserting mechanism {mech} in {sec.name()}.")
    #except Exception as e:
    #    print(f"Unhandled error in setting mechanism params {sec.name()}: {str(e)} {type(e)}")
       
    
    # Name the file based on the current process ID to avoid conflicts
    filename = f'assignments_{os.getpid()}.csv'
    
    # Check if the file exists to determine if we should write headers
    file_exists = os.path.isfile(filename)
    
    # Append the section dictionary to the rows list
    rows.append(section_row)
    
    # Write the rows to the CSV
    keys = rows[0].keys()  # get the headers (parameter names)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

def create_cell_from_template_and_pickle():
    '''
    Main Function that uses the above functions related to the latest Neymotin detailed model from Cell Reports 2023
    '''
    # ... (the rest of the cell creation remains the same)

    # After all parameters are set, merge individual CSVs
    merge_csv_files("assignments.csv")

    return complex_cell

def merge_csv_files(output_filename):
    # Find all CSV files that start with "assignments_"
    all_files = glob.glob("assignments_*.csv")

    with open(output_filename, 'w', newline='') as outfile:
        for i, filename in enumerate(all_files):
            with open(filename, 'r') as infile:
                # Write header only for the first file
                if i == 0:
                    outfile.write(infile.readline())
                else:
                    infile.readline()  # skip header line
                
                # Copy the rest of the lines
                outfile.writelines(infile.readlines())

            # Optionally, remove the individual CSV after merging
            os.remove(filename)

      
    
def create_cell_from_template_and_pickle(indicate_soma_and_axon_updates):
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
            assign_parameters_to_section(sec, params['secs'][section_name_as_stored_in_pickle], indicate_soma_and_axon_updates)
            #except Exception as e:
            #    print(f"Error in assigning parameters to {section_name_as_stored_in_pickle}: {e}. Parameters: {params['secs'][section_name_as_stored_in_pickle]}")
        else:
            print(f"Warning: No parameters found for {section_name_as_stored_in_pickle}.")
            
    #inspect_pickle()
    # May need to update to use more of the unpickled params
    return complex_cell
    
def set_pickled_parameters_to_sections(sections, indicate_soma_and_axon_updates, decrease_axon_Ra_with_update):
    params = unpickle_params()
    for sec in sections:
        cell_section_name = sec.name()
        section_name = sec.name().split(".")[1]  # Remove Cell from name

        if "[" in section_name:
            section_type, section_type_index = section_name.split("[")
            section_type_index = section_type_index.strip("]")
            
            # Concatenate with "_"
            section_name_as_stored_in_pickle = f"{section_type}"#_{section_type_index}"
    
        else:
            section_name_as_stored_in_pickle = section_name  # For sections like soma and axon
    
        if section_name_as_stored_in_pickle in params['secs']:
            assign_parameters_to_section(sec, params['secs'][section_name_as_stored_in_pickle], indicate_soma_and_axon_updates, decrease_axon_Ra_with_update)
        else:
            raise ValueError(f"No parameters found for {section_name_as_stored_in_pickle}.")

      

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

import pandas as pd

def is_indexable(obj):
    """Check if the object is indexable."""
    try:
        _ = obj[0]
        return True
    except (TypeError, IndexError):
        return False

def assign_parameters_from_csv(cell, filename='assignments_1593562.csv'):
    df = pd.read_csv(filename)

    for index, row in df.iterrows():
        section_name = row['Section']

        if "soma" in section_name:
            section = cell.soma[0] if is_indexable(cell.soma) else cell.soma
        elif "axon" in section_name:
            section = cell.axon[0] if is_indexable(cell.axon) else cell.axon
        else:
            continue

        for col, value in row.items():
            if col == "Section":
                continue

            categories = col.split('.')
            
            if len(categories) < 2:
                continue

            category, attr_name = categories[:2]
            param = categories[-1]
            
            if category == "geom":
                setattr(section, attr_name, value)
            elif category == "ions":
                ion_name = categories[1]
                if attr_name in ["i", "e", "o"]:
                    actual_attr_name = f"{ion_name}{attr_name}"
                    setattr(section, actual_attr_name, value)
            elif category == "mechs":
                mechanism_name = categories[1]
                actual_attr_name = f"{mechanism_name}_{param}"  # e.g., "kBK_tau"

                # Before attempting to set, check if the attribute exists using getattr
                for seg in section:
                    try:
                        _ = getattr(seg, actual_attr_name)  # This will throw an error if attribute doesn't exist
                        setattr(seg, actual_attr_name, value)
                    except AttributeError as e:
                        print(f"Attribute {actual_attr_name} not found in {section_name} with mechanism {mechanism_name}.")
                        continue  # Optionally raise the error if needed


# OLD
#def assign_parameters_to_section(sec, section_data):
#    '''
#    assigns parameters to section/segments from data that was stored in pickle
#    '''
#    # List of common state variables
#    state_variables = []#'o_na', 'o_k', 'o_ca', 'm', 'h', 'n', 'i_na', ]
#
#    import csv
#    
#    # List to hold the rows for the CSV
#    rows = []
#    
#    # Initialize a dictionary for the section
#    section_row = {'Section': sec.name()}
#    
#    # Set and record geometry parameters
#    geom = section_data.get('geom', {})
#    for param, value in geom.items():
#        if str(param) not in ['pt3d']:
#            setattr(sec, param, value)
#            section_row[f"geom.{param}"] = value
#    
#    # Set and record ion parameters
#    try:
#        ions = section_data.get('ions', {})
#        for ion, params in ions.items():
#            for param, value in params.items():
#                if param not in state_variables:
#                    main_attr_name = f"{ion}_ion"
#                    
#                    if param[-1] == 'o':
#                        sub_attr_name = f"{ion}{param}"
#                    else:
#                        sub_attr_name = f"{param}{ion}"
#    
#                    try:
#                        for seg in sec:
#                            ion_obj = getattr(seg, main_attr_name)
#                            setattr(ion_obj, sub_attr_name, value)
#                    except AttributeError as e:
#                        print(f"AttributeError in {sec.name()}: {str(e)}")
#                    except ValueError as e:
#                        print(f"ValueError in {sec.name()}: {str(e)}")
#                    
#                    section_row[f"ions.{ion}.{param}"] = value
#    except Exception as e:
#        print(f"Unhandled error in setting ion params {sec.name()}: {str(e)}")
#    
#    # Set and record mechanism parameters
#    mechs = section_data.get('mechs', {})
#    for mech, params in mechs.items():
#        sec.insert(mech)
#        for param, value in params.items():
#            if param not in state_variables:
#                for i, seg in enumerate(sec):
#                    if isinstance(value, list):
#                        try:
#                            setattr(seg, f"{param}_{mech}", value[i])
#                        except Exception as e:
#                            print(f"Warning: Issue setting {mech} {param} in {sec.name()} to {value[i]}. {e} | value type: {type(value[i])} | nseg: {sec.nseg}; len(value): {len(value)}")
#                    else:
#                        try:
#                            setattr(seg, f"{param}_{mech}", value)
#                        except Exception as e:
#                            print(f"Warning: Issue setting {mech} {param} in {sec.name()} to {value}. {e} value type {type(value)}")
#    
#                section_row[f"mechs.{mech}.{param}"] = value
#    
#    # Check if the file exists to determine if we should write headers
#    file_exists = os.path.isfile('assignments.csv')
#    
#    # Append the section dictionary to the rows list
#    rows.append(section_row)
#    
#    # Write the rows to a CSV
#    keys = rows[0].keys()  # get the headers (parameter names)
#    # Append the data to the CSV
#    with open('assignments.csv', 'a', newline='') as csvfile:
#        writer = csv.DictWriter(csvfile, fieldnames=keys)
#        if not file_exists:
#            writer.writeheader()
#    writer.writerows(rows)
#
#
#        #except AttributeError:
#            #    print(f"Warning: Issue with inserting mechanism {mech} in {sec.name()}.")
#    #except Exception as e:
#    #    print(f"Unhandled error in setting mechanism params {sec.name()}: {str(e)} {type(e)}")
#       
#    #TODO add these
#    #pt3d = section_data.get('pt3d', {})
#    # pt3d_data = geom.get('pt3d', {})
#    # set section attributes: 'parentseg', 'psection', 'pt3dadd', 'pt3dchange', 'pt3dclear', 'pt3dinsert' 'x3d', 'y3d', 'z3d']
#    #topol = section_data.get('topol', {}) #{childX: 0.0, parentSec: 'apic_0', parentX: 1.0}