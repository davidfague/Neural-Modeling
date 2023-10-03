from neuron import h
import pickle
import pickletools

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
    h.load_file(cell_folder + template)
    cell = h.CP_Cell()
    
    return cell
    
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
