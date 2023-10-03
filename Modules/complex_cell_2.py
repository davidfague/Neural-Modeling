from neuron import h
import pickle
import pickletools
import netpyne
    
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
