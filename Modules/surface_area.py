from adjacency import get_all_descendants
import numpy as np

def record_model_SA_and_L(cell_model):
  SA_df = {}
  L_df = {}
  for model_part in ['all','soma','dend','apic','axon']:
    sec_list = getattr(cell_model, model_part)
    SA_df[model_part] = get_SA_for_sec_list(sec_list)
    L_df[model_part] = get_L_for_sec_list(sec_list)
    return SA_df, L_df
  
def calc_SA_and_L_beyond_seg(cell_model, seg_index, adjacency_matrix):
  seg_indices = get_all_descendants(adjacency_matrix=adjacency_matrix, start_segment=seg_index)
  all_segs, _ = cell_model.get_segments(["all"])
  segments = [all_segs[i] for i in seg_indices]
  return get_SA_for_seg_list(segments), get_L_for_seg_list(segments)

def get_L_for_sec_list(list_of_obj):
  L=0
  for obj in list_of_obj:
    L += obj.L
  return L
  
def get_L_for_seg_list(list_of_obj):
  L=0
  for obj in list_of_obj:
    L += obj.sec.L/obj.sec.nseg
  return L

def get_SA_for_seg_list(list_of_obj):
  SA=0
  for obj in list_of_obj:
    SA += calc_SA_of_seg(obj)
  return SA

def get_SA_for_sec_list(list_of_obj):
  SA=0
  for obj in list_of_obj:
    SA += calc_SA_of_sec(obj)
  return SA

def calc_SA_of_sec(obj): # obj can be section or segment
  L = obj.L
  R = obj.diam/2 # need to check if diameter is the same along segments or if this will be the average value
  SA, _, _ = calculate_cylinder_SA(L,R) # wrap SA is the only one relevant to NEURON
  return SA
  
def calc_SA_of_seg(obj): # obj can be section or segment
  L = obj.sec.L/obj.sec.nseg
  R = obj.diam/2 # need to check if diameter is the same along segments or if this will be the average value
  SA, _, _ = calculate_cylinder_SA(L,R) # wrap SA is the only one relevant to NEURON
  return SA

def calculate_cylinder_SA(L,R):
  '''
  calculate surface area of a cylinder : 2prh+2pr2
  '''
  wrap_SA = 2*np.pi*R*L
  circles_SA = 2*np.pi*R*2 #may not want to include circle_SA
  total_SA= wrap_SA + circles_SA
  return wrap_SA, circles_SA, total_SA