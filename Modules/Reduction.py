from neuron_reduce import subtree_reductor
from Modules.cable_expander_func import cable_expander

class Reductor():
  def __init__(cell, method=str, synapses_list=None, netcons_list=None, reduction_frequency=0, sections_to_expand=None, furcations_x=None, nbranches=None, return_seg_to_seg: bool = False):
    '''
    cell: hoc model cell object (TO TEST: providing python cell_model object instead)
    method: str for method to use ex. 'expand cable', 'neuron_reduce'
    synapses_list: list of synapse objects
    netcons_list: list of netcon objects
    reduction_frequency: frequency used in calculated transfer impedance
    sections_to_expand: list of sections for 'cable expand' method
    furcations_x: list of x (0 to 1) locs corresponding to sections_to_expand to bifurcate section at
    nbranches: list of integar number of branches corresponding to sections_to_expand to choose the number of new branches at furcation_x
    return_seg_to_seg: bool for returning a dictionary mapping original segments to reduced.
    '''
    if method == 'expand cable': # sanity checks
      if sections_to_expand:
        if furcations_x:
          if nbranches:
            reduced_cell, synapses_list, netcons_list, txt = cable_expander(cell, sections_to_expand, furcations_x, nbranches,
                                                                              synapses_list, netcons_list, reduction_frequency=reduction_frequency,return_seg_to_seg=True)
          else:
            raise ValueError('Must specify nbranches list for cable_expander()')
        else:
          raise ValueError('Must specify furcations_x list for cable_expander()')
      else:
        raise ValueError('Must specify sections_to_expand for cable_expander()')
    elif method == 'neuron_reduce':
      reduced_cell, synapses_list, netcons_list, txt = subtree_reductor(cell, synapses_list, netcons_list, reduction_frequency=reduction_frequency,return_seg_to_seg=True)
    else:
      raise ValueError(f"Method '{method}' not implemented.")

    if return_seg_to_seg:
      return reduced_cell, synapses_list, netcons_list, txt
    else:
      return reduced_cell, synapses_list, netcons_list

  def get_other_seg_from_seg_to_seg(segments, seg=str,seg_to_seg=dict):
    '''
    WORK IN PROGRESS
    segments: list of segments to search through
    seg: str or nrn.segment for which you wish to return the mapped segments
    works with find_seg_to_seg, get_str_from_dict, and get_seg_from_str to return segment objects from seg_to_seg
    '''
    seg_mapping = find_seg_to_seg(seg, seg_to_seg)
    # seg_to_find=not_original_seg
    seg_to_find_str = get_str_from_dict(seg_mapping, seg_to_find)
    seg = get_seg_from_str(segments, seg_to_find_str)
    return seg
    
  
  def find_seg_to_seg(seg=str, seg_to_seg=dict):
    '''
    TO DO:
    Finds and returns a str of the given segment's mapping where the segment could be either a key (original model seg) or item (reduced model seg)
    in the case where the new seg is an item from a cable_expand model, the dictionary will have {original seg: reduced seg, reduced seg, ... } if seg is expanded section
    can add a bool for if you specify reduced seg in this case to only return {original seg: desired reduced seg}
    return mapping
    '''
    pass


  def get_str_from_dict():
    '''
    separates a desired string from a dictionary (to use in combination
    '''
    pass
  
  def get_seg_from_str(segments, str):
    '''
    searches list of segments for segment corresponding to str
    returns segment
    '''
    pass

def find_space_const_in_cm(diameter, rm, ra):
    ''' returns space constant (lambda) in cm, according to: space_const = sqrt(rm/(ri+r0)) '''
    # rm = Rm/(PI * diam), diam is in cm and Rm is in ohm * cm^2
    rm = float(rm) / (math.pi * diameter)
    # ri = 4*Ra/ (PI * diam^2), diam is in cm and Ra is in ohm * cm
    ri = float(4 * ra) / (math.pi * (diameter**2))
    space_const = math.sqrt(rm / ri)  # r0 is negligible
    return space_const

  def calculate_nseg_from_lambda(section):
    rm = 1.0 / section.g_pas  # in ohm * cm^2
    ra = section.Ra  # in ohm * cm
    diam_in_cm = section.L / 10000
    space_const_in_cm = find_space_const_in_cm(diam_in_cm,
                                                      rm,
                                                      ra)
    space_const_in_micron = 10000 * space_const_in_cm
    nseg=int((float(section.L) / space_const_in_micron) * 10 / 2) * 2 + 1
    return nseg
  
  def update_model_nseg_using_lambda(cell):
    '''
    Optomizes number of segments using length constant
    '''
    initial_nseg = 0
    new_nseg = 0
    for sec in cell.all:
      initial_nseg += sec.nseg
      sec.nseg = calculate_nseg_from_lambda(sec)
      new_nseg += sec.nseg
    if initial_nseg != new_nseg:
      print('Model nseg changed from', initial_nseg, 'to', new_nseg)
    else:
      print('Model nseg did not change')
