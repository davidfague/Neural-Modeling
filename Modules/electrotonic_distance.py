import collections
import re
import cmath
import datetime
import numpy as np
import neuron
from neuron import h
import math

from adjacency import *

def calculate_electrotonic_length_of_section(section):
  cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(section, frequency=0) # Assumes that these properties do not vary across section
  sec_elec_L = 0
  for seg in section:
    seg_elec_L = calculate_electrotonic_length_of_segment(seg, cm, rm, ra, e_pas, q)
    sec_elec_L += seg_elec_L
    
  return sec_elec_L
  
def calculate_electrotonic_length_of_segment(seg, cm=None, rm=None, ra=None, e_pas=None, q=None):
    if cm is None or rm is None or ra is None or e_pas is None or q is None:
        cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(seg.sec)
    cable_space_const_in_cm = find_space_const_in_cm(seg.diam/10000, rm, ra) # think this will approximate the diameter as a segment as the diameter in the middle, which should also be the average if it is linearly changing.
    seg_L = seg.sec.L/seg.sec.nseg
    seg_elec_L = seg_L/(cable_space_const_in_cm*10000)
    
    return seg_elec_L
    
def calculate_electrotonic_length_of_dendrite_path(cell_model, starting_segment_id, terminating_segment_id, adjacency_matrix):
  segment_indices = find_path_segments(adjacency_matrix, starting_segment_id, terminating_segment_id)
  #print(f"path from {starting_segment_id} to {terminating_segment_id} is: {segment_indices}")
  segments, _ = cell_model.get_segments(['all'])
  segments_to_calc = [segments[i] for i in segment_indices]
  
  total_elec_L = 0
  for seg in segments_to_calc:
    elec_L = calculate_electrotonic_length_of_segment(seg)
    total_elec_L += elec_L
    
  return total_elec_L

def calc_length_constant_in_microns(sec):
  cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(sec) # sourced
  cable_space_const_in_cm = find_space_const_in_cm(sec.diam/10000, rm, ra) # sourced
  length_constant_in_microns = cable_space_const_in_cm*10000 # unit correction
  return length_constant_in_microns

# Adapted to handle sec objects from Source: https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L52
def _get_subtree_biophysical_properties(section, frequency=0):
    ''' gets the biophysical cable properties (Rm, Ra, Rc) and q

    for the subtree to be reduced according to the properties of the root section of the subtree
    '''
    #section = subtree_root_ref.sec #subtree_root_ref was passed originally

    rm = 1.0 / section.g_pas  # in ohm * cm^2
    # in secs, with conversion of the capacitance from uF/cm2 to F/cm2
    RC = rm * (float(section.cm) / 1000000)

    # defining q=sqrt(1+iwRC))
    angular_freq = 2 * math.pi * frequency   # = w
    q_imaginary = angular_freq * RC
    q = complex(1, q_imaginary)   # q=1+iwRC
    q = cmath.sqrt(q)		# q = sqrt(1+iwRC)

    return (section.cm,
            rm,
            section.Ra,  # in ohm * cm
            section.e_pas,
            q) 

# SOURCE: https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L208     
def find_space_const_in_cm(diameter, rm, ra):
    ''' returns space constant (lambda) in cm, according to: space_const = sqrt(rm/(ri+r0)) '''
    # rm = Rm/(PI * diam), diam is in cm and Rm is in ohm * cm^2
    rm = float(rm) / (math.pi * diameter)
    # ri = 4*Ra/ (PI * diam^2), diam is in cm and Ra is in ohm * cm
    ri = float(4 * ra) / (math.pi * (diameter**2))
    space_const = math.sqrt(rm / ri)  # r0 is negligible
    return space_const
# -------------------------------------------------    
'''	# calculate the electrotonic length of the cable
	cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(h.SectionRef(sec=section_to_expand), frequency)
	cable_space_const_in_cm = find_space_const_in_cm(section_to_expand(0.5).diam/10000, rm, ra)
	cable_elec_L = section_to_expand.L/(cable_space_const_in_cm*10000)
	
	# calculate the diameter of each branch
	trunk_diam = section_to_expand.diam
	branch_diam_in_micron = (trunk_diam**(3/2)/nbranch)**(2/3)
	branch_diam_in_cm = branch_diam_in_micron/10000
	
	# calculate the electrotonic length of each branch
	trunk_elec_L = furcation_x * cable_elec_L
	branch_elec_L = cable_elec_L - trunk_elec_L
	branch_space_const_in_cm = find_space_const_in_cm(branch_diam_in_cm, rm, ra)  # Convert back to cm
	branch_space_const_in_micron = 10000 * branch_space_const_in_cm
	branch_L = branch_elec_L * branch_space_const_in_micron
	
	# calculate the other parameters for each branch
	trunk_diam_in_cm = trunk_diam/10000
	trunk_L = section_to_expand.L*furcation_x
	sec_type = section_to_expand.name().split(".")[1][:4]
	
	# create CableParams objects for the trunk and branch
	trunk_params = CableParams(length=trunk_L, diam=trunk_diam, space_const=cable_space_const_in_cm*10000,
							   cm=cm, rm=rm, ra=ra, e_pas=e_pas, electrotonic_length=trunk_elec_L,
							   type=sec_type, furcation_x=furcation_x)
	
	branch_params = CableParams(length=branch_L, diam=branch_diam_in_micron, space_const=branch_space_const_in_micron,
								cm=cm, rm=rm, ra=ra, e_pas=e_pas, electrotonic_length=branch_elec_L,
								type=sec_type, furcation_x=furcation_x)
                                                                                    '''