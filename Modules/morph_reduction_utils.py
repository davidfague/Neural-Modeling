import collections
CableParams = collections.namedtuple('CableParams',
                                     'length, diam, space_const,'
                                     'cm, rm, ra, e_pas, electrotonic_length')

import math
import cmath
import numpy as np
import neuron
from neuron import h
import itertools as it
import logging
logger = logging.getLogger(__name__)
from logger import Logger
EXCLUDE_MECHANISMS = ('pas', 'na_ion', 'k_ion', 'ca_ion', 'h_ion', 'ttx_ion', )

def get_terminal_coordinates(seg_data, terminal_descend_indices):
    '''Check if [[1,2,3],[1,2,3]] is a good way to store output.'''
    terminal_seg_datas = [seg_data[idx] for idx in terminal_descend_indices]
    terminal_seg_coords = [[terminal_seg_data.coords.p1_0, terminal_seg_data.coords.p1_1, terminal_seg_data.coords.p1_2] for terminal_seg_data in terminal_seg_datas]
    return terminal_seg_coords

def disconnect_root(root_sec):
    pseg = root_sec.parentseg()
    if pseg is not None:
        root_ascendant_x = pseg.x
        h.disconnect(sec=root_sec) # disconnect section from its parent
        return pseg.sec, root_ascendant_x

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L45
import contextlib
@contextlib.contextmanager
def push_section(section):
    '''push a section onto the top of the NEURON stack, pop it when leaving the context'''
    section.push()
    yield
    h.pop_section()

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/subtree_reductor_func.py#L212
def create_segments_to_mech_vals(sections_to_delete,
                                 remove_mechs=False,
                                 exclude=EXCLUDE_MECHANISMS):
    '''This function copy the create a mapping between a dictionary and the mechanisms that it have
       plus the values of those mechanisms. It also remove the mechanisms from the model in order to
       create a passive model

       Arguments:
           remove_mechs - False|True
               if True remove the mechs after creating the mapping, False - keep the mechs
           exclude - List of all the mechs name that should not be removed
       '''
    exclude = set(exclude)
    segment_to_mech_vals, mech_names = {}, set()

    for seg in it.chain.from_iterable(sections_to_delete):
        segment_to_mech_vals[seg] = {}
        for mech in seg:
            mech_name = mech.name()
            segment_to_mech_vals[seg][mech_name] = {}
            for n in dir(mech):
                if n.startswith('__') or n in ('next', 'name', 'is_ion', 'segment', ):
                    continue

                if not n.endswith('_' + mech_name) and not mech_name.endswith('_ion'):
                    n += '_' + mech_name

                segment_to_mech_vals[seg][mech_name][n] = getattr(seg, n)
                mech_names.add(mech_name)

    mech_names -= exclude

    if remove_mechs:  # Remove all the mechs from the sections
        for sec in sections_to_delete:
            with push_section(sec):
                for mech in mech_names:
                    h("uninsert " + mech)

    return segment_to_mech_vals

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L52
def _get_subtree_biophysical_properties(subtree_root_ref, frequency):
    ''' gets the biophysical cable properties (Rm, Ra, Rc) and q

    for the subtree to be reduced according to the properties of the root section of the subtree
    '''
    section = subtree_root_ref.sec

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

#https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L293
def measure_input_impedance_of_subtree(subtree_root_section, frequency):
    '''measures the input impedance of the subtree with the given root section

    (at the "0" tip, the soma-proximal end),
    returns the Impedance hoc object and the input impedance as a complex value
    '''

    imp_obj = h.Impedance()
    CLOSE_TO_SOMA_EDGE = 0
    # sets origin for impedance calculations (soma-proximal end of root section)
    imp_obj.loc(CLOSE_TO_SOMA_EDGE, sec=subtree_root_section)

    # computes transfer impedance from every segment in the model in relation
    # to the origin location above
    imp_obj.compute(frequency + 1 / 9e9, 0)

    # in Ohms (impedance measured at soma-proximal end of root section)
    root_input_impedance = imp_obj.input(CLOSE_TO_SOMA_EDGE, sec=subtree_root_section) * 1000000
    root_input_phase = imp_obj.input_phase(CLOSE_TO_SOMA_EDGE, sec=subtree_root_section)
    # creates a complex impedance value out of the given polar coordinates
    root_input_impedance = cmath.rect(root_input_impedance, root_input_phase)
    return imp_obj, root_input_impedance

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L76
def find_lowest_subtree_impedance(subtree_root_ref, imp_obj):
    '''
    finds the segment in the subtree with the lowest transfer impedance in
    relation to the proximal-to-soma end of the given subtree root section,
    using a recursive hoc function,

    returns the lowest impedance in Ohms
    '''
    # returns [lowest subtree transfer impedance in Mohms, transfer phase]
    lowest_impedance = h.lowest_impedance_recursive(subtree_root_ref, imp_obj)
    # impedance saved as a complex number after converting Mohms to ohms
    curr_lowest_subtree_imp = cmath.rect(lowest_impedance.x[0] * 1000000, lowest_impedance.x[1])
    return curr_lowest_subtree_imp

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L157
def find_subtree_new_electrotonic_length(root_input_impedance, lowest_subtree_impedance, q):
    ''' finds the subtree's reduced cable's electrotonic length

    based on the following equation:
    lowest_subtree_impedance = subtree_root_input_impedance/cosh(q*L)
    according to the given complex impedance values
    '''

    # this equation could be solved analytically using:
    # L = 1/q * arcosh(subtree_root_input_impedance/lowest_subtree_impedance),
    # But since L in this equation is complex number and we chose to focus on
    # finding the correct attenuation
    # we decided to search the L that will result with correct attenuation from
    # the tip of the dendrite to the soma.
    # We chose to use only real L (without a complex part)

    L = find_best_real_L(root_input_impedance, lowest_subtree_impedance, q)
    return L

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L100
def find_best_real_L(Z0, ZL_goal, q, max_L=10.0, max_depth=50):
    '''finds the best real L

    s.t. the modulus part of the impedance of ZL in eq 2.9 will be correct
    Since the modulus is a decreasing function of L, it is easy to find it using binary search.
    '''
    min_L = 0.0
    current_L = (min_L + max_L) / 2.0
    ZL_goal_A = cmath.polar(ZL_goal)[0]

    for _ in range(max_depth):
        Z_current_L_A = compute_zl_polar(Z0, current_L, q)[0]
        if abs(ZL_goal_A - Z_current_L_A) <= 0.001:  # Z are in Ohms , normal values are >10^6
            break
        elif ZL_goal_A > Z_current_L_A:
            current_L, max_L = (min_L + current_L) / 2.0, current_L
        else:
            current_L, min_L = (max_L + current_L) / 2.0, current_L
    else:
        logger.info("The difference between L and the goal L is larger than 0.001")
    return current_L

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L91
def compute_zl_polar(Z0, L, q):
    '''
    given Z0 , L and q computes the polar represntation of ZL (equation 2.9 in Gals thesis)
    '''
    ZL = Z0 * 1.0 / cmath.cosh(q * L)
    ZL = cmath.polar(ZL)
    return ZL

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L208
def find_space_const_in_cm(diameter, rm, ra):
    ''' returns space constant (lambda) in cm, according to: space_const = sqrt(rm/(ri+r0)) '''
    # rm = Rm/(PI * diam), diam is in cm and Rm is in ohm * cm^2
    rm = float(rm) / (math.pi * diameter)
    # ri = 4*Ra/ (PI * diam^2), diam is in cm and Ra is in ohm * cm
    ri = float(4 * ra) / (math.pi * (diameter**2))
    space_const = math.sqrt(rm / ri)  # r0 is negligible
    return space_const

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L177
def _find_subtree_new_diam_in_cm(root_input_impedance, electrotonic_length_as_complex, rm, ra, q):
    '''finds the subtree's new cable's diameter (in cm)

    according to the given complex input impedance at the segment in the
    original subtree that is closest to the soma (the tip), and the given cable
    electrotonic length,

    with the following equation:
    d (in cm) = (2/PI * (sqrt(RM*RA)/(q*subtree_root_input_impedance)) *
                 (coth(q * NewCableElectrotonicLength)) )^(2/3)
    derived from Rall's cable theory for dendrites (Gal Eliraz)
    '''

    diam_in_cm = (2.0 / math.pi *
                  (math.sqrt(rm * ra) / (q * root_input_impedance)) *
                  (1 / cmath.tanh(q * electrotonic_length_as_complex))  # coth = 1/tanh
                  ) ** (2.0 / 3)

    '''
    # for debugging inaccuracies:
    if diam_in_cm.imag != 0:
        if abs(diam_in_cm.imag) > 0.03:
        print "PROBLEM - DIAM HAS SUBSTANTIAL IMAGINARY PART"
        print "\n"
    '''

    # the radius of the complex number received from the equation
    new_subtree_dend_diam_in_cm = cmath.polar(diam_in_cm)[0]
    return new_subtree_dend_diam_in_cm

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L218
def reduce_subtree(subtree_root, frequency=0):
    '''Reduces the subtree  from the original_cell into one single section (cable).

    The reduction is done by finding the length and diameter of the cable (a
    single solution) that preserves the subtree's input impedance at the
    somatic end, and the transfer impedance in the subtree from the distal end
    to the proximal somatic end (between the new cable's two tips).
    '''

    subtree_root_ref = h.SectionRef(sec=subtree_root)
    cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(subtree_root_ref, frequency)

    # finds the subtree's input impedance (at the somatic-proximal end of the
    # subtree root section) and the lowest transfer impedance in the subtree in
    # relation to the somatic-proximal end (see more in Readme on NeuroReduce)
    imp_obj, root_input_impedance = measure_input_impedance_of_subtree(subtree_root, frequency)

    # in Ohms (a complex number)
    curr_lowest_subtree_imp = find_lowest_subtree_impedance(subtree_root_ref, imp_obj)

    # reducing the whole subtree into one section:
    # L = 1/q * arcosh(ZtreeIn(f)/min(ZtreeX,0(f)),
    # d = ( (2/pi * (sqrt(Rm*Ra)/q*ZtreeIn(f)) * coth(qL) )^(2/3) - from Gal Eliraz's thesis 1999
    new_cable_electrotonic_length = find_subtree_new_electrotonic_length(root_input_impedance,
                                                                         curr_lowest_subtree_imp,
                                                                         q)
    cable_electrotonic_length_as_complex = complex(new_cable_electrotonic_length, 0)
    new_cable_diameter_in_cm = _find_subtree_new_diam_in_cm(root_input_impedance,
                                                            cable_electrotonic_length_as_complex,
                                                            rm,
                                                            ra,
                                                            q)
    new_cable_diameter = new_cable_diameter_in_cm * 10000   # in microns

    # calculating the space constant, in order to find the cylinder's length:
    # space_const = sqrt(rm/(ri+r0))
    curr_space_const_in_cm = find_space_const_in_cm(new_cable_diameter_in_cm,
                                                    rm,
                                                    ra)
    curr_space_const_in_micron = 10000 * curr_space_const_in_cm
    new_cable_length = curr_space_const_in_micron * new_cable_electrotonic_length  # in microns

    return CableParams(length=new_cable_length,
                       diam=new_cable_diameter,
                       space_const=curr_space_const_in_micron,
                       cm=cm,
                       rm=rm,
                       ra=ra,
                       e_pas=e_pas,
                       electrotonic_length=new_cable_electrotonic_length)

def calculate_nsegs_from_lambda(new_cable_properties, nsegs_per_lambda=10):
    '''calculate the number of segments for each section in the reduced model

    according to the length (in microns) and space constant (= lambda - in
    microns) that were previously calculated for each section and are given in
    subtree_dimensions.  According to this calculation, a segment is formed for
    every 0.1 * lambda in a section. (lambda = space constant = electrotonic length unit).
    '''
    # for every unit of electronic length (length/space_constant such units)
    # ~10 segments are formed
    nseg = (int((float(new_cable_properties.length) / new_cable_properties.space_const) * nsegs_per_lambda / 2) * 2 + 1)
    return nseg

# adapted from https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/subtree_reductor_func.py#L450
def apply_params_to_section(section, cable_params, nseg):
    section.L = cable_params.length
    section.diam = cable_params.diam
    section.nseg = nseg

    #append_to_section_lists(name, type_of_sectionlist, instance_as_str)

    section.insert('pas')
    section.cm = cable_params.cm
    section.g_pas = 1.0 / cable_params.rm
    section.Ra = cable_params.ra
    section.e_pas = cable_params.e_pas

# adapted from https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/subtree_reductor_func.py#L253
def create_seg_to_seg(new_section,
                      root_descendants,
                      root_sec,
                      #mapping_sections_to_subtree_index,
                      new_cable_properties,
                    #   has_apical,
                    #   apic,
                    #   basals,
                    #   subtree_ind_to_q,
                      mapping_type,
                      reduction_frequency):
    '''create mapping between segments in the original model to segments in the reduced model

       if mapping_type == impedance the mapping will be a response to the
       transfer impedance of each segment to the soma (like the synapses)

       if mapping_type == distance  the mapping will be a response to the
       distance of each segment to the soma (like the synapses) NOT IMPLEMENTED
       YET

       '''

    assert mapping_type == 'impedance', 'distance mapping not implemented yet'
    # the keys are the segments of the original model, the values are the
    # segments of the reduced model
    original_seg_to_reduced_seg = {}
    reduced_seg_to_original_seg = collections.defaultdict(list)
    
    imp_obj, subtree_input_impedance = measure_input_impedance_of_subtree(root_sec, reduction_frequency)
    
    root_q = calculate_subtree_q(root_sec, reduction_frequency)
    sections_to_map = [root_sec] + root_descendants
    for sec in sections_to_map:
        for seg in sec:
            synapse_location = find_synapse_loc(seg)
            mid_of_segment_loc = reduce_synapse(
                synapse_location,
                sec,
                imp_obj,
                subtree_input_impedance,
                new_cable_properties.electrotonic_length,
                root_q)

            reduced_seg = new_section(mid_of_segment_loc)
            original_seg_to_reduced_seg[seg] = reduced_seg
            reduced_seg_to_original_seg[reduced_seg].append(seg)

    return original_seg_to_reduced_seg, dict(reduced_seg_to_original_seg)

# adapted from https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/subtree_reductor_func.py#L151C1-L178C63
def find_synapse_loc(synapse_or_segment):
    ''' Returns the normalized location (x) of the given synapse or segment'''
    if not isinstance(synapse_or_segment, neuron.nrn.Segment):
        synapse_or_segment = synapse_or_segment.get_segment()
    return synapse_or_segment.x

# adapted from https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L317
def reduce_synapse(
                   synapse_location,
                   section,
                   imp_obj,
                   root_input_impedance,
                   new_cable_electrotonic_length,
                   q_subtree
                   ):
    '''
    Receives an instance of a cell, the location (section + relative
    location(x)) of a synapse to be reduced, a boolean on_basal that is True if
    the synapse is on a basal subtree, the number of segments in the reduced
    cable that this synapse is in, an Impedance calculating Hoc object, the
    input impedance at the root of this subtree, and the electrotonic length of
    the reduced cable that represents the current subtree
    (as a real and as a complex number) -
    and maps the given synapse to its new location on the reduced cable
    according to the NeuroReduce algorithm.  Returns the new "post-merging"
    relative location of the synapse on the reduced cable (x, 0<=x<=1), that
    represents the middle of the segment that this synapse is located at in the
    new reduced cable.
    '''
    # measures the original transfer impedance from the synapse to the
    # somatic-proximal end in the subtree root section

    with push_section(section):
        orig_transfer_imp = imp_obj.transfer(synapse_location) * 1000000  # ohms
        orig_transfer_phase = imp_obj.transfer_phase(synapse_location)
        # creates a complex Impedance value with the given polar coordinates
        orig_synapse_transfer_impedance = cmath.rect(orig_transfer_imp, orig_transfer_phase)

    # synapse location could be calculated using:
    # X = L - (1/q) * arcosh( (Zx,0(f) / ZtreeIn(f)) * cosh(q*L) ),
    # derived from Rall's cable theory for dendrites (Gal Eliraz)
    # but we chose to find the X that will give the correct modulus. See comment about L values

    synapse_new_electrotonic_location = find_best_real_X(root_input_impedance,
                                                         orig_synapse_transfer_impedance,
                                                         q_subtree,
                                                         new_cable_electrotonic_length)
    new_relative_loc_in_section = (float(synapse_new_electrotonic_location) /
                                   new_cable_electrotonic_length)

    if new_relative_loc_in_section > 1:  # PATCH
        new_relative_loc_in_section = 0.999999

    return new_relative_loc_in_section

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/subtree_reductor_func.py#L464
def calculate_subtree_q(root, reduction_frequency):
    rm = 1.0 / root.g_pas
    rc = rm * (float(root.cm) / 1000000)
    angular_freq = 2 * math.pi * reduction_frequency
    q_imaginary = angular_freq * rc
    q_subtree = complex(1, q_imaginary)   # q=1+iwRC
    q_subtree = cmath.sqrt(q_subtree)
    return q_subtree

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L131
def find_best_real_X(Z0, ZX_goal, q, L, max_depth=50):
    '''finds the best location of a synapse (X)

    s.t. the modulus part of the impedance of ZX in eq 2.8 will be correct.
    Since the modulus is a decreasing function of L, it is easy to find it using binary search.
    '''
    min_x, max_x = 0.0, L
    current_x = (min_x + max_x) / 2.0

    ZX_goal = cmath.polar(ZX_goal)[0]

    for _ in range(max_depth):
        Z_current_X_A = compute_zx_polar(Z0, L, q, current_x)[0]

        if abs(ZX_goal - Z_current_X_A) <= 0.001:
            break
        elif ZX_goal > Z_current_X_A:
            current_x, max_x = (min_x + current_x) / 2.0, current_x
        else:
            current_x, min_x = (max_x + current_x) / 2.0, current_x
    else:
        logger.info("The difference between X and the goal X is larger than 0.001")

    return current_x

# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/reducing_methods.py#L123
def compute_zx_polar(Z0, L, q, x):
    '''computes the polar represntation of Zx (equation 2.8 in Gals thesis)
    '''
    ZX = Z0 * cmath.cosh(q * (L - x)) / cmath.cosh(q * L)
    ZX = cmath.polar(ZX)
    return ZX

# adapted from https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/subtree_reductor_func.py#L314
def copy_dendritic_mech(original_seg_to_reduced_seg,
                        reduced_seg_to_original_seg,
                        segment_to_mech_vals,
                        new_section,
                        all_segments,
                        mapping_type='impedance'):
    ''' copies the mechanisms from the original model to the reduced model'''

    # copy mechanisms
    # this is needed for the case where some segements were not been mapped
    mech_names_per_segment = collections.defaultdict(list)
    vals_per_mech_per_segment = {}
    for reduced_seg, original_segs in reduced_seg_to_original_seg.items():
        vals_per_mech_per_segment[reduced_seg] = collections.defaultdict(list)

        for original_seg in original_segs:
            for mech_name, mech_params in segment_to_mech_vals[original_seg].items():
                for param_name, param_value in mech_params.items():
                    vals_per_mech_per_segment[reduced_seg][param_name].append(param_value)

                mech_names_per_segment[reduced_seg].append(mech_name)
                reduced_seg.sec.insert(mech_name)

        for param_name, param_values in vals_per_mech_per_segment[reduced_seg].items():
            setattr(reduced_seg, param_name, np.mean(param_values))

    all_reduced_segments = [list(new_section)]
    

    if len(all_reduced_segments) != len(reduced_seg_to_original_seg):
        logger.warning('There is no segment to segment copy, it means that some segments in the'
                    'reduced model did not receive channels from the original cell.'
                    'Trying to compensate by copying channels from neighboring segments')
        handle_orphan_segments(original_seg_to_reduced_seg,
                               all_segments,
                               vals_per_mech_per_segment,
                               mech_names_per_segment)
 
# https://github.com/orena1/neuron_reduce/blob/94b1850607f9c62a205ad8fb695f2fd91d84d87d/neuron_reduce/subtree_reductor_func.py#L357C1-L424C57       
def handle_orphan_segments(original_seg_to_reduced_seg,
                           all_segments,
                           vals_per_mech_per_segment,
                           mech_names_per_segment):
    ''' This function handle reduced segments that did not had original segments mapped to them'''
    # Get all reduced segments that have been mapped by a original model segment
    all_mapped_control_segments = original_seg_to_reduced_seg.values()
    non_mapped_segments = set(all_segments) - set(all_mapped_control_segments)

    for reduced_seg in non_mapped_segments:
        seg_secs = list(reduced_seg.sec)
        # find valid parent
        parent_seg_index = seg_secs.index(reduced_seg) - 1
        parent_seg = None
        while parent_seg_index > -1:
            if seg_secs[parent_seg_index] in all_mapped_control_segments:
                parent_seg = seg_secs[parent_seg_index]
                break
            else:
                parent_seg_index -= 1

        # find valid child
        child_seg_index = seg_secs.index(reduced_seg) + 1
        child_seg = None
        while child_seg_index < len(seg_secs):
            if seg_secs[child_seg_index] in all_mapped_control_segments:
                child_seg = seg_secs[child_seg_index]
                break
            else:
                child_seg_index += 1

        if not parent_seg and not child_seg:
            raise Exception("no child seg nor parent seg, with active channels, was found")

        if parent_seg and not child_seg:
            for mech in mech_names_per_segment[parent_seg]:
                reduced_seg.sec.insert(mech)
            for n in vals_per_mech_per_segment[parent_seg]:
                setattr(reduced_seg, n, np.mean(vals_per_mech_per_segment[parent_seg][n]))

        if not parent_seg and child_seg:
            for mech in mech_names_per_segment[child_seg]:
                reduced_seg.sec.insert(mech)
            for n in vals_per_mech_per_segment[child_seg]:
                setattr(reduced_seg, n, np.mean(vals_per_mech_per_segment[child_seg][n]))

        # if both parent and child were found, we add to the segment all the mech in both
        # this is just a decision

        if parent_seg and child_seg:
            for mech in set(mech_names_per_segment[child_seg]) & set(mech_names_per_segment[parent_seg]):
                reduced_seg.sec.insert(mech)

            for n in vals_per_mech_per_segment[child_seg]:
                child_mean = np.mean(vals_per_mech_per_segment[child_seg][n])
                if n in vals_per_mech_per_segment[parent_seg]:
                    parent_mean = np.mean(vals_per_mech_per_segment[parent_seg][n])
                    setattr(reduced_seg, n, (child_mean + parent_mean) / 2)
                else:
                    setattr(reduced_seg, n, child_mean)

            for n in vals_per_mech_per_segment[parent_seg]:
                parent_mean = np.mean(vals_per_mech_per_segment[parent_seg][n])
                if n in vals_per_mech_per_segment[child_seg]:
                    child_mean = np.mean(vals_per_mech_per_segment[child_seg][n])
                    setattr(reduced_seg, n, (child_mean + parent_mean) / 2)
                else:
                    setattr(reduced_seg, n, parent_mean)