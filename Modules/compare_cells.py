'''
Module for comparing section and segment parameters between models
'''



def is_numerical(val):
    """Check if value is numeric."""
    return isinstance(val, (int, float, complex))

def get_numeric_attributes(obj):
    """Get numeric attributes of an object."""
    return {attr: getattr(obj, attr) for attr in dir(obj) if is_numerical(getattr(obj, attr))}

def compare_objects(obj1, obj2):
    """Compare numeric attributes of two objects."""
    attrs1 = get_numeric_attributes(obj1)
    attrs2 = get_numeric_attributes(obj2)

    differences = {}

    for attr, value in attrs1.items():
        if attr in attrs2:
            if attrs2[attr] != value:
                differences[attr] = (value, attrs2[attr])
        else:
            differences[attr] = (value, None)

    for attr, value in attrs2.items():
        if attr not in attrs1:
            differences[attr] = (None, value)

    return differences

def compare_sections(sec1, sec2):
    """Compare two NEURON sections."""
    if sec1.nseg != sec2.nseg:
        raise ValueError("Sections have different numbers of segments!")

    # Compare section attributes
    section_diff = compare_objects(sec1, sec2)

    # Compare segment attributes for each segment
    segment_diffs = {}
    for i in range(sec1.nseg):
        x = (i + 0.5) / sec1.nseg  # midpoint of segment
        segment_diff = compare_objects(sec1(x), sec2(x))
        
        # Explicitly check for ionic properties
        ions = ['ca_ion', 'na_ion', 'k_ion']
        for ion in ions:
            ion1 = getattr(sec1(x), ion, None)
            ion2 = getattr(sec2(x), ion, None)
            
            if ion1 is not None and ion2 is not None:
                ion_diff = compare_objects(ion1, ion2)
                if ion_diff:
                    segment_diff[ion] = ion_diff
            elif ion1 is not None:
                segment_diff[ion] = (ion1, None)
            elif ion2 is not None:
                segment_diff[ion] = (None, ion2)
                
        if segment_diff:
            segment_diffs[x] = segment_diff

    # Compare mechanisms and their attributes
    mechanisms_diff = {}

    # Get mechanisms present in each section
    mechanisms1 = set(sec1.psection()['density_mechs'].keys())
    mechanisms2 = set(sec2.psection()['density_mechs'].keys())

    # Mechanisms present only in one of the sections
    exclusive_mechanisms1 = mechanisms1 - mechanisms2
    exclusive_mechanisms2 = mechanisms2 - mechanisms1

    for mech_name in exclusive_mechanisms1:
        mechanisms_diff[mech_name] = {"status": "Only in sec1"}

    for mech_name in exclusive_mechanisms2:
        mechanisms_diff[mech_name] = {"status": "Only in sec2"}

    # For mechanisms present in both sections, compare their properties
    common_mechanisms = mechanisms1.intersection(mechanisms2)
    for mech_name in common_mechanisms:
        mechanism_diffs_for_segments = {}
        for i in range(sec1.nseg):
            x = (i + 0.5) / sec1.nseg  # midpoint of segment
            mech1 = getattr(sec1(x), mech_name)
            mech2 = getattr(sec2(x), mech_name)
            mech_diff = compare_objects(mech1, mech2)
            if mech_diff:
                mechanism_diffs_for_segments[x] = mech_diff

        if mechanism_diffs_for_segments:
            mechanisms_diff[mech_name] = mechanism_diffs_for_segments

    return {
        "section": section_diff,
        "segments": segment_diffs,
        "mechanisms": mechanisms_diff
    }


# capture_h_globals AND compare_dictionaries go together

def capture_h_globals(h_obj):
    """Capture all numerical global variables in `h`."""
    h_globals = {}
    for name in dir(h_obj):
        if name != 'h':
            try:
                value = getattr(h_obj, name)
                if is_numerical(value) and not callable(value):
                    h_globals[name] = value
            except (AttributeError, TypeError):  
                # Catching both AttributeError and TypeError
                pass  # If we can't access the attribute for some reason, we just move on
    return h_globals


def compare_dictionaries(dict1, dict2):
    """Compare two dictionaries."""
    diffs = {}
    all_keys = set(dict1.keys()).union(set(dict2.keys()))
    for key in all_keys:
        if key not in dict1:
            diffs[key] = ("Not in dict1", dict2[key])
        elif key not in dict2:
            diffs[key] = (dict1[key], "Not in dict2")
        elif dict1[key] != dict2[key]:
            diffs[key] = (dict1[key], dict2[key])
    return diffs

def adjust_axon_diameter(cell, diff):
    """Adjusts the axon diameter of the given cell based on the differences provided."""
    if "segments" in diff:
        segments_diff = diff["segments"]
        for x, segment_diff in segments_diff.items():
            if "diam" in segment_diff:
                new_diam, _ = segment_diff["diam"]
                cell.axon[0](x).diam = new_diam