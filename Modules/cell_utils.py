import numpy as np

def get_segments_and_len_per_segment(cell):
    # List of segments
    all_segments = []

    # List of segment lengths
    all_len_per_segment = []

    for sec in cell.all:
        for seg in sec:
            all_segments.append(seg)
            all_len_per_segment.append(seg.sec.L / seg.sec.nseg)
    
    all_len_per_segment = np.array(all_len_per_segment)

    return all_segments, all_len_per_segment