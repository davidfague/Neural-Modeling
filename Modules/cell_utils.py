import numpy as np

def get_segments_and_len_per_segment(cell):
    seg_coords = calc_seg_coords(cell)
    # all = except axon
    # soma = only soma
    # no_soma = except soma, except axon
    
    # list of segment centers
    all_segments_centers=seg_coords['pc'] #with axon
    all_segments_center=[] # without axon
    soma_segments_center=[]
    no_soma_segments_center=[]
    
    # List of segments
    all_segments = []
    soma_segments = []
    no_soma_segments = []
    
    # List of segment lengths
    soma_len_per_segment = []
    all_len_per_segment = []
    no_soma_len_per_segment = []
    
    # List od segment surface areas
    soma_SA_per_segment = []
    all_SA_per_segment = []
    no_soma_SA_per_segment = []

    i= 0 #seg index
    for sec in cell.all:
        if 'axon' in sec.name():#(sec in cell.axon) or (sec is cell.axon):
            for seg in sec:
                i+=1
        elif 'soma' in sec.name():#(sec in cell.soma) or (sec is cell.soma):
            for seg in sec:
                soma_segments.append(seg)
                soma_len_per_segment.append(seg.sec.L / seg.sec.nseg)
                soma_SA_per_segment.append(np.pi*seg.diam*(seg.sec.L/seg.sec.nseg))
                all_segments.append(seg)
                all_len_per_segment.append(seg.sec.L / seg.sec.nseg)
                all_SA_per_segment.append(np.pi*seg.diam*(seg.sec.L/seg.sec.nseg))
                soma_segments_center.append(all_segments_centers[i])
                all_segments_center.append(all_segments_centers[i])
                i+=1
        else:
            for seg in sec:
                all_segments.append(seg)
                all_len_per_segment.append(seg.sec.L / seg.sec.nseg)
                all_SA_per_segment.append(np.pi*seg.diam*(seg.sec.L/seg.sec.nseg))
                no_soma_segments.append(seg)
                no_soma_len_per_segment.append(seg.sec.L / seg.sec.nseg)
                no_soma_SA_per_segment.append(np.pi*seg.diam*(seg.sec.L/seg.sec.nseg))
                no_soma_segments_center.append(all_segments_centers[i])
                all_segments_center.append(all_segments_centers[i])
                i+=1
    
    all_len_per_segment = np.array(all_len_per_segment)
    all_SA_per_segment = np.array(all_SA_per_segment)
    soma_len_per_segment = np.array(soma_len_per_segment)
    soma_SA_per_segment = np.array(soma_SA_per_segment)    
    no_soma_len_per_segment = np.array(no_soma_len_per_segment)
    no_soma_SA_per_segment = np.array(no_soma_SA_per_segment)

    return all_segments, all_len_per_segment, all_SA_per_segment, all_segments_center, soma_segments, soma_len_per_segment, soma_SA_per_segment, soma_segments_center, no_soma_segments, no_soma_len_per_segment, no_soma_SA_per_segment, no_soma_segments_center

#def calc_seg_coords(cell) -> dict:
#
#    nseg_total = sum(sec.nseg for sec in cell.all)
#    p0, p05, p1 = np.zeros((nseg_total, 3)), np.zeros((nseg_total, 3)), np.zeros((nseg_total, 3))
#    r = np.zeros(nseg_total)
#
#    seg_idx = 0
#    for sec in cell.all:
#
#        seg_length = sec.L / sec.nseg # get the segment length
#        # find the segment coordinates from the section coordinates
#        for i in range(sec.n3d()-1): # iterate through section 3d coordinates
#            arc_length = [sec.arc3d(i), sec.arc3d(i+1)] # arc length from the beginning of the section to the current sec 3d coordinate and the next 3d coordinate
#            for seg in sec:
#                if (arc_length[0] / sec.L) <= seg.x < (arc_length[1] / sec.L): # if the center of the segment is between the two 3D coordinates
#                    seg_x_between_coordinates = (seg.x * sec.L - arc_length[0]) / (arc_length[1] - arc_length[0]) # normalized distance between the two 3D coordaintes
#                    xyz_before = [sec.x3d(i), sec.y3d(i), sec.z3d(i)] # sec coordinate before segment middle
#                    xyz_after = [sec.x3d(i+1), sec.y3d(i+1), sec.z3d(i+1)] # sec coordinate after segment middle
#
#                    pt = np.array([xyz_before[k] + (xyz_after[k] - xyz_before[k]) * seg_x_between_coordinates for k in range(3)])
#                    dxdydz = np.array([(xyz_after[k] - xyz_before[k]) * (seg_length / 2) / (arc_length[1] - arc_length[0]) for k in range(3)])
#                    
#                    pt_back, pt_forward = pt - dxdydz, pt + dxdydz
#
#                    p0[seg_idx], p05[seg_idx], p1[seg_idx] = pt_back, pt, pt_forward
#                    r[seg_idx] = seg.diam / 2
#
#                    seg_idx += 1
#
#    seg_coords = {'p0': p0, 'p1': p1, 'pc': p05, 'r': r, 'dl': p1 - p0}
#
#    return seg_coords

import numpy as np

def calc_seg_coords(cell) -> dict:
    nseg_total = sum(sec.nseg for sec in cell.all)
    p0, p05, p1 = np.zeros((nseg_total, 3)), np.zeros((nseg_total, 3)), np.zeros((nseg_total, 3))
    r = np.zeros(nseg_total)

    seg_idx = 0
    for sec in cell.all:
        seg_length = sec.L / sec.nseg

        arc_lengths = [sec.arc3d(i) for i in range(sec.n3d())]
        coords = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(sec.n3d())])

        for seg in sec:
            start = seg.x * sec.L - seg_length / 2
            end = seg.x * sec.L + seg_length / 2
            mid = seg.x * sec.L
        
            for i in range(len(arc_lengths) - 1):
                # Check if segment's middle is between two 3D coordinates
                if arc_lengths[i] <= mid < arc_lengths[i+1]:
                    t = (mid - arc_lengths[i]) / (arc_lengths[i+1] - arc_lengths[i])
                    pt = coords[i] + (coords[i+1] - coords[i]) * t
        
                    # Calculate the start and end points of the segment
                    direction = (coords[i+1] - coords[i]) / np.linalg.norm(coords[i+1] - coords[i])
                    p0[seg_idx] = pt - direction * seg_length / 2
                    p1[seg_idx] = pt + direction * seg_length / 2
        
                    # Correct the start point if it goes before 3D coordinates
                    while i > 0 and start < arc_lengths[i]:  # Added boundary check i > 0
                        i -= 1
                        direction = (coords[i+1] - coords[i]) / np.linalg.norm(coords[i+1] - coords[i])
                        p0[seg_idx] = coords[i] + direction * (start - arc_lengths[i])
        
                    # Correct the end point if it goes beyond 3D coordinates
                    while end > arc_lengths[i+1] and i+2 < len(arc_lengths):
                        i += 1
                        direction = (coords[i+1] - coords[i]) / np.linalg.norm(coords[i+1] - coords[i])
                        p1[seg_idx] = coords[i] + direction * (end - arc_lengths[i])
        
                    p05[seg_idx] = pt
                    r[seg_idx] = seg.diam / 2
                    seg_idx += 1
                    break

    seg_coords = {'p0': p0, 'p1': p1, 'pc': p05, 'r': r, 'dl': p1 - p0}

    return seg_coords

