import numpy as np

def get_segments_and_len_per_segment(cell):
    seg_coords = calc_seg_coords(complex_cell)
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
        if sec in cell.axon:
            for seg in sec:
                i+=1
        elif sec in cell.soma:
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

def calc_seg_coords(cell) -> dict:

    nseg_total = sum(sec.nseg for sec in cell.all)
    p0, p05, p1 = np.zeros((nseg_total, 3)), np.zeros((nseg_total, 3)), np.zeros((nseg_total, 3))
    r = np.zeros(nseg_total)

    seg_idx = 0
    for sec in cell.all:

        seg_length = sec.L / sec.nseg

        for i in range(sec.n3d()-1):
            arc_length = [sec.arc3d(i), sec.arc3d(i+1)] # Before, after
            for seg in sec:
                if (arc_length[0] / sec.L) <= seg.x < (arc_length[1] / sec.L):
                    seg_x_between_coordinates = (seg.x * sec.L - arc_length[0]) / (arc_length[1] - arc_length[0])
                    xyz_before = [sec.x3d(i), sec.y3d(i), sec.z3d(i)]
                    xyz_after = [sec.x3d(i+1), sec.y3d(i+1), sec.z3d(i+1)]

                    pt = np.array([xyz_before[k] + (xyz_after[k] - xyz_before[k]) * seg_x_between_coordinates for k in range(3)])
                    dxdydz = np.array([(xyz_after[k] - xyz_before[k]) * (seg_length / 2) / (arc_length[1] - arc_length[0]) for k in range(3)])
                    
                    pt_back, pt_forward = pt - dxdydz, pt + dxdydz

                    p0[seg_idx], p05[seg_idx], p1[seg_idx] = pt_back, pt, pt_forward
                    r[seg_idx] = seg.diam / 2

                    seg_idx += 1

    seg_coords = {'p0': p0, 'p1': p1, 'pc': p05, 'r': r, 'dl': p1 - p0}

    return seg_coords
