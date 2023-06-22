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
