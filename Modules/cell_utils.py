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

