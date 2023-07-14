from neuron import h

def build_L5_cell(cell_folder, biophys = 'L5PCbiophys3.hoc', 3d_morph = 'import3d.hoc'):
    # Load biophysics
    h.load_file(cell_folder + biophys)

    # Load morphology
    h.load_file(3d_morph)

    # Load builder
    h.load_file(cell_folder + 'L5PCtemplate.hoc')

    # Build complex_cell object
    complex_cell = h.L5PCtemplate(cell_folder + 'cell1.asc')

    return complex_cell


    
