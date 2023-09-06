from neuron import h

def build_L5_cell(cell_folder, biophys = 'L5PCbiophys3.hoc', morph = 'cell1.asc', template = 'L5PCtemplate.hoc'):
    # Load biophysics
    h.load_file(cell_folder + biophys)

    # Load morphology
    h.load_file("import3d.hoc")

    # Load builder
    h.load_file(cell_folder + template)

    # Build complex_cell object
    complex_cell = eval("h." + template.split('.')[0] + '(cell_folder + morph)')
    #complex_cell = h.L5PCtemplate(cell_folder + morph)

    return complex_cell


    
def build_L5_cell_ziao(cell_folder, template='ziao_templates.hoc'):
    h.load_file(cell_folder + template)
    complex_cell = h.CP_Cell()
    
    return complex_cell
