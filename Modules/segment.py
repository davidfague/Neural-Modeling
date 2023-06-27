class Segment():
    '''
    class for storing seg info and recorded data
    '''
    def __init__(self, seg_info: dict = None, seg_data: dict = None):
        # seg_info is a row from a dataframe
        # set seg_info into Segment attributes
        for info_key, info_value in seg_info.items():
            clean_key = str(info_key).replace(" ", "_").replace(".", "_") # change space and '.' both to '_'
            setattr(self, clean_key, info_value)

        self.name = self.seg # for clarity
        # set seg_data into Segment attributes
        for data_type in seg_data:
            setattr(self, str(data_type), seg_data[data_type])

        # set segment color
        if self.Type == 'soma':
          self.color = 'purple'
        elif self.Type == 'dend':
          self.color = 'red'
        elif self.Type == 'apic':
          self.color = 'blue'
        elif self.Type == 'axon':
          self.color = 'green'
        else:
          raise(ValueError("Section type not implemented", self.Type))
        # initialize lists for later
        self.axial_currents = []
        self.adj_segs = [] # adjacent segments list
        self.child_segs = []
        self.parent_segs = []
        self.parent_axial_currents = []
        self.child_axial_currents = []
