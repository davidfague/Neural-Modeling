# Work in progress
# module for running simulation
import time

class Simulation():
  def __init__(self, filename: str = None, record_ecp: bool = False):
    self.filename=filename

  def run(self):
    self.start_timer()
    h.run()
    self.end_timer()
    if filename is not None:
      self.write_data_to_file():

  def write_data_to_file(self):
    pass
  
  def record_ecp(self):
    pass
  
  def start_timer(self):
    pass
    
  def end_timer(self):
    pass
