import numpy as np
from neuron import h
from neuron import nrn

class SEClamp:
  def __init__(self, seg, dur1: float, amp1: float, rs: float = 0.01, record: bool = False):
    # Voltage Clamp
    self.clamp_nrn_obj = h.SEClamp(seg) # create voltage clamp object #Use single electrode Vclamp over double electrode Vclamp because it is better.
    self.clamp_nrn_obj.dur1 = dur1 # (ms) duration 
    self.clamp_nrn_obj.amp1 = amp1 # (mV) amplitude
    self.clamp_nrn_obj.rs = rs # (MOhm) # access resistance from cell to electrode once electrode has penetrated cell (lower is better, but less experimentally obtainable.)
    if record:
      self.setup_recorder()
    
  def setup_recorder(self):
        size = [round(h.tstop / h.dt) + 1]
        self.rec_vec = h.Vector(*size).record(self.clamp_nrn_obj._ref_i) #record