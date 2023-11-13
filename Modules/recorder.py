from neuron import h

class SectionRecorder:

	def __init__(self, seg: object, var_name: str):
		self.var_name = var_name
		self.vec = h.Vector()

		attr = getattr(seg(0.5), '_ref_' + var_name)
		self.vec.record(attr)

class SynapseRecorder:

	def __init__(self, syn: object, var_name: str):
		self.var_name = var_name
		self.vec = h.Vector()

		attr = getattr(syn, '_ref_' + var_name)
		self.vec.record(attr)

class SpikeRecorder:

	def __init__(self, obj: object, var_name: str, spike_threshold: float):
		self.var_name = var_name
		self.vec = h.Vector()
		self.spike_threhsold = spike_threshold

		nc = h.NetCon(obj(0.5)._ref_v, None, sec = obj)
		nc.threshold = spike_threshold
		nc.record(self.vec)
