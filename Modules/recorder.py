from neuron import h

class Recorder:

	def __init__(self, obj: object, name: str, vector_length: int):
		self.name = name
		self.vec = h.Vector(vector_length)

		attr_name = '_ref_' + name
		attr = getattr(obj(0.5), attr_name)
		self.vec.record(attr)

class SpikeRecorder:

	def __init__(self, obj: object, name: str, spike_threshold: float):
		self.name = name
		self.vec = h.Vector()
		self.spike_threhsold = spike_threshold

		nc = h.NetCon(obj(0.5)._ref_v, None, sec = obj)
		nc.threshold = spike_threshold
		nc.record(self.vec)
