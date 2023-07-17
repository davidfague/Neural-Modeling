from neuron import h
import numpy as np

class Recorder:

    def __init__(self, obj_list: list, var_name: str = 'v', vector_length: int = None):
        """
        Parameters:
        ----------
        obj_list: list
            List of target objects.
        
        var_name: str
            Name of variable to be recorded.

        """
        self.obj_list = obj_list
        self.var_name = var_name
        self.vectors = None
        self.setup_recorder(vector_length)

    def setup_recorder(self, vector_length: int) -> None:
        size = round(vector_length / h.dt) + 1
        attr_name = '_ref_' + self.var_name
        self.vectors = [h.Vector(size).record(getattr(obj, attr_name)) for obj in self.obj_list]

    #TODO: why copy?
    def as_numpy(self, copy: bool = True) -> np.ndarray:
        """
        Parameters:
        ----------
        copy: bool = True
            ...

        Returns:
        ----------
        x: np.ndarray(shape = (num_objects, time_len))
            Array of recording.
        """
        x = np.array([v.as_numpy() for v in self.vectors])
        if copy: x = x.copy()
        return x
