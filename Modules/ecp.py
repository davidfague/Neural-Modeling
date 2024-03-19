import numpy as np

# ECP MOD: https://github.com/chenziao/Stylized-Cell-Inference/blob/main/cell_inference/utils/currents/ecp.py

# previously used cell object: https://github.com/chenziao/Stylized-Cell-Inference/blob/main/cell_inference/cells/stylizedcell.py

# This is a post-processing version of the above code.
class ECP(object):
  def __init__(self, i_membrane, seg_coords, min_distance):
      self.i_membrane = i_membrane
      self.seg_coords = seg_coords
      self.move_cell= None
      self.scale=1
      self.min_distance = min_distance
      self._nseg=len(i_membrane)
      #print(self._nseg)
      #self.pc_array = np.stack(seg_coords['pc'].values)
      #self.dl_array = np.stack(seg_coords['dl'].values)
      

  def set_electrode_positions(self, electrode_positions: np.ndarray) -> None:
      self.elec_coords = np.asarray(electrode_positions)
      if self.elec_coords.ndim != 2 or self.elec_coords.shape[1] != 3:
          raise ValueError("'electrode_positions' must be an n-by-3 2-D array")
      self.nelec = self.elec_coords.shape[0]
  
  def calc_transfer_resistance(self, move_cell: list = None,
                               scale: float = 1.0, min_distance: float = None,
                               move_elec: bool = False, sigma: float = 0.3,) -> None:
      """
      Precompute mapping from segment to electrode locations
      move_cell: list/tuple/2-by-3 array of (translate,rotate), rotate the cell followed by translating it
      scale: scaling factor of ECP magnitude
      min_distance: minimum distance allowed between segment and electrode, if specified
      sigma: resistivity of medium (mS/mm)
      move_elec: whether or not to relatively move electrodes for calculation
      """
      seg_coords = self.seg_coords
      if move_cell is not None:
          move_cell = np.asarray(move_cell).reshape((2, 3))
      if move_elec and move_cell is not None:
          elec_coords = move_position(move_cell[0], move_cell[1], self.elec_coords, True)
      else:
          elec_coords = self.elec_coords
      if not move_elec and move_cell is not None:
          dl = move_position([0., 0., 0.], move_cell[1], seg_coords['dl'])
          pc = move_position(move_cell[0], move_cell[1], seg_coords['pc'])
      else:
          dl = seg_coords['dl']
          pc = seg_coords['pc']
      if min_distance is None:
          r = seg_coords['r']
      else:
          r = np.fmax(seg_coords['r'], min_distance)
      rr = r ** 2
      
      tr = np.empty((self.nelec, self._nseg))
      for j in range(self.nelec):  # calculate mapping for each site on the electrode
          rel_pc = elec_coords[j, :] - pc  # distance between electrode and segment centers
          # compute dot product row-wise, the resulting array has as many rows as original
          r2 = np.einsum('ij,ij->i', rel_pc, rel_pc)
          rlldl = np.einsum('ij,ij->i', rel_pc, dl)
          dlmag = np.linalg.norm(dl, axis=1)  # length of each segment
          rll = abs(rlldl / dlmag)  # component of r parallel to the segment axis it must be always positive
          r_t2 = r2 - rll ** 2  # square of perpendicular component
          up = rll + dlmag / 2
          low = rll - dlmag / 2
          np.fmax(r_t2, rr, out=r_t2, where=low - r < 0)
          num = up + np.sqrt(up ** 2 + r_t2)
          den = low + np.sqrt(low ** 2 + r_t2)
          tr[j, :] = np.log(num / den) / dlmag  # units of (um) use with im_ (total seg current)
      tr *= scale / (4 * np.pi * sigma)
      return tr
  
  def calc_im(self, index: int = None) -> np.ndarray:
      """Calculate transmembrane current after simulation. Unit: nA."""
      index = slice(None) if index is None else index
      im = self.i_membrane[:, index].copy() #im = self.i_membrane.as_numpy(copy=False)[:, index].copy()
#      for inj in self.cell.injection:
#          im[inj.get_segment_id(), :] -= inj.rec_vec.as_numpy()[index]
      return im
  
  def calc_ecp(self, index: int = None, **kwargs) -> np.ndarray:
      """Calculate ECP after simulation. Unit: mV."""
      kwargs0 = {
                  'move_cell': self.move_cell,
                  'scale': self.scale,
                  'min_distance': self.min_distance,
                 }
      kwargs0.update(kwargs)
      tr = self.calc_transfer_resistance(**kwargs0)
      im = self.calc_im(index=index)
      return tr @ im
  
  def calc_ecps(self, move_cell: list = None, index: int = None, **kwargs) -> np.ndarray:
      """Calculate ECP with multiple positions after simulation. Unit: mV."""
      kwargs0 = {
                  'scale': self.scale,
                  'min_distance': self.min_distance,
                 }
      kwargs0.update(kwargs)
      if move_cell is None:
          move_cell = [self.move_cell]
      im = self.calc_im(index=index)
      ecp = []
      for mc in move_cell:
          ecp.append(self.calc_transfer_resistance(move_cell=mc, **kwargs0) @ im)
      return np.stack(ecp, axis=0)
  
  
  def move_position(translate: np.ndarray,
                    rotate: np.ndarray,
                    old_position: np.ndarray = None,
                    move_frame: bool = False) -> np.ndarray:
      """
      Rotate and translate an object with old_position and calculate its new coordinates.
      Rotate(alpha, h, phi): first rotate alpha about the y-axis (spin),
      then rotate arccos(h) about the x-axis (elevation),
      then rotate phi about the y-axis (azimuth).
      Finally translate the object by translate(x, y, z).
      If move_frame is True, use the object as reference frame and move the
      old reference frame, calculate new coordinates of the old_position.
      """
      translate = np.asarray(translate)
      if old_position is None:
          old_position = [0., 0., 0.]
      old_position = np.asarray(old_position)
      rot = Rotation.from_euler('yxy', [rotate[0], np.arccos(rotate[1]), rotate[2]])
      if move_frame:
          new_position = rot.inv().apply(old_position - translate)
      else:
          new_position = rot.apply(old_position) + translate
      return new_position