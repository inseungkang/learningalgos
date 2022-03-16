def c3d_marker_header_loc(filename):
# extract marker header location for each c3d files

  with open(filename, 'rb') as handle:
    reader = c3d.Reader(handle)
    markerLabel = reader.point_labels    
    markerName = ['RASI','LASI','RPSI','LPSI','RANL','LANL']

    idxList = []
    for _, name in enumerate(markerName):
      for idx, label in enumerate(markerLabel):
        if name == label.strip():
          idxList.append(idx)

  return idxList

def conv_gc_resample(input_data, stance_idx):
# Using stance index, segment data out to concat every gait cycle

  gait_cycle_data = np.empty((1000, len(stance_idx)-1, 1))
  
  # Each axis data resampled to 1000 datapoints
  for axis in np.arange(0, 3):
    axis_vec = np.empty((1000, 1))

    for idx in np.arange(len(stance_idx)-1):
      input_vec = input_data[stance_idx[idx,0]:stance_idx[idx+1,0], axis]
      out_vec = np.interp(
              np.linspace(0.0, 1.0, 1000, endpoint=False),
              np.linspace(0.0, 1.0, len(input_vec), endpoint=False),
              input_vec,)
      axis_vec = np.concatenate((axis_vec, np.expand_dims(out_vec, axis=1)), axis=1)

    axis_vec = axis_vec[:,1:]
    gait_cycle_data = np.concatenate((gait_cycle_data, np.expand_dims(axis_vec, axis=2)), axis=2)
  
  return gait_cycle_data[:, :, 1:]

def comput_stance_idx(force_right, force_left):
  stance_right = np.where(force_right < -100, 1, np.zeros(len(force_right)))
  stance_start_right = np.where(np.diff(stance_right) == 1)[0]

  stance_left = np.where(force_left < -100, 1, np.zeros(len(force_left)))
  stance_start_left = np.where(np.diff(stance_left) == 1)[0]

  # Check the stance start and end index to match total stance index array size

  if stance_start_right[0] > stance_start_left[0]:
    stance_start_left = stance_start_left[1:]

  if len(stance_start_right) < len(stance_start_left):
    stance_start_left = stance_start_left[:len(stance_start_right)]
  else:
    stance_start_right = stance_start_right[:len(stance_start_left)]

  # stack two stance index array together
  stance_idx = np.concatenate((stance_start_right.reshape(-1,1), stance_start_left.reshape(-1,1)), axis=1)

  return stance_idx  

def compute_SLA(stance_idx, fast_leg):
# computing SLA with a stance inx input (right and left heel contact)
# indicate fast leg (0 is right, 1 is left)
  SLA = np.empty([0,])
  for ii in np.arange(len(stance_idx)):
    step_r = foot_r[stance_idx[ii,0],1] - foot_l[stance_idx[ii,0],1]
    step_l = foot_l[stance_idx[ii,1],1] - foot_r[stance_idx[ii,1],1]

    if fast_leg == 0:
      current_sla = (step_r - step_l)/(step_r + step_l)
    else:
      current_sla = (step_l - step_r)/(step_r + step_l)

    SLA = np.append(SLA, current_sla)

  return SLA