import procFunc as pf
import numpy as np
import pandas as pd
import os
import fnmatch
import c3d
import matplotlib.pyplot as plt
import pickle

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def calc_slope(x):
    slope = np.polyfit(range(len(x)), x, 1)[0]
    return slope

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

class subjectData:
        def __init__(self, subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg):
            self.subjectNum = subjectNum
            self.delta = delta
            self.session = session
            self.com = com
            self.foot_r = foot_r
            self.foot_l = foot_l
            self.force_r = force_r
            self.force_l = force_l
            self.fast_leg = fast_leg

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

def extract_from_c3d(file_name, index_list, trial_min):
    b = np.load(file_name)
    markerData = b['points']
    forceData = b['analog']

    force_sync = np.mean(forceData[:,14,:], axis=1)

    # main adaptation trial started after 12 min of constant walking
    sync_start = np.argwhere(abs(force_sync) > np.max(abs(force_sync))/2)[0,0]
    start_idx = sync_start + 72000
    end_idx = start_idx + trial_min*6000

    foot_r = markerData[start_idx:end_idx, index_list[4], 0:3]
    foot_l = markerData[start_idx:end_idx, index_list[5], 0:3]
    com = np.mean(markerData[start_idx:end_idx, index_list[0:4], 0:3], axis=1)
    force_r = np.mean(forceData[start_idx:end_idx, 0:3, :], axis=2)
    force_l = np.mean(forceData[start_idx:end_idx, 6:9, :], axis=2)

    save_name = file_name.split('.npz')[0]+'_synced.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump([com, foot_r, foot_l, force_r, force_l], f)

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

def compute_stance_idx(force_right, force_left):
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

def compute_SLA(stance_idx, foot_r, foot_l, fast_leg):
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

def compute_foot_placement(stance_idx, foot_r, foot_l, com):
  foot_place_mag_r = np.empty([0,])
  foot_place_mag_l = np.empty([0,])

  for ii in np.arange(len(stance_idx))-1:
      foot_place_r_x = foot_r[stance_idx[ii,0],0] - com[stance_idx[ii,0],0]
      foot_place_r_y = foot_r[stance_idx[ii,0],1] - com[stance_idx[ii,0],1]
      foot_place_r_mag_1 = np.sqrt(foot_place_r_x**2 + foot_place_r_y**2)

      foot_place_r_x = foot_r[stance_idx[ii+1,0],0] - com[stance_idx[ii+1,0],0]
      foot_place_r_y = foot_r[stance_idx[ii+1,0],1] - com[stance_idx[ii+1,0],1]
      foot_place_r_mag_2 = np.sqrt(foot_place_r_x**2 + foot_place_r_y**2)

      foot_place_r_mag_delta = foot_place_r_mag_2 - foot_place_r_mag_1
      foot_place_mag_r = np.append(foot_place_mag_r, foot_place_r_mag_delta)
  ##
      foot_place_l_x = foot_l[stance_idx[ii,1],0] - com[stance_idx[ii,1],0]
      foot_place_l_y = foot_l[stance_idx[ii,1],1] - com[stance_idx[ii,1],1]
      foot_place_l_mag_1 = np.sqrt(foot_place_l_x**2 + foot_place_l_y**2)

      foot_place_l_x = foot_l[stance_idx[ii+1,1],0] - com[stance_idx[ii+1,1],0]
      foot_place_l_y = foot_l[stance_idx[ii+1,1],1] - com[stance_idx[ii+1,1],1]
      foot_place_l_mag_2 = np.sqrt(foot_place_l_x**2 + foot_place_l_y**2)

      foot_place_l_mag_delta = foot_place_l_mag_2 - foot_place_l_mag_1
      foot_place_mag_l = np.append(foot_place_mag_l, foot_place_l_mag_delta)

  return foot_place_mag_r, foot_place_mag_l

def moving_std(x, w):
  return np.convolve(x, np.ones(w), 'valid') / w

def compute_foot_placement_y(stance_idx, foot_r, foot_l, com):

  foot_place_mag_r = np.empty([0,])
  foot_place_mag_l = np.empty([0,])

  for ii in np.arange(len(stance_idx))-1:
      foot_place_r_y1 = foot_r[stance_idx[ii,0],1] - com[stance_idx[ii,0],1]
      foot_place_r_y2 = foot_r[stance_idx[ii+1,0],1] - com[stance_idx[ii+1,0],1]
      foot_place_r_y = foot_place_r_y2 - foot_place_r_y1

      foot_place_l_y1 = foot_l[stance_idx[ii,1],1] - com[stance_idx[ii,1],1]
      foot_place_l_y2 = foot_l[stance_idx[ii+1,1],1] - com[stance_idx[ii+1,1],1]
      foot_place_l_y = foot_place_l_y2 - foot_place_l_y1

      foot_place_mag_r = np.append(foot_place_mag_r, foot_place_r_y)
      foot_place_mag_l = np.append(foot_place_mag_l, foot_place_l_y)

  return foot_place_mag_r, foot_place_mag_l

def load_synced_data():
  data_dir = '/Users/inseungkang/Documents/learningalgos data/synced data/'
  file_list = [i for i in os.listdir(data_dir) if '.pkl' in i]

  for file_name in file_list:
      subjectNum = int(file_name.split('_Session')[0].split('AB')[1])
      session = int(file_name.split('Session')[1].split('_Right')[0])
      rSpeed = int(file_name.split('Right')[1].split('_Left')[0])
      lSpeed = int(file_name.split('Left')[1].split('_synced')[0])

      if rSpeed == 10:
          delta = int((lSpeed - rSpeed)/2)
      else:
          delta = int((rSpeed - lSpeed)/2)

      delta = delta + 2
      
      if rSpeed > lSpeed:
          fast_leg = 0
      else:
          fast_leg = 1

      with open(data_dir+file_name, 'rb') as handle:
          com, foot_r, foot_l, force_r, force_l = pickle.load(handle)

          if subjectNum == 1:
              if delta == 0:
                  AB1_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 1:
                  AB1_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 3:
                  AB1_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 4:
                  AB1_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

          if subjectNum == 2:
              if delta == 0:
                  AB2_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 1:
                  AB2_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 3:
                  AB2_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 4:
                  AB2_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

          if subjectNum == 3:
              if delta == 0:
                  AB3_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 1:
                  AB3_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 3:
                  AB3_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 4:
                  AB3_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

          if subjectNum == 4:
              if delta == 0:
                  AB4_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 1:
                  AB4_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 3:
                  AB4_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 4:
                  AB4_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

          if subjectNum == 5:
              if delta == 0:
                  AB5_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 1:
                  AB5_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 3:
                  AB5_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
              if delta == 4:
                  AB5_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
  result = [AB1_delta0, AB1_delta1, AB1_delta2, AB1_delta3,
          AB2_delta0, AB2_delta1, AB2_delta2, AB2_delta3,
          AB3_delta0, AB3_delta1, AB3_delta2, AB3_delta3,
          AB4_delta0, AB4_delta1, AB4_delta2, AB4_delta3,
          AB5_delta0, AB5_delta1, AB5_delta2, AB5_delta3]

  return result