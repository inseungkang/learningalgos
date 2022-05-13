import numpy as np
import pandas as pd
import os
import fnmatch
import c3d
import pickle
from scipy import signal
from sklearn.linear_model import LinearRegression
import utilityFunc as UF

def compute_gait_event(com, foot_r, foot_l):
    '''
    Compute gait event index number (left and right). This will be in the order of left mid stance, right heel contact, right mid stance, and left heel contact
    This will be used to compute the com state during mid stance and evaluate the foot placement in the following heel contact
    here, using the com y position and foot y position, compute the mid stance (zero crossing) and heel contact (min value) from the signal: diff(com, foot)
    this is because com y location will be negative when the foot y is front of com and vice versa
    '''
    # left heel contact and mid stance index based on COM Y position subtracted from left foot Y 
    COM_Y_l = com[:,1] - foot_l[:,1]
    zc = np.where(np.diff(np.sign(COM_Y_l)))[0]

    hc_l = []
    ms_l = []
    for idx, value in enumerate(zc):
        if COM_Y_l[value+1] < 0:
            if idx < len(zc)-1:
                min_point = np.argmin(COM_Y_l[zc[idx]:zc[idx+1]])
                hc_idx = value+min_point
                ms_idx = zc[idx+1]
                hc_l.append(hc_idx)
                ms_l.append(ms_idx)

    # right heel contact and mid stance index based on COM Y position subtracted from right foot Y 
    COM_Y_r = com[:,1] - foot_r[:,1]
    zc = np.where(np.diff(np.sign(COM_Y_r)))[0]

    hc_r = []
    ms_r = []
    for idx, value in enumerate(zc):
        if COM_Y_r[value+1] < 0:
            if idx < len(zc)-1:
                min_point = np.argmin(COM_Y_r[zc[idx]:zc[idx+1]])
                hc_idx = value+min_point
                ms_idx = zc[idx+1]
                hc_r.append(hc_idx)
                ms_r.append(ms_idx)

    # mid stance and heel contact index for left and right might match in size so truncate the first on to match the MX4 size and concatenate
    # order of the gait event every loop is: ms_l, hc_r, ms_r, hc_l
    gait_event_idx = np.zeros((0,4))
    for idx, x in enumerate(ms_l):

        if idx+1 > len(hc_r)-1 or idx+1 > len(ms_r)-1 or idx+1 > len(hc_l)-1:
            break
        a = x

        if a < hc_r[idx]:
            b = hc_r[idx]
        else:
            b = hc_r[idx+1]

        if b < ms_r[idx]:
            c = ms_r[idx]
        else:
            c = ms_r[idx+1]

        if c < hc_l[idx]:
            d = hc_l[idx]
        else:
            d = hc_l[idx+1]
    
        gait_vec = np.array([int(a), int(b), int(c), int(d)])
        gait_event_idx = np.row_stack((gait_event_idx, gait_vec))
    return gait_event_idx

# Obtain delta P and Q using the COM state and foot XY position
def compute_pelvis_foot_state_const(com, foot_r, foot_l, gait_event_idx):

    COM_X_r = com[:,0] - foot_r[:,0]
    COM_Y_r = com[:,1] - foot_r[:,1]
    COM_Z_r = com[:,2] - foot_r[:,2]
    COM_X_l = com[:,0] - foot_l[:,0]
    COM_Y_l = com[:,1] - foot_l[:,1]
    COM_Z_l = com[:,2] - foot_l[:,2]

    COM_XX_r = np.append(np.diff(COM_X_r), np.diff(COM_X_r)[-1])
    COM_YY_r = np.append(np.diff(COM_Y_r), np.diff(COM_Y_r)[-1])
    COM_ZZ_r = np.append(np.diff(COM_Z_r), np.diff(COM_Z_r)[-1])
    COM_XX_l = np.append(np.diff(COM_X_l), np.diff(COM_X_l)[-1])
    COM_YY_l = np.append(np.diff(COM_Y_l), np.diff(COM_Y_l)[-1])
    COM_ZZ_l = np.append(np.diff(COM_Z_l), np.diff(COM_Z_l)[-1])

    b, a = signal.butter(2, 0.125)
    COM_XX_r = signal.filtfilt(b, a, COM_XX_r)
    COM_YY_r = signal.filtfilt(b, a, COM_YY_r)
    COM_ZZ_r = signal.filtfilt(b, a, COM_ZZ_r)
    COM_XX_l = signal.filtfilt(b, a, COM_XX_l)
    COM_YY_l = signal.filtfilt(b, a, COM_YY_l)
    COM_ZZ_l = signal.filtfilt(b, a, COM_ZZ_l)

    COM_state_r = np.transpose(np.asarray((COM_X_r, COM_Z_r, COM_XX_r, COM_YY_r, COM_ZZ_r)))
    COM_state_l = np.transpose(np.asarray((COM_X_l, COM_Z_l, COM_XX_l, COM_YY_l, COM_ZZ_l)))

    # Obtain foot placement vector (X, Y). This is only the position vector but we need to subtract the relevant COM position to obtain foot place relative to the COM
    footXY_l = np.column_stack((foot_l[:,0] - foot_r[:,0], foot_l[:,1] - foot_r[:,1]))
    footXY_r = np.column_stack((foot_r[:,0] - foot_l[:,0], foot_r[:,1] - foot_l[:,1]))

    # Obtain COM state and foot placement at mid stance / heel contact for left and right leg using the corresponding gait event index
    COM_ms_l = COM_state_l[[int(i) for i in gait_event_idx[:,0].tolist()]]
    COM_ms_r = COM_state_r[[int(i) for i in gait_event_idx[:,2].tolist()]]
    footXY_hc_l = footXY_l[[int(i) for i in gait_event_idx[:,3].tolist()]]
    footXY_hc_r = footXY_r[[int(i) for i in gait_event_idx[:,1].tolist()]]

    # compute the nominal P_star and Q_star which is just a mean value during certain part of the gait cycle (mid stance / heel contact)
    COM_ms_norm_l = np.mean(COM_ms_l, axis=0)
    COM_ms_norm_r = np.mean(COM_ms_r, axis=0)
    footXY_hc_norm_l = np.mean(footXY_hc_l, axis=0)
    footXY_hc_norm_r = np.mean(footXY_hc_r, axis=0)

    delta_P_r = COM_ms_r - COM_ms_norm_r
    delta_P_l = COM_ms_l - COM_ms_norm_l
    delta_Q_r = footXY_hc_l - footXY_hc_norm_l
    delta_Q_l = footXY_hc_r - footXY_hc_norm_r

    return delta_P_r, delta_P_l, delta_Q_r, delta_Q_l

def compute_continous_com_state(com, foot_r, foot_l):
    COM_X_r = com[:,0] - foot_r[:,0]
    COM_Y_r = com[:,1] - foot_r[:,1]
    COM_Z_r = com[:,2] - foot_r[:,2]
    COM_X_l = com[:,0] - foot_l[:,0]
    COM_Y_l = com[:,1] - foot_l[:,1]
    COM_Z_l = com[:,2] - foot_l[:,2]

    COM_XX_r = np.append(np.diff(COM_X_r), np.diff(COM_X_r)[-1])
    COM_YY_r = np.append(np.diff(COM_Y_r), np.diff(COM_Y_r)[-1])
    COM_ZZ_r = np.append(np.diff(COM_Z_r), np.diff(COM_Z_r)[-1])
    COM_XX_l = np.append(np.diff(COM_X_l), np.diff(COM_X_l)[-1])
    COM_YY_l = np.append(np.diff(COM_Y_l), np.diff(COM_Y_l)[-1])
    COM_ZZ_l = np.append(np.diff(COM_Z_l), np.diff(COM_Z_l)[-1])

    b, a = signal.butter(2, 0.125)
    COM_XX_r = signal.filtfilt(b, a, COM_XX_r)
    COM_YY_r = signal.filtfilt(b, a, COM_YY_r)
    COM_ZZ_r = signal.filtfilt(b, a, COM_ZZ_r)
    COM_XX_l = signal.filtfilt(b, a, COM_XX_l)
    COM_YY_l = signal.filtfilt(b, a, COM_YY_l)
    COM_ZZ_l = signal.filtfilt(b, a, COM_ZZ_l)

    COM_state_r = np.transpose(np.asarray((COM_X_r, COM_Z_r, COM_XX_r, COM_YY_r, COM_ZZ_r)))
    COM_state_l = np.transpose(np.asarray((COM_X_l, COM_Z_l, COM_XX_l, COM_YY_l, COM_ZZ_l)))

    return COM_state_r, COM_state_l

def compute_com_state_mid_stance(COM_state_r, COM_state_l, gait_event_idx):
    # Obtain COM state and foot placement at mid stance / heel contact for left and right leg using the corresponding gait event index
    COM_ms_r = COM_state_r[[int(i) for i in gait_event_idx[:,2].tolist()]]
    COM_ms_l = COM_state_l[[int(i) for i in gait_event_idx[:,0].tolist()]]
    return COM_ms_r, COM_ms_l

def compute_foot_placement(foot_r, foot_l, gait_event_idx):
    # Obtain foot placement during heel contact (X, Y).
    # Note that we are subtracting the contra leg because foot placement location is X, Y position relative to the origin leg (stance phase)
    footXY_r = np.column_stack((foot_r[:,0] - foot_l[:,0], foot_r[:,1] - foot_l[:,1]))
    footXY_l = np.column_stack((foot_l[:,0] - foot_r[:,0], foot_l[:,1] - foot_r[:,1]))

    # Using the corresponding gait event index, we can extract the X and Y position during a certain gait event
    footXY_hc_r = footXY_r[[int(i) for i in gait_event_idx[:,1].tolist()]]
    footXY_hc_l = footXY_l[[int(i) for i in gait_event_idx[:,3].tolist()]]
    return footXY_hc_r, footXY_hc_l

def find_evaluting_range_idx_time(time_start, range, gait_event_idx):
    range = range*6000
    start = time_start*6000
    end = start + range
    start = np.argmin(np.abs(gait_event_idx[:,0] - start))
    end = np.argmin(np.abs(gait_event_idx[:,0] - end))

    return start, end

def find_evaluting_range_idx_segment(start, segment_num, gait_event_idx):
    size = gait_event_idx.shape[0]
    start = int(size/segment_num)*start
    end = int(size/segment_num)*start + int(size/segment_num)

    return start, end

def detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l):
    delta_P_r = com_ms_l - UF.low_pass_filter(com_ms_l)
    delta_P_l = com_ms_r - UF.low_pass_filter(com_ms_r)
    delta_Q_r = foot_hc_r - UF.low_pass_filter(foot_hc_r)
    delta_Q_l = foot_hc_l - UF.low_pass_filter(foot_hc_l)

    return delta_P_r, delta_P_l, delta_Q_r, delta_Q_l

def detrend_data_detrend(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l):
    delta_P_r = UF.linear_detrend(com_ms_l, 5)
    delta_P_l = UF.linear_detrend(com_ms_r, 5)
    delta_Q_r = UF.linear_detrend(foot_hc_r, 5)
    delta_Q_l = UF.linear_detrend(foot_hc_l, 5)

    return delta_P_r, delta_P_l, delta_Q_r, delta_Q_l

def detrend_data_moveave(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l):
    delta_P_r = UF.moving_average(com_ms_l, 50)
    delta_P_l = UF.moving_average(com_ms_r, 50)
    delta_Q_r = UF.moving_average(foot_hc_r, 50)
    delta_Q_l = UF.moving_average(foot_hc_l, 50)

    return delta_P_r, delta_P_l, delta_Q_r, delta_Q_l

def controller_fit(delta_P, delta_Q):
    linear_model_X = LinearRegression().fit(delta_P, delta_Q[:,0])
    linear_model_Y = LinearRegression().fit(delta_P, delta_Q[:,1])
    X_score = linear_model_X.score(delta_P, delta_Q[:,0])
    Y_score = linear_model_Y.score(delta_P, delta_Q[:,1])
    X_gain = np.around(linear_model_X.coef_, 2)
    Y_gain = np.around(linear_model_Y.coef_, 2)

    return X_score, Y_score, X_gain, Y_gain

def extract_trial_info(file_name):

  subject = int((file_name.split('AB'))[1].split('_')[0])
  rSpeed = int((file_name.split('Right'))[1].split('_')[0])
  lSpeed = int((file_name.split('Left'))[1].split('.')[0])

  if rSpeed > lSpeed:
    fast_leg = 'right'
    fSpeed = rSpeed
    sSpeed = lSpeed
  else:
    fast_leg = 'left'
    fSpeed = lSpeed
    sSpeed = rSpeed

  if rSpeed == 10:
    delta = lSpeed - rSpeed
  else:
    delta = rSpeed - lSpeed

  return subject, delta, fast_leg, fSpeed, sSpeed

def delta_to_idx(delta):
  if delta == -4:
    out = 0
  elif delta == -2:
    out = 1
  elif delta == 2:
    out = 2
  elif delta == 4:
    out = 3
  return out