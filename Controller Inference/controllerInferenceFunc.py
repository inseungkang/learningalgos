import numpy as np
import pandas as pd
import os
import fnmatch
import c3d
import pickle
from scipy import signal
from sklearn.linear_model import LinearRegression
import utilityFunc as UF
import readFiles as RF

def computeGaitEventIdx(pelvisPosition, footPositionR, footPositionL):
    # Left midstance index can be computed by finding the zero crossing point based on foot Y position relative to pelvis Y position.
    footYPosRelative2PelvisL = footPositionL[:,1] - pelvisPosition[:,1]
    footYPosRelative2PelvisR = footPositionR[:,1] - pelvisPosition[:,1]
    midStanceIdxL = np.where(np.diff(np.sign(footYPosRelative2PelvisL)) < 0)[0]

    # Gait event index can be computed in the sequence of MS_L, HC_R, MS_R, HC_L.
    gaitEventIdx = np.zeros((0,4))
    for ii in np.arange(len(midStanceIdxL)-1):
        startIdx = midStanceIdxL[ii]
        endIdx = midStanceIdxL[ii+1]

        heelContactIdxR = startIdx + np.argmax(footYPosRelative2PelvisR[startIdx:endIdx])
        midStanceIdxR = startIdx + np.where(np.diff(np.sign(footYPosRelative2PelvisR[startIdx:endIdx])) < 0)[0][0]
        heelContactIdxL = startIdx + np.argmax(footYPosRelative2PelvisL[startIdx:endIdx])

        singleGaitCycleIdx = np.array([midStanceIdxL[ii], heelContactIdxR, midStanceIdxR, heelContactIdxL])
        gaitEventIdx = np.row_stack((gaitEventIdx, singleGaitCycleIdx))
    gaitEventIdx = gaitEventIdx.astype(int)
    return gaitEventIdx

def computePelvisState(pelvisPosition, footPositionR, footPositionL, gaitEventIdx):
    # Pelvis position needs to be computed relative to the correpsodning leg's position for all X, Y, and Z. Units are assumed to be in m.
    pelvisXPosRelative2footR = (pelvisPosition[:,0] - footPositionR[:,0])
    pelvisYPosRelative2footR = (pelvisPosition[:,1] - footPositionR[:,1])
    pelvisZPosRelative2footR = (pelvisPosition[:,2] - footPositionR[:,2])
    pelvisXPosRelative2footL = (pelvisPosition[:,0] - footPositionL[:,0])
    pelvisYPosRelative2footL = (pelvisPosition[:,1] - footPositionL[:,1])
    pelvisZPosRelative2footL = (pelvisPosition[:,2] - footPositionL[:,2])
    
    # Differente the pelvis position to compute velocity. Mocap marker sampled at 100 Hz and units are assumed to be m/s.
    pelvisXVelRelative2footR = np.append(np.diff(pelvisXPosRelative2footR), np.diff(pelvisXPosRelative2footR)[-1])/0.01
    pelvisYVelRelative2footR = np.append(np.diff(pelvisYPosRelative2footR), np.diff(pelvisYPosRelative2footR)[-1])/0.01
    pelvisZVelRelative2footR = np.append(np.diff(pelvisZPosRelative2footR), np.diff(pelvisZPosRelative2footR)[-1])/0.01
    pelvisXVelRelative2footL = np.append(np.diff(pelvisXPosRelative2footL), np.diff(pelvisXPosRelative2footL)[-1])/0.01
    pelvisYVelRelative2footL = np.append(np.diff(pelvisYPosRelative2footL), np.diff(pelvisYPosRelative2footL)[-1])/0.01
    pelvisZVelRelative2footL = np.append(np.diff(pelvisZPosRelative2footL), np.diff(pelvisZPosRelative2footL)[-1])/0.01

    # Concatenate pelvis position and velocity to make a pelvis state matrix.
    pelvisStateR = np.transpose(np.asarray((pelvisXPosRelative2footR, pelvisZPosRelative2footR,
                                            pelvisXVelRelative2footR, pelvisYVelRelative2footR,
                                            pelvisZVelRelative2footR)))
    pelvisStateL = np.transpose(np.asarray((pelvisXPosRelative2footL, pelvisZPosRelative2footL,
                                            pelvisXVelRelative2footL, pelvisYVelRelative2footL,
                                            pelvisZVelRelative2footL)))

    # Compute pelvis state during mid-stance of the gait cycle.
    pelvisStateMidStanceR = pelvisStateR[[int(i) for i in gaitEventIdx[:,2].tolist()]]
    pelvisStateMidStanceL = pelvisStateL[[int(i) for i in gaitEventIdx[:,0].tolist()]]

    return pelvisStateMidStanceR, pelvisStateMidStanceL

def computeFootPlacementLocation(footPositionR, footPositionL, gaitEventIdx):
    # Foot placement location is all relative to the stance leg (origin).
    footPlacementR = np.column_stack((footPositionR[:,0] - footPositionL[:,0], footPositionR[:,1] - footPositionL[:,1]))
    footPlacementL = np.column_stack((footPositionL[:,0] - footPositionR[:,0], footPositionL[:,1] - footPositionR[:,1]))

    # Using the gait event index matrix, X and Y foot placement location can be extract during heel contact.
    footPlacementHeelContactR = footPlacementR[[int(i) for i in gaitEventIdx[:,1].tolist()]]
    footPlacementHeelContactL = footPlacementL[[int(i) for i in gaitEventIdx[:,3].tolist()]]

    return footPlacementHeelContactR, footPlacementHeelContactL

def computeDeltaState(pelvisStateMidStanceR, pelvisStateMidStanceL, footPlacementHeelContactR, footPlacementHeelContactL):
    # Compute delta P and Q from the mean value of pelvis and foot placement during each trial (works for tied-belt conditions).
    deltaPelvisR = pelvisStateMidStanceR - np.mean(pelvisStateMidStanceR, axis=0)
    deltaPelvisL = pelvisStateMidStanceL - np.mean(pelvisStateMidStanceL, axis=0)
    deltaFootR = footPlacementHeelContactR - np.mean(footPlacementHeelContactR, axis=0)
    deltaFootL = footPlacementHeelContactL - np.mean(footPlacementHeelContactL, axis=0)

    return deltaPelvisR, deltaPelvisL, deltaFootR, deltaFootL

def controllerInference(deltaPelvis, deltaFoot):
    linearModelX = LinearRegression().fit(deltaPelvis, deltaFoot[:,0])
    linearModelY = LinearRegression().fit(deltaPelvis, deltaFoot[:,1])
    modelScoreX = linearModelX.score(deltaPelvis, deltaFoot[:,0])
    modelScoreY = linearModelY.score(deltaPelvis, deltaFoot[:,1])
    controllerGainX = np.around(linearModelX.coef_, 3)
    controllerGainY = np.around(linearModelY.coef_, 3)

    return controllerGainX, controllerGainY, modelScoreX, modelScoreY, linearModelX, linearModelY







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

def detrend_data_constant(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l):
    delta_P_r = com_ms_l - np.mean(com_ms_l)
    delta_P_l = com_ms_r - np.mean(com_ms_r)
    delta_Q_r = foot_hc_r - np.mean(foot_hc_r)
    delta_Q_l = foot_hc_l - np.mean(foot_hc_l)

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

    return X_score, Y_score, X_gain, Y_gain, linear_model_X, linear_model_Y

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
  return int(out)


def computeDeltaPandQ(file_dir, file_header, compute_leg, start_time, stop_time):
    com, foot_r, foot_l, force_r, force_l = RF.extract_data(file_dir, file_header, start_time, stop_time)
    gait_event_idx = compute_gait_event(com, foot_r, foot_l)
    com_cont_r, com_cont_l = compute_continous_com_state(com, foot_r, foot_l)
    com_ms_r, com_ms_l = compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
    foot_hc_r, foot_hc_l = compute_foot_placement(foot_r, foot_l, gait_event_idx)
    delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)

    if compute_leg == 'right':
        delta_P = delta_P_r
        delta_Q = delta_Q_r
    elif compute_leg == 'left':
        delta_P = delta_P_l
        delta_Q = delta_Q_l
        
    return delta_P, delta_Q, gait_event_idx

def GRF_based_gait_event_idx(com, foot_r, foot_l, force_r, force_l):
    '''
    Compute gait event index number (left and right). This will be in the order of left mid stance, right heel contact, right mid stance, and left heel contact
    This will be used to compute the com state during mid stance and evaluate the foot placement in the following heel contact
    here, using the com y position and foot y position, compute the mid stance (zero crossing) and heel contact (min value) from the signal: diff(com, foot)
    this is because com y location will be negative when the foot y is front of com and vice versa
    '''
    z_vec = np.abs(force_r[:,2])
    gait_event = np.zeros(len(z_vec))
    gait_event[np.where(z_vec > 200)] = 1
    gait_event_diff = np.diff(gait_event)
    hc_idx_r = np.where(gait_event_diff == 1)[0]

    ms_idx_r = []
    for ii in np.arange(len(hc_idx_r)-1):
            relative_com_pos_y = foot_r[hc_idx_r[ii]:hc_idx_r[ii+1],1] - com[hc_idx_r[ii]:hc_idx_r[ii+1],1]
            zero_crossing = np.where(np.diff(np.sign(relative_com_pos_y)) < 0)[0][0]       #first zc corrpesonding to mid-stance
            ms_idx_r.append(zero_crossing+hc_idx_r[ii])
    ms_idx_r = np.asarray(ms_idx_r)

    z_vec = np.abs(force_l[:,2])
    gait_event = np.zeros(len(z_vec))
    gait_event[np.where(z_vec > 200)] = 1
    gait_event_diff = np.diff(gait_event)
    hc_idx_l = np.where(gait_event_diff == 1)[0]

    ms_idx_l = []
    for ii in np.arange(len(hc_idx_l)-1):
            relative_com_pos_y = foot_l[hc_idx_l[ii]:hc_idx_l[ii+1],1] - com[hc_idx_l[ii]:hc_idx_l[ii+1],1]
            zero_crossing = np.where(np.diff(np.sign(relative_com_pos_y)) < 0)[0][0]       #first zc corrpesonding to mid-stance
            ms_idx_l.append(zero_crossing+hc_idx_l[ii])
    ms_idx_l = np.asarray(ms_idx_l)

    gait_event_idx = np.empty((0,4))
    # order of the gait event index will be ms_l, hc_r, ms_r, hc_l
    for ii in np.arange(len(hc_idx_r)-2):       # May need to revisit later to take care of size mismatch on the last stride
            if ii < len(hc_idx_l)-2:
                    empty_vec = np.empty((4,))
                    empty_vec[1] = hc_idx_r[ii]

                    relative_com_pos_y = foot_r[hc_idx_r[ii]:hc_idx_r[ii+1],1] - com[hc_idx_r[ii]:hc_idx_r[ii+1],1]
                    zero_crossing = np.where(np.diff(np.sign(relative_com_pos_y)) > 0)[0][0]       #first zc corrpesonding to mid-stance
                    empty_vec[2] = hc_idx_r[ii] + zero_crossing

                    next_hc = hc_idx_l[hc_idx_l > hc_idx_r[ii]].min()
                    empty_vec[3] = next_hc

                    next_hc_ii = np.where(hc_idx_l == next_hc)[0][0]
                    relative_com_pos_y = foot_l[hc_idx_l[next_hc_ii]:hc_idx_l[next_hc_ii+1],1] - com[hc_idx_l[next_hc_ii]:hc_idx_l[next_hc_ii+1],1] 
                    zero_crossing = np.where(np.diff(np.sign(relative_com_pos_y)) > 0)[0][0]       #first zc corrpesonding to mid-stance
                    empty_vec[0] = next_hc + zero_crossing
                    
                    gait_event_idx = np.row_stack((gait_event_idx, empty_vec))

    new_ms_idx_l = []

    for ii in np.arange(len(hc_idx_l)-1):
            index_vec = hc_idx_l[ii+1] - hc_idx_l[ii]
            new_ms_idx_l.append(int(index_vec*0.7)+hc_idx_l[ii])
    new_ms_idx_l = np.asarray(new_ms_idx_l)

    return gait_event_idx, ms_idx_r, ms_idx_l, new_ms_idx_l