import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import c3d
import pickle

import readFiles as RF
import controllerInferenceFunc as CIF
import utilityFunc as UF

from sklearn.linear_model import LinearRegression


file_dir = '/Users/inseungkang/Documents/learningalgos data/WangSrinivasanBiologyLetters_DryadDataUpload/'
subjectPool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
trialPool = [1, 2, 3]

controllerGainXRight = np.zeros((0, 5))
controllerGainYRight = np.zeros((0, 5))

for subject in subjectPool:
    GainXTrialRight = np.zeros((0, 5))
    GainYTrialRight = np.zeros((0, 5))
    for trial in trialPool:
        file_name = 'Subject'+str(subject)+'Trial'+str(trial)+'.txt'


        header = ['TorsoX', 'TorsoY', 'TorsoZ', 'LfootX' ,'LfootY', 'LfootZ' ,'RfootX', 'RfootY' ,'RfootZ']
        data = pd.read_csv(file_dir+file_name, delimiter=r"\s+", header= 0, names= header).to_numpy()

        pelvisPosition = data[:,0:3]
        footPositionL = data[:,3:6]
        footPositionR = data[:,6:9]

        # right heel contact and mid stance index based on COM Y position subtracted from right foot Y 
        footYPosRelative2PelvisL = footPositionL[:,1] - pelvisPosition[:,1]
        footYPosRelative2PelvisR = footPositionR[:,1] - pelvisPosition[:,1]
        midStanceIdxL = np.where(np.diff(np.sign(footYPosRelative2PelvisL)) < 0)[0]

        # Computing gait event index. The order of this index matrix has to be in MS_L, HC_R, MS_R, HC_L
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

        # Section to compute the pelvis position (in meter) relative to the correpsodning leg 
        pelvisXPosRelative2footR = (pelvisPosition[:,0] - footPositionR[:,0])
        pelvisYPosRelative2footR = (pelvisPosition[:,1] - footPositionR[:,1])
        pelvisZPosRelative2footR = (pelvisPosition[:,2] - footPositionR[:,2])
        pelvisXPosRelative2footL = (pelvisPosition[:,0] - footPositionL[:,0])
        pelvisYPosRelative2footL = (pelvisPosition[:,1] - footPositionL[:,1])
        pelvisZPosRelative2footL = (pelvisPosition[:,2] - footPositionL[:,2])
        
        # Section to compute the pelvis velocity (in m/s). Input marker data is sampled at 100 Hz, so divide by 0.01 
        pelvisXVelRelative2footR = np.append(np.diff(pelvisXPosRelative2footR), np.diff(pelvisXPosRelative2footR)[-1])/0.01
        pelvisYVelRelative2footR = np.append(np.diff(pelvisYPosRelative2footR), np.diff(pelvisYPosRelative2footR)[-1])/0.01
        pelvisZVelRelative2footR = np.append(np.diff(pelvisZPosRelative2footR), np.diff(pelvisZPosRelative2footR)[-1])/0.01
        pelvisXVelRelative2footL = np.append(np.diff(pelvisXPosRelative2footL), np.diff(pelvisXPosRelative2footL)[-1])/0.01
        pelvisYVelRelative2footL = np.append(np.diff(pelvisYPosRelative2footL), np.diff(pelvisYPosRelative2footL)[-1])/0.01
        pelvisZVelRelative2footL = np.append(np.diff(pelvisZPosRelative2footL), np.diff(pelvisZPosRelative2footL)[-1])/0.01

        # Concatenate pos and vel to make a pelvis state matrix
        pelvisStateR = np.transpose(np.asarray((pelvisXPosRelative2footR, pelvisZPosRelative2footR,
                                                pelvisXVelRelative2footR, pelvisYVelRelative2footR,
                                                pelvisZVelRelative2footR)))
        pelvisStateL = np.transpose(np.asarray((pelvisXPosRelative2footL, pelvisZPosRelative2footL,
                                                pelvisXVelRelative2footL, pelvisYVelRelative2footL,
                                                pelvisZVelRelative2footL)))

        footPlacementR = np.column_stack((footPositionR[:,0] - footPositionL[:,0], footPositionR[:,1] - footPositionL[:,1]))
        footPlacementL = np.column_stack((footPositionL[:,0] - footPositionR[:,0], footPositionL[:,1] - footPositionR[:,1]))

        # Gait event based pelvis state and foot placement value
        pelvisStateMidStanceL = pelvisStateL[[int(i) for i in gaitEventIdx[:,0].tolist()]]
        footPlacementHeelContactR = footPlacementR[[int(i) for i in gaitEventIdx[:,1].tolist()]]
        pelvisStateMidStanceR = pelvisStateR[[int(i) for i in gaitEventIdx[:,2].tolist()]]
        footPlacementHeelContactL = footPlacementL[[int(i) for i in gaitEventIdx[:,3].tolist()]]

        # Compute delta P and Q from the mean value of pelvis and foot placement during the trial
        deltaPelvisL = pelvisStateMidStanceR - np.mean(pelvisStateMidStanceR, axis=0)
        deltaPelvisR = pelvisStateMidStanceL - np.mean(pelvisStateMidStanceL, axis=0)
        deltaFootR = footPlacementHeelContactR - np.mean(footPlacementHeelContactR, axis=0)
        deltaFootL = footPlacementHeelContactL - np.mean(footPlacementHeelContactL, axis=0)

        r2_right_x, r2_right_y, gain_right_x, gain_right_y, linear_model_X, linear_model_Y = CIF.controller_fit(deltaPelvisR, deltaFootR)
        r2_left_x, r2_left_y, gain_left_x, gain_left_y, _, _ = CIF.controller_fit(deltaPelvisL, deltaFootL)

        GainXTrialRight = np.row_stack((GainXTrialRight, gain_right_x))
        GainYTrialRight = np.row_stack((GainYTrialRight, gain_right_y))

        from regressors import stats
        print(stats.coef_pval(linear_model_X, deltaPelvisR, deltaFootR[:,0]))

    GainXTrialRight = np.mean(GainXTrialRight, axis = 0)
    GainYTrialRight = np.mean(GainYTrialRight, axis = 0)

    controllerGainXRight = np.row_stack((controllerGainXRight, GainXTrialRight))
    controllerGainYRight = np.row_stack((controllerGainYRight, GainYTrialRight))

controllerGainXMeanRight = np.mean(controllerGainXRight, axis = 0)
controllerGainYMeanRight = np.mean(controllerGainYRight, axis = 0)



# import scipy as sp
# for ii in np.arange(5):
#     test_vec = controllerGainXRight[:,ii]
#     zero_vec = np.zeros(len(test_vec))
#     print(sp.stats.ttest_ind(test_vec, zero_vec, axis=0))
# for ii in np.arange(5):
#     test_vec = controllerGainYRight[:,ii]
#     zero_vec = np.zeros(len(test_vec))
#     print(sp.stats.ttest_ind(test_vec, zero_vec, axis=0))

reference = [0]


fig, axs = plt.subplots(1, 10, sharey=True)

for ii in np.arange(5):
    ci = 0.05 * np.std(controllerGainXRight[:,ii]) / np.mean(controllerGainXRight[:,ii])
    axs[ii].boxplot(controllerGainXRight[:,ii], showfliers=False)

for ii in np.arange(5):
    ci = 0.05 * np.std(controllerGainYRight[:,ii]) / np.mean(controllerGainYRight[:,ii])
    axs[ii+5].boxplot(controllerGainYRight[:,ii], showfliers=False)

for ii in np.arange(10):
    axs[ii].axhline(reference, c='k')

plt.ylim((-3, 3))
plt.show()