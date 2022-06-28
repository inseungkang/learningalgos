import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import c3d
import pickle

import readFiles as RF
import controllerInferenceFunc as CIF
import utilityFunc as UF

c3dFileDirectory = '/Users/inseungkang/Documents/learningalgos data/c3d files_tiedbelt/'
npzFileDirectory = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'
fileInformation = RF.extract_file_info(c3dFileDirectory)

for file in fileInformation:
        fileName = file[0]+'.npz'
        fileHeader = file[1]

        subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(fileName)

        if subject == 3:
                print(fileName)

                pelvisPosition, footPositionR, footPositionL, _, _ = RF.extractMarkerForceData(npzFileDirectory+fileName, fileHeader, 0, 6)
                gaitEventIdx = CIF.computeGaitEventIdx(pelvisPosition, footPositionR, footPositionL)
                pelvisStateMidStanceR, pelvisStateMidStanceL = CIF.computePelvisState(pelvisPosition, footPositionR, footPositionL, gaitEventIdx)
                footPlacementHeelContactR, footPlacementHeelContactL = CIF.computeFootPlacementLocation(footPositionR, footPositionL, gaitEventIdx)
                deltaPelvisR, deltaPelvisL, deltaFootR, deltaFootL = CIF.computeDeltaState(pelvisStateMidStanceR, pelvisStateMidStanceL, footPlacementHeelContactR, footPlacementHeelContactL)

                controllerGainXR, controllerGainYR, _, _, _, _ = CIF.controllerInference(deltaPelvisL, deltaFootR)
                controllerGainXL, controllerGainYL, _, _, _, _ = CIF.controllerInference(deltaPelvisR, deltaFootL)  

                plt.plot(controllerGainXR)
                plt.plot(controllerGainXL)
plt.show()
