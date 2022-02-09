import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import myFunc
import matplotlib.pyplot as plt

current_path = os.getcwd() + "/learningalgos/Metabolic Cost/Data/"
start = 720
end = 3420
mass = 60

subjectID = ["AB3"]
sessionNum = ["2"]

def computeMetabolics(filepath, mass, start_time, end_time):
    data = pd.read_csv(filepath, header=None)

    timeStamp = data.loc[:,0].str.split(":")
    timeConv = np.zeros(len(timeStamp))
    for idx in range(len(timeStamp)):
        if len(timeStamp[idx]) == 2:
            timeConv[idx] = int(timeStamp[idx][0]) * 60 + int(timeStamp[idx][1])
        else:
            if int(timeStamp[idx][0]) != 1:
                timeConv[idx] = int(timeStamp[idx][0]) * 60 + int(timeStamp[idx][1])
            else:
                timeConv[idx] = int(timeStamp[idx][0]) * 3600 + int(timeStamp[idx][1]) * 60 + int(timeStamp[idx][2])

    # print(int(timeStamp[idx][0]) * 3600 + int(timeStamp[idx][1]) * 60 + int(timeStamp[idx][2]))
    # print(int(timeStamp[idx][0]) * 3600)
    # print(int(timeStamp[idx][1]) * 60)
    # print(int(timeStamp[idx][2]))
    
    # plt.plot(timeConv)
    # plt.show()

    timeConv = timeConv - timeConv[0]
    print(timeConv)
    adapStart_idx = np.abs(timeConv - start_time).argmin()
    print(adapStart_idx)
    if timeConv[adapStart_idx] < start_time:
        adapStart_idx = adapStart_idx + 1
        print(adapStart_idx)
    adapEnd_idx = np.abs(timeConv - end_time).argmin()
    if timeConv[adapEnd_idx] > end_time:
        adapEnd_idx = adapStart_idx - 1

    print(adapStart_idx)
    # timeConv = timeConv[adapStart_idx:adapEnd_idx].astype(int)-start_time

    # VO2 = data.iloc[adapStart_idx:adapEnd_idx,4].to_numpy()
    # VCO2 = data.iloc[adapStart_idx:adapEnd_idx,6].to_numpy()
    # metabolics = (0.278*VO2 + 0.075*VCO2)/mass

    # newData = np.zeros(end_time-start_time)
    # # print(metabolics.shape)
    # # print(timeConv.shape)
    # # print(timeConv[-1])
    # # plt.plot(timeConv)
    # # plt.show()
    # for idx, xx in enumerate(metabolics):
    #     newData[timeConv[idx]] = xx

    # for idx, xx in enumerate(newData):
    #     if xx < 0.1:
    #         newData[idx] = newData[idx-1]

    # return newData


for subject in subjectID:
    for session in sessionNum:
        filename = subject + "_Metabolics_Session" + session

        for file in os.listdir(current_path):
            if file.startswith(filename):

                data = computeMetabolics(current_path+file, mass, start, end)




