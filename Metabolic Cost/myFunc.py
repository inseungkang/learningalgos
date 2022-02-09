import numpy as np
import pandas as pd

def computeMetabolics(filepath, mass, start_time, end_time):
    data = pd.read_csv(filepath, header=None)

    timeStamp = data.loc[:,0].str.split(":")
    timeConv = np.zeros(len(timeStamp))
    for idx in range(len(timeStamp)):
        if len(timeStamp[idx]) == 2:
            timeConv[idx] = int(timeStamp[idx][0]) * 60 + int(timeStamp[idx][1])
        else:
            if int(timeStamp[idx][0]) > 1:
                timeConv[idx] = int(timeStamp[idx][0]) * 60 + int(timeStamp[idx][1])
            else:
                timeConv[idx] = int(timeStamp[idx][0]) * 3600 + int(timeStamp[idx][1]) * 60 + int(timeStamp[idx][2])

    timeConv = timeConv - timeConv[0]

    adapStart_idx = np.abs(timeConv - start_time).argmin()
    if timeConv[adapStart_idx] < start_time:
        adapStart_idx =+ 1
    adapEnd_idx = np.abs(timeConv - end_time).argmin()
    if timeConv[adapEnd_idx] > end_time:
        adapEnd_idx =- 1

    timeConv = timeConv[adapStart_idx:adapEnd_idx].astype(int)-start_time

    VO2 = data.iloc[adapStart_idx:adapEnd_idx,4].to_numpy()
    VCO2 = data.iloc[adapStart_idx:adapEnd_idx,6].to_numpy()
    metabolics = (0.278*VO2 + 0.075*VCO2)/mass

    newData = np.zeros(end_time-start_time)
    for idx, xx in enumerate(metabolics):
        newData[timeConv[idx]] = xx

    for idx, xx in enumerate(newData):
        if xx < 0.1:
            newData[idx] = newData[idx-1]

    return newData
