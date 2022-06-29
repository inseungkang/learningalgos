import numpy as np
import pandas as pd

def computeOxyconMetabolics(filepath, bodyMass, startTime, endTime):
    # Computing the user's metabolic cost from the Oxycon Moblie. Recorded data was reformated to a csv data.
    # Input start and end time to segment data (in minutes)
    # Column 0: Timestamp, Column 4: VO2, Column 6: VCO2

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

    adapStart_idx = np.abs(timeConv - startTime).argmin()
    if timeConv[adapStart_idx] < startTime:
        adapStart_idx = adapStart_idx + 1
    adapEnd_idx = np.abs(timeConv - endTime).argmin()
    if timeConv[adapEnd_idx] > endTime:
        adapEnd_idx = adapEnd_idx - 1

    timeConv = timeConv[adapStart_idx:adapEnd_idx].astype(int)-startTime

    VO2 = data.iloc[adapStart_idx:adapEnd_idx, 4].to_numpy()
    VCO2 = data.iloc[adapStart_idx:adapEnd_idx, 6].to_numpy()
    metabolics = (0.278*VO2 + 0.075*VCO2)/bodyMass
    # metabolics = (16.48/VO2 + 4.48/VCO2)/mass

    newData = np.zeros(endTime-startTime)
    for idx, xx in enumerate(metabolics):
        newData[timeConv[idx]] = xx

    for idx, xx in enumerate(newData):
        if xx < 0.1:
            newData[idx] = newData[idx-1]

    return newData


def computeSplitBeltSpeedDelta(fileName):
    speedDeltaPool = [-4, -2, 2, 4]
    speedR = int(fileName.split("Right")[1].split("_")[0])
    speedL = int(fileName.split("Left")[1].split(".")[0]) 

    if speedR == 10:
        speedDelta = speedL - speedR
    else:
        speedDelta = speedR - speedL

    deltaIdx = speedDeltaPool.index(speedDelta)
    return speedDelta, deltaIdx

def expFunc(x, A, B, C):
    return A * np.exp(B*x) + C

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w