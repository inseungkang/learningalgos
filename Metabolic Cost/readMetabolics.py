from concurrent.futures import process
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import myFunc
import matplotlib.pyplot as plt

current_path = os.getcwd() + "/Data/"
# "\\Data\\"
start = 720
end = 3420
processedData = [np.empty((2700,0)), np.empty((2700,0)), np.empty((2700,0)), np.empty((2700,0))]
massPool = [[68.7, 68.7, 67, 67.2],
            [62, 59, 59.8, 60.3],
            [59.6, 58.2, 58.4, 58.3],
            [77.3, 77.7, 78.4, 78.2],
            [59.5, 59.8, 59.8, 60.3]]

subjectID = ["AB1", "AB2", "AB3", "AB4", "AB5"]
sessionNum = ["2", "3", "4", "5"]

for sub_idx, subject in enumerate(subjectID):
    for sess_idx, session in enumerate(sessionNum):
        mass = massPool[sub_idx][sess_idx]
        filename = subject + "_Metabolics_Session" + session

        for file in os.listdir(current_path):
            if file.startswith(filename):
                print("Processing: " + filename)

                rSpeed = int(file.split("Right")[1].split("_")[0])
                lSpeed = int(file.split("Right")[1].split("Left")[1].split(".")[0])

                if rSpeed == 10:
                    deltaSpeed = [6, 8, 12, 14].index(lSpeed)
                else:
                    deltaSpeed = [6, 8, 12, 14].index(rSpeed)

                data = myFunc.computeMetabolics(current_path+file, mass, start, end)
                data = np.expand_dims(data, axis=1)
                processedData[deltaSpeed] = np.column_stack((processedData[deltaSpeed], data))

def expFunc(x, A, B, C):
    return A * np.exp(B*x) + C

y_vec = processedData[0].flatten()
x_vec = np.array(range(len(y_vec)))

np.save("metabolicData.npy", processedData)

