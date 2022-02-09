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

subjectID = ["AB1", "AB2", "AB3", "AB4", "AB5"]
sessionNum = ["2", "3", "4", "5"]

processedData = [np.empty((2700,0)), np.empty((2700,0)), np.empty((2700,0)), np.empty((2700,0))]

for subject in subjectID:
    for session in sessionNum:
        filename = subject + "_Metabolics_Session" + session

        for file in os.listdir(current_path):
            if file.startswith(filename):
                rSpeed = int(file.split("Right")[1].split("_")[0])
                lSpeed = int(file.split("Right")[1].split("Left")[1].split(".")[0])

                if rSpeed == 10:
                    deltaSpeed = [6, 8, 12, 14].index(lSpeed)
                else:
                    deltaSpeed = [6, 8, 12, 14].index(rSpeed)
                print(filename)
                data = myFunc.computeMetabolics(current_path+file, mass, start, end)
                data = np.expand_dims(data, axis=1)
                processedData[deltaSpeed] = np.column_stack((processedData[deltaSpeed], data))

def expFunc(x, A, B, C):
    return A * np.exp(B*x) + C

y_vec = processedData[0].flatten()
x_vec = np.array(range(len(y_vec)))

# params, _  = curve_fit(expFunc, x_vec, y_vec)
# a, b, c = params[0], params[1], params[2]
# print(params)
# yfit = a*np.exp(b*x_vec)+c

# z = np.polyfit(x_vec, y_vec, 30)


# plt.plot(y_vec)
# plt.plot(z)
# plt.show()


