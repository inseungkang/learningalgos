from concurrent.futures import process
from turtle import speed
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import myFunc
import matplotlib.pyplot as plt

current_path = os.getcwd() + "/Data/"
start = 0
end = 360
processedData = [np.empty((360,0)), np.empty((360,0)), np.empty((360,0)), np.empty((360,0))]
massPool = [[68.7, 68.7, 67, 67.2],
            [62, 59, 59.8, 60.3],
            [59.6, 58.2, 58.4, 58.3],
            [77.3, 77.7, 78.4, 78.2],
            [59.5, 59.8, 59.8, 60.3]]

massPool = np.mean(np.array(massPool), axis=-1)
subjectID = ["AB1", "AB2", "AB3", "AB4", "AB5"]
sessionNum = ["1"]

result = np.empty((5,5))
for sub_idx, subject in enumerate(subjectID):
    mass = massPool[sub_idx]
    filename = subject + "_Metabolics_Session1"
    for file in os.listdir(current_path):
        if file.startswith(filename):
            # print("Processing: " + file)

            Speed = int(file.split("Speed")[1].split(".csv")[0])
            speed_idx = int((Speed - 6)/2)
            data = myFunc.computeMetabolics(current_path+file, mass, start, end)
            meanMeta = np.mean(data[240:])
            result[sub_idx,speed_idx] = meanMeta
            # data = np.expand_dims(data, axis=1)
            # processedData[Speed] = np.column_stack((processedData[deltaSpeed], data))


# plt.show()

x = np.arange(0.6, 1.6, step=0.2)
legend = ['AB1', 'AB2', 'AB3', 'AB4', 'AB5'] 
plt.plot(x, result[0,:])
plt.plot(x, result[1,:])
plt.plot(x, result[2,:])
plt.plot(x, result[3,:])
plt.plot(x, result[4,:])
# plt.xticks(np.arange(6, 16, step=2))
plt.title('Average Metabolic Cost for Tied-Belt')
plt.ylabel('Metabolic Cost (W/kg)')
plt.xlabel('Speed (m/s)')
plt.legend(legend)
plt.show()
# plt.plot(result)
# plt.show()
# def expFunc(x, A, B, C):
#     return A * np.exp(B*x) + C

# y_vec = processedData[0].flatten()
# x_vec = np.array(range(len(y_vec)))

# np.save("metabolicData_const.npy", processedData)

