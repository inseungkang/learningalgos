import numpy as np
import pandas as pd
import os
import myFunc
import matplotlib.pyplot as plt

start = 720
end = 3420
mass = 60

subjectID = ["AB1"]
sessionNum = ["2", "3", "4", "5"]
processedData = np.empty((4,))
# for idx in range(4):
#     processedData[idx] = np.empty((2700,0))

# for subject in subjectID:
#     for session in sessionNum:
#         filename = subject + "_Metabolics_Session" + session

#         for file in os.listdir():
#             if file.startswith(filename):
#                 rSpeed = int(file.split("Right")[1].split("_")[0])
#                 lSpeed = int(file.split("Right")[1].split("Left")[1].split(".")[0])
#                 if rSpeed == 10:
#                     deltaSpeed = [6, 8, 12, 14].index(lSpeed)
#                 else:
#                     deltaSpeed = [6, 8, 12, 14].index(rSpeed)
#                 print(deltaSpeed)

#                 data = myFunc.computeMetabolics(file, mass, start, end)
#                 data = np.expand_dims(data, axis=1)
                
#                 processedData[deltaSpeed] = np.column_stack((processedData[deltaSpeed], data))
#                 deltaSpeed 

# plt.plot(processedData)
# plt.show()
