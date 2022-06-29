import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import metabolicFunc as MF

fileDirectory = os.getcwd() + "/Data/"
fileNamePool = ['AB2_Metabolics_Session6.csv', 'AB4_Metabolics_Session6.csv', 'AB5_Metabolics_Session6.csv']
subjectMassPool = [68, 60, 59, 78, 60]

extraSessionResult = np.empty((2700,0))
for ii in np.arange(3):
    fileName = fileNamePool[ii]
    data = MF.computeOxyconMetabolics(fileDirectory+fileName, subjectMassPool[ii], 720, 3420)
    extraSessionResult = np.column_stack((extraSessionResult, data))

    # meanData = MF.moving_average(data, 180)
#     plt.plot(meanData,'o')
# plt.show()

regularSessionResult0 = np.empty((2700,0))
regularSessionResult1 = np.empty((2700,0))
regularSessionResult2 = np.empty((2700,0))
regularSessionResult3 = np.empty((2700,0))

regularSessionResult = 

subjectNamePool = ["AB1", "AB2", "AB3", "AB4", "AB5"]
sessionNumPool = ["2", "3", "4", "5"]

for subjectIdx, subjectName in enumerate(subjectNamePool):
    for sessionNum in sessionNumPool:
        fileName = subjectName + "_Metabolics_Session" + sessionNum
        for fileList in os.listdir(fileDirectory):
            if fileName in fileList:
                fileName = fileList

        data = MF.computeOxyconMetabolics(fileDirectory+fileName, subjectMassPool[subjectIdx], 720, 3420)
        _, deltaIdx = MF.computeSplitBeltSpeedDelta(fileName)

        





                # data = np.expand_dims(data, axis=1)
#                 processedData[deltaSpeed] = np.column_stack((processedData[deltaSpeed], data))

# np.save("metabolicData.npy", processedData)

