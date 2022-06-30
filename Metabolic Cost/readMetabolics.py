import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import metabolicFunc as MF

fileDirectory = os.getcwd() + "/Data/"
fileNamePool = ['AB2_Metabolics_Session6.csv', 'AB4_Metabolics_Session6.csv', 'AB5_Metabolics_Session6.csv']
subjectMassPool = [68, 60, 59, 78, 60]

dataRange = 2700
filterWindow = 180
extraSessionResult = np.empty((dataRange-filterWindow+1,0))

for ii in np.arange(3):
    fileName = fileNamePool[ii]
    data = MF.computeOxyconMetabolics(fileDirectory+fileName, subjectMassPool[ii], 720, 3420)
    data = MF.moving_average(data, filterWindow)
    extraSessionResult = np.column_stack((extraSessionResult, data))

regularSessionResult0 = np.empty((dataRange-filterWindow+1,0))
regularSessionResult1 = np.empty((dataRange-filterWindow+1,0))
regularSessionResult2 = np.empty((dataRange-filterWindow+1,0))
regularSessionResult3 = np.empty((dataRange-filterWindow+1,0))

regularSessionResult = [regularSessionResult0, regularSessionResult1, regularSessionResult2, regularSessionResult3]

subjectNamePool = ["AB1", "AB2", "AB3", "AB4", "AB5"]
sessionNumPool = ["2", "3", "4", "5"]

for subjectIdx, subjectName in enumerate(subjectNamePool):
    for sessionNum in sessionNumPool:
        fileName = subjectName + "_Metabolics_Session" + sessionNum
        for fileList in os.listdir(fileDirectory):
            if fileName in fileList:
                fileName = fileList

        data = MF.computeOxyconMetabolics(fileDirectory+fileName, subjectMassPool[subjectIdx], 720, 3420)
        data = MF.moving_average(data, filterWindow)

        _, deltaIdx = MF.computeSplitBeltSpeedDelta(fileName)

        regularSessionResult[deltaIdx] = np.column_stack((regularSessionResult[deltaIdx], data))



palette = sns.color_palette("Paired_r", 5)
fig, axs = plt.subplots(5, 5, sharex=True)
axs[0, 0].set_title('-2$\Delta$')
axs[0, 1].set_title('-1$\Delta$')
axs[0, 2].set_title('+1$\Delta$')
axs[0, 3].set_title('+2$\Delta$')
axs[0, 4].set_title('+5$\Delta$')

axs[0, 0].set_ylabel('AB1')
axs[1, 0].set_ylabel('AB2')
axs[2, 0].set_ylabel('AB3')
axs[3, 0].set_ylabel('AB4')
axs[4, 0].set_ylabel('AB5')

regularSessionResult[deltaIdx]
xtime = np.arange(dataRange-filterWindow+1)/60
for deltaIdx in np.arange(4):
    axs[0, deltaIdx].plot(xtime, (regularSessionResult[deltaIdx][:,0]/regularSessionResult[deltaIdx][0,0]-1)*100, color = palette[0], linewidth=2.5)
    axs[1, deltaIdx].plot(xtime, (regularSessionResult[deltaIdx][:,1]/regularSessionResult[deltaIdx][0,1]-1)*100, color = palette[1], linewidth=2.5)
    axs[2, deltaIdx].plot(xtime, (regularSessionResult[deltaIdx][:,2]/regularSessionResult[deltaIdx][0,2]-1)*100, color = palette[2], linewidth=2.5)
    axs[3, deltaIdx].plot(xtime, (regularSessionResult[deltaIdx][:,3]/regularSessionResult[deltaIdx][0,3]-1)*100, color = palette[3], linewidth=2.5)
    axs[4, deltaIdx].plot(xtime, (regularSessionResult[deltaIdx][:,4]/regularSessionResult[deltaIdx][0,4]-1)*100, color = palette[4], linewidth=2.5)

axs[1, 4].plot(xtime, (extraSessionResult[:,0]/extraSessionResult[0,0]-1)*100, color = palette[1], linewidth=2.5)
axs[3, 4].plot(xtime, (extraSessionResult[:,1]/extraSessionResult[0,1]-1)*100, color = palette[3], linewidth=2.5)
axs[4, 4].plot(xtime, (extraSessionResult[:,2]/extraSessionResult[0,2]-1)*100, color = palette[4], linewidth=2.5)


for ii in np.arange(5):
    axs[0,ii].set_ylim((-15, 5))
    axs[1,ii].set_ylim((-45, 5))
    axs[2,ii].set_ylim((-15, 15))
    axs[3,ii].set_ylim((-25, 15))
    axs[4,ii].set_ylim((-25, 10))

# axs[0, 0].plot(fitData[0][:,0], 'k')
# axs[1, 0].plot(fitData[0][:,1], 'k')
# axs[2, 0].plot(fitData[0][:,2], 'k')
# axs[3, 0].plot(fitData[0][:,3], 'k')
# axs[4, 0].plot(fitData[0][:,4], 'k')
fig.supylabel('Change in Metabolic Cost (%)',fontweight="bold")
fig.supxlabel('Time (min)',fontweight="bold")
fig.suptitle('Metabolic cost during different split-belt walking conditions',fontweight="bold")
plt.show()












                # data = np.expand_dims(data, axis=1)
#                 processedData[deltaSpeed] = np.column_stack((processedData[deltaSpeed], data))

# np.save("metabolicData.npy", processedData)

