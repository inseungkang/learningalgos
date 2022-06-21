from turtle import color
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import math
import myFunc
import matplotlib.pyplot as plt

data = np.load('metabolicData.npy')
movWindow = 120
dataRange = 2700-movWindow+1

dataFilt = [np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0))]
fitData = [np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0))]
x_vec = np.array(range(dataRange))

print(data.shape)

for sub_idx in range(5):
    for del_idx in range(4):
        y_vec = np.round(data[del_idx][:,sub_idx].flatten(), 4)
        y_vec = myFunc.moving_average(y_vec, movWindow)
        # z = np.polyfit(np.log(x_vec), y_vec, 1)
        # p = np.poly1d(z)
        dataFilt[del_idx] = np.column_stack((dataFilt[del_idx], y_vec))
        # fitData[del_idx] = np.column_stack((fitData[del_idx], p(np.log(x_vec))))
import seaborn as sns
# from matplotlib import rcParams
# rcParams['font.family'] = 'serif'
# palette = sns.color_palette("Spectral", 5).as_hex()
palette = sns.color_palette("Paired_r", 5)
# fig, axs = plt.subplots(1,5, sharey = True)
# axs[0, 0].set_title('-2$\Delta$')
# axs[0, 1].set_title('-1$\Delta$')
# axs[0, 2].set_title('+1$\Delta$')
# axs[0, 3].set_title('+2$\Delta$')

# axs[0, 0].set_ylabel('AB1')
# axs[1, 0].set_ylabel('AB2')
# axs[2, 0].set_ylabel('AB3')
# axs[3, 0].set_ylabel('AB4')
# axs[4, 0].set_ylabel('AB5')
xtime = np.arange(dataRange)/60
x = ['-2$\Delta$','-1$\Delta$','+1$\Delta$','+2$\Delta$']
import seaborn as sns
colors = sns.color_palette("coolwarm", 4)

mean1 = []
mean2 = []
mean3 = []
mean4 = []

for subject in np.arange(5):


    metaChange1 = -dataFilt[0][0,subject] + np.mean(dataFilt[0][-120:,subject])
    metaChange2 = -dataFilt[1][0,subject] + np.mean(dataFilt[1][-120:,subject])
    metaChange3 = -dataFilt[2][0,subject] + np.mean(dataFilt[2][-120:,subject])
    metaChange4 = -dataFilt[3][0,subject] + np.mean(dataFilt[3][-120:,subject])

    mean1.append(metaChange1)
    mean2.append(metaChange2)
    mean3.append(metaChange3)
    mean4.append(metaChange4)

    # y = [metaChange1, metaChange2, metaChange3, metaChange4]   
    # axs[subject].bar(x,y, color=colors)

    # axs[0, ii].plot(xtime, dataFilt[ii][:,0], color = palette[0], linewidth=2.5)
    # axs[1, ii].plot(xtime, dataFilt[ii][:,1], color = palette[1], linewidth=2.5)
    # axs[2, ii].plot(xtime, dataFilt[ii][:,2], color = palette[2], linewidth=2.5)
    # axs[3, ii].plot(xtime, dataFilt[ii][:,3], color = palette[3], linewidth=2.5)
    # axs[4, ii].plot(xtime, dataFilt[ii][:,4], color = palette[4], linewidth=2.5)
# axs[0, 0].plot(fitData[0][:,0], 'k')
# axs[1, 0].plot(fitData[0][:,1], 'k')
# axs[2, 0].plot(fitData[0][:,2], 'k')
# axs[3, 0].plot(fitData[0][:,3], 'k')
# axs[4, 0].plot(fitData[0][:,4], 'k')

d1 = np.mean(mean1)
d2 = np.mean(mean2)
d3 = np.mean(mean3)
d4 = np.mean(mean4)
yvec = [d1, d2, d3, d4]
stvec = [np.std(mean1), np.std(mean2), np.std(mean3), np.std(mean4)]
plt.bar(x, yvec, yerr=stvec, color=colors)
plt.ylabel('Change in Metabolic Cost (W/kg)',fontweight="bold")
plt.xlabel('Time (min)',fontweight="bold")
plt.title('Metabolic cost change before and after split-belt walking',fontweight="bold")
plt.show()

# axs[0].set_title('AB1',fontweight="bold")
# axs[1].set_title('AB2',fontweight="bold")
# axs[2].set_title('AB3',fontweight="bold")
# axs[3].set_title('AB4',fontweight="bold")
# axs[4].set_title('AB5',fontweight="bold")

# axs[0].set_ylabel('Change in Metabolic Cost (W/kg)',fontweight="bold")
# axs[2].set_xlabel('Time (min)',fontweight="bold")
# fig.suptitle('Metabolic cost change before and after split-belt walking',fontweight="bold")

# # fig.supylabel('Change in Metabolic Cost (W/kg)',fontweight="bold")
# # fig.supxlabel('Time (min)',fontweight="bold")
# # fig.suptitle('Metabolic cost during different split-belt walking conditions',fontweight="bold")
# plt.show()


# fig, axs = plt.subplots(1, 4, sharex=True)
# axs[0].set_title('-2$\Delta$')
# axs[0].set_ylabel('AB1')

# axs[0].plot(dataFilt[0][:,0], 'o', color = 'orange', markersize=1)

# axs[1].set_title('-$\Delta$')
# axs[1].plot(dataFilt[1][:,0], 'o', color = 'orange', markersize=1)

# axs[2].set_title('$\Delta$')
# axs[2].plot(dataFilt[2][:,0], 'o', color = 'orange', markersize=1)

# axs[3].set_title('2$\Delta$')
# axs[3].plot(dataFilt[3][:,0], 'o', color = 'orange', markersize=1)

# plt.show()