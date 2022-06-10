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
fig, axs = plt.subplots(5, 4, sharex=True, sharey=True)
# axs.Axes.set_yticklabels(labels=None)
axs[0, 0].set_title('-2$\Delta$')
axs[0, 1].set_title('-1$\Delta$')
axs[0, 2].set_title('+1$\Delta$')
axs[0, 3].set_title('+2$\Delta$')

axs[0, 0].set_ylabel('AB1')
axs[1, 0].set_ylabel('AB2')
axs[2, 0].set_ylabel('AB3')
axs[3, 0].set_ylabel('AB4')
axs[4, 0].set_ylabel('AB5')
xtime = np.arange(dataRange)/60
for ii in np.arange(4):
    axs[0, ii].plot(xtime, dataFilt[ii][:,0]/dataFilt[ii][0,0]-1, color = palette[0], linewidth=2.5)
    axs[1, ii].plot(xtime, dataFilt[ii][:,1]/dataFilt[ii][0,1]-1, color = palette[1], linewidth=2.5)
    axs[2, ii].plot(xtime, dataFilt[ii][:,2]/dataFilt[ii][0,2]-1, color = palette[2], linewidth=2.5)
    axs[3, ii].plot(xtime, dataFilt[ii][:,3]/dataFilt[ii][0,3]-1, color = palette[3], linewidth=2.5)
    axs[4, ii].plot(xtime, dataFilt[ii][:,4]/dataFilt[ii][0,4]-1, color = palette[4], linewidth=2.5)
# axs[0, 0].plot(fitData[0][:,0], 'k')
# axs[1, 0].plot(fitData[0][:,1], 'k')
# axs[2, 0].plot(fitData[0][:,2], 'k')
# axs[3, 0].plot(fitData[0][:,3], 'k')
# axs[4, 0].plot(fitData[0][:,4], 'k')
fig.supylabel('Change in Metabolic Cost (W/kg)',fontweight="bold")
fig.supxlabel('Time (min)',fontweight="bold")
fig.suptitle('Metabolic cost during different split-belt walking conditions',fontweight="bold")
plt.show()




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