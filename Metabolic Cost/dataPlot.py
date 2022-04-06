import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import math
import myFunc
import matplotlib.pyplot as plt

data = np.load('metabolicData.npy')
movWindow = 180
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

# fig, axs = plt.subplots(5, 4, sharex=True)
# # axs.Axes.set_yticklabels(labels=None)
# axs[0, 0].set_title('-2$\Delta$')
# axs[0, 0].set_ylabel('AB1')
# axs[1, 0].set_ylabel('AB2')
# axs[2, 0].set_ylabel('AB3')
# axs[3, 0].set_ylabel('AB4')
# axs[4, 0].set_ylabel('AB5')

# axs[0, 0].plot(dataFilt[0][:,0], 'o', color = 'blue', markersize=1)
# axs[1, 0].plot(dataFilt[0][:,1], 'o', color = 'orange', markersize=1)
# axs[2, 0].plot(dataFilt[0][:,2], 'o', color = 'green', markersize=1)
# axs[3, 0].plot(dataFilt[0][:,3], 'o', color = 'red', markersize=1)
# axs[4, 0].plot(dataFilt[0][:,4], 'o', color = 'purple', markersize=1)
# # axs[0, 0].plot(fitData[0][:,0], 'k')
# # axs[1, 0].plot(fitData[0][:,1], 'k')
# # axs[2, 0].plot(fitData[0][:,2], 'k')
# # axs[3, 0].plot(fitData[0][:,3], 'k')
# # axs[4, 0].plot(fitData[0][:,4], 'k')

# axs[0, 1].set_title('-$\Delta$')
# axs[0, 1].plot(dataFilt[1][:,0], 'o', color = 'blue', markersize=1)
# axs[1, 1].plot(dataFilt[1][:,1], 'o', color = 'orange', markersize=1)
# axs[2, 1].plot(dataFilt[1][:,2], 'o', color = 'green', markersize=1)
# axs[3, 1].plot(dataFilt[1][:,3], 'o', color = 'red', markersize=1)
# axs[4, 1].plot(dataFilt[1][:,4], 'o', color = 'purple', markersize=1)
# # axs[0, 1].plot(fitData[1][:,0], 'k')
# # axs[1, 1].plot(fitData[1][:,1], 'k')
# # axs[2, 1].plot(fitData[1][:,2], 'k')
# # axs[3, 1].plot(fitData[1][:,3], 'k')
# # axs[4, 1].plot(fitData[1][:,4], 'k')

# axs[0, 2].set_title('$\Delta$')
# axs[0, 2].plot(dataFilt[2][:,0], 'o', color = 'blue', markersize=1)
# axs[1, 2].plot(dataFilt[2][:,1], 'o', color = 'orange', markersize=1)
# axs[2, 2].plot(dataFilt[2][:,2], 'o', color = 'green', markersize=1)
# axs[3, 2].plot(dataFilt[2][:,3], 'o', color = 'red', markersize=1)
# axs[4, 2].plot(dataFilt[2][:,4], 'o', color = 'purple', markersize=1)
# # axs[0, 2].plot(fitData[2][:,0], 'k')
# # axs[1, 2].plot(fitData[2][:,1], 'k')
# # axs[2, 2].plot(fitData[2][:,2], 'k')
# # axs[3, 2].plot(fitData[2][:,3], 'k')
# # axs[4, 2].plot(fitData[2][:,4], 'k')

# axs[0, 3].set_title('2$\Delta$')
# axs[0, 3].plot(dataFilt[3][:,0], 'o', color = 'blue', markersize=1)
# axs[1, 3].plot(dataFilt[3][:,1], 'o', color = 'orange', markersize=1)
# axs[2, 3].plot(dataFilt[3][:,2], 'o', color = 'green', markersize=1)
# axs[3, 3].plot(dataFilt[3][:,3], 'o', color = 'red', markersize=1)
# axs[4, 3].plot(dataFilt[3][:,4], 'o', color = 'purple', markersize=1)
# # axs[0, 3].plot(fitData[3][:,0], 'k')
# # axs[1, 3].plot(fitData[3][:,1], 'k')
# # axs[2, 3].plot(fitData[3][:,2], 'k')
# # axs[3, 3].plot(fitData[3][:,3], 'k')
# # axs[4, 3].plot(fitData[3][:,4], 'k')
# fig.suptitle('Metabolic Cost moving 3 min for all subjects across trials')
# plt.show()




fig, axs = plt.subplots(1, 4, sharex=True)
axs[0].set_title('-2$\Delta$')
axs[0].set_ylabel('AB1')

axs[0].plot(dataFilt[0][:,0], 'o', color = 'orange', markersize=1)

axs[1].set_title('-$\Delta$')
axs[1].plot(dataFilt[1][:,0], 'o', color = 'orange', markersize=1)


axs[2].set_title('$\Delta$')
axs[2].plot(dataFilt[2][:,0], 'o', color = 'orange', markersize=1)


axs[3].set_title('2$\Delta$')
axs[3].plot(dataFilt[3][:,0], 'o', color = 'orange', markersize=1)

plt.show()