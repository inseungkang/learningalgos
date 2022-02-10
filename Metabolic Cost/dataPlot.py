import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import myFunc
import matplotlib.pyplot as plt

data = np.load('metabolicData.npy')
# for idx in range(4):
#     data[idx] = data[idx][:1000,:]

dataMean = np.mean(data, axis=2)
dataMean = np.array(dataMean, dtype=np.float64)

fitData = [np.empty((2700,0)), np.empty((2700,0)), np.empty((2700,0)), np.empty((2700,0))]
x_vec = np.array(range(len(data[0][:,0])))
for sub_idx in range(5):
    for del_idx in range(4):
        y_vec = np.round(data[del_idx][:,sub_idx].flatten(), 4)
        z = np.polyfit(x_vec, y_vec, 3)
        p = np.poly1d(z)
        fitData[del_idx] = np.column_stack((fitData[del_idx], p(x_vec)))

fig, axs = plt.subplots(5, 4, sharex=True)
# axs.Axes.set_yticklabels(labels=None)
axs[0, 0].set_title('-2$\Delta$')
axs[0, 0].set_ylabel('AB1')
axs[1, 0].set_ylabel('AB2')
axs[2, 0].set_ylabel('AB3')
axs[3, 0].set_ylabel('AB4')
axs[4, 0].set_ylabel('AB5')

axs[0, 0].plot(data[0][:,0], 'tab:blue')
axs[1, 0].plot(data[0][:,1], 'tab:orange')
axs[2, 0].plot(data[0][:,2], 'tab:green')
axs[3, 0].plot(data[0][:,3], 'tab:red')
axs[4, 0].plot(data[0][:,4], 'tab:purple')
axs[0, 0].plot(fitData[0][:,0], 'k')
axs[1, 0].plot(fitData[0][:,1], 'k')
axs[2, 0].plot(fitData[0][:,2], 'k')
axs[3, 0].plot(fitData[0][:,3], 'k')
axs[4, 0].plot(fitData[0][:,4], 'k')

axs[0, 1].set_title('-$\Delta$')
axs[0, 1].plot(data[1][:,0], 'tab:blue')
axs[1, 1].plot(data[1][:,1], 'tab:orange')
axs[2, 1].plot(data[1][:,2], 'tab:green')
axs[3, 1].plot(data[1][:,3], 'tab:red')
axs[4, 1].plot(data[1][:,4], 'tab:purple')
axs[0, 1].plot(fitData[1][:,0], 'k')
axs[1, 1].plot(fitData[1][:,1], 'k')
axs[2, 1].plot(fitData[1][:,2], 'k')
axs[3, 1].plot(fitData[1][:,3], 'k')
axs[4, 1].plot(fitData[1][:,4], 'k')

axs[0, 2].set_title('$\Delta$')
axs[0, 2].plot(data[2][:,0], 'tab:blue')
axs[1, 2].plot(data[2][:,1], 'tab:orange')
axs[2, 2].plot(data[2][:,2], 'tab:green')
axs[3, 2].plot(data[2][:,3], 'tab:red')
axs[4, 2].plot(data[2][:,4], 'tab:purple')
axs[0, 2].plot(fitData[2][:,0], 'k')
axs[1, 2].plot(fitData[2][:,1], 'k')
axs[2, 2].plot(fitData[2][:,2], 'k')
axs[3, 2].plot(fitData[2][:,3], 'k')
axs[4, 2].plot(fitData[2][:,4], 'k')

axs[0, 3].set_title('2$\Delta$')
axs[0, 3].plot(data[3][:,0], 'tab:blue')
axs[1, 3].plot(data[3][:,1], 'tab:orange')
axs[2, 3].plot(data[3][:,2], 'tab:green')
axs[3, 3].plot(data[3][:,3], 'tab:red')
axs[4, 3].plot(data[3][:,4], 'tab:purple')
axs[0, 3].plot(fitData[3][:,0], 'k')
axs[1, 3].plot(fitData[3][:,1], 'k')
axs[2, 3].plot(fitData[3][:,2], 'k')
axs[3, 3].plot(fitData[3][:,3], 'k')
axs[4, 3].plot(fitData[3][:,4], 'k')
plt.show()

def expFunc(x, A, B, C):
    return A * np.exp(B*x) + C

# plt.plot(x_vec, y_vec)
# plt.plot(x_vec, fit_vec)
# plt.show()



# a, b, c = fit_exp_nonlinear(x_vec, y_vec)
# fit_y = expFunc(x_vec, a, b, c)

# plt.plot(x_vec, y_vec, fit_y)
# plt.show()
# print(dataMean.shape)
# plt.plot(data[3])
# plt.show()