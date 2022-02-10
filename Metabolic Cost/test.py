import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import myFunc
import matplotlib.pyplot as plt

# massPool = [[1, 2, 3, 4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]] 
# mass = massPool[2][3]
# print(mass)



data = np.load('metabolicData.npy')
dataMean = np.mean(data, axis=2)
dataMean = np.array(dataMean, dtype=np.float64)
def expFunc(x, A, B, C):
    return A * np.exp(B*x) + C

def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = curve_fit(expFunc, x_vec, y_vec, maxfev=1000)
    A, K, C = opt_parms
    return A, K, C

y_vec = np.round(dataMean[3].flatten(), 4)
x_vec = np.array(range(len(y_vec)))

print(y_vec[1])

a, b, c = fit_exp_nonlinear(x_vec, y_vec)
fit_y = expFunc(x_vec, a, b, c)

plt.plot(x_vec, y_vec, fit_y)
plt.show()
# print(dataMean.shape)
# plt.plot(dataMean[0])
# plt.show()