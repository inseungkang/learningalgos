import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import myFunc
import matplotlib.pyplot as plt

massPool = [[1, 2, 3, 4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]] 
mass = massPool[2][3]
print(mass)

# data = np.load('metabolicData.npy')
# dataMean = np.mean(data, axis=2)
# print(dataMean.shape)
# plt.plot(data[3])
# plt.show()