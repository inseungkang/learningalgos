import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

mkr = np.load('testMarkerData.npy')
frc = np.load('testForceData.npy')

# RPSI = mkr[0,1,:]
# LPSI = mkr[1,1,:]
# LASI = mkr[2,1,:]
# RASI = mkr[3,1,:]

pelvisPosX = np.mean(mkr[0:3,0,:], axis=0)
pelvisPosY = np.mean(mkr[0:3,1,:], axis=0)
pelvisPosZ = np.mean(mkr[0:3,2,:], axis=0)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(mkr[0,0,5000:6000], mkr[0,1,5000:6000], mkr[0,2,5000:6000], 'gray')
# plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(pelvisPosX[1000:1500],pelvisPosY[1000:1500],pelvisPosZ[1000:1500], 'blue')
plt.show()

# plt.plot(mkr[0,3,1:])
# plt.plot(mkr[1,3,1:])
# plt.plot(mkr[2,3,1:])
# plt.plot(mkr[3,3,1:])

# plt.plot(pelvisPosZ[1:])
# plt.show()
