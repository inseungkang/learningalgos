import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from random import randint

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

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(pelvisPosX[1000:1500],pelvisPosY[1000:1500],pelvisPosZ[1000:1500], 'blue')
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xs = []
ys = []

for ii, xx in enumerate(pelvisPosX[2:]):
    xs.append(ii)
    ys.append(xx)

def animate(i):
    ax.clear()
    ax.plot(xs, ys)

# Set up plot to call animate() function every 1000 milliseconds
ani = FuncAnimation(fig, animate, interval=1000)
plt.show()