import procFunc as pf
import numpy as np
import pandas as pd
import os
import fnmatch
import c3d
import matplotlib.pyplot as plt
import pickle

# load pre-synced data across all subjects
result = pf.load_synced_data()

# Evalauting SLA from result
stance_idx_list = []
sla_list = []
sla_gradient_list = []
for idx, data in enumerate(result):
    stance_idx = pf.compute_stance_idx(data.force_r[:,2], data.force_l[:,2])
    sla = pf.compute_SLA(stance_idx, data.foot_r, data.foot_l, data.fast_leg)
    stance_idx_list.append(stance_idx)
    sla_list.append(sla)


rolling = pf.rolling_window(sla_list[3], 180)
test_list = []
for idx, window in enumerate(rolling):
    print(rolling[idx,:].shape)
    test_list.append(np.gradient(rolling[idx,:]))


test= np.diff(sla_list[3])
gradient = np.gradient(sla_list[3])

plt.plot(test)
plt.ylim((-0.5,0.5))
plt.show()





# foot_place_std_r = []
# foot_place_std_l = []
# for idx, data in enumerate(result):
#     stance_idx = pf.compute_stance_idx(data.force_r[:,2], data.force_l[:,2])
#     mag_r, mag_l = pf.compute_foot_placement(stance_idx, foot_r, foot_l, com)
#     y_r, y_l = pf.compute_foot_placement_y(stance_idx, foot_r, foot_l, com)

#     movWindow = 10
#     std_r = np.std(pf.rolling_window(y_r, movWindow), axis=-1)
#     std_l = np.std(pf.rolling_window(y_l, movWindow), axis=-1)

#     if fast_leg == 0:
#         foot_place_std_r.append(std_r)
#         foot_place_std_l.append(std_l)
#     elif fast_leg == 1:
#         foot_place_std_r.append(std_l)
#         foot_place_std_l.append(std_r)
    

# data = np.load('/Users/inseungkang/Documents/learningalgos/Metabolic Cost/metabolicData.npy')
# movWindow = 180
# dataRange = 2700-movWindow+1

# dataFilt = [np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0))]
# fitData = [np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0))]
# x_vec = np.array(range(dataRange))

# print(data.shape)
# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w
# for sub_idx in range(5):
#     for del_idx in range(4):
#         y_vec = np.round(data[del_idx][:,sub_idx].flatten(), 4)
#         y_vec = moving_average(y_vec, movWindow)
#         dataFilt[del_idx] = np.column_stack((dataFilt[del_idx], y_vec))



# fig, axs = plt.subplots(3, 3, sharex=True)
# axs[0, 0].set_title('AB1')
# axs[0, 1].set_title('AB2')
# axs[0, 2].set_title('AB4')
# axs[0, 0].set_ylabel('Metabolic Cost')
# axs[1, 0].set_ylabel('SLA')
# axs[2, 0].set_ylabel('Exploration')

# axs[0, 0].plot(dataFilt[3][:,0], 'o', markersize=1)
# axs[0, 1].plot(dataFilt[3][:,1], 'o', markersize=1)
# axs[0, 2].plot(dataFilt[3][:,3], 'o', markersize=1)

# axs[1, 0].plot(sla_list[3])
# axs[1, 0].plot(np.arange(len(sla_list[3])),np.zeros(len(sla_list[3])))
# axs[1, 1].plot(sla_list[7])
# axs[1, 1].plot(np.arange(len(sla_list[7])),np.zeros(len(sla_list[7])))
# axs[1, 2].plot(sla_list[15])
# axs[1, 2].plot(np.arange(len(sla_list[15])),np.zeros(len(sla_list[15])))

# axs[2, 0].plot(foot_place_std_r[3])
# axs[2, 1].plot(foot_place_std_r[7])
# axs[2, 2].plot(foot_place_std_r[15])

# # for subject in [0, 1, 2, 3, 4]:
# #     for delta in [0, 1, 2, 3]:
# #         axs[subject, delta].plot(foot_place_std_r[4*subject+delta])
# custom_ylim = (-0.25, 0.1)
# plt.setp(axs[1,:], ylim=custom_ylim)
# fig.suptitle('Trials with clear trends (+2 Delta case)')
# plt.show()















