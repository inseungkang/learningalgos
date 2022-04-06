import procFunc as pf
import numpy as np
import pandas as pd
import os
import fnmatch
import c3d
import matplotlib.pyplot as plt
import pickle

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

## load c3d file data and extract header column info and save 
# file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d data/'
# file_list = os.listdir(file_dir)

# result = []
# for file_name in file_list:
#     idxlist = c3d_marker_header_loc(os.path.join(file_dir,file_name))
#     result.append([file_name, idxlist])

# save_dir = '/Users/inseungkang/Documents/learningalgos data/c3d marker header location.pkl'
# with open(save_file_name, 'wb') as handle:
#     pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # Load marker header file to get the index column
# header_dir = '/Users/inseungkang/Documents/learningalgos data/c3d marker header location.pkl'
# with open(header_dir, 'rb') as handle:
#     marker_header_idx = pickle.load(handle)

# # Open npz marker and force data and match the relevaant index column
# data_dir = '/Users/inseungkang/Documents/learningalgos data/npz data/'
# file_list = os.listdir(data_dir)
# file_list = [i for i in file_list if 'npz' in i]

# for file_name in file_list:

#     for idx, [name, list] in enumerate(marker_header_idx):
#         if file_name.split('.npz')[0] == name.split('.c3d')[0]:
#             indexList = list

#     if file_name.find('Static') & file_name.find('Session1') == -1:
#         print(file_name)
#         pf.extract_from_c3d(data_dir+file_name, indexList, 45)


# data_dir = '/Users/inseungkang/Documents/learningalgos data/synced data/'
# file_list = [i for i in os.listdir(data_dir) if '.pkl' in i]

# for file_name in file_list:
#     subjectNum = int(file_name.split('_Session')[0].split('AB')[1])
#     session = int(file_name.split('Session')[1].split('_Right')[0])
#     rSpeed = int(file_name.split('Right')[1].split('_Left')[0])
#     lSpeed = int(file_name.split('Left')[1].split('_synced')[0])

#     if rSpeed == 10:
#         delta = int((lSpeed - rSpeed)/2)
#     else:
#         delta = int((rSpeed - lSpeed)/2)

#     delta = delta + 2
    
#     if rSpeed > lSpeed:
#         fast_leg = 0
#     else:
#         fast_leg = 1

#     with open(data_dir+file_name, 'rb') as handle:
#         com, foot_r, foot_l, force_r, force_l = pickle.load(handle)

#         if subjectNum == 1:
#             if delta == 0:
#                 AB1_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 1:
#                 AB1_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 3:
#                 AB1_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 4:
#                 AB1_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

#         if subjectNum == 2:
#             if delta == 0:
#                 AB2_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 1:
#                 AB2_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 3:
#                 AB2_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 4:
#                 AB2_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

#         if subjectNum == 3:
#             if delta == 0:
#                 AB3_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 1:
#                 AB3_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 3:
#                 AB3_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 4:
#                 AB3_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

#         if subjectNum == 4:
#             if delta == 0:
#                 AB4_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 1:
#                 AB4_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 3:
#                 AB4_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 4:
#                 AB4_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

#         if subjectNum == 5:
#             if delta == 0:
#                 AB5_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 1:
#                 AB5_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 3:
#                 AB5_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
#             if delta == 4:
#                 AB5_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

# result = [AB1_delta0, AB1_delta1, AB1_delta2, AB1_delta3,
#         AB2_delta0, AB2_delta1, AB2_delta2, AB2_delta3,
#         AB3_delta0, AB3_delta1, AB3_delta2, AB3_delta3,
#         AB4_delta0, AB4_delta1, AB4_delta2, AB4_delta3,
#         AB5_delta0, AB5_delta1, AB5_delta2, AB5_delta3]


# foot_place_std_r = []
# foot_place_std_l = []

# for idx, data in enumerate(result):
#     stance_idx = pf.compute_stance_idx(data.force_r[:,2], data.force_l[:,2])
#     mag_r, mag_l = pf.compute_foot_placement(stance_idx, foot_r, foot_l, com)
#     y_r, y_l = pf.compute_foot_placement_y(stance_idx, foot_r, foot_l, com)

#     movWindow = 10
#     std_r = np.std(rolling_window(y_r, movWindow), axis=-1)
#     std_l = np.std(rolling_window(y_l, movWindow), axis=-1)

#     if fast_leg == 0:
#         foot_place_std_r.append(std_r)
#         foot_place_std_l.append(std_l)
#     elif fast_leg == 1:
#         foot_place_std_r.append(std_l)
#         foot_place_std_l.append(std_r)
    
# fig, axs = plt.subplots(5, 4, sharex=True)
# axs[0, 0].set_title('-2$\Delta$')
# axs[0, 1].set_title('-$\Delta$')
# axs[0, 2].set_title('+1$\Delta$')
# axs[0, 3].set_title('+2$\Delta$')
# axs[0, 0].set_ylabel('AB1')
# axs[1, 0].set_ylabel('AB2')
# axs[2, 0].set_ylabel('AB3')
# axs[3, 0].set_ylabel('AB4')
# axs[4, 0].set_ylabel('AB5')
# for subject in [0, 1, 2, 3, 4]:
#     for delta in [0, 1, 2, 3]:
#         axs[subject, delta].plot(foot_place_std_r[4*subject+delta])

# fig.suptitle('Foot placement exploration (moving' + str(movWindow) + 'std)')
# plt.show()






###################
# Ploting all sla

# stance_idx_list = []
# for idx, data in enumerate(result):
#     stance_idx = pf.compute_stance_idx(data.force_r[:,2], data.force_l[:,2])
#     stance_idx_list.append(stance_idx)

# check index 8: AB3_delta0
# stance_idx_list = []
# sla_list = []
# for idx, data in enumerate(result):
#     stance_idx = pf.compute_stance_idx(data.force_r[:,2], data.force_l[:,2])
#     sla = pf.compute_SLA(stance_idx, data.foot_r, data.foot_l, data.fast_leg)
#     stance_idx_list.append(stance_idx)
#     sla_list.append(sla)

# fig, axs = plt.subplots(5, 4, sharex=True)

# # axs.set_yticklabels(labels=None)
# axs[0, 0].set_title('-2$\Delta$')
# axs[0, 1].set_title('-$\Delta$')
# axs[0, 2].set_title('+1$\Delta$')
# axs[0, 3].set_title('+2$\Delta$')
# axs[0, 0].set_ylabel('AB1')
# axs[1, 0].set_ylabel('AB2')
# axs[2, 0].set_ylabel('AB3')
# axs[3, 0].set_ylabel('AB4')
# axs[4, 0].set_ylabel('AB5')

# axs[0, 0].plot(sla_list[0])
# axs[0, 0].plot(np.arange(len(sla_list[0])),np.zeros(len(sla_list[0])))
# axs[0, 1].plot(sla_list[1])
# axs[0, 1].plot(np.arange(len(sla_list[1])),np.zeros(len(sla_list[1])))
# axs[0, 2].plot(sla_list[2])
# axs[0, 2].plot(np.arange(len(sla_list[2])),np.zeros(len(sla_list[2])))
# axs[0, 3].plot(sla_list[3])
# axs[0, 3].plot(np.arange(len(sla_list[3])),np.zeros(len(sla_list[3])))


# axs[1, 0].plot(sla_list[4])
# axs[1, 0].plot(np.arange(len(sla_list[4])),np.zeros(len(sla_list[4])))
# axs[1, 1].plot(sla_list[5])
# axs[1, 1].plot(np.arange(len(sla_list[5])),np.zeros(len(sla_list[5])))
# axs[1, 2].plot(sla_list[6])
# axs[1, 2].plot(np.arange(len(sla_list[6])),np.zeros(len(sla_list[6])))
# axs[1, 3].plot(sla_list[7])
# axs[1, 3].plot(np.arange(len(sla_list[7])),np.zeros(len(sla_list[7])))

# axs[2, 0].plot(sla_list[8])
# axs[2, 0].plot(np.arange(len(sla_list[8])),np.zeros(len(sla_list[8])))
# axs[2, 1].plot(sla_list[9])
# axs[2, 1].plot(np.arange(len(sla_list[9])),np.zeros(len(sla_list[9])))
# axs[2, 2].plot(sla_list[10])
# axs[2, 2].plot(np.arange(len(sla_list[10])),np.zeros(len(sla_list[10])))
# axs[2, 3].plot(sla_list[11])
# axs[2, 3].plot(np.arange(len(sla_list[11])),np.zeros(len(sla_list[11])))

# axs[3, 0].plot(sla_list[12])
# axs[3, 0].plot(np.arange(len(sla_list[12])),np.zeros(len(sla_list[12])))
# axs[3, 1].plot(sla_list[13])
# axs[3, 1].plot(np.arange(len(sla_list[13])),np.zeros(len(sla_list[13])))
# axs[3, 2].plot(sla_list[14])
# axs[3, 2].plot(np.arange(len(sla_list[14])),np.zeros(len(sla_list[14])))
# axs[3, 3].plot(sla_list[15])
# axs[3, 3].plot(np.arange(len(sla_list[15])),np.zeros(len(sla_list[15])))


# axs[4, 0].plot(sla_list[16])
# axs[4, 0].plot(np.arange(len(sla_list[16])),np.zeros(len(sla_list[16])))
# axs[4, 1].plot(sla_list[17])
# axs[4, 1].plot(np.arange(len(sla_list[17])),np.zeros(len(sla_list[17])))
# axs[4, 2].plot(sla_list[18])
# axs[4, 2].plot(np.arange(len(sla_list[18])),np.zeros(len(sla_list[18])))
# axs[4, 3].plot(sla_list[19])
# axs[4, 3].plot(np.arange(len(sla_list[19])),np.zeros(len(sla_list[19])))

# custom_ylim = (-0.25, 0.1)
# fig.suptitle('SLA for all subjects across trials')
# plt.setp(axs, ylim=custom_ylim)
# plt.show()

#####################################
# plotting AB1, AB2, AB4 2D cases

data_dir = '/Users/inseungkang/Documents/learningalgos data/synced data/'
file_list = [i for i in os.listdir(data_dir) if '.pkl' in i]

for file_name in file_list:
    subjectNum = int(file_name.split('_Session')[0].split('AB')[1])
    session = int(file_name.split('Session')[1].split('_Right')[0])
    rSpeed = int(file_name.split('Right')[1].split('_Left')[0])
    lSpeed = int(file_name.split('Left')[1].split('_synced')[0])

    if rSpeed == 10:
        delta = int((lSpeed - rSpeed)/2)
    else:
        delta = int((rSpeed - lSpeed)/2)

    delta = delta + 2
    
    if rSpeed > lSpeed:
        fast_leg = 0
    else:
        fast_leg = 1

    with open(data_dir+file_name, 'rb') as handle:
        com, foot_r, foot_l, force_r, force_l = pickle.load(handle)

        if subjectNum == 1:
            if delta == 0:
                AB1_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 1:
                AB1_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 3:
                AB1_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 4:
                AB1_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

        if subjectNum == 2:
            if delta == 0:
                AB2_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 1:
                AB2_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 3:
                AB2_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 4:
                AB2_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

        if subjectNum == 3:
            if delta == 0:
                AB3_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 1:
                AB3_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 3:
                AB3_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 4:
                AB3_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

        if subjectNum == 4:
            if delta == 0:
                AB4_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 1:
                AB4_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 3:
                AB4_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 4:
                AB4_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

        if subjectNum == 5:
            if delta == 0:
                AB5_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 1:
                AB5_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 3:
                AB5_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
            if delta == 4:
                AB5_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
result = [AB1_delta0, AB1_delta1, AB1_delta2, AB1_delta3,
        AB2_delta0, AB2_delta1, AB2_delta2, AB2_delta3,
        AB3_delta0, AB3_delta1, AB3_delta2, AB3_delta3,
        AB4_delta0, AB4_delta1, AB4_delta2, AB4_delta3,
        AB5_delta0, AB5_delta1, AB5_delta2, AB5_delta3]

foot_place_std_r = []
foot_place_std_l = []

stance_idx_list = []
sla_list = []
sla_deriv = []
movWindow = 180

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

for idx, data in enumerate(result):
    stance_idx = pf.compute_stance_idx(data.force_r[:,2], data.force_l[:,2])
    sla = pf.compute_SLA(stance_idx, data.foot_r, data.foot_l, data.fast_leg)
    sla[sla > 0.5] = 0
    sla[sla < -0.5] = 0
    
    stance_idx_list.append(stance_idx)
    # sla = moving_average(sla, 180)
    sla_list.append(sla)

    deriv_vec = []
    for ii in np.arange(len(sla)-movWindow):
        test_vec = sla[ii:ii+movWindow]
        slope = pf.calc_slope(test_vec)
        deriv_vec.append(slope)
    sla_deriv.append(deriv_vec)


for idx, data in enumerate(result):
    stance_idx = pf.compute_stance_idx(data.force_r[:,2], data.force_l[:,2])
    mag_r, mag_l = pf.compute_foot_placement(stance_idx, foot_r, foot_l, com)
    y_r, y_l = pf.compute_foot_placement_y(stance_idx, foot_r, foot_l, com)

    movWindow = 10
    std_r = np.std(rolling_window(y_r, movWindow), axis=-1)
    std_l = np.std(rolling_window(y_l, movWindow), axis=-1)

    if fast_leg == 0:
        foot_place_std_r.append(std_r)
        foot_place_std_l.append(std_l)
    elif fast_leg == 1:
        foot_place_std_r.append(std_l)
        foot_place_std_l.append(std_r)
    
data = np.load('/Users/inseungkang/Documents/learningalgos/Metabolic Cost/metabolicData.npy')
movWindow = 180
dataRange = 2700-movWindow+1

dataFilt = [np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0))]
fitData = [np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0))]
x_vec = np.array(range(dataRange))

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

metabolic_filt = []
metabolic_deriv = []

for sub_idx in range(5):
    for del_idx in range(4):
        y_vec = np.round(data[del_idx][:,sub_idx].flatten(), 4)
        y_vec = moving_average(y_vec, movWindow)
        metabolic_filt.append(y_vec)

        deriv_vec = []
        for ii in np.arange(len(y_vec)-movWindow):
            test_vec = y_vec[ii:ii+movWindow]
            slope = pf.calc_slope(test_vec)
            deriv_vec.append(slope)
        metabolic_deriv.append(deriv_vec)










from matplotlib.ticker import FormatStrFormatter

# plt.figure(1)
# plt.subplot(221)
# plt.plot(metabolic_filt[3])
# # plt.yticks([])
# plt.title('Metabolic Cost')
# plt.subplot(222)
# plt.plot(metabolic_deriv[3])
# # plt.yticks([])
# plt.title('Rate of Metabolic Change')

# plt.subplot(223)
# plt.plot(sla_list[3])
# # plt.yticks([])
# plt.title('SLA')
# plt.subplot(224)
# plt.plot(sla_deriv[3])
# # plt.yticks([])
# plt.title('Rate of SLA Change')
# plt.show()






####PLOT####
fig, axs = plt.subplots(1, 4, sharex=True)
axs[0].set_ylabel('SLA')

axs[0].plot(sla_list[0])
axs[0].plot(np.arange(len(sla_list[0])),np.zeros(len(sla_list[0])))

axs[1].plot(sla_list[1])
axs[1].plot(np.arange(len(sla_list[1])),np.zeros(len(sla_list[1])))

axs[2].plot(sla_list[2])
axs[2].plot(np.arange(len(sla_list[2])),np.zeros(len(sla_list[2])))

axs[3].plot(sla_list[3])
axs[3].plot(np.arange(len(sla_list[3])),np.zeros(len(sla_list[3])))

custom_ylim = (-0.4, 0.1)
plt.setp(axs, ylim=custom_ylim)
plt.show()




















###PLOT####
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

# custom_ylim = (-0.25, 0.1)
# plt.setp(axs[1,:], ylim=custom_ylim)
# fig.suptitle('Trials with clear trends (+2 Delta case)')
# plt.show()
