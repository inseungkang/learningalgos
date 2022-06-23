import enum
import numpy as np
import pickle
import readFiles as RF
import matplotlib.pyplot as plt
import controllerInferenceFunc as CIF
import utilityFunc as UF
import os
import c3d

file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_tiedbelt/'
npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'
file_info = RF.extract_file_info(file_dir)

for file in file_info:
# file = file_info[2]
        file_name = file[0]+'.npz'
        file_header = file[1]
        print(file_name)

        subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name)
        trial_info = (subject, delta, fast_leg, v_fast, v_slow)

        com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 0, 6)
        gait_event_idx, ms_idx_r, ms_idx_l, hc_idx_r, hc_idx_l = CIF.GRF_based_gait_event_idx(com, foot_r, foot_l, force_r, force_l)

        plt.plot(gait_event_idx - np.expand_dims(gait_event_idx[:,0], axis=1))
        plt.show()


