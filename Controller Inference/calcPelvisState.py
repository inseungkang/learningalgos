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

file_name = 'AB1_Session1_Right12_Left12.npz' 
file_header = [34, 2, 0, 1, 19, 25]
mass = 70
subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name)
com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 0, 6)

right_z = -force_r[:,2]
left_z = -force_l[:,2]
acc_z = (right_z+left_z)/mass - 9.81
acc_z = acc_z - np.mean(acc_z)
# /mass - 9.81
import scipy.integrate as integrate
import scipy as sp
time = np.linspace(0, len(acc_z)/100, len(acc_z))
print(time[-1])
vel_z = sp.integrate.cumtrapz(acc_z, x=time)
vel_z_new = sp.signal.detrend(vel_z,type='linear')
pos_z = sp.integrate.cumtrapz(vel_z_new, x=time[:-1])
pos_z_new = sp.signal.detrend(pos_z)
# plt.plot(acc_z)
plt.plot(vel_z)
plt.show()
# plt.plot(acc_z)
# plt.show()

# gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
# com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
# com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
# foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
# delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)
        
        # r2_fast_x, r2_fast_y, gain_fast_x, gain_fast_y = CIF.controller_fit(delta_P_r, delta_Q_r)
        # r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y = CIF.controller_fit(delta_P_l, delta_Q_l)
