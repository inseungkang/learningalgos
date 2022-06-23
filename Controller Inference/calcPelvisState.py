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
        file_name = file[0]+'.npz'
        file_header = file[1]

        if file_name == 'AB3_Session2_Right10_Left6.npz':
                file_header = [3, 2, 0, 1, 30, 33]

        subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name)
        trial_info = (subject, delta, fast_leg, v_fast, v_slow)

        # print(file_name)
        # npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_splitbelt/'
        com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 0, 6)
        gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
        right_hc = gait_event_idx[3:,1]
        right_ms = gait_event_idx[3:,2]
        signal_every_gc = []
        mid_stance_idx = []

        plt.plot(gait_event_idx - np.expand_dims(gait_event_idx[:,0],axis=1))
        plt.show()
        for i in np.arange(len(right_hc)-1):
                idx_1 = int(right_hc[i])
                idx_2 = int(right_hc[i+1])
                signal_every_gc.append(com[idx_1:idx_2, 2])
                mid_stance_idx.append(int(right_ms[i]) - int(right_hc[i]))

                
        # print(len(mid_stance_idx))
        # for i, x in enumerate(signal_every_gc):
        #         plt.plot(x)
        #         plt.plot(mid_stance_idx[i], x[mid_stance_idx[i]], 'x')
        #         plt.title(file_name)
        # plt.show()







# file_name = 'AB1_Session1_Right6_Left6.npz'
# c3d_name = 'AB1_Session1_Right6_Left6.c3d'
# file_header = [34, 2, 0, 1, 20, 26]


# file_header = RF.extract_marker_column(file_dir+c3d_name)

# # file_name = 'AB5_Session1_Right14_Left14.npz'
# # file_header = [2, 35, 0, 1, 21, 27]

# subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name)
# com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 0, 6)
# gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)

# right_hc = gait_event_idx[:,1]

# test_vec = []
# for i in np.arange(len(right_hc)-1):
#         idx_1 = int(right_hc[i])
#         idx_2 = int(right_hc[i+1])
#         test_vec.append(com[idx_1:idx_2,2])

# for i, x in enumerate(test_vec):
#         plt.plot(x)
#         plt.title(file_name)
# plt.show()




# f0 = 3.5  # Frequency to be removed from signal (Hz)
# Q = 30.0  # Quality factor
# b, a = signal.iirnotch(f0, Q, fs)
# outputSignal = signal.filtfilt(b, a, outputSignal)

# f0 = 1.75  # Frequency to be removed from signal (Hz)
# Q = 30.0  # Quality factor
# b, a = signal.iirnotch(f0, Q, fs)
# outputSignal = signal.filtfilt(b, a, outputSignal)


# plt.plot(testing_signal)
# # plt.plot(outputSignal)

# plt.show()


# class trial_info:
#   def __init__(self, subject, delta, fastleg, speed_fastleg, speed_slowleg):
#     self.subject = subject
#     self.delta = delta
#     self.fastleg = fastleg
#     self.speed_fastleg = speed_fastleg
#     self.speed_slowleg = speed_slowleg

# information = trial_info(subject, delta, fast_leg, v_fast, v_slow)
# print(information.delta)


# plt.plot(foot_r)
# plt.show()



# print(com.shape)







        # com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
        # com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
        # foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
        # delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)





# mass = 60
# right_z = -force_r[:,2]
# left_z = -force_l[:,2]
# acc_z = (right_z+left_z)/(mass*9.81) - 9.81
# # acc_z = acc_z - np.mean(acc_z)
# # /mass - 9.81
# import scipy.integrate as integrate
# import scipy as sp
# time = np.linspace(0, len(acc_z)/100, len(acc_z))
# print(time[-1])
# vel_z = sp.integrate.cumtrapz(acc_z, x=time)
# # vel_z_new = sp.signal.detrend(vel_z,type='linear')
# pos_z = sp.integrate.cumtrapz(vel_z, x=time[:-1])
# # pos_z_new = sp.signal.detrend(pos_z)
# # plt.plot(acc_z)
# plt.plot(vel_z)
# plt.show()
# plt.plot(acc_z)
# plt.show()

# gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
# com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
# com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
# foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
# delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)
        
        # r2_fast_x, r2_fast_y, gain_fast_x, gain_fast_y = CIF.controller_fit(delta_P_r, delta_Q_r)
        # r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y = CIF.controller_fit(delta_P_l, delta_Q_l)
