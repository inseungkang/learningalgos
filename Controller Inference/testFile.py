import numpy as np
import readFiles as RF
import matplotlib.pyplot as plt
import controllerInferenceFunc as CIF
import utilityFunc as UF

file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_splitbelt/'
file_info = RF.extract_file_info(file_dir)

file_name = file_info[4][0]+'.npz'
print(file_name)
file_header= file_info[4][1]
npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_splitbelt/'

com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 12, 45)
gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)

delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)

Xresult = []
Yresult = []
for ii in np.arange(45):
    start, end = CIF.find_evaluting_range_idx(ii, 1, gait_event_idx)
    R2_X, R2_Y, gain_X, gain_Y = CIF.controller_fit(delta_P_l[start:end,:], delta_Q_l[start:end,:])
    Xresult.append(R2_X)
    Yresult.append(R2_Y)


fig, axs = plt.subplots(1,2)
axs[0].plot(Xresult)
axs[1].plot(Yresult)
plt.show()

