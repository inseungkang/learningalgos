import numpy as np
import pickle
import readFiles as RF
import matplotlib.pyplot as plt
import controllerInferenceFunc as CIF
import utilityFunc as UF

file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_tiedbelt/'
file_info = RF.extract_file_info(file_dir)

info_result = []
fastgainX_result = []
fastgainY_result = []
slowgainX_result = []
slowgainY_result = []

for file in file_info:
    file_name = file[0]+'.npz'
    file_header = file[1]
    subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name)

    if v_fast == 12:
        trial_info = (subject, delta, fast_leg, v_fast, v_slow)
        npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'

        com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 0, 6)
        gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
        com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
        com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
        foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
        delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)
        
        # print(str(start)+" "+str(gait_event_idx.shape)+" "+str(gait_event_idx[-1,0]))
        r2_fast_x, r2_fast_y, gain_fast_x, gain_fast_y = CIF.controller_fit(delta_P_r, delta_Q_r)
        r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y = CIF.controller_fit(delta_P_l, delta_Q_l)
        
        info_result.append(trial_info)
        fastgainX_result.append(gain_fast_x)
        fastgainY_result.append(gain_fast_y)
        slowgainX_result.append(gain_slow_x)
        slowgainY_result.append(gain_slow_y)


file = open('controllerInferenceConst_12.pkl','wb')
pickle.dump(info_result, file)
pickle.dump(fastgainX_result, file)
pickle.dump(fastgainY_result, file)
pickle.dump(slowgainX_result, file)
pickle.dump(slowgainY_result, file)
file.close()

# print(result[1])
# print(fast_gain[0][0][0])

# Plotting gain as a function of time
# fig, axs = plt.subplots(1,2)
# axs[0].plot(Xresult)
# axs[1].plot(Yresult)
# plt.show()
