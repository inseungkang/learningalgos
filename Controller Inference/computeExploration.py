from stat import S_ISDOOR
import numpy as np
import pickle
import readFiles as RF
import matplotlib.pyplot as plt
import controllerInferenceFunc as CIF
import utilityFunc as UF
import os





# for file in file_info:
#     file_name = file[0]+'.npz'
#     file_header = file[1]
#     subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name)

#     if v_fast == 14 and subject == 3:

#         trial_info = (subject, delta, fast_leg, v_fast, v_slow)
#         npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'
#         print(file_name)

#         com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 0, 6)
#         gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
#         com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
#         com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
#         foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
#         delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)
        
#         r2_fast_x, r2_fast_y, gain_fast_x, gain_fast_y, fast_model_x, fast_model_y = CIF.controller_fit(delta_P_l, delta_Q_l)

#         fast_natural_var_x = fast_model_x.predict(delta_P_l) - delta_Q_l[:,0]
#         fast_natural_var_y = fast_model_y.predict(delta_P_l) - delta_Q_l[:,1]
#         fast_var_mean_x = np.mean(fast_natural_var_x)
#         fast_var_mean_y = np.mean(fast_natural_var_y)
#         fast_var_std_x = np.std(fast_natural_var_x)
#         fast_var_std_y = np.std(fast_natural_var_y)





#     if v_fast == 10 and subject == 3:

#         trial_info = (subject, delta, fast_leg, v_fast, v_slow)
#         npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'
#         print(file_name)

#         com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 0, 6)
#         gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
#         com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
#         com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
#         foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
#         delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)
        
#         # print(str(start)+" "+str(gait_event_idx.shape)+" "+str(gait_event_idx[-1,0]))
#         r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y, slow_model_x, slow_model_y = CIF.controller_fit(delta_P_r, delta_Q_r)

#         slow_natural_var_x = slow_model_x.predict(delta_P_r) - delta_Q_r[:,0]
#         slow_natural_var_y = slow_model_y.predict(delta_P_r) - delta_Q_r[:,1]
#         slow_var_mean_x = np.mean(slow_natural_var_x)
#         slow_var_mean_y = np.mean(slow_natural_var_y)
#         slow_var_std_x = np.std(slow_natural_var_x)
#         slow_var_std_y = np.std(slow_natural_var_y)













split_file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_splitbelt/'
tied_file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_tiedbelt/'
split_npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_splitbelt/'
tied_npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'
for file in RF.extract_file_info(split_file_dir):
    file_name = file[0]+'.npz'
    file_header = file[1]

    if file_name == 'AB3_Session2_Right10_Left6.npz':
        file_header = [3, 2, 0, 1, 30, 33]

    subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name)
    trial_info = (subject, delta, fast_leg, v_fast, v_slow)

    if  subject == 1 and delta == 2:
        print(file_name)

        fastleg_filename = 'AB'+str(subject)+'_Session1_Right'+str(v_fast)+'_'+'Left'+str(v_fast)
        slowleg_filename = 'AB'+str(subject)+'_Session1_Right'+str(v_slow)+'_'+'Left'+str(v_slow)

        file_header_fast = RF.extract_marker_column(tied_file_dir+fastleg_filename+'.c3d')
        file_header_slow = RF.extract_marker_column(tied_file_dir+fastleg_filename+'.c3d')

        if fast_leg == 'right':
            delta_P_fast, delta_Q_fast, gait_idx = CIF.computeDeltaPandQ(tied_npz_dir+fastleg_filename+'.npz', file_header_fast, 'right', 0, 6)
            delta_P_slow, delta_Q_slow, gait_idx = CIF.computeDeltaPandQ(tied_npz_dir+slowleg_filename+'.npz', file_header_slow, 'left', 0, 6)
            split_delta_P_fast, split_delta_Q_fast, gait_idx = CIF.computeDeltaPandQ(split_npz_dir+file_name, file_header, 'right', 12, 45)
            split_delta_P_slow, split_delta_Q_slow, gait_idx = CIF.computeDeltaPandQ(split_npz_dir+file_name, file_header, 'left', 12, 45)

        elif fast_leg == 'left':
            delta_P_fast, delta_Q_fast, gait_idx = CIF.computeDeltaPandQ(tied_npz_dir+fastleg_filename+'.npz', file_header_fast, 'left', 0, 6)
            delta_P_slow, delta_Q_slow, gait_idx = CIF.computeDeltaPandQ(tied_npz_dir+slowleg_filename+'.npz', file_header_slow, 'right', 0, 6)       
            split_delta_P_fast, split_delta_Q_fast, gait_idx = CIF.computeDeltaPandQ(split_npz_dir+file_name, file_header, 'left', 12, 45)
            split_delta_P_slow, split_delta_Q_slow, gait_idx = CIF.computeDeltaPandQ(split_npz_dir+file_name, file_header, 'right', 12, 45)

        r2_fast_x, r2_fast_y, gain_fast_x, gain_fast_y, fast_model_x, fast_model_y = CIF.controller_fit(delta_P_fast, delta_Q_fast)
        r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y, slow_model_x, slow_model_y = CIF.controller_fit(delta_P_slow, delta_Q_slow)

        fast_var_std_x = np.std(fast_model_x.predict(delta_P_fast) - delta_Q_fast[:,0])
        fast_var_std_y = np.std(fast_model_y.predict(delta_P_fast) - delta_Q_fast[:,1])            
        slow_var_std_x = np.std(slow_model_x.predict(delta_P_slow) - delta_Q_slow[:,0])
        slow_var_std_y = np.std(slow_model_y.predict(delta_P_slow) - delta_Q_slow[:,1])  

        start_1, end_1 = CIF.find_evaluting_range_idx_time(0, 2, gait_idx)
        start_2, end_2 = CIF.find_evaluting_range_idx_time(43, 2, gait_idx)

        fast_pred_x_1 = fast_model_x.predict(split_delta_P_fast[start_1:end_1,:]) - split_delta_Q_fast[start_1:end_1,0]
        fast_pred_y_1 = fast_model_y.predict(split_delta_P_fast[start_1:end_1,:]) - split_delta_Q_fast[start_1:end_1,1]
        slow_pred_x_1 = slow_model_x.predict(split_delta_P_slow[start_1:end_1,:]) - split_delta_Q_slow[start_1:end_1,0]
        slow_pred_y_1 = slow_model_y.predict(split_delta_P_slow[start_1:end_1,:]) - split_delta_Q_slow[start_1:end_1,1]

        fast_pred_x_2 = fast_model_x.predict(split_delta_P_fast[start_2:end_2,:]) - split_delta_Q_fast[start_2:end_2,0]
        fast_pred_y_2 = fast_model_y.predict(split_delta_P_fast[start_2:end_2,:]) - split_delta_Q_fast[start_2:end_2,1]
        slow_pred_x_2 = slow_model_x.predict(split_delta_P_slow[start_2:end_2,:]) - split_delta_Q_slow[start_2:end_2,0]
        slow_pred_y_2 = slow_model_y.predict(split_delta_P_slow[start_2:end_2,:]) - split_delta_Q_slow[start_2:end_2,1]


        split_fast_var_std_x_1 = np.std(fast_pred_x_1)
        split_fast_var_std_y_1 = np.std(fast_pred_y_1)
        split_slow_var_std_x_1 = np.std(slow_pred_x_1)
        split_slow_var_std_y_1 = np.std(slow_pred_y_1)

        split_fast_var_std_x_2 = np.std(fast_pred_x_2)
        split_fast_var_std_y_2 = np.std(fast_pred_y_2)
        split_slow_var_std_x_2 = np.std(slow_pred_x_2)
        split_slow_var_std_y_2 = np.std(slow_pred_y_2)



index = ['Baseline','Start','End']

fig, axs = plt.subplots(2,2)
fig.suptitle('1 Delta Case')
axs[0,0].bar(index, [fast_var_std_x, split_fast_var_std_x_1, split_fast_var_std_x_2])
axs[0,0].set_title('Fast Leg Variability X')

axs[0,1].bar(index, [slow_var_std_x, split_slow_var_std_x_1, split_slow_var_std_x_2])
axs[0,1].set_title('Slow Leg Variability X')

axs[1,0].bar(index, [fast_var_std_y, split_fast_var_std_y_1, split_fast_var_std_y_2])
axs[1,0].set_title('Fast Leg Variability Y')

axs[1,1].bar(index, [slow_var_std_y, split_slow_var_std_y_1, split_slow_var_std_y_2])
axs[1,1].set_title('Slow Leg Variability Y')

axs[0,0].set_ylabel('Foot Placement (mm)')
axs[1,0].set_ylabel('Foot Placement (mm)')

# axs[0].errorbar(index, mean_x, yerr=std_x, fmt="o", color="r")
# axs[1].errorbar(index, mean_y, yerr=std_y, fmt="o", color="r")
plt.show()

        # file_list = []
        # for ii, name in enumerate(os.listdir(tied_file_dir)):
        #     if name != None and 'AB3' in name:
        #         file_list.append(name)



        # # tied_file_name = 'AB1_Session1_Right6_Left6.c3d'
        # # header_list = RF.extract_marker_column(file_dir+file_name)
        
        # for filename in file_list:
        #     if fastleg_name in filename:
        #         const_filename_fast = filename
        #     elif slowleg_name in filename:
        #         const_filename_slow = filename



