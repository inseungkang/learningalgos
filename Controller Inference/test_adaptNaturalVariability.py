import numpy as np
import pickle
import readFiles as RF
import matplotlib.pyplot as plt
import controllerInferenceFunc as CIF
import utilityFunc as UF

file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_tiedbelt/'
file_info = RF.extract_file_info(file_dir)

for file in file_info:
    file_name = file[0]+'.npz'
    file_header = file[1]
    subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name)

    if v_fast == 14 and subject == 3:

        trial_info = (subject, delta, fast_leg, v_fast, v_slow)
        npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'
        print(file_name)

        com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 0, 6)
        gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
        com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
        com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
        foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
        delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)
        
        # print(str(start)+" "+str(gait_event_idx.shape)+" "+str(gait_event_idx[-1,0]))
        r2_fast_x, r2_fast_y, gain_fast_x, gain_fast_y, fast_model_x, fast_model_y = CIF.controller_fit(delta_P_l, delta_Q_l)
        # r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y, slow_model_x, slow_model_y = CIF.controller_fit(delta_P_l, delta_Q_l)

        fast_natural_var_x = fast_model_x.predict(delta_P_l) - delta_Q_l[:,0]
        fast_natural_var_y = fast_model_y.predict(delta_P_l) - delta_Q_l[:,1]
        fast_var_mean_x = np.mean(fast_natural_var_x)
        fast_var_mean_y = np.mean(fast_natural_var_y)
        fast_var_std_x = np.std(fast_natural_var_x)
        fast_var_std_y = np.std(fast_natural_var_y)

    if v_fast == 10 and subject == 3:

        trial_info = (subject, delta, fast_leg, v_fast, v_slow)
        npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'
        print(file_name)

        com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 0, 6)
        gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
        com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
        com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
        foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
        delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)
        
        # print(str(start)+" "+str(gait_event_idx.shape)+" "+str(gait_event_idx[-1,0]))
        r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y, slow_model_x, slow_model_y = CIF.controller_fit(delta_P_r, delta_Q_r)

        slow_natural_var_x = slow_model_x.predict(delta_P_r) - delta_Q_r[:,0]
        slow_natural_var_y = slow_model_y.predict(delta_P_r) - delta_Q_r[:,1]
        slow_var_mean_x = np.mean(slow_natural_var_x)
        slow_var_mean_y = np.mean(slow_natural_var_y)
        slow_var_std_x = np.std(slow_natural_var_x)
        slow_var_std_y = np.std(slow_natural_var_y)






file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_splitbelt/'
file_info = RF.extract_file_info(file_dir)

info_result = []
fastgainX_result = []
fastgainY_result = []
slowgainX_result = []
slowgainY_result = []

for file in file_info:
    file_name = file[0]+'.npz'
    file_header = file[1]

    if file_name == 'AB3_Session2_Right10_Left6.npz':
        file_header = [3, 2, 0, 1, 30, 33]

    subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name)
    trial_info = (subject, delta, fast_leg, v_fast, v_slow)
    if delta == 4 and subject == 3:
        print(file_name)
        print(fast_leg)
        npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_splitbelt/'

        com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 12, 45)
        gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
        com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
        com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
        foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
        delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)

        start_1, end_1 = CIF.find_evaluting_range_idx_time(0, 2, gait_event_idx)
        start_2, end_2 = CIF.find_evaluting_range_idx_time(43, 2, gait_event_idx)

fast_pred_x_1 = fast_model_x.predict(delta_P_l[start_1:end_1,:]) - delta_Q_l[start_1:end_1,0]
fast_pred_y_1 = fast_model_y.predict(delta_P_l[start_1:end_1,:]) - delta_Q_l[start_1:end_1,1]
slow_pred_x_1 = slow_model_x.predict(delta_P_r[start_1:end_1,:]) - delta_Q_r[start_1:end_1,0]
slow_pred_y_1 = slow_model_y.predict(delta_P_r[start_1:end_1,:]) - delta_Q_r[start_1:end_1,1]

fast_pred_x_2 = fast_model_x.predict(delta_P_l[start_2:end_2,:]) - delta_Q_l[start_2:end_2,0]
fast_pred_y_2 = fast_model_y.predict(delta_P_l[start_2:end_2,:]) - delta_Q_l[start_2:end_2,1]
slow_pred_x_2 = slow_model_x.predict(delta_P_r[start_2:end_2,:]) - delta_Q_r[start_2:end_2,0]
slow_pred_y_2 = slow_model_y.predict(delta_P_r[start_2:end_2,:]) - delta_Q_r[start_2:end_2,1]

split_fast_var_mean_x_1 = np.mean(fast_pred_x_1)
split_fast_var_mean_y_1 = np.mean(fast_pred_y_1)
split_fast_var_std_x_1 = np.std(fast_pred_x_1)
split_fast_var_std_y_1 = np.std(fast_pred_y_1)
split_slow_var_mean_x_1 = np.mean(slow_pred_x_1)
split_slow_var_mean_y_1 = np.mean(slow_pred_y_1)
split_slow_var_std_x_1 = np.std(slow_pred_x_1)
split_slow_var_std_y_1 = np.std(slow_pred_y_1)

split_fast_var_mean_x_2 = np.mean(fast_pred_x_2)
split_fast_var_mean_y_2 = np.mean(fast_pred_y_2)
split_fast_var_std_x_2 = np.std(fast_pred_x_2)
split_fast_var_std_y_2 = np.std(fast_pred_y_2)
split_slow_var_mean_x_2 = np.mean(slow_pred_x_2)
split_slow_var_mean_y_2 = np.mean(slow_pred_y_2)
split_slow_var_std_x_2 = np.std(slow_pred_x_2)
split_slow_var_std_y_2 = np.std(slow_pred_y_2)


index = ['Baseline','Start','End']

fig, axs = plt.subplots(2,2)
fig.suptitle('Foot placement exploration during split-belt walking (+2 delta, AB3)')
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


# print(fast_model_x(delta_P_r))




# fig, axs = plt.subplots(2,2)
# axs[0,0].plot(fast_delta_pred_x)
# axs[1,0].plot(fast_delta_pred_y)
# axs[0,1].plot(slow_delta_pred_x)
# axs[1,1].plot(slow_delta_pred_y)

# plt.setp(axs[0,0], ylabel='X Foot Placement (mm)')
# plt.setp(axs[1,0], ylabel='Y Foot Placement (mm)')
# plt.setp(axs[0,0], title='fast leg (1.4)')
# plt.setp(axs[0,1], title='slow leg (1.0)')
# # plt.setp(axs[1,0], xlabel='Number of Strides')
# fig.suptitle('AB3 Split-belt walking 2 delta exploration')
# plt.show()

# print(result[1])
# print(fast_gain[0][0][0])

# Plotting gain as a function of time
# fig, axs = plt.subplots(1,2)
# axs[0].plot(Xresult)
# axs[1].plot(Yresult)
# plt.show()




# # del const_6_fastgainX_result[4]
# # del const_6_slowgainX_result[4]
# # del const_8_fastgainY_result[3]
# # del const_8_slowgainY_result[3]
# # del const_10_fastgainX_result[3]
# # del const_10_slowgainX_result[3]
# # del const_10_fastgainY_result[3]
# # del const_10_slowgainY_result[3]

# # del const_12_fastgainY_result[3]
# # del const_12_slowgainY_result[3]
# # del const_14_fastgainY_result[3]
# # del const_14_slowgainY_result[3]

# const6_fast_X = UF.list_to_array(const_6_fastgainX_result)
# const6_slow_X = UF.list_to_array(const_6_slowgainX_result)
# const6_fast_X_avg = np.mean(const6_fast_X, axis=1)
# const6_slow_X_avg = np.mean(const6_slow_X, axis=1)
# const6_fast_Y = UF.list_to_array(const_6_fastgainY_result)
# const6_slow_Y = UF.list_to_array(const_6_slowgainY_result)
# const6_fast_Y_avg = np.mean(const6_fast_Y, axis=1)
# const6_slow_Y_avg = np.mean(const6_slow_Y, axis=1)

# const8_fast_X = UF.list_to_array(const_8_fastgainX_result)
# const8_slow_X = UF.list_to_array(const_8_slowgainX_result)
# const8_fast_X_avg = np.mean(const8_fast_X, axis=1)
# const8_slow_X_avg = np.mean(const8_slow_X, axis=1)
# const8_fast_Y = UF.list_to_array(const_8_fastgainY_result)
# const8_slow_Y = UF.list_to_array(const_8_slowgainY_result)
# const8_fast_Y_avg = np.mean(const8_fast_Y, axis=1)
# const8_slow_Y_avg = np.mean(const8_slow_Y, axis=1)

# const10_fast_X = UF.list_to_array(const_10_fastgainX_result)
# const10_slow_X = UF.list_to_array(const_10_slowgainX_result)
# const10_fast_X_avg = np.mean(const10_fast_X, axis=1)
# const10_slow_X_avg = np.mean(const10_slow_X, axis=1)
# const10_fast_Y = UF.list_to_array(const_10_fastgainY_result)
# const10_slow_Y = UF.list_to_array(const_10_slowgainY_result)
# const10_fast_Y_avg = np.mean(const10_fast_Y, axis=1)
# const10_slow_Y_avg = np.mean(const10_slow_Y, axis=1)

# const12_fast_X = UF.list_to_array(const_12_fastgainX_result)
# const12_slow_X = UF.list_to_array(const_12_slowgainX_result)
# const12_fast_X_avg = np.mean(const12_fast_X, axis=1)
# const12_slow_X_avg = np.mean(const12_slow_X, axis=1)
# const12_fast_Y = UF.list_to_array(const_12_fastgainY_result)
# const12_slow_Y = UF.list_to_array(const_12_slowgainY_result)
# const12_fast_Y_avg = np.mean(const12_fast_Y, axis=1)
# const12_slow_Y_avg = np.mean(const12_slow_Y, axis=1)

# const14_fast_X = UF.list_to_array(const_14_fastgainX_result)
# const14_slow_X = UF.list_to_array(const_14_slowgainX_result)
# const14_fast_X_avg = np.mean(const14_fast_X, axis=1)
# const14_slow_X_avg = np.mean(const14_slow_X, axis=1)
# const14_fast_Y = UF.list_to_array(const_14_fastgainY_result)
# const14_slow_Y = UF.list_to_array(const_14_slowgainY_result)
# const14_fast_Y_avg = np.mean(const14_fast_Y, axis=1)
# const14_slow_Y_avg = np.mean(const14_slow_Y, axis=1)

# const10_X = (const10_slow_X_avg + const10_fast_X_avg)/2
# const14_X = (const14_slow_X_avg + const14_fast_X_avg)/2
# const10_Y = (const10_slow_Y_avg + const10_fast_Y_avg)/2
# const14_Y = (const14_slow_Y_avg + const14_fast_Y_avg)/2

# fig, axs = plt.subplots(5,4)
# for ii, xx in enumerate([6, 8, 10, 12, 14]):
#     nameX = [const6_fast_X, const8_fast_X, const10_fast_X, const12_fast_X, const14_fast_X,
#     const6_slow_X, const8_slow_X, const10_slow_X, const12_slow_X, const14_slow_X]
#     nameY = [const6_fast_Y, const8_fast_Y, const10_fast_Y, const12_fast_Y, const14_fast_Y,
#     const6_slow_Y, const8_slow_Y, const10_slow_Y, const12_slow_Y, const14_slow_Y]

#     axs[ii,0].plot(nameX[ii], 'o-')
#     axs[ii,1].plot(nameX[ii+5], 'o-')
#     axs[ii,2].plot(nameY[ii], 'o-')
#     axs[ii,3].plot(nameY[ii+5], 'o-')


# plt.setp(axs[0,0], title='FootX Right')
# plt.setp(axs[0,1], title='FootX Left')
# plt.setp(axs[0,2], title='FootY Right')
# plt.setp(axs[0,3], title='FootY Left')

# plt.setp(axs[0,0], ylabel='0.6 m/s')
# plt.setp(axs[1,0], ylabel='0.8 m/s')
# plt.setp(axs[2,0], ylabel='1.0 m/s')
# plt.setp(axs[3,0], ylabel='1.2 m/s')
# plt.setp(axs[4,0], ylabel='1.4 m/s')
# plt.legend(['AB1','AB2','AB3','AB4','AB5'], ncol=5, bbox_to_anchor=(0.2, -0.4))
# fig.suptitle('Controller gains across different walking speeds')
# plt.show()

# x = np.arange(45)
# plt.plot(x, np.ones(len(x))*const14_fast_Y_avg[0])
# plt.show()



# fig, axs = plt.subplots(2,5)
# for iii in np.arange(5):
#     x = np.arange(45)
#     # fast X
#     test_vec = np.empty((len(adapt_fastgainX_result[0][:,iii]),1))
#     for ii in np.arange(5):
#         vec = np.expand_dims(np.array(adapt_fastgainX_result[ii][:,iii]), axis=1)
#         test_vec = np.concatenate((test_vec, vec), axis=1)
#     test_vec = test_vec[:,1:]
#     test_vec_avg = np.mean(test_vec, axis=1)
#     test_vec_std = np.std(test_vec, axis=1)

#     axs[0,iii].errorbar(x, test_vec_avg, yerr=test_vec_std, fmt='o', markersize=2.5, color='darkred', ecolor='lightcoral', elinewidth=1.5, capsize=0)
#     # fast Y
#     test_vec = np.empty((len(adapt_fastgainY_result[0][:,iii]),1))
#     for ii in np.arange(5):
#         vec = np.expand_dims(np.array(adapt_fastgainY_result[ii][:,iii]), axis=1)
#         test_vec = np.concatenate((test_vec, vec), axis=1)
#     test_vec = test_vec[:,1:]
#     test_vec_avg = np.mean(test_vec, axis=1)
#     test_vec_std = np.std(test_vec, axis=1)

#     axs[1,iii].errorbar(x, test_vec_avg, yerr=test_vec_std, fmt='o', markersize=2.5, color='darkred', ecolor='lightcoral', elinewidth=1.5, capsize=0)


#     x = np.arange(45)+0.5
#     # Slow X
#     test_vec = np.empty((len(adapt_slowgainX_result[0][:,iii]),1))
#     for ii in np.arange(5):
#         vec = np.expand_dims(np.array(adapt_slowgainX_result[ii][:,iii]), axis=1)
#         test_vec = np.concatenate((test_vec, vec), axis=1)
#     test_vec = test_vec[:,1:]
#     test_vec_avg = np.mean(test_vec, axis=1)
#     test_vec_std = np.std(test_vec, axis=1)


#     axs[0,iii].errorbar(x, test_vec_avg, yerr=test_vec_std, fmt='o', markersize=2.5, color='darkblue', ecolor='lightsteelblue', elinewidth=1.5, capsize=0)
#     # Slow Y
#     test_vec = np.empty((len(adapt_slowgainY_result[0][:,iii]),1))
#     for ii in np.arange(5):
#         vec = np.expand_dims(np.array(adapt_slowgainY_result[ii][:,iii]), axis=1)
#         test_vec = np.concatenate((test_vec, vec), axis=1)
#     test_vec = test_vec[:,1:]
#     test_vec_avg = np.mean(test_vec, axis=1)
#     test_vec_std = np.std(test_vec, axis=1)

#     axs[1,iii].errorbar(x, test_vec_avg, yerr=test_vec_std, fmt='o', markersize=2.5, color='darkblue', ecolor='lightsteelblue', elinewidth=1.5, capsize=0)
#     axs[0,iii].plot(x, np.ones(len(x))*const10_X[iii],color='k')
#     axs[1,iii].plot(x, np.ones(len(x))*const10_Y[iii],color='k')

# plt.setp(axs[0,0], title='COM X Pos')
# plt.setp(axs[0,1], title='COM Z Pos')
# plt.setp(axs[0,2], title='COM X Vel')
# plt.setp(axs[0,3], title='COM Y Vel')
# plt.setp(axs[0,4], title='COM Z Vel')

# plt.setp(axs[0,0], ylabel='Foot X')
# plt.setp(axs[1,0], ylabel='Foot Y')
# fig.suptitle('Step-to-step controller gain as a function of locomotor adaptation (+2 Delta Case)')
# plt.legend(['Tied-Belt Baseline','Fast Leg', 'Slow Leg'])
# # ax = plt.gca()
# # ax.spines['right'].set_color('none')
# # ax.spines['top'].set_color('none')
# plt.show()


# file = open('controllerInferenceConst_14.pkl', 'rb')
# const_14_info_result = pickle.load(file)
# const_14_fastgainX_result = pickle.load(file)
# const_14_fastgainY_result = pickle.load(file)
# const_14_slowgainX_result = pickle.load(file)
# const_14_slowgainY_result = pickle.load(file)
# file.close()

# fig, axs = plt.subplots(2,10)
# for ii in np.arange(5):
#     if ii == 0:
#         color = 'r'
#     elif ii == 1:
#         color = 'g'
#     elif ii == 2:
#         color = 'b'
#     elif ii == 3:
#         color = 'm'
#     elif ii == 4:
#         color = 'y'

#     axs[0,0].plot(adapt_fastgainX_result[ii][:,0],'.', color=color)
#     axs[0,1].plot(adapt_fastgainX_result[ii][:,1],'.', color=color)
#     axs[0,2].plot(adapt_fastgainX_result[ii][:,2],'.', color=color)
#     axs[0,3].plot(adapt_fastgainX_result[ii][:,3],'.', color=color)
#     axs[0,4].plot(adapt_fastgainX_result[ii][:,4],'.', color=color)
#     axs[0,5].plot(adapt_slowgainX_result[ii][:,0],'.', color=color)
#     axs[0,6].plot(adapt_slowgainX_result[ii][:,1],'.', color=color)
#     axs[0,7].plot(adapt_slowgainX_result[ii][:,2],'.', color=color)
#     axs[0,8].plot(adapt_slowgainX_result[ii][:,3],'.', color=color)
#     axs[0,9].plot(adapt_slowgainX_result[ii][:,4],'.', color=color)
#     axs[0,0].plot(const_14_fastgainX_result[ii][:,0],'x', color=color)
#     axs[0,1].plot(const_14_fastgainX_result[ii][:,1],'x', color=color)
#     axs[0,2].plot(const_14_fastgainX_result[ii][:,2],'x', color=color)
#     axs[0,3].plot(const_14_fastgainX_result[ii][:,3],'x', color=color)
#     axs[0,4].plot(const_14_fastgainX_result[ii][:,4],'x', color=color)
#     axs[0,5].plot(const_10_fastgainX_result[ii][:,0],'x', color=color)
#     axs[0,6].plot(const_10_fastgainX_result[ii][:,1],'x', color=color)
#     axs[0,7].plot(const_10_fastgainX_result[ii][:,2],'x', color=color)
#     axs[0,8].plot(const_10_fastgainX_result[ii][:,3],'x', color=color)
#     axs[0,9].plot(const_10_fastgainX_result[ii][:,4],'x', color=color)

#     axs[1,0].plot(adapt_fastgainY_result[ii][:,0],'.', color=color)
#     axs[1,1].plot(adapt_fastgainY_result[ii][:,1],'.', color=color)
#     axs[1,2].plot(adapt_fastgainY_result[ii][:,2],'.', color=color)
#     axs[1,3].plot(adapt_fastgainY_result[ii][:,3],'.', color=color)
#     axs[1,4].plot(adapt_fastgainY_result[ii][:,4],'.', color=color)
#     axs[1,5].plot(adapt_fastgainY_result[ii][:,0],'.', color=color)
#     axs[1,6].plot(adapt_fastgainY_result[ii][:,1],'.', color=color)
#     axs[1,7].plot(adapt_fastgainY_result[ii][:,2],'.', color=color)
#     axs[1,8].plot(adapt_fastgainY_result[ii][:,3],'.', color=color)
#     axs[1,9].plot(adapt_fastgainY_result[ii][:,4],'.', color=color)
#     axs[1,0].plot(const_14_fastgainY_result[ii][:,0],'x', color=color)
#     axs[1,1].plot(const_14_fastgainY_result[ii][:,1],'x', color=color)
#     axs[1,2].plot(const_14_fastgainY_result[ii][:,2],'x', color=color)
#     axs[1,3].plot(const_14_fastgainY_result[ii][:,3],'x', color=color)
#     axs[1,4].plot(const_14_fastgainY_result[ii][:,4],'x', color=color)
#     axs[1,5].plot(const_10_fastgainY_result[ii][:,0],'x', color=color)
#     axs[1,6].plot(const_10_fastgainY_result[ii][:,1],'x', color=color)
#     axs[1,7].plot(const_10_fastgainY_result[ii][:,2],'x', color=color)
#     axs[1,8].plot(const_10_fastgainY_result[ii][:,3],'x', color=color)
#     axs[1,9].plot(const_10_fastgainY_result[ii][:,4],'x', color=color)

# plt.setp(axs[0,0], title='COM X Pos')
# plt.setp(axs[0,1], title='COM Z Pos')
# plt.setp(axs[0,2], title='Fast Leg \n COM X Vel')
# plt.setp(axs[0,3], title='COM Y Vel')
# plt.setp(axs[0,4], title='COM Z Vel')

# plt.setp(axs[0,5], title='COM X Pos')
# plt.setp(axs[0,6], title='COM Z Pos')
# plt.setp(axs[0,7], title='Slow Leg \n COM X Vel')
# plt.setp(axs[0,8], title='COM Y Vel')
# plt.setp(axs[0,9], title='COM Z Vel')
# plt.setp(axs[0,0], ylabel='Foot X')
# plt.setp(axs[1,0], ylabel='Foot Y')
# fig.suptitle('Step-to-step controller gain as a function of locomotor adaptation (+2 Delta Case)')
# plt.show()