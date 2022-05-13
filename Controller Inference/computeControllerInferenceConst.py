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

    if v_fast == 10:
        trial_info = (subject, delta, fast_leg, v_fast, v_slow)

        npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'

        com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 0, 6)
        gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
        com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
        com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
        foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
        delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)

        fast_gain_total_x = np.empty((45,5))
        fast_gain_total_y = np.empty((45,5))
        slow_gain_total_x = np.empty((45,5))
        slow_gain_total_y = np.empty((45,5))
        
        for ii in np.arange(45):
            start, end = CIF.find_evaluting_range_idx_segment(ii, 45, gait_event_idx)
            
            # print(str(start)+" "+str(gait_event_idx.shape)+" "+str(gait_event_idx[-1,0]))
            r2_fast_x, r2_fast_y, gain_fast_x, gain_fast_y = CIF.controller_fit(delta_P_r[start:end,:], delta_Q_r[start:end,:])
            r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y = CIF.controller_fit(delta_P_l[start:end,:], delta_Q_l[start:end,:])

            fast_gain_total_x[ii,:] = gain_fast_x
            fast_gain_total_y[ii,:] = gain_fast_y
            slow_gain_total_x[ii,:] = gain_slow_x
            slow_gain_total_y[ii,:] = gain_slow_y
        
        info_result.append(trial_info)
        fastgainX_result.append(fast_gain_total_x)
        fastgainY_result.append(fast_gain_total_y)
        slowgainX_result.append(slow_gain_total_x)
        slowgainY_result.append(slow_gain_total_y)


file = open('controllerInferenceConst_10.pkl','wb')
pickle.dump(info_result, file)
pickle.dump(fastgainX_result, file)
pickle.dump(fastgainY_result, file)
pickle.dump(slowgainX_result, file)
pickle.dump(slowgainY_result, file)
file.close()

# file = open('test.pkl', 'rb')
# obj_1 = pickle.load(file)
# obj_2 = pickle.load(file)
# obj_3 = pickle.load(file)
# print(obj_1)
# print(obj_2)
# print(obj_3)
# file.close()

fig, axs = plt.subplots(2,10)
for ii in np.arange(5):
    axs[0,0].plot(fastgainX_result[ii][:,0],'o')
    axs[0,1].plot(fastgainX_result[ii][:,1],'o')
    axs[0,2].plot(fastgainX_result[ii][:,2],'o')
    axs[0,3].plot(fastgainX_result[ii][:,3],'o')
    axs[0,4].plot(fastgainX_result[ii][:,4],'o')
    axs[0,5].plot(slowgainX_result[ii][1:,0],'o')
    axs[0,6].plot(slowgainX_result[ii][1:,1],'o')
    axs[0,7].plot(slowgainX_result[ii][1:,2],'o')
    axs[0,8].plot(slowgainX_result[ii][1:,3],'o')
    axs[0,9].plot(slowgainX_result[ii][1:,4],'o')

    axs[1,0].plot(fastgainY_result[ii][:,0],'o')
    axs[1,1].plot(fastgainY_result[ii][:,1],'o')
    axs[1,2].plot(fastgainY_result[ii][:,2],'o')
    axs[1,3].plot(fastgainY_result[ii][:,3],'o')
    axs[1,4].plot(fastgainY_result[ii][:,4],'o')
    axs[1,5].plot(slowgainY_result[ii][1:,0],'o')
    axs[1,6].plot(slowgainY_result[ii][1:,1],'o')
    axs[1,7].plot(slowgainY_result[ii][1:,2],'o')
    axs[1,8].plot(slowgainY_result[ii][1:,3],'o')
    axs[1,9].plot(slowgainY_result[ii][1:,4],'o')

plt.setp(axs[0,0], title='COM X Pos')
plt.setp(axs[0,1], title='COM Z Pos')
plt.setp(axs[0,2], title='Right Leg \n COM X Vel')
plt.setp(axs[0,3], title='COM Y Vel')
plt.setp(axs[0,4], title='COM Z Vel')

plt.setp(axs[0,5], title='COM X Pos')
plt.setp(axs[0,6], title='COM Z Pos')
plt.setp(axs[0,7], title='Left Leg \n COM X Vel')
plt.setp(axs[0,8], title='COM Y Vel')
plt.setp(axs[0,9], title='COM Z Vel')
plt.setp(axs[0,0], ylabel='Foot X')
plt.setp(axs[1,0], ylabel='Foot Y')
fig.suptitle('Step-to-step controller gain for a tied-belt condition (1 m/s case)')
plt.show()

# print(result[1])
# print(fast_gain[0][0][0])

# Plotting gain as a function of time
# fig, axs = plt.subplots(1,2)
# axs[0].plot(Xresult)
# axs[1].plot(Yresult)
# plt.show()
