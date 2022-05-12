from http.client import MOVED_PERMANENTLY
from shutil import move
import numpy as np
import readFiles as RF
import matplotlib.pyplot as plt
import controllerInferenceFunc as CIF
import utilityFunc as UF

file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_splitbelt/'
file_info = RF.extract_file_info(file_dir)


info_result = []
fastgainX_result = []
fastgainY_result = []

slowgain_result = []

for file in file_info:
    file_name = file[0]+'.npz'
    file_header = file[1]
    subject, delta, fast_leg = CIF.extract_trial_info(file_name)

    if delta == 4: 
        print(file_name)
        npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_splitbelt/'

        com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 12, 45)
        gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
        com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
        com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
        foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)

        subject, delta, fast_leg = CIF.extract_trial_info(file_name)
        delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)

        fast_gain_total_x = np.empty((45,5))
        fast_gain_total_y = np.empty((45,5))

        for ii in np.arange(45):
            start, end = CIF.find_evaluting_range_idx(ii, 1, gait_event_idx)
            if fast_leg == 'right':
                r2_fast_x, r2_fast_y, gain_fast_x, gain_fast_y = CIF.controller_fit(delta_P_r[start:end,:], delta_Q_r[start:end,:])
                r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y = CIF.controller_fit(delta_P_l[start:end,:], delta_Q_l[start:end,:])
            else:
                r2_fast_x, r2_fast_y, gain_fast_x, gain_fast_y = CIF.controller_fit(delta_P_l[start:end,:], delta_Q_l[start:end,:])
                r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y = CIF.controller_fit(delta_P_r[start:end,:], delta_Q_r[start:end,:])

            # print(gain_fast_x.shape)
            fast_gain_total_x[ii,:] = gain_slow_x
            fast_gain_total_y[ii,:] = gain_slow_y

            # fast_gain = np.concatenate(fast_gain, gain_fast_x)
            # fast_gain.append([gain_fast_x, gain_fast_y])
            # slow_gain.append([gain_slow_x, gain_slow_y])

        # subject_info = [subject, delta, fast_leg]
        # info_result.append(subject_info)
        fastgainX_result.append(fast_gain_total_x)
        fastgainY_result.append(fast_gain_total_y)

        # slowgain_result.append(slow_gain)
        # result.append([subject_info, fast_gain, slow_gain])


# plt.plot(fastgain_result[0][:,0])
# plt.show()

fig, axs = plt.subplots(2,5)
for ii in np.arange(5):
    axs[0,0].plot(fastgainX_result[ii][:,0])
    axs[0,1].plot(fastgainX_result[ii][:,1])
    axs[0,2].plot(fastgainX_result[ii][:,2])
    axs[0,3].plot(fastgainX_result[ii][:,3])
    axs[0,4].plot(fastgainX_result[ii][:,4])

    axs[1,0].plot(fastgainY_result[ii][:,0])
    axs[1,1].plot(fastgainY_result[ii][:,1])
    axs[1,2].plot(fastgainY_result[ii][:,2])
    axs[1,3].plot(fastgainY_result[ii][:,3])
    axs[1,4].plot(fastgainY_result[ii][:,4])

plt.setp(axs[0,0], title='COM X Pos')
plt.setp(axs[0,1], title='COM Z Pos')
plt.setp(axs[0,2], title='COM X Vel')
plt.setp(axs[0,3], title='COM Y Vel')
plt.setp(axs[0,4], title='COM Z Vel')
plt.setp(axs[0,0], ylabel='Foot X')
plt.setp(axs[1,0], ylabel='Foot Y')
plt.setp(axs[1,2], xlabel='Time (min)')
fig.suptitle('Controller Gain as a function of adaptation for Slow Leg (Delta +2)')
plt.show()

# print(result[1])
# print(fast_gain[0][0][0])

# Plotting gain as a function of time

# fig, axs = plt.subplots(1,2)
# axs[0].plot(Xresult)
# axs[1].plot(Yresult)
# plt.show()
