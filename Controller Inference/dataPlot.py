import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import readFiles as RF
import matplotlib.pyplot as plt
import controllerInferenceFunc as CIF
import utilityFunc as UF
import os
import c3d

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

data = np.load('metabolicData.npy')
movWindow = 180
dataRange = 2700-movWindow+1

dataFilt = [np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0)), np.empty((dataRange,0))]

for sub_idx in range(5):
    for del_idx in range(4):
        y_vec = np.round(data[del_idx][:,sub_idx].flatten(), 4)
        y_vec = moving_average(y_vec, movWindow)
        dataFilt[del_idx] = np.column_stack((dataFilt[del_idx], y_vec))




split_file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_splitbelt/'
tied_file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_tiedbelt/'
split_npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_splitbelt/'
tied_npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'

eval_time = 1
subject_result = []

test_list = os.listdir(split_file_dir)
for subjectname in ['AB3']:
# for subjectname in ['AB1', 'AB2', 'AB3', 'AB4', 'AB5']:
    newlist = []
    idxlist = []
    result_fast_x = []
    result_fast_y = []
    result_slow_x = []
    result_slow_y = []
    result_sum = []

    for file in test_list:
        if subjectname in file:
            newlist.append(file)
            subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file)
            delta_idx = CIF.delta_to_idx(delta)
            idxlist.append(delta_idx)
    newlist[:] = [newlist[i] for i in np.argsort(idxlist)]

    for file in newlist:
        file_name = file.split('.c3d')[0]+'.npz'
        file_header = RF.extract_marker_column(split_file_dir+file)
        print(file_name)

        if file_name == 'AB3_Session2_Right10_Left6.npz':
            file_header = [3, 2, 0, 1, 30, 33]

        subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name)
        trial_info = (subject, delta, fast_leg, v_fast, v_slow)


        delta_idx = CIF.delta_to_idx(delta)

        fastleg_filename = 'AB'+str(subject)+'_Session1_Right'+str(v_fast)+'_'+'Left'+str(v_fast)
        slowleg_filename = 'AB'+str(subject)+'_Session1_Right'+str(v_slow)+'_'+'Left'+str(v_slow)

        file_header_fast = RF.extract_marker_column(tied_file_dir+fastleg_filename+'.c3d')
        file_header_slow = RF.extract_marker_column(tied_file_dir+slowleg_filename+'.c3d')

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


        tied_fast_mean_x = np.mean(fast_model_x.predict(delta_P_fast)- delta_Q_fast[:,0])
        tied_fast_mean_y = np.mean(fast_model_y.predict(delta_P_fast)- delta_Q_fast[:,1])            
        tied_slow_mean_x = np.mean(slow_model_x.predict(delta_P_slow)- delta_Q_slow[:,0])
        tied_slow_mean_y = np.mean(slow_model_y.predict(delta_P_slow)- delta_Q_slow[:,1])  
        tied_fast_var_x = UF.rmse(fast_model_x.predict(delta_P_fast), delta_Q_fast[:,0])
        tied_fast_var_y = UF.rmse(fast_model_y.predict(delta_P_fast), delta_Q_fast[:,1])            
        tied_slow_var_x = UF.rmse(slow_model_x.predict(delta_P_slow), delta_Q_slow[:,0])
        tied_slow_var_y = UF.rmse(slow_model_y.predict(delta_P_slow), delta_Q_slow[:,1])  

        split_fast_rmse_x = np.empty((int(45/eval_time)))
        split_fast_rmse_y = np.empty((int(45/eval_time)))
        split_slow_rmse_x = np.empty((int(45/eval_time)))
        split_slow_rmse_y = np.empty((int(45/eval_time)))
        split_summed_rmse = np.empty((int(45/eval_time)))

        for ii in np.arange(int(45/eval_time)):
            start, end = CIF.find_evaluting_range_idx_time(ii, eval_time, gait_idx)
            split_fast_var_x = UF.rmse(fast_model_x.predict(split_delta_P_fast[start:end,:]), split_delta_Q_fast[start:end,0])
            split_fast_var_y = UF.rmse(fast_model_y.predict(split_delta_P_fast[start:end,:]), split_delta_Q_fast[start:end,1])
            split_slow_var_x = UF.rmse(slow_model_x.predict(split_delta_P_slow[start:end,:]), split_delta_Q_slow[start:end,0])
            split_slow_var_y = UF.rmse(slow_model_y.predict(split_delta_P_slow[start:end,:]), split_delta_Q_slow[start:end,1])
            # summed_data = split_fast_var_x + split_fast_var_y +split_slow_var_x + split_slow_var_y
            # summed_data = split_fast_var_y + split_slow_var_y
            summed_data = ((split_fast_var_x/tied_fast_var_x + split_fast_var_y/tied_fast_var_y + split_slow_var_x/tied_slow_var_x + split_slow_var_y/tied_slow_var_x)/4 - 1)*100

        
            # split_fast_rmse_x[ii] = split_fast_var_x-tied_fast_var_x
            # split_fast_rmse_y[ii] = split_fast_var_y-tied_fast_var_y
            # split_slow_rmse_x[ii] = split_slow_var_x-tied_slow_var_x
            # split_slow_rmse_y[ii] = split_slow_var_y-tied_slow_var_y
            # split_summed_rmse[ii] = summed_data
            split_fast_rmse_x[ii] = split_fast_var_x 
            split_fast_rmse_y[ii] = split_fast_var_y 
            split_slow_rmse_x[ii] = split_slow_var_x 
            split_slow_rmse_y[ii] = split_slow_var_y 
            split_summed_rmse[ii] = summed_data
        split_fast_rmse_x = np.sum(UF.filt_outlier(split_fast_rmse_x)- np.ones(45)*tied_fast_var_x)
        split_fast_rmse_y = np.sum(UF.filt_outlier(split_fast_rmse_y)- np.ones(45)*tied_fast_var_y)
        split_slow_rmse_x = np.sum(UF.filt_outlier(split_slow_rmse_x)- np.ones(45)*tied_slow_var_x)
        split_slow_rmse_y = np.sum(UF.filt_outlier(split_slow_rmse_y)- np.ones(45)*tied_slow_var_y)

        result_fast_x.append(split_fast_rmse_x)
        result_fast_y.append(split_fast_rmse_y)
        result_slow_x.append(split_slow_rmse_x)    
        result_slow_y.append(split_slow_rmse_y)

        result_sum.append(UF.filt_outlier(split_summed_rmse))

    subject_result.append([result_fast_x, result_fast_y, result_slow_x, result_slow_y])


# fig, axs = plt.subplots(5, 4, sharey=True)
# fig.suptitle('Foot placement exploration during split-belt walking')

# import seaborn as sns
# colors = sns.color_palette("coolwarm", 4)
# labels = ['-2D','-1D','+1D','+2D']
# for ii in np.arange(5):
#     axs[ii,0].bar(labels, subject_result[ii][0])
#     axs[ii,1].bar(labels, subject_result[ii][1])
#     axs[ii,2].bar(labels, subject_result[ii][2])
#     axs[ii,3].bar(labels, subject_result[ii][3])

#     # ax2 = axs[1].bar(subject_result[1][ii], '-o', markersize = 2.5, color=colors[ii], label=labels[ii])
#     # ax3 = axs[2].bar(subject_result[2][ii], '-o', markersize = 2.5, color=colors[ii], label=labels[ii])
#     # ax4 = axs[3].bar(subject_result[3][ii], '-o', markersize = 2.5, color=colors[ii], label=labels[ii])    
#     # ax5 = axs[4].bar(subject_result[4][ii], '-o', markersize = 2.5, color=colors[ii], label=labels[ii])
# axs[0,0].set_title('Fast X')
# axs[0,1].set_title('Fast Y')
# axs[0,2].set_title('Slow X') 
# axs[0,3].set_title('Slow Y') 

# axs[0,0].set_ylabel('AB1')
# axs[1,0].set_ylabel('AB2')
# axs[2,0].set_ylabel('AB3')
# axs[3,0].set_ylabel('AB4')
# axs[4,0].set_ylabel('AB5')

# plt.show()   

fig, axs = plt.subplots(3,4)

for ii in np.arange(4):
    axs[0,ii].plot(subject_result[0][ii])

    test = dataFilt[ii][:,0]
    testvec = []
    for iii in np.arange(45):
        value = iii*60+30
        if value > len(test):
            value = len(test)-1
        testvec.append(test[value])

    axs[1,ii].plot(testvec)
    axs[2,ii].plot(testvec, subject_result[0][ii], 'o')

axs[0, 0].set_ylabel('Exploration')
axs[1, 0].set_ylabel('Energetics')
axs[2, 0].set_ylabel('Correlation') 
axs[0, 0].set_title('-2 delta')
axs[0, 1].set_title('-1 delta')
axs[0, 2].set_title('+1 delta') 
axs[0, 3].set_title('+2 delta') 
fig.suptitle('Example of plotting exploration with metabolics (AB1)',fontweight="bold")
plt.show()

# palette = sns.color_palette("Paired_r", 5)
# fig, axs = plt.subplots(5, 4, sharex=True, sharey=True)
# axs[0, 0].set_title('-2$\Delta$')
# axs[0, 1].set_title('-1$\Delta$')
# axs[0, 2].set_title('+1$\Delta$')
# axs[0, 3].set_title('+2$\Delta$')

# axs[0, 0].set_ylabel('AB1')
# axs[1, 0].set_ylabel('AB2')
# axs[2, 0].set_ylabel('AB3')
# axs[3, 0].set_ylabel('AB4')
# axs[4, 0].set_ylabel('AB5')
# xtime = np.arange(dataRange)/60
# for ii in np.arange(4):
#     axs[0, ii].plot(xtime, dataFilt[ii][:,0]/dataFilt[ii][0,0]-1, color = palette[0], linewidth=2.5)
#     axs[1, ii].plot(xtime, dataFilt[ii][:,1]/dataFilt[ii][0,1]-1, color = palette[1], linewidth=2.5)
#     axs[2, ii].plot(xtime, dataFilt[ii][:,2]/dataFilt[ii][0,2]-1, color = palette[2], linewidth=2.5)
#     axs[3, ii].plot(xtime, dataFilt[ii][:,3]/dataFilt[ii][0,3]-1, color = palette[3], linewidth=2.5)
#     axs[4, ii].plot(xtime, dataFilt[ii][:,4]/dataFilt[ii][0,4]-1, color = palette[4], linewidth=2.5)
# fig.supylabel('Change in Metabolic Cost (W/kg)',fontweight="bold")
# fig.supxlabel('Time (min)',fontweight="bold")
# fig.suptitle('Metabolic cost during different split-belt walking conditions',fontweight="bold")
# plt.show()



