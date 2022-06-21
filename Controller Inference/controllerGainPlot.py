from cgi import test
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


split_file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_splitbelt/'
tied_file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_tiedbelt/'
split_npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_splitbelt/'
tied_npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'

eval_time = 1
subject_tied_fastx = []
subject_tied_slowx = []
subject_tied_fasty = []
subject_tied_slowy = []
subject_split_fastx = []
subject_split_slowx = []
subject_split_fasty = []
subject_split_slowy = []

test_list = os.listdir(split_file_dir)
# for subjectname in ['AB1']:
for subjectname in ['AB1', 'AB2', 'AB3', 'AB4', 'AB5']:
    newlist = []
    idxlist = []
    result_tied_fast_x = []
    result_tied_fast_y = []
    result_tied_slow_x = []  
    result_tied_slow_y = []
    result_split_fast_x = []
    result_split_fast_y = []
    result_split_slow_x = []   
    result_split_slow_y = []
        
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

        _, _, gain_fast_x, gain_fast_y, fast_model_x, fast_model_y = CIF.controller_fit(delta_P_fast, delta_Q_fast)
        _, _, gain_slow_x, gain_slow_y, slow_model_x, slow_model_y = CIF.controller_fit(delta_P_slow, delta_Q_slow)

        tied_gain_fast_x = gain_fast_x
        tied_gain_fast_y = gain_fast_y
        tied_gain_slow_x = gain_slow_x
        tied_gain_slow_y = gain_slow_y

        _, _, gain_fast_x, gain_fast_y, fast_model_x, _ = CIF.controller_fit(split_delta_P_fast, split_delta_Q_fast)
        _, _, gain_slow_x, gain_slow_y, slow_model_x, _ = CIF.controller_fit(split_delta_P_slow, split_delta_Q_slow)

        split_gain_fast_x = gain_fast_x
        split_gain_fast_y = gain_fast_y
        split_gain_slow_x = gain_slow_x
        split_gain_slow_y = gain_slow_y


        result_tied_fast_x.append(tied_gain_fast_x)
        result_tied_fast_y.append(tied_gain_fast_y)
        result_tied_slow_x.append(tied_gain_slow_x)    
        result_tied_slow_y.append(tied_gain_slow_y)        
        result_split_fast_x.append(split_gain_fast_x)
        result_split_fast_y.append(split_gain_fast_y)
        result_split_slow_x.append(split_gain_slow_x)    
        result_split_slow_y.append(split_gain_slow_y)

    subject_tied_fastx.append(result_tied_fast_x)
    subject_tied_fasty.append(result_tied_fast_y)
    subject_tied_slowx.append(result_tied_slow_x)
    subject_tied_slowy.append(result_tied_slow_y)
    subject_split_fastx.append(result_split_fast_x)
    subject_split_fasty.append(result_split_fast_y)
    subject_split_slowx.append(result_split_slow_x)
    subject_split_slowy.append(result_split_slow_y)


# print(subject_tied_fasty)
# print(subject_tied_fasty[:][0])
def test_func(input):
    mean1 = []
    mean2 = []
    mean3 = []
    mean4 = []
    mean5 = []

    for delta in np.arange(4):
        test1 = []
        test2 = []
        test3 = []
        test4 = []
        test5 = []
        for subject in np.arange(5):
            d1_a1 = input[subject][delta][0]
            d1_a2 = input[subject][delta][1]
            d1_a3 = input[subject][delta][2]
            d1_a4 = input[subject][delta][3]
            d1_a5 = input[subject][delta][4]    
            test1.append(d1_a1)
            test2.append(d1_a2)
            test3.append(d1_a3)
            test4.append(d1_a4)
            test5.append(d1_a5)
        mean1.append(np.mean(test1))
        mean2.append(np.mean(test2))
        mean3.append(np.mean(test3))
        mean4.append(np.mean(test4))  
        mean5.append(np.mean(test5))  

    return mean1, mean2, mean3, mean4, mean5



m1, m2, m3, m4, m5 = test_func(subject_tied_fasty)
n1, n2, n3, n4, n5 = test_func(subject_split_fasty)


# print(np.mean(subject_tied_fasty))
fig, axs = plt.subplots(1, 5)

axs[0].plot(m1, n1,'o')
axs[1].plot(m2, n2,'o')
axs[2].plot(m3, n3,'o')
axs[3].plot(m4, n4,'o')
axs[4].plot(m5, n5,'o')
plt.show()
# axs[0].plot(np.mean(subject_tied_fasty[:][0]), np.mean(subject_split_fasty[:][0]), 'o')
# axs[0].plot(np.mean(subject_tied_fasty[:][1]), np.mean(subject_split_fasty[:][0]), 'o')
# axs[0].plot(np.mean(subject_tied_fasty[:][2]), np.mean(subject_split_fasty[:][0]), 'o')
# axs[0].plot(np.mean(subject_tied_fasty[:][3]), np.mean(subject_split_fasty[:][0]), 'o')


# # axs[0].plot(np.mean(subject_tied_fasty[:][4][0]), np.mean(subject_split_fasty[:][4][0]), 'o')

# # axs[1].plot(subject_tied_fasty[0][0][1], subject_split_fasty[0][0][1], 'o')
# # axs[1].plot(subject_tied_fasty[0][1][1], subject_split_fasty[0][1][1], 'o')
# # axs[1].plot(subject_tied_fasty[0][2][1], subject_split_fasty[0][2][1], 'o')
# # axs[1].plot(subject_tied_fasty[0][3][1], subject_split_fasty[0][3][1], 'o')

# plt.show()
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
# print(subject_result[0][0])
# plt.plot(subject_result[0][0])
# plt.show()

# fig, axs = plt.subplots(3,4)
# for ii in [0, 2]:

#     axs[0,ii].plot(subject_result_fasty[0][int(ii/2)+2], 'o-')
#     axs[0,ii+1].plot(subject_result_slowy[0][int(ii/2)+2], 'o-')    

#     #AB1 is change the last number to corresponding index
#     test = dataFilt[int(ii/2)+2][:,0]
#     testvec = []
#     for iii in np.arange(45):
#         value = iii*60+30
#         if value > len(test):
#             value = len(test)-1
#         testvec.append(test[value])

#     axs[1,ii].plot(testvec, 'o-')
#     axs[1,ii+1].plot(testvec, 'o-')
    
#     axs[2,ii].plot(testvec, subject_result_fasty[0][int(ii/2)+1], 'o')
#     axs[2,ii+1].plot(testvec, subject_result_slowy[0][int(ii/2)+1], 'o')

# axs[0, 0].set_ylabel('Exploration')
# axs[1, 0].set_ylabel('Energetics')
# axs[2, 0].set_ylabel('Correlation')
# # axs[0, 0].set_title('+1 delta')
# # axs[0, 1].set_title('+2 delta')
# # axs[0, 2].set_title('+1 delta') 
# # axs[0, 3].set_title('+2 delta') 
# fig.suptitle('Example of plotting exploration with metabolics (AB1)',fontweight="bold")
# plt.show()

# y_pred = reg.predict(x)

# ax7.set_title("R2 = "+str((r2_score(testvec, subject_result_fasty[0][2]))))
# print(str(r2_score(y, y_pred)))


# ax5.set_xlabel('Time (min)')
# ax6.set_xlabel('Time (min)')

# ax7.set_xlabel('metabolics')
# ax8.set_xlabel('Time (min)')
# ax9.set_xlabel('Time (min)')
# ax10.set_xlabel('Time (min)')


# ax1.set_ylabel('Exploration')
# ax5.set_ylabel('Energetics')
# ax7.set_ylabel('Correlation')
# axs[0, 0].set_title('+1 delta')
# axs[0, 1].set_title('+2 delta')
# axs[0, 2].set_title('+1 delta') 
# axs[0, 3].set_title('+2 delta') 
# fig.suptitle('Example of plotting exploration with metabolics (AB1)',fontweight="bold")


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



