import numpy as np
import pickle
import readFiles as RF
import matplotlib.pyplot as plt
import controllerInferenceFunc as CIF
import utilityFunc as UF
import os


split_file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_splitbelt/'
tied_file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_tiedbelt/'
split_npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_splitbelt/'
tied_npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'

file_name = 'AB4_Session5_Right8_Left10'

file_header = RF.extract_marker_column(split_file_dir+file_name+'.c3d')

subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name+'.npz')
trial_info = (subject, delta, fast_leg, v_fast, v_slow)

fastleg_filename = 'AB'+str(subject)+'_Session1_Right'+str(v_fast)+'_'+'Left'+str(v_fast)
slowleg_filename = 'AB'+str(subject)+'_Session1_Right'+str(v_slow)+'_'+'Left'+str(v_slow)

file_header_fast = RF.extract_marker_column(tied_file_dir+fastleg_filename+'.c3d')
file_header_slow = RF.extract_marker_column(tied_file_dir+fastleg_filename+'.c3d')

if fast_leg == 'right':
    delta_P_fast, delta_Q_fast, gait_idx = CIF.computeDeltaPandQ(tied_npz_dir+fastleg_filename+'.npz', file_header_fast, 'right', 0, 6)
    delta_P_slow, delta_Q_slow, gait_idx = CIF.computeDeltaPandQ(tied_npz_dir+slowleg_filename+'.npz', file_header_slow, 'left', 0, 6)
    split_delta_P_fast, split_delta_Q_fast, gait_idx = CIF.computeDeltaPandQ(split_npz_dir+file_name+'.npz', file_header, 'right', 12, 45)
    split_delta_P_slow, split_delta_Q_slow, gait_idx = CIF.computeDeltaPandQ(split_npz_dir+file_name+'.npz', file_header, 'left', 12, 45)

elif fast_leg == 'left':
    delta_P_fast, delta_Q_fast, gait_idx = CIF.computeDeltaPandQ(tied_npz_dir+fastleg_filename+'.npz', file_header_fast, 'left', 0, 6)
    delta_P_slow, delta_Q_slow, gait_idx = CIF.computeDeltaPandQ(tied_npz_dir+slowleg_filename+'.npz', file_header_slow, 'right', 0, 6)       
    split_delta_P_fast, split_delta_Q_fast, gait_idx = CIF.computeDeltaPandQ(split_npz_dir+file_name+'.npz', file_header, 'left', 12, 45)
    split_delta_P_slow, split_delta_Q_slow, gait_idx = CIF.computeDeltaPandQ(split_npz_dir+file_name+'.npz', file_header, 'right', 12, 45)


com, foot_r, foot_l, force_r, force_l = RF.extract_data(tied_npz_dir+slowleg_filename+'.npz', file_header_slow, 0, 6)

plt.plot(com)
plt.show()
r2_fast_x, r2_fast_y, gain_fast_x, gain_fast_y, fast_model_x, fast_model_y = CIF.controller_fit(delta_P_fast, delta_Q_fast)
# r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y, slow_model_x, slow_model_y = CIF.controller_fit(delta_P_slow, delta_Q_slow)

# tied_fast_var_x = UF.rmse(fast_model_x.predict(delta_P_fast), delta_Q_fast[:,0])
# tied_fast_var_y = UF.rmse(fast_model_y.predict(delta_P_fast), delta_Q_fast[:,1])            
# tied_slow_var_x = UF.rmse(slow_model_x.predict(delta_P_slow), delta_Q_slow[:,0])
# tied_slow_var_y = UF.rmse(slow_model_y.predict(delta_P_slow), delta_Q_slow[:,1])  