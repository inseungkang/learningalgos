import numpy as np
import pickle
import readFiles as RF
import matplotlib.pyplot as plt
import controllerInferenceFunc as CIF
import utilityFunc as UF
import c3d

file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_splitbelt/'
file_name = 'AB3_Session2_Right10_Left6.c3d'

reader = c3d.Reader(open(file_dir+file_name, 'rb'))
markerLabel = reader.point_labels
markerName = ['RASI','LASI','RPSI','LPSI','RHEE','LHEE']
header_list = []
for _, name in enumerate(markerName):
    for idx, label in enumerate(markerLabel):
        if name == label.strip():
            header_list.append(idx)
print(header_list)
file_name = file_name.split('.')[0]+'.npz'
file_header = header_list
npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_splitbelt/'

com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 12, 45)
gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)
com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)
foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_filt(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)

plt.plot(foot_r)
plt.show()