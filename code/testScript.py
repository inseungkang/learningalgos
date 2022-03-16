import procFunc as pf
import numpy as np
import pandas as pd
import os
import fnmatch
import c3d
import matplotlib.pyplot as plt
import pickle

## load c3d file data and extract header column info and save 
# file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d data/'
# file_list = os.listdir(file_dir)

# result = []
# for file_name in file_list:
#     idxlist = c3d_marker_header_loc(os.path.join(file_dir,file_name))
#     result.append([file_name, idxlist])

# save_dir = '/Users/inseungkang/Documents/learningalgos data/c3d marker header location.pkl'
# with open(save_file_name, 'wb') as handle:
#     pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # Load marker header file to get the index column
# header_dir = '/Users/inseungkang/Documents/learningalgos data/c3d marker header location.pkl'
# with open(header_dir, 'rb') as handle:
#     marker_header_idx = pickle.load(handle)

# # Open npz marker and force data and match the relevaant index column
# data_dir = '/Users/inseungkang/Documents/learningalgos data/npz data/'
# file_list = os.listdir(data_dir)
# file_list = [i for i in file_list if 'npz' in i]

# for file_name in file_list:

#     for idx, [name, list] in enumerate(marker_header_idx):
#         if file_name.split('.npz')[0] == name.split('.c3d')[0]:
#             indexList = list

#     if file_name.find('Static') & file_name.find('Session1') == -1:
#         print(file_name)
#         pf.extract_from_c3d(data_dir+file_name, indexList, 45)


data_dir = '/Users/inseungkang/Documents/learningalgos data/synced data/'
file_list = [i for i in os.listdir(data_dir) if '.pkl' in i]

for file_name in file_list:
    with open(data_dir+file_name, 'rb') as handle:
        com, foot_r, foot_l, force_r, force_l = pickle.load(handle)
        session = 1
        (com, foot_r, foot_l, force_r, force_l)
        AB1_delta0 = pf.subjectData(com, foot_r, foot_l, force_r, force_l, session)



# stance_idx = pf.comput_stance_idx(force_r[:,2], force_l[:,2])
# sla = pf.compute_SLA(stance_idx, foot_r, foot_l, 1)
