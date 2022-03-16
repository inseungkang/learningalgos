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


## Load marker header data
# load_dir = '/Users/inseungkang/Documents/learningalgos data/c3d marker header location.pkl'
# with open(load_dir, 'rb') as handle:
#     marker_header_idx = pickle.load(handle)

# print(marker_header_idx)


## convert c3d file into npz format
file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d data/'
listOfFiles = os.listdir(file_dir)
listOfFiles = listOfFiles[0]

pattern = "*.c3d"
result = []
for file_name in listOfFiles:
    if fnmatch.fnmatch(file_name, pattern):
        idxList = extract_marker_col(file_name)
        result.append((file_name, idxList))
        # c3d2npz file_name
