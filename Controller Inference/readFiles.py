import numpy as np
import pandas as pd
import os
import c3d
import matplotlib.pyplot as plt

#This function uses the file title to extract the marker's header index number from a C3D file (in a full list format)
def extract_file_info(file_dir):
    file_list = os.listdir(file_dir)
    file_list = [i for i in file_list if 'c3d' in i]
    file_list.sort()
    file_info = []

    for filename in file_list:
        reader = c3d.Reader(open(file_dir+filename, 'rb'))
        markerLabel = reader.point_labels
        markerName = ['RASI','LASI','RPSI','LPSI','RANL','LANL']
        header_list = []
        for _, name in enumerate(markerName):
            for idx, label in enumerate(markerLabel):
                if name == label.strip():
                    header_list.append(idx)
        file_info.append((filename.split('.c3d')[0], header_list))
    return file_info

#This function uses the file title to extract the marker's header index number from a C3D file (directory included)
def extract_marker_column(filename):
    reader = c3d.Reader(open(filename, 'rb'))
    markerLabel = reader.point_labels
    markerName = ['RASI','LASI','RPSI','LPSI','RANL','LANL']
    header_list = []
    for _, name in enumerate(markerName):
        for idx, label in enumerate(markerLabel):
            if name == label.strip():
                header_list.append(idx)
    return header_list

# This function uses the file input (npz file format) and extract relevant data (marker, force)
# for the trial time, input in the start time in seconds and duration is in seconds
def extract_data(file_dir, file_header, start_time, duration):
  with np.load(file_dir) as data:
    marker_data = data['points']
    force_data = data['analog']

  force_sync = np.mean(force_data[:,14,:], axis=1)
  start_idx = np.argwhere(abs(force_sync) > np.max(abs(force_sync)/2))[0,0]
  start_idx = start_idx + start_time*6000
  end_idx = start_idx + duration*6000        

  foot_r = marker_data[start_idx:end_idx, file_header[4], 0:3]
  foot_l = marker_data[start_idx:end_idx, file_header[5], 0:3]
  # com = np.mean(marker_data[start_idx:end_idx, file_header[0:2], 0:3], axis=1)
  com = np.mean(marker_data[start_idx:end_idx, file_header[0:4], 0:3], axis=1)
  force_r = np.mean(force_data[start_idx:end_idx, 0:3, :], axis=2)
  force_l = np.mean(force_data[start_idx:end_idx, 6:9, :], axis=2)
  
  return com, foot_r, foot_l, force_r, force_l
