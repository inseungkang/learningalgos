import numpy as np
import pandas as pd
import os
import c3d
import matplotlib.pyplot as plt


file_dir = '/Users/inseungkang/Documents/c3d_Day2/'
file_name = 'adapt1.c3d'

header_list = []
reader = c3d.Reader(open(file_dir+file_name, 'rb'))
markerLabel = reader.point_labels  
markerName = ['RASI','LASI','RPSI','LPSI','RHEE','LHEE']
for _, name in enumerate(markerName):
    for idx, label in enumerate(markerLabel):
        if name == label.strip():
            header_list.append(idx)

c3d2npz adapt1.c3d

print(header_list)




# def extract_file_info(file_dir):
#     file_list = os.listdir(file_dir)
#     file_list = [i for i in file_list if 'c3d' in i]
#     file_list.sort()
#     file_info = []

#     for filename in file_list:
#         reader = c3d.Reader(open(file_dir+filename, 'rb'))
#         markerLabel = reader.point_labels
#         markerName = ['RASI','LASI','RPSI','LPSI','RHEE','LHEE']
#         header_list = []
#         for _, name in enumerate(markerName):
#             for idx, label in enumerate(markerLabel):
#                 if name == label.strip():
#                     header_list.append(idx)
#         file_info.append((filename.split('.c3d')[0], header_list))
#     return file_info