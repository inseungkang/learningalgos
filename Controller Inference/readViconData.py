import numpy as np
import pandas as pd
import os
import c3d

# Put Vicon C3D file diectory here
dir_path = "C://Users//Inseung Kang//Desktop//Vicon Data//"

file_name = os.listdir(dir_path)
# print(file_name[0])
markerData = np.empty((36, 5, 1))
forceData = np.empty((18, 10, 1))

with open(os.path.join(dir_path, file_name[0]), 'rb') as handle:
    print(file_name[0])
#     reader = c3d.Reader(handle)
#     for _, (idx, marker, force) in enumerate(reader.read_frames()):
#         print(markerData.shape)
#         markerData = np.concatenate((markerData, np.expand_dims(marker, axis=2)), axis=2)
#         forceData = np.concatenate((forceData, np.expand_dims(force, axis=2)), axis=2)
#         if idx == 7999:
#             break

# np.save('testMarkerData.npy', markerData)
# np.save('testForceData.npy', forceData)




# A = reader.read_frames()


    # print(reader.re)
    # for i, (points, analog) in enumerate(reader.read_frames()):
    #     print('Frame {}: {}'.format(i, points.round(2)))


# for file in file_list[0]:
    # print(file)        
# print(arr)






