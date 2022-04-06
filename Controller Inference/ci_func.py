import numpy as np
import pandas as pd
import os
import fnmatch
import c3d
import pickle



## Load data function
def load_data(data_dir):
    file_list = [i for i in os.listdir(data_dir) if '.pkl' in i]

    for file_name in file_list:
        subjectNum = int(file_name.split('_Session')[0].split('AB')[1])
        session = int(file_name.split('Session')[1].split('_Right')[0])
        rSpeed = int(file_name.split('Right')[1].split('_Left')[0])
        lSpeed = int(file_name.split('Left')[1].split('_synced')[0])

        if rSpeed == 10:
            delta = int((lSpeed - rSpeed)/2)
        else:
            delta = int((rSpeed - lSpeed)/2)

        delta = delta + 2
        
        if rSpeed > lSpeed:
            fast_leg = 0
        else:
            fast_leg = 1

        with open(data_dir+file_name, 'rb') as handle:
            com, foot_r, foot_l, force_r, force_l = pickle.load(handle)

            if subjectNum == 1:
                if delta == 0:
                    AB1_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 1:
                    AB1_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 3:
                    AB1_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 4:
                    AB1_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

            if subjectNum == 2:
                if delta == 0:
                    AB2_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 1:
                    AB2_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 3:
                    AB2_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 4:
                    AB2_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

            if subjectNum == 3:
                if delta == 0:
                    AB3_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 1:
                    AB3_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 3:
                    AB3_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 4:
                    AB3_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

            if subjectNum == 4:
                if delta == 0:
                    AB4_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 1:
                    AB4_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 3:
                    AB4_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 4:
                    AB4_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

            if subjectNum == 5:
                if delta == 0:
                    AB5_delta0 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 1:
                    AB5_delta1 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 3:
                    AB5_delta2 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)
                if delta == 4:
                    AB5_delta3 = pf.subjectData(subjectNum, delta, session, com, foot_r, foot_l, force_r, force_l, fast_leg)

    return [AB1_delta0, AB1_delta1, AB1_delta2, AB1_delta3,
            AB2_delta0, AB2_delta1, AB2_delta2, AB2_delta3,
            AB3_delta0, AB3_delta1, AB3_delta2, AB3_delta3,
            AB4_delta0, AB4_delta1, AB4_delta2, AB4_delta3,
            AB5_delta0, AB5_delta1, AB5_delta2, AB5_delta3]
