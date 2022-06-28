import enum
import numpy as np
import pickle
import readFiles as RF
import matplotlib.pyplot as plt
import controllerInferenceFunc as CIF
import utilityFunc as UF
import os
import c3d

file_dir = '/Users/inseungkang/Documents/learningalgos data/c3d files_tiedbelt/'
npz_dir = '/Users/inseungkang/Documents/learningalgos data/npz files_tiedbelt/'
file_info = RF.extract_file_info(file_dir)

for file in file_info:
        # file = file_info[5]

        file_name = file[0]+'.npz'
        file_header = file[1]

        subject, delta, fast_leg, v_fast, v_slow = CIF.extract_trial_info(file_name)
        trial_info = (subject, delta, fast_leg, v_fast, v_slow)
        if subject == 3 and v_fast == 14:
                print(file_name)

                com, foot_r, foot_l, force_r, force_l = RF.extract_data(npz_dir+file_name, file_header, 0, 6)
                gait_event_idx, right_ms_idx, left_ms_idx, test_left_ms_idx = CIF.GRF_based_gait_event_idx(com, foot_r, foot_l, force_r, force_l)
                # gait_event_idx = CIF.compute_gait_event(com, foot_r, foot_l)                
                com_cont_r, com_cont_l = CIF.compute_continous_com_state(com, foot_r, foot_l)
                # COM_X_r = (com[:,0])*0.1
                # COM_Y_r = (com[:,1])*0.1
                # COM_Z_r = (com[:,2])*0.1
                # COM_X_l = (com[:,0])*0.1
                # COM_Y_l = (com[:,1])*0.1
                # COM_Z_l = (com[:,2])*0.1
                COM_X_r = (com[:,0] - foot_r[:,0])*0.1
                COM_Y_r = (com[:,1] - foot_r[:,1])*0.1
                COM_Z_r = (com[:,2] - foot_r[:,2])*0.1
                COM_X_l = (com[:,0] - foot_l[:,0])*0.1
                COM_Y_l = (com[:,1] - foot_l[:,1])*0.1
                COM_Z_l = (com[:,2] - foot_l[:,2])*0.1

                # plt.subplot(311)
                # plt.plot(COM_X_r)
                # plt.xlim((10000, 11000))
                # plt.subplot(312)
                # plt.xlim((10000, 11000))
                # plt.plot(COM_Y_r)
                # plt.subplot(313)
                # plt.xlim((10000, 11000))
                # plt.plot(COM_Z_r)

                # input data is sample at 100 Hz so to convert velocity in cm/s, divide by 0.01 
                COM_XX_r = np.append(np.diff(COM_X_r), np.diff(COM_X_r)[-1])/0.01
                COM_YY_r = np.append(np.diff(COM_Y_r), np.diff(COM_Y_r)[-1])/0.01
                COM_ZZ_r = np.append(np.diff(COM_Z_r), np.diff(COM_Z_r)[-1])/0.01
                COM_XX_l = np.append(np.diff(COM_X_l), np.diff(COM_X_l)[-1])/0.01
                COM_YY_l = np.append(np.diff(COM_Y_l), np.diff(COM_Y_l)[-1])/0.01
                COM_ZZ_l = np.append(np.diff(COM_Z_l), np.diff(COM_Z_l)[-1])/0.01

                COM_state_r = np.transpose(np.asarray((COM_X_r, COM_Z_r, COM_XX_r, COM_YY_r, COM_ZZ_r)))
                COM_state_l = np.transpose(np.asarray((COM_X_l, COM_Z_l, COM_XX_l, COM_YY_l, COM_ZZ_l)))

                com_ms_r = COM_state_r[[int(i) for i in gait_event_idx[:,2].tolist()]]
                com_ms_l = COM_state_l[[int(i) for i in gait_event_idx[:,0].tolist()]]
                # com_ms_r, com_ms_l = CIF.compute_com_state_mid_stance(com_cont_r, com_cont_l, gait_event_idx)

                # footXY_r = np.column_stack((foot_r[:,0]*0.1, foot_r[:,1]*0.1))
                # footXY_l = np.column_stack((foot_l[:,0]*0.1, foot_l[:,1]*0.1))
                # footXY_r = np.column_stack((foot_r[:,0] - com[:,0], foot_r[:,1] - com[:,1]))*0.1
                # footXY_l = np.column_stack((foot_l[:,0] - com[:,0], foot_l[:,1] - com[:,1]))*0.1
                footXY_r = np.column_stack((foot_r[:,0] - foot_l[:,0], foot_r[:,1] - foot_l[:,1]))*0.1
                footXY_l = np.column_stack((foot_l[:,0] - foot_r[:,0], foot_l[:,1] - foot_r[:,1]))*0.1
                footXY_hc_r = footXY_r[[int(i) for i in gait_event_idx[:,1].tolist()]]
                footXY_hc_l = footXY_l[[int(i) for i in gait_event_idx[:,3].tolist()]]

                left_hc_idx = gait_event_idx[:,3].astype(int)
                right_ms_idx = gait_event_idx[:,2].astype(int)


                # plt.subplot(211)
                # plt.plot(footXY_r[:,0])
                # yval_x = footXY_r[:,0]
                # yval_x = yval_x[right_hc_idx]
                # plt.plot(right_hc_idx, yval_x, 'o')
                # plt.xlim((10000, 11000))
                
                # plt.subplot(212)
                # plt.plot(footXY_r[:,1])
                # yval_y = footXY_r[:,1]
                # yval_y = yval_y[right_hc_idx]
                # plt.plot(right_hc_idx, yval_y, 'o')
                # plt.xlim((10000, 11000))

                
                # plt.subplot(311)
                # plt.plot(COM_X_r)
                # yval_x = COM_X_r[right_ms_idx]    
                # plt.plot(right_ms_idx, yval_x, 'o')
                # plt.xlim((10000, 11000))

                # plt.subplot(312)
                # plt.plot(footXY_l[:,0])
                # yval_x = footXY_l[:,0]
                # yval_x = yval_x[left_hc_idx]
                # plt.plot(left_hc_idx, yval_x, 'o')
                # plt.xlim((10000, 11000))

                # plt.subplot(313)
                # plt.plot(force_l[:,2])
                # yval_x = force_l[:,0]
                # yval_x = yval_x[left_hc_idx]
                # plt.plot(left_hc_idx, yval_x, 'o')
                # plt.xlim((10000, 11000))

                foot_hc_r, foot_hc_l = CIF.compute_foot_placement(foot_r, foot_l, gait_event_idx)
                delta_P_r, delta_P_l, delta_Q_r, delta_Q_l = CIF.detrend_data_constant(com_ms_r, com_ms_l, foot_hc_r, foot_hc_l)

                r2_fast_x, r2_fast_y, gain_fast_x, gain_fast_y, _, _ = CIF.controller_fit(delta_P_r, delta_Q_r)
                r2_slow_x, r2_slow_y, gain_slow_x, gain_slow_y, _, _ = CIF.controller_fit(delta_P_l, delta_Q_l)

                print(gain_fast_x)
                print(gain_slow_x)
                plt.plot(gain_fast_x)
                plt.plot(gain_slow_x)
                plt.legend(['right leg', 'left leg'])

plt.show()



        # plt.plot(gait_event_idx - np.expand_dims(gait_event_idx[:,0], axis=1))
        # plt.show()


