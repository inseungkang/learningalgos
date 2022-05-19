import numpy as np
import pickle
import readFiles as RF
import matplotlib.pyplot as plt
import controllerInferenceFunc as CIF
import utilityFunc as UF

file = open('controllerInferenceAdapt.pkl', 'rb')
adapt_info_result = pickle.load(file)
adapt_fastgainX_result = pickle.load(file)
adapt_fastgainY_result = pickle.load(file)
adapt_slowgainX_result = pickle.load(file)
adapt_slowgainY_result = pickle.load(file)
file.close()

file = open('controllerInferenceConst_6.pkl', 'rb')
const_6_info_result = pickle.load(file)
const_6_fastgainX_result = pickle.load(file)
const_6_fastgainY_result = pickle.load(file)
const_6_slowgainX_result = pickle.load(file)
const_6_slowgainY_result = pickle.load(file)
file.close()

file = open('controllerInferenceConst_8.pkl', 'rb')
const_8_info_result = pickle.load(file)
const_8_fastgainX_result = pickle.load(file)
const_8_fastgainY_result = pickle.load(file)
const_8_slowgainX_result = pickle.load(file)
const_8_slowgainY_result = pickle.load(file)
file.close()

file = open('controllerInferenceConst_10.pkl', 'rb')
const_10_info_result = pickle.load(file)
const_10_fastgainX_result = pickle.load(file)
const_10_fastgainY_result = pickle.load(file)
const_10_slowgainX_result = pickle.load(file)
const_10_slowgainY_result = pickle.load(file)
file.close()

file = open('controllerInferenceConst_12.pkl', 'rb')
const_12_info_result = pickle.load(file)
const_12_fastgainX_result = pickle.load(file)
const_12_fastgainY_result = pickle.load(file)
const_12_slowgainX_result = pickle.load(file)
const_12_slowgainY_result = pickle.load(file)
file.close()

file = open('controllerInferenceConst_14.pkl', 'rb')
const_14_info_result = pickle.load(file)
const_14_fastgainX_result = pickle.load(file)
const_14_fastgainY_result = pickle.load(file)
const_14_slowgainX_result = pickle.load(file)
const_14_slowgainY_result = pickle.load(file)
file.close()

# del const_6_fastgainX_result[4]
# del const_6_slowgainX_result[4]
# del const_8_fastgainY_result[3]
# del const_8_slowgainY_result[3]
# del const_10_fastgainX_result[3]
# del const_10_slowgainX_result[3]
# del const_10_fastgainY_result[3]
# del const_10_slowgainY_result[3]

# del const_12_fastgainY_result[3]
# del const_12_slowgainY_result[3]
# del const_14_fastgainY_result[3]
# del const_14_slowgainY_result[3]

const6_fast_X = UF.list_to_array(const_6_fastgainX_result)
const6_slow_X = UF.list_to_array(const_6_slowgainX_result)
const6_fast_X_avg = np.mean(const6_fast_X, axis=1)
const6_slow_X_avg = np.mean(const6_slow_X, axis=1)
const6_fast_Y = UF.list_to_array(const_6_fastgainY_result)
const6_slow_Y = UF.list_to_array(const_6_slowgainY_result)
const6_fast_Y_avg = np.mean(const6_fast_Y, axis=1)
const6_slow_Y_avg = np.mean(const6_slow_Y, axis=1)

const8_fast_X = UF.list_to_array(const_8_fastgainX_result)
const8_slow_X = UF.list_to_array(const_8_slowgainX_result)
const8_fast_X_avg = np.mean(const8_fast_X, axis=1)
const8_slow_X_avg = np.mean(const8_slow_X, axis=1)
const8_fast_Y = UF.list_to_array(const_8_fastgainY_result)
const8_slow_Y = UF.list_to_array(const_8_slowgainY_result)
const8_fast_Y_avg = np.mean(const8_fast_Y, axis=1)
const8_slow_Y_avg = np.mean(const8_slow_Y, axis=1)

const10_fast_X = UF.list_to_array(const_10_fastgainX_result)
const10_slow_X = UF.list_to_array(const_10_slowgainX_result)
const10_fast_X_avg = np.mean(const10_fast_X, axis=1)
const10_slow_X_avg = np.mean(const10_slow_X, axis=1)
const10_fast_Y = UF.list_to_array(const_10_fastgainY_result)
const10_slow_Y = UF.list_to_array(const_10_slowgainY_result)
const10_fast_Y_avg = np.mean(const10_fast_Y, axis=1)
const10_slow_Y_avg = np.mean(const10_slow_Y, axis=1)

const12_fast_X = UF.list_to_array(const_12_fastgainX_result)
const12_slow_X = UF.list_to_array(const_12_slowgainX_result)
const12_fast_X_avg = np.mean(const12_fast_X, axis=1)
const12_slow_X_avg = np.mean(const12_slow_X, axis=1)
const12_fast_Y = UF.list_to_array(const_12_fastgainY_result)
const12_slow_Y = UF.list_to_array(const_12_slowgainY_result)
const12_fast_Y_avg = np.mean(const12_fast_Y, axis=1)
const12_slow_Y_avg = np.mean(const12_slow_Y, axis=1)

const14_fast_X = UF.list_to_array(const_14_fastgainX_result)
const14_slow_X = UF.list_to_array(const_14_slowgainX_result)
const14_fast_X_avg = np.mean(const14_fast_X, axis=1)
const14_slow_X_avg = np.mean(const14_slow_X, axis=1)
const14_fast_Y = UF.list_to_array(const_14_fastgainY_result)
const14_slow_Y = UF.list_to_array(const_14_slowgainY_result)
const14_fast_Y_avg = np.mean(const14_fast_Y, axis=1)
const14_slow_Y_avg = np.mean(const14_slow_Y, axis=1)

const10_X = (const10_slow_X_avg + const10_fast_X_avg)/2
const14_X = (const14_slow_X_avg + const14_fast_X_avg)/2
const10_Y = (const10_slow_Y_avg + const10_fast_Y_avg)/2
const14_Y = (const14_slow_Y_avg + const14_fast_Y_avg)/2

fig, axs = plt.subplots(5,4)
for ii, xx in enumerate([6, 8, 10, 12, 14]):
    nameX = [const6_fast_X, const8_fast_X, const10_fast_X, const12_fast_X, const14_fast_X,
    const6_slow_X, const8_slow_X, const10_slow_X, const12_slow_X, const14_slow_X]
    nameY = [const6_fast_Y, const8_fast_Y, const10_fast_Y, const12_fast_Y, const14_fast_Y,
    const6_slow_Y, const8_slow_Y, const10_slow_Y, const12_slow_Y, const14_slow_Y]

    axs[ii,0].plot(nameX[ii], 'o-')
    axs[ii,1].plot(nameX[ii+5], 'o-')
    axs[ii,2].plot(nameY[ii], 'o-')
    axs[ii,3].plot(nameY[ii+5], 'o-')


plt.setp(axs[0,0], title='FootX Right')
plt.setp(axs[0,1], title='FootX Left')
plt.setp(axs[0,2], title='FootY Right')
plt.setp(axs[0,3], title='FootY Left')

plt.setp(axs[0,0], ylabel='0.6 m/s')
plt.setp(axs[1,0], ylabel='0.8 m/s')
plt.setp(axs[2,0], ylabel='1.0 m/s')
plt.setp(axs[3,0], ylabel='1.2 m/s')
plt.setp(axs[4,0], ylabel='1.4 m/s')
plt.legend(['AB1','AB2','AB3','AB4','AB5'], ncol=5, bbox_to_anchor=(0.2, -0.4))
fig.suptitle('Controller gains across different walking speeds')
plt.show()

# x = np.arange(45)
# plt.plot(x, np.ones(len(x))*const14_fast_Y_avg[0])
# plt.show()



# fig, axs = plt.subplots(2,5)
# for iii in np.arange(5):
#     x = np.arange(45)
#     # fast X
#     test_vec = np.empty((len(adapt_fastgainX_result[0][:,iii]),1))
#     for ii in np.arange(5):
#         vec = np.expand_dims(np.array(adapt_fastgainX_result[ii][:,iii]), axis=1)
#         test_vec = np.concatenate((test_vec, vec), axis=1)
#     test_vec = test_vec[:,1:]
#     test_vec_avg = np.mean(test_vec, axis=1)
#     test_vec_std = np.std(test_vec, axis=1)

#     axs[0,iii].errorbar(x, test_vec_avg, yerr=test_vec_std, fmt='o', markersize=2.5, color='darkred', ecolor='lightcoral', elinewidth=1.5, capsize=0)
#     # fast Y
#     test_vec = np.empty((len(adapt_fastgainY_result[0][:,iii]),1))
#     for ii in np.arange(5):
#         vec = np.expand_dims(np.array(adapt_fastgainY_result[ii][:,iii]), axis=1)
#         test_vec = np.concatenate((test_vec, vec), axis=1)
#     test_vec = test_vec[:,1:]
#     test_vec_avg = np.mean(test_vec, axis=1)
#     test_vec_std = np.std(test_vec, axis=1)

#     axs[1,iii].errorbar(x, test_vec_avg, yerr=test_vec_std, fmt='o', markersize=2.5, color='darkred', ecolor='lightcoral', elinewidth=1.5, capsize=0)


#     x = np.arange(45)+0.5
#     # Slow X
#     test_vec = np.empty((len(adapt_slowgainX_result[0][:,iii]),1))
#     for ii in np.arange(5):
#         vec = np.expand_dims(np.array(adapt_slowgainX_result[ii][:,iii]), axis=1)
#         test_vec = np.concatenate((test_vec, vec), axis=1)
#     test_vec = test_vec[:,1:]
#     test_vec_avg = np.mean(test_vec, axis=1)
#     test_vec_std = np.std(test_vec, axis=1)


#     axs[0,iii].errorbar(x, test_vec_avg, yerr=test_vec_std, fmt='o', markersize=2.5, color='darkblue', ecolor='lightsteelblue', elinewidth=1.5, capsize=0)
#     # Slow Y
#     test_vec = np.empty((len(adapt_slowgainY_result[0][:,iii]),1))
#     for ii in np.arange(5):
#         vec = np.expand_dims(np.array(adapt_slowgainY_result[ii][:,iii]), axis=1)
#         test_vec = np.concatenate((test_vec, vec), axis=1)
#     test_vec = test_vec[:,1:]
#     test_vec_avg = np.mean(test_vec, axis=1)
#     test_vec_std = np.std(test_vec, axis=1)

#     axs[1,iii].errorbar(x, test_vec_avg, yerr=test_vec_std, fmt='o', markersize=2.5, color='darkblue', ecolor='lightsteelblue', elinewidth=1.5, capsize=0)
#     axs[0,iii].plot(x, np.ones(len(x))*const10_X[iii],color='k')
#     axs[1,iii].plot(x, np.ones(len(x))*const10_Y[iii],color='k')

# plt.setp(axs[0,0], title='COM X Pos')
# plt.setp(axs[0,1], title='COM Z Pos')
# plt.setp(axs[0,2], title='COM X Vel')
# plt.setp(axs[0,3], title='COM Y Vel')
# plt.setp(axs[0,4], title='COM Z Vel')

# plt.setp(axs[0,0], ylabel='Foot X')
# plt.setp(axs[1,0], ylabel='Foot Y')
# fig.suptitle('Step-to-step controller gain as a function of locomotor adaptation (+2 Delta Case)')
# plt.legend(['Tied-Belt Baseline','Fast Leg', 'Slow Leg'])
# # ax = plt.gca()
# # ax.spines['right'].set_color('none')
# # ax.spines['top'].set_color('none')
# plt.show()


# file = open('controllerInferenceConst_14.pkl', 'rb')
# const_14_info_result = pickle.load(file)
# const_14_fastgainX_result = pickle.load(file)
# const_14_fastgainY_result = pickle.load(file)
# const_14_slowgainX_result = pickle.load(file)
# const_14_slowgainY_result = pickle.load(file)
# file.close()

# fig, axs = plt.subplots(2,10)
# for ii in np.arange(5):
#     if ii == 0:
#         color = 'r'
#     elif ii == 1:
#         color = 'g'
#     elif ii == 2:
#         color = 'b'
#     elif ii == 3:
#         color = 'm'
#     elif ii == 4:
#         color = 'y'

#     axs[0,0].plot(adapt_fastgainX_result[ii][:,0],'.', color=color)
#     axs[0,1].plot(adapt_fastgainX_result[ii][:,1],'.', color=color)
#     axs[0,2].plot(adapt_fastgainX_result[ii][:,2],'.', color=color)
#     axs[0,3].plot(adapt_fastgainX_result[ii][:,3],'.', color=color)
#     axs[0,4].plot(adapt_fastgainX_result[ii][:,4],'.', color=color)
#     axs[0,5].plot(adapt_slowgainX_result[ii][:,0],'.', color=color)
#     axs[0,6].plot(adapt_slowgainX_result[ii][:,1],'.', color=color)
#     axs[0,7].plot(adapt_slowgainX_result[ii][:,2],'.', color=color)
#     axs[0,8].plot(adapt_slowgainX_result[ii][:,3],'.', color=color)
#     axs[0,9].plot(adapt_slowgainX_result[ii][:,4],'.', color=color)
#     axs[0,0].plot(const_14_fastgainX_result[ii][:,0],'x', color=color)
#     axs[0,1].plot(const_14_fastgainX_result[ii][:,1],'x', color=color)
#     axs[0,2].plot(const_14_fastgainX_result[ii][:,2],'x', color=color)
#     axs[0,3].plot(const_14_fastgainX_result[ii][:,3],'x', color=color)
#     axs[0,4].plot(const_14_fastgainX_result[ii][:,4],'x', color=color)
#     axs[0,5].plot(const_10_fastgainX_result[ii][:,0],'x', color=color)
#     axs[0,6].plot(const_10_fastgainX_result[ii][:,1],'x', color=color)
#     axs[0,7].plot(const_10_fastgainX_result[ii][:,2],'x', color=color)
#     axs[0,8].plot(const_10_fastgainX_result[ii][:,3],'x', color=color)
#     axs[0,9].plot(const_10_fastgainX_result[ii][:,4],'x', color=color)

#     axs[1,0].plot(adapt_fastgainY_result[ii][:,0],'.', color=color)
#     axs[1,1].plot(adapt_fastgainY_result[ii][:,1],'.', color=color)
#     axs[1,2].plot(adapt_fastgainY_result[ii][:,2],'.', color=color)
#     axs[1,3].plot(adapt_fastgainY_result[ii][:,3],'.', color=color)
#     axs[1,4].plot(adapt_fastgainY_result[ii][:,4],'.', color=color)
#     axs[1,5].plot(adapt_fastgainY_result[ii][:,0],'.', color=color)
#     axs[1,6].plot(adapt_fastgainY_result[ii][:,1],'.', color=color)
#     axs[1,7].plot(adapt_fastgainY_result[ii][:,2],'.', color=color)
#     axs[1,8].plot(adapt_fastgainY_result[ii][:,3],'.', color=color)
#     axs[1,9].plot(adapt_fastgainY_result[ii][:,4],'.', color=color)
#     axs[1,0].plot(const_14_fastgainY_result[ii][:,0],'x', color=color)
#     axs[1,1].plot(const_14_fastgainY_result[ii][:,1],'x', color=color)
#     axs[1,2].plot(const_14_fastgainY_result[ii][:,2],'x', color=color)
#     axs[1,3].plot(const_14_fastgainY_result[ii][:,3],'x', color=color)
#     axs[1,4].plot(const_14_fastgainY_result[ii][:,4],'x', color=color)
#     axs[1,5].plot(const_10_fastgainY_result[ii][:,0],'x', color=color)
#     axs[1,6].plot(const_10_fastgainY_result[ii][:,1],'x', color=color)
#     axs[1,7].plot(const_10_fastgainY_result[ii][:,2],'x', color=color)
#     axs[1,8].plot(const_10_fastgainY_result[ii][:,3],'x', color=color)
#     axs[1,9].plot(const_10_fastgainY_result[ii][:,4],'x', color=color)

# plt.setp(axs[0,0], title='COM X Pos')
# plt.setp(axs[0,1], title='COM Z Pos')
# plt.setp(axs[0,2], title='Fast Leg \n COM X Vel')
# plt.setp(axs[0,3], title='COM Y Vel')
# plt.setp(axs[0,4], title='COM Z Vel')

# plt.setp(axs[0,5], title='COM X Pos')
# plt.setp(axs[0,6], title='COM Z Pos')
# plt.setp(axs[0,7], title='Slow Leg \n COM X Vel')
# plt.setp(axs[0,8], title='COM Y Vel')
# plt.setp(axs[0,9], title='COM Z Vel')
# plt.setp(axs[0,0], ylabel='Foot X')
# plt.setp(axs[1,0], ylabel='Foot Y')
# fig.suptitle('Step-to-step controller gain as a function of locomotor adaptation (+2 Delta Case)')
# plt.show()
