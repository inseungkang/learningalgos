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

file = open('controllerInferenceConst_10.pkl', 'rb')
const_10_info_result = pickle.load(file)
const_10_fastgainX_result = pickle.load(file)
const_10_fastgainY_result = pickle.load(file)
const_10_slowgainX_result = pickle.load(file)
const_10_slowgainY_result = pickle.load(file)
file.close()

file = open('controllerInferenceConst_14.pkl', 'rb')
const_14_info_result = pickle.load(file)
const_14_fastgainX_result = pickle.load(file)
const_14_fastgainY_result = pickle.load(file)
const_14_slowgainX_result = pickle.load(file)
const_14_slowgainY_result = pickle.load(file)
file.close()

fig, axs = plt.subplots(2,10)
for ii in np.arange(5):
    if ii == 0:
        color = 'r'
    elif ii == 1:
        color = 'g'
    elif ii == 2:
        color = 'b'
    elif ii == 3:
        color = 'm'
    elif ii == 4:
        color = 'y'

    axs[0,0].plot(adapt_fastgainX_result[ii][:,0],'.', color=color)
    axs[0,1].plot(adapt_fastgainX_result[ii][:,1],'.', color=color)
    axs[0,2].plot(adapt_fastgainX_result[ii][:,2],'.', color=color)
    axs[0,3].plot(adapt_fastgainX_result[ii][:,3],'.', color=color)
    axs[0,4].plot(adapt_fastgainX_result[ii][:,4],'.', color=color)
    axs[0,5].plot(adapt_slowgainX_result[ii][:,0],'.', color=color)
    axs[0,6].plot(adapt_slowgainX_result[ii][:,1],'.', color=color)
    axs[0,7].plot(adapt_slowgainX_result[ii][:,2],'.', color=color)
    axs[0,8].plot(adapt_slowgainX_result[ii][:,3],'.', color=color)
    axs[0,9].plot(adapt_slowgainX_result[ii][:,4],'.', color=color)
    axs[0,0].plot(const_14_fastgainX_result[ii][:,0],'x', color=color)
    axs[0,1].plot(const_14_fastgainX_result[ii][:,1],'x', color=color)
    axs[0,2].plot(const_14_fastgainX_result[ii][:,2],'x', color=color)
    axs[0,3].plot(const_14_fastgainX_result[ii][:,3],'x', color=color)
    axs[0,4].plot(const_14_fastgainX_result[ii][:,4],'x', color=color)
    axs[0,5].plot(const_10_fastgainX_result[ii][:,0],'x', color=color)
    axs[0,6].plot(const_10_fastgainX_result[ii][:,1],'x', color=color)
    axs[0,7].plot(const_10_fastgainX_result[ii][:,2],'x', color=color)
    axs[0,8].plot(const_10_fastgainX_result[ii][:,3],'x', color=color)
    axs[0,9].plot(const_10_fastgainX_result[ii][:,4],'x', color=color)

    axs[1,0].plot(adapt_fastgainY_result[ii][:,0],'.', color=color)
    axs[1,1].plot(adapt_fastgainY_result[ii][:,1],'.', color=color)
    axs[1,2].plot(adapt_fastgainY_result[ii][:,2],'.', color=color)
    axs[1,3].plot(adapt_fastgainY_result[ii][:,3],'.', color=color)
    axs[1,4].plot(adapt_fastgainY_result[ii][:,4],'.', color=color)
    axs[1,5].plot(adapt_fastgainY_result[ii][:,0],'.', color=color)
    axs[1,6].plot(adapt_fastgainY_result[ii][:,1],'.', color=color)
    axs[1,7].plot(adapt_fastgainY_result[ii][:,2],'.', color=color)
    axs[1,8].plot(adapt_fastgainY_result[ii][:,3],'.', color=color)
    axs[1,9].plot(adapt_fastgainY_result[ii][:,4],'.', color=color)
    axs[1,0].plot(const_14_fastgainY_result[ii][:,0],'x', color=color)
    axs[1,1].plot(const_14_fastgainY_result[ii][:,1],'x', color=color)
    axs[1,2].plot(const_14_fastgainY_result[ii][:,2],'x', color=color)
    axs[1,3].plot(const_14_fastgainY_result[ii][:,3],'x', color=color)
    axs[1,4].plot(const_14_fastgainY_result[ii][:,4],'x', color=color)
    axs[1,5].plot(const_10_fastgainY_result[ii][:,0],'x', color=color)
    axs[1,6].plot(const_10_fastgainY_result[ii][:,1],'x', color=color)
    axs[1,7].plot(const_10_fastgainY_result[ii][:,2],'x', color=color)
    axs[1,8].plot(const_10_fastgainY_result[ii][:,3],'x', color=color)
    axs[1,9].plot(const_10_fastgainY_result[ii][:,4],'x', color=color)

plt.setp(axs[0,0], title='COM X Pos')
plt.setp(axs[0,1], title='COM Z Pos')
plt.setp(axs[0,2], title='Fast Leg \n COM X Vel')
plt.setp(axs[0,3], title='COM Y Vel')
plt.setp(axs[0,4], title='COM Z Vel')

plt.setp(axs[0,5], title='COM X Pos')
plt.setp(axs[0,6], title='COM Z Pos')
plt.setp(axs[0,7], title='Slow Leg \n COM X Vel')
plt.setp(axs[0,8], title='COM Y Vel')
plt.setp(axs[0,9], title='COM Z Vel')
plt.setp(axs[0,0], ylabel='Foot X')
plt.setp(axs[1,0], ylabel='Foot Y')
fig.suptitle('Step-to-step controller gain as a function of locomotor adaptation (+2 Delta Case)')
plt.show()
