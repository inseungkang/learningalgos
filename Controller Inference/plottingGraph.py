

fig, axs = plt.subplots(5,2)

for ii in np.arange(5):
    axs[ii,0].plot(com_ms_r[:,ii], label = 'raw data')
    axs[ii,0].plot(testvec_1[:,ii], label = 'mov ave')
    axs[ii,1].plot(com_ms_r[:,ii], label = 'raw data')
    axs[ii,1].plot(testvec_2[:,ii], label = 'filter data')

axs[0,0].title.set_text('Moving Ave')
axs[0,1].title.set_text('Low Pass Filter')
# axs[0,0].title.set_text('Raw data')
# axs[0,1].title.set_text('Detrend')
# axs[0,2].title.set_text('Filt Delta')
plt.setp(axs[0,0], ylabel='COM X Pos')
plt.setp(axs[1,0], ylabel='COM Z Pos')
plt.setp(axs[2,0], ylabel='COM X Vel')
plt.setp(axs[3,0], ylabel='COM Y Vel')
plt.setp(axs[4,0], ylabel='COM Z Vel')
axs[0,0].legend()
axs[0,1].legend()
# axs[1,0].plot(COM_ms_r[:,1])
# axs[1,0].plot(y_1)
# axs[1,1].plot(delta_P_r[:,1])
# axs[1,2].plot(newY_1)

plt.show()