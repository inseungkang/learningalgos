
moveave_data = UF.moving_average(com_cont_r, 50)
filt_data = UF.low_pass_filter(com_cont_r)

print(moveave_data.shape)
print(filt_data.shape)
fig, axs = plt.subplots(5,2)

for ii in np.arange(5):
    axs[ii,0].plot(com_cont_r[:,ii], label = 'raw data')
    # axs[ii,0].plot(moveave_data[:,ii], label = 'mov ave')
    axs[ii,1].plot(com_cont_r[:,ii], label = 'raw data')
    axs[ii,0].set_xlim(73000, 75000)
    # axs[ii,1].plot(filt_data[:,ii], label = 'filter data')

axs[0,0].title.set_text('Moving Ave')
# axs[0,1].title.set_text('Low Pass Filter')
plt.setp(axs[0,0], ylabel='COM X Pos')
plt.setp(axs[1,0], ylabel='COM Z Pos')
plt.setp(axs[2,0], ylabel='COM X Vel')
plt.setp(axs[3,0], ylabel='COM Y Vel')
plt.setp(axs[4,0], ylabel='COM Z Vel')
axs[0,0].legend()
axs[0,1].legend()

plt.show()
