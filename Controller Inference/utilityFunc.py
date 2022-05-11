import numpy as np
import pandas as pd
from scipy import signal

def moving_average(y, N):
  y_final = np.empty(y.shape)
  for idx in np.arange(y.shape[1]):
    y_padded = np.pad(y[:,idx], (N//2, N-1-N//2), mode='edge')
    y_final[:,idx] = np.convolve(y_padded, np.ones((N,))/N, mode='valid')
  return y_final

def linear_detrend(input_vec, bin_size):
  output_vec = np.empty((input_vec.shape[0],))
  for ii in np.arange(input_vec.shape[1]):

    testing_vec = input_vec[:,ii]
    slices = np.linspace(0, len(testing_vec), int(len(testing_vec)/bin_size) + 1, True).astype(int)

    detrended_vec = []
    for ii, _ in enumerate(slices[:-1]):
      detrended_vec = np.append(detrended_vec, signal.detrend(testing_vec[slices[ii]:slices[ii+1]]))
    output_vec = np.column_stack((output_vec, detrended_vec))

  return output_vec[:,1:]

def low_pass_filter(input_vec):
    b, a = signal.butter(2, 0.02)
    output_vec = np.empty((input_vec.shape[0],))

    for ii in np.arange(input_vec.shape[1]):
        testing_vec = input_vec[:,ii]
        detrended_vec = signal.filtfilt(b, a, testing_vec, method="gust")
        output_vec = np.column_stack((output_vec, detrended_vec))

    return output_vec[:,1:]