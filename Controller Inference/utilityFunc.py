import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import mean_squared_error


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

def list_to_array(input_vec):
  test_vec = np.empty((len(input_vec[0]),1))
  for ii in np.arange(len(input_vec)):
    vec = np.expand_dims(np.array(input_vec[ii]), axis=1)
    test_vec = np.concatenate((test_vec, vec), axis=1)
  new_vec = test_vec[:,1:]
  return new_vec

def rmse(actual, predict):
  rms = np.sqrt(mean_squared_error(actual, predict))
  return rms

def zscore(actual, predict, mean, std):
  error = actual - predict
  mean = np.expand_dims(np.ones(len(error))*mean, axis=1)
  score = ((actual - predict) - mean)/std
  score = np.mean(np.abs(score))
  return score

def filt_outlier(input_vec):
  test_vec_d = np.abs(np.diff(input_vec))
  for ii in np.arange(len(test_vec_d)):
      if test_vec_d[ii] > np.std(test_vec_d)*2 and ii > 3:
          input_vec[ii] = (input_vec[ii-1]+ input_vec[ii+1])/2
  return input_vec