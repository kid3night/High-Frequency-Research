import numpy as np
import pandas as pd

def high_(data):
	if len(data) == 0:
		return np.nan
	return np.max(data)

def low_(data):
	if len(data) == 0:
		return np.nan
	return np.min(data)

def close_(data):
	if len(data) == 0:
		return np.nan
	return data[-1]

def vol_sum(data):
	if len(data) == 0:
		return np.nan
	return data.sum()

def max_(Series1, Series2):
	if len(Series1) < 1:
		return Series1
	concated_series = pd.concat([Series1, Series2], axis=1)
	return concated_series.max(axis=1)

def min_(Series1, Series2):
	if len(Series1) < 1:
		return Series1
	concated_series = pd.concat([Series1, Series2], axis=1)
	return concated_series.min(axis=1)

def mean_(data):
	if len(data) == 0:
		return np.nan
	return data.mean()

def iF(condition, value_1, value_2, index_series): 
	# first three parameters are numpy arrays
	if len(condition) < 1:
		return pd.Series(index=[])
	res = pd.Series(np.where(condition, value_1, value_2), index=index_series)
	return res

def centralize_std(data):
	if len(data) == 0:
		return np.nan
	mean_ = np.mean(data)
	std_ = np.std(data)
	if np.abs(std_) < 1e-6:
		return 0
	return (data[-1] - mean_) / std_

def centralize_max_min(data):
	if len(data) == 0:
		return np.nan
	mean_ = np.mean(data)
	high_ = np.max(data)
	low_ = np.min(data)
	if high_ == low_:
		return 0
	return (data[-1] - mean_) / (high_ - low_)

def lc_(data):
	if len(data) == 0:
		return np.nan
	return data[0]

def high_minus_low_(data):
	if len(data) == 0:
		return np.nan
	return np.max(data) - np.min(data)

def diff_mean(data):
	if len(data) < 1:
		return np.nan
	res = data[-1] - np.mean(data)
	return res

def argsort_final_pos(data):
	if len(data) < 1:
		return np.nan
	return np.argsort(data)[-1]

def diff_(data):
	if len(data) < 1:
		return np.nan
	res = data[-1] - data[0]
	return res

def pct_change_(data):
	if len(data) < 1:
		return np.nan
	if data[0] == data[-1] == 0:
		return 0
	if data[0] == 0 and data[-1] != 0:
		return np.sign(data[-1]) * 1
	return (data[-1] - data[0]) / data[0]

def close_lc_max(data):
	if len(data) < 1:
		return np.nan
	return np.max([data[-1] - data[0], 0])

def close_lc_abs(data):
	if len(data) < 1:
		return np.nan
	return np.abs(data[-1] - data[0])

def series_clean(Series_):
	result = pd.Series(np.where(np.isfinite(Series_.values), Series_.values, np.nan), index=Series_.index).ffill().fillna(0)
	return result

def array_clean(arrays):
	result = np.where(np.isfinite(arrays), arrays, 0)
	return result

def rolling_cal(raw_series, rolling_time, func):

	return raw_series.rolling(rolling_time, closed='left').apply(func, raw=True).ffill().fillna(0)

def rolling_corr(series1, series2, rolling_time):

	return series1.rolling(rolling_time, closed='left').corr(series2)
