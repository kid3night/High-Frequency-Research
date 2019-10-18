import pandas as pd
import numpy as np
from baseSignal_Modified import base_feature
import talib as ta

##  possibly need to be transfered

class Transaction_UOS(base_feature):
	param_list = ['nperiod']
	def feature(self, data_fed, nperiod):
		stk_data = data_fed['transaction_data'].dropna()

		def high_(data):
			if len(data) == 0:
				return 0
			return np.max(data)

		def low_(data):
			if len(data) == 0:
				return 0
			return np.min(data)

		def close_(data):
			if len(data) == 0:
				return 0
			return data[-1]

		def vol_sum(data):
			if len(data) == 0:
				return 0
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

		rolling_time = str(nperiod * 3) + 'S'
		M1 = nperiod * 3
		M2 = nperiod * 6
		M3 = nperiod * 12
		rolling_time_M1 = str(M1) + 'S'
		rolling_time_M2 = str(M2) + 'S'
		rolling_time_M3 = str(M3) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		high_series = price_series.rolling(rolling_time,  closed='left').apply(high_, raw=True)
		low_series = price_series.rolling(rolling_time,  closed='left').apply(low_, raw=True)
		close_series = price_series.rolling(rolling_time,  closed='left').apply(close_, raw=True)
		TH = max_(high_series, close_series.shift(1))
		TL = min_(close_series.shift(1), low_series)
		numerator_ = close_series - TL
		denominator_ = TH - TL
		ACC1 = numerator_.rolling(rolling_time_M1,  closed='left').apply(vol_sum, raw=True) / denominator_.rolling(rolling_time_M1,  closed='left').apply(vol_sum, raw=True)
		ACC2 = numerator_.rolling(rolling_time_M2,  closed='left').apply(vol_sum, raw=True) / denominator_.rolling(rolling_time_M2,  closed='left').apply(vol_sum, raw=True)
		ACC3 = numerator_.rolling(rolling_time_M3,  closed='left').apply(vol_sum, raw=True) / denominator_.rolling(rolling_time_M3,  closed='left').apply(vol_sum, raw=True)
		UOS = ACC1 * M1 * M2 + ACC2 * M1 * M3 + ACC3 * M1 * M2 / (M1 * M2 + M1 * M3 + M1 * M2)
		UOS = pd.Series(np.where(np.isfinite(UOS.values), UOS.values, np.nan), index=UOS.index).ffill().fillna(0)
		# EMAUOS = pd.Series(ta.EMA(UOS.values, M1), index=UOS.index)
		return UOS


class Transaction_WAD(base_feature):
	param_list = ['nperiod']
	def feature(self, data_fed, nperiod):
		stk_data = data_fed['transaction_data'].dropna()

		def high_(data):
			if len(data) == 0:
				return 0
			return np.max(data)

		def low_(data):
			if len(data) == 0:
				return 0
			return np.min(data)

		def close_(data):
			if len(data) == 0:
				return 0
			return data[-1]

		def vol_sum(data):
			if len(data) == 0:
				return 0
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

		def iF(condition, value_1, value_2, index_series): 
			# first three parameters are numpy arrays
			if len(condition) < 1:
				return pd.Series(index=[])
			res = pd.Series(np.where(condition, value_1, value_2), index=index_series)
			return res


		rolling_time = str(nperiod * 3) + 'S'
		M2 = nperiod * 6
		M3 = nperiod * 12
		rolling_time_M2 = str(M2) + 'S'
		rolling_time_M3 = str(M3) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		high_series = price_series.rolling(rolling_time,  closed='left').apply(high_, raw=True)
		low_series = price_series.rolling(rolling_time,  closed='left').apply(low_, raw=True)
		close_series = price_series.rolling(rolling_time,  closed='left').apply(close_, raw=True)
		res_MIDA = min_(close_series.shift(1), low_series)
		MIDA = (close_series - res_MIDA) / res_MIDA
		res_MIDB = max_(close_series.shift(1), high_series)
		r1 = (close_series - res_MIDB) / res_MIDB
		r2 = 0
		condition_ = close_series <= close_series.shift(1)
		MIDB = iF(condition_, r1, r2, close_series.index)
		condition_new = ~condition_
		WAD = (iF(condition_new, MIDA, MIDB, close_series.index)).rolling(rolling_time_M3, closed='left').apply(vol_sum, raw=True)
		# EMAWAD = pd.Series(ta.EMA(WAD.values, M3), index=WAD.index)
		return WAD


class Transaction_VR(base_feature):
	param_list = ['nperiod']
	def feature(self, data_fed, nperiod):
		stk_data = data_fed['transaction_data'].dropna()
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		vol_series = stk_data['Turnover'][stk_data['FunctionCode'] == 48]

		def close_(data):
			if len(data) == 0:
				return 0
			return data[-1]

		def vol_sum(data):
			if len(data) == 0:
				return 0
			return data.sum()

		def iF(condition, value_1, value_2, index_series): 
			# first three parameters are numpy arrays
			if len(condition) < 1:
				return pd.Series(index=[])
			res = pd.Series(np.where(condition, value_1, value_2), index=index_series)
			return res

		close_series = price_series.rolling(rolling_time,  closed='left').apply(close_, raw=True)
		vol_series = price_series.rolling(rolling_time,  closed='left').apply(vol_sum, raw=True)
		condition_1 = close_series > close_series.shift(1)
		condition_2 = close_series < close_series.shift(1)
		condition_3 = close_series == close_series.shift(1)
		rolling_time = str(nperiod * 12) + 'S'
		TH = (iF(condition_1, vol_series, 0, close_series.index)).rolling(rolling_time, closed='left').apply(vol_sum, raw=True)
		TH = (iF(condition_2, vol_series, 0, close_series.index)).rolling(rolling_time, closed='left').apply(vol_sum, raw=True)
		TQ = (iF(condition_3, vol_series, 0, close_series.index)).rolling(rolling_time, closed='left').apply(vol_sum, raw=True)
		M = nperiod * 6
		VR = (TH * 2 + TQ) / (TL * 2 + TQ)
		VR = pd.Series(np.where(np.isfinite(VR.values), VR.values, np.nan), index=VR.index).ffill().fillna(0)
		# EMAVR = pd.Series(ta.EMA(VR.values, M), index=VR.index)
		return VR


class Transaction_ATR(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()

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

		rolling_time_n = str(nperiod * 9) + 'S'
		M = nperiod * 24
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		high_series = price_series.rolling(rolling_time_n,  closed='left').apply(high_, raw=True)
		low_series = price_series.rolling(rolling_time_n,  closed='left').apply(low_, raw=True)
		close_series = price_series.rolling(rolling_time_n,  closed='left').apply(close_, raw=True)

		high_low_minus = high_series - low_series
		high_minus_ref_close = (high_series - close_series.shift(1)).abs()
		ref_close_minus_low = (close_series.shift(1) - low_series).abs()
		MTR = (max_(max_(high_low_minus, high_minus_ref_close), ref_close_minus_low)).ffill().fillna(0)
		# if len(MTR) < 1:
		# 	return MTR
		# EMAMTR = pd.Series(ta.MA(MTR.values, N), index=MTR.index)
		# res = MTR / EMAMTR - 1
		# result = pd.Series(np.where(np.isfinite(res.values), res.values, np.nan), index=res.index).ffill().fillna(0)
		return MTR
