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
		if len(UOS) < 1:
			return UOS
		EMAUOS = pd.Series(ta.EMA(UOS.values, M1), index=UOS.index)
		return UOS - EMAUOS


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
		if len(price_series) < 1:
			return price_series
		high_series = price_series.rolling(rolling_time_M3,  closed='left').apply(high_, raw=True)
		low_series = price_series.rolling(rolling_time_M3,  closed='left').apply(low_, raw=True)
		close_series = price_series.rolling(rolling_time_M3,  closed='left').apply(close_, raw=True)
		res_MIDA = min_(close_series.shift(1), low_series)
		MIDA = (close_series - res_MIDA) / res_MIDA
		res_MIDB = max_(close_series.shift(1), high_series)
		r1 = (close_series - res_MIDB) / res_MIDB
		r2 = 0
		condition_ = close_series <= close_series.shift(1)
		MIDB = iF(condition_, r1.values, r2, close_series.index)
		condition_new = ~condition_
		WAD = (iF(condition_new, MIDA.values, MIDB.values, close_series.index)).rolling(rolling_time_M2, closed='left').apply(vol_sum, raw=True)
		WAD = pd.Series(np.where(np.isfinite(WAD.values), WAD.values, np.nan), index=WAD.index).ffill().fillna(0)
		if len(WAD) < 1:
			return WAD
		EMAWAD = pd.Series(ta.EMA(WAD.values, M2), index=WAD.index)
		return WAD - EMAWAD


class Transaction_VR(base_feature):
	param_list = ['nperiod']
	def feature(self, data_fed, nperiod):
		stk_data = data_fed['transaction_data'].dropna()
		prices = stk_data['Price'][stk_data['FunctionCode'] == 48]
		vols = stk_data['Turnover'][stk_data['FunctionCode'] == 48]
		if len(prices) < 1:
			return prices
		if len(vols) < 1:
			return vols

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

		rolling_time = str(nperiod * 12) + 'S'
		close_series = prices.rolling(rolling_time,  closed='left').apply(close_, raw=True)
		vol_series = vols.rolling(rolling_time,  closed='left').apply(vol_sum, raw=True)
		condition_1 = close_series > close_series.shift(1)
		condition_2 = close_series < close_series.shift(1)
		condition_3 = close_series == close_series.shift(1)
		TH = (iF(condition_1, vol_series.values, 0, close_series.index)).rolling(rolling_time, closed='left').apply(vol_sum, raw=True)
		TL = (iF(condition_2, vol_series.values, 0, close_series.index)).rolling(rolling_time, closed='left').apply(vol_sum, raw=True)
		TQ = (iF(condition_3, vol_series.values, 0, close_series.index)).rolling(rolling_time, closed='left').apply(vol_sum, raw=True)
		M = nperiod * 6
		VR = (TH * 2 + TQ) / (TL * 2 + TQ)
		VR = pd.Series(np.where(np.isfinite(VR.values), VR.values, np.nan), index=VR.index).ffill().fillna(0)
		if len(VR) < 1:
			return VR
		EMAVR = pd.Series(ta.EMA(VR.values, M), index=VR.index)
		return VR - EMAVR


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

		rolling_time_n = str(nperiod * 6) + 'S'
		N = nperiod * 12
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series
		high_series = price_series.rolling(rolling_time_n,  closed='left').apply(high_, raw=True)
		low_series = price_series.rolling(rolling_time_n,  closed='left').apply(low_, raw=True)
		close_series = price_series.rolling(rolling_time_n,  closed='left').apply(close_, raw=True)

		high_low_minus = high_series - low_series
		high_minus_ref_close = (high_series - close_series.shift(1)).abs()
		ref_close_minus_low = (close_series.shift(1) - low_series).abs()
		MTR = (max_(max_(high_low_minus, high_minus_ref_close), ref_close_minus_low)).ffill().fillna(0)
		if len(MTR) < 1:
			return MTR
		EMAMTR = pd.Series(ta.MA(MTR.values, N), index=MTR.index)
		res = MTR / EMAMTR - 1
		result = pd.Series(np.where(np.isfinite(res.values), res.values, np.nan), index=res.index).ffill().fillna(0)
		return result


class Ask_Bid_CYR(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].ffill().bfill()
		tran_vol = stk_data['TransactionVol'].astype(float)
		tran_amount = stk_data['TransactionAmount'].astype(float)
		N = nperiod * 3
		M = nperiod
		if len(tran_vol) < 1:
			return tran_vol
		DIVE = pd.Series(ta.EMA(tran_amount.values, N) / ta.EMA(tran_vol.values, N), index=tran_vol.index)
		CYR = DIVE.pct_change()
		CYR = pd.Series(np.where(np.isfinite(CYR.values), CYR.values, np.nan), index=CYR.index).ffill().fillna(0)
		MACYR = pd.Series(ta.EMA(CYR.values, M), index=CYR.index)
		result = CYR - MACYR
		return result


class Transaction_MASS(base_feature):
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

		def centralize(data):
			if len(data) == 0:
				return 0
			mean_ = np.mean(data)
			high_ = np.max(data)
			low_ = np.min(data)
			if high_ == low_:
				return 0
			return (data[-1] - mean_) / (high_ - low_)

		rolling_time_n = str(nperiod * 8) + 'S'
		N = nperiod * 3
		N2 = nperiod * 8
		M = nperiod * 2
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series
		high_series = price_series.rolling(rolling_time_n, closed='left').apply(high_, raw=True)
		low_series = price_series.rolling(rolling_time_n, closed='left').apply(low_, raw=True)
		high_minus_low = high_series - low_series
		ma_high_minus_low = ta.EMA(high_minus_low.values, N)
		mama_high_minus_low = ta.EMA(ma_high_minus_low, N)
		res_percent = ma_high_minus_low / mama_high_minus_low
		MASS_temp = pd.Series(np.where(np.isfinite(res_percent), res_percent, np.nan), index=high_minus_low.index).ffill().fillna(0)
		MASS = MASS_temp.rolling(rolling_time_n, closed='left').apply(vol_sum, raw=True).ffill()
		MAMASS = pd.Series(ta.EMA(MASS.values, M), index=MASS.index)
		RES = MASS - MAMASS
		result = RES.rolling(rolling_time_n, closed='left').apply(vol_sum, raw=True).ffill().fillna(0)
		return result


class Transaction_ACD(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()

		def lc_(data):
			if len(data) == 0:
				return np.nan
			return data[0]

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

		def centralize(data):
			if len(data) == 0:
				return 0
			std_ = np.std(data)
			if np.abs(std_) < 1e-6:
				return 0
			return (data[-1] - mean_) / std_

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

		def diff_mean(data):
			if len(data) < 1:
				return np.nan
			res = data[-1] - np.mean(data)


		rolling_time_n = str(nperiod * 3) + 'S'
		rolling_time_n2 = str(nperiod * 25) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series
		close_ = price_series.rolling(rolling_time_n, closed='left').apply(close_, raw=True)
		high_ = price_series.rolling(rolling_time_n, closed='left').apply(high_, raw=True)
		low_ = price_series.rolling(rolling_time_n, closed='left').apply(low_, raw=True)
		LC = price_series.rolling(rolling_time_n, closed='left').apply(lc_, raw=True)
		condition_ = (close_ > LC).values
		DIF = close_ - iF(condition_, min_(low_, LC).values, max_(high_, LC).values, LC.index)
		condition_2 = (close_ == LC).values
		ACD_temp = iF(condition_2, 0, DIF.values, DIF.index)
		ACD = ACD_temp.rolling(rolling_time_n2, closed='left').apply(vol_sum, raw=True)
		ACD_minus_MA_ACD = ACD.rolling(rolling_time_n2, closed='left').apply(diff_mean, raw=True)
		result = ACD_minus_MA_ACD.rolling(rolling_time_n2, closed='left').apply(centralize, raw=True)
		return result
