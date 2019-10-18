import pandas as pd
import numpy as np
from baseSignal_Modified import base_feature
import talib as ta
# import feature_cal_supporting as fcs
from feature_cal_supporting import *


class Transaction_ACD(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		rolling_time_n = str(nperiod * 3) + 'S'
		rolling_time_n2 = str(nperiod * 20) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series
		close_series = rolling_cal(price_series, rolling_time_n, close_)
		high_series = rolling_cal(price_series, rolling_time_n, high_)
		low_series = rolling_cal(price_series, rolling_time_n, low_)
		LC = rolling_cal(price_series, rolling_time_n, lc_)
		condition_ = (close_series > LC).values
		DIF = close_series - iF(condition_, min_(low_series, LC).values, max_(high_series, LC).values, LC.index)
		condition_2 = (close_series == LC).values
		ACD_temp = iF(condition_2, 0, DIF.values, DIF.index)
		# ACD = ACD_temp.rolling(rolling_time_n2, closed='left').apply(vol_sum, raw=True)
		ACD = rolling_cal(ACD_temp, rolling_time_n2, vol_sum)
		# print(ACD)
		# print(ACD, rolling_time_n2)
		ACD_minus_MA_ACD = rolling_cal(ACD, rolling_time_n2, diff_mean)
		# ACD_minus_MA_ACD = ACD.rolling(rolling_time_n2, closed='left').apply(diff_mean, raw=True)
		# result = ACD_minus_MA_ACD.rolling(rolling_time_n2, closed='left').apply(centralize_std, raw=True)
		result = rolling_cal(ACD_minus_MA_ACD, rolling_time_n2, centralize_std)
		return result


class Transaction_MASS(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()

		rolling_time_n = str(nperiod * 3) + 'S'
		rolling_time_m = str(nperiod * 6) + 'S'

		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		high_minus_low = rolling_cal(price_series, rolling_time_n, high_minus_low_)
		ma_high_minus_low = rolling_cal(high_minus_low, rolling_time_n, mean_)
		mama_high_minus_low = rolling_cal(ma_high_minus_low, rolling_time_n, mean_)
		res_percent = ma_high_minus_low / mama_high_minus_low
		MASS_temp = pd.Series(np.where(np.isfinite(res_percent.values), res_percent.values, np.nan), index=high_minus_low.index).ffill().fillna(0)
		MASS = rolling_cal(MASS_temp, rolling_time_n, vol_sum)
		RES = rolling_cal(MASS, rolling_time_m, diff_mean)
		result = rolling_cal(RES, rolling_time_n, vol_sum)
		return result


class Transaction_VR(base_feature):
	param_list = ['nperiod']
	def feature(self, data_fed, nperiod):
		stk_data = data_fed['transaction_data'].dropna()
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		vols = stk_data['Turnover'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series
		if len(vols) < 1:
			return vols

		rolling_time_n = str(nperiod * 3) + 'S'
		rolling_time_m = str(nperiod * 6) + 'S'
		rolling_time = str(nperiod * 9) + 'S'

		close_series = rolling_cal(price_series, rolling_time_n, close_)
		lc_series = rolling_cal(price_series, rolling_time_n, lc_)
		vol_series = rolling_cal(vols, rolling_time_n, vol_sum)

		condition_1 = close_series > lc_series
		condition_2 = close_series < lc_series
		condition_3 = close_series == lc_series
		TH = (iF(condition_1, vol_series.values, 0, close_series.index)).rolling(rolling_time, closed='left').apply(vol_sum, raw=True)
		TL = (iF(condition_2, vol_series.values, 0, close_series.index)).rolling(rolling_time, closed='left').apply(vol_sum, raw=True)
		TQ = (iF(condition_3, vol_series.values, 0, close_series.index)).rolling(rolling_time, closed='left').apply(vol_sum, raw=True)
		VR = (TH * 2 + TQ) / (TL * 2 + TQ)
		VR = pd.Series(np.where(np.isfinite(VR.values), VR.values, np.nan), index=VR.index).ffill().fillna(0)
		if len(VR) < 1:
			return VR
		result = rolling_cal(VR, rolling_time_m, diff_mean)
		return result


class Transaction_ATR(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()

		rolling_time_n = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 6) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		close_series = rolling_cal(price_series, rolling_time_n, close_)
		high_series = rolling_cal(price_series, rolling_time_n, high_)
		low_series = rolling_cal(price_series, rolling_time_n, low_)
		lc_series = rolling_cal(price_series, rolling_time_n, lc_)

		high_low_minus = high_series - low_series
		high_minus_ref_close = (high_series - lc_series).abs()
		ref_close_minus_low = (lc_series - low_series).abs()
		MTR = (max_(max_(high_low_minus, high_minus_ref_close), ref_close_minus_low)).ffill().fillna(0)
		if len(MTR) < 1:
			return MTR
		result = rolling_cal(MTR, rolling_time_n, centralize_max_min)
		return result


class Transaction_WAD(base_feature):
	param_list = ['nperiod']
	def feature(self, data_fed, nperiod):
		stk_data = data_fed['transaction_data'].dropna()


		rolling_time_n = str(nperiod * 3) + 'S'
		M2 = nperiod * 6
		rolling_time_M2 = str(M2) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		close_series = rolling_cal(price_series, rolling_time_n, close_)
		high_series = rolling_cal(price_series, rolling_time_n, high_)
		low_series = rolling_cal(price_series, rolling_time_n, low_)
		lc_series = rolling_cal(price_series, rolling_time_n, lc_)

		res_MIDA = min_(lc_series, low_series)
		MIDA = (close_series - res_MIDA) / res_MIDA
		res_MIDB = max_(lc_series, high_series)
		r1 = (close_series - res_MIDB) / res_MIDB
		r2 = 0
		condition_ = close_series <= lc_series
		MIDB = iF(condition_, r1.values, r2, close_series.index)
		condition_new = ~condition_
		WAD_temp = iF(condition_new, MIDA.values, MIDB.values, close_series.index)
		WAD = rolling_cal(WAD_temp, rolling_time_M2, vol_sum)
		WAD = pd.Series(np.where(np.isfinite(WAD.values), WAD.values, np.nan), index=WAD.index).ffill().fillna(0)
		if len(WAD) < 1:
			return WAD
		result = rolling_cal(WAD, rolling_time_M2, diff_mean)
		return result


class Transaction_UOS(base_feature):
	param_list = ['nperiod']
	def feature(self, data_fed, nperiod):
		stk_data = data_fed['transaction_data'].dropna()


		rolling_time = str(nperiod * 3) + 'S'
		M1 = nperiod * 3
		M2 = nperiod * 6
		M3 = nperiod * 12
		rolling_time_M1 = str(M1) + 'S'
		rolling_time_M2 = str(M2) + 'S'
		rolling_time_M3 = str(M3) + 'S'

		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		close_series = rolling_cal(price_series, rolling_time, close_)
		high_series = rolling_cal(price_series, rolling_time, high_)
		low_series = rolling_cal(price_series, rolling_time, low_)
		lc_series = rolling_cal(price_series, rolling_time, lc_)

		TH = max_(high_series, lc_series)
		TL = min_(lc_series, low_series)
		numerator_ = close_series - TL
		denominator_ = TH - TL
		ACC1 = numerator_.rolling(rolling_time_M1,  closed='left').apply(vol_sum, raw=True) / denominator_.rolling(rolling_time_M1,  closed='left').apply(vol_sum, raw=True)
		ACC2 = numerator_.rolling(rolling_time_M2,  closed='left').apply(vol_sum, raw=True) / denominator_.rolling(rolling_time_M2,  closed='left').apply(vol_sum, raw=True)
		ACC3 = numerator_.rolling(rolling_time_M3,  closed='left').apply(vol_sum, raw=True) / denominator_.rolling(rolling_time_M3,  closed='left').apply(vol_sum, raw=True)
		UOS = ACC1 * M1 * M2 + ACC2 * M1 * M3 + ACC3 * M1 * M2 / (M1 * M2 + M1 * M3 + M1 * M2)
		UOS = pd.Series(np.where(np.isfinite(UOS.values), UOS.values, np.nan), index=UOS.index).ffill().fillna(0)
		if len(UOS) < 1:
			return UOS
		result = rolling_cal(UOS, rolling_time_M2, diff_mean)
		return result


class Transaction_DMI(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()

		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 18) + 'S'
		rolling_time_m = str(nperiod * 9) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		close_series = rolling_cal(price_series, rolling_time, close_)
		high_series = rolling_cal(price_series, rolling_time, high_)
		low_series = rolling_cal(price_series, rolling_time, low_)
		lc_series = rolling_cal(price_series, rolling_time, lc_)

		high_low_minus = high_series - low_series
		high_minus_ref_close = (high_series - lc_series).abs()
		ref_close_minus_low = (lc_series - low_series).abs()
		MTR_temp = max_(max_(high_low_minus, high_minus_ref_close), ref_close_minus_low)
		MTR = rolling_cal(MTR_temp, rolling_time, vol_sum)
		HD = rolling_cal(high_series, rolling_time, diff_)
		LD = -rolling_cal(low_series, rolling_time, diff_)
		DMP_temp = iF(((HD > 0) & (HD > LD)).values, HD.values, 0, HD.index)
		DMM_temp = iF(((LD > 0) & (LD > HD)).values, LD.values, 0, LD.index)
		if len(DMP_temp) < 1:
			return DMP_temp
		DMP = rolling_cal(DMP_temp, rolling_time_n, vol_sum)
		DMM = rolling_cal(DMM_temp, rolling_time_n, vol_sum)
		PDI = DMP * 100 / MTR
		MDI = DMM * 100 / MTR
		ADX_ = series_clean(((MDI - PDI) / (MDI + PDI)).abs())
		ADX_ = rolling_cal(ADX_, rolling_time_m, mean_)
		ADXR = (ADX_ + rolling_cal(ADX_, rolling_time_m, lc_)) / 2
		return ADXR


class Transaction_DMI_no_abs(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()

		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 18) + 'S'
		rolling_time_m = str(nperiod * 9) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		close_series = rolling_cal(price_series, rolling_time, close_)
		high_series = rolling_cal(price_series, rolling_time, high_)
		low_series = rolling_cal(price_series, rolling_time, low_)
		lc_series = rolling_cal(price_series, rolling_time, lc_)

		high_low_minus = high_series - low_series
		high_minus_ref_close = (high_series - lc_series).abs()
		ref_close_minus_low = (lc_series - low_series).abs()
		MTR_temp = max_(max_(high_low_minus, high_minus_ref_close), ref_close_minus_low)
		MTR = rolling_cal(MTR_temp, rolling_time_n, vol_sum)
		HD = rolling_cal(high_series, rolling_time, diff_)
		LD = -rolling_cal(low_series, rolling_time, diff_)
		DMP_temp = iF(((HD > 0) & (HD > LD)).values, HD.values, 0, HD.index)
		DMM_temp = iF(((LD > 0) & (LD > HD)).values, LD.values, 0, LD.index)
		if len(DMP_temp) < 1:
			return DMP_temp
		DMP = rolling_cal(DMP_temp, rolling_time_n, vol_sum)
		DMM = rolling_cal(DMM_temp, rolling_time_n, vol_sum)
		PDI = DMP * 100 / MTR
		MDI = DMM * 100 / MTR
		ADX_ = series_clean(((MDI - PDI) / (MDI + PDI)))
		result = rolling_cal(ADX_, rolling_time_m, mean_)
		return result


class Transaction_EMV(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		stk_data = stk_data[stk_data['FunctionCode'] == 48]
		vol_series = stk_data['Volume']
		price_series = stk_data['Price']
		if len(price_series) < 1:
			return price_series
		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 14) + 'S'
		rolling_time_m = str(nperiod * 9) + 'S'
		Vol = rolling_cal(vol_series, rolling_time, vol_sum)
		MA_Vol = rolling_cal(Vol, rolling_time_n, mean_)
		Volume = series_clean(MA_Vol / Vol)

		close_series = rolling_cal(price_series, rolling_time, close_)
		high_series = rolling_cal(price_series, rolling_time, high_)
		low_series = rolling_cal(price_series, rolling_time, low_)
		high_low_add = low_series + high_series
		high_low_minus = high_series - low_series
		Mid = rolling_cal(high_low_add, rolling_time, pct_change_)
		high_low_minus_MA = rolling_cal(high_low_minus, rolling_time_n, mean_)

		EMV_inside = series_clean(Mid * Volume * high_low_minus / high_low_minus_MA)
		EMV = rolling_cal(EMV_inside, rolling_time_m, mean_)
		MAEMV = rolling_cal(EMV, rolling_time_m, mean_)
		return EMV - MAEMV


class Transaction_CHO(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		vol_series = stk_data['Volume'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series
		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n1 = str(nperiod * 10) + 'S'
		rolling_time_n2 = str(nperiod * 20) + 'S'
		rolling_time_m = str(nperiod * 6) + 'S'

		close_series = rolling_cal(price_series, rolling_time, close_)
		high_series = rolling_cal(price_series, rolling_time, high_)
		low_series = rolling_cal(price_series, rolling_time, low_)
		Vol = rolling_cal(vol_series, rolling_time, vol_sum)
		Mid_p1 = Vol * (2 * close_series - high_series - low_series) / (high_series + low_series)
		Mid = rolling_cal(Mid_p1, rolling_time_n2, vol_sum)
		CHO = rolling_cal(Mid, rolling_time_n1, mean_) - rolling_cal(Mid, rolling_time_n2, mean_)
		MACHO = rolling_cal(CHO, rolling_time_n2, mean_)
		RES = CHO - MACHO
		result = rolling_cal(RES, rolling_time_m, centralize_max_min)
		return result


class Transaction_KDJ(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 9) + 'S'
		rolling_time_m = str(nperiod * 6) + 'S'

		close_series = rolling_cal(price_series, rolling_time, close_)
		high_series = rolling_cal(price_series, rolling_time, high_)
		low_series = rolling_cal(price_series, rolling_time, low_)
		llv = rolling_cal(low_series, rolling_time_n, low_)
		hhv = rolling_cal(high_series, rolling_time_n, high_)
		RSV = series_clean((close_series - llv) / (hhv - llv) * 100)
		K = rolling_cal(RSV, rolling_time_m, mean_)
		D = rolling_cal(RSV, rolling_time_n, mean_)
		J = 3 * K - 2 * D
		return J


class Transaction_RSV(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 9) + 'S'
		rolling_time_m = str(nperiod * 6) + 'S'

		close_series = rolling_cal(price_series, rolling_time, close_)
		high_series = rolling_cal(price_series, rolling_time, high_)
		low_series = rolling_cal(price_series, rolling_time, low_)
		llv = rolling_cal(low_series, rolling_time_n, low_)
		hhv = rolling_cal(high_series, rolling_time_n, high_)
		RSV = series_clean((close_series - llv) / (hhv - llv) * 100)
		return RSV


class Transaction_RSI(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		def func_info(data):
			if len(data) == 0:
				return 0
			close_change = np.max((data[-1] - data[0], 0))
			abs_change = np.abs(data[-1] - data[0]) if data[-1] != data[0] else 1
			return close_change / abs_change

		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 15) + 'S'
		RSI_numerator = rolling_cal(price_series, rolling_time, close_lc_max)
		RSI_denominator = rolling_cal(price_series, rolling_time, close_lc_abs)
		num = rolling_cal(RSI_numerator, rolling_time_n, mean_)
		denom = rolling_cal(RSI_denominator, rolling_time_n, mean_)
		result = series_clean(num / denom)
		return result


class Mid_Change_Origin(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				result = np.mean(data)
				return np.log(result) if result > 0 else -np.log(-result) if result < 0 else 0

		order_vols = stk_data.loc[:, ['BidVol1', 'AskVol1']].sum(axis=1)

		rolling_time = str(nperiod * 3) + 'S'
		mid_change_origin = order_vols * stk_data['Close'] * stk_data['Mid'].pct_change(1)
		mco_ = series_clean(mid_change_origin)
		return mco_.rolling(rolling_time, closed='left').apply(function_inside, raw=True)


class Tran_Price_Change_Vol(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data']
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		vol_series = stk_data['Volume'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		def function_inside1(data):
			if len(data) == 0:
				return 0
			else:
				return np.sign(data[-1] - data[0])

		def function_inside2(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = (data).sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 10) + 'S'
		selected_data = stk_data[stk_data['Price'] != 0]
		price_change_sign = price_series.rolling(rolling_time,  closed='left').apply(function_inside1, raw=True)
		vol_sum_decay = vol_series.rolling(rolling_time,  closed='left').apply(function_inside2, raw=True)
		RES = vol_sum_decay * price_change_sign
		result = rolling_cal(RES, rolling_time_n, centralize_std)
		return result


class VRSI(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].dropna()
		vol = stk_data['TransactionVol'].astype(float)
		if len(vol) < 1:
			return vol

		n = nperiod
		N = nperiod * 3
		diff_vol = vol.diff(n).fillna(0).values
		num = ta.EMA(np.where(diff_vol > 0, diff_vol, 0), N)
		denom = ta.EMA(np.abs(diff_vol), N)
		denom = np.where((num == 0), 1, denom)
		result = pd.Series(num / denom, index=vol.index)
		return result


class RSI_TA(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']
		vol = stk_data['Mid'].astype(float)
		if len(vol) < 1:
			return vol

		n = nperiod
		N = nperiod * 3
		diff_vol = vol.diff(n).fillna(0).values
		num = ta.EMA(np.where(diff_vol > 0, diff_vol, 0), N)
		denom = ta.EMA(np.abs(diff_vol), N)
		denom = np.where((num == 0), 1, denom)
		result = pd.Series(num / denom, index=vol.index)
		return result


class BIAS(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].dropna()
		price_series = stk_data['Mid'].astype(float)
		if len(price_series) < 1:
			return price_series

		n = nperiod
		N = nperiod * 5
		ma_price = ta.MA(price_series.values, N)
		ma_price = np.where(price_series.values == ma_price, 1, ma_price)
		result = pd.Series((price_series.values - ma_price) / ma_price, index=price_series.index).ffill().fillna(0) * 100
		return result


class PSY(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].dropna()
		price_series = stk_data['Mid'].astype(float)
		if len(price_series) < 1:
			return price_series

		n = nperiod
		N = nperiod * 3
		price = price_series.values
		ref_price = price_series.shift(1).fillna(0).values
		res = ta.SUM((price > ref_price).astype(float), N)
		res = np.where(np.isfinite(res), res, 0)
		result = pd.Series(ta.MA(res, N), index=price_series.index)
		return result


class Ask_Bid_CYR(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].dropna()
		tran_vol = stk_data['TransactionVol'].astype(float)
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


class Ask_Bid_OBV(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].dropna()
		tran_vol = stk_data['TransactionVol'].astype(float)
		price_series = stk_data['Mid'].astype(float)
		N = nperiod * 3
		M = nperiod
		if len(tran_vol) < 1:
			return tran_vol
		ref_price = price_series.shift(1).fillna(0)
		VA = np.sign(price_series.values - ref_price.values) * tran_vol.values
		OBV = ta.SUM(VA, M)
		MAOBV = ta.MA(OBV, N)
		STDOBV = ta.STDDEV(OBV, N)
		res = pd.Series((OBV - MAOBV)) / pd.Series(STDOBV)
		result = pd.Series(array_clean(res.values), index=price_series.index)
		return result


class Transaction_CYM(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data']
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		vol_series = stk_data['Volume'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 10) + 'S'
		close_series = rolling_cal(price_series, rolling_time, close_)
		high_series = rolling_cal(price_series, rolling_time, high_)
		low_series = rolling_cal(price_series, rolling_time, low_)
		Vol = rolling_cal(vol_series, rolling_time, vol_sum)
		temp_var = (2 * close_series - high_series - low_series) / (high_series - low_series) * Vol
		VAR4 = pd.Series(array_clean(np.where((high_series > low_series).values, temp_var.values, 0)), index=high_series.index)
		# print(rolling_time_n)
		MAVAR4 = rolling_cal(VAR4, rolling_time_n, mean_)
		RES = VAR4 - MAVAR4
		result = rolling_cal(RES, rolling_time_n, centralize_max_min)
		return result


class Ask_Bid_CYS(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].dropna()
		tran_vol = stk_data['TransactionVol'].astype(float)
		tran_Amount = stk_data['TransactionAmount'].astype(float)
		price_series = stk_data['Mid'].astype(float)
		N = nperiod * 3
		rolling_time_n = str(nperiod * 6) + 'S'
		if len(tran_vol) < 1:
			return tran_vol

		res = ta.EMA(tran_Amount.values, N) / ta.EMA(tran_vol.values, N) * 10000
		result = pd.Series(array_clean((price_series.values - res) / res * 100), index=price_series.index)
		result = rolling_cal(result, rolling_time_n, centralize_max_min)
		return result


class Ask_Bid_VMACD(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].ffill().bfill()
		mid = stk_data['TransactionVol'].values.astype(float)
		if len(mid) < 1:
			return stk_data['TransactionVol']
		short_n = nperiod * 3
		long_n = nperiod * 6
		mid_n = nperiod * 2
		macd = ta.MACD(mid, fastperiod=short_n, slowperiod=long_n, signalperiod=mid_n)[0]
		result = pd.Series(macd, index=stk_data.index).pct_change(1)
		return result


class Ask_Bid_MACD(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].ffill().bfill()
		mid = stk_data['Mid'].values.astype(float)
		if len(mid) < 1:
			return stk_data['Mid']
		short_n = nperiod * 3
		long_n = nperiod * 6
		mid_n = nperiod * 2
		macd = ta.MACD(mid, fastperiod=short_n, slowperiod=long_n, signalperiod=mid_n)[0]
		result = pd.Series(macd, index=stk_data.index).pct_change(1)
		return result


class Ask_Bid_1_decay(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				result = data[-1] - data[0]
				return result if np.isfinite(result) else 0

		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
		rolling_time = str(nperiod * 3) + 'S'
		bid_part_decay = np.sum(stk_data.loc[:, bid_part], axis=1)
		ask_part_decay = np.sum(stk_data.loc[:, ask_part], axis=1)
		diff_percent = (bid_part_decay - ask_part_decay) / (bid_part_decay + ask_part_decay)
		vol_diff = pd.Series(diff_percent, index=stk_data.index)
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)


class Tran_Type_Num_Diff(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = data.sum()
				return net_sum

		def function_inside_denominator(data):
			if len(data) == 0:
				return 1
			else:
				net_sum = data.sum()
				return net_sum if net_sum != 0 else 1

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[stk_data['FunctionCode'] != 67]
		bs_multiplier = pd.Series(np.where(selected_data['BSFlag'].values == 66, 1, -1), index=selected_data.index)
		abs_multiplier = bs_multiplier.abs()
		numerator = bs_multiplier.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)
		denominator = abs_multiplier.rolling(rolling_time,  closed='left').apply(function_inside_denominator, raw=True)
		return numerator / denominator


class Transaction_Net_Vol(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		def function_inside_denominator(data):
			if len(data) == 0:
				return 1
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 1

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[(stk_data['BSFlag'] == 66) | (stk_data['BSFlag'] == 83)]
		bs_multiplier = np.where(selected_data['BSFlag'].values == 66, 1, -1)
		signed_turnover = selected_data['Volume'] * bs_multiplier
		unsigned_turnover = selected_data['Volume']
		numerator = signed_turnover.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)
		denominator = unsigned_turnover.rolling(rolling_time,  closed='left').apply(function_inside_denominator, raw=True)
		return numerator / denominator


class Order_Direction_Volume_decay(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['order_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		def function_inside_denominator(data):
			if len(data) == 0:
				return 1
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 1

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[(stk_data['FunctionCode'] == 66) | (stk_data['FunctionCode'] == 83)]
		direction_data = -np.sign(selected_data['FunctionCode'] - 70) * selected_data['Volume']
		numerator = direction_data.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)
		denominator = selected_data['Volume'].rolling(rolling_time,  closed='left').apply(function_inside_denominator, raw=True)
		return numerator / denominator


class Ask_Bid_Spread_rate(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].dropna()
		mid = stk_data['Mid'].values
		ask1 = stk_data['AskPrice1'].values
		bid1 = stk_data['BidPrice1'].values
		high_limit = stk_data['HighLimit'].values
		low_limit = stk_data['LowLimit'].values
		if len(mid) == 0:
			return stk_data['Mid']
		elif len(mid) == 1:
			return pd.Series(0, index=stk_data.index)
		result_ = (ask1 - bid1) / mid / 2
		result = np.where(((ask1 == 0) | (bid1 == 0) | (mid >= high_limit) | (mid <= low_limit)), 0, result_)
		result = pd.Series(result, index=stk_data.index)
		return result


class Transaction_VPT(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		stk_data = stk_data[stk_data['FunctionCode'] == 48]
		vol_series = stk_data['Volume']
		price_series = stk_data['Price']
		if len(price_series) < 1:
			return price_series
		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 9) + 'S'
		rolling_time_m = str(nperiod * 6) + 'S'
		Vol = rolling_cal(vol_series, rolling_time, vol_sum)
		close_series = rolling_cal(price_series, rolling_time, close_)
		lc_close_series = rolling_cal(price_series, rolling_time, lc_)
		temp_vpt = series_clean((close_series - lc_close_series) / lc_close_series * Vol)
		VPT = rolling_cal(temp_vpt, rolling_time_n, vol_sum)
		RES = rolling_cal(MA_VPT, rolling_time_m, centralize_std)
		return RES


class Transaction_WVAD(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		stk_data = stk_data[stk_data['FunctionCode'] == 48]
		vol_series = stk_data['Volume']
		price_series = stk_data['Price']
		if len(price_series) < 1:
			return price_series
		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 9) + 'S'
		rolling_time_m = str(nperiod * 6) + 'S'
		Vol = rolling_cal(vol_series, rolling_time, vol_sum)
		close_series = rolling_cal(price_series, rolling_time, close_)
		high_series = rolling_cal(price_series, rolling_time, high_)
		low_series = rolling_cal(price_series, rolling_time, low_)
		open_series = rolling_cal(price_series, rolling_time, lc_)
		pre_WVAD = series_clean((close_series - open_series) / (high_series - low_series) * Vol)
		WVAD = rolling_cal(pre_WVAD, rolling_time_n, vol_sum)
		RES = rolling_cal(WVAD, rolling_time_m, centralize_std)
		return RES


class Transaction_AMV(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		stk_data = stk_data[stk_data['FunctionCode'] == 48]
		vol_series = stk_data['Volume']
		price_series = stk_data['Price']
		if len(price_series) < 1:
			return price_series
		rolling_time = str(nperiod * 3) + 'S'
		rolling_time_n = str(nperiod * 9) + 'S'
		rolling_time_m = str(nperiod * 6) + 'S'
		Vol = rolling_cal(vol_series, rolling_time, vol_sum)
		close_series = rolling_cal(price_series, rolling_time, close_)
		high_series = rolling_cal(price_series, rolling_time, high_)
		low_series = rolling_cal(price_series, rolling_time, low_)
		open_series = rolling_cal(price_series, rolling_time, lc_)
		AMOV = Vol * (close_series + open_series) / 2
		SUM_AMOV = rolling_cal(AMOV, rolling_time_n, vol_sum)
		SUM_VOL = rolling_cal(Vol, rolling_time_n, vol_sum)
		Mid_High_low = (high_series + low_series) / 2
		RES = (SUM_AMOV / SUM_VOL / Mid_High_low) * 100
		return RES


class Transaction_RSI_EMA(base_feature):  ##  基本数值离散0, 1
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()

		def func_info(data):
			if len(data) == 0:
				return 0
			close_change = np.max((data[-1] - data[0], 0))
			abs_change = np.abs(data[-1] - data[0]) if data[-1] != data[0] else 1
			return close_change / abs_change

		rolling_time = str(nperiod * 3) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		res = series_clean(price_series.rolling(rolling_time,  closed='left').apply(func_info, raw=True))
		if len(res) < 1:
			return res
		
		result = pd.Series(ta.EMA(res.values, int(nperiod * 3)), index=res.index)
		return result