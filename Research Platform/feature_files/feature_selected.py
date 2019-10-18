import pandas as pd
import numpy as np
from baseSignal_Modified import base_feature
import talib as ta
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
		ACD = rolling_cal(ACD_temp, rolling_time_n2, vol_sum)
		result = rolling_cal(ACD, rolling_time_n2, centralize_std)
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
		# if len(UOS) < 1:
		# 	return UOS
		# result = rolling_cal(UOS, rolling_time_M2, diff_mean)  ### Modify
		return (UOS - 1) * 100


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
		# VR = pd.Series(np.where(np.isfinite(VR.values), VR.values, np.nan), index=VR.index).ffill().fillna(0)
		VR = pd.Series(np.where(np.isfinite(VR.values), VR.values, np.nan), index=VR.index).ffill().fillna(0) - 1
		if len(VR) < 1:
			return VR
		# result = rolling_cal(VR, rolling_time_m, diff_mean)  Modify
		return VR * 100


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
		result = rolling_cal(RES, rolling_time_n, centralize_std) # Modify
		# result = ((RES / rolling_cal(RES, rolling_time_n, mean_)) - 1) * 100
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
		result = rolling_cal(RES, rolling_time_n, centralize_max_min) # modify
		# result = ((VAR4 / MAVAR4) - 1) * 100
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

		# nominator_ = np.where(tran_Amount == 0, 0, tran_vol)
		# denominator_ = np.where(tran_Amount == 0, 1, tran_Amount)  ## added to get rid of warning
		# res = ta.EMA(nominator_, N) / ta.EMA(denominator_, N) * 10000 ## added to get rid of warning
		res = ta.EMA(tran_Amount.values, N) / ta.EMA(tran_vol.values, N) * 10000 ## comment to get rid of warning
		result = pd.Series(array_clean((price_series.values - res) / res * 100), index=price_series.index)
		# result = rolling_cal(result, rolling_time_n, centralize_max_min)
		return result



class Order_Average_Order(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['order_data'].dropna()
		bs_flag = stk_data['FunctionCode']
		prices = stk_data['Price']

		def func_long(data):

			long_orders = np.where(data == 66)[0]
			if len(long_orders) == 0:
				return len(data)
			elif len(long_orders) == 1:
				return long_orders[0]
			else:
				return np.diff(long_orders).mean()

		def func_short(data):

			short_orders = np.where(data == 83)[0]
			if len(short_orders) == 0:
				return len(data)
			elif len(short_orders) == 1:
				return short_orders[0]
			else:
				return np.diff(short_orders).mean()

		rolling_time = str(nperiod * 3) + 'S'
		bs_not_cancel = bs_flag[prices != 0]
		average_long = bs_not_cancel.rolling(rolling_time,  closed='left').apply(func_long, raw=True)
		average_short = bs_not_cancel.rolling(rolling_time,  closed='left').apply(func_short, raw=True)
		result = average_short - average_long
		return result


class Transaction_Average_Order(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		bs_flag = stk_data['BSFlag']

		def func_long(data):

			long_orders = np.where(data == 66)[0]
			if len(long_orders) == 0:
				return len(data)
			elif len(long_orders) == 1:
				return long_orders[0]
			else:
				return np.diff(long_orders).mean()

		def func_short(data):

			short_orders = np.where(data == 83)[0]
			if len(short_orders) == 0:
				return len(data)
			elif len(short_orders) == 1:
				return short_orders[0]
			else:
				return np.diff(short_orders).mean()

		rolling_time = str(nperiod * 3) + 'S'
		average_long = bs_flag.rolling(rolling_time,  closed='left').apply(func_long, raw=True)
		average_short = bs_flag.rolling(rolling_time,  closed='left').apply(func_short, raw=True)
		result = average_short - average_long
		return result


# class Transaction_CHO(base_feature):
# 	param_list = ['nperiod']

# 	def feature(self, data_fed, nperiod):

# 		stk_data = data_fed['transaction_data'].dropna()
# 		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
# 		vol_series = stk_data['Volume'][stk_data['FunctionCode'] == 48]
# 		if len(price_series) < 1:
# 			return price_series
# 		rolling_time = str(nperiod * 3) + 'S'
# 		rolling_time_n1 = str(nperiod * 10) + 'S'
# 		rolling_time_n2 = str(nperiod * 20) + 'S'
# 		rolling_time_m = str(nperiod * 6) + 'S'

# 		close_series = rolling_cal(price_series, rolling_time, close_)
# 		high_series = rolling_cal(price_series, rolling_time, high_)
# 		low_series = rolling_cal(price_series, rolling_time, low_)
# 		Vol = rolling_cal(vol_series, rolling_time, vol_sum)
# 		Mid_p1 = Vol * (2 * close_series - high_series - low_series) / (high_series + low_series)
# 		Mid = rolling_cal(Mid_p1, rolling_time_n2, vol_sum)
# 		CHO = rolling_cal(Mid, rolling_time_n1, mean_) - rolling_cal(Mid, rolling_time_n2, mean_)
# 		MACHO = rolling_cal(CHO, rolling_time_n2, mean_)
# 		# RES = CHO - MACHO
# 		# result = rolling_cal(RES, rolling_time_m, centralize_max_min) # modify
# 		result = (CHO / MACHO - 1) * 100
# 		return result


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
		# MACHO = rolling_cal(CHO, rolling_time_n2, mean_)
		# RES = CHO - MACHO
		result = rolling_cal(CHO, rolling_time_m, centralize_std) # modify
		# result = (CHO / MACHO - 1) * 100
		return result



class Ask_Bid_1_New(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				max_point = np.max(data)
				min_point = np.min(data)
				result = np.sign(data[-1] - data[0]) * (max_point - min_point) / (max_point + min_point)
				return result if np.isfinite(result) else 0

		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
		rolling_time = str(nperiod * 3) + 'S'
		bid_part_decay = np.sum(stk_data.loc[:, bid_part], axis=1)
		ask_part_decay = np.sum(stk_data.loc[:, ask_part], axis=1)
		new_bid_ask_vol = bid_part_decay - ask_part_decay
		vol_diff = pd.Series(new_bid_ask_vol, index=stk_data.index)
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True).abs()


class Ask_Bid_Sum_Vol_decay(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				return data[np.isfinite(data)].sum()

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


class Order_Direction_Amount_decay(base_feature):
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
		direction_data = -np.sign(selected_data['FunctionCode'] - 70) * selected_data['Amount']
		numerator = direction_data.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)
		denominator = selected_data['Amount'].rolling(rolling_time,  closed='left').apply(function_inside_denominator, raw=True)
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


class Transaction_Net_DIFF(base_feature):
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
		signed_turnover = selected_data['Turnover'] * bs_multiplier
		unsigned_turnover = selected_data['Turnover']
		numerator = signed_turnover.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)
		denominator = unsigned_turnover.rolling(rolling_time,  closed='left').apply(function_inside_denominator, raw=True)
		return numerator / denominator


class Transaction_Order_Percent_Diff(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		order_data = data_fed['order_data']
		transaction_data = data_fed['transaction_data']
		tick_data = data_fed['tick_data']

		def pos_sum_tran(x):
		    sum_pos = x[x > 0].sum()
		    return sum_pos

		def neg_sum_tran(x):
			sum_neg = x[x < 0].sum()
			return -sum_neg

		def pos_sum_order(x):
		    sum_pos = x[x > 0].sum()
		    return sum_pos if sum_pos > 0 else 1e-3

		def neg_sum_order(x):
			sum_neg = x[x < 0].sum()
			return -sum_neg if sum_neg < 0 else 1e-3

		rolling_time = str(nperiod * 3) + 'S'
		standard_series = tick_data["Close"]
		
		selected_order_data = order_data[(order_data['FunctionCode'] == 66) | (order_data['FunctionCode'] == 83)]
		selected_tran_data = transaction_data[(transaction_data['BSFlag'] == 66) | (transaction_data['BSFlag'] == 83)]

		bs_multiplier_order = np.where(selected_order_data['FunctionCode'].values == 66, 1, -1)
		bs_multiplier_tran = np.where(selected_tran_data['BSFlag'].values == 66, 1, -1)

		sign_vol_order = selected_order_data['Volume'] * bs_multiplier_order
		sign_vol_tran = selected_tran_data['Volume'] * bs_multiplier_tran

		buy_tran_vol_pos = sign_vol_tran.rolling(rolling_time, closed='left').apply(pos_sum_tran, raw=True)
		sell_tran_vol_neg = sign_vol_tran.rolling(rolling_time, closed='left').apply(neg_sum_tran, raw=True)
		buy_order_vol_pos = sign_vol_order.rolling(rolling_time, closed='left').apply(pos_sum_order, raw=True)
		sell_order_vol_pos = sign_vol_order.rolling(rolling_time, closed='left').apply(neg_sum_order, raw=True)

		new_index_tran_sum_buy = ~buy_tran_vol_pos.index.duplicated(keep='last')
		new_index_tran_sum_sell = ~sell_tran_vol_neg.index.duplicated(keep='last')
		tran_buy_sum = buy_tran_vol_pos.loc[new_index_tran_sum_buy]
		tran_sell_sum = sell_tran_vol_neg.loc[new_index_tran_sum_sell]

		new_index_ord_sum_buy = ~buy_order_vol_pos.index.duplicated(keep='last')
		new_index_ord_sum_sell = ~sell_order_vol_pos.index.duplicated(keep='last')
		ord_buy_sum = buy_order_vol_pos.loc[new_index_ord_sum_buy]
		ord_sell_sum = sell_order_vol_pos.loc[new_index_ord_sum_sell]

		low_bound = np.max([tran_buy_sum.index[0], ord_buy_sum.index[0], standard_series.index[0]])
		up_bound = np.min([tran_buy_sum.index[-1], ord_buy_sum.index[-1], standard_series.index[-1]])

		target_index = standard_series.index[(standard_series.index > low_bound) & (standard_series.index < up_bound)]

		pos_tran_buy = tran_buy_sum.index.searchsorted(target_index, side='left')
		pos_tran_sell = tran_sell_sum.index.searchsorted(target_index, side='left')

		pos_ord_buy = ord_buy_sum.index.searchsorted(target_index, side='left')
		pos_ord_sell = ord_sell_sum.index.searchsorted(target_index, side='left')

		tran_result_buy = pd.Series(tran_buy_sum.iloc[pos_tran_buy].values, index=target_index)
		tran_result_sell = pd.Series(tran_sell_sum.iloc[pos_tran_sell].values, index=target_index)

		ord_result_buy = pd.Series(ord_buy_sum.iloc[pos_ord_buy].values, index=target_index)
		ord_result_sell = pd.Series(ord_sell_sum.iloc[pos_ord_sell].values, index=target_index)

		return tran_result_buy / ord_result_buy - tran_result_sell / ord_result_sell


class Transaction_Order_Times_Diff(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		order_data = data_fed['order_data']
		transaction_data = data_fed['transaction_data']
		tick_data = data_fed['tick_data']

		def pos_sum_tran(x):
		    sum_pos = x[x > 0].sum()
		    return sum_pos if sum_pos > 0 else 1e-3

		def neg_sum_tran(x):
			sum_neg = x[x < 0].sum()
			return -sum_neg if sum_neg < 0 else 1e-3

		def pos_sum_order(x):
		    sum_pos = x[x > 0].sum()
		    return sum_pos

		def neg_sum_order(x):
			sum_neg = x[x < 0].sum()
			return -sum_neg

		rolling_time = str(nperiod * 3) + 'S'
		standard_series = tick_data["Close"]
		
		selected_order_data = order_data[(order_data['FunctionCode'] == 66) | (order_data['FunctionCode'] == 83)]
		selected_tran_data = transaction_data[(transaction_data['BSFlag'] == 66) | (transaction_data['BSFlag'] == 83)]

		bs_multiplier_order = np.where(selected_order_data['FunctionCode'].values == 66, 1, -1)
		bs_multiplier_tran = np.where(selected_tran_data['BSFlag'].values == 66, 1, -1)

		# sign_vol_order = selected_order_data['Volume'] * bs_multiplier_order
		# sign_vol_tran = selected_tran_data['Volume'] * bs_multiplier_tran

		sign_vol_order = pd.Series(bs_multiplier_order, index=selected_order_data['Volume'].index)
		sign_vol_tran = pd.Series(bs_multiplier_tran, index=selected_tran_data['Volume'].index)

		buy_tran_vol_pos = sign_vol_tran.rolling(rolling_time, closed='left').apply(pos_sum_tran, raw=True)
		sell_tran_vol_neg = sign_vol_tran.rolling(rolling_time, closed='left').apply(neg_sum_tran, raw=True)
		buy_order_vol_pos = sign_vol_order.rolling(rolling_time, closed='left').apply(pos_sum_order, raw=True)
		sell_order_vol_pos = sign_vol_order.rolling(rolling_time, closed='left').apply(neg_sum_order, raw=True)

		new_index_tran_sum_buy = ~buy_tran_vol_pos.index.duplicated(keep='last')
		new_index_tran_sum_sell = ~sell_tran_vol_neg.index.duplicated(keep='last')
		tran_buy_sum = buy_tran_vol_pos.loc[new_index_tran_sum_buy]
		tran_sell_sum = sell_tran_vol_neg.loc[new_index_tran_sum_sell]

		new_index_ord_sum_buy = ~buy_order_vol_pos.index.duplicated(keep='last')
		new_index_ord_sum_sell = ~sell_order_vol_pos.index.duplicated(keep='last')
		ord_buy_sum = buy_order_vol_pos.loc[new_index_ord_sum_buy]
		ord_sell_sum = sell_order_vol_pos.loc[new_index_ord_sum_sell]

		low_bound = np.max([tran_buy_sum.index[0], ord_buy_sum.index[0], standard_series.index[0]])
		up_bound = np.min([tran_buy_sum.index[-1], ord_buy_sum.index[-1], standard_series.index[-1]])

		target_index = standard_series.index[(standard_series.index > low_bound) & (standard_series.index < up_bound)]

		pos_tran_buy = tran_buy_sum.index.searchsorted(target_index, side='left')
		pos_tran_sell = tran_sell_sum.index.searchsorted(target_index, side='left')

		pos_ord_buy = ord_buy_sum.index.searchsorted(target_index, side='left')
		pos_ord_sell = ord_sell_sum.index.searchsorted(target_index, side='left')

		tran_result_buy = pd.Series(tran_buy_sum.iloc[pos_tran_buy].values, index=target_index)
		tran_result_sell = pd.Series(tran_sell_sum.iloc[pos_tran_sell].values, index=target_index)

		ord_result_buy = pd.Series(ord_buy_sum.iloc[pos_ord_buy].values, index=target_index)
		ord_result_sell = pd.Series(ord_sell_sum.iloc[pos_ord_sell].values, index=target_index)

		return tran_result_buy / ord_result_buy - tran_result_sell / ord_result_sell


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





##=======================TEST New Features============================================


class Transaction_UDL(base_feature):
	param_list = ['nperiod']
	def feature(self, data_fed, nperiod):
		stk_data = data_fed['transaction_data'].dropna()
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		rolling_time_n = str(nperiod * 3) + 'S'
		rolling_time_n2 = str(nperiod * 6) + 'S'
		rolling_time_n3 = str(nperiod * 10) + 'S'
		rolling_time_n4 = str(nperiod * 15) + 'S'
		rolling_time_n5 = str(nperiod * 20) + 'S'

		close_series = rolling_cal(price_series, rolling_time_n, close_)

		MA2 = rolling_cal(close_series, rolling_time_n2, mean_)
		MA3 = rolling_cal(close_series, rolling_time_n3, mean_)
		MA4 = rolling_cal(close_series, rolling_time_n4, mean_)
		MA5 = rolling_cal(close_series, rolling_time_n5, mean_)

		UDL = (MA2 + MA3 + MA4 + MA5) / 4
		MAUDL = rolling_cal(UDL, rolling_time_n2, mean_)
		result = ((UDL / MAUDL) - 1) * 100
		return result


class Transaction_Returns(base_feature):
	param_list = ['nperiod']
	def feature(self, data_fed, nperiod):
		stk_data = data_fed['transaction_data'].dropna()
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		rolling_time_n = str(nperiod * 3) + 'S'

		close_series = rolling_cal(price_series, rolling_time_n, close_)
		LC = rolling_cal(price_series, rolling_time_n, lc_)   ## 考虑加入EMA
		result = ((close_series / LC) - 1) * 100
		return result


class Ask_Bid_1_New2(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				result = (data[-1] - np.mean(data)) / np.mean(data)
				return result if np.isfinite(result) else 0

		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
		rolling_time = str(nperiod * 3) + 'S'
		bid_part_decay = np.sum(stk_data.loc[:, bid_part], axis=1)
		ask_part_decay = np.sum(stk_data.loc[:, ask_part], axis=1)
		new_bid_ask_vol = bid_part_decay - ask_part_decay
		vol_diff = pd.Series(new_bid_ask_vol, index=stk_data.index)
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True).abs()


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
		RES = rolling_cal(VPT, rolling_time_m, centralize_std)
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
		RES = (SUM_AMOV / SUM_VOL / Mid_High_low - 1) * 100
		return RES


class Ask_Bid_AMV(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']
		Open = stk_data['Open']
		Close = stk_data['Close']
		Mid = stk_data['Mid']
		Vol = stk_data['TransactionVol']
		if len(Close) < 1:
			return Close

		rolling_time = str(nperiod * 9) + 'S'
		AMOV = Vol * (Open + Close) / 2
		SUM_AMOV = rolling_cal(AMOV, rolling_time, vol_sum)
		SUM_VOL = rolling_cal(Vol, rolling_time, vol_sum)
		RES = (SUM_AMOV / SUM_VOL / Mid - 1) * 100
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


class Transaction_OLD_VR(base_feature):
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


class Transaction_OLD_UOS(base_feature):
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


class Transaction_alpha101_32(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):
		stk_data = data_fed['transaction_data'].dropna()
		rolling_time_n = str(nperiod * 3) + 'S'
		rolling_time_n2 = str(nperiod * 6) + 'S'
		rolling_time_n3 = str(nperiod * 20) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		vol_series = stk_data['Volume'][stk_data['FunctionCode'] == 48]
		if len(price_series) < 1:
			return price_series

		close_series = price_series
		print(rolling_time_n)
		close_series.to_csv('debug.csv')
		high_series = close_series.rolling(rolling_time_n, closed='left').max()
		print('in feature')
		mean_close = close_series.rolling(rolling_time_n2, closed='left').mean()
		p1 = (mean_close - close_series) / mean_close
		close_t_vol = close_series * vol_series
		VOL = vol_series.rolling(rolling_time_n, closed='left').sum()
		VWAP = close_t_vol.rolling(rolling_time_n, closed='left').sum() / VOL
		LC = rolling_cal(price_series, rolling_time_n, lc_)
		roll_corr = rolling_corr(VWAP, LC, rolling_time_n2)
		mean_roll_corr = roll_corr.rolling(rolling_time_n3, closed='left').mean()
		p2 = roll_corr / mean_roll_corr
		result = p1 + p2
		return result