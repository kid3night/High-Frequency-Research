import pandas as pd
import numpy as np
from baseSignal_Modified import base_feature
import talib as ta
from feature_cal_supporting import *


# class Ask_Bid_1_New(base_feature):
# 	param_list = ['nperiod']

# 	def feature(self, data_fed, nperiod):

# 		stk_data = data_fed['tick_data']

# 		def function_inside(data):
# 			if len(data) == 0:
# 				return 0
# 			else:
# 				max_point = np.max(data)
# 				min_point = np.min(data)
# 				result = np.abs(data[-1] - data[0]) / (np.abs(max_point) + np.abs(min_point))
# 				return result if np.isfinite(result) else 0

# 		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
# 					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
# 		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
# 					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
# 		rolling_time = str(nperiod * 3) + 'S'
# 		bid_part_decay = np.sum(stk_data.loc[:, bid_part], axis=1)
# 		ask_part_decay = np.sum(stk_data.loc[:, ask_part], axis=1)
# 		new_bid_ask_vol = bid_part_decay - ask_part_decay
# 		vol_diff = pd.Series(new_bid_ask_vol, index=stk_data.index)
# 		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True).abs()


# class VRSI(base_feature):
# 	param_list = ['nperiod']

# 	def feature(self, data_fed, nperiod):

# 		stk_data = data_fed['tick_data'].dropna()
# 		vol = stk_data['TransactionVol'].astype(float)
# 		Close = stk_data['Close']
# 		if len(vol) < 1:
# 			return vol

# 		n = nperiod
# 		N = nperiod * 3
# 		sign_vol = np.sign(Close.diff(n).fillna(0))
# 		diff_vol = (vol).diff(n).fillna(0).values
# 		num = ta.EMA(np.where(diff_vol != 0, diff_vol, 0), N)
# 		denom = ta.EMA(np.abs(diff_vol), N)
# 		denom = np.where((num == 0), 1, denom)
# 		result = pd.Series(num / denom * sign_vol.values, index=vol.index)
# 		return result


class VRSI_update(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].dropna()
		vol = stk_data['TransactionVol'].astype(float)
		Close = stk_data['Close']
		if len(vol) < 1:
			return vol

		n = nperiod
		N = nperiod * 3
		sign_vol = np.sign(Close.diff(n).fillna(0))
		diff_vol = (vol * sign_vol).diff(n).fillna(0).values
		num = ta.EMA(np.where(diff_vol != 0, diff_vol, 0), N)
		denom = ta.EMA(np.abs(diff_vol), N)
		denom = np.where((num == 0), 1, denom)
		result = pd.Series(num / denom, index=vol.index)
		return result


# class VRSI(base_feature):
# 	param_list = ['nperiod']

# 	def feature(self, data_fed, nperiod):

# 		stk_data = data_fed['tick_data'].dropna()
# 		vol = stk_data['TransactionVol'].astype(float)
# 		Close = stk_data['Close']
# 		if len(vol) < 1:
# 			return vol

# 		n = nperiod
# 		N = nperiod * 3
# 		sign_vol = np.sign(Close.diff(n).fillna(0))
# 		diff_vol = (vol).diff(n).fillna(0).values
# 		num = ta.MA(np.where(diff_vol > 0, diff_vol, 0), N) * sign_vol
# 		denom = ta.MA(np.abs(diff_vol), N)
# 		denom = np.where((num == 0), 1, denom)
# 		result = pd.Series(num / denom, index=vol.index)
# 		return result


class Ask_Bid_New_update(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				max_point = np.max(data)
				min_point = np.min(data)
				result = (data[-1] - data[0]) / (np.abs(max_point) + np.abs(min_point))
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
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)


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

		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
		rolling_time = str(nperiod * 3) + 'S'

		order_vols_bid = stk_data.loc[:, bid_part].sum(axis=1)
		order_vols_ask = stk_data.loc[:, ask_part].sum(axis=1)
		bid_change_perc = order_vols_bid.rolling(rolling_time,  closed='left').apply(pct_change_, raw=True).values
		ask_change_perc = order_vols_ask.rolling(rolling_time,  closed='left').apply(pct_change_, raw=True).values
		mid_change_perc = stk_data['Mid'].rolling(rolling_time,  closed='left').apply(pct_change_, raw=True).values
		perc_diff_rate = (bid_change_perc - ask_change_perc) * 100
		res = np.where((mid_change_perc > 0) & (perc_diff_rate > 0), mid_change_perc * mid_change_perc, np.where((mid_change_perc < 0) & (perc_diff_rate < 0), -(mid_change_perc * mid_change_perc), 0))
		result = pd.Series(res, index=order_vols_bid.index)
		result = series_clean(result)
		return result
