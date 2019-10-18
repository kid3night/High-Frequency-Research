import pandas as pd
import numpy as np
from baseSignal_Modified import base_feature
# import feature_auxiliary as fa


# class Ask_Bid_1(base_feature):
# 	param_list = ['nperiod']

# 	def feature(self, data_fed, nperiod):

# 		stk_data = data_fed['tick_data']

# 		def function_inside(data):
# 			if len(data) == 0:
# 				return 0
# 			else:
# 				return (data[-1] - data[0]) / len(data)

# 		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
# 					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
# 		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
# 					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
# 		rolling_time = str(nperiod * 3) + 'S'
# 		vol_diff = stk_data.loc[:, bid_part].sum(axis=1) - stk_data.loc[:, ask_part].sum(axis=1)
# 		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)

def decay_within_tickers(leng, decay_factor):
	weight = np.array([decay_factor ** i if decay_factor ** i > 0.1 else 0.1 for i in range(leng - 1, -1, -1)])
	return weight

class Ask_Bid_1(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				result = (data[-1] - data[0]) / len(data)
				return np.log(result) if result > 0 else -np.log(-result) if result < 0 else 0

		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
		rolling_time = str(nperiod * 3) + 'S'
		vol_diff = stk_data.loc[:, bid_part].sum(axis=1) - stk_data.loc[:, ask_part].sum(axis=1)
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)


# class Transaction_1(base_feature):
# 	param_list = ['nperiod']

# 	def feature(self, data_fed, nperiod):

# 		stk_data = data_fed['transaction_data']

# 		def function_inside(data):
# 			if len(data) == 0:
# 				return 0
# 			else:
# 				return data.sum()

# 		rolling_time = str(nperiod * 3) + 'S'
# 		bs_multiplier = np.where(stk_data['BSFlag'].values == 66, 1, np.where(stk_data['BSFlag'].values == 83, -1, 0))
# 		signed_turnover = stk_data['Turnover'] * bs_multiplier
# 		return signed_turnover.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)


class Transaction_1_re(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		bs_multiplier = np.where(stk_data['BSFlag'].values == 66, 1, np.where(stk_data['BSFlag'].values == 83, -1, 0))
		signed_turnover = stk_data['Turnover'] * bs_multiplier
		return signed_turnover.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)


class Transaction_1_decay(base_feature):
	param_list = ['nperiod', 'decay_factor']

	def feature(self, data_fed, nperiod, decay_factor):

		stk_data = data_fed['transaction_data']

		def function_inside(data, decay_factor):
			if len(data) == 0:
				return 0
			else:
				decay_weight = decay_within_tickers(len(data), decay_factor)
				net_sum = (data * decay_weight).sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[(stk_data['BSFlag'] == 66) | (stk_data['BSFlag'] == 83)]
		bs_multiplier = np.where(selected_data['BSFlag'].values == 66, 1, -1)
		signed_turnover = selected_data['Turnover'] * bs_multiplier
		return signed_turnover.rolling(rolling_time,  closed='left').apply(function_inside, args=(decay_factor,), raw=True)


class Transaction_Cancellation_re(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		cancel_data = stk_data[stk_data['FunctionCode'] == 67]
		sign_cancel = np.sign(cancel_data['AskOrder'] - cancel_data['BidOrder']) * cancel_data['Volume']
		return sign_cancel.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)


class Transaction_Cancellation_Square(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = (data.sum()) ** 2
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		cancel_data = stk_data[stk_data['FunctionCode'] == 67]
		sign_cancel = np.sign(cancel_data['AskOrder'] - cancel_data['BidOrder']) * cancel_data['Volume']
		return (sign_cancel.rolling(rolling_time,  closed='left').apply(function_inside, raw=True))


class Ask_Bid_Vol_Diff_All_Mean(base_feature):
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
		vol_diff = (stk_data.loc[:, bid_part].sum(axis=1) - stk_data.loc[:, ask_part].sum(axis=1)).diff(1).dropna()
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)



class Ask_Bid_Vol_Diff_All_Mean_Rate(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				result = np.nanmean(data)
				return result

		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
		rolling_time = str(nperiod * 3) + 'S'
		vol_diff = (stk_data.loc[:, bid_part].sum(axis=1) - stk_data.loc[:, ask_part].sum(axis=1)).pct_change(1)
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True).fillna(0)


class Ask_Bid_Vol_Diff_All_Mean_Rate_Square(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				result = (np.nanmean(data)) ** 2 * 1e6
				return result

		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
		rolling_time = str(nperiod * 3) + 'S'
		vol_diff = (stk_data.loc[:, bid_part].sum(axis=1) - stk_data.loc[:, ask_part].sum(axis=1)).pct_change(1)
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True).fillna(0)



class Order_Direction_Amount(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['order_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[(stk_data['FunctionCode'] == 66) | (stk_data['FunctionCode'] == 83)]
		direction_data = np.sign(selected_data['FunctionCode'] - 70) * selected_data['Amount']
		return direction_data.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)


class Order_Direction_Volume(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['order_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[(stk_data['FunctionCode'] == 66) | (stk_data['FunctionCode'] == 83)]
		direction_data = np.sign(selected_data['FunctionCode'] - 70) * selected_data['Volume']
		return direction_data.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)


class Ask_Bid_Mean_Sum(base_feature):
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
		vol_diff = stk_data.loc[:, bid_part].sum(axis=1) - stk_data.loc[:, ask_part].sum(axis=1)
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)


class Ask_Bid_Mean_Sum(base_feature):
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
		vol_diff = stk_data.loc[:, bid_part].sum(axis=1) - stk_data.loc[:, ask_part].sum(axis=1)
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)


class Ask_Bid_Volume_Price_Multiple(base_feature):
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
		bid_prices = ['BidPrice10', 'BidPrice9', 'BidPrice8', 'BidPrice7', 'BidPrice6',
       				  'BidPrice5', 'BidPrice4', 'BidPrice3', 'BidPrice2', 'BidPrice1']
		ask_prices = ["AskPrice1","AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5",
					  "AskPrice6","AskPrice7", "AskPrice8", "AskPrice9", "AskPrice10"]
		rolling_time = str(nperiod * 3) + 'S'
		vol_diff = (stk_data.loc[:, bid_part] * stk_data.loc[:, bid_prices]).sum(axis=1) - (stk_data.loc[:, ask_part] * stk_data.loc[:, bid_prices]).sum(axis=1)
		vf_ = vol_diff.loc[np.isfinite(vol_diff)]
		return vf_.rolling(rolling_time, closed='left').apply(function_inside, raw=True)


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
		mco_ = mid_change_origin.loc[np.isfinite(mid_change_origin)]
		return mco_.rolling(rolling_time, closed='left').apply(function_inside, raw=True)


class Transaction_Order_Percent(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		order_data = data_fed['order_data']
		transaction_data = data_fed['transaction_data']
		tick_data = data_fed['tick_data']

		def function_inside1(data):
			if len(data) == 0:
				return 0
			else:
				result = np.mean(data)
				return np.log(result) if result > 0 else -np.log(-result) if result < 0 else 0

		def function_inside2(data):
			if len(data) == 0:
				return 1
			else:
				result = np.mean(data)
				return np.log(result) if result > 0 else -np.log(-result) if result < 0 else 1

		rolling_time = str(nperiod * 3) + 'S'
		standard_series = tick_data["Close"]
		ord_sum = order_data['Amount'].rolling(rolling_time, closed='left').apply(function_inside2, raw=True)
		tran_sum = transaction_data['Turnover'].rolling(rolling_time, closed='left').apply(function_inside1, raw=True)
		low_bound = np.max([tran_sum.index[0], ord_sum.index[0], tick_data.index[0]])
		up_bound = np.min([tran_sum.index[-1], ord_sum.index[-1], tick_data.index[-1]])
		target_index = tick_data.index[(tick_data.index > low_bound) & (tick_data.index < up_bound)]
		new_index_tran_sum = ~tran_sum.index.duplicated(keep='last')
		new_index_ord_sum = ~ord_sum.index.duplicated(keep='last')
		new_tran_sum = tran_sum.iloc[new_index_tran_sum]
		new_order_sum = ord_sum.iloc[new_index_ord_sum]
		pos_tran = new_tran_sum.index.searchsorted(target_index, side='left')
		pos_order = new_order_sum.index.searchsorted(target_index, side='left')
		tran_result = pd.Series(new_tran_sum.iloc[pos_tran].values, index=target_index)
		order_result = pd.Series(new_order_sum.iloc[pos_order].values, index=target_index)
		divide_order_tran = tran_result / order_result
		return divide_order_tran


class Transaction_1_Modify(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data']
		tick_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		bs_multiplier = np.where(stk_data['BSFlag'].values == 66, 1, np.where(stk_data['BSFlag'].values == 83, -1, 0))
		signed_turnover = stk_data['Turnover'] * bs_multiplier
		rolling_result = signed_turnover.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)
		standard_series = tick_data["Close"]
		low_bound = np.max([rolling_result.index[0], tick_data.index[0]])
		up_bound = np.min([rolling_result.index[-1], tick_data.index[-1]])
		target_index = tick_data.index[(tick_data.index > low_bound) & (tick_data.index < up_bound)]
		new_index_rolling_result = ~rolling_result.index.duplicated(keep='last')
		no_duplicated = rolling_result.iloc[new_index_rolling_result]
		pos_new_result = no_duplicated.index.searchsorted(target_index, side='left')
		result = pd.Series(no_duplicated.iloc[pos_new_result].values, index=target_index)
		return result


class Transaction_Cancellation_Modify(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data']
		tick_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0
        
		rolling_time = str(nperiod * 3) + 'S'
		bs_multiplier = np.where(stk_data['FunctionCode'].values == 67, 1, 0)
		sign_cancel = np.sign(stk_data['AskOrder'] - stk_data['BidOrder']) * stk_data['Volume'] * bs_multiplier
		rolling_result = sign_cancel.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)
		standard_series = tick_data["Close"]
		low_bound = np.max([rolling_result.index[0], standard_series.index[0]])
		up_bound = np.min([rolling_result.index[-1], standard_series.index[-1]])
		target_index = standard_series.index[(standard_series.index > low_bound) & (standard_series.index < up_bound)]
		new_index_rolling_result = ~rolling_result.index.duplicated(keep='last')
		no_duplicated = rolling_result.iloc[new_index_rolling_result]
		pos_new_result = no_duplicated.index.searchsorted(target_index, side='left')
		result = pd.Series(no_duplicated.iloc[pos_new_result].values, index=target_index)
		return result


class Order_Direction_Amount_Modify_1(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['order_data']
		tick_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		# bs_multiplier = np.where(stk_data['FunctionCode'].values == 66, 1, np.where(stk_data['FunctionCode'].values == 83, -1, 0))
		# signed_turnover = stk_data['Amount'] * bs_multiplier
		valid_points = stk_data[(stk_data['FunctionCode'].values == 66) | (stk_data['FunctionCode'].values == 83)]
		bs_multiplier = np.where(valid_points['FunctionCode'].values == 66, 1, np.where(valid_points['FunctionCode'].values == 83, -1, 0))
		signed_turnover = valid_points['Amount'] * bs_multiplier

		rolling_result = signed_turnover.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)
		standard_series = tick_data["Close"]
		low_bound = np.max([rolling_result.index[0], standard_series.index[0]])
		up_bound = np.min([rolling_result.index[-1], standard_series.index[-1]])
		target_index = standard_series.index[(standard_series.index > low_bound) & (standard_series.index < up_bound)]
		new_index_rolling_result = ~rolling_result.index.duplicated(keep='last')
		no_duplicated = rolling_result.iloc[new_index_rolling_result]
		pos_new_result = no_duplicated.index.searchsorted(target_index, side='left')
		result = pd.Series(no_duplicated.iloc[pos_new_result].values, index=target_index)
		return result


class Order_Direction_Volume_Modify(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['order_data']
		tick_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				net_sum = data.sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		# bs_multiplier = np.where(stk_data['FunctionCode'].values == 66, 1, np.where(stk_data['FunctionCode'].values == 83, -1, 0))
		valid_points = stk_data[(stk_data['FunctionCode'].values == 66) | (stk_data['FunctionCode'].values == 83)]
		bs_multiplier = np.where(valid_points['FunctionCode'].values == 66, 1, np.where(valid_points['FunctionCode'].values == 83, -1, 0))
		signed_turnover = valid_points['Volume'] * bs_multiplier
		rolling_result = signed_turnover.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)
		standard_series = tick_data["Close"]
		low_bound = np.max([rolling_result.index[0], standard_series.index[0]])
		up_bound = np.min([rolling_result.index[-1], standard_series.index[-1]])
		target_index = standard_series.index[(standard_series.index > low_bound) & (standard_series.index < up_bound)]
		new_index_rolling_result = ~rolling_result.index.duplicated(keep='last')
		no_duplicated = rolling_result.iloc[new_index_rolling_result]
		pos_new_result = no_duplicated.index.searchsorted(target_index, side='left')
		result = pd.Series(no_duplicated.iloc[pos_new_result].values, index=target_index)
		return result


class Ask_Bid_1_decay(base_feature):
	param_list = ['nperiod', 'decay_factor']

	def feature(self, data_fed, nperiod, decay_factor):

		stk_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				result = (data[-1] - data[0]) / len(data)
				return np.log(result) if result > 0 else -np.log(-result) if result < 0 else 0

		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
		rolling_time = str(nperiod * 3) + 'S'
		# bid_ask_decay = np.concatenate((decay_within_tickers(10, decay_factor), -decay_within_tickers(10, decay_factor)[::-1]))
		bid_part_decay = np.average(stk_data.loc[:, bid_part], weights=decay_within_tickers(10, decay_factor), axis=1)
		ask_part_decay = np.average(stk_data.loc[:, ask_part], weights=decay_within_tickers(10, decay_factor)[::-1], axis=1)
		vol_diff = pd.Series(bid_part_decay - ask_part_decay, index=stk_data.index)
		# vol_diff = pd.Series(np.average(stk_data, weights=bid_ask_decay, axis=1), index=stk_data.index)
		# vol_diff = stk_data.loc[:, bid_part].sum(axis=1) - stk_data.loc[:, ask_part].sum(axis=1)
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)
