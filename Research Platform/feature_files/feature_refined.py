import pandas as pd
import numpy as np
from baseSignal_Modified import base_feature
import talib as ta


def decay_within_ticks(leng, decay_factor):
	weight = np.array([decay_factor ** i if decay_factor ** i > 0.1 else 0.1 for i in range(leng - 1, -1, -1)])
	return weight


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
		print(stk_data)

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
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)


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


class Transaction_Cancellation_decay(base_feature):
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
		cancel_data = stk_data[stk_data['FunctionCode'] == 67]
		sign_cancel = np.sign(cancel_data['BidOrder'] - cancel_data['AskOrder']) * cancel_data['Volume']
		numerator = sign_cancel.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)
		denominator = cancel_data['Volume'].rolling(rolling_time,  closed='left').apply(function_inside_denominator, raw=True)
		return numerator / denominator


class Tran_Price_Change_Points(base_feature):
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
		selected_data = stk_data[stk_data['Price'] != 0]
		price_diff = selected_data['Price'].diff(1).fillna(0)
		price_change_pos = price_diff != 0
		price_change_part_vol = selected_data['Volume'][price_change_pos]
		rolling_series = price_change_part_vol * price_diff[price_change_pos]
		numerator = rolling_series.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)
		denominator = selected_data['Volume'].rolling(rolling_time,  closed='left').apply(function_inside_denominator, raw=True)
		return numerator / denominator



# class Tran_Price_Change_Vol_decay(base_feature):
# 	param_list = ['nperiod', 'decay_factor']

# 	def feature(self, data_fed, nperiod, decay_factor):

# 		stk_data = data_fed['transaction_data']

# 		def function_inside1(data):
# 			if len(data) == 0:
# 				return 0
# 			else:
# 				return np.sign(data[-1] - data[0])

# 		def function_inside2(data, decay_factor):
# 			if len(data) == 0:
# 				return 0
# 			else:
# 				decay_weight = decay_within_ticks(len(data), decay_factor)
# 				net_sum = (data * decay_weight).sum()
# 				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

# 		rolling_time = str(nperiod * 3) + 'S'
# 		selected_data = stk_data[stk_data['Price'] != 0]
# 		price_change_sign = selected_data['Price'].rolling(rolling_time,  closed='left').apply(function_inside1, raw=True)
# 		vol_sum_decay = selected_data['Volume'].rolling(rolling_time,  closed='left').apply(function_inside2, args=(decay_factor,), raw=True)
# 		return vol_sum_decay * price_change_sign





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


# class tran_price_diff_percent_change(base_feature):
# 	param_list = ['nperiod']

# 	def feature(self, data_fed, nperiod):

# 		stk_data = data_fed['transaction_data']

# 		def function_inside(data, decay_factor):
# 			if len(data) == 0:
# 				return 0
# 			else:
# 				decay_weight = decay_within_ticks(len(data), decay_factor)
# 				net_sum = (data * decay_weight).sum()
# 				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

# 		rolling_time = str(nperiod * 3) + 'S'
# 		selected_data = stk_data[stk_data['FunctionCode'] != 67]
# 		bs_multiplier = np.where(selected_data['BSFlag'].values == 66, 1, -1)
# 		signed_price = selected_data * bs_multiplier
# 		rolling_series = price_change_part_vol * selected_data['Price']
# 		return rolling_series.rolling(rolling_time,  closed='left').apply(function_inside, args=(decay_factor,), raw=True)


class Transaction_RSI(base_feature):  ##  基本数值离散0, 1
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
		result = price_series.rolling(rolling_time,  closed='left').apply(func_info, raw=True)
		# result = pd.Series(ta.EMA(raw_series.values, int(nperiod * 3)), index=raw_series.index)
		return result

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
		res = price_series.rolling(rolling_time,  closed='left').apply(func_info, raw=True)
		if len(res) < 1:
			return res

		result = pd.Series(ta.EMA(res.values, int(nperiod * 3)), index=res.index)
		return result


class Transaction_RSV(base_feature):##  基本数值离散(multiple values)
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()

		def func_info(data):
			if len(data) == 0:
				return 0
			close = data[-1]
			high = np.max(data)
			low = np.min(data)
			numerator = close - low
			denominator = (high - low) if (high - low) != 0 else 1
			return numerator / denominator

		rolling_time = str(nperiod * 3) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		result = price_series.rolling(rolling_time,  closed='left').apply(func_info, raw=True)
		return result


class Transaction_KDJ(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()

		def func_info(data):
			if len(data) == 0:
				return 0
			close = data[-1]
			high = np.max(data)
			low = np.min(data)
			numerator = close - low
			denominator = (high - low) if (high - low) != 0 else 1
			return numerator / denominator

		rolling_time = str(nperiod * 3) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		RSV = price_series.rolling(rolling_time,  closed='left').apply(func_info, raw=True).ffill().fillna(0)
		if len(RSV) < 1:
			return RSV
		K = ta.EMA(RSV.values, int(nperiod * 3))
		D = ta.EMA(K, int(nperiod * 3))
		J = 3 * K - 2 * D
		return pd.Series(J, index=RSV.index)


class Transaction_CHO(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()

		def func_info(data):
			if len(data) == 0:
				return 0
			close = data[-1]
			high = np.max(data)
			low = np.min(data)
			numerator = 2 * close - low - high
			denominator = high + low
			return numerator / denominator

		def vol_sum(data):
			if len(data) == 0:
				return 0
			return data.sum()

		rolling_time = str(nperiod * 3) + 'S'
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		vol = stk_data['Volume'][stk_data['FunctionCode'] == 48]
		MID_p1 = price_series.rolling(rolling_time,  closed='left').apply(func_info, raw=True).ffill().fillna(0)
		MID_p2 = vol.rolling(rolling_time,  closed='left').apply(vol_sum, raw=True).ffill().fillna(0)
		if len(MID_p1) < 1:
			return MID_p1
		if len(MID_p2) < 1:
			return MID_p2
		MID = np.cumsum(MID_p1 * MID_p2 / 1000000).values
		CHO = ta.MA(MID, int(nperiod * 3)) - ta.MA(MID, int(nperiod * 6))
		DIF = CHO - ta.MA(CHO, int(nperiod * 3))
		result = pd.Series(DIF, index=MID_p1.index)
		return result


class Ask_Bid_MACD(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].ffill().bfill()
		mid = stk_data['Mid'].values.astype(float)
		if len(mid) < 1:
			return stk_data['Mid']
		print_start = mid[0]
		short_n = nperiod * 2
		long_n = nperiod * 4
		mid_n = nperiod
		macd = ta.MACD(mid, fastperiod=short_n, slowperiod=long_n, signalperiod=mid_n)[0] / print_start
		result = pd.Series(macd, index=stk_data['Mid'].index)
		return result


class Transaction_EMV(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		stk_data = stk_data[stk_data['FunctionCode'] == 48]
		def func_info1(data):
			if len(data) == 0:
				return 0
			return np.max(data) - np.min(data)

		def func_info2(data):
			if len(data) == 0:
				return 0
			return np.max(data) + np.min(data)

		def func_sum(data):
			return data.sum()

		rolling_time = str(nperiod * 3) + 'S'
		N = nperiod * 3
		M = nperiod * 6
		Vol = stk_data['Volume'].rolling(rolling_time,  closed='left').apply(func_sum, raw=True).ffill().fillna(0)
		if len(Vol) < 1:
			return Vol
		Volume_temp = ta.MA(Vol.values, N) / Vol.values
		Volume = pd.Series(np.where(np.isfinite(Volume_temp), Volume_temp, np.nan), index=Vol.index).ffill().fillna(0)
		high_low_add = stk_data['Price'].rolling(rolling_time,  closed='left').apply(func_info2, raw=True).ffill().fillna(0)
		high_low_minus = stk_data['Price'].rolling(rolling_time,  closed='left').apply(func_info1, raw=True).ffill().fillna(0)
		if len(high_low_minus) < 1:
			return high_low_minus
		high_low_minus_MA = pd.Series(ta.MA(high_low_minus.values, N), index=high_low_minus.index)

		Mid = high_low_add.pct_change()
		Mid[~np.isfinite(Mid)] = np.nan
		Mid = Mid.ffill().fillna(0)
		EMV_inside = Mid * Volume * high_low_minus / high_low_minus_MA
		EMV_inside[~np.isfinite(EMV_inside)] = np.nan
		EMV_inside = EMV_inside.ffill().fillna(0)
		if len(EMV_inside) < 1:
			return EMV_inside
		EMV = pd.Series(ta.MA(EMV_inside.values, N), index=EMV_inside.index)
		return EMV


class Transaction_MAEMV(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data'].dropna()
		stk_data = stk_data[stk_data['FunctionCode'] == 48]
		def func_info1(data):
			if len(data) == 0:
				return 0
			return np.max(data) - np.min(data)

		def func_info2(data):
			if len(data) == 0:
				return 0
			return np.max(data) + np.min(data)

		def func_sum(data):
			return data.sum()

		rolling_time = str(nperiod * 3) + 'S'
		N = nperiod * 3
		M = nperiod * 6
		Vol = stk_data['Volume'].rolling(rolling_time,  closed='left').apply(func_sum, raw=True).ffill().fillna(0)
		if len(Vol) < 1:
			return Vol
		Volume_temp = ta.MA(Vol.values, N) / Vol.values
		Volume = pd.Series(np.where(np.isfinite(Volume_temp), Volume_temp, np.nan), index=Vol.index).ffill().fillna(0)
		high_low_add = stk_data['Price'].rolling(rolling_time,  closed='left').apply(func_info2, raw=True).ffill().fillna(0)
		high_low_minus = stk_data['Price'].rolling(rolling_time,  closed='left').apply(func_info1, raw=True).ffill().fillna(0)
		if len(high_low_minus) < 1:
			return high_low_minus
		high_low_minus_MA = pd.Series(ta.MA(high_low_minus.values, N), index=high_low_minus.index)

		Mid = high_low_add.pct_change()
		Mid[~np.isfinite(Mid)] = np.nan
		Mid = Mid.ffill().fillna(0)
		EMV_inside = Mid * Volume * high_low_minus / high_low_minus_MA
		EMV_inside[~np.isfinite(EMV_inside)] = np.nan
		EMV_inside = EMV_inside.ffill().fillna(0)
		if len(EMV_inside) < 1:
			return EMV_inside
		MAEMV = pd.Series(ta.MA(ta.MA(EMV_inside.values, N), M), index=EMV_inside.index)
		return MAEMV


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


class Ask_Bid_VMACD(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data'].ffill().bfill()
		mid = stk_data['TransactionVol'].values.astype(float)
		if len(mid):
			return stk_data['TransactionVol']
		print_start = mid[0]
		short_n = nperiod * 3
		long_n = nperiod * 6
		mid_n = nperiod * 2
		macd = ta.MACD(mid, fastperiod=short_n, slowperiod=long_n, signalperiod=mid_n)[0] / print_start
		result = pd.Series(macd, index=stk_data['TransactionVol'].index)
		return result


class Transaction_DMI(base_feature):
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

		def sum_(data):
			return np.sum(data)

		def max_(Series1, Series2):
			if len(Series1) < 1:
				return Series1
			concated_series = pd.concat([Series1, Series2], axis=1)
			return concated_series.max(axis=1)

		def iF(condition, value_1, value_2, index_series): 
			# first three parameters are numpy arrays
			if len(condition) < 1:
				return pd.Series(index=[])
			res = pd.Series(np.where(condition, value_1, value_2), index=index_series)
			return res

		rolling_time = str(nperiod * 3) + 'S'
		N = nperiod * 18
		rolling_time_n = str(nperiod * 18) + 'S'
		M = nperiod * 9
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		high_series = price_series.rolling(rolling_time,  closed='left').apply(high_, raw=True)
		low_series = price_series.rolling(rolling_time,  closed='left').apply(low_, raw=True)
		close_series = price_series.rolling(rolling_time,  closed='left').apply(close_, raw=True)

		high_low_minus = high_series - low_series
		high_minus_ref_close = (high_series - close_series.shift(1)).abs()
		ref_close_minus_low = (close_series.shift(1) - low_series).abs()
		MTR_temp = max_(max_(high_low_minus, high_minus_ref_close), ref_close_minus_low)
		MTR = MTR_temp.rolling(rolling_time_n, closed='left').apply(sum_, raw=True)
		HD = high_series.diff(1)
		LD = -low_series.diff(1)
		DMP_temp = iF(((HD > 0) & (HD > LD)).values, HD.values, 0, HD.index)
		DMM_temp = iF(((LD > 0) & (LD > HD)).values, LD.values, 0, LD.index)
		if len(DMP_temp) < 1:
			return DMP_temp
		DMP = DMP_temp.rolling(rolling_time_n, closed='left').apply(sum_, raw=True)
		DMM = DMM_temp.rolling(rolling_time_n, closed='left').apply(sum_, raw=True)
		PDI = DMP * 100 / MTR
		MDI = DMM * 100 / MTR
		ADX_ = (MDI - PDI / (MDI + PDI)).ffill().fillna(0)
		if len(ADX_) < 1:
			return ADX_
		result = pd.Series(ta.MA(ADX_.values, M), index=ADX_.index)
		return result


class Transaction_DMI_abs(base_feature):
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

		def sum_(data):
			return np.sum(data)

		def max_(Series1, Series2):
			if len(Series1) < 1:
				return Series1
			concated_series = pd.concat([Series1, Series2], axis=1)
			return concated_series.max(axis=1)

		def iF(condition, value_1, value_2, index_series): 
			# first three parameters are numpy arrays
			if len(condition) < 1:
				return pd.Series(index=[])
			res = pd.Series(np.where(condition, value_1, value_2), index=index_series)
			return res

		rolling_time = str(nperiod * 3) + 'S'
		N = nperiod * 18
		rolling_time_n = str(nperiod * 18) + 'S'
		M = nperiod * 9
		price_series = stk_data['Price'][stk_data['FunctionCode'] == 48]
		high_series = price_series.rolling(rolling_time,  closed='left').apply(high_, raw=True)
		low_series = price_series.rolling(rolling_time,  closed='left').apply(low_, raw=True)
		close_series = price_series.rolling(rolling_time,  closed='left').apply(close_, raw=True)

		high_low_minus = high_series - low_series
		high_minus_ref_close = (high_series - close_series.shift(1)).abs()
		ref_close_minus_low = (close_series.shift(1) - low_series).abs()
		MTR_temp = max_(max_(high_low_minus, high_minus_ref_close), ref_close_minus_low)
		MTR = MTR_temp.rolling(rolling_time_n, closed='left').apply(sum_, raw=True)
		HD = high_series.diff(1)
		LD = -low_series.diff(1)
		DMP_temp = iF(((HD > 0) & (HD > LD)).values, HD.values, 0, HD.index)
		DMM_temp = iF(((LD > 0) & (LD > HD)).values, LD.values, 0, LD.index)
		DMP = DMP_temp.rolling(rolling_time_n, closed='left').apply(sum_, raw=True)
		DMM = DMM_temp.rolling(rolling_time_n, closed='left').apply(sum_, raw=True)
		PDI = DMP * 100 / MTR
		MDI = DMM * 100 / MTR
		ADX_ = ((MDI - PDI_).abs() / (MDI + PDI)).ffill().fillna(0)
		if len(ADX_) < 1:
			return ADX_
		result = pd.Series(ta.MA(ADX_.values, M), index=ADX_.index)
		return result


class SH_feature(base_feature):
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
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)


class SZ_feature(base_feature):
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
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)