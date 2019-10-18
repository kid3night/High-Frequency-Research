import pandas as pd
import numpy as np
from baseSignal_Modified import base_feature


def decay_within_ticks(leng, decay_factor):
	weight = np.array([decay_factor ** i if decay_factor ** i > 0.1 else 0.1 for i in range(leng - 1, -1, -1)])
	return weight


class Transaction_1_decay(base_feature):
	param_list = ['nperiod', 'decay_factor']

	def feature(self, data_fed, nperiod, decay_factor):

		stk_data = data_fed['transaction_data']

		def function_inside(data, decay_factor):
			if len(data) == 0:
				return 0
			else:
				decay_weight = decay_within_ticks(len(data), decay_factor)
				net_sum = (data * decay_weight).sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		def function_inside_denominator(data, decay_factor):
			if len(data) == 0:
				return 1
			else:
				decay_weight = decay_within_ticks(len(data), decay_factor)
				net_sum = (data * decay_weight).sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 1

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[(stk_data['BSFlag'] == 66) | (stk_data['BSFlag'] == 83)]
		bs_multiplier = np.where(selected_data['BSFlag'].values == 66, 1, -1)
		signed_turnover = selected_data['Turnover'] * bs_multiplier
		unsigned_turnover = selected_data['Turnover']
		numerator = signed_turnover.rolling(rolling_time,  closed='left').apply(function_inside, args=(decay_factor,), raw=True)
		denominator = unsigned_turnover.rolling(rolling_time,  closed='left').apply(function_inside_denominator, args=(decay_factor,), raw=True)
		return numerator / denominator


class Transaction_1_decay_Volume(base_feature):
	param_list = ['nperiod', 'decay_factor']

	def feature(self, data_fed, nperiod, decay_factor):

		stk_data = data_fed['transaction_data']

		def function_inside(data, decay_factor):
			if len(data) == 0:
				return 0
			else:
				decay_weight = decay_within_ticks(len(data), decay_factor)
				net_sum = (data * decay_weight).sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[(stk_data['BSFlag'] == 66) | (stk_data['BSFlag'] == 83)]
		bs_multiplier = np.where(selected_data['BSFlag'].values == 66, 1, -1)
		signed_turnover = selected_data['Volume'] * bs_multiplier
		return signed_turnover.rolling(rolling_time,  closed='left').apply(function_inside, args=(decay_factor,), raw=True)


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
		bid_part_decay = np.average(stk_data.loc[:, bid_part], weights=decay_within_ticks(10, decay_factor), axis=1)
		ask_part_decay = np.average(stk_data.loc[:, ask_part], weights=decay_within_ticks(10, decay_factor)[::-1], axis=1)
		vol_diff = pd.Series(bid_part_decay - ask_part_decay, index=stk_data.index)
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, raw=True)


class Ask_Bid_Sum_Vol_decay(base_feature):
	param_list = ['nperiod', 'decay_factor']

	def feature(self, data_fed, nperiod, decay_factor):

		stk_data = data_fed['tick_data']

		def function_inside(data, decay_factor):
			if len(data) == 0:
				return 0
			else:
				decay_weight = decay_within_ticks(len(data), decay_factor)
				net_sum = (data * decay_weight).sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
		rolling_time = str(nperiod * 3) + 'S'
		bid_part_decay = np.average(stk_data.loc[:, bid_part], weights=decay_within_ticks(10, decay_factor), axis=1)
		ask_part_decay = np.average(stk_data.loc[:, ask_part], weights=decay_within_ticks(10, decay_factor)[::-1], axis=1)
		vol_diff = pd.Series(bid_part_decay - ask_part_decay, index=stk_data.index)
		return vol_diff.rolling(rolling_time, closed='left').apply(function_inside, args=(decay_factor,), raw=True)


class Order_Direction_Volume_decay(base_feature):
	param_list = ['nperiod', 'decay_factor']

	def feature(self, data_fed, nperiod, decay_factor):

		stk_data = data_fed['order_data']

		def function_inside(data, decay_factor):
			if len(data) == 0:
				return 0
			else:
				decay_weight = decay_within_ticks(len(data), decay_factor)
				net_sum = (data * decay_weight).sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[(stk_data['FunctionCode'] == 66) | (stk_data['FunctionCode'] == 83)]
		direction_data = np.sign(selected_data['FunctionCode'] - 70) * selected_data['Volume']
		return direction_data.rolling(rolling_time,  closed='left').apply(function_inside, args=(decay_factor,), raw=True)


class Order_Direction_Amount_decay(base_feature):
	param_list = ['nperiod', 'decay_factor']

	def feature(self, data_fed, nperiod, decay_factor):

		stk_data = data_fed['order_data']

		def function_inside(data, decay_factor):
			if len(data) == 0:
				return 0
			else:
				decay_weight = decay_within_ticks(len(data), decay_factor)
				net_sum = (data * decay_weight).sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[(stk_data['FunctionCode'] == 66) | (stk_data['FunctionCode'] == 83)]
		direction_data = np.sign(selected_data['FunctionCode'] - 70) * selected_data['Amount']
		return direction_data.rolling(rolling_time,  closed='left').apply(function_inside, args=(decay_factor,), raw=True)


class Transaction_Cancellation_decay(base_feature):
	param_list = ['nperiod', 'decay_factor']

	def feature(self, data_fed, nperiod, decay_factor):

		stk_data = data_fed['transaction_data']

		def function_inside(data, decay_factor):
			if len(data) == 0:
				return 0
			else:
				decay_weight = decay_within_ticks(len(data), decay_factor)
				net_sum = (data * decay_weight).sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		cancel_data = stk_data[stk_data['FunctionCode'] == 67]
		sign_cancel = np.sign(cancel_data['BidOrder'] - cancel_data['AskOrder']) * cancel_data['Volume']
		return sign_cancel.rolling(rolling_time,  closed='left').apply(function_inside, args=(decay_factor,), raw=True)


class tran_price_change_vol_decay(base_feature):
	param_list = ['nperiod', 'decay_factor']

	def feature(self, data_fed, nperiod, decay_factor):

		stk_data = data_fed['transaction_data']

		def function_inside1(data):
			if len(data) == 0:
				return 0
			else:
				return np.sign(data[-1] - data[0])

		def function_inside2(data, decay_factor):
			if len(data) == 0:
				return 0
			else:
				decay_weight = decay_within_ticks(len(data), decay_factor)
				net_sum = (data * decay_weight).sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[stk_data['Price'] != 0]
		price_change_sign = selected_data['Price'].rolling(rolling_time,  closed='left').apply(function_inside1, raw=True)
		vol_sum_decay = selected_data['Volume'].rolling(rolling_time,  closed='left').apply(function_inside2, args=(decay_factor,), raw=True)
		return vol_sum_decay * price_change_sign


class tran_price_change_points_decay(base_feature):
	param_list = ['nperiod', 'decay_factor']

	def feature(self, data_fed, nperiod, decay_factor):

		stk_data = data_fed['transaction_data']

		def function_inside(data, decay_factor):
			if len(data) == 0:
				return 0
			else:
				decay_weight = decay_within_ticks(len(data), decay_factor)
				net_sum = (data * decay_weight).sum()
				return np.log(net_sum) if net_sum > 0 else -np.log(-net_sum) if net_sum < 0 else 0

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[stk_data['Price'] != 0]
		price_diff = selected_data['Price'].diff(1).fillna(0)
		price_change_pos = price_diff != 0
		price_change_part_vol = selected_data['Volume'][price_change_pos]
		rolling_series = price_change_part_vol * price_diff[price_change_pos]
		return rolling_series.rolling(rolling_time,  closed='left').apply(function_inside, args=(decay_factor,), raw=True)


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


class tran_type_num_diff(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				result = data.sum()
				return result

		rolling_time = str(nperiod * 3) + 'S'
		selected_data = stk_data[stk_data['FunctionCode'] != 67]
		bs_multiplier = pd.Series(np.where(selected_data['BSFlag'].values == 66, 1, -1), index=selected_data.index)
		return bs_multiplier.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)