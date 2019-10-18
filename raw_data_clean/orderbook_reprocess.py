import pandas as pd
import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool

# class orderbook_reprocess():

# 	def __init__(self):

# 		self.

def get_mid(x, y):
	if np.isnan(x):
		return y - 1
	elif np.isnan(y):
		return x + 1
	else:
		return (x + y) / 2

v_mid = np.vectorize(get_mid)


def slice_date(date):

    day = date % 100
    month = date // 100 % 100
    year = date // 10000
    return (year, month, day)


def slice_time(day, time):
    hour = time // 10000000
    minute = time // 100000 % 100
    sec =  time // 1000 % 100
    return datetime(*slice_date(day), hour, minute, sec)

v_time_split = np.vectorize(slice_time)


def get_nono_zeros_part(vector, prices, pos, size):

    base = np.zeros(size * 2)
    bid_part = vector[:pos]
    ask_part = vector[pos:]
    bid_nonzero = bid_part[np.nonzero(bid_part)[0][-size:]]
    ask_nonzero = ask_part[np.nonzero(ask_part)[0][:size]]

    price_list = np.zeros(size * 2)
    bid_price = prices[:pos]
    ask_price = prices[pos:]

    bid_price_nonzero = bid_price[np.nonzero(bid_part)[0][-size:]]
    ask_price_nonzero = ask_price[np.nonzero(ask_part)[0][:size]]

    if len(bid_nonzero) > 0:
        base[size - len(bid_nonzero):size] = bid_nonzero
        base[size:size + len(ask_nonzero)] = ask_nonzero
        price_list[size - len(bid_nonzero):size] = bid_price_nonzero
        price_list[size:size + len(ask_nonzero)] = ask_price_nonzero
    return base, price_list
        

def get_new_orderbook(data, size, trading_day):

	length, width = data.shape[0], data.shape[1] - 3

	if length > 2:
		midprices = v_mid(data['Bid1'].values, data['Ask1'].values)
		price_list = data.columns[:-3].values.astype(int)
		mid_pos = np.searchsorted(price_list, midprices, side='right')
		new_orderbook = np.zeros((length, size * 2))
		new_orderbook_prices = np.zeros((length, size * 2))
		data_numpy = data.values[:, :-3]
		time_index = v_time_split(trading_day, data['Time'].values)
		bid_ask_columns = ['Bid{}'.format(i) for i in range(size, 0, -1)] + ['Ask{}'.format(i) for i in range(1, size + 1)]
		for i in range(length):
			new_orderbook[i, :], new_orderbook_prices[i, :] = get_nono_zeros_part(data_numpy[i, :], price_list, mid_pos[i], size)
		new_orderbook_df = pd.DataFrame(new_orderbook, index=time_index, columns=bid_ask_columns)
		new_orderbook_prices_df = pd.DataFrame(new_orderbook_prices, index=time_index, columns=bid_ask_columns)
		return new_orderbook_df, new_orderbook_prices_df
	else:
		return (None, None)


def need_dates(data_file_path, begin_date, end_date):

    h5_list = np.array(os.listdir(data_file_path))
    h5_list.sort()
    need_days = h5_list[(h5_list >= '{}'.format(begin_date)) & (h5_list <= '{}'.format(end_date))]
    return need_days
	

def folder_process(data_path, day, size, save_path_orderbook, save_path_orderbook_price):

	files = os.listdir(data_path)
	for ticker in files:
		print(day, ticker)
		key_ = ticker[:-3]
		data = pd.read_hdf(data_path + '/' + ticker, key=key_, mode='r')
		new_order_book, new_orderbook_price = get_new_orderbook(data, size, int(day))
		if new_order_book is not None and new_orderbook_price is not None:
			new_order_book.to_hdf(save_path_orderbook + '/' + day + '.h5', key=ticker, mode='a')
			new_orderbook_price.to_hdf(save_path_orderbook_price + '/' + day + '.h5', key=ticker, mode='a')



def to_new_h5(folder_path, size, save_path_orderbook, save_path_orderbook_price, multi_proc, begin_date, end_date):

	needed_dates = need_dates(folder_path, begin_date, end_date)
	if multi_proc:
		p = Pool(multi_proc)
		for day in needed_dates:
			data_path = folder_path + '/' + day
			p.apply_async(folder_process, args=(data_path, day, size, save_path_orderbook, save_path_orderbook_price,))
		p.close()
		p.join()
	else:
		for day in needed_dates:
			data_path = folder_path + '/' + day
			folder_process(data_path, day, size, save_path_orderbook, save_path_orderbook_price)


if __name__ == '__main__':
	
	folder_path = 'F:/orderbook_raw'
	size = 100
	save_path_orderbook = 'F:/orderbook_process/'
	save_path_orderbook_price = 'F:/orderbook_price_process/'
	if not os.path.exists(save_path_orderbook):
		os.makedirs(save_path_orderbook)
	if not os.path.exists(save_path_orderbook_price):
		os.makedirs(save_path_orderbook_price)
	multi_proc = 0
	begin_date = 20180601
	end_date = 20180601
	to_new_h5(folder_path, size, save_path_orderbook, save_path_orderbook_price, multi_proc, begin_date, end_date)


