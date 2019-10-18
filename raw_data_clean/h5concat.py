import pandas as pd
import numpy as np
import os
from datetime import datetime


def slice_date(date):

    day = date % 100
    month = date // 100 % 100
    year = date // 10000
    return (year, month, day)


def slice_time(day, time):
    hour = time // 10000000
    minute = time // 100000 % 100
    sec =  time // 1000 % 100
#     ms = time  % 1000 * 1000
    return datetime(*slice_date(day), hour, minute, sec)


v_time_split = np.vectorize(slice_time)


def begin_tick_creator(begin_name, h5_path, save_name, save_path):

	data_path = h5_path + '/' + begin_name
	save_data_path = save_path + '/' + save_name
	if not os.path.exists(save_path):
	    os.makedirs(save_path)
	old_store = pd.HDFStore(data_path)
	keys = old_store.keys()
	for key in keys:
		print(begin_name, key)
		old_store[key].to_hdf(save_data_path, key=key, mode='a', format='table')


def begin_other_creator(begin_name, h5_path, save_name, save_path):

	data_path = h5_path + '/' + begin_name
	save_data_path = save_path + '/' + save_name
	if not os.path.exists(save_path):
	    os.makedirs(save_path)
	day = int(begin_name[:-3])
	old_store = pd.HDFStore(data_path)
	keys = old_store.keys()
	for key in keys:
		print(day, key)
		data = old_store[key]
		new_df = data.loc[:, data.columns != 'Time']
		new_df.index = v_time_split(day, data['Time'])
		day_time = new_df.index.hour * 100 + new_df.index.minute
		select_vector = (day_time >= 930) & (day_time < 1457)
		new_df = new_df.loc[select_vector, :]
		new_df.to_hdf(save_data_path, key=key, mode='a', format='table')


def concat_tick(save_name, save_path, h5_list, h5_path):

	# concat cleaned tick bars

	save_data_path = save_path + '/' + save_name
	begin = pd.HDFStore(save_data_path)

	for day in h5_list:
		data_path = h5_path + '/' + day
		new_store = pd.HDFStore(data_path)
		keys = new_store.keys()
		for key in keys:
			print(day, key)
			begin.append(key, new_store[key])
		new_store.close()


def concat_other(save_name, save_path, h5_list, h5_path):

	# concat cleaned order, transaction data

	save_data_path = save_path + '/' + save_name
	begin = pd.HDFStore(save_data_path)

	for d in h5_list:
		day = int(d[:-3])
		data_path = h5_path + '/' + d
		new_store = pd.HDFStore(data_path)
		keys = new_store.keys()
		for key in keys:
			print(day, key)
			data = new_store[key]
			new_df = data.loc[:, data.columns != 'Time']
			new_df.index = v_time_split(day, data['Time'])
			day_time = new_df.index.hour * 100 + new_df.index.minute
			select_vector = (day_time >= 930) & (day_time < 1457)
			new_df = new_df.loc[select_vector, :]
			begin.append(key, new_df)

		new_store.close()


def need_dates(data_file_path, begin_date, end_date):

    h5_list = np.array(os.listdir(data_file_path))
    h5_list.sort()
    need_days = h5_list[(h5_list >= '{}.h5'.format(begin_date)) & (h5_list <= '{}.h5'.format(end_date))]
    return need_days


class h5_concat():

	def __init__(self, h5_path, save_path, save_name, concat_mode, data_type, begin_date, end_date):

		self.h5_path = h5_path
		self.save_path = save_path
		self.save_name = save_name
		self.concat_mode = concat_mode
		self.data_type = data_type
		self.begin_date = begin_date
		self.end_date = end_date

	def run_concat(self):
		
		need_days = need_dates(self.h5_path, self.begin_date, self.end_date)
		if self.concat_mode == 'from_begin':

			if self.data_type == 'tick' or self.data_type == 'targets':
				begin_name = need_days[0]
				begin_tick_creator(begin_name, self.h5_path, self.save_name, self.save_path)
				concat_tick(self.save_name, self.save_path, need_days[1:], self.h5_path)
			else:
				begin_name = need_days[0]
				begin_other_creator(begin_name, self.h5_path, self.save_name, self.save_path)
				concat_other(self.save_name, self.save_path, need_days[1:], self.h5_path)
		else:
			if self.data_type == 'tick' or self.data_type == 'targets':
				concat_tick(self.save_name, self.save_path, need_days, self.h5_path)
			else:
				concat_other(self.save_name, self.save_path, need_days, self.h5_path)


if __name__ == "__main__":

	import warnings
	warnings.filterwarnings('ignore')
	# h5_path = 'F:/tick_bar'
	# h5_path = 'F:/targets_bar_new'
	# h5_path = 'F:/transaction_raw'
	h5_path = 'F:/tick_bar'
	# h5_path = 'F:/order_raw'
	data_type = 'tick'
	save_path = 'F:/{}_concated_data_check_08_09'.format(data_type)
	# save_path = 'F:/{}_concated'.format(data_type)
	save_name = '{}_concated.h5'.format(data_type)
	concat_mode = 'from_begin'
	# concat_mode = 'other'
	begin_date = 20180802
	end_date = 20180905
	h5s_concat = h5_concat(h5_path, save_path, save_name, concat_mode, data_type, begin_date, end_date)
	h5s_concat.run_concat()
