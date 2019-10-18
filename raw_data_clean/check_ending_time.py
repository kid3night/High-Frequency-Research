import pandas as pd
import numpy as np
import h5py
import os


def get_keysHDF(path):

    f = h5py.File(path)
    keys = list(f.keys())
    return keys


if __name__ == '__main__':

	day_list = os.listdir('F:/tick_raw')
	index = 1
	data_path = 'F:/tick_raw/{}'.format(day_list[index])
	data = pd.HDFStore(data_path, 'r')
	keys = get_keysHDF(data_path)
	last_time = np.zeros(len(keys))
	for k in range(len(keys)):
		last_time[k] = data[keys[k]]['Time'].iloc[-1]

	keys_array = np.array(keys)
	stocks_lessthan_1500 = keys_array[(last_time < 145900000)]
	status = np.zeros(len(stocks_lessthan_1500))
	for i in range(len(stocks_lessthan_1500)):
		status[i] = data[stocks_lessthan_1500[i]]['Status'].iloc[-1]

