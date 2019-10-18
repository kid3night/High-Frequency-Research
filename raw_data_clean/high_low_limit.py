import numpy as np
import pandas as pd
import os
import h5py
from multiprocessing import Pool



def get_keysHDF(path):

    f = h5py.File(path)
    keys = list(f.keys())
    return keys


def generate_high_low_limit(h5path, savepath):

	h5 = pd.HDFStore(h5path, 'r')
	keys = get_keysHDF(h5path)
	for key in keys:
		print(key)
		sr = h5[key][['HighLimit', 'LowLimit']].iloc[-1]
		sr.to_hdf(savepath, key, mode='a')


def need_dates(path, begin_date, end_date):

    h5_list = np.array(os.listdir(path))
    h5_list.sort()
    need_days = h5_list[(h5_list >= '{}.h5'.format(begin_date)) & (h5_list <= '{}.h5'.format(end_date))]
    return need_days


if __name__ == "__main__":
	path = 'F:/tick_raw'
	objpath = 'F:/high_low_limit'
	begin_date = 20180703
	end_date = 20180723
	multi_proc = 10
	needed_dates = need_dates(path, begin_date, end_date)

	if not os.path.exists(objpath):
		os.mkdir(objpath)
	h5s = os.listdir(path)

	if multi_proc:
		p = Pool(multi_proc)
		for day in needed_dates:
			print(day)
			h5path = path + '/' + day
			savepath = objpath + '/' + day
			p.apply_async(generate_high_low_limit, args=(h5path, savepath,))
		p.close()
		p.join()
	else:
	    for day in needed_dates:
	        print(day)
	        h5path = path + '/' + day
	        savepath = objpath + '/' + day
	        generate_high_low_limit(h5path, savepath)

