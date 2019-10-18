import os
import pandas as pd
import numpy as np


def find_missing_days_raw(all_days, types)

	# find out how many days haven't been processed for raw data clean
	days = set(all_days)
	dates = list(days)
	dates.sort()
	types = 'transaction'
	dirt = 'F:\\{}_raw'.format(types)
	folders = os.listdir(dirt)
	files = [int(i[:-3]) for i in folders]
	order_not = [i for i in dates if i not in files]
	return order_not