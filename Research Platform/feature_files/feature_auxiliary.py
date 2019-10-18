import numpy as np

def decay_within_tickers(leng, decay_factor):
	weight = np.array([decay_factor ** i if decay_factor ** i > 0.1 else 0.1 for i in range(leng - 1, -1, -1)])
	return weight