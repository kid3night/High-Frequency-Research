import pandas as pd
import numpy as np
import h5py


def get_keysHDF(path):

    f = h5py.File(path)
    keys = list(f.keys())
    return keys


def get_universe_from_index_excel(data_path):

    c1 = '成分券代码\nConstituent Code'
    c2 = '交易所\nExchange'
    data = pd.read_excel(data_path)
    f1 = lambda x: '{:06}'.format(x)
    f2 = lambda x: '.SZ' if x == 'Shenzhen' else '.SH'
    stk_code = np.array(data[c1].apply(f1))
    exchange_code = np.array(data[c2].apply(f2))
    result = stk_code + exchange_code
    return result


def get_universe_from_index_csv(data_path):

    universe_input = pd.read_csv(data_path, header=None)
    result = universe_input[1].values.astype(str)
    return result


# def verify_tickers(data_path, order_data):

# 	universe_input = get_universe_from_index_csv(data_path)
# 	# if order_data:
# 	# 	ticker_saved = np.array(get_keysHDF('F:/order_concated_data_check_08_09/order_concated.h5'))
# 	# else:
# 	# 	ticker_saved = np.array(get_keysHDF('F:/transaction_concated_data_check_08_09/transaction_concated.h5'))
#     if order_data:
#         ticker_saved = np.array(get_keysHDF('F:/order_concated_data_check/order_concated.h5'))
#     else:
#         ticker_saved = np.array(get_keysHDF('F:/transaction_concated_data_check/transaction_concated.h5'))
#     # if order_data:
#     #     ticker_saved = np.array(get_keysHDF('F:/order_concated_data_check/order_concated.h5'))
#     # else:
#     #     ticker_saved = np.array(get_keysHDF('F:/transaction_concated_data_check/transaction_concated.h5'))

#     intersect = np.intersect1d(universe_input, ticker_saved)    
#     diff = np.setdiff1d(universe_input, ticker_saved)
# 	return intersect, diff


# def get_intersect_diff(input_tickers, order_data):
#     # if order_data:
#     #     ticker_saved = np.array(get_keysHDF('F:/order_concated_data_check_08_09/order_concated.h5'))
#     # else:
#     #     ticker_saved = np.array(get_keysHDF('F:/transaction_concated_data_check_08_09/transaction_concated.h5'))
#     if order_data:
#         ticker_saved = np.array(get_keysHDF('F:/order_concated_data_check/order_concated.h5'))
#     else:
#         ticker_s np.array(get_keysHDF('F:/transaction_concated_data_check/transaction_concated.h5'))
        
        
#     intersect = np.intersect1d(input_tickers, ticker_saved)
#     diff = np.setdiff1d(input_tickers, ticker_saved)
#     return intersect, diff



# def verify_tickers(data_path, order_data):
#     universe_input = get_universe_from_index_csv(data_path)
#     if order_data:
#       ticker_saved = np.array(get_keysHDF('F:/order_concated_data_check/order_concated.h5'))
#     else:
#       ticker_saved = np.array(get_keysHDF('F:/transaction_concated_data_check/transaction_concated.h5'))
#     intersect = np.intersect1d(universe_input, ticker_saved)    
#     diff = np.setdiff1d(universe_input, ticker_saved)
#     return intersect, diff

# def get_intersect_diff(input_tickers, order_data):
#     if order_data:
#         ticker_saved = np.array(get_keysHDF('F:/order_concated_data_check/order_concated.h5'))
#     else:
#         ticker_saved = np.array(get_keysHDF('F:/transaction_concated_data_check/transaction_concated.h5'))
#     intersect = np.intersect1d(input_tickers, ticker_saved)
#     diff = np.setdiff1d(input_tickers, ticker_saved)
#     return intersect, diff


def verify_tickers(data_path, order_data):
    universe_input = get_universe_from_index_csv(data_path)
    if order_data:
      ticker_saved = np.array(get_keysHDF('F:/order_concated_data_check_08_09/order_concated.h5'))
    else:
      ticker_saved = np.array(get_keysHDF('F:/transaction_concated_data_check_08_09/transaction_concated.h5'))
    intersect = np.intersect1d(universe_input, ticker_saved)    
    diff = np.setdiff1d(universe_input, ticker_saved)
    return intersect, diff

def get_intersect_diff(input_tickers, order_data):
    if order_data:
        ticker_saved = np.array(get_keysHDF('F:/order_concated_data_check_08_09/order_concated.h5'))
    else:
        ticker_saved = np.array(get_keysHDF('F:/transaction_concated_data_check_08_09/transaction_concated.h5'))
    intersect = np.intersect1d(input_tickers, ticker_saved)
    diff = np.setdiff1d(input_tickers, ticker_saved)
    return intersect, diff