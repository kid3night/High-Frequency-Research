import pandas as pd
import numpy as np
import os
import csv


ord_ = lambda x: ord(x) if isinstance(x, str) else ord(str(int(x)))
v_order = np.vectorize(ord_)


def split_Files(data, data_type, save_path, fixed_format=True):

    if data_type == 'tick':
        split_ticker_file(data, save_path, fixed_format)
    elif data_type == 'transaction':
        split_transaction_file(data, save_path, fixed_format)
    elif data_type == 'order':
        split_order_file(data, save_path, fixed_format)
    elif data_type == 'index':
        split_index_file(data, save_path, fixed_format)

        
def split_transaction_file(data, save_path, fixed_format=True):

    data = data.fillna(0)
    result = pd.DataFrame()
    result[['Time', 'Index', 'BSFlag', 'Price', 'Volume', 'Turnover', 'AskOrder', 'BidOrder']] = data[['Time', 'Index', 'BSFlag', 'Price', 'Volume', 'Turnover', 'AskOrder', 'BidOrder']]
    result['OrderKind'] = data['OrderKind'].values.astype(int)
    result['FunctionCode'] = v_order(data['FunctionCode'].values)
    result = result.sort_values(by=['Index'])
    result = result.drop_duplicates()
    day = str(data['TradingDay'].iloc[0])
    ticker = data['WindCode'].values[0]
    print('Drop_Duplicates:', ticker)
    save_name = save_path + '/' + day + '.h5'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if fixed_format:
        result.to_hdf(save_name, key=ticker, mode='a')
    else:
        result.to_hdf(save_name, format='table', key=ticker, mode='a')


def split_order_file(data, save_path, fixed_format=True):

    result = pd.DataFrame()
    result[['Time', 'Order', 'Price', 'Volume']] = data[['Time', 'Order', 'Price', 'Volume']]
    result['Amount'] = ((result['Price'].values * result['Volume'].values) / 10000).astype(int)
    result['OrderKind'] = v_order(data['OrderKind'].values)
    result['FunctionCode'] = v_order(data['FunctionCode'].values)
    result = result.sort_values(by=['Order'])
    result = result.drop_duplicates()
    day = str(data['TradingDay'].iloc[0])
    ticker = data['WindCode'].values[0]
    print('Drop_Duplicates:', ticker)
    save_name = save_path + '/' + day + '.h5'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if fixed_format:
        result.to_hdf(save_name, key=ticker, mode='a')
    else:
        result.to_hdf(save_name, format='table', key=ticker, mode='a')


def split_level(x, name):
    temp_table = x[name].str.split(',',expand=True)
    temp_table.iloc[:, 0] = temp_table.iloc[:, 0].apply(lambda x: x[1:])
    return temp_table.iloc[:, :10].astype(int)


def split_ticker_file(data, save_path, fixed_format=True):
    df_new = data.loc[:, ['Time', 'Status', 'PreClose', 'Open', 'High', 'Low', 'Close',
                          'TransactionNum', 'TransactionVol', 'TransactionAmount',
                          'TotalBidVol','TotalAskVol', 'WeightedAvgBidPrice',
                          'WeightedAvgAskPrice','HighLimit', 'LowLimit']]
    
    muti_col = ['AskPrices', 'AskVols','BidPrices','BidVols']
    for name in muti_col:
        col_names = ["{}{}".format(name[:-1], i) for i in range(1, 11)]
        splitted_data = split_level(data, name)
        splitted_data.columns = col_names
        df_new[col_names] = splitted_data
    df_new = df_new.sort_values(by=['Time'])
    day = str(data['TradingDay'].iloc[0])
    df_new.index = df_new['Time']
    df_new = df_new.drop_duplicates()
    ticker = data['WindCode'].values[0]
    print('Drop_Duplicates:', ticker)
    save_name = save_path + '/' + day + '.h5'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if fixed_format:
        df_new.to_hdf(save_name, key=ticker, mode='a')
    else:
        df_new.to_hdf(save_name, format='table', key=ticker, mode='a')
    return df_new


def split_index_file(data, save_path, fixed_format=True):
    df_new = data.loc[:, ['Time', 'PreClose', 'Open', 'High', 'Low', 'Close',
                          'TransactionVol', 'TransactionAmount']]

    df_new = df_new.sort_values(by=['Time'])
    day = str(data['TradingDay'].iloc[0])
    df_new.index = df_new['Time']
    df_new = df_new.drop_duplicates()
    ticker = data['WindCode'].values[0]
    print('Drop_Duplicates:', ticker)
    save_name = save_path + '/' + day + '.h5'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if fixed_format:
        df_new.to_hdf(save_name, key=ticker, mode='a')
    else:
        df_new.to_hdf(save_name, format='table', key=ticker, mode='a')
    return df_new


# def get_lines_csv(data_path):

#     def fix_nulls(s):
#         for line in s:
#             yield line.replace('\0', ' ')

#     r = csv.reader(fix_nulls(open(data_path)))
#     return sum(1 for row in r)


def get_lines_csv(data_path):
    i = 0
    with open(data_path, 'rb') as f:
        for l in f:
            i = i + 1
    return i


def read_csv_tick(data_path, nrow, skip_rows=[0]):
    data_type_dict = {0: 'int64', 1: 'int64', 2: 'int64', 3: 'int64', 4: 'O',
                      5: 'int64', 6: 'int64', 7: 'int64', 8: 'int64', 9: 'int64',
                      10: 'int64', 11: 'int64', 12: 'int64', 13: 'int64', 14: 'O', 
                      15: 'O', 16: 'O', 17: 'O', 18: 'int64', 19: 'int64', 
                      20: 'int64', 21: 'int64', 22: 'int64', 23: 'int64', 24: 'int64', 
                      25: 'int64', 26: 'int64', 27: 'int64', 28: 'int64', 29: 'O', 
                      30: 'int64', 31: 'int64', 32: 'int64'}
    if not nrow:
        file = open(data_path)
        # lines = sum(1 for row in csv.reader(file))
        
        drop_lines = []
        # ==========================
        # lines = get_lines_csv(data_path)
        # drop_lines = list(range(lines - 1, lines))  #  drop last 1 useless lines
        # ==========================  get total lines 

        data = pd.read_csv(data_path, skiprows=skip_rows + drop_lines, header=None, nrows=None,
                           error_bad_lines=False, low_memory=False, dtype=data_type_dict)
        # data = pd.read_csv(data_path, skiprows=skip_rows, skipfooter=1, header=None, nrows=None,
        #                    error_bad_lines=False, dtype=data_type_dict, engine='python')
    else:
        data = pd.read_csv(data_path, skiprows=skip_rows, header=None, nrows=nrow)
    return data



def read_index_tick(data_path, nrow, skip_rows=[0]):
    data_type_dict = {0: 'int64', 1: 'int64', 2: 'int64', 3: 'int64',
                      4: 'O', 5: 'int64', 6: 'int64', 7: 'int64', 8: 'int64', 
                      9: 'int64', 10: 'int64', 11: 'int64', 12: 'int64', 13: 'int64', 14: 'int64'}
    if not nrow:
        file = open(data_path)
        # lines = sum(1 for row in csv.reader(file))
        
        drop_lines = []

        # ===============================
        # lines = get_lines_csv(data_path)
        # drop_lines = list(range(lines - 1, lines))  #  drop last 1 useless lines
        # =============================== get total lines
        data = pd.read_csv(data_path, skiprows=skip_rows + drop_lines, header=None, nrows=None,
                           error_bad_lines=False, low_memory=False, dtype=data_type_dict)
        # data = pd.read_csv(data_path, skiprows=skip_rows, skipfooter=1, header=None, nrows=None,
        #                    error_bad_lines=False, dtype=data_type_dict, engine='python')
    else:
        data = pd.read_csv(data_path, skiprows=skip_rows, header=None, nrows=nrow)
    return data


def read_csv_transaction(data_path, nrow, skip_rows=[0]):
    data_type_dict = {0: 'int64', 1: 'int64', 2: 'int64', 3: 'int64', 
                      4: 'O', 5: 'int64', 6: 'int64', 7: 'int64', 
                      8: 'int64', 9: 'int64', 10: 'int64', 11: 'int64', 
                      12: 'float64', 13: 'O', 14: 'int64', 15: 'int64'}
    if not nrow:
        file = open(data_path)
        # print(data_path)
        # lines = sum(1 for row in csv.reader(file))
        
        drop_lines = []
        # ===============================
        # lines = get_lines_csv(data_path) 
        # drop_lines = list(range(lines - 1, lines))  #  drop last 1 useless lines
        # =============================== get total lines
        data = pd.read_csv(data_path, skiprows=skip_rows + drop_lines, header=None, nrows=None,
                           error_bad_lines=False, low_memory=False, dtype=data_type_dict)
        # data = pd.read_csv(data_path, skiprows=skip_rows, skipfooter=1, header=None, nrows=None,
        #                    error_bad_lines=False, dtype=data_type_dict)
    else:
        data = pd.read_csv(data_path, skiprows=skip_rows, header=None, nrows=nrow)
    return data

 
def read_csv_order(data_path, nrow, skip_rows=[0]):
    data_type_dict = {0: 'int64', 1: 'int64', 2: 'int64', 3: 'int64',
                      4: 'O', 5: 'int64', 6: 'int64', 7: 'int64', 8: 'int64', 
                      9: 'int64', 10: 'O', 11: 'O'}
    if not nrow:
        file = open(data_path)
        # lines = sum(1 for row in csv.reader(file))
        
        drop_lines = []
        # ===============================
        # lines = get_lines_csv(data_path)
        # drop_lines = list(range(lines - 1, lines))  #  drop last 1 useless lines
        # =============================== get total lines
        data = pd.read_csv(data_path, skiprows=skip_rows + drop_lines, header=None, nrows=None,
                           error_bad_lines=False, low_memory=False, dtype=data_type_dict)
        # data = pd.read_csv(data_path, skiprows=skip_rows, skipfooter=1, header=None, nrows=None,
        #                    error_bad_lines=False, dtype=data_type_dict, engine='python')
    else:
        data = pd.read_csv(data_path, skiprows=skip_rows, header=None, nrows=nrow)
    return data


def need_days(data_file_path, begin_date, end_date):

    sub_folder_list = np.array(os.listdir(data_file_path))
    sub_folder_list.sort()
    need_days = sub_folder_list[(sub_folder_list >= begin_date) & (sub_folder_list <= end_date)]
    return need_days