import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import split_help as sh


class data_prcoess:

    def __init__(self, data_type, data_file_path, save_path, begin_date, end_date, multi_proc=False, nrow=None):

        self.data_type = data_type
        self.data_file_path = data_file_path
        self.save_path = save_path
        self.begin_date = begin_date
        self.end_date = end_date
        self.multi_proc = multi_proc
        self.nrow = nrow
        self.data = None


    def data_clean(self, folder):

        if self.data_type == 'tick':
            self.ticker_data_clean(folder)
        elif self.data_type == 'transaction':
            self.transac_data_clean(folder)
        elif self.data_type == 'order':
            self.order_data_clean(folder)
        elif self.data_type == 'index':
            self.index_data_clean(folder)


    def data_clean_batch(self):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        need_date = sh.need_days(data_file_path, self.begin_date, self.end_date)
        print(need_date)
        if self.multi_proc:
            p = Pool(multi_proc)
            for i, folder in enumerate(need_date):
                p.apply_async(self.data_clean, args=(folder,))
            p.close()
            p.join()
        else:
            for i, folder in enumerate(need_date):
                self.data_clean(folder)


    def transac_data_clean(self, folder):

        file_name = self.data_file_path + folder + "/transaction.csv"
        save_path = self.save_path
        data_type=self.data_type
        nrow = self.nrow

        print(file_name)

        data = sh.read_csv_transaction(file_name,  nrow=nrow, skip_rows=[0])
        data.columns = ['TradingDay', 'LocalTime', 'ServerTime', 'Time', 'WindCode',
                        'OriginCode', 'NatureDay', 'Index', 'Price', 'Volume', 'Turnover',
                        'BSFlag', 'OrderKind', 'FunctionCode', 'AskOrder', 'BidOrder']
        data.groupby('WindCode').apply(sh.split_Files,  data_type=data_type, save_path=save_path, fixed_format=True)


    def order_data_clean(self, folder):

        file_name = self.data_file_path + folder + "/order.csv"
        save_path = self.save_path
        data_type=self.data_type
        nrow = self.nrow
        print(file_name)
        data = sh.read_csv_order(file_name,  nrow=nrow, skip_rows=[0])
        data.columns = ['TradingDay', 'LocalTime', 'ServerTime', 'Time', 'WindCode', 
                        'OriginCode', 'NatureDay', 'Order', 'Price', 'Volume', 'OrderKind',
                        'FunctionCode']
        data.groupby('WindCode').apply(sh.split_Files, data_type=data_type, save_path=save_path, fixed_format=True)

                
    def ticker_data_clean(self, folder):

        file_name = self.data_file_path + folder + "/stock.csv"
        save_path = self.save_path
        data_type = self.data_type
        nrow = self.nrow

        print(file_name)

        # data = pd.read_csv(file_name, skiprows=[0], header=None, nrows=nrow)
        data = sh.read_csv_tick(file_name, nrow=nrow, skip_rows=[0])
        data.columns = ['TradingDay', 'LocalTime', 'ServerTime', 'Time', 'WindCode', 'OriginCode', 'NatureDay',
                        'ExecutionDay', 'Status', 'PreClose', 'Open', 'High', 'Low', 'Close', 'AskPrices', 'AskVols',
                        'BidPrices', 'BidVols', 'TransactionNum', 'TransactionVol', 'TransactionAmount',  'TotalBidVol',
                        'TotalAskVol', 'WeightedAvgBidPrice','WeightedAvgAskPrice', 'IPOV', 'RET', 'HighLimit',
                        'LowLimit', 'SecurityInfo', 'PE1', 'PE2', 'SD']
        data.groupby('WindCode').apply(sh.split_Files, data_type=data_type, save_path=save_path, fixed_format=True)


    def index_data_clean(self, folder):

        file_name = self.data_file_path + folder + "/index.csv"
        save_path = self.save_path
        data_type = self.data_type
        nrow = self.nrow

        print(file_name)

        # data = pd.read_csv(file_name, skiprows=[0], header=None, nrows=nrow)
        data = sh.read_index_tick(file_name, nrow=nrow, skip_rows=[0])
        data.columns = ['TradingDay', 'LocalTime', 'ServerTime', 'Time', 'WindCode', 'OriginCode', 'NatureDay',
                        'ExecutionDay', 'Open', 'High', 'Low', 'Close', 'TransactionVol', 'TransactionAmount',
                        'PreClose']
        data.groupby('WindCode').apply(sh.split_Files, data_type=data_type, save_path=save_path, fixed_format=True)



if __name__ == "__main__":

    import warnings
    warnings.filterwarnings('ignore')
    print('hhh')
    data_file_path = "F:/qtdata/"
    data_type = 'tick'
    save_path = 'F:/{}_raw'.format(data_type)
    bgd = "20180824"
    edd = "20180904"
    nrow = None
    multi_proc = 4
    transac_raw = data_prcoess(data_type, data_file_path, save_path, bgd, edd, multi_proc, nrow)
    transac_raw.data_clean_batch()







