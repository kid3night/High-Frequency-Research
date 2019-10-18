# import sys
# sys.path.append('./backtest_main_files/')
# print(sys.path)

import auxiliaryFunctionsNew1009 as af
import pandas as pd

# data feed class
# Used for getting data needed


class dataFeed():

    def __init__(self, start_time, end_time, ticker, tick_path, target_path, order_path, transaction_path, orderbook_path, feature_type, each_type_data_need):

        self.target_scope = ['MidReturn15', 'MidReturn30', 'MidReturn60',
                             'MidReturn90', 'MidReturn300', 'MidReturn600',
                             'MidReturn900', 'MidReturn1500', 'MidReturn2400']
        self.tick_path = tick_path
        self.target_path = target_path
        self.order_path = order_path 
        self.transaction_path = transaction_path 
        self.orderbook_path = orderbook_path
        self.ticker = ticker
        self.start_time = start_time
        self.end_time = end_time
        self.feature_type = feature_type  # a list include feature data used [tick, orderbook, order, transaction] etc...
        self.each_type_data_need = each_type_data_need  # a dict stored columns needed for each type of data 
        self.selectStockTest(tick_path, target_path)

    def selectStockTest(self, tick_path, target_path):

        self.data_feed, self.data_affiliated = af.fetchData(self.tick_path, self.target_path, self.order_path,
                                                            self.transaction_path, self.orderbook_path, self.ticker, 
                                                            self.start_time, self.end_time, self.target_scope, self.feature_type,
                                                            self.each_type_data_need)
