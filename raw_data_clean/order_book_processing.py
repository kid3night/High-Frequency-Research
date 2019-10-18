import time
from datetime import datetime, timedelta, date
from decimal import Decimal
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
import xml.etree.ElementTree as ET
import utils
from multiprocessing import Pool
import warnings
import h5py
warnings.filterwarnings("ignore")


def get_keysHDF(path):

    f = h5py.File(path)
    keys = list(f.keys())
    return keys


def read_xml(in_path):
    """读取并解析xml文件
           in_path: xml路径
           return: tree"""
    tree = ET.parse(in_path)
    return tree


def creat_dict(root):
    """xml生成为dict：，
    将tree中个节点添加到list中，将list转换为字典dict_init
    叠加生成多层字典dict_new"""
    dict_new = {}
    for key, valu in enumerate(root):
        dict_init = {}
        list_init = []
        for item in valu:
            list_init.append([item.tag, item.text])
            for lists in list_init:
                dict_init[lists[0]] = lists[1]
        dict_new[key] = dict_init
    return dict_new


def caculate_limit_price(ticker, day):

    high_low_h5 = pd.read_hdf('F:/high_low_limit/{}.h5'.format(day), key=ticker, mode='r')
    (down_limit, up_limit) =  high_low_h5['LowLimit'], high_low_h5['HighLimit']
    return down_limit,up_limit


class Order():
    '''
    Order类
    '''
    def __init__(self, itime, iorderid, iprice, iquantity, iorder_kind, ifunction_code):
        self.timestamp = itime
        self.orderid = iorderid
        self.price = iprice
        self.quantity = iquantity  # Volume
        self.order_kind = iorder_kind
        self.function_code = ifunction_code
        self.order_type()

    def order_type(self):
        '''
        现在看来，可以将order分成三类：限价单，市价单，本方最优单。
        限价单的match规则最简单，本方最优单可以理解为最新bid或ask的限价单。
        市价单的match最复杂，分为通常的市价（涨跌停价的限价）；对方最优成交，剩余挂单；5
        档最优成交，剩余撤单
        遗憾的事，依据现有数据特征，不能直接区分市价单的类别，只能依据接下来的transaction来识别。
        市价单的处理方式：默认对方最优成交，通过接下来的transaction来识别是否需要修正。
        限价单：order_type_num = 0
        市价单：order_type_num = 1
        本方最优：order_type_num = "U"
        :return:
        '''
        if self.order_kind == 48:  # 限价
            self.order_type_num = 0
        elif self.order_kind == 49:  # 通常的市价
            self.order_type_num = 1
        elif self.order_kind == 85:  # 本方最优排队
            self.order_type_num = 2
        else:
            print("order_type error")


class Transaction():
    '''
    Transaction类
    '''
    def __init__(self, itime, iindex, iask_order_id, ibid_order_id, ibs_flag, iprice, iquantity, iorder_kind, ifunction_code):
        self.timestamp = itime
        self.index = iindex
        self.ask_orderid = iask_order_id
        self.bid_orderid = ibid_order_id
        self.bs_flag = ibs_flag
        self.price = iprice
        self.quantity = iquantity
        self.order_kind = iorder_kind
        self.function_code = ifunction_code


class OrderTree():
    '''
    order树
    '''
    def __init__(self, limit_down, limit_up, tick_size):
        self.limit_down = limit_down
        self.limit_up = limit_up
        self.tick_size = tick_size
        self.price_list = np.arange(limit_down, limit_up + self.tick_size, self.tick_size)
        self.initialize_tree()
        # orderid和price的mapping,在每次insert_order函数调用时增加，作用：cancel order时方便查找
        self.orderid_price_mapping = pd.Series()
        pass

    def initialize_tree(self):
        self.tree = dict()
        for price in self.price_list:
            self.tree[price] = OrderedDict()


class OrderBook():

    def __init__(self, DAY, TICKER, DOWNLIMIT, UPLIMIT, TICKSIZE):
        self.DOWNLIMIT = DOWNLIMIT
        self.UPLIMIT = UPLIMIT
        self.TICKSIZE = TICKSIZE
        self.DAY = DAY
        self.TICKER = TICKER
        self.bid_tree = OrderTree(DOWNLIMIT,UPLIMIT,TICKSIZE)
        self.ask_tree = OrderTree(DOWNLIMIT,UPLIMIT,TICKSIZE)
        self.order_instance = Order(91459990,0,0,0,48,32)
        self.transaction_instance = Transaction(91459990,0,0,0,32,0,0,0,32)
        self.b_a_total = list()

    def order_update(self,itime,iorderid,iprice,iquantity,iorder_kind,ifunction_code):
        self.order_instance = Order(itime,
                                    iorderid,
                                    iprice,
                                    iquantity,
                                    iorder_kind,
                                    ifunction_code)

    def transaction_update(self,itime,iindex,iask_order_id,ibid_order_id,ibs_flag,iprice,iquantity,iorder_kind,ifunction_code):
        self.transaction_instance = Transaction(itime,
                                                iindex,
                                                iask_order_id,
                                                ibid_order_id,
                                                ibs_flag,
                                                iprice,
                                                iquantity,
                                                iorder_kind,
                                                ifunction_code)

    def insert_order(self):
        if self.order_instance.function_code==66:    # buy order
            self.bid_tree.tree[self.order_instance.price][self.order_instance.orderid] = self.order_instance.quantity
            self.bid_tree.orderid_price_mapping.loc[self.order_instance.orderid] = self.order_instance.price
        else:   # sell order
            # print(self.order_instance.keys())
            # print(type(self.order_instance))
            print(self.order_instance.timestamp, self.order_instance.orderid, self.order_instance.price, self.order_instance.quantity, self.order_instance.order_kind, self.order_instance.function_code)
            self.ask_tree.tree[self.order_instance.price][self.order_instance.orderid] = self.order_instance.quantity
            self.ask_tree.orderid_price_mapping.loc[self.order_instance.orderid] = self.order_instance.price

    def remove_cancel_order(self):
        '''
        处理transaction中的撤单
        :return:
        '''
        if self.transaction_instance.ask_orderid<1e-6:   # cancel bid order
            try:
                remove_price = self.bid_tree.orderid_price_mapping.loc[self.transaction_instance.bid_orderid]
                self.bid_tree.tree[remove_price].pop(self.transaction_instance.bid_orderid)
                self.bid_tree.orderid_price_mapping.pop(self.transaction_instance.bid_orderid)
            except:
                pass
        else:   # cancel ask order
            try:
                remove_price = self.ask_tree.orderid_price_mapping[self.transaction_instance.ask_orderid]
                self.ask_tree.tree[remove_price].pop(self.transaction_instance.ask_orderid)
                self.ask_tree.orderid_price_mapping.pop(self.transaction_instance.ask_orderid)
            except:
                pass

    def remove_open_auction_order(self):
        remove_ask_price = self.ask_tree.orderid_price_mapping[self.transaction_instance.ask_orderid]
        self.ask_tree.tree[remove_ask_price][self.transaction_instance.ask_orderid] -= self.transaction_instance.quantity
        if self.ask_tree.tree[remove_ask_price][self.transaction_instance.ask_orderid]<1e-6:
            self.ask_tree.tree[remove_ask_price].pop(self.transaction_instance.ask_orderid)
            self.ask_tree.orderid_price_mapping.pop(self.transaction_instance.ask_orderid)
        remove_bid_price = self.bid_tree.orderid_price_mapping[self.transaction_instance.bid_orderid]
        self.bid_tree.tree[remove_bid_price][self.transaction_instance.bid_orderid] -= self.transaction_instance.quantity
        if self.bid_tree.tree[remove_bid_price][self.transaction_instance.bid_orderid] < 1e-6:
            self.bid_tree.tree[remove_bid_price].pop(self.transaction_instance.bid_orderid)
            self.bid_tree.orderid_price_mapping.pop(self.transaction_instance.bid_orderid)

    def judge_order_kind(self):
        '''
        见Order.order_type
        :return:
        '''
        if self.order_instance.order_kind == 49:             # 如果委托为市价委托
            if self.order_instance.price in [0,10000,self.DOWNLIMIT,self.UPLIMIT]:
                if self.order_instance.function_code == 66:  # 如果是买单 则生成对应ask order里最优价格
                    self.order_instance.price = self.ask_tree.orderid_price_mapping.min()
                else:                                        # 如果是卖单 则生成对应bid order里最优价格
                    print('test', self.bid_tree.orderid_price_mapping)
                    self.order_instance.price = self.bid_tree.orderid_price_mapping.max()
            else:
                if self.order_instance.function_code == 66:
                    self.order_instance.price = self.ask_tree.orderid_price_mapping.min()
                else:
                    self.order_instance.price = self.bid_tree.orderid_price_mapping.max()
        elif self.order_instance.order_kind == 85:  # 本方最优价委托，修改成最新的bid1或ask1
            if self.order_instance.function_code == 66:
                temp0 = self.bid_tree.orderid_price_mapping.max()
                if not np.isnan(temp0):
                    self.order_instance.price = self.bid_tree.orderid_price_mapping.max()
                else:
                    self.order_instance.price = self.DOWNLIMIT
            else:
                temp0 = self.ask_tree.orderid_price_mapping.min()
                if not np.isnan(temp0):
                    self.order_instance.price = self.ask_tree.orderid_price_mapping.min()
                else:
                    self.order_instance.price = self.UPLIMIT
        else:   # 限价委托单
            pass

    def process_order(self):
        self.judge_order_kind()
        print(1, self.order_instance.timestamp, self.order_instance.orderid, self.order_instance.price, self.order_instance.quantity, self.order_instance.order_kind, self.order_instance.function_code)
        self.insert_order()
        self.match_order()
        if self.transaction_instance.timestamp > 92459990:
            bid_ask_qty = self.combine_bid_ask_tree()
            self.write_to_csv(bid_ask_qty)

    def process_transaction(self):
        '''
        处理撤单
        处理开盘92500000的match
        :return:
        '''
        if self.transaction_instance.function_code == 67:
            self.remove_cancel_order()
        if self.transaction_instance.timestamp == 92500000:
            self.remove_open_auction_order()

    def match_order(self):
        if self.order_instance.timestamp > 92459990:
            if self.order_instance.order_type_num in [0,2]:    # in case of limit, best price order 
                if self.order_instance.function_code == 66:    # buy order，match ask_tree
                    min_ask = self.ask_tree.orderid_price_mapping.min()
                    if self.order_instance.price<min_ask or np.isnan(min_ask): # judge whether there is any price to match
                        # print("不能match {}".format(self.order_instance.timestamp))
                        pass
                    else:
                        iprice = min_ask
                        while self.order_instance.price+1e-6>iprice>min_ask-1e6:
                            if not self.ask_tree.tree[iprice].__len__()==0:  # if such price can be matched
                                ask_tree_cp = self.ask_tree.tree[iprice].copy()
                                for k,v in ask_tree_cp.items():
                                    k_deal_qty = min(self.order_instance.quantity,v)   # deal with the correspond volume
                                    self.order_instance.quantity -= k_deal_qty
#                                     print("deal {0} in ask tree,price {1}, orderid {2} time {3}".format(k_deal_qty,
#                                                                                                         iprice,
#                                                                                                         k,
#                                                                                                         self.order_instance.timestamp))
                                    self.ask_tree.tree[iprice][k] -= k_deal_qty
                                    self.bid_tree.tree[self.order_instance.price][self.order_instance.orderid] -= k_deal_qty
                                    if self.ask_tree.tree[iprice][k]<1e-6:  # if old order vol is costed by new orders, pop the old order
                                        self.ask_tree.tree[iprice].pop(k)
                                        self.ask_tree.orderid_price_mapping.pop(k)
                                    if self.order_instance.quantity<=1e-6:  # if new order is costed by old orders, pop the new older
                                        self.bid_tree.tree[self.order_instance.price].pop(self.order_instance.orderid)
                                        self.bid_tree.orderid_price_mapping.pop(self.order_instance.orderid)
                                        break
                                if self.order_instance.quantity <= 1e-6:
                                    break
                            iprice += self.TICKSIZE
                elif self.order_instance.function_code == 83:    # sell order，match bid_tree
                    max_bid = self.bid_tree.orderid_price_mapping.max()
                    if self.order_instance.price>max_bid or np.isnan(max_bid):
                        # print("不能match {}".format(self.order_instance.timestamp))
                        pass
                    else:
                        iprice = max_bid
                        while self.order_instance.price-1e-6<iprice<max_bid+1e-6:
                            if not self.bid_tree.tree[iprice].__len__()==0:
                                bid_tree_cp = self.bid_tree.tree[iprice].copy()
                                for k,v in bid_tree_cp.items():
                                    k_deal_qty = min(self.order_instance.quantity,v)
                                    self.order_instance.quantity -= k_deal_qty
#                                     print("deal {0} in bid tree,price {1}, orderid {2} time {3}".format(k_deal_qty,
#                                                                                                        iprice,
#                                                                                                        k,
#                                                                                                        self.order_instance.timestamp))
                                    self.bid_tree.tree[iprice][k] -= k_deal_qty
                                    self.ask_tree.tree[self.order_instance.price][self.order_instance.orderid] -= k_deal_qty
                                    if self.bid_tree.tree[iprice][k]<1e-6:
                                        self.bid_tree.tree[iprice].pop(k)
                                        self.bid_tree.orderid_price_mapping.pop(k)
                                    if self.order_instance.quantity<=1e-6:
                                        self.ask_tree.tree[self.order_instance.price].pop(self.order_instance.orderid)
                                        self.ask_tree.orderid_price_mapping.pop(self.order_instance.orderid)
                                        break
                                if self.order_instance.quantity <= 1e-6:
                                    break
                            iprice -= self.TICKSIZE
            elif self.order_instance.order_type_num==1:  # in case of market order
                if self.order_instance.function_code == 66:    # buy order，match ask_tree
                    min_ask = self.ask_tree.orderid_price_mapping.min()
                    if self.order_instance.price<min_ask or np.isnan(min_ask):
                        # print("不能match {}".format(self.order_instance.timestamp))
                        pass
                    else:
                        iprice = min_ask
                        if not self.ask_tree.tree[iprice].__len__()==0:
                            ask_tree_cp = self.ask_tree.tree[iprice].copy()
                            for k,v in ask_tree_cp.items():
                                k_deal_qty = min(self.order_instance.quantity,v)
                                self.order_instance.quantity -= k_deal_qty
#                                 print("deal {0} in ask tree,price {1}, orderid {2} time {3}".format(k_deal_qty,
#                                                                                                     iprice,
#                                                                                                     k,
#                                                                                                     self.order_instance.timestamp))
                                self.ask_tree.tree[iprice][k] -= k_deal_qty
                                self.bid_tree.tree[self.order_instance.price][self.order_instance.orderid] -= k_deal_qty
                                if self.ask_tree.tree[iprice][k]<1e-6:
                                    self.ask_tree.tree[iprice].pop(k)
                                    self.ask_tree.orderid_price_mapping.pop(k)
                                if self.order_instance.quantity<=1e-6:
                                    self.bid_tree.tree[self.order_instance.price].pop(self.order_instance.orderid)
                                    self.bid_tree.orderid_price_mapping.pop(self.order_instance.orderid)
                                    break
                            if self.order_instance.quantity>1e-6:
                                self.match_verify()
                elif self.order_instance.function_code == 83:    # sell order，match bid_tree
                    max_bid = self.bid_tree.orderid_price_mapping.max()
                    if self.order_instance.price>max_bid or np.isnan(max_bid):
                        # print("不能match {}".format(self.order_instance.timestamp))
                        pass
                    else:
                        iprice = max_bid
                        if not self.bid_tree.tree[iprice].__len__()==0:
                            bid_tree_cp = self.bid_tree.tree[iprice].copy()
                            for k,v in bid_tree_cp.items():
                                k_deal_qty = min(self.order_instance.quantity,v)
                                self.order_instance.quantity -= k_deal_qty
#                                 print("deal {0} in bid tree,price {1}, orderid {2} time {3}".format(k_deal_qty,
#                                                                                                    iprice,
#                                                                                                    k,
#                                                                                                    self.order_instance.timestamp))
                                self.bid_tree.tree[iprice][k] -= k_deal_qty
                                self.ask_tree.tree[self.order_instance.price][self.order_instance.orderid] -= k_deal_qty
                                if self.bid_tree.tree[iprice][k]<1e-6:
                                    self.bid_tree.tree[iprice].pop(k)
                                    self.bid_tree.orderid_price_mapping.pop(k)
                                if self.order_instance.quantity<=1e-6:
                                    self.ask_tree.tree[self.order_instance.price].pop(self.order_instance.orderid)
                                    self.ask_tree.orderid_price_mapping.pop(self.order_instance.orderid)
                                    break
                            if self.order_instance.quantity>1e-6:
                                self.match_verify()

    def match_verify(self):
        '''
        验证order_type=3时的类型
        :return: 
        '''
        next_order_data = self.df0.order_data[self.order_cursor+1]
        next_order_id = next_order_data[1]
        temp1 = list(np.argwhere(self.df0.transaction_data[:, 1] > self.order_instance.orderid)[:, 0])
        temp2 = list(np.argwhere(self.df0.transaction_data[:, 1] < next_order_id)[:, 0])
        transaction_id_period = list(set(temp1).intersection(set(temp2)))
        if self.order_instance.function_code==83:
            for itran in transaction_id_period:
                if self.df0.transaction_data[itran,2]==self.order_instance.orderid:
                    if self.df0.transaction_data[itran,5]<self.order_instance.price and \
                       self.df0.transaction_data[itran, 8] == 48:   # 需要修改成5档成交
                        # 撤掉ask1上的挂单
                        if self.ask_tree.tree[self.order_instance.price].__len__()>1e-6:
                            self.ask_tree.tree[self.order_instance.price].pop(self.order_instance.orderid)
                            self.ask_tree.orderid_price_mapping.pop(self.order_instance.orderid)
                        # match at self.df0.transaction_data[itran,5]
                        self.bid_tree.tree[self.df0.transaction_data[itran,5]][self.df0.transaction_data[itran,3]] -= \
                            self.df0.transaction_data[itran, 6]
                        if self.bid_tree.tree[self.df0.transaction_data[itran,5]][self.df0.transaction_data[itran,3]]<1e-6:
                            self.bid_tree.tree[self.df0.transaction_data[itran, 5]].pop(self.df0.transaction_data[itran,3])
                            self.bid_tree.orderid_price_mapping.pop(self.df0.transaction_data[itran,3])
        if self.order_instance.function_code==66:
            for itran in transaction_id_period:
                if self.df0.transaction_data[itran,3]==self.order_instance.orderid:
                    if self.df0.transaction_data[itran,5]>self.order_instance.price and \
                       self.df0.transaction_data[itran, 8] == 48:   # 需要修改成5档成交
                        # 撤掉bid1上的挂单
                        if self.bid_tree.tree[self.order_instance.price].__len__()>1e-6:
                            self.bid_tree.tree[self.order_instance.price].pop(self.order_instance.orderid)
                            self.bid_tree.orderid_price_mapping.pop(self.order_instance.orderid)
                        # match at self.df0.transaction_data[itran,5]
                        self.ask_tree.tree[self.df0.transaction_data[itran,5]][self.df0.transaction_data[itran,2]] -= \
                            self.df0.transaction_data[itran, 6]
                        if self.ask_tree.tree[self.df0.transaction_data[itran,5]][self.df0.transaction_data[itran,2]]<1e-6:
                            self.ask_tree.tree[self.df0.transaction_data[itran, 5]].pop(self.df0.transaction_data[itran,2])
                            self.ask_tree.orderid_price_mapping.pop(self.df0.transaction_data[itran, 2])

    def combine_bid_ask_tree(self):
        price_list = np.arange(self.DOWNLIMIT, self.UPLIMIT+self.TICKSIZE, self.TICKSIZE)
        bid_ask_tot_qty = pd.Series(0,index=price_list)
        bid_ask_tot_qty["Time"] = self.order_instance.timestamp
        bid_ask_tot_qty["Bid1"] = self.bid_tree.orderid_price_mapping.max()
        bid_ask_tot_qty["Ask1"] = self.ask_tree.orderid_price_mapping.min()
        for price in price_list:
            bid_ask_tot_qty[price] -= sum(self.ask_tree.tree[price].values())
            bid_ask_tot_qty[price] += sum(self.bid_tree.tree[price].values())
            pass
        return bid_ask_tot_qty.to_frame().T

    def write_to_csv(self,bid_ask_tot_qty):
#         self.b_a_total = pd.concat([self.b_a_total,bid_ask_tot_qty],ignore_index=True)
        self.b_a_total.append(bid_ask_tot_qty.values[0])
        pass

    def save_to_h5(self):
        save_path0 = 'F:/orderbook_raw'
        if not os.path.exists(save_path0):
            os.mkdir(save_path0)
        y = str(self.DAY)[:4]
        save_path1 = os.path.join(save_path0,y)
        if not os.path.exists(save_path1):
            os.mkdir(save_path1)
        m = str(self.DAY)[4:6]
        save_path2 = os.path.join(save_path1, m)
        if not os.path.exists(save_path2):
            os.mkdir(save_path2)
        d = str(self.DAY)[6:]
        save_path3 = os.path.join(save_path2, d)
        if not os.path.exists(save_path3):
            os.mkdir(save_path3)
        save_path4 = os.path.join(save_path3,self.TICKER+".h5")
#   self.b_a_total.to_hdf(save_path4,self.TICKER)
        col_name = list(self.bid_tree.price_list)+["Time","Bid1","Ask1"]
        pd_data = pd.DataFrame(self.b_a_total,columns=col_name)
        pd_data.to_hdf(save_path4,self.TICKER)

    def run(self):
        self.order_cursor = 0
        self.transaction_cursor = 0
        while True:
            this_order_data = self.df0.order_data[self.order_cursor]
            this_transaction_data = self.df0.transaction_data[self.transaction_cursor]
            self.order_recent_id = this_order_data[1]
            self.transaction_recent_id = this_transaction_data[1]
            current_id = min(self.order_recent_id,self.transaction_recent_id)
            if current_id==self.order_recent_id<self.transaction_recent_id:
                self.order_update(this_order_data[0],
                                  this_order_data[1],
                                  this_order_data[2],
                                  this_order_data[3],
                                  this_order_data[4],
                                  this_order_data[5])
                print(this_order_data)
                print(self.order_instance.timestamp, self.order_instance.orderid, self.order_instance.price, self.order_instance.quantity, self.order_instance.order_kind, self.order_instance.function_code)
                self.process_order()
                self.order_cursor += 1
            elif current_id==self.transaction_recent_id<self.order_recent_id:
                self.transaction_update(this_transaction_data[0],
                                        this_transaction_data[1],
                                        this_transaction_data[2],
                                        this_transaction_data[3],
                                        this_transaction_data[4],
                                        this_transaction_data[5],
                                        this_transaction_data[6],
                                        this_transaction_data[7],
                                        this_transaction_data[8])
                self.process_transaction()
                self.transaction_cursor += 1

            if self.transaction_instance.timestamp>145500000:
                self.save_to_h5()
                break


class DataFeed():

    def __init__(self,DAY,TICKER):
        self.DAY = DAY
        self.TICKER = TICKER

    def read_orders(self):
        self.order_data = pd.read_hdf("F:/order_raw/{}.h5".format(self.DAY), self.TICKER, 'r')[["Time","Order","Price","Volume","OrderKind","FunctionCode"]].values
        self.order_data_len = self.order_data.shape[0]

    def read_transactions(self):

        self.transaction_data = pd.read_hdf("F:/transaction_raw/{}.h5".format(self.DAY), self.TICKER, 'r')[["Time","Index","AskOrder","BidOrder",
                                                                                                       "BSFlag","Price","Volume","OrderKind","FunctionCode"]].values
        self.transaction_data_len = self.transaction_data.shape[0]

    def get_data(self):
        self.read_orders()
        self.read_transactions()


def main():
    gz = r"./order_book.xml"
    gz_tree = read_xml(gz)
    gz_node = creat_dict(gz_tree.getroot())
    for i in range(gz_node.__len__()):
        start_day = gz_node[i]["STARTDAY"]
        end_day = gz_node[i]["ENDDAY"]
        # DAY = gz_node[i]["DAY"]
        TICKER = gz_node[i]["TICKER"]
        TICKSIZE = int(gz_node[i]["TICKSIZE"])
        trading_days = utils.get_period_trading_days(start_day, end_day)
        for DAY in trading_days:
            print("start {} {}".format(TICKER,DAY))
            DOWNLIMIT,UPLIMIT = caculate_limit_price(TICKER,DAY)
            df0 = DataFeed(DAY, TICKER)
            ob0 = OrderBook(DAY, TICKER, DOWNLIMIT, UPLIMIT, TICKSIZE)
            df0.get_data()
            ob0.df0 = df0
            ob0.run()
            print("end {} {}".format(TICKER,DAY))


def main_multiprocess():
    
    start_day = "20180601"
    end_day = "20180601"
    trading_days = utils.get_period_trading_days(start_day, end_day)
    print(trading_days)
    TICKSIZE = 100
    if len(trading_days)>0: 
        # p = Pool(10)
        # for DAY in trading_days:
        #     p.apply_async(mutiprocess_run, args=(DAY,TICKSIZE,))
        # p.close()
        # p.join()  
        mutiprocess_run(trading_days[0], TICKSIZE)


def mutiprocess_run(DAY, TICKSIZE):
    # tickers = get_keysHDF('F:/order_raw/{}.h5'.format(DAY))
    key = pd.read_csv('F:/TANG_NEW/ticker_0601.csv', header=None)
    tickers = key[1]

    for TICKER in tickers:

        print("start {} {}".format(TICKER,DAY))
        DOWNLIMIT,UPLIMIT = caculate_limit_price(TICKER, DAY)
        df0 = DataFeed(DAY, TICKER)
        ob0 = OrderBook(DAY, TICKER, DOWNLIMIT, UPLIMIT, TICKSIZE)
        df0.get_data()
        ob0.df0 = df0
        ob0.run()
        print("end {} {}".format(TICKER,DAY))


# def mutiprocess_run(DAY, TICKSIZE):
#     tickers = get_keysHDF('F:/order_raw/{}.h5'.format(DAY))
#     for TICKER in tickers:
#         try:
#             print("start {} {}".format(TICKER,DAY))
#             DOWNLIMIT,UPLIMIT = caculate_limit_price(TICKER, DAY)
#             df0 = DataFeed(DAY, TICKER)
#             ob0 = OrderBook(DAY, TICKER, DOWNLIMIT, UPLIMIT, TICKSIZE)
#             df0.get_data()
#             ob0.df0 = df0
#             ob0.run()
#             print("end {} {}".format(TICKER,DAY))
#         except:
#             print("error:{},{}".format(DAY,TICKER))


def every_day_run():
    current_day = int(date.today().strftime("%Y%m%d"))
    start_day = current_day
    end_day = current_day
    trading_days = utils.get_period_trading_days(start_day, end_day)
    TICKSIZE = 100
    if len(trading_days)>0: 
        for DAY in trading_days:
            mutiprocess_run(DAY,TICKSIZE)  

 

if __name__=="__main__":
#     main()
    main_multiprocess()
    #every_day_run()
