# encoding: UTF-8
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from decimal import Decimal

def read_xml(in_path):
    """读取并解析xml文件
           in_path: xml路径
           return: tree"""
    tree = ET.parse(in_path)
    return tree


def create_dict(root):
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


def get_period_trading_days(start_day, end_day):
    calendar_path = r"calendar.csv"
    calendar = pd.read_csv(calendar_path, header=None).set_index([0])[1]
    calendar = calendar[calendar == 1]
    calendar = calendar[calendar.index >= int(start_day)]
    trading_days = calendar[calendar.index <= int(end_day)].index
    return list(trading_days)

def change_hdf_to_csv(ticker, td):
    
    file_path = "/20data/orderbook/orderbook/2018/05/02/{}.h5".format(td[:4],td[4:6],td[6:],ticker)
    ticker_ob = pd.read_hdf(file_path,key=ticker)
    ticker_ob.to_csv("{}_{}.csv".format(ticker,td))
    
def generate_standard_snapshort_time():
    am_start_time = datetime(2018,1,1,9,30,0)
    am_end_time = datetime(2018,1,1,11,30,0)
    pm_start_time = datetime(2018,1,1,13,0,3)
    pm_end_time = datetime(2018,1,1,14,55,0)
    time_list = []
    ct = am_start_time
    delta = timedelta(seconds=3)
    while ct<pm_end_time:
        ct += delta
        if (am_end_time>ct>am_start_time) or (pm_end_time>ct>pm_start_time):
            time_list.append(ct.hour*10000000+ct.minute*100000+ct.second*1000)
    return time_list

def caculate_limit_price(ticker,day):
    column_name = ticker[:-3]
    data = pd.read_csv(r"/13data/PVData/preclose").set_index("Unnamed: 0")
    pre_index = list(data.index).index(int(day))
    pre_close = data[column_name].iloc[pre_index]
    up_limit = int(round(Decimal(pre_close*1.1)+Decimal(1e-6),2)*10000+Decimal(1e-6))
    down_limit = int(round(Decimal(pre_close*0.9)+Decimal(1e-6),2)*10000+Decimal(1e-6))
    
   
#     print(up_limit,down_limit)
    return down_limit,up_limit
    
if __name__=="__main__":
    generate_standard_snapshort_time()