# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:48:28 2022

@author: rtchg
"""
from PIL.ImageOps import expand
from sqlalchemy import create_engine
import sqlalchemy
from scipy.interpolate import interp1d
from dateutil.relativedelta import relativedelta, FR, MO
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import dataframe_image as dfi
import scipy
import csv
import openpyxl
from collections.abc import Iterable
from matplotlib.ticker import MultipleLocator
import statistics

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xlrd import xldate_as_datetime
import matplotlib
import subprocess
import matplotlib as mpl
import tushare
import scipy.io as sio
import seaborn as sns
import tushare as ts
import multiprocessing as mp
import dolphindb as ddb
import h5py
import zipfile
import datetime
import requests
import warnings
import os
import os.path
import time
import math
import sys
import re
import random
import json
import traceback
import pickle
import shutil
import libconf
import smtplib
import zipfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email import encoders

from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from pretty_html_table import build_table

from toolmodules.modules_constvars import *
from toolmodules.modules_downemail import *
from toolmodules.modules_generate_pdf import *
from toolmodules.modules_sendemail import *
from toolmodules.modules_ssh_server import *
from toolmodules.modules_upload_database import *
from toolmodules.cfidataapi.trade_info_loader import *

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def cprint(df):
    print(df.columns)
    print(df.index)
    print(df)


def hprint(df):
    print(len(df))
    print(df.head())

def htprint(df):
    print(len(df))
    print(df.head(1).T)


def tprint(df):
    print(len(df))
    print(df.tail())


def bprint(x='break'):
    print(x)
    breakpoint()


def bhprint(x):
    hprint(x)
    breakpoint()


def backup_directory(origin_path=None, target_path=None):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for origin_p__ in os.listdir(origin_path):
        if os.path.isdir(origin_path + origin_p__):
            backup_directory(origin_path=origin_path + origin_p__ + '/',
                         target_path=target_path + origin_p__ + '/')
        else:
            if not os.path.exists(target_path + origin_p__):
                try:
                    shutil.copy(origin_path + origin_p__, target_path + origin_p__)
                except:
                    print(origin_path + origin_p__)


def half_to_full(s):
    """
    Convert half-width characters to full-width characters.
    """
    full_width_s = ""
    for char in s:
        # 如果是半角字符（除了空格）则转换
        if 0x21 <= ord(char) <= 0x7e:
            full_width_s += chr(ord(char) + 0xfee0)
        elif ord(char) == 0x20:  # 对空格进行特殊处理
            full_width_s += chr(0x3000)  # 全角空格的Unicode编码
        else:
            full_width_s += char
    return full_width_s



class GetOrdersData():
    def __init__(self):
        pass

    def get_orders_by_pta_report(self, curdate, production):
        df_infor = pd.read_excel(f'{PLATFORM_PATH_DICT["t_path"]}ProcessedData/PTA/report/{curdate}/{curdate}.xlsx')
        df_infor = df_infor[['production', 'colo', '总订单', '订单数(SZ)', '订单数(SH)', '报错单数']]

        df_infor = df_infor[df_infor['production'] == production]
        if len(df_infor) == 1:
            df_infor = df_infor[['总订单', '报错单数']]
            order_num = df_infor['总订单'].sum() - df_infor['报错单数'].sum()
            return order_num
        else:
            df_infor = df_infor[['colo', '订单数(SZ)', '订单数(SH)', '报错单数']]
            df_infor['订单数(SZ)'] = df_infor.apply(lambda row: row['订单数(SZ)'] if 'sz' in row['colo'] else 0, axis=1)
            df_infor['订单数(SH)'] = df_infor.apply(lambda row: row['订单数(SH)'] if 'sh' in row['colo'] else 0, axis=1)
            order_num = df_infor['订单数(SH)'].sum() + df_infor['订单数(SZ)'].sum() - df_infor['报错单数'].mean()
            return order_num

    def get_orders_by_md_data(self, curdate, production, ret_ori_data=False):
        code_list = get_position(curdate, production)['SecuCode'].to_list()

        conlist = []
        for code in code_list:
            print(curdate, production, code)
            SecuCode_pref = 'SH' + code if code[0] == '6' else 'SZ' + code

            df_order = get_trade_info(SecuCode_pref, curdate)
            if df_order.empty: continue
            df_order = df_order[df_order['production'] == production]

            df_order_agg = trade_agg_func(df_order)
            if df_order_agg.empty: continue
            conlist.append(df_order_agg)

        df_order = pd.concat(conlist, axis=0)
        if ret_ori_data: return df_order

        df_order = df_order[['insid', 'volume', 'price', 'trade_volume', 'trade_turnover']]
        df_order['turnover'] = df_order['volume'] * df_order['price']
        order_value_list = np.array(df_order['turnover'].to_list())
        send_order = len(df_order)
        canceled_order = np.sum(df_order['volume'] > df_order['trade_volume'])
        total_order = send_order + canceled_order
        canceled_ratio = canceled_order / send_order
        volume_list = df_order['volume'].to_list()
        return order_value_list, send_order, canceled_order, total_order, canceled_ratio, volume_list

    def get_orders_by_worker(self, date, production):
        colo_name = production_2_colo(production)
        path_colo = f'{PLATFORM_PATH_DICT["z_path"]}ProcessedData/PTA/parsed_log/{colo_name}/{date}/'
        print(production, path_colo)
        if not os.path.exists(path_colo):
            return None
        alpha_dict = get_trading_accounts_paras_dict('productions', path_colo)
        if alpha_dict.get(production, None) is None:
            return None
        alpha_i = alpha_dict[production]

        df_worker = pd.read_csv(f'{path_colo}{colo_name}-worker-{date}.csv')
        if 'send_err_time' in df_worker.columns.to_list():
            df_worker = df_worker[df_worker["send_err_time"].isnull()]
        df_worker = df_worker[df_worker['stid'].astype('int') == alpha_i]

        if production in DUALCENTER_PRODUCTION:
            colo_name = colo_name.replace('sz', 'sh')
            path_colo = f'{PLATFORM_PATH_DICT["z_path"]}ProcessedData/PTA/parsed_log/{colo_name}/{date}/'
            if os.path.exists(f'{path_colo}{colo_name}-worker-{date}.csv'):
                df_worker_sh = pd.read_csv(f'{path_colo}{colo_name}-worker-{date}.csv')
                df_worker_sh = df_worker_sh[df_worker_sh['stid'] == alpha_i]
                df_worker_sh = df_worker_sh[df_worker_sh['insid'].apply(lambda x: x[2] == '6')]
                if 'send_err_time' in df_worker_sh.columns.to_list():
                    df_worker_sh = df_worker_sh[df_worker_sh["send_err_time"].isnull()]

                df_worker = df_worker[df_worker['insid'].apply(lambda x: x[2] != '6')]
                df_worker = pd.concat([df_worker, df_worker_sh], axis=0)
        if df_worker.empty:
            return None

        send_order = len(df_worker['canceled'])
        canceled_order = df_worker['canceled'].sum()
        total_order = send_order + canceled_order
        canceled_ratio = canceled_order / send_order

        df_worker = df_worker[~ df_worker['canceled']]
        df_worker['order_value'] = df_worker['volume'].astype('float') * df_worker['price'].astype('float')
        volume_list = df_worker['volume'].astype('float').to_list()
        order_value_list = np.array(df_worker['order_value'].to_list())

        return order_value_list, send_order, canceled_order, total_order, canceled_ratio, volume_list

    def get_orders_by_worker_exchange(self, date, production, exchange):
        colo_name = production_2_colo(production)
        path_colo = f'{PLATFORM_PATH_DICT["z_path"]}ProcessedData/PTA/parsed_log/{colo_name}/{date}/'
        print(production, path_colo)
        if not os.path.exists(path_colo):
            return None
        alpha_dict = get_trading_accounts_paras_dict('productions', path_colo)
        if alpha_dict.get(production, None) is None:
            return None
        alpha_i = alpha_dict[production]

        if production in DUALCENTER_PRODUCTION:
            if exchange == 'SZ':
                df_worker = pd.read_csv(f'{path_colo}{colo_name}-worker-{date}.csv')
                df_worker = df_worker[df_worker['stid'].astype('int') == alpha_i]
                df_worker = df_worker[df_worker['insid'].apply(lambda x: x[2] != '6')]
            else:
                colo_name = colo_name.replace('sz', 'sh')
                path_colo = f'{PLATFORM_PATH_DICT["z_path"]}ProcessedData/PTA/parsed_log/{colo_name}/{date}/'
                df_worker = pd.read_csv(f'{path_colo}{colo_name}-worker-{date}.csv')
                df_worker = df_worker[df_worker['stid'] == alpha_i]
                df_worker = df_worker[df_worker['insid'].apply(lambda x: x[2] == '6')]

        else:
            df_worker = pd.read_csv(f'{path_colo}{colo_name}-worker-{date}.csv')
            df_worker = df_worker[df_worker['stid'].astype('int') == alpha_i]
            if exchange == 'SZ':
                df_worker = df_worker[df_worker['insid'].apply(lambda x: x[2] != '6')]
            else:
                df_worker = df_worker[df_worker['insid'].apply(lambda x: x[2] == '6')]
        send_order = len(df_worker['canceled'])
        canceled_order = df_worker['canceled'].sum()
        total_order = send_order + canceled_order
        canceled_ratio = canceled_order / send_order

        df_worker['order_value'] = df_worker['volume'].astype('float') * df_worker['price'].astype('float')
        order_value_list = np.array(df_worker['order_value'].to_list())
        volume_list = df_worker['volume'].astype('float').to_list()

        return order_value_list, send_order, canceled_order, total_order, canceled_ratio, volume_list

    def get_orders_by_trades_files(self, date, product, exchange=None):
        df_trades = get_trades(date, product)

        if exchange == 'SZ':
            df_trades = df_trades[df_trades['SecuCode'].apply(lambda x: x[2] != '6')]
        elif exchange == 'SH':
            df_trades = df_trades[df_trades['SecuCode'].apply(lambda x: x[2] == '6')]

        order_tn = np.array((df_trades['Volume'] * df_trades['Price']).to_list())
        send_order = len(df_trades)
        canceled_order = 0
        total_order = send_order + canceled_order
        canceled_order_r = 0
        volume_list = df_trades['Volume'].to_list()

        return order_tn, send_order, canceled_order, total_order, canceled_order_r, volume_list


class GetPortfolioParasDaily():
    def __init__(self):
        pass

    def get_open_t0_flag(self, start_date='20240101', end_date='20241122', product='ZZJXZQ1'):
        infor_list = []
        for date in get_trading_days(start_date, end_date):
            path_file = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/adjust_cfg_paras/paras_config/' \
                        f'{date}_paras_config.cfg'
            with open(path_file, 'r') as f:
                config = libconf.load(f)
                
                infor_list.append([date, product in config['open_t0_mode_sh'][0]['target_list']])
        df_t0_mode = pd.DataFrame(infor_list, columns=['Date', 'Flag'])
        return df_t0_mode
    
    
    def get_slmt_ratio(self, start_date, end_date, product):
        infor_list = []
        for date in get_trading_days(start_date, end_date):
            if os.path.exists(f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/adjust_cfg_paras/paras_portfolio/{date}_portfolio_paras.csv'):
                df = pd.read_csv(f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/adjust_cfg_paras/paras_portfolio/{date}_portfolio_paras.csv')
                df = df[(df['Product'] == product) & (df['Parameter'] == 'reserve_weight_ratio')].head(1)
                infor_list.append([date, df['Value'].iloc[0]])
        df_slmt = pd.DataFrame(infor_list, columns=['Date', 'reserve_weight_ratio'])
        return df_slmt


class GetProductionInformation():
    """
        产品信息数据
    """

    def __init__(self):
        self.MulAcc_Main_dict = Production_MulAccDC_Main_Dict
        self.MulAcc_Main_dict.update(Production_MulAccZQ_Main_Dict)
        self.MulAcc_Main_dict.update(Production_MulAccYX_Main_Dict)

        self.MulAcc_FutureMain_Dict = Production_MulAccDC_Main_Future
        self.MulAcc_FutureMain_Dict.update(Production_MulAccZQ_Main_Future)
        self.MulAcc_FutureMain_Dict.update(Production_MulAccYX_Main_Future)

        self.Future2Trading_Replace_Dict = Dict_FutureNameReplace
        self.Trading2Future_Replace_Dict = Dict_ProductionName_Replace

        self.index_list = IndexNameList
        self.future_list = FutureNameList

        self.future_account_2_product = self.get_future_acc_2_product()
        self.simu_position_path = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/SimulationData/%s/new-comb/'

    def get_product_trading_main_name(self, prd):
        prd = self.MulAcc_FutureMain_Dict.get(self.MulAcc_Main_dict.get(prd, prd), prd)
        return self.Future2Trading_Replace_Dict.get(prd, prd)

    def get_product_future_main_name(self, prd):
        prd = self.MulAcc_FutureMain_Dict.get(self.MulAcc_Main_dict.get(prd, prd), prd)
        return self.Trading2Future_Replace_Dict.get(prd, prd)
    
    def get_product_type_infor(self):
        config_path = f'{PLATFORM_PATH_DICT["v_path"]}StockData/Trading/AnalysisTrading/config/'

        with open(os.path.join(config_path, "product_type.json"), "r") as f: prod_type_dict = json.loads(f.read())
        return prod_type_dict

    def get_stock_account_config(self, date):
        path_config = f'{PLATFORM_PATH_DICT["v_path"]}stock/Trading/AutoCapitalManager/StockConfigAccount/'
        df = pd.read_csv(f'{path_config}{date}_stock_account_config.csv', dtype='str', encoding='GBK')
        return df

    def get_future_accounts_auto_infor(self, drop_product_list=None):
        if drop_product_list is None: drop_product_list = []
        future_trans_products_dict = {}
        path_cfg = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/hedge_account_infor/future_trans_accounts.cfg'
        with open(path_cfg, 'r') as f:
            for line in f.readlines():
                prod_type, product_list = line.split('=')
                future_trans_products_dict[prod_type] = product_list.replace('(', '').replace(')', '').split()
        auto_future_acc = {
            acc: True for acc, product in zip(future_trans_products_dict['automated'] + future_trans_products_dict['automated_hs'], future_trans_products_dict['automated_product'] + future_trans_products_dict['nonautomated_product'])
            if product not in drop_product_list
        }
        return future_trans_products_dict, auto_future_acc

    def get_whether_exists_trades_files(self, curdate, product, trade_type='market', only_check_exists=False):
        if trade_type.lower() == 'market':
            file_path_pos = get_position(curdate, product, return_path=True)
            file_path_trades = get_trades(curdate, product, return_path=True)
        else:
            file_path_pos = get_position_simu(curdate, product, simu_type=trade_type, return_path=True)
            file_path_trades = get_trades_simu(curdate, product, simu_type=trade_type, return_path=True)

        if only_check_exists: return os.path.exists(file_path_pos) and os.path.exists(file_path_trades)
        return os.path.exists(file_path_pos) and os.path.exists(file_path_trades) and os.path.getsize(
            file_path_pos) > 0 and os.path.getsize(file_path_trades) > 0

    def get_product_trading_class_info(self):
        production_infor_list = []
        for prod in WinterFallProductionList:
            strategy = str(production_2_strategy(prod))
            index_origin = str(production_2_index(prod, type='long'))
            index = index_origin.split('.')[0]
            index = 'OTHERS' if not index else index
            production_infor_list.append({
                'Product': prod,
                'Index': index,
                'Strategy': strategy,
                'Class': index_origin + strategy,
                'Colo': production_2_colo(prod)
            })

        df_prod_infor = pd.DataFrame(production_infor_list)

        index_2_prodlist = {index: sorted(df_prod['Product'].to_list()) for index, df_prod in df_prod_infor.groupby('Index')}
        class_2_prodlist = {clss: sorted(df_prod['Product'].to_list() ) for clss, df_prod in df_prod_infor.groupby('Class')}
        colo_2_prodlist = {colo: sorted(df_prod['Product'].to_list()) for colo, df_prod in df_prod_infor.groupby('Colo')}

        return index_2_prodlist, class_2_prodlist, colo_2_prodlist

    def match_product_list_with_class_info(self, curdate, filter_dict, class_2_prodlist, add_priority):
        if filter_dict[curdate]['all_prod']: target_production_list = deepcopy(WinterFallProductionList)
        else: target_production_list = []

        add_target_production_list = []
        for clss in filter_dict[curdate]['class']: add_target_production_list += class_2_prodlist.get(clss, [])

        drop_target_production_list = []
        for clss in filter_dict[curdate]['class_drop']: add_target_production_list += class_2_prodlist.get(clss, [])

        target_prod_list = list(set(target_production_list + add_target_production_list) - set(drop_target_production_list))
        if add_priority:
            target_prod_list = list((set(target_prod_list) - set(filter_dict[curdate]['drop_plist'])) | set(filter_dict[curdate]['add_plist']))
        else:
            target_prod_list = list(set(target_prod_list + filter_dict[curdate]['add_plist']) - set(filter_dict[curdate]['drop_plist']))

        return target_prod_list

    def get_future_acc_2_product(self):
        df = pd.read_csv(f'{DATA_PATH_SELF}product_hedge_account_infor.csv', dtype='str')
        return {acc: prod for acc, prod in zip(df['Account'], df['Product'])}


class GetBaseData(GetProductionInformation):
    """
        读取交易数据
    """

    def __init__(self):
        super().__init__()

    def get_bm_index_weight_distribution(self, curdate, proportion_dict, dict_index_stock_list=None):
        if dict_index_stock_list is None:
            dict_index_stock_list = {
                index_name: get_stockdaily_indexweight(curdate, index_name).reset_index()[['SecuCode', 'IndexWeight']]
                for index_name in proportion_dict
            }

        conlist = []
        for index_name, prop in proportion_dict.items():
            conlist.append(dict_index_stock_list[index_name].copy(deep=True).set_index('SecuCode') * float(prop))

        df_weight = pd.concat(conlist, axis=0).fillna(0).sum(axis=1).to_frame().reset_index()
        df_weight = df_weight.rename({0: 'IndexWeight'}, axis='columns')
        return df_weight

    def get_order_nums(self, date, product):
        df = pd.read_excel(f'{PLATFORM_PATH_DICT["z_path"]}ProcessedData/PTA/report/{date}/{date}.xlsx')
        if isinstance(product, str): product = [product]
        df = df[df['production'].isin(product)]
        df['总订单'] *= 1 + df['撤单数比'].fillna(0) / 100
        if df.empty: return 0
        return int(df['总订单'].fillna(0).mean())

    def get_stamp_tax_ratio(self, date):
        if int(date) < 20230828:
            stampr = 0.001
        else:
            stampr = STAMP_RATE
        return stampr

    def get_commision_and_order_fee(self, date, product):
        if product == 'ZS9B':
            return 0.00011, 0
        df = pd.read_excel(f'{DATA_PATH_SELF}{date}/report.xls')
        df = df[df['内部代码'] == product].reset_index(drop=True)
        if df.empty:
            return 0.00012, 0

        commision_name = [col for col in df.columns.to_list() if col.startswith('佣金')][0]
        commision = df[commision_name].fillna('万1.2').iloc[0].replace('万', '').replace('..', '.').replace('，', ',')
        if ',' in commision:
            commision = round(float(commision.split(',')[0].strip()) / 10000, 7)
            order_fee = 1
        else:
            commision = round(float(commision.split(',')[0].strip()) / 10000, 7)
            if '流量费' not in df.columns.to_list():
                order_fee = 0
            else:
                order_fee = df['流量费'].fillna(0).iloc[0]

        if product == 'DC19':
            if (int(date) >= 20230830) & (int(date) <= 20230922):
                order_fee = 1

        if product == 'JT1':
            if (int(date) >= 20230830) & (int(date) <= 20230922):
                order_fee = 1

            if int(date) < 20230830:
                commision = {'Long': 0.00013, 'Short': 0.00019}
            # else:
            #     commision = 0.000115

        return commision, order_fee

    def get_broker_fee_and_profit(self, start_date, end_date, product=None, colo=None, save_file=False):
        if isinstance(product, str): product = [product]

        conlist = []
        for date in get_trading_days(start_date, end_date):
            df = pd.read_excel(f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/{date}/BrokerFeeSummary/{date}_daily_profit.xlsx')
            if product is not None: df = df[df['产品名'].isin(product)]
            conlist.append(df)
        df = pd.concat(conlist, axis=0)

        if save_file: df.to_csv(f'{LOG_TEMP_PATH}{start_date}-{end_date}_{"-".join(product)}.csv', index=False, encoding='GBK')
        return df
    
    def get_valuation_sheet_filepath(self, date, product):
        product_name = PRODUCTION_2_PRODUCT_NAME.get(product)
        hosting = PRODUCTION_2_VALUATION_SHEET_PATH.get(product)
        evalue_sheet_path = f'{VALUATION_STATEMENT_PATH}{date}/{hosting}/'
        filepath = list(Path(evalue_sheet_path).glob(f'*{product_name}*'))
        if not filepath: return None
        return filepath, hosting

    def get_valuation_sheet_data(self, date, product=None, ret_df_data=False, product_name_dict=None):
        if product_name_dict is None: product_name_dict = ProductionName_2_Production
        df_netvalue = pd.read_excel(MARKET_SUMMARY_PATH % date)
        df_netvalue['Production'] = df_netvalue['产品名称'].apply(lambda x: product_name_dict.get(x, x))

        if ret_df_data:
            df_netvalue = df_netvalue[df_netvalue['产品名称'].isin(product_name_dict.keys())]
            if product is not None:
                df_netvalue = df_netvalue[df_netvalue['Production'] == Production_NetValueMatch.get(product, product)]
            return df_netvalue

        if product is not None:
            df_netvalue = df_netvalue[df_netvalue['Production'] == Production_NetValueMatch.get(product, product)]
            date_net_value = df_netvalue['日期'].astype('float').iloc[0]
            net_value = df_netvalue['资产净值'].astype('float').iloc[0]
            net_value_unit = df_netvalue['单位净值'].astype('float').iloc[0]
            net_value_share = df_netvalue['实收资本'].astype('float').iloc[0]
            future_margin = df_netvalue['期货及衍生品交易保证金'].astype('float').iloc[0]

            try: bank_cash = df_netvalue['银行存款'].astype('float').iloc[0]
            except: bank_cash = 0
            try: cash = df_netvalue['现金类资产'].astype('float').iloc[0]
            except: cash = 0

            if product == 'DC19':
                if date == '20230914':
                    net_value = 350215117.18
                    net_value_unit = 1.2169
                elif date == '20231113':
                    net_value = 391405741.35
                    net_value_unit = 1.3286
            if product == 'HXDC21':
                if date == '20250212':
                    net_value = 70205255.17
                    net_value_unit = 7.3005
            if (product == 'YPA') or (product == 'GDYPA'):
                if date == '20230914':
                    net_value = 56877240.21
                    net_value_unit = 0.678

            return {
                'product': product,
                'net_value_date': date_net_value,
                'net_value': net_value,
                'net_value_unit': net_value_unit,
                'net_value_share': net_value_share,
                'margin': future_margin,
                'bank_cash': bank_cash,
                'cash': cash,
            }
        return df_netvalue.to_dict(orient='records')

    def get_swap_details(self, file_path=None, format_mode=None):
        if file_path is None:
            file_path = SWAP_DETAILS_PATH
            if not os.path.exists(file_path):
                print(file_path, '不存在!!')
                file_path = DATA_PATH_SELF

        file_path += '收益互换明细汇总(新).xlsx'
        dict_swap_infor = pd.read_excel(file_path, sheet_name=None)
        if format_mode is not None:
            conlist = []
            for key in dict_swap_infor:
                df_swap_infor = dict_swap_infor[key].copy(deep=True)
                if not df_swap_infor.empty:
                    conlist.append(df_swap_infor)
            df_swap_infor = pd.concat(conlist, axis=0).dropna(how='all', axis=1)
            df_swap_infor = df_swap_infor[~df_swap_infor['合约标的'].replace(['nan', 'NaN'], np.nan).isna()]
            df_swap_infor = df_swap_infor[[col for col in ['产品名称', '合约标的', '数量', '起始日期', '到期日期', '期末时间', '利率', '期初价格', '期末价格'] if col in df_swap_infor.columns.to_list()]]

            df_swap_infor = df_swap_infor[df_swap_infor['产品名称'].isin(ProductionName_2_Production.keys())]

            df_swap_infor['产品名称'] = df_swap_infor['产品名称'].apply(lambda x: ProductionName_2_Production[x])
            if '利率' in df_swap_infor.columns.to_list():
                df_swap_infor = df_swap_infor[df_swap_infor['利率'].isna()].dropna(how='all', axis=1)
            df_swap_infor['起始日期'] = df_swap_infor['起始日期'].apply(lambda x: format_date_2_str(x))
            df_swap_infor['到期日期'] = df_swap_infor['到期日期'].apply(lambda x: format_date_2_str(x))
            df_swap_infor['期末时间'] = df_swap_infor['期末时间'].apply(lambda x: format_date_2_str(x))
            if format_mode == 'format2pos':
                return df_swap_infor

            if format_mode == 'format2trades':
                conlist_infor_list = []
                for product, fut_name, volume, startdate, enddate, endtradingdate, startprice, endprice in df_swap_infor.values:
                    conlist_infor_list.append([product, fut_name, volume, 0, startdate, startprice])
                    if str(endprice).lower() != 'nan':
                        conlist_infor_list.append([product, fut_name, volume, 1, endtradingdate, endprice])
                df_swap_trading = pd.DataFrame(
                    conlist_infor_list, columns=['Product', 'SecuCode', 'Volume', 'LS', 'Date', 'Price'])

                df_swap_trading['Volume'] *= -1
                return df_swap_trading
            else:
                raise 'ValueError'

        return dict_swap_infor

    def get_option_details(self, file_path=None, format_mode=None):
        if file_path is None:
            file_path = SWAP_DETAILS_PATH
            if not os.path.exists(file_path):
                print(file_path, '不存在!!')
                file_path = DATA_PATH_SELF

        file_path += '场外期权明细汇总.xlsx'
        dict_option_infor = pd.read_excel(file_path, sheet_name=None)
        if format_mode is not None:
            conlist = []
            for key in dict_option_infor:
                df_infor = dict_option_infor[key].copy(deep=True)
                if not df_infor.empty:
                    conlist.append(df_infor)
            df_infor = pd.concat(conlist, axis=0).dropna(how='all', axis=1)
            df_infor = df_infor[~df_infor['合约标的'].replace(['nan', 'NaN'], np.nan).isna()]
            df_infor = df_infor[[
                col for col in ['产品名称', '合约标的', '数量', '起始日期', '到期日期', '期末时间', '利率', '期初价格', '期末价格']
                if col in df_infor.columns.to_list()]]
            df_infor = df_infor[df_infor['产品名称'].isin(ProductionName_2_Production.keys())]

            df_infor['产品名称'] = df_infor['产品名称'].apply(lambda x: ProductionName_2_Production[x])
            if '利率' in df_infor.columns.to_list():
                df_infor = df_infor[df_infor['利率'].isna()].dropna(how='all', axis=1)

            df_infor['合约标的'] = df_infor['合约标的'].apply(lambda x: expand_stockcode(x))
            df_infor['合约标的'] = df_infor['合约标的'].apply(lambda x: IndexSecuCode_2_IndexName_FutureName[x][1])

            df_infor['起始日期'] = df_infor['起始日期'].apply(lambda x: format_date_2_str(x))
            df_infor['到期日期'] = df_infor['到期日期'].apply(lambda x: format_date_2_str(x))
            df_infor['期末时间'] = df_infor['期末时间'].apply(lambda x: format_date_2_str(x))
            if format_mode == 'format2pos':
                return df_infor
            else:
                raise 'ValueError'

        return dict_option_infor

    def get_details_future_margin_data(self, curdate, product=None):
        df_hedge_margin = pd.read_csv(f'{PLATFORM_PATH_DICT["t_path"]}HEDGE/indexPatch/{curdate}.csv')
        df_hedge_margin['Product'] = df_hedge_margin.apply(
            lambda row: self.future_account_2_product.get(str(row['Account']), row['Product'])
            if (str(row['Product']).lower() == 'xxx') else row['Product'], axis=1).replace(Dict_FutureNameReplace)
        if product is not None:
            if isinstance(product, str): product = [product]
            df_hedge_margin = df_hedge_margin[df_hedge_margin['Product'].isin(product)]
        return df_hedge_margin[['Product', 'Account', 'UpdateTime', 'Capital', 'Margin', 'Withdraw', 'Available', 'Pos_Long', 'Pos_Short']]

    def get_details_future_position_data(self, curdate, data_type='df', product=None):
        df_hedge_position = pd.read_csv(f'{PLATFORM_PATH_DICT["t_path"]}HEDGE/indexPatch/{curdate}.csv')
        df_hedge_position['Product'] = df_hedge_position.apply(
            lambda row: self.future_account_2_product.get(str(row['Account']), row['Product'])
            if (str(row['Product']).lower() == 'xxx') else row['Product'], axis=1).replace(Dict_FutureNameReplace)
        if product is not None:
            if isinstance(product, str): product = [product]
            df_hedge_position = df_hedge_position[df_hedge_position['Product'].isin(product)]

        pos_columns = [
            col for col in df_hedge_position.columns
            if col.startswith('I') and (col.endswith('Long') or col.endswith('Short'))
        ]
        df_hedge_position = pd.melt(df_hedge_position, id_vars='Product', value_vars=pos_columns)
        df_hedge_position = df_hedge_position.groupby(['Product', 'variable'])[['value']].sum().reset_index()
        df_hedge_position['Direction'] = df_hedge_position['variable'].apply(lambda x: x.split('_')[1])
        df_hedge_position['Instrument'] = df_hedge_position['variable'].apply(lambda x: x.split('_')[0])
        df_hedge_position = pd.pivot_table(
            df_hedge_position, index=['Product', 'Instrument'], columns='Direction', values='value').reset_index()

        df_hedge_position['FutureName'] = df_hedge_position['Instrument'].str[:2]
        if data_type == 'df':
            return df_hedge_position

        col_list = df_hedge_position.columns.to_list()
        if 'Short' not in col_list: df_hedge_position['Short'] = 0
        if 'Long' not in col_list: df_hedge_position['Long'] = 0

        df_hedge_position['Short'] += df_hedge_position['Long']
        if data_type == 'FutureName':
            return pd.DataFrame([
                [
                    product,
                    df.groupby('FutureName')['Short'].sum().to_dict()
                ] for product, df in df_hedge_position.groupby('Product')
            ], columns=['Product', 'HedgePos'])

        if data_type == 'Instrument':
            return pd.DataFrame([
                [
                    product,
                    df.groupby('Instrument')['Short'].sum().to_dict()
                ] for product, df in df_hedge_position.groupby('Product')
            ], columns=['Product', 'HedgePosDict'])

        return pd.DataFrame([
            [
                product,
                df.groupby('FutureName')['Short'].sum().to_dict(),
                df.groupby('Instrument')['Short'].sum().to_dict()
            ] for product, df in df_hedge_position.groupby('Product')
        ], columns=['Product', 'HedgePos', 'HedgePosDict'])

    def get_trades_data(self, date, product, backtest_date=None, suffix_name=''):
        if SimuDict.get(product, None) is not None:
            if SimuDict[product]['type'] == 'simu':
                df_trades = get_trades_simu(date, SimuDict[product].get('trading', product), filepath=SimuDict[product].get('trading_path'))
            elif SimuDict[product]['type'] == 'bktest':
                back_test_path = SimuDict[product].get('trading_path')
                if backtest_date is not None: back_test_path = back_test_path % backtest_date
                df_trades = get_trades_simu(date, SimuDict[product].get('trading', product), filepath=back_test_path)
            else: raise ValueError
        else: df_trades = get_trades(date, product, suffix_name=suffix_name)
        return df_trades

    def get_position_close(self, date, product, backtest_date=None, pre_mode=False):
        if product in ProductionList_AlphaShort_Short:
            df_position = get_position(date, product)
            if pre_mode:
                df_position = df_position[['SecuCode', 'PreCloseVolume']]
            else:
                df_position = df_position[['SecuCode', 'Volume']]
                df_position['Volume'] *= -1
                df_position = get_destpos_alpha_short_ti8_recall(date, df_position, product)
                df_position = df_position.rename({"Volume": 'PreCloseVolume'}, axis='columns')
        elif SimuDict.get(product, None) is not None:
            if SimuDict[product]['type'] == 'simu':
                df_position = get_position_simu(date, SimuDict[product].get('trading', product))
            elif SimuDict[product]['type'] == 'bktest':
                back_test_path = SimuDict[product].get('trading_path')
                if backtest_date is not None: back_test_path = back_test_path % backtest_date
                df_position = get_position_backtest(date, SimuDict[product]['trading'], filepath=back_test_path)
            else: raise ValueError
            if pre_mode:
                print("读取文件 ['SecuCode', 'PreCloseVolume']")
                df_position = df_position[['SecuCode', 'PreCloseVolume']]
            else:
                print("读取文件 ['SecuCode', 'Volume']")
                df_position = df_position[['SecuCode', 'Volume']].rename({"Volume": 'PreCloseVolume'}, axis='columns')
        else:
            df_position = get_position(date, product)
            if pre_mode: df_position = df_position[['SecuCode', 'PreCloseVolume']]
            else: df_position = df_position[['SecuCode', 'Volume']].rename({"Volume": 'PreCloseVolume'}, axis='columns')
        return df_position

    def get_quota_close(self, date, product, bar=8):
        if bar == 8:
            file_path = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/d0_destpos/'
        else:
            file_path = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/d0_destpos_5m/'
        if product in ProductionList_AlphaShort_Short:
            df_quota_pos = get_alpha_quota_bar(date, product, bar, filepath=file_path, short_mode=True, recall_mode=True, bar_mode=bar).rename(
                {"Volume": 'PreCloseVolume'}, axis='columns')
        elif SimuDict.get(product, None) is not None:
            df_quota_pos = get_alpha_quota_bar(date, SimuDict[product].get('quota', product), bar, filepath=SimuDict[product].get('quota_path', file_path), bar_mode=bar).rename(
                {"Volume": 'PreCloseVolume'}, axis='columns')
        else:
            df_quota_pos = get_alpha_quota_bar(date, product, bar, filepath=file_path, bar_mode=bar).rename(
                {"Volume": 'PreCloseVolume'}, axis='columns')
        return df_quota_pos

    def get_quota_trade_diff_volume(self, date, predate, product, filepath=None, mode='dest-pos', ret_df_mode='Diff', bar_mode=8, drop_quota_list=None):
        if product in ProductionList_AlphaShort_Short:
            df_trade_volume = get_simulate_diff_volume_mat_by_quota(date, predate, product, filepath=filepath, mode=mode, ret_df_mode=ret_df_mode, short_mode=True, bar_mode=bar_mode, drop_quota_list=drop_quota_list)
        elif SimuDict.get(product, None) is not None:
            df_trade_volume = get_simulate_diff_volume_mat_by_quota(date, predate, SimuDict[product].get('quota', product), filepath=filepath, mode=mode, ret_df_mode=ret_df_mode, bar_mode=bar_mode, drop_quota_list=drop_quota_list)
        else:
            df_trade_volume = get_simulate_diff_volume_mat_by_quota(date, predate, product, filepath=filepath, mode=mode, ret_df_mode=ret_df_mode, bar_mode=bar_mode, drop_quota_list=drop_quota_list)

        return df_trade_volume

    def get_future_trades_pnl(self, start_date, end_date, product=None, return_mode='df'):
        if isinstance(product, str):
            product = [product]

        conlist = []
        for date in get_trading_days(start_date, end_date):
            df = pd.read_csv(f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/future_trades_pnl/{date}_FutureTradesPnLSummary.csv', dtype={'Date': 'str'})
            if product is not None:
                df = df[df['Product'].isin(product)]
            conlist.append(df)
        df = pd.concat(conlist, axis=0).rename({'FutTradesPnL': 'FutureTradingPnL'}, axis='columns')

        if return_mode == 'df':
            return df
        elif return_mode == 'pnl':
            return df['FutureTradingPnL'].sum()
        else: raise ValueError
    
    def get_future_basis_data_daily(self, curdate):
        conlist = []
        for file_path in Path(
                f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/FutureBasisInformation/{curdate}/').glob(
                f'{curdate}_Compare_Contract_*.csv'):
            conlist.append(pd.read_csv(file_path))
        
        if conlist:
            df = pd.concat(conlist, axis=0)
        else:
            df = None
        return df
    
    def get_close_position_weight_all_productions(self, curdate):
        output_dir = f'{PLATFORM_PATH_DICT["v_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'
        df = pd.read_csv(f'{output_dir}{curdate}_position_weight_all_productions.csv')
        df['SecuCode'] = df['SecuCode'].apply(lambda x: expand_stockcode(x))
        return df

    def save_simu_position_2_marketing(self, date, product_list):
        simu_path = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/SimulationData/{date}/new-comb/'
        if not os.path.exists(simu_path): os.makedirs(simu_path)
        if isinstance(product_list, str): product_list = [product_list]

        tsl = TransferServerLocal()
        for product in product_list:
            tsl.download_file(
                server_path=f'/home/jumper/temp/{date}/new-comb/{date}_position_simu-{product}.txt',
                local_path=f'{simu_path}{date}_position_simu-{product}.txt')

            df_pos = get_position_simu(date, product)
            tradingdir = f'{PLATFORM_PATH_DICT["z_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{date}/'
            df_pos.to_csv(f'{tradingdir}{date}_position-{product}.txt', sep=',', index=False, header=False, quoting=csv.QUOTE_NONE)

    def read_alpha_simu(self, date, alpha_name):
        path_alpha_simu = os.path.join(f'{PLATFORM_PATH_DICT["v_path"]}StockData/Trading/AlphasimPnL/', f'{alpha_name}.csv')
        if os.path.exists(path_alpha_simu):
            df_alpha_simu = pd.read_csv(path_alpha_simu, dtype={'Date': 'str'})[['Date', 'pnl', 'hld_pnl']]
            df_alpha_simu = df_alpha_simu[df_alpha_simu['Date'] == date].set_index('Date')
            df_alpha_simu['hld_pnl'] /= 1e7
            df_alpha_simu['pnl'] -= df_alpha_simu['hld_pnl']
            return {
                'asim-trds': round(df_alpha_simu['pnl'].sum() * 1e4, 3),
                'asim-pos': round(df_alpha_simu['hld_pnl'].sum() * 1e4, 3),
            }
        else: return {}


class ReadAlphaMatData():
    def process_alpha_mat_data(self, AlphaMat):
        NameList = list(AlphaMat.dtype.names)
        AlphaDict = {NameList[i]: AlphaMat[NameList[i]][0][0] for i in range(len(NameList))}
        TempNameL = [NameList[i] for i in range(len(AlphaDict)) if AlphaDict[NameList[i]].dtype == 'O']
        for curname in TempNameL:
            if np.shape(AlphaDict[curname])[1] == 1:
                AlphaDict[curname] = np.asarray([
                    str(AlphaDict[curname][i][0][0])
                    for i in range(np.shape(AlphaDict[curname])[0])
                    if len(AlphaDict[curname][i][0]) > 0], dtype=np.str_)
            elif np.shape(AlphaDict[curname])[0] == 1:
                AlphaDict[curname] = np.asarray([str(AlphaDict[curname][0][i][0])
                                                 for i in range(np.shape(AlphaDict[curname])[1]) if
                                                 len(AlphaDict[curname][0][i]) > 0], dtype=np.str_)
            else:
                AlphaDict[curname] = np.asarray([[str(AlphaDict[curname][i][j][0])
                                                  for j in range(np.shape(AlphaDict[curname])[1])] for i in
                                                 range(np.shape(AlphaDict[curname])[0])], dtype=np.str_)
        return AlphaDict

    def get_alpha_dict(self, curday, alpha_tag, DailyAlphaPath=None):
        alpha_path = os.path.join(PLATFORM_PATH_DICT["v_path"], 'Data/StockData/InterdayAlpha/ProductionInfo/ConfigInfo', curday, alpha_tag + '.mat')
        if DailyAlphaPath is not None:
            alpha_path = os.path.join(DailyAlphaPath, alpha_tag + '.mat')
        if not os.path.exists(alpha_path):
            alpha_path = os.path.join(PLATFORM_PATH_DICT["v_path"], 'Data/StockData/InterdayAlpha/ProductionInfo/ConfigInfo', curday, alpha_tag + '.mat')
            if not os.path.exists(alpha_path):
                alpha_path = os.path.join(PLATFORM_PATH_DICT["v_path"], 'Data/StockData/InterdayAlpha/ProductionInfo/ConfigInfo', curday, alpha_tag + '.mat')
                if not os.path.exists(alpha_path):
                    print(f'{alpha_path}: NOT FOUND !!')
                    return None
        alpha_mat = sio.loadmat(alpha_path)['AF']
        alpha_dict = self.process_alpha_mat_data(alpha_mat)

        alpha_dict['DateL'] = np.squeeze(alpha_dict['DateL'])

        Alpha_SecuCodeL = alpha_dict['SecuCodeL']
        sort_idx = np.argsort(Alpha_SecuCodeL)
        SecuCodeL_sorted = np.sort(Alpha_SecuCodeL)
        alpha_dict['SecuCodeL'] = SecuCodeL_sorted
        for curkey, curvalue in alpha_dict.items():
            if len(np.shape(curvalue)) > 1:
                if np.shape(curvalue)[1] > 4000:
                    alpha_dict[curkey] = curvalue[:, sort_idx]
        return alpha_dict


def get_indexret_1min(curdate, index_name='ZZ500', return_type='array'):
    prev_closeprice = get_indexprice(curdate, curdate, IndexName=index_name)['PrevClosePrice'].sum()
    path_index = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/Intra_data/{curdate}/{curdate}_{index_name}_1MIN.csv'
    if os.path.exists(path_index):
        df_index_1min = pd.read_csv(path_index)
        index_ret = np.array(df_index_1min.sort_values('time')['close'].to_list()) / prev_closeprice - 1
    else:
        path_index = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/Intra_data/{curdate}/{curdate}_ZZ500_1MIN.csv'
        df_index_1min = pd.read_csv(path_index)
        index_ret = np.array(df_index_1min.sort_values('time')['close'].to_list()) / prev_closeprice - 1
        index_ret *= np.nan

    if return_type == 'array':
        return index_ret
    else:
        return pd.DataFrame(index_ret, columns=[index_name], index=sorted(df_index_1min['time'].unique())).T


def get_indexret_1min_dict(curdate, return_type='array'):
    index_ret_dict = {
        'ZZ500': get_indexret_1min(curdate, index_name='ZZ500', return_type=return_type),
        'ZZA500': get_indexret_1min(curdate, index_name='ZZA500', return_type=return_type),
        'HS300': get_indexret_1min(curdate, index_name='HS300', return_type=return_type),
        'ZZ800': get_indexret_1min(curdate, index_name='ZZ800', return_type=return_type),
        'ZZ1000': get_indexret_1min(curdate, index_name='ZZ1000', return_type=return_type),
        'ZZ2000': get_indexret_1min(curdate, index_name='ZZ2000', return_type=return_type),
        'ZZHL': get_indexret_1min(curdate, index_name='ZZHL', return_type=return_type),
        'ZZQZ': get_indexret_1min(curdate, index_name='ZZQZ', return_type=return_type),
    }
    return index_ret_dict


def get_stockdaily_indexweight_dict(curdate):
    dict_index_stock_dict = {
        'ZZ500': get_stockdaily_indexweight(curdate, 'ZZ500').reset_index()[['SecuCode', 'IndexWeight']],
        'ZZA500': get_stockdaily_indexweight(curdate, 'ZZA500').reset_index()[['SecuCode', 'IndexWeight']],
        'HS300': get_stockdaily_indexweight(curdate, 'HS300').reset_index()[['SecuCode', 'IndexWeight']],
        'ZZ800': get_stockdaily_indexweight(curdate, 'ZZ800').reset_index()[['SecuCode', 'IndexWeight']],
        'ZZ1000': get_stockdaily_indexweight(curdate, 'ZZ1000').reset_index()[['SecuCode', 'IndexWeight']],
        'ZZ2000': get_stockdaily_indexweight(curdate, 'ZZ2000').reset_index()[['SecuCode', 'IndexWeight']],
        'ZZHL': get_stockdaily_indexweight(curdate, 'ZZHL').reset_index()[['SecuCode', 'IndexWeight']],
        'ZZQZ': get_stockdaily_indexweight(curdate, 'ZZQZ').reset_index()[['SecuCode', 'IndexWeight']],
    }
    return dict_index_stock_dict


def pool_starmap_multiprocessing(pool_func, paras_list, process_num=30, remove_none=False):
    pool = mp.Pool(process_num)
    df_list = pool.starmap(pool_func, paras_list)
    pool.close()
    pool.join()
    
    if remove_none: df_list = [_ for _ in df_list if _ is not None]

    return df_list


def rename_df_columns_name(df, mode='sum/max/min', precision=2, return_mode='str'):
    if mode == 'sum/max':
        return df.rename(
            {col: f'{col}: {round(df[col].sum(), 2)}/{round(df[col].max(), 2)}'
             for col in df.columns.to_list()}, axis='columns')
    elif mode == 'sum/max/min':
        return df.rename(
            {col: f'{col}: {round(df[col].sum(), 2)}/{round(df[col].max(), 2)}/{round(df[col].min(), 2)}'
             for col in df.columns.to_list()}, axis='columns')
    elif mode == 'max/min':
        return df.rename(
            {col: f'{col}: {round(df[col].max(), 2)}/{round(df[col].min(), 2)}'
             for col in df.columns.to_list()}, axis='columns')
    elif mode == 'last/max/min/mean/median':
        return df.rename(
            {col: f'{col}: {round(df[col].iloc[-1], precision)}/'
                  f'{round(df[col].max(), precision)}/'
                  f'{round(df[col].min(), precision)}/'
                  f'{round(df[col].mean(), precision)}/'
                  f'{round(df[col].median(), precision)}'
             for col in df.columns.to_list()}, axis='columns')
    elif mode == 'last/mean':
        return df.rename(
            {col: f'{col}: {round(df[col].iloc[-1], precision)}/{round(df[col].mean(), precision)}'
             for col in df.columns.to_list()}, axis='columns')
    elif mode == 'last':
        return df.rename(
            {col: f'{col}: {round(df[col].iloc[-1], precision)}'
             for col in df.columns.to_list()}, axis='columns')
    elif mode == 'first/last':
        return df.rename(
            {col: f'{col}: \n{round(df[col].iloc[0], precision)}/{round(df[col].iloc[-1], precision)}'
             for col in df.columns.to_list()}, axis='columns')
    elif mode == '4':
        return df.rename({
            col: f'{col}: {calculate_sharpe_annual_maxdd(df[col].dropna().to_list(), precision=precision, return_mode=return_mode)}'
            for col in df.columns.to_list()}, axis='columns')


def get_conversed_bonus_xd_details(curdate=None, production=None, quota_stock_list=None, curdate_only=True):
    df_sql = pd.read_sql("select * from AShareEXRightDividendRecord where EX_DATE>= '%s'" % curdate,
                         create_engine("mssql+pymssql://ht:huangtao@dbs.cfi:1433/WDDB"))

    df_sql['SecuCode'] = df_sql['S_INFO_WINDCODE'].apply(lambda x: x.split('.')[0])
    df_sql['Multi_Ratio'] = np.maximum(df_sql['CONVERSED_RATIO'], df_sql['BONUS_SHARE_RATIO'])
    df_sql = df_sql[df_sql['Multi_Ratio'] > 0]
    
    if curdate_only:
        df_sql = df_sql[df_sql['EX_DATE'].apply(lambda x: x == curdate)]
    df_sql = df_sql.rename({'EX_DATE': 'Date', 'EX_DESCRIPTION': 'Details'}, axis='columns')

    if not df_sql.empty:
        df_sql = df_sql[['SecuCode', 'Date', 'Multi_Ratio', 'Details']]
    else:
        df_sql = pd.DataFrame(columns=['SecuCode', 'Date', 'Multi_Ratio', 'Details'])

    if production is None:
        if quota_stock_list is None:
            return df_sql
        else:
            df_sql = df_sql[df_sql['SecuCode'].isin(quota_stock_list)]
            return df_sql
    else:
        stockquota = pd.read_csv(f'{PLATFORM_PATH_DICT["z_path"]}Trading/StockQuota_{production}.csv', header=None)
        stockquota.columns = ['SecuCode', 'EVolume', 'tmp']
        stockquota['SecuCode'] = stockquota['SecuCode'].apply(lambda x: expand_stockcode(x))
        stockquota = stockquota.groupby('SecuCode').sum().reset_index()

        df_sql = df_sql[df_sql['SecuCode'].isin(stockquota['SecuCode'].to_list())]
        return df_sql


def get_maximum_drawdown(list_ret):

    return - np.maximum.accumulate(list_ret) + np.array(list_ret)


def calculate_cancel_ratio_by_worker(date, production):
    colo_name = production_2_colo(production)
    path_colo = f'{PLATFORM_PATH_DICT["z_path"]}ProcessedData/PTA/parsed_log/{colo_name}/{date}/'
    print(production, path_colo)
    if not os.path.exists(path_colo):
        return np.nan
    alpha_dict = get_trading_accounts_paras_dict('productions', path_colo)
    if alpha_dict.get(production, None) is None:
        return None, None, None, None, None
    alpha_i = alpha_dict[production]

    df_worker = pd.read_csv(f'{path_colo}{colo_name}-worker-{date}.csv')
    df_worker = df_worker[df_worker['stid'] == alpha_i]

    if production in DUALCENTER_PRODUCTION:
        colo_name = colo_name.replace('sz', 'sh')
        path_colo = f'{PLATFORM_PATH_DICT["z_path"]}ProcessedData/PTA/parsed_log/{colo_name}/{date}/'
        df_worker_sh = pd.read_csv(f'{path_colo}{colo_name}-worker-{date}.csv')
        df_worker_sh = df_worker_sh[df_worker_sh['stid'] == alpha_i]
        df_worker_sh = df_worker_sh[df_worker_sh['insid'].apply(lambda x: x[2] == '6')]
        df_worker = df_worker[df_worker['insid'].apply(lambda x: x[2] != '6')]
        df_worker = pd.concat([df_worker, df_worker_sh], axis=0)

    if df_worker.empty:
        return 0
    send_order = len(df_worker['canceled'])
    canceled_order = df_worker['canceled'].sum()
    canceled_ratio = canceled_order / send_order

    return canceled_ratio


def cal_long_short_manual(date, production_list=None):
    infor_list = []
    df_price = get_price(date, date).reset_index()[['SecuCode', 'ClosePrice']]
    df_targetmv = pd.read_csv(
        f'{PLATFORM_PATH_DICT["v_path"]}StockData/IndexPortfolioFile/Operation/OpenCloseAmount_{date}.csv')
    df_targetmv['Amount'] = df_targetmv['Amount'].fillna('0W').apply(lambda x: int(x.replace('W', '')) * 10000)

    if production_list is None:
        production_list = WinterFallProductionList_Alpha

    for prod in production_list:
        print(date, prod)
        df_pos = get_position(date, prod)
        if df_pos.empty:
            continue

        df_pos = pd.merge(df_pos, df_price, on='SecuCode', how='left').fillna(0)
        holdmv = (df_pos['ClosePrice'] * df_pos['Volume']).sum()

        df_destpos = get_alpha_quota_bar(date, prod, 8)
        if not df_destpos.empty:
            df_destpos = pd.merge(df_destpos, df_price, on='SecuCode', how='left').fillna(0)
            holdmv_quota = (df_destpos['ClosePrice'] * df_destpos['Volume']).sum()
        else:
            holdmv_quota = 0

        targetmv = df_targetmv[df_targetmv['Account'] == prod].tail(1)['Amount'].sum()
        infor_list.append([date, prod, holdmv, holdmv_quota, targetmv])

    df_long_infor = pd.DataFrame(
        infor_list, columns=['Date', 'Product', 'HoldMV', 'DestPos', 'TargetMV']).sort_values(['Product', 'Date'])
    df_long_infor['Operation'] = 'TwapManual'
    df_long_infor['TWAP'] = ((df_long_infor['HoldMV'] - df_long_infor['DestPos']) / 10000).apply(lambda x: int(round(x))) * 10000
    df_long_infor['TWAP'] *= (df_long_infor['TargetMV'] == 0)
    df_long_infor['DiffRatio'] = np.abs(df_long_infor['TWAP']) / df_long_infor['HoldMV']
    df_long_infor = df_long_infor[(df_long_infor['DiffRatio'] > 0.10) | (df_long_infor['TargetMV'] != 0)]

    outputdir = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/{date}/LongShortManual/'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    df_long_infor = df_long_infor[df_long_infor['TWAP'] != 0]
    df_long_infor = df_long_infor.reset_index(drop=True).rename(
        {'Product': 'Production', 'TWAP': 'Amount'}, axis='columns')[
        ['Production', 'Amount', 'Operation', 'Date']]
    print(df_long_infor)
    df_long_infor.to_csv(f'{outputdir}{date}_LONG_INFOR_OPERATION.csv', index=False)

    return df_long_infor


def get_statistic_results_max_min_mean_median(value_list, precision: int, type='origin'):
    if not value_list:
        return 'nan/nan/nan/nan'
    if type == 'origin':
        statis = '/'.join([str(round(float(x), precision)) for x in [
            np.nanmax(value_list),
            np.nanmin(value_list),
            np.nanmean(value_list),
            np.nanmedian(value_list)]])
    elif type == '+':
        statis = '/'.join([str(round(float(x), precision)) for x in [
            np.nanmax(value_list),
            np.nanmin(value_list),
            np.nanmean(value_list),
            np.nanmedian(value_list)]])
    elif type == '-':
        statis = '/'.join([str(round(float(x), precision)) for x in [
            np.nanmin(value_list),
            np.nanmax(value_list),
            np.nanmean(value_list),
            np.nanmedian(value_list)]])
    else:
        assert False, 'Incorrect parameters not in [origin, +, -]'
    return statis


def save_price_data_npy(df_data, file_path, filename):
    values_list = [col for col in df_data.columns if col not in ['code', 'time', 'date']]

    df_data_melt = pd.pivot_table(df_data, index='code', values=values_list, columns='time')

    if not os.path.exists(file_path + filename + '/'):
        os.makedirs(file_path + filename + '/')

    file_path = file_path + filename + '/'

    for value in values_list:
        file_path_value = file_path + f'{filename}_{value}.npy'
        value_array = np.array(df_data_melt[value].copy(deep=True).values, dtype=np.float64)
        fp = np.memmap(file_path_value, dtype=np.float64, mode='w+', shape=value_array.shape)
        # print(value_array.shape)
        fp[:] = value_array[:]
        fp.flush()
        # if value == 'close':
        #     hprint(df_data_melt[value])
        #     print(value_array)

    df_index = pd.DataFrame([df_data_melt.index.to_list()], index=['SecuCode']).T
    df_index.to_csv(file_path + f'{filename}_CodeList.csv', index=False)

    df_columns = pd.DataFrame([df_data_melt[values_list[0]].columns.to_list()], index=['TimeList']).T
    df_columns.to_csv(file_path + f'{filename}_TimeList.csv', index=False)


def read_price_data_npy(date, data_name='close', outputdir=f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/', file_type='_VWAP_1MIN_RG'):
    code_list = pd.read_csv(outputdir + f'{date}/{date}{file_type}/{date}{file_type}_CodeList.csv')[
        'SecuCode'].to_list()
    code_list = [expand_stockcode(x) for x in code_list]
    time_list = pd.read_csv(outputdir + f'{date}/{date}{file_type}/{date}{file_type}_TimeList.csv')[
        'TimeList'].to_list()

    memmap_ret = np.memmap(outputdir + f'{date}/{date}{file_type}/{date}{file_type}_{data_name}.npy',
                           mode='r', shape=(len(code_list), len(time_list)), dtype=np.float64)
    df_mem = pd.DataFrame(memmap_ret, index=code_list, columns=time_list)
    return df_mem


def get_zj_t0_order_values(curdate, product, file_path=None, file_short_path=None, code_2_price=None):
    product_num = {
        'ZJDC12': '24618',
        'ZJDC8': '25286'
    }

    if code_2_price is None:
        code_2_price = {code: price
                        for code, price in
                        get_price(curdate, curdate).reset_index()[['SecuCode', 'ClosePrice']].values}
    values_msg = ''
    if file_path is None:
        file_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/T0_TradingInfor/{product}_position/借券下单/'
    nextdate = get_predate(curdate, -1)
    order_file = f'{file_path}{product_num[product]}-借券下单-{nextdate}.xlsx'

    if os.path.exists(order_file):
        df_order = pd.read_excel(order_file).rename(
            {'证券代码': 'SecuCode', '锁定股数': 'Volume', '股数': 'Volume'}, axis='columns')

        df_order = df_order[df_order['Volume'] > 0]
        if len(df_order) > 0:
            df_order['SecuCode'] = df_order['SecuCode'].apply(lambda x: x.split('.')[0])
            code_2_price = {code: price
                            for code, price in
                            get_price(curdate, curdate).reset_index()[['SecuCode', 'ClosePrice']].values}
            msg_list, values_list = [], []
            for code, vol in df_order[['SecuCode', 'Volume']].values:
                value = round(vol * code_2_price[code], 2)
                msg_list.append(f'\t{code}: {vol},{value};')
                values_list.append(value)
            values_msg += f'{nextdate}-{product}-OrderLong总市值：{round(sum(values_list), 2)}:\n' + \
                         '\n'.join(msg_list) + '\n'
        else:
            values_msg += f'{nextdate}-{product}-OrderLong：无\n'
    else:
        values_msg += f'{nextdate}-{product}-OrderLong：无\n'

    if file_short_path is None:
        file_short_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/T0_TradingInfor/{product}_position/还券下单/'
    order_file = f'{file_short_path}{product_num[product]}-还券下单-{nextdate}.xlsx'
    if os.path.exists(order_file):
        df_order = pd.read_excel(order_file).rename(
            {'证券代码': 'SecuCode', '锁定股数': 'Volume', '股数': 'Volume'}, axis='columns')
        df_order = df_order[df_order['Volume'] > 0]
        if len(df_order) > 0:
            df_order['SecuCode'] = df_order['SecuCode'].apply(lambda x: x.split('.')[0])

            msg_list, values_list = [], []
            for code, vol in df_order[['SecuCode', 'Volume']].values:
                value = round(vol * code_2_price[code], 2)
                msg_list.append(f'\t{code}: {vol},{value};')
                values_list.append(value)
            values_msg += f'{nextdate}-{product}-OrderShort总市值：{round(sum(values_list), 2)}:\n' + \
                         '\n'.join(msg_list) + '\n'
        else:
            values_msg += f'{nextdate}-{product}-OrderShort：无\n'
    else:
        values_msg += f'{nextdate}-{product}-OrderShort：无\n'

    return values_msg


def format_date_2_str(origin_date):
    if str(origin_date).lower().startswith('n'):
        return '99999999'
    elif isinstance(origin_date, float) or isinstance(origin_date, int):
        return xldate_as_datetime(origin_date, 0).strftime('%Y%m%d')
    else:
        try:
            return origin_date.strftime('%Y%m%d')
        except:
            origin_date = origin_date.strip().split(' ')[0].strip()
            origin_date = origin_date.replace('年', '-').replace('月', '-').replace('与', '-').replace('日', '')
            origin_date = origin_date.replace(' ', '').replace('/', '-')
            return datetime.datetime.strptime(origin_date, '%Y-%m-%d').strftime('%Y%m%d')
        

def get_trading_accounts_paras_dict(para_name, file_path):
    configdir = f'{file_path}/trading-accounts.cfg'

    cfg = {}
    with open(configdir) as f:
        for fi in f.readlines():
            if fi.strip():
                key, value = fi.strip().split('=')
                value = tuple(value[1:-1].split())
                cfg[key] = value
    paras_dict = {}
    for i, prod in enumerate(cfg[para_name]):
        paras_dict[prod] = int(cfg['strategies'][i].replace('alpha', '')) - 1

    return paras_dict


def generate_quota_use_position(curdate, production, filepath=None):
    if filepath is None:
        df_position = get_position(curdate, production)
    else:
        df_position = get_position(curdate, production, filepath)
    df_position = df_position[['SecuCode', 'Volume']]
    df_position['Volume'] = (np.round(df_position['Volume'] / 100) * 100).astype('int')
    df_position['tmp'] = 0
    df_position = df_position[df_position['Volume'] != 0]
    df_position.to_csv(f'{PLATFORM_PATH_DICT["z_path"]}Trading/StockQuota_{production}.csv', index=False, header=False)
    print(df_position)
    print(('Quota 市值： ', get_quota_values(curdate, curdate, production, df_position)))


def time_2_bar_n_min(t, mode='30min'):
    if mode == '30min':
        bar_30_list = [90000, 100000, 103000, 110000, 113000, 133000, 140000, 143000, 153000]
        for i, (t1, t2) in enumerate(zip(bar_30_list, bar_30_list[1:])):
            if t1 < int(t) <= t2:
                return i + 1
        # whichbar = {(93000, 100001): 1, (100001, 103001): 2, (103001, 110001): 3, (110001, 125801): 4,
        #             (125801, 133001): 5, (133001, 140001): 6, (140001, 143001): 7, (143001, 150100): 8}
        # for key in whichbar.keys():
        #     if key[0] <= int(t) < key[1]: return whichbar[key]
    elif mode == '5min':
        bar_5_list = [
            90000,  93500,  94000,  94500,  95000,  95500, 100000, 100500, 101000,
            101500, 102000, 102500, 103000, 103500, 104000, 104500, 105000,
            105500, 110000, 110500, 111000, 111500, 112000, 112500, 113000,
            130500, 131000, 131500, 132000, 132500, 133000, 133500, 134000,
            134500, 135000, 135500, 140000, 140500, 141000, 141500, 142000,
            142500, 143000, 143500, 144000, 144500, 145000, 145500, 153000
        ]
        for i, (t1, t2) in enumerate(zip(bar_5_list, bar_5_list[1:])):
            if t1 < int(t) <= t2:
                return i + 1
    else: raise ValueError


def calculate_predict_netvalue_daily(curdate, product, holdmv=None):
    gbd = GetBaseData()
    df_hedge = gbd.get_details_future_margin_data(curdate)
    df_hedge = df_hedge[df_hedge['Product'] == product].reset_index(drop=True)
    margin_capital = df_hedge['Capital'].astype('float').sum()

    subprocess.getstatusoutput(
        f'"C:/Program Files/Git/usr/bin/scp.exe" jumper:~/rtchg/T0_Capital/HRTG2*.csv '
        f'{DATA_PATH_SELF}{curdate}')

    stocks_capital = pd.concat([
        pd.read_csv(f'{DATA_PATH_SELF}{curdate}/HRTG2-sz-capital.csv', header=None),
        pd.read_csv(f'{DATA_PATH_SELF}{curdate}/HRTG2-sh-capital.csv', header=None),
    ], axis=0)[9].sum()

    if holdmv is None:
        holdmv = get_position_values(curdate, curdate, product)

    return margin_capital + stocks_capital + holdmv


def calculate_hedging_future_values(curdate, product, df_hedgepos=None):
    if df_hedgepos is None: df_hedgepos = get_hedge_position(curdate, product)

    if df_hedgepos is None:
        return None

    if df_hedgepos.empty:
        return 0.

    df_future = get_futureprice(curdate, curdate)[['SecuCode', 'ClosePrice']]

    df_values = pd.merge(df_hedgepos, df_future, on='SecuCode', how='left')
    df_values['Multiplier'] = df_values['SecuCode'].apply(lambda x: Future_Value_Multiplier[x[:2]])
    df_values['Value'] = df_values['PosNum'] * df_values['Multiplier'] * df_values['ClosePrice']

    future_values = np.round(df_values['Value'].sum(), 2)
    return future_values


def export_position_singal_code(curdate, code):
    conlist = []
    for prod in WinterFallProductionList:
        df_pos = get_position(curdate, prod)[['SecuCode', 'Exchange', 'Volume']]
        df_pos = df_pos[df_pos['SecuCode'] == code]
        if not df_pos.empty:
            df_pos['Product'] = prod
            conlist.append(df_pos)
    df_pos = pd.concat(conlist, axis=0)
    df_pos.to_excel(f'{LOG_TEMP_PATH}all_position_{code}.xlsx', index=False)
    return df_pos


def get_hedge_position(curdate, product, df_hedge=None):
    if (df_hedge is None) and (curdate < '20220216'): return None

    gbd = GetBaseData()
    df_hedge = gbd.get_details_future_position_data(curdate, data_type='df', product=product).rename(
        {'Instrument': 'SecuCode'}, axis='columns')
    df_hedge['PosNum'] = df_hedge['Long'] + df_hedge['Short']
    df_hedge['Multiplier'] = df_hedge['SecuCode'].apply(lambda x: Future_Value_Multiplier[x[:2]])
    df_hedge = df_hedge[['SecuCode', 'PosNum', 'Multiplier']]

    return df_hedge


def generate_colo_list_use_trading(curdate):
    colo_list = []
    df_infor = pd.read_csv(
        f'{DATA_PATH_SELF}{curdate}/'
        f'Summary_Capital_{curdate}_Monitor.csv', encoding='GBK')[['Dual', 'Colo']]

    print(df_infor)
    for dual, colo in df_infor.values:
        if colo == '---':
            continue

        colo_list.append(colo)
        if dual == 2:
            colo_list.append(colo.replace('sz', 'sh').replace('dg', 'sh'))

    if Temp_Colo_Mechine_Dict.get(curdate, None) is not None:
        colo_list = list(set(colo_list + Temp_Colo_Mechine_Dict[curdate]))

    if Temp_Colo_Mechine_Dict['droplist']:
        for colo in Temp_Colo_Mechine_Dict['droplist']:
            if colo in colo_list:
                colo_list.remove(colo)

    colo_list = list(set(colo_list + Temp_Colo_Mechine_Dict['curdate']))

    predate = get_predate(curdate, 1)
    with open(f'{DATA_PATH_SELF}colo_list_trading.txt', 'w') as f:
        f.write(','.join(colo_list))
    with open(f'{DATA_PATH_SELF}{curdate}/colo_list_trading.txt', 'w') as f:
        f.write(','.join(colo_list))

    with open(f'{DATA_PATH_SELF}{predate}/colo_list_trading.txt', 'r') as f:
        colo_list_pre = f.read().split(',')

    msg = f'{curdate}-Colo List Check:\n'
    msg += f'\t昨日 Colo 数量：{len(colo_list_pre)}\n'
    msg += f'\t当日 Colo 数量：{len(colo_list)}\n'
    msg += f'\t当日增加 Colo name： {list(set(colo_list) - set(colo_list_pre))}\n'
    msg += f'\t当日减少 Colo name： {list(set(colo_list_pre) - set(colo_list))}'

    wechat_bot_msg_check(msg)


def process_df_trades(x, tart_type):
    try:
        return tart_type(x)
    except:
        try:
            return tart_type(x.replace('..', '.'))
        except:
            return np.nan


def get_all_production_close_position(curdate):
    gbd = GetBaseData()
    df_accsumm, dict_alpha, dict_holdmv, dict_bar_mode, accsumm_product_list = get_account_summary_info(curdate)
    product_list = accsumm_product_list + [sprd for sprd, sdict in SimuDict.items() if sdict.get('type') != 'bktest']
    conlist = []
    for product in product_list:
        df = gbd.get_position_close(curdate, product).rename({'PreCloseVolume': product}, axis='columns').set_index('SecuCode')
        conlist.append(df)

    df = pd.concat(conlist, axis=1).fillna(0)
    output_dir = f'{PLATFORM_PATH_DICT["v_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    df.to_csv(f'{output_dir}{curdate}_position_all_productions.csv')

    df_price = get_price(curdate, curdate).reset_index().set_index('SecuCode')
    df_price = df_price.reindex(df.index, fill_value=0)

    for product in product_list:
        df[product] *= df_price['ClosePrice']
        df[product] /= df[product].sum()

    df.to_csv(f'{output_dir}{curdate}_position_weight_all_productions.csv')


def get_all_production_alpha_quota_trading_volume(curdate, mode='dest-pos'):
    predate = get_predate(curdate, 1)
    df_accsumm, dict_alpha, dict_holdmv, dict_bar_mode, accsumm_product_list = get_account_summary_info(curdate)

    conlist, conlist_5m = [], []
    for product in accsumm_product_list:
        bar_mode = dict_bar_mode.get(product, 8)
        df_qtrading = get_simulate_diff_volume_mat_by_quota(curdate, predate, product, ret_df_mode='Diff', mode=mode, bar_mode=bar_mode)
        df_qtrading['Product'] = product
        if not df_qtrading.empty:
            if bar_mode == 8: conlist.append(df_qtrading)
            elif bar_mode == 48: conlist_5m.append(df_qtrading)

    df_qtrading = pd.concat(conlist, axis=0).reset_index()
    df_qtrading_5m = pd.concat(conlist_5m, axis=0).reset_index()

    output_dir = f'{PLATFORM_PATH_DICT["v_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    suffix = '' if mode == 'dest-pos' else f'_{mode}'
    df_qtrading.to_csv(f'{output_dir}{curdate}_quota_diff_all{suffix}.csv', index=False)
    df_qtrading_5m.to_csv(f'{output_dir}{curdate}_quota_diff_all_5m{suffix}.csv', index=False)


def get_option_product_future_pos(curdate, pos_mode='close', resp_mode=None, instrument=None):
    position_path = f'{PLATFORM_PATH_DICT["v_path"]}Trading/Alpha_TradingInfor/Summary/{curdate}_cfi_option_{pos_mode}_position.csv'
    if (not os.path.exists(position_path)) or (resp_mode == 'Origin'):
        gtd = GetBaseData()
        df_option = gtd.get_option_details(format_mode='format2pos')

        df_option['期末时间'] = np.minimum(df_option['期末时间'].astype('int'), df_option['到期日期'].astype('int'))
        if pos_mode == 'close':
            df_option = df_option[(df_option['起始日期'].astype('int') <= int(curdate)) & (df_option['期末时间'].astype('int') > int(curdate))]
        elif pos_mode == 'pre-close':
            df_option = df_option[(df_option['起始日期'].astype('int') < int(curdate)) & (df_option['期末时间'].astype('int') >= int(curdate))]
        else:
            raise 'ValueError'

        df_option['Multiplier'] = df_option['合约标的'].apply(lambda x: Future_Value_Multiplier[x])
        df_option['PosFut'] = df_option['数量'] / df_option['期初价格'] / df_option['Multiplier']
        if resp_mode == 'Origin': return df_option

        df_option = df_option.rename({'产品名称': 'Product',  '合约标的': 'FutureName', 'PosFut': 'Volume'}, axis='columns')
        if instrument is None:
            cur_format_date = datetime.datetime.strptime(curdate, '%Y%m%d')
            if cur_format_date.month == 12: instrument = datetime.datetime(cur_format_date.year + 1, 1, 1).strftime('%Y%m')[2:]
            else: instrument = datetime.datetime(cur_format_date.year, cur_format_date.month + 1, 1).strftime('%Y%m')[2:]
        df_option['Instrument'] = df_option['FutureName'] + str(instrument)
        df_option['Volume'] *= -1
        df_option = df_option.groupby(['Product', 'Instrument', 'FutureName']).agg({'Volume': 'sum', 'Multiplier': 'mean'}).reset_index()
        df_option = df_option[['Product', 'Instrument', 'FutureName', 'Volume', 'Multiplier']]
    else:
        df_option = pd.read_csv(position_path)
        if instrument is not None: df_option['Instrument'] = df_option['FutureName'] + str(instrument)
    if resp_mode is None: return df_option

    if resp_mode == 'dict':
        return {
            prod: df_pos_fut.groupby('Instrument')['Volume'].sum().to_dict()
            for prod, df_pos_fut in df_option.groupby('Product')
        }

    prod_option_2_pos = {}
    index_name_list = ['IF', 'IC', 'IM']
    for prod, df_pos_fut in df_option.groupby('Product'):
        hedge_pos_list = []
        for index_name in index_name_list:
            df_index = df_pos_fut[df_pos_fut['FutureName'] == index_name]
            if df_index.empty:
                hedge_pos_list.append(0.0)
            else:
                hedge_pos_list.append(np.round(df_index['Volume'].sum(), 2))
        prod_option_2_pos[prod] = hedge_pos_list

    return prod_option_2_pos


def get_swap_product_infor(curdate, product=None, remain_mode=True, file_path=None):
    gtpp = GetBaseData()
    Swap_df_list = gtpp.get_swap_details()

    conlist = []
    for i, (qs, df_total) in enumerate(Swap_df_list.items()):
        if i == 0:
            continue

        df_total = df_total.dropna(axis=0, subset=['产品名称']).dropna(axis=1, how='all')
        df_total = df_total[df_total['产品名称'].isin(ProductionName_2_Production.keys())]
        if df_total.empty:
            continue

        df_total['产品名称'] = df_total['产品名称'].apply(lambda x: ProductionName_2_Production[x])
        if '到期日期' in df_total.columns.to_list():
            df_total['到期日期'] = df_total['到期日期'].apply(lambda x: format_date_2_str(x))
            if remain_mode:
                df_total = df_total[df_total['到期日期'] >= curdate]
            else:
                df_total = df_total[df_total['到期日期'] > curdate]

        if df_total.empty:
            continue
        if '期末时间' in df_total.columns.to_list():
            df_total['期末时间'] = df_total['期末时间'].apply(lambda x: format_date_2_str(x))
            if remain_mode:
                df_total = df_total[df_total['期末时间'] >= curdate]
            else:
                df_total = df_total[df_total['期末时间'] > curdate]

        df_total = df_total.dropna(axis=0, how='all').dropna(axis=1, how='all')
        if df_total.empty:
            continue

        if '起始日期' in df_total.columns.to_list():
            df_total['起始日期'] = df_total['起始日期'].apply(lambda x: format_date_2_str(x))
            df_total = df_total[df_total['起始日期'] <= curdate]

        if df_total.empty:
            continue

        conlist.append(df_total)

    df_total = pd.concat(conlist, axis=0)
    if product is None:
        return df_total

    df_total = df_total[df_total['产品名称'] == product]
    return df_total


def get_swap_product_future_pos(curdate, pos_mode='close', resp_mode=None, product=None, file_path=None, df_swap=None):
    position_path = f'{PLATFORM_PATH_DICT["v_path"]}Trading/Alpha_TradingInfor/Summary/{curdate}_cfi_swap_{pos_mode}_position.csv'
    if (not os.path.exists(position_path)) or (resp_mode == 'Origin'):
        if df_swap is None:
            gbd = GetBaseData()
            df_total = gbd.get_swap_details(file_path=file_path, format_mode='format2pos')
        else:
            df_total = df_swap.copy(deep=True)

        df_total['期末时间'] = np.minimum(df_total['期末时间'].astype('int'), df_total['到期日期'].astype('int'))
        if pos_mode == 'close':
            df_total = df_total[(df_total['起始日期'].astype('int') <= int(curdate)) & (df_total['期末时间'].astype('int') > int(curdate))]
        elif pos_mode == 'pre-close':
            df_total = df_total[(df_total['起始日期'].astype('int') < int(curdate)) & (df_total['期末时间'].astype('int') >= int(curdate))]
        else: raise 'ValueError'

        df_total = df_total[df_total['合约标的'].str[:2].isin(FutureName_2_IndexName.keys())]
        if product is not None:
            if isinstance(product, str): product = [product]
            df_total = df_total[df_total['产品名称'].isin(product)]

        if resp_mode == 'Origin': return df_total
        df_total = df_total.rename({'产品名称': 'Product', '合约标的': 'Instrument', '数量': 'Volume'}, axis='columns')
        df_total['FutureName'] = df_total['Instrument'].str[:2]
        df_total = df_total.groupby(['Product', 'Instrument', 'FutureName'])['Volume'].sum().reset_index()
        df_total['Multiplier'] = df_total['FutureName'].apply(lambda x: Future_Value_Multiplier[x])
        df_total['Volume'] = - np.abs(df_total['Volume']) / df_total['Multiplier']
        df_total = df_total[df_total['Volume'] != 0]
    else:
        df_total = pd.read_csv(position_path)

    if resp_mode is None: return df_total

    if resp_mode == 'Instrument':
        return {
            prod: df_pos_fut.set_index('Instrument')['Volume'].to_dict() for prod, df_pos_fut in
            df_total.groupby('Product')
        }

    df_total = df_total.groupby(['Product', 'FutureName'])['Volume'].sum().reset_index()
    if resp_mode == 'FutureName-dict':
        return {
            prod: df_pos_fut.set_index('FutureName')['Volume'].to_dict() for prod, df_pos_fut in
            df_total.groupby('Product')
        }
    else:
        raise 'ValueError'


def get_alpha_quota_bar(curdate, product, bar, filepath=None, short_mode=False, recall_mode=False, bar_mode=8):
    filepath_30m = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/d0_destpos/'
    if filepath is None:
        if bar_mode == 8: filepath = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/d0_destpos/'
        else: filepath = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/d0_destpos_5m/'

    if bar == 1:
        if os.path.exists(f'{filepath}{curdate}/{bar}/{curdate}-{product}-destpos.csv'):
            print('Use', f'{filepath}{curdate}/{bar}/{curdate}-{product}-destpos.csv')
            df_trade_volume = pd.read_csv(f'{filepath}{curdate}/1/{curdate}-{product}-destpos.csv')
            df_trade_volume['InstrumentID'] = df_trade_volume['InstrumentID'].apply(lambda x: x[2:])
        else: df_trade_volume = pd.DataFrame(columns=['SecuCode', 'Volume'])
    elif bar == 'd1':
        if os.path.exists(f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/{curdate}/alpha-config/{curdate}-{product}-destpos.csv'):
            print('Use', f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/{curdate}/alpha-config/{curdate}-{product}-destpos.csv')
            df_trade_volume = pd.read_csv(f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/{curdate}/alpha-config/{curdate}-{product}-destpos.csv')
            df_trade_volume['InstrumentID'] = df_trade_volume['InstrumentID'].apply(lambda x: x[2:])
        else: df_trade_volume = pd.DataFrame(columns=['SecuCode', 'Volume'])
    else:
        if os.path.exists(f'{filepath}{curdate}/{bar}/{curdate}-{product}-destpos.csv'):
            df_trade_volume = pd.read_csv(f'{filepath}{curdate}/{bar}/{curdate}-{product}-destpos.csv')
            df_trade_volume['InstrumentID'] = df_trade_volume['InstrumentID'].apply(lambda x: x[2:])
        elif (bar_mode == 48) and os.path.exists(f'{filepath_30m}{curdate}/8/{curdate}-{product}-destpos.csv'):
            df_trade_volume = pd.read_csv(f'{filepath_30m}{curdate}/8/{curdate}-{product}-destpos.csv')
            df_trade_volume['InstrumentID'] = df_trade_volume['InstrumentID'].apply(lambda x: x[2:])
        else: df_trade_volume = pd.DataFrame(columns=['SecuCode', 'Volume'])
        
    if not df_trade_volume.empty:
        if short_mode: df_trade_volume['yshort'] = df_trade_volume['yshort'] - df_trade_volume['tshort']
        df_trade_volume = df_trade_volume.rename({'InstrumentID': 'SecuCode', 'yshort': 'Volume'}, axis='columns')[['SecuCode', 'Volume']]

        if short_mode and ((int(bar) == 8) or (int(bar) == 48)) and recall_mode:
            df_trade_volume = get_destpos_alpha_short_ti8_recall(curdate, df_trade_volume, product, recall_mode='quota')

    return df_trade_volume


def get_destpos_alpha_short_ti8_recall(curdate, df_destpos, product, recall_mode='position'):
    ms_recall_path = f'{PLATFORM_PATH_DICT["v_path"]}homes/amully/sbl/ms/pool/{curdate}/'
    zj_recall_path = f'{PLATFORM_PATH_DICT["v_path"]}homes/duanlian/stock_lend/cicc/stock_req/'

    if recall_mode == 'position':
        recall_path_dict = {
            'SHORTMSLS1': f'{ms_recall_path}MSLS1_eod_recall_{curdate}.csv',
            'SHORTMSLS2': f'{ms_recall_path}MSLS2_eod_recall_{curdate}.csv'
        }
    elif recall_mode == 'quota':
        recall_path_dict = {
            'SHORTMSLS1': f'{ms_recall_path}MSLS1_recall_{curdate}1530.csv',
            'SHORTMSLS2': f'{ms_recall_path}MSLS2_recall_{curdate}1530.csv'
        }
    else: raise ValueError

    recall_path_dict['SHORTZJJT1'] = f'{zj_recall_path}{curdate}_zjjt1_recall_short_result.csv'

    recall_path = recall_path_dict[product]
    if not os.path.exists(recall_path): return df_destpos

    df_recall = pd.read_csv(recall_path).rename({'stock_code': 'SecuCode', 'Security': 'SecuCode', 'Quantity': 'recall_qty'}, axis='columns')[['SecuCode', 'recall_qty']]
    df_recall['SecuCode'] = df_recall['SecuCode'].str[2:]
    df_destpos = pd.concat([df_recall.set_index('SecuCode'), df_destpos.set_index('SecuCode')], axis=1).fillna(0)
    df_destpos['Volume'] += df_destpos['recall_qty']

    df_destpos = df_destpos.reset_index()[['SecuCode', 'Volume']]

    return df_destpos


def get_intra_recall_zj(curdate, product):
    predate = get_predate(curdate, 1)
    recall_flag = {
        'SHORTZJJT1': '世纪前沿鲸涛1号-DMA'
    }[product]
    file_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/Alpha_TradingInfor/中金/Recall/'
    predate_fmt = datetime.datetime.strptime(predate, "%Y%m%d").strftime("%Y.%m.%d")
    file_list = list(Path(file_path).glob(f'*{recall_flag} {predate_fmt}*.xlsx'))
    if not file_list: return None

    df_rcll = pd.read_excel(file_list[-1])
    df_rcll['指令'] = df_rcll['指令'].replace('MOC', '0915-0930 TWAP 卖出')
    df_rcll['starttime'] = df_rcll['指令'].apply(lambda x: x.split()[0].split('-')[0])
    df_rcll['endtime'] = df_rcll['指令'].apply(lambda x: x.split()[0].split('-')[-1])
    df_rcll['how'] = df_rcll['指令'].apply(lambda x: x.split()[1].replace('卖出', ''))
    df_rcll['LongShort'] = 0
    df_rcll = df_rcll.rename({'代码': 'code', '世纪前沿鲸涛1号-DMA': 'Volume'}, axis='columns').drop(['指令', '简称'], axis=1)
    df_rcll['code'] = df_rcll['code'].apply(lambda x: expand_stockcode(x))
    df_rcll.to_csv(f'{file_path}{predate}_{product}_recall.csv', index=False)

    df_1min = get_n_min_stock_daily_data(curdate, period='1min')
    conlist = []
    for (starttime, endtime, how), df in df_rcll.groupby(['starttime', 'endtime', 'how']):
        print(starttime, endtime, how, curdate)
        code_list = df['code'].to_list()
        df_price = df_1min[df_1min['code'].isin(code_list)]
        df_price['starttime'] = int(starttime) * 100
        df_price['endtime'] = int(endtime) * 100
        df_price = df_price[(df_price['starttime'] < df_price['time']) & (df_price['time'] <= df_price['endtime'])]
        if how.lower() == 'vwap':
            df_price = df_price.groupby('code').agg({'turnover': 'mean', 'volume': 'mean', 'vwap': 'mean'}).reset_index()
            df_price['Price'] = (df_price['turnover'] / df_price['volume']).replace([np.inf, -np.inf], np.nan)
            df_price['Price'] = np.where(~ df_price['Price'].isna(), df_price['Price'], df_price['vwap'])
        elif how.lower() == 'twap':
            df_price = df_price.groupby('code')[['twapmid']].mean().reset_index().rename({'twapmid': 'Price'}, axis='columns')
        else: raise ValueError

        df_price = df_price[['code', 'Price']]
        df = pd.merge(df, df_price, on='code', how='left')
        conlist.append(df)

    df_trades = pd.concat(conlist, axis=0).rename({'code': 'SecuCode', 'endtime': 'time'}, axis='columns')
    df_trades['time'] = df_trades['time'].astype('str') + '00'
    df_trades['Date'] = curdate
    df_trades['tmp'] = 0
    df_trades['SecuCode'] = df_trades['SecuCode'].apply(lambda x: expend_market(x, suf_num=2, suf_mode=False))
    df_trades = df_trades.reindex(columns=['Date', 'time', 'SecuCode', 'LongShort', 'tmp', 'Price', 'Volume'])

    output_dir = f'{PLATFORM_PATH_DICT["v_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'
    filepath = output_dir + f'{curdate}-{production_2_account(product)}-trades_recall.txt'
    print(filepath)
    df_trades.to_csv(filepath, sep=',', index=False, header=False, quoting=csv.QUOTE_NONE)

    return df_trades


def get_intra_recall_ms(curdate, product, rtn_trds=False):
    ms_recall_path = f'{PLATFORM_PATH_DICT["v_path"]}homes/amully/sbl/ms/pool/{curdate}/'
    rcll_tm_2_bar = {
        1000: 2, 
        1030: 3, 
        1100: 4, 
        1130: 5, 
        1330: 6, 
        1400: 7, 
        1430: 8,
    }
    recall_flag = {
        'SHORTMSLS1': 'MSLS1',
        'SHORTMSLS2': 'MSLS2',
    }[product]
    conlist = []
    for rcll_tm, bar in rcll_tm_2_bar.items():
        file_rcll = ms_recall_path + f'{recall_flag}_recall_{curdate}{rcll_tm}.csv'
        if not os.path.exists(file_rcll): continue
        df_rcll = pd.read_csv(file_rcll).rename(
            {'Quantity': bar, 'Security': 'code'}, axis='columns')
        df_rcll['code'] = df_rcll['code'].str[2:]
        conlist.append(df_rcll.set_index('code'))

    if not rtn_trds: return pd.concat(conlist, axis=1).fillna(0)

    df_rcll = pd.read_csv(ms_recall_path + f'{recall_flag}_recall_{curdate}1530.csv').rename(
        {'Quantity': 9, 'Security': 'code'}, axis='columns')
    df_rcll['code'] = df_rcll['code'].str[2:]
    conlist.append(df_rcll.set_index('code'))

    df_rcll = pd.concat(conlist, axis=1).fillna(0).reset_index().melt(id_vars='code', value_name='Volume', var_name='bar')
    df_rcll['bar'] -= 1
    df_rcll = df_rcll[df_rcll['Volume'] != 0]

    df_30min = get_n_min_stock_daily_data(curdate, period='30min', mode30min='')[
        ['code', 'bar', 'vwap', 'time']].rename({'vwap': 'Price'}, axis='columns')

    df_trades = pd.merge(df_rcll, df_30min, on=['code', 'bar'], how='left').rename({'code': 'SecuCode'}, axis='columns')
    df_trades['LongShort'] = 0
    df_trades['Date'] = curdate
    df_trades['tmp'] = 0
    df_trades = df_trades.reindex(columns=['Date', 'time', 'SecuCode', 'LongShort', 'tmp', 'Price', 'Volume'])
    df_trades['SecuCode'] = df_trades['SecuCode'].apply(lambda x: expend_market(x, suf_num=2, suf_mode=False))
    output_dir = f'{PLATFORM_PATH_DICT["v_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'
    filepath = output_dir + f'{curdate}-{production_2_account(product)}-trades_recall.txt'
    print(filepath)
    df_trades.to_csv(filepath, sep=',', index=False, header=False, quoting=csv.QUOTE_NONE)


def get_simulate_diff_volume_mat_by_quota(date, predate=None, production=None, filepath=None, mode='dest-pos', ret_df_mode='Diff', short_mode=False, bar_mode=8, drop_bar_delay=True, scale_pqt_bar=None, drop_quota_list=None):
    if predate is None: predate = get_predate(date, 1)
    file_path_30m = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/d0_destpos/'
    file_path_5m = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/d0_destpos_5m/'
    if filepath is None:
        if bar_mode == 8: filepath = file_path_30m
        else: filepath = file_path_5m

    file_path_ex_right = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/ex_right_data/{date}_ex_right_data.csv'
    if os.path.exists(file_path_ex_right):
        df_exit_right = pd.read_csv(file_path_ex_right).rename({"SecuCode": 'code'}, axis='columns')
        df_exit_right['code'] = df_exit_right['code'].apply(lambda x: expand_stockcode(x))
    else:
        df_exit_right = pd.DataFrame()

    # try: print(f'{date}, {production}: {mode}'); except: pass
    if mode == 'dest-pos':
        target_path = f'{filepath}{predate}/{bar_mode}/{predate}-{production}-destpos.csv'
        if not os.path.exists(target_path):
            if bar_mode == 48: target_path = f'{file_path_30m}{predate}/8/{predate}-{production}-destpos.csv'
            else: target_path = f'{file_path_5m}{predate}/48/{predate}-{production}-destpos.csv'
            
        if os.path.exists(target_path):
            df_trade_pre_volume_bar_0 = pd.read_csv(target_path)
            if short_mode:
                df_trade_pre_volume_bar_0['yshort'] = df_trade_pre_volume_bar_0['yshort'] - df_trade_pre_volume_bar_0['tshort']

            df_trade_pre_volume_bar_0 = df_trade_pre_volume_bar_0[['InstrumentID', 'yshort']]
            df_trade_pre_volume_bar_0.columns = ['code', 'volume_bar_0']
            df_trade_pre_volume_bar_0['code'] = df_trade_pre_volume_bar_0['code'].apply(lambda x: x[2:])
            if short_mode:
                df_trade_pre_volume_bar_0 = df_trade_pre_volume_bar_0.rename(
                    {"code": 'SecuCode', 'volume_bar_0': 'Volume'}, axis='columns')
                df_trade_pre_volume_bar_0 = get_destpos_alpha_short_ti8_recall(
                    predate, df_trade_pre_volume_bar_0, production, recall_mode='quota').rename(
                    {"SecuCode": 'code', 'Volume': 'volume_bar_0'}, axis='columns')

            if not df_exit_right.empty:
                df_trade_pre_volume_bar_0 = pd.merge(df_trade_pre_volume_bar_0, df_exit_right, on='code', how='left')
                df_trade_pre_volume_bar_0['volume_bar_0'] *= df_trade_pre_volume_bar_0['ExRightRatio'].fillna(1)

            df_trade_pre_volume_bar_0 = df_trade_pre_volume_bar_0.set_index('code')[['volume_bar_0']]
        else:
            df_trade_pre_volume_bar_0 = pd.DataFrame(columns=['code', 'volume_bar_0']).set_index('code')
    elif mode == 'curr-pos':
        target_path = f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/{date}/alpha-config/{date}-{production}-currpos.csv'
        if os.path.exists(target_path):
            df_trade_pre_volume_bar_0 = pd.read_csv(target_path)
    
            df_trade_pre_volume_bar_0 = df_trade_pre_volume_bar_0[['InstrumentID', 'ylong']]
            df_trade_pre_volume_bar_0.columns = ['code', 'volume_bar_0']
            df_trade_pre_volume_bar_0['code'] = df_trade_pre_volume_bar_0['code'].apply(lambda x: x[2:])
            df_trade_pre_volume_bar_0 = df_trade_pre_volume_bar_0.set_index('code')
        else:
            df_trade_pre_volume_bar_0 = pd.DataFrame(columns=['code', 'volume_bar_0']).set_index('code')
    elif mode == 'position':
        df_trade_pre_volume_bar_0 = get_position(date, production)[['SecuCode', 'PreCloseVolume']].rename(
            {'SecuCode': 'code', 'PreCloseVolume': 'volume_bar_0'}, axis='columns')
        if short_mode:
            df_trade_pre_volume_bar_0['volume_bar_0'] *= -1
        df_trade_pre_volume_bar_0['code'] = df_trade_pre_volume_bar_0['code'].apply(lambda x: expand_stockcode(x))
        df_trade_pre_volume_bar_0 = df_trade_pre_volume_bar_0.set_index('code')
    elif mode == 'pre-position':
        df_trade_pre_volume_bar_0 = get_position(predate, production)[['SecuCode', 'Volume']].rename(
            {'SecuCode': 'code', 'Volume': 'volume_bar_0'}, axis='columns')
        if short_mode:
            df_trade_pre_volume_bar_0['volume_bar_0'] *= -1
        df_trade_pre_volume_bar_0['code'] = df_trade_pre_volume_bar_0['code'].apply(lambda x: expand_stockcode(x))

        if short_mode:
            df_trade_pre_volume_bar_0 = df_trade_pre_volume_bar_0.rename({"code": 'SecuCode', 'volume_bar_0': 'Volume'}, axis='columns')
            df_trade_pre_volume_bar_0 = get_destpos_alpha_short_ti8_recall(
                predate, df_trade_pre_volume_bar_0, production).rename({"SecuCode": 'code', 'Volume': 'volume_bar_0'}, axis='columns')

        if not df_exit_right.empty:
            df_trade_pre_volume_bar_0 = pd.merge(df_trade_pre_volume_bar_0, df_exit_right, on='code', how='left')
            df_trade_pre_volume_bar_0['volume_bar_0'] *= df_trade_pre_volume_bar_0['ExRightRatio'].fillna(1)

        df_trade_pre_volume_bar_0 = df_trade_pre_volume_bar_0.set_index('code')[['volume_bar_0']]
    else:
        raise 'ValueError'
    # if os.path.exists(f'{filepath}{date}/1/{date}-{production}-destpos.csv'):
    #     df_trade_volume_bar_1 = pd.read_csv(f'{filepath}{date}/1/{date}-{production}-destpos.csv')
    #     if short_mode:
    #         df_trade_volume_bar_1['yshort'] = df_trade_volume_bar_1['yshort'] - df_trade_volume_bar_1['tshort']
    #         # if os.path.exists(f'{filepath}{date}/1/{date}-{production}-shortquota.csv'):
    #         #     df_apply = pd.read_csv(f'{filepath}{date}/1/{date}-{production}-shortquota.csv', header=None)
    #         #     df_apply.columns = ['InstrumentID', 'Apply']
    #         #     df_trade_volume_bar_1 = pd.merge(df_trade_volume_bar_1, df_apply, on='InstrumentID', how='outer').fillna(0)
    #         #     df_trade_volume_bar_1['yshort'] -= df_trade_volume_bar_1['Apply']
    #
    #     df_trade_volume_bar_1 = df_trade_volume_bar_1[['InstrumentID', 'yshort']]
    #     df_trade_volume_bar_1.columns = ['code', 'volume_bar_1']
    #     df_trade_volume_bar_1['code'] = df_trade_volume_bar_1['code'].apply(lambda x: x[2:])
    #     df_trade_volume_bar_1 = df_trade_volume_bar_1.set_index('code')
    # elif os.path.exists(f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/{date}/alpha-config/{date}-{production}-destpos.csv'):
    #     df_trade_volume_bar_1 = pd.read_csv(
    #         f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/{date}/alpha-config/{date}-{production}-destpos.csv')
    #     if short_mode:
    #         df_trade_volume_bar_1['yshort'] = df_trade_volume_bar_1['yshort'] - df_trade_volume_bar_1['tshort']
    #         # if os.path.exists(f'{filepath}{date}/1/{date}-{production}-shortquota.csv'):
    #         #     df_apply = pd.read_csv(f'{filepath}{date}/1/{date}-{production}-shortquota.csv', header=None)
    #         #     df_apply.columns = ['InstrumentID', 'Apply']
    #         #     df_trade_volume_bar_1 = pd.merge(df_trade_volume_bar_1, df_apply, on='InstrumentID', how='outer').fillna(0)
    #         #     df_trade_volume_bar_1['yshort'] -= df_trade_volume_bar_1['Apply']
    #
    #     df_trade_volume_bar_1 = df_trade_volume_bar_1[['InstrumentID', 'yshort']]
    #     df_trade_volume_bar_1.columns = ['code', 'volume_bar_1']
    #     df_trade_volume_bar_1['code'] = df_trade_volume_bar_1['code'].apply(lambda x: x[2:])
    #     df_trade_volume_bar_1 = df_trade_volume_bar_1.set_index('code')
    # elif os.path.exists(f'{PLATFORM_PATH_DICT["u_path"]}StockData/ConfigDailyStore/{date}/alpha-config/{date}-{production}-destpos.csv'):
    #     df_trade_volume_bar_1 = pd.read_csv(
    #         f'{PLATFORM_PATH_DICT["u_path"]}StockData/ConfigDailyStore/{date}/alpha-config/{date}-{production}-destpos.csv')
    #     if short_mode:
    #         df_trade_volume_bar_1['yshort'] = df_trade_volume_bar_1['yshort'] - df_trade_volume_bar_1['tshort']
    #         # if os.path.exists(f'{filepath}{date}/1/{date}-{production}-shortquota.csv'):
    #         #     df_apply = pd.read_csv(f'{filepath}{date}/1/{date}-{production}-shortquota.csv', header=None)
    #         #     df_apply.columns = ['InstrumentID', 'Apply']
    #         #     df_trade_volume_bar_1 = pd.merge(df_trade_volume_bar_1, df_apply, on='InstrumentID', how='outer').fillna(0)
    #         #     df_trade_volume_bar_1['yshort'] -= df_trade_volume_bar_1['Apply']
    #
    #     df_trade_volume_bar_1 = df_trade_volume_bar_1[['InstrumentID', 'yshort']]
    #     df_trade_volume_bar_1.columns = ['code', 'volume_bar_1']
    #     df_trade_volume_bar_1['code'] = df_trade_volume_bar_1['code'].apply(lambda x: x[2:])
    #     df_trade_volume_bar_1 = df_trade_volume_bar_1.set_index('code')
    # else:
    #     df_trade_volume_bar_1 = pd.DataFrame(columns=['code', 'volume_bar_1']).set_index('code')
    # df_trade_volume_list = [df_trade_pre_volume_bar_0, df_trade_volume_bar_1]

    if scale_pqt_bar is not None: df_trade_pre_volume_bar_0['volume_bar_0'] *= scale_pqt_bar
    df_trade_volume_list = [df_trade_pre_volume_bar_0]

    # ls_flag = production in ProductionList_AlphaShort
    for i in range(1, bar_mode + 1):
        if drop_quota_list is not None:
            flag_refresh_bar = i not in drop_quota_list
        else:
            if bar_mode == 8:
                flag_refresh_bar = os.path.exists(f'{filepath}{date}/{i}/{date}_ti{i}_quota_diff.csv')
            else:
                flag_refresh_bar = os.path.exists(f'{filepath}{date}/{i}/{date}_ti{i}_quota_diff_5m.csv')

        destpos_path = f'{filepath}{date}/{i}/{date}-{production}-destpos.csv'
        if flag_refresh_bar:
            if os.path.exists(destpos_path):
                df_trade_volume = pd.read_csv(destpos_path)
                if short_mode: df_trade_volume['yshort'] = df_trade_volume['yshort'] - df_trade_volume['tshort']

                df_trade_volume = df_trade_volume[['InstrumentID', 'yshort']]
                df_trade_volume.columns = ['code', f'volume_bar_{i}']
                df_trade_volume['code'] = df_trade_volume['code'].apply(lambda x: x[2:])
                df_trade_volume = df_trade_volume.set_index('code')
                df_trade_volume_list.append(df_trade_volume)
            else:
                # if ls_flag: assert False, f'请检查 {destpos_path} 不存在!'
                df_trade_volume_list.append(deepcopy(
                    df_trade_volume_list[-1].rename({f'volume_bar_{i - 1}': f'volume_bar_{i}'}, axis='columns')))
        else:
            df_trade_volume_list.append(deepcopy(df_trade_volume_list[-1].rename({f'volume_bar_{i - 1}': f'volume_bar_{i}'}, axis='columns')))

    df_trade_volume = pd.concat(df_trade_volume_list, axis=1).fillna(0)  # .fillna(method='ffill', axis=1).fillna(0)
    if short_mode and ProductionDict_LongShortClss.get(production, '').startswith('ms'):
        df_rcll = get_intra_recall_ms(date, production).rename(
            {_bar: f'volume_bar_{_bar}' for _bar in range(1, bar_mode + 1)}, axis='columns')
        df_rcll = pd.DataFrame(np.cumsum(df_rcll, axis=1), index=df_rcll.index, columns=df_rcll.columns)
        df_rcll = df_rcll.reindex(df_trade_volume.index, columns=df_trade_volume.columns, fill_value=0)
        df_trade_volume -= df_rcll
        
    if ret_df_mode == 'Bar10':
        if os.path.exists(f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/{date}/alpha-config/{date}-{production}-destpos.csv'):
            df_trade_volume_bar_10 = pd.read_csv(
                f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/{date}/alpha-config/{date}-{production}-destpos.csv')
            if short_mode:
                df_trade_volume_bar_10['yshort'] = df_trade_volume_bar_10['yshort'] - df_trade_volume_bar_10['tshort']
                # if os.path.exists(f'{filepath}{date}/1/{date}-{production}-shortquota.csv'):
                #     df_apply = pd.read_csv(f'{filepath}{date}/1/{date}-{production}-shortquota.csv', header=None)
                #     df_apply.columns = ['InstrumentID', 'Apply']
                #     df_trade_volume_bar_10 = pd.merge(df_trade_volume_bar_10, df_apply, on='InstrumentID',
                #                                      how='outer').fillna(0)
                #     df_trade_volume_bar_10['yshort'] -= df_trade_volume_bar_10['Apply']

            df_trade_volume_bar_10 = df_trade_volume_bar_10[['InstrumentID', 'yshort']]
            df_trade_volume_bar_10.columns = ['code', 'volume_bar_d1']
            df_trade_volume_bar_10['code'] = df_trade_volume_bar_10['code'].apply(lambda x: x[2:])
            df_trade_volume_bar_10 = df_trade_volume_bar_10.set_index('code')
        else:
            df_trade_volume_bar_10 = pd.DataFrame(columns=['code', 'volume_bar_d1'])

        df_trade_volume = pd.merge(df_trade_volume, df_trade_volume_bar_10, on='code', how='outer').fillna(0)

    if ret_df_mode == 'Diff':
        for i in range(1, bar_mode + 1):
            df_trade_volume[f'Ti{i}_Weight_Diff'] = df_trade_volume[f'volume_bar_{i}'] - df_trade_volume[f'volume_bar_{i - 1}']
        df_trade_volume = df_trade_volume[[f'Ti{i}_Weight_Diff' for i in range(1, bar_mode + 1)]].sort_index()
        return df_trade_volume
    elif ret_df_mode == 'Bar8':
        return df_trade_volume[[f'volume_bar_{i}' for i in range(1, bar_mode + 1)]]
    elif ret_df_mode == 'Bar9':
        return df_trade_volume[[f'volume_bar_{i}' for i in range(bar_mode + 1)]]
    elif ret_df_mode == 'Bar10':
        return df_trade_volume[[f'volume_bar_{i}' for i in range(bar_mode + 1)] + ['volume_bar_d1']]
    elif ret_df_mode == 'All':
        for i in range(1, bar_mode + 1):
            df_trade_volume[f'Ti{i}_Weight_Diff'] = df_trade_volume[f'volume_bar_{i}'] - df_trade_volume[f'volume_bar_{i - 1}']
        return df_trade_volume.sort_index()
    else:
        assert False, 'Paras Error!'


def get_product_list_trades(curdate, product_list=None, add_product=False):
    conlist = []
    if product_list is None:
        if int(curdate) < int(datetime.datetime.now().strftime('%Y%m%d')):
            prodlist = get_production_list_trading(curdate, production=None) + Test_ProductionList + ProductionList_AlphaShort + T0_ProductionList
        else:
            prodlist = WinterFallProductionList

    for production in product_list:
        df_trades = get_trades(curdate, production)
        if not df_trades.empty:
            if add_product:
                df_trades['Product'] = production
            conlist.append(df_trades)
    df_trades = pd.concat(conlist, axis=0).reset_index(drop=True)

    print(curdate, len(df_trades))
    return df_trades


def get_product_list_position(curdate, product_list=None):
    conlist = []
    for production in product_list:
        df_position = get_position(curdate, production)
        if not df_position.empty:
            conlist.append(df_position)

    if conlist: df_position = pd.concat(conlist, axis=0).reset_index(drop=True)
    else: df_position = pd.DataFrame()

    print(curdate, len(df_position))
    return df_position


def get_code_list(curdate, mode='SZ-'):
    df_code = get_price(curdate, curdate).reset_index()[['SecuCode']].values.T[0]
    code_list = []
    for code in df_code:
        if code[0] in ['0', '3', '6']:
            if code[0] == '6':
                if mode == 'SZ-' or mode == 'SH-':
                    code_list.append('SH' + code)
                elif mode == '-SZ' or mode == '-SH':
                    code_list.append(code + '.SH')
                elif mode == '-SZE' or mode == '-SSE':
                    code_list.append(code + '.SSE')
                else:
                    assert False, 'Incorrect Parameters!'
            else:
                if mode == 'SZ-' or mode == 'SH-':
                    code_list.append('SZ' + code)
                elif mode == '-SZ' or mode == '-SH':
                    code_list.append(code + '.SZ')
                elif mode == '-SZE' or mode == '-SSE':
                    code_list.append(code + '.SZE')
                else:
                    assert False, 'Incorrect Parameters!'
    return code_list


def resample_n_min_data(curdate, resample_mode='0940'):
    outputdir = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/'
    df_5min = pd.read_hdf(f'{outputdir}{curdate}/{curdate}_VWAP_5MIN_RG.h5', key='VWAP')
    df_30min = pd.read_hdf(f'{outputdir}{curdate}/{curdate}_VWAP_30MIN_RG.h5', key='VWAP')
    df_5min['timeline'] = df_5min.apply(
        lambda row: datetime.datetime.strptime(str(row['date']) + ' ' + str(row['time']), '%Y%m%d %H%M%S'), axis=1)

    conlist = []
    # '93000 93500 94000 94500 95000 95500 100000 100500 101000 101500 102000 102500 103000 ' \
    # '103500 104000 104500 105000 105500 110000 110500 111000 111500 112000 112500 113000 ' \
    # '130500 131000 131500 132000 132500 133000 133500 134000 134500 135000 135500 140000 ' \
    # '140500 141000 141500 142000 142500 143000 143500 144000 144500 145000 145500 150000'

    for (date, code), df_code in df_5min.groupby(['date', 'code']):
        if resample_mode == '0940':
            df_code = df_code[~df_code['time'].isin([93500, 94000])]
        elif resample_mode == 'first5min':
            target_list = [93000,
                           94000, 100500, 103500, 110500, 130500, 133500, 140500, 143500]
            df_code = df_code[df_code['time'].isin(target_list)]
        elif resample_mode == 'first10min':
            target_list = [93000,
                           94000, 100500, 103500, 110500, 130500, 133500, 140500, 143500,
                           94500, 101000, 104000, 111000, 131000, 134000, 141000, 144000,]
            df_code = df_code[df_code['time'].isin(target_list)]
        elif resample_mode == 'first15min':
            target_list = [93000,
                           94000, 100500, 103500, 110500, 130500, 133500, 140500, 143500,
                           94500, 101000, 104000, 111000, 131000, 134000, 141000, 144000,
                           95000, 101500, 104500, 111500, 131500, 134500, 141500, 144500,]
            df_code = df_code[df_code['time'].isin(target_list)]

        df_code = df_code.set_index('timeline')[['volume', 'turnover']]

        df_code = df_code.resample('30min', closed='right', label='right').sum().reset_index()
        df_code['time'] = df_code['timeline'].apply(lambda x: x.strftime('%H%M%S'))
        df_code = df_code[~df_code['time'].isin(['120000', '123000', '130000'])].reset_index(drop=True)
        df_code['vwap'] = df_code['turnover'] / df_code['volume']

        df_30min_code = df_30min[df_30min['code'] == code]

        if resample_mode == '0940':
            df_30min_code['volume'] = np.array(
                df_code['volume'].to_list()[:8] + df_30min_code['volume'].to_list()[-2:])
            df_30min_code['turnover'] = np.array(
                df_code['turnover'].to_list()[:8] + df_30min_code['turnover'].to_list()[-2:])
            df_30min_code['vwap'] = np.array(df_code['vwap'].to_list()[:8] + df_30min_code['vwap'].to_list()[-2:])
        elif resample_mode in ['first5min', 'first10min', 'first15min']:
            df_30min_code['volume'] = np.array(
                df_code['volume'].to_list() + df_30min_code['volume'].to_list()[-1:])
            df_30min_code['turnover'] = np.array(
                df_code['turnover'].to_list() + df_30min_code['turnover'].to_list()[-1:])
            df_30min_code['vwap'] = np.array(df_code['vwap'].to_list() + df_30min_code['vwap'].to_list()[-1:])

        conlist.append(df_30min_code)

    df_30min = pd.concat(conlist, axis=0).reset_index(drop=True)
    df_30min.to_hdf(outputdir + f'{curdate}/{curdate}_VWAP_30MIN_RG_{resample_mode}_LACK.h5', key='VWAP', mode='w')
    df_30min.to_csv(outputdir + f'{curdate}/{curdate}_VWAP_30MIN_RG_{resample_mode}_LACK.csv', index=False)


def resample_n_min_data_details(curdate, resample_mode='0935'):
    outputdir = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/'
    df_5min = pd.read_hdf(f'{outputdir}{curdate}/{curdate}_VWAP_5MIN_RG.h5', key='VWAP')
    df_5min['timeline'] = df_5min.apply(lambda row: datetime.datetime.strptime(str(row['date']) + ' ' + str(row['time']), '%Y%m%d %H%M%S'), axis=1)

    if resample_mode == '0935': df_5min = df_5min[~ df_5min['time'].isin([93500])].copy(deep=True)

    conlist = []
    for (date, code), df_code in df_5min.groupby(['date', 'code']):
        df_code = df_code.sort_values(['code', 'time'], ascending=True).set_index('timeline').resample(
            '30min', closed='right', label='right').agg(
            {
                'date': lambda x: x.head(1),
                'time': lambda x: x.head(1),
                'code': lambda x: x.head(1),
                'open': lambda x: x.head(1),
                'close': lambda x: x.tail(1),
                'high': np.nanmax,
                'low': np.nanmin,
                'volume': np.nansum,
                'turnover': np.nansum,
                'vwap': lambda x: x.head(1),
                # 'twapask': np.nanmean,
                # 'twapbid': np.nanmean,
                # 'twapmid': np.nanmean,
            }).reset_index()
        df_code['time'] = df_code['timeline'].apply(lambda x: x.strftime('%H%M%S'))
        df_code = df_code[~df_code['time'].isin(['120000', '123000', '130000'])].reset_index(drop=True)
        df_code = df_code.drop('timeline', axis=1)
        df_code['vwap'] = df_code['turnover'] / df_code['volume']

        conlist.append(df_code)

    df_30min = pd.concat(conlist, axis=0)

    df_30min.to_hdf(outputdir + f'{curdate}/{curdate}_VWAP_30MIN_RG_{resample_mode}_LACK.h5', key='VWAP', mode='w')
    df_30min.to_csv(outputdir + f'{curdate}/{curdate}_VWAP_30MIN_RG_{resample_mode}_LACK.csv', index=False)


def generate_30min_vwap_deduct_trades_vol(curdate):
    production_list = get_production_list_trading(curdate)
    df_trades = get_product_list_trades(curdate, production_list)
    df_trades['bar'] = df_trades['time'].apply(lambda x: time_2_bar_n_min(x))
    df_trades['Volume'] = df_trades['Volume'].astype('float').fillna(0)

    df_trades['Amount'] = df_trades['Price'].astype('float').fillna(0) * df_trades['Volume']

    df_trades = df_trades.groupby(['SecuCode', 'bar'])[['Amount', 'Volume']].sum().reset_index()

    df_trades['bar'] = df_trades['bar'].apply(lambda x: -8 if x == 8 else x)
    df_trades = df_trades.rename({'SecuCode': 'code', 'Volume': 'VolCfi'}, axis=1)
    df_30min = get_n_min_stock_daily_data(curdate, period='30min', mode30min='_0935')
    columns_list = df_30min.columns.to_list()
    df_30min = pd.merge(df_30min, df_trades, on=['code', 'bar'], how='outer').fillna(0)

    df_30min['tn_amount'] = df_30min['turnover'] - df_30min['Amount']
    df_30min['diff_vol'] = df_30min['volume'] - df_30min['VolCfi']

    df_30min['vwap'] = df_30min.apply(
        lambda row: row['tn_amount'] / row['diff_vol']
        if (row['diff_vol'] != 0) and (row['tn_amount'] != 0) else np.nan, axis=1)

    df_30min = df_30min[columns_list]
    outputdir = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/'
    df_30min.to_hdf(outputdir + f'{curdate}/{curdate}_VWAP_30MIN_RG_0935_Adjusted.h5', key='VWAP', mode='w')
    df_30min.to_csv(outputdir + f'{curdate}/{curdate}_VWAP_30MIN_RG_0935_Adjusted.csv', index=False)
    df_30min_except = df_30min[df_30min['vwap'] <= 0].copy()
    if not df_30min_except.empty:
        wechat_bot_msg_check(f"{curdate}: \n{df_30min_except}")
    return df_30min


def generate_n_min_price_limit(curdate, modenmin='_0935', output_dir=None, rename_col=False, return_fix_df=True, bar_mode=8, save_mode=False):
    if (bar_mode == 48) and (int(curdate) < 20250120): return {}
    if output_dir is None:
        output_dir = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/{curdate}/PriceLimit/'
        if not os.path.exists(output_dir): os.makedirs(output_dir)

    if bar_mode == 8: file_path = f'{output_dir}{curdate}_VWAP_30MIN_RG{modenmin}_PriceLimit.csv'
    elif bar_mode == 48: file_path = f'{output_dir}{curdate}_VWAP_5MIN_RG{modenmin}_PriceLimit.csv'
    else: raise ValueError

    if (not os.path.exists(file_path)) or save_mode:
        if bar_mode == 8:
            df = get_n_min_stock_daily_data(curdate, period='30min', mode30min=modenmin)
            df = df[~df['bar'].isin([0, 8])]
            df['bar'] = np.abs(df['bar'])
            flag_bar = 'bar'
        elif bar_mode == 48:
            df = get_n_min_stock_daily_data(curdate, period='5min', mode5min=modenmin)
            df = df[df['bar_5'] >= 0]
            df['bar_5'] += 1
            flag_bar = 'bar_5'
        else: raise ValueError

        df_price = get_price(curdate, curdate).reset_index()
        df = pd.merge(df, df_price[['SecuCode', 'Uplimit', 'Downlimit']], left_on='code', right_on='SecuCode', how='left')
        df['IsUpLimit'] = (df['high'] == df['low']) & (df['high'] == df['Uplimit'])
        df['IsDownLimit'] = (df['high'] == df['low']) & (df['low'] == df['Downlimit'])
        df['Limit'] = df['IsUpLimit'].astype('int') - df['IsDownLimit'].astype('int')
        df_limit = df[df['Limit'] != 0]
        df_limit = pd.pivot_table(df_limit, index='code', columns=flag_bar, values='Limit').fillna(0).reset_index()

        df_limit.to_csv(file_path, index=False)
    else:
        df_limit = pd.read_csv(file_path)
        df_limit['code'] = df_limit['code'].apply(lambda x: expand_stockcode(x))

    df_limit = df_limit.set_index('code')

    if rename_col: df_limit.columns = list(range(1, bar_mode + 1))

    if return_fix_df:
        shape = df_limit.shape
        df_limit_bm = pd.DataFrame(np.tile(np.array(range(shape[1], 0, -1)), (shape[0], 1)), index=df_limit.index, columns=df_limit.columns)

        df_limit_reverse = df_limit.copy(deep=True).iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
        df_limit_fix = df_limit.copy(deep=True)[df_limit_bm == df_limit_reverse].fillna(0)
        return {'origin': df_limit, 'fix': df_limit_fix}
    return df_limit


def get_n_min_stock_daily_data(date, period='1min', mode30min='_0935', mode5min=''):
    if period == '1min':
        if os.path.exists(f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/{date}/{date}_VWAP_1MIN_RG_last_close.h5'):
            DATA_1_MIN_DIR = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/{date}/{date}_VWAP_1MIN_RG_last_close.h5'
            df_1_min_closeprice_data = pd.read_hdf(DATA_1_MIN_DIR, key='VWAP')

            return df_1_min_closeprice_data
        else:
            DATA_1_MIN_DIR = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/{date}/{date}_VWAP_1MIN_RG.h5'
            df_1_min_closeprice_data = pd.read_hdf(DATA_1_MIN_DIR, key='VWAP').sort_values(['date', 'code', 'time'])
            time_list = sorted(list(df_1_min_closeprice_data['time'].unique()))

            df_1_min_closeprice_data['volume'] = df_1_min_closeprice_data['volume'].fillna(0)
            df_1_min_closeprice_data['turnover'] = df_1_min_closeprice_data['turnover'].fillna(0)
            df_1_min_closeprice_data = df_1_min_closeprice_data.set_index(['time', 'code'])
            for time, code in df_1_min_closeprice_data[np.isnan(df_1_min_closeprice_data['close'])].index:
                index_time = time_list.index(time)
                if index_time > 0:
                    last_price = df_1_min_closeprice_data.loc[(time_list[index_time - 1], code), 'close']
                else:
                    df_price = get_price(date, date).reset_index()[['SecuCode', 'PreClosePrice']]
                    df_price = df_price[df_price['SecuCode'] == code]
                    if len(df_price) > 0:
                        last_price = df_price['PreClosePrice'].sum()
                    else:
                        last_price = np.nan
                if not np.isnan(last_price):
                    df_1_min_closeprice_data.loc[(time, code), 'open'] = last_price
                    df_1_min_closeprice_data.loc[(time, code), 'close'] = last_price
                    df_1_min_closeprice_data.loc[(time, code), 'high'] = last_price
                    df_1_min_closeprice_data.loc[(time, code), 'low'] = last_price
                    df_1_min_closeprice_data.loc[(time, code), 'vwap'] = last_price
                    df_1_min_closeprice_data.loc[(time, code), 'twapask'] = last_price
                    df_1_min_closeprice_data.loc[(time, code), 'twapbid'] = last_price
                    df_1_min_closeprice_data.loc[(time, code), 'twapmid'] = last_price

            df_1_min_closeprice_data = df_1_min_closeprice_data.reset_index()
            outputdir = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/'
            df_1_min_closeprice_data.to_hdf(outputdir + f'{date}/{date}_VWAP_1MIN_RG_last_close.h5', key='VWAP', mode='w')
            df_1_min_closeprice_data.to_csv(outputdir + f'{date}/{date}_VWAP_1MIN_RG_last_close.csv', index=False)

            return df_1_min_closeprice_data
    elif period == '30min':
        DATA_30_MIN_DIR = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/{date}/{date}_VWAP_30MIN_RG{mode30min}.h5'
        if not os.path.exists(DATA_30_MIN_DIR) and mode30min == '_0940_LACK':
            resample_n_min_data(date, resample_mode='0940')

        if not os.path.exists(DATA_30_MIN_DIR) and mode30min == '_first5min_LACK':
            resample_n_min_data(date, resample_mode='first5min')

        if not os.path.exists(DATA_30_MIN_DIR) and mode30min == '_first10min_LACK':
            resample_n_min_data(date, resample_mode='first10min')

        if not os.path.exists(DATA_30_MIN_DIR) and mode30min == '_first15min_LACK':
            resample_n_min_data(date, resample_mode='first15min')

        # if not os.path.exists(DATA_30_MIN_DIR) and mode30min == '_first5min_ONLY':
        #     resample_n_min_data(date, resample_mode='first5min_ONLY')
        #
        # if not os.path.exists(DATA_30_MIN_DIR) and mode30min == '_first10min_ONLY':
        #     resample_n_min_data(date, resample_mode='first10min_ONLY')
        #
        # if not os.path.exists(DATA_30_MIN_DIR) and mode30min == '_first15min_ONLY':
        #     resample_n_min_data(date, resample_mode='first15min_ONLY')

        if not os.path.exists(DATA_30_MIN_DIR) and mode30min == '_0935_Adjusted':
            generate_30min_vwap_deduct_trades_vol(date)

        df_30_min_closeprice_data = pd.read_hdf(DATA_30_MIN_DIR, key='VWAP')
        df_30_min_closeprice_data['vwap'] = np.where(
            df_30_min_closeprice_data['vwap'].isna(), df_30_min_closeprice_data['close'], df_30_min_closeprice_data['vwap'])

        return df_30_min_closeprice_data
    elif period == '5min':
        if not mode5min:
            df_5_min_closeprice_data = pd.read_csv(f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/{date}/{date}_VWAP_5MIN_RG.csv')
        else:
            df_5_min_closeprice_data = pd.read_csv(f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/vwap_data/{date}/delay_5min_interval_10s/{date}_VWAP_5MIN_RG{mode5min}.csv')
        df_5_min_closeprice_data['code'] = df_5_min_closeprice_data['code'].apply(lambda x: expand_stockcode(x))
        return df_5_min_closeprice_data
    elif period == '10s':
        df_10s_data = pd.read_feather(f'{PLATFORM_PATH_DICT["z_path"]}ProcessedData/stock_interval_10s/{date}-interval-10s.feather')
        df_10s_data['code'] = df_10s_data['code'].apply(lambda x: expand_stockcode(x))
        df_10s_data['date'] = df_10s_data['date'].apply(lambda x: x.strftime('%Y%m%d'))
        df_10s_data['time'] = df_10s_data['time'].apply(lambda x: int(x.strftime('%H%M%S')))
        return df_10s_data
    else:
        assert False, '错误的参数！'


def get_position_volume_value(date, production):
    df_position = get_position(date, production)[['SecuCode', 'Volume']]
    df_price = get_price(date, date).reset_index()[['SecuCode', 'ClosePrice']]
    df_merge = pd.merge(df_position, df_price, on='SecuCode', how='left')
    df_merge['Value'] = df_merge['Volume'] * df_merge['ClosePrice']
    df_merge = df_merge.sort_values('Value', ascending=False).reset_index(drop=True)
    df_merge.to_excel(f'{LOG_TEMP_PATH}{production}_position_volume_value.xlsx', index=False)
    hprint(df_merge)


def format_trades_time_2_minute(t, mode='1min', dtype='int'):
    adjust_hour = {96000: 100000, 106000: 110000, 136000: 140000, 146000: 150000}
    if mode == '1min':
        t_min = math.ceil(float(t) / 100) * 100
    elif mode == '5min':
        t_min = math.ceil(float(t) / 500) * 500
    elif mode == '30min':
        t_min = math.ceil(float(t) / 3000) * 3000
    else:
        assert False, 'Incorrect Mode Parameter!'
    if adjust_hour.get(t_min, None) is not None:
        t_min = adjust_hour[t_min]

    if 113000 < t_min <= 130000:
        t_min = 130100
    if 150000 < t_min:
        t_min = 150000

    if dtype == 'str':
        t_min = str(t_min)

    return t_min


def calculate_sharpe_annual_maxdd(list_ret, precision=4, return_mode='dict'):
    if not list_ret:
        list_ret = [np.nan]

    if return_mode == 'dict':
        return {
            'CumRet': round(sum(list_ret), precision),
            'AnnRet': round(sum(list_ret) * 244 / len(list_ret), precision),
            'Shp': round(np.mean(list_ret) / np.std(list_ret) * 244 ** 0.5, precision),
            'MDD': round(max(np.maximum.accumulate(np.cumsum(list_ret)) - np.array(np.cumsum(list_ret))), precision)
        }
    elif return_mode == 'dict-cumprod':
        cum_prod_list_ret = np.nancumprod(np.array(list_ret) + 1) - 1
        return {
            'CumRet': round(cum_prod_list_ret[-1], precision),
            'AnnRet': round(cum_prod_list_ret[-1] * 244 / len(list_ret), precision),
            'Shp': round(np.mean(list_ret) / np.std(list_ret) * 244 ** 0.5, precision),
            'MDD': round(max(np.maximum.accumulate(cum_prod_list_ret) - cum_prod_list_ret), precision)
        }
    elif (return_mode.split()[0] == 'str'):
        return_mode_2 = return_mode.split()[1] if len(return_mode.split()) == 2 else ''
        if return_mode_2 == '%':
            return '/'.join([
                     str(round(sum(list_ret), precision)),
                     str(round(sum(list_ret) * 244 / len(list_ret) / 100, precision)) + '%',
                     str(round(np.mean(list_ret) / np.std(list_ret) * 244 ** 0.5, precision)),
                     str(round(max(np.maximum.accumulate(np.cumsum(list_ret)) - np.array(np.cumsum(list_ret))), precision))
                 ])
        elif return_mode_2 == '%%':
            return '/'.join([
                     str(round(sum(list_ret) * 100, precision)) + '%',
                     str(round(sum(list_ret) * 244 / len(list_ret) * 100, precision)) + '%',
                     str(round(np.mean(list_ret) / np.std(list_ret) * 244 ** 0.5, precision)),
                     str(round(max(np.maximum.accumulate(np.cumsum(list_ret)) - np.array(np.cumsum(list_ret))) * 100, precision)) + '%'
                 ])
        else:
            return '/'.join(
                [str(res)
                 for res in [
                     round(sum(list_ret), precision),
                     round(sum(list_ret) * 244 / len(list_ret), precision),
                     round(np.mean(list_ret) / np.std(list_ret) * 244 ** 0.5, precision),
                     round(max(np.maximum.accumulate(np.cumsum(list_ret)) - np.array(np.cumsum(list_ret))), precision)
                 ]])
    else:
        assert False, 'Incorrect Parameter!'


def get_position_values(
        date, pricedate=None, Production=None, df_price=None, mode='market', quota_bar=8, code_list=None, df_position=None, bar_mode=8):

    if pricedate is None: pricedate = date

    if df_position is None:
        gbd = GetBaseData()
        if mode == 'market':
            df_position = gbd.get_position_close(date, Production).rename({'PreCloseVolume': 'Volume'}, axis='columns')
        elif mode == 'quota':
            df_position = gbd.get_quota_close(date, Production, bar=quota_bar).rename({'PreCloseVolume': 'Volume'}, axis='columns')
        else:
            raise ValueError

    if code_list is not None:
        print(set(df_position['SecuCode'].to_list()) - set(code_list))
        df_position = df_position[df_position['SecuCode'].isin(code_list)]

    if df_price is None:
        df_price = get_price(pricedate, pricedate).reset_index()[['SecuCode', 'ClosePrice']]
    else:
        df_price = df_price.reset_index()[['SecuCode', 'ClosePrice']]
    df_merge = pd.merge(df_position, df_price, on='SecuCode', how='left')
    df_merge['Value'] = df_merge['Volume'] * df_merge['ClosePrice']
    return round(df_merge['Value'].sum(), 2)


def get_position_weight(date, pricedate, Production, mode='market'):
    if mode == 'market':
        df_position = get_position(date, Production)[['SecuCode', 'Volume']]
    else:
        df_position = get_position_cal(date, Production)[['SecuCode', 'Volume']]
    df_price = get_price(pricedate, pricedate).reset_index()[['SecuCode', 'ClosePrice']]
    df_merge = pd.merge(df_position, df_price, on='SecuCode', how='left')
    df_merge['Value'] = df_merge['Volume'] * df_merge['ClosePrice']
    
    df_merge['weight']= df_merge['Value']/ np.sum(df_merge['Value'])

    return df_merge[['SecuCode', 'weight']].set_index('SecuCode')


def get_quota_values(date, pricedate, Production, df_quota=None):
    if df_quota is None:
        df_quota = get_quota(date, Production)
    if df_quota.empty:
        return 0

    df_price = get_price(pricedate, pricedate).reset_index()[['SecuCode', 'ClosePrice']]
    if df_price.empty:
        prepricedate = get_predate(pricedate, 1)
        df_price = get_price(prepricedate, prepricedate).reset_index()[['SecuCode', 'ClosePrice']]
    df_merge = pd.merge(df_quota, df_price, on='SecuCode', how='left')
    df_merge['Value'] = df_merge['Volume'] * df_merge['ClosePrice']
    return round(df_merge['Value'].sum(), 2)


def get_quota(curdate, Production, filepath=None):
    output_bk = f'{PLATFORM_PATH_DICT["z_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'
    if filepath is None:
        if os.path.exists(f'{output_bk}StockQuota_{Production}.csv'):
            if os.path.getsize(f'{output_bk}StockQuota_{Production}.csv') == 0:
                return pd.DataFrame()

            print(f'{output_bk}StockQuota_{Production}.csv')
            df_quota = pd.read_csv(f'{output_bk}StockQuota_{Production}.csv', header=None)
        else:
            print(f'{PLATFORM_PATH_DICT["z_path"]}Trading/StockQuota_{Production}.csv')
            df_quota = pd.read_csv(f'{PLATFORM_PATH_DICT["z_path"]}Trading/StockQuota_{Production}.csv', header=None)
    else:
        print(filepath)
        df_quota = pd.read_csv(f'{filepath}StockQuota_{Production}.csv', header=None)

    df_quota.columns = ['SecuCode', 'Volume', 'tmp']
    df_quota = df_quota[['SecuCode', 'Volume']]
    df_quota['Volume'] = df_quota['Volume'].astype('int')
    df_quota['SecuCode'] = df_quota['SecuCode'] .apply(lambda x: expand_stockcode(x))
    return df_quota

def production_2_feeratio(production): return PRODUCTION_2_FEE_RATIO.get(production, 0.00015)

def production_2_fee_min_value(product): return PRODUCTION_2_FEE_MIN_VALUE.get(product, 0)

def production_2_order_fee_mode(product): return PRODUCTION_2_ORDER_FEE_MODE.get(product, 0)

def production_2_bankname(production): return PRODUCTION_2_BANKNAME.get(production, '-')

def production_2_product_name(production): return PRODUCTION_2_PRODUCT_NAME.get(production, '-')

def production_2_strategy(production): return PRODUCTION_2_STRATEGY.get(production, 'ZQ')

def production_2_index(production, type='short'):
    if PRODUCTION_2_INDEX.get(production, None) is None:
        print(production, 'index not found!')
        return 'ZZ500'
    else:
        index = PRODUCTION_2_INDEX[production]
        if type == 'short':
            return IndexName_Long_2_Short.get(index, index)
        else:
            return index

def production_2_proportion(production):
    if PRODUCTION_2_PROPORTION.get(production, None) is None:
        print(f'production_2_proportion 没有查询到 {production} 的信息, 默认使用 500\n')
        return MIX_INDEX_PROPORTION_DEFAULT_DICT
    else:
        return PRODUCTION_2_PROPORTION[production]


def production_2_index_date(date, production):
    df_infor = pd.read_csv(f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/prod_infor_monitor/{date}_prod_infor.csv')
    product_2_index_date = {prod: index.upper() for prod, index in df_infor[['Production', 'Index']].values}

    if product_2_index_date.get(production, None) is None:
        print(production, 'index not found!')
        return 'ZZ500'
    else:
        return product_2_index_date[production]


def production_2_strategy_date(date, production):
    df_infor = pd.read_csv(f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/prod_infor_monitor/{date}_prod_infor.csv')
    product_2_strategy_date = {prod: strategy.upper() for prod, strategy in df_infor[['Production', 'Strategy']].values}

    if product_2_strategy_date.get(production, None) is None:
        print(production, 'strategy not found!')
        return 'ZQ'
    else:
        return product_2_strategy_date[production]


def production_2_proportion_date(date, production):
    if int(date) < 20221101: date = '20221201'

    df_infor = pd.read_csv(f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/prod_infor_monitor/{date}_prod_infor.csv')

    product_2_ratio_date = {}
    for prod, index_list, ratio_list, index_origin in df_infor[['Production', 'IndexList', 'Ratio', 'Index']].values:
        index_list = eval(index_list)
        ratio_list = eval(ratio_list)
        if sum(ratio_list) == 0:
            ratio_dict = {index_origin: 1} if IndexName_2_IndexSecuCode.get(index_origin) is not None else MIX_INDEX_PROPORTION_DEFAULT_DICT
        else:
            ratio_dict = {index: ratio for index, ratio in zip(index_list, ratio_list)}

        product_2_ratio_date[prod] = ratio_dict

    if product_2_ratio_date.get(production, None) is None:
        print(f'production_2_proportion_date 没有查询到 {date} {production} 的信息, 默认使用 500\n')
        return MIX_INDEX_PROPORTION_DEFAULT_DICT
    else:
        return deepcopy(product_2_ratio_date[production])


def production_2_account(product, reverse=False, non_match_raise=True, non_match_ret='product'):
    old_product = {
        'HTHK': 'cjx_jyy001',
        'HSZS6': '198800045449',
        'ZS2': '218000101',
        'ZS5I': '28219468',
        'XHd1': '28269759',
        'DC3': '198800888731',
        'DC9': '666810039886',
        'YRB': '101800000975',
        'YPB3': '28850485',
        'ZQ10': '932291669',
        'ZJDC8': 'sjqy02',
        'opt300simu': 'opt300simu'
    }
    if reverse:
        return ACCOUNT_2_PRODUCTION.get(product, None)

    if PRODUCTION_2_ACCOUNT.get(product, None) is None:
        if old_product.get(product, None) is None:
            df = pd.read_csv(f'{DATA_PATH_SELF}product_account_infor.csv')
            account_others = {prod: acc for prod, acc in zip(df['production'], df['account'])}
            if account_others.get(product, None) is None:
                if non_match_raise:
                    raise ValueError(f'{product} 没有账户信息!')
                else:
                    if non_match_ret == 'product':
                        return product
                    elif non_match_ret == 'nan':
                        return None
                    else:
                        raise 'ValueError'
            else:
                return account_others[product]
        else:
            return old_product[product]
    else:
        return PRODUCTION_2_ACCOUNT[product]


def get_date_nums(startdate, enddate):
    trade_cal = pd.read_csv(CALENDAR_CSV_PATH)
    trade_cal = trade_cal[trade_cal.apply(
        lambda row: int(startdate) <= int(row['cal_date']) <= int(enddate), axis=1)]
    return len(trade_cal) - 1, [str(date) for date in trade_cal['cal_date'].values[1:]]


def get_stockdaily_indexweight(QueryDate, index_name='ZZ500', OrderStyle='ListedDate'):
    if (int(QueryDate) < 20230831) and (index_name == 'ZZ2000'): QueryDate = '20230831'

    end_date = QueryDate
    cur_date = datetime.datetime.strptime(QueryDate, '%Y%m%d')
    one_month_ago = cur_date + datetime.timedelta(weeks=-5)
    start_date = one_month_ago.strftime('%Y%m%d')
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/JYDB")

    if IndexName_2_IndexinteriorCode.get(index_name, None) is not None:
        index_code = IndexName_2_IndexinteriorCode[index_name]
    else:
        raise ('Incorrect Index Name!')

    data_sql = ''' 
            select TSL.SecuCode, Weight as IndexWeight, EndDate
            from (select InnerCode, EndDate, Weight
            from LC_IndexComponentsWeight ICW
            where ICW.IndexCode = %s and EndDate between '%s' and '%s' )T1
            left join Table_AllStockList TSL
            on T1.InnerCode = TSL.InnerCode
            order by EndDate, SecuCode
            '''
    data_str = data_sql % (index_code, start_date, end_date)
    df = pd.read_sql(data_str, engine)
    df = df.set_index(['SecuCode'])
    if df.empty:
        return pd.DataFrame(columns=['SecuCode', 'IndexWeight', 'EndDate', 'PctChg'])
    df = df[df['EndDate'] == df['EndDate'][-1]]
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/WDDB")
    adj_sql = '''
    select S_INFO_WINDCODE,TRADE_DT,S_DQ_ADJCLOSE/S_DQ_ADJPRECLOSE as PctChg from ASHAREEODPRICES 
    where TRADE_DT between '%s' and '%s'
    '''
    adj_str = adj_sql % (df['EndDate'][-1].strftime('%Y%m%d'), end_date)
    adj_df = pd.read_sql(adj_str, engine)
    adj_df['SecuCode'] = adj_df['S_INFO_WINDCODE'].apply(lambda x: x.split('.')[0])
    adj_factor = adj_df.groupby('SecuCode')['PctChg'].prod()
    df = pd.merge(df, adj_factor, on=['SecuCode'])
    df['IndexWeight'] = df['IndexWeight'] * df['PctChg']
    df['IndexWeight'] = df['IndexWeight'] / df['IndexWeight'].sum()
    return df


def get_index_constituent_stock(date='20221213', indexcode='000852'):
    engine = create_engine("mssql+pymssql://public_data:public_data@dbs.cfi/CSMAR_DATA")
    data_sql = '''
        select SAMPLESECURITYCODE as SecuCode,TRADINGDATE as TradeDate, WEIGHT as weight
        from idx_weightnextday
        where TRADINGDATE = '%s' and SYMBOL = '%s'
        '''
    data_str = data_sql % (date, indexcode)

    df = pd.read_sql(data_str, engine, index_col=['SecuCode'])

    return df


def get_code_industry_class(CurDate, OrderStyle='ListedDate'):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi:1433/JYDB?charset=GBK")
    if OrderStyle == 'ListedDate':
        data_sql = '''
        select SecuCode, SecuAbbr,FirstIndustryCode, FirstIndustryName,
        SecondIndustryCode, SecondIndustryName, ThirdIndustryCode, ThirdIndustryName,
        convert(varchar(10),ListedDate,112) as ListedDate
        from Table_AllStockList where ListedDate <= '%s' order by ListedDate,SecuCode;
        '''
    elif OrderStyle == 'SecuCode':
        data_sql = '''
        select SecuCode, SecuAbbr,FirstIndustryCode, FirstIndustryName,
        SecondIndustryCode, SecondIndustryName, ThirdIndustryCode, ThirdIndustryName,
        convert(varchar(10),ListedDate,112) as ListedDate
        from Table_AllStockList where ListedDate <= %s order by SecuCode;
        '''
    else:
        raise ('Incorrect OrderStyle')
    data_str = data_sql % (CurDate)
    df = pd.read_sql(data_str, engine, index_col=['SecuCode'])
    return df


def get_long_short_date(start_date, end_date):
    engine = create_engine("mssql+pymssql://mluo:Cfi888_@dbs.cfi:1433/TRDB")
    data_sql = '''
                select * from Production_OpenShortRecord
                WHERE Date between %s and %s
                ''' % (start_date, end_date)
    df = pd.read_sql(data_sql, engine)
    return df


def get_exchange_rate_from_database(curdate, symbol='M0340585', return_type='df', start_date='20230101'):
    target_path = f'{PLATFORM_PATH_DICT["y_path"]}海外产品/exchange_rate/{curdate}_exchange_rate.csv'
    if not os.path.exists(target_path):
        engine = create_engine("mssql+pymssql://mluo:Cfi888_@dbs.cfi:1433/DATA_GROUP")
        data_sql = '''
                select * from wind_edb_daily wed 
            where symbol = '%s' 
            order by api_date
            '''
        data_str = data_sql % (symbol)
        df = pd.read_sql(data_str, engine).sort_values('insert_time', ascending=True).drop_duplicates(subset='api_date', keep='last')
        df['api_date'] = df['api_date'].apply(lambda x: str(x).replace('-', ''))
        df = df[df['api_date'].astype('int') >= int(start_date)]
        df = df.reset_index(drop=True)
        df.to_csv(target_path, index=False)
    else:
        df = pd.read_csv(target_path)

    if int(curdate) >= 20250403:
        df_new = get_exchange_rate_from_ddb(curdate, symbol='M0067855', return_type='df', start_date='20250403')
        df = pd.concat([df, df_new], axis=0)

    if return_type == 'dict':
        exchange_rate_dict = {str(date): ratio for date, ratio in df[['api_date', 'value']].values}
        exchange_rate_dict['20250318'] = 7.22857
        exchange_rate_dict['20250319'] = 7.23118
        exchange_rate_dict['20250320'] = 7.24848
        exchange_rate_dict['20250321'] = 7.25163
        exchange_rate_dict['20250324'] = 7.25953
        exchange_rate_dict['20250325'] = 7.25847
        exchange_rate_dict['20250326'] = 7.26427
        exchange_rate_dict['20250327'] = 7.2648
        exchange_rate_dict['20250328'] = 7.2633
        exchange_rate_dict['20250331'] = 7.2576
        exchange_rate_dict['20250401'] = 7.2703
        exchange_rate_dict['20250402'] = 7.2679
        # exchange_rate_dict['20250430'] = 7.2
        return exchange_rate_dict

    return df


def get_exchange_rate_from_ddb(curdate, symbol='M0067855', return_type='df', start_date='20231211'):
    target_path = f'{PLATFORM_PATH_DICT["y_path"]}海外产品/exchange_rate/{curdate}_exchange_rate_new.csv'
    if not os.path.exists(target_path):
        s = ddb.session()
        s.connect(host="192.168.0.144", port=8902, userid="rtchg", password="Cfi888_clustergr")
        dos = f"""
            select * from loadTable("dfs://ods_wind_api_range_year","edb")
            where symbol = "{symbol}" and api_date > 2023.12.11
        """
        df = s.run(dos) 
        df.to_csv(target_path, index=False)
    else:
        df = pd.read_csv(target_path)
    
    df['api_date'] = df['api_date'].apply(lambda x: str(x).split()[0].replace('.', '').replace('-', ''))
    df = df[df['api_date'].astype('int') >= int(start_date)]
    if return_type == 'dict':
        exchange_rate_dict = {str(date): ratio for date, ratio in df[['api_date', 'value']].values}
        return exchange_rate_dict

    return df


def get_industry_code():
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/WDDB?charset=GBK")
    data_sql = '''
            select * from AShareIndustriesCode
            WHERE LEVELNUM= 2 and USED= 1
            '''
    df = pd.read_sql(data_sql, engine)[['INDUSTRIESCODE', 'INDUSTRIESNAME']]
    return df


def get_sw_industry_class():
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/WDDB?charset=GBK")
    data_sql = '''
       select * from AShareSWNIndustriesClass 
               '''
    industry_class = pd.read_sql(data_sql, engine).fillna('0')
    industry_class = industry_class[industry_class['REMOVE_DT'] < '1']
    industry_class['S_INFO_WINDCODE'] = industry_class['S_INFO_WINDCODE'].apply(lambda x: x.split('.')[0])

    industry_class = industry_class[['S_INFO_WINDCODE', 'SW_IND_CODE', 'ENTRY_DT']]
    industry_code = get_industry_code()

    industry_code['INDUSTRIESCODE'] = industry_code['INDUSTRIESCODE'].apply(lambda x: x[:4])
    industry_class['SW_IND_CODE'] = industry_class['SW_IND_CODE'].apply(lambda x: x[:4])
    df_industry_sw = pd.merge(
        industry_code, industry_class, left_on='INDUSTRIESCODE', right_on='SW_IND_CODE', how='right')
    df_industry_sw = df_industry_sw[['S_INFO_WINDCODE', 'INDUSTRIESNAME', 'ENTRY_DT']].rename(
        {'S_INFO_WINDCODE': 'SecuCode', 'INDUSTRIESNAME': 'IndustryName'}, axis='columns')
    return df_industry_sw


def get_code_citic_industry():
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/WDDB?charset=GBK")
    data_sql = '''
       select * from AShareIndustriesClassCITICS 
               '''
    industry_class = pd.read_sql(data_sql, engine).fillna('0')
    industry_class = industry_class[industry_class['REMOVE_DT'] < '1']
    industry_class['S_INFO_WINDCODE'] = industry_class['S_INFO_WINDCODE'].apply(lambda x: x.split('.')[0])

    industry_class = industry_class[['S_INFO_WINDCODE', 'CITICS_IND_CODE', 'ENTRY_DT']]
    industry_code = get_industry_code()

    industry_code['INDUSTRIESCODE'] = industry_code['INDUSTRIESCODE'].apply(lambda x: x[:4])
    industry_class['CITICS_IND_CODE'] = industry_class['CITICS_IND_CODE'].apply(lambda x: x[:4])
    df_industry_citic = pd.merge(
        industry_code, industry_class, left_on='INDUSTRIESCODE', right_on='CITICS_IND_CODE', how='right')
    df_industry_citic = df_industry_citic[['S_INFO_WINDCODE', 'INDUSTRIESNAME', 'ENTRY_DT']].rename(
        {'S_INFO_WINDCODE': 'SecuCode', 'INDUSTRIESNAME': 'IndustryName'}, axis='columns')
    return df_industry_citic


def get_st_stocklist(curdate, ret_list=True):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/WDDB")
    data_sql = '''
        select S_INFO_WINDCODE, S_TYPE_ST, REMOVE_DT, ENTRY_DT
        from ASHAREST
    '''
    data_str = data_sql
    df = pd.read_sql(data_str, engine)
    df = df[(df['REMOVE_DT'].isna() | (df['REMOVE_DT'] > curdate)) & ((df['ENTRY_DT'] <= curdate) & (df['S_TYPE_ST'] != 'R'))]
    df['SecuCode'] = df['S_INFO_WINDCODE'].apply(lambda x: x.split('.')[0])
    df = df.sort_values('ENTRY_DT', ascending=True).groupby('SecuCode').head(1)
    if ret_list:
        st_list = list(df['SecuCode'].unique())
        return st_list

    return df


def get_st_remove_stocklist(start_date):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/WDDB")
    data_sql = '''
        select S_INFO_WINDCODE, S_TYPE_ST, REMOVE_DT, ENTRY_DT
        from ASHAREST
    '''
    data_str = data_sql
    df = pd.read_sql(data_str, engine)
    df = df[(df['S_TYPE_ST'].isin(['T', 'L'])) & (df['ENTRY_DT'].astype('int') >= int(start_date))]
    return df['S_INFO_WINDCODE'].str[:6].unique()


def get_suspension_stocklist(curdate):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/WDDB")
    data_sql = '''
        declare @CurQueryDate smalldatetime
        set @CurQueryDate = '%s'
        select T1.SecuCode, T1.S_INFO_WINDCODE as stockcode,  convert(varchar(10),@CurQueryDate,112) as QueryDate,
        T1.S_DQ_SUSPENDDATE as SuspendDate,
        T1.S_DQ_RESUMPDATE as ResumptionDate,
        T1.S_DQ_CHANGEREASON,
        T1.S_DQ_SUSPENDTYPE
        from  (select ROW_NUMBER() over(partition by SecuCode order by SR.S_DQ_SUSPENDDATE desc) as RowNum, 
        TSL.SecuCode,  SR.* from
        Table_WDAllStockList TSL left join AShareTradingSuspension SR on TSL.S_INFO_WINDCODE = SR.S_INFO_WINDCODE ) T1
        where T1.RowNum = 1 and S_DQ_SUSPENDDATE <= @CurQueryDate and 
        ( (S_DQ_SUSPENDTYPE = '444003000' and @CurQueryDate < ISNULL(S_DQ_RESUMPDATE,'20500101')
        or (S_DQ_SUSPENDDATE = @CurQueryDate and S_DQ_SUSPENDTYPE <> '444003000') ) )  
        order by SecuCode
    '''
    data_str = data_sql % curdate
    df = pd.read_sql(data_str, engine, index_col=['SecuCode'])

    return list(df.index)


def get_secucode_2_abbr(abbr_2_code=False):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/JYDB?charset=GBK")
    data_sql = """
            select 
            SecuCode, SecuAbbr
            from SecuMain
            """
    data_str = data_sql
    df = pd.read_sql(data_str, engine)

    if not abbr_2_code: return {code: abbr for code, abbr in df[['SecuCode', 'SecuAbbr']].values}
    else: return {abbr: code for code, abbr in df[['SecuCode', 'SecuAbbr']].values}


def get_backtest_datelist(start_date, end_date):
    predate_STR = start_date[4:6] + '/' + start_date[6:] + '/' + start_date[:4]
    nowtime_STR = end_date[4:6] + '/' + end_date[6:] + '/' + end_date[:4]
    datelist = [i.strftime('%Y%m%d') for i in list(pd.bdate_range(predate_STR, nowtime_STR))]
    return datelist


def get_trading_days(start_date, end_date, list_mode='day'):
    trade_cal = pd.read_csv(CALENDAR_CSV_PATH)[['cal_date', 'is_open']]
    trade_cal = trade_cal[trade_cal.apply(
        lambda row: (int(start_date) <= int(row['cal_date']) <= int(end_date)) and (row['is_open'] == 1), axis=1)]
    trade_cal = trade_cal.sort_values('cal_date', ascending=True)

    if list_mode == 'day':
        date_list = [ilis for ilis in trade_cal['cal_date'].astype('str').to_list()]
        return date_list
    elif list_mode == 'month':
        trade_cal['month'] = trade_cal['cal_date'].apply(
            lambda x: datetime.datetime.strptime(str(x), "%Y%m%d").strftime('%Y%m'))
        date_dict = {mode: df_date['cal_date'].astype('str').to_list() for mode, df_date in trade_cal.groupby('month')}

        return date_dict
    elif list_mode == 'year':
        trade_cal['year'] = trade_cal['cal_date'].apply(
            lambda x: datetime.datetime.strptime(str(x), "%Y%m%d").strftime('%Y'))
        date_dict = {mode: df_date['cal_date'].astype('str').to_list() for mode, df_date in trade_cal.groupby('year')}

        return date_dict
    elif list_mode == 'week':
        trade_cal['week'] = trade_cal['cal_date'].apply(
            lambda x: datetime.datetime.strptime(str(x), "%Y%m%d").weekday())
        trade_cal['flag'] = np.cumsum((trade_cal['week'] < trade_cal['week'].shift(1).fillna(0))) + 1
        date_dict = {mode: df_date['cal_date'].astype('str').to_list() for mode, df_date in trade_cal.groupby('flag')}

        return date_dict
    else:
        raise 'ValueError'


def get_datelist(start_date, end_date):
    trade_cal = pd.read_csv(CALENDAR_CSV_PATH)[['cal_date']]
    trade_cal = trade_cal[trade_cal.apply(
        lambda row: int(start_date) <= int(row['cal_date']) <= int(end_date), axis=1)]
    return list([str(date) for date in trade_cal['cal_date'].values])


def get_account_summary_info(date, dropsimu=True):
    df_accsumm = get_production_list_trading(date, ret_df_data=True, drop_simu=dropsimu)
    dict_alpha = {prod: alpha for prod, alpha in zip(df_accsumm['Account'], df_accsumm['Alpha'])}
    dict_holdmv = {prod: bar_mode for prod, bar_mode in zip(df_accsumm['Account'], df_accsumm['HoldMV'])}
    dict_bar_mode = {prod: bar_mode for prod, bar_mode in zip(df_accsumm['Account'], df_accsumm['bar'])}
    accsumm_product_list = df_accsumm['Account'].to_list()
    return df_accsumm, dict_alpha, dict_holdmv, dict_bar_mode, accsumm_product_list


def get_production_list_trading(date, production=None, ret_type='alpha', ret_df_data=False, drop_simu=True, df_accsum=None, ret_nan=None):
    if df_accsum is None:
        acc_summ_path = f'{PLATFORM_PATH_DICT["v_path"]}StockData/IndexPortfolioFile/weightRef/'

        acc_alp_500 = pd.read_csv(acc_summ_path + f'{date}/accountSummary_{date}.csv')[['Account', 'Alpha', 'HoldMV', 'simu']]
        if os.path.exists(acc_summ_path + f'{date}_HS300/accountSummary_{date}_300.csv'):
            acc_alp_300 = pd.read_csv(acc_summ_path + f'{date}_HS300/accountSummary_{date}_300.csv')[['Account', 'Alpha', 'HoldMV', 'simu']]
            df_accsum = pd.concat([acc_alp_500, acc_alp_300], axis=0).dropna(axis=0).drop_duplicates(subset='Account')
        else:df_accsum = acc_alp_500.dropna(axis=0)

        df_accsum['bar'] = 8
        if os.path.exists(acc_summ_path + f'{date}/accountSummary_{date}_5m.csv'):
            acc_alp_5m = pd.read_csv(acc_summ_path + f'{date}/accountSummary_{date}_5m.csv')[['Account', 'Alpha', 'HoldMV', 'simu']]
            acc_alp_5m['bar'] = 48
            df_accsum = df_accsum[~ df_accsum['Account'].isin(acc_alp_5m['Account'].to_list())]
            df_accsum = pd.concat([df_accsum, acc_alp_5m], axis=0).dropna(axis=0).drop_duplicates(subset='Account')

        df_accsum['Alpha'] = df_accsum['Alpha'].apply(lambda x: x if x[:7] != 'DLZZ500' else 'hzhang.weight' + x[7:].replace('r', 'opt_r'))

        if drop_simu:
            df_accsum = df_accsum[df_accsum['Account'].apply(lambda x: ('simu' not in x) and ('HS300' not in x) and ('summary' not in x))]
            df_accsum = df_accsum[df_accsum['simu'].apply(lambda x: x != 1)].reset_index(drop=True)

        predate = get_predate(date, 1)
        path_pos = f'{PLATFORM_PATH_DICT["v_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{predate}/{predate}_ProductQuotaPositionExpose.csv'
        if os.path.exists(path_pos):
            df = pd.read_csv(path_pos)
            pre_holdmv_dict = np.abs(df.set_index('Product')['HoldMV']).to_dict()
        else:
            pre_holdmv_dict = {}

        if int(date) < 20250424:
            df_accsum_ls = pd.DataFrame([
                [prd, 'longshort', pre_holdmv_dict.get(prd, np.nan), 0, 8] for prd in ProductionList_AlphaShort
            ], columns=['Account', 'Alpha', 'HoldMV', 'simu', 'bar'])
        else:
            df_accsum_ls = pd.DataFrame([
                [prd, 'longshort', pre_holdmv_dict.get(prd, np.nan), 0, 8]
                 if ProductionDict_LongShortClss.get(prd) != 'msls2' 
                 else [prd, 'longshort', pre_holdmv_dict.get(prd, np.nan), 0, 48] for prd in ProductionList_AlphaShort
            ], columns=['Account', 'Alpha', 'HoldMV', 'simu', 'bar'])
        
        df_accsum = pd.concat([df_accsum, df_accsum_ls], axis=0)

    if ret_df_data:
        return df_accsum

    if production is None:
        return df_accsum['Account'].to_list()
    else:
        df_accsum = df_accsum[df_accsum['Account'] == production]

        if ret_type == 'alpha':
            return df_accsum['Alpha'].iloc[0] if (not df_accsum.empty) else None
        elif ret_type == 'holdmv':
            if ret_nan is None:
                return df_accsum['HoldMV'].iloc[0] if (not df_accsum.empty) else None
            elif ret_nan == 'nan':
                return df_accsum['HoldMV'].iloc[0] if (not df_accsum.empty) else np.nan
            elif ret_nan == 0:
                return df_accsum['HoldMV'].iloc[0] if (not df_accsum.empty) else 0
            else:
                raise 'ValueError'
        elif ret_type == 'bar':
            return df_accsum['bar'].iloc[0] if (not df_accsum.empty) else 8
        else:
            assert False, 'Incorrect Parameter!!'


def get_current_week_day(curdate, week_day=0):
    '''
     作用: 获取当前周指定星期的具体日期
     参数:
    '''
    today = datetime.datetime.strptime(str(curdate), '%Y%m%d')
    # datetime模块中,星期一到星期天对应数字0到6
    delta_hour = datetime.timedelta(days=1)  # 改变幅度为1天
    while today.weekday() != week_day:
        if today.weekday() > week_day:
            today -= delta_hour
        elif today.weekday() < week_day:
            today += delta_hour
        else:
            pass
    return today.strftime("%Y%m%d")


def get_next_trading_date(date):
    if get_whether_trading_day(date):
        nextdate = get_predate(date, -1)
        return nextdate
    else:
        datelist = get_datelist(date, get_predate(date, -1))
        for date in datelist:
            if get_whether_trading_day(date):
                return date


def get_whether_trading_day(date):
    trade_cal = pd.read_csv(CALENDAR_CSV_PATH)[['cal_date', 'is_open']]
    trade_cal = trade_cal[trade_cal['cal_date'].astype('int') == int(date)]
    date_2_isopen = {str(idate): bool(is_open) for idate, is_open in trade_cal.values}

    return date_2_isopen[date]


def get_week_days(start_date, end_date):
    trade_cal = pd.read_csv(CALENDAR_CSV_PATH)[['cal_date', 'is_open']]
    trade_cal = trade_cal[trade_cal.apply(
        lambda row: int(start_date) <= int(row['cal_date']) <= int(end_date), axis=1)]
    days, is_open = list(trade_cal['cal_date'].values), list(trade_cal['is_open'].values)
    i, weeklist = -1, []
    while True:
        if is_open[i] == 1:
            weeklist.append(str(days[i]))
            i -= 1
        else:
            break
    weeklist.reverse()
    return weeklist


def get_monthly_trading_day(cur_date):
    cur_date = datetime.datetime.strptime(cur_date, '%Y%m%d')
    month_start = datetime.datetime(cur_date.year, cur_date.month, 1).strftime('%Y%m%d')
    month_end = datetime.datetime(
        cur_date.year, cur_date.month, calendar.monthrange(cur_date.year, cur_date.month)[1]).strftime('%Y%m%d')
    tradinglist = get_trading_days(month_start, month_end)
    return tradinglist


def get_yearly_trading_day(year):
    tradinglist = get_trading_days(year + '0101', year + '1231')
    return tradinglist


def expand_stockcode(x):
    if isinstance(x, float):
        x_format = str(int(round(x)))
    elif isinstance(x, int):
        x_format = str(x)
    else:
        x_format = str(x)
    return x_format if len(x_format) == 6 else '0' * (6 - len(x_format)) + x_format


def format_stockcode(code, type=0):
    """
    type eg: {
        0: 000000,
        2: 000000.SZ,
        3: 000000.SZE
    }
    """
    if isinstance(code, float):
        code = str(int(round(code)))
    else:
        code = str(code)


def expend_market(x, suf_num=3, suf_mode=True, scr_typ=None):
    if scr_typ is None:
        if suf_mode:
            if suf_num == 2:
                return x + '.SZ' if x < '600000' else x + '.SH'
            elif suf_num == 3:
                return x + '.SZE' if x < '600000' else x + '.SSE'
            elif suf_num == 4:
                return x + ".SZSE" if x < '600000' else x + '.SHSE'
            else:
                assert False, 'suf_num not in [2, 3, 4]'
        else:
            if suf_num == 2:
                return 'SZ' + x if x < '600000' else 'SH' + x
            else:
                assert False, 'suf_num not in [2, ]'
    elif scr_typ == 'qfii':
        return f'{x} CH Equity'
    elif scr_typ == 'connect':
        return f'{x} C2 Equity' if x < '600000' else f'{x} C1 Equity'
    else:
        raise ValueError


def calculate_pnl_basis(curdate, id_data=None, ret_data_mode='pnl&mv'):
    if isinstance(id_data, str):
        id_data = [id_data]

    gbd = GetBaseData()
    if isinstance(id_data, list):
        predate = get_predate(curdate, 1)
        id_data = gbd.get_details_future_position_data(predate, data_type='Instrument', product=id_data)

    id_data = pd.DataFrame(id_data.set_index('Product')['HedgePosDict'].to_dict()).reset_index().rename({'index': 'Instrument'}, axis='columns')
    id_data = pd.melt(id_data, id_vars='Instrument', var_name='Product', value_name='Volume')

    df_basis = gbd.get_future_basis_data_daily(curdate)
    if df_basis is None:
        if ret_data_mode == 'pnl&mv':
            return 0, 0
        elif ret_data_mode == 'pnl':
            return 0
        else:
            raise 'ValueError'
    df_basis = df_basis.rename({"SecuCode": 'Instrument'}, axis='columns')
    df = pd.merge(id_data, df_basis, on='Instrument', how='left')
    pnl = (df['Volume'] * df['Multiplier'] * df['BasisChange']).sum()
    future_mv = (df['Volume'] * df['Multiplier'] * df['PreSettlePrice']).sum()
    if ret_data_mode == 'pnl&mv':
        return pnl, future_mv
    elif ret_data_mode == 'pnl':
        return pnl
    else:
        raise 'ValueError'

def calculate_pnl_position(date, product, df_price=None, df_position=None, ret_pre_mv=False):
    if df_price is None:
        df_price = get_price(date, date).reset_index()

    if df_position is None:
        df_position = get_position(date, product)[['SecuCode', 'PreCloseVolume', 'Volume']]
        if product in ProductionList_AlphaShort_Short:
            df_position['PreCloseVolume'] *= -1
            df_position['Volume'] *= -1

    df_position = pd.merge(df_position, df_price, on='SecuCode', how='left').fillna(0)
    df_position['PreCloseValue'] = df_position['PreCloseVolume'] * df_position['PreClosePrice']
    df_position['CloseValue'] = df_position['PreCloseVolume'] * df_position['ClosePrice']
    df_position['CurCloseValue'] = df_position['Volume'] * df_position['ClosePrice']

    position_pnl = (df_position['CloseValue'] - df_position['PreCloseValue']).sum()
    if not ret_pre_mv:
        return position_pnl
    else:
        return position_pnl, abs(df_position['CurCloseValue'].sum()), abs(df_position['PreCloseValue'].sum())


def calculate_pnl(curdate, id_data, df_price=None, fee_r=0.00012, return_mode='DF', ret_trade_value=False, stampr=STAMP_RATE, ipo_pnl_mode=False):
    """
    df_record: DataFrame, columns = ['SecuCode', 'Volume', 'LongShort', 'Price']
    """

    if isinstance(id_data, str): df_trades = get_trades(curdate, id_data)
    else: df_trades = id_data.copy(deep=True)

    df_trades = df_trades[['SecuCode', 'Volume', 'LongShort', 'Price']]
    if ret_trade_value: return (df_trades['Volume'] * df_trades['Price']).sum()

    if df_price is None: df_price = get_price(curdate, curdate).reset_index()[['SecuCode', 'ClosePrice', 'PreClosePrice']]
    else: df_price = df_price.copy(deep=True).reset_index()[['SecuCode', 'ClosePrice', 'PreClosePrice']]

    if df_trades.empty:
        if return_mode == 'DF':
            return pd.DataFrame(columns=['SecuCode', 'Long', 'Short', 'PnL'])
        elif return_mode == 'PnL&trades-value':
            return 0, 0
        else:
            return 0

    if not isinstance(fee_r, dict): fee_r = {'Long': fee_r, 'Short': fee_r}

    df_merge = pd.merge(df_trades, df_price, on='SecuCode', how='left')
    df_merge['Volume'] *= (1 - 2 * df_merge['LongShort'])
    ls_flag = df_merge['Volume'] >= 0

    if not ipo_pnl_mode: df_merge['PnL'] = df_merge['Volume'] * (df_merge['ClosePrice'] - df_merge['Price'] * (ls_flag * (1 + fee_r['Long']) + (1 - ls_flag) * (1 - stampr - fee_r['Short'])))
    else: df_merge['PnL'] = - df_merge['Volume'] * (df_merge['Price'] * (ls_flag * (1 + fee_r['Long']) + (1 - ls_flag) * (1 - stampr - fee_r['Short'])) - df_merge['PreClosePrice'])

    df_merge['TradeValue'] = np.abs(df_merge['Volume'] * df_merge['Price'])

    trade_value = df_merge['TradeValue'].sum()
    if return_mode == 'trades-value': return trade_value

    if return_mode == 'DF':
        df_ls_info = df_merge.groupby(['SecuCode', 'LongShort'])['Volume'].sum().to_frame().reset_index()
        df_ls_info = pd.pivot_table(df_ls_info, values='Volume', index='SecuCode', columns='LongShort').rename(
            {0: 'Long', 1: 'Short'}, axis='columns')
        df_merge_pnl = df_merge.groupby('SecuCode')[['PnL']].sum()
        df_concat = pd.concat([df_ls_info, df_merge_pnl], axis=1).fillna(0).reset_index()
        return df_concat[['SecuCode', 'Long', 'Short', 'PnL']]
    elif return_mode == 'DFValue':
        return df_merge.groupby('SecuCode')[['TradeValue', 'PnL']].sum().reset_index()
    elif return_mode == 'PnL&trades-value':
        return trade_value, round(df_merge['PnL'].sum(), 2)
    else:
        return round(df_merge['PnL'].sum(), 2)


def get_position(curdate, Production, filepath=None, return_path=False):
    if filepath is None:
        filepath = f'{PLATFORM_PATH_DICT["z_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'
        filepath += f'{curdate}_position-{Production}.txt'

        if not os.path.exists(filepath):
            filepath = f'{PLATFORM_PATH_DICT["t_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'
            filepath += f'{curdate}_position-{Production}.txt'
    if return_path: return filepath
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        position_df = pd.read_table(filepath, sep=',', header=None).dropna()
        position_df.columns = ['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume']
        position_df['SecuCode'] = position_df['SecuCode'].apply(lambda x: expand_stockcode(x))
        position_df = position_df.groupby(['SecuCode', 'Exchange']).sum().reset_index()
        return position_df
    else:
        return pd.DataFrame(columns=['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume'])


def get_position_af_ex_right(date, df_position):
    file_path_ex_right = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/ex_right_data/{date}_ex_right_data.csv'
    if os.path.exists(file_path_ex_right):
        df_exit_right = pd.read_csv(file_path_ex_right)
        if not df_exit_right.empty:
            df_exit_right['SecuCode'] = df_exit_right['SecuCode'].apply(lambda x: expand_stockcode(x))
            df_position = pd.merge(df_position, df_exit_right, on='SecuCode', how='left')
            
            df_position['Volume'] += df_position['PreCloseVolume'] * (df_position['ExRightRatio'].fillna(1) - 1)
            df_position['Volume'] = np.round(df_position['Volume']).astype('int')

    return df_position


def get_position_cal(curdate, Production, filepath=None):
    if filepath is None:
        filepath = f'{PLATFORM_PATH_DICT["z_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/CalPosition/{curdate}/'
        filepath += f'{curdate}_position-{Production}.txt'

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        position_df = pd.read_table(filepath, sep=',', header=None).dropna()
        position_df.columns = ['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume']
        position_df['SecuCode'] = position_df['SecuCode'].apply(lambda x: expand_stockcode(x))
        position_df = position_df.groupby(['SecuCode', 'Exchange']).sum().reset_index()
        return position_df
    else:
        return pd.DataFrame(columns=['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume'])


def get_position_simu(curdate, Production, filepath=None, simu_type='simu2', return_path=False):
    if filepath is None:
        if simu_type.lower() == 'simu':
            filepath = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/SimulationData/{curdate}/'
        elif simu_type.lower() == 'simu2':
            filepath = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/SimulationData/{curdate}/new-comb/'
        else: raise ValueError

    if os.path.exists(filepath + f'{curdate}_position_simu-{Production}.txt'):
        filepath += f'{curdate}_position_simu-{Production}.txt'
    else:
        filepath += f'{curdate}_position-{Production}.txt'
    print(filepath)
    if return_path: return filepath

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        position_df = pd.read_table(filepath, sep=',', header=None).dropna()
        position_df.columns = ['SecuCode', 'PreCloseVolume', 'YesShort', 'OpenVolume']
        position_df['Volume'] = position_df['OpenVolume'] + position_df['YesShort']
        position_df['Exchange'] = position_df['SecuCode'].apply(lambda x: {'SH': 'SSE', 'SZ': 'SZE'}[x[:2]])
        position_df['SecuCode'] = position_df['SecuCode'].apply(lambda x: x[2:])

        position_df = position_df.groupby(['SecuCode', 'Exchange']).sum().reset_index()
        return position_df[['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume']]
    else:
        return pd.DataFrame(columns=['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume'])


def get_position_backtest(curdate, Production, filepath):
    filepath += f'{curdate}_position-{Production}.txt'
    print(filepath)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        position_df = pd.read_table(filepath, sep=',', header=None)
        position_df.columns = ['Exchange', 'SecuCode', 'PreCloseVolume', 'OpenVolume', 'Volume']
        position_df['SecuCode'] = position_df['SecuCode'].apply(lambda x: expand_stockcode(x))
        position_df = position_df.groupby(['SecuCode', 'Exchange']).sum().reset_index()
        position_df = position_df[['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume']]
        return position_df
    else:
        return pd.DataFrame(columns=['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume'])


def get_position_backup_first(curdate, Production, filepath=None):
    if filepath is None:
        filepath = f'{PLATFORM_PATH_DICT["z_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'

    filepath_backup = f'{filepath}{curdate}_position-{Production}-backup.txt'
    filepath = f'{filepath}{curdate}_position-{Production}.txt'

    if os.path.exists(filepath_backup) and os.path.getsize(filepath_backup) > 0:
        position_df = pd.read_table(filepath_backup, sep=',', header=None).dropna()
        position_df.columns = ['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume']
        position_df['SecuCode'] = position_df['SecuCode'].apply(lambda x: expand_stockcode(x))
        position_df = position_df.groupby(['SecuCode', 'Exchange']).sum().reset_index()
        return position_df
    elif os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        position_df = pd.read_table(filepath, sep=',', header=None).dropna()
        position_df.columns = ['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume']
        position_df['SecuCode'] = position_df['SecuCode'].apply(lambda x: expand_stockcode(x))
        position_df = position_df.groupby(['SecuCode', 'Exchange']).sum().reset_index()
        return position_df
    else:
        return pd.DataFrame(columns=['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume'])


def get_trades_simu(cur_date, Production, filepath=None, simu_type='simu2', return_path=False):
    if filepath is None:
        if simu_type.lower() == 'simu':
            filepath = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/SimulationData/{cur_date}/'
        elif simu_type.lower() == 'simu2':
            filepath = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/SimulationData/{cur_date}/new-comb/'
        else: raise ValueError

    filepath = filepath + f'{cur_date}-{Production}-trades.txt'

    print(filepath)
    if return_path: return filepath
    if os.path.exists(filepath) and (os.path.getsize(filepath) > 0):
        return get_trades(cur_date, Production, filepath)
    else:
        return pd.DataFrame(columns=Trades_Columns_List)


def get_trades_backtest(cur_date, Production, filepath):
    filepath += f'{cur_date}-{Production}-trades.txt'

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        df = pd.read_table(filepath, sep=',', header=None).dropna()
        df.columns = Trades_Columns_List[:len(df.columns)]

        df['SecuCode'] = df['SecuCode'].apply(lambda x: x[2:])
        df['LongShort'] = df['LongShort'].astype('int')
        df['Date'] = df['Date'].astype('str')
        df = df.reset_index(drop=True)
        try:
            df['Price'] = df['Price'].astype('float')
        except:
            df['Price'] = df['Price'].apply(lambda x: process_df_trades(x, float))
        try:
            df['Volume'] = df['Volume'].astype('int')
        except:
            df['Volume'] = df['Volume'].apply(lambda x: process_df_trades(x, int))

        df = df.dropna(axis=0)
        return df
    else:
        return pd.DataFrame(columns=Trades_Columns_List)


def get_trades_octopus(curdate, product, filepath=None):
    if filepath is None:
        filepath = f'{PLATFORM_PATH_DICT["z_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'
        filepath = filepath + f'{curdate}_T0_trades-{product}.txt'

        if not os.path.exists(filepath):
            filepath = f'{PLATFORM_PATH_DICT["t_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'
            filepath = filepath + f'{curdate}_T0_trades-{product}.txt'
    else:
        filepath = filepath + f'{curdate}_T0_trades-{product}.txt'

    if os.path.exists(filepath) and (os.path.getsize(filepath) > 0):
        df = pd.read_table(filepath, sep=',', header=None).dropna()[[2, 3, 4, 5, 6]]
        df.columns = ['SecuCode', 'LongShort', 'Volume', 'Price', 'Time']
        df['SecuCode'] = df['SecuCode'].apply(lambda x: x.split('.')[0])
        df['Date'] = curdate
        df['Time'] = df['Time'].apply(lambda x: expand_stockcode(math.ceil(float(x.split('T')[1]))))
    else:
        df = pd.DataFrame(columns=['SecuCode', 'LongShort', 'Volume', 'Price', 'Time'])
    return df


def get_trades(cur_date, Production, filepath=None, return_path=False, suffix_name=''):
    columns_list = Trades_Columns_List

    if filepath is None:
        filepath = f'{PLATFORM_PATH_DICT["z_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{cur_date}/'
        filepath = filepath + f'{cur_date}-{production_2_account(Production)}-trades{suffix_name}.txt'
        if not os.path.exists(filepath):
            filepath = f'{PLATFORM_PATH_DICT["t_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{cur_date}/'
            filepath = filepath + f'{cur_date}-{production_2_account(Production)}-trades{suffix_name}.txt'

    if return_path: return filepath
    # assert os.path.exists(filepath), '当日文件未上传！'
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            df = pd.read_table(filepath, sep=',', header=None)
            df = df[[i for i in range(min(len(df.columns), len(columns_list)))]]
            df.columns = columns_list[:len(df.columns)]
            df = df[df['SecuCode'].str.startswith('S')]
            df['SecuCode'] = df['SecuCode'].apply(lambda x: x[2:])
            df['LongShort'] = df['LongShort'].astype('int')
            # df = df[(df['LongShort'] == 0) | (df['LongShort'] == 1)]
            df['Date'] = df['Date'].astype('str')

            df = df.reset_index(drop=True)
            try:
                df['Price'] = df['Price'].astype('float')
            except:
                df['Price'] = df['Price'].apply(lambda x: process_df_trades(x, float))
            try:
                df['Volume'] = df['Volume'].astype('int')
            except:
                df['Volume'] = df['Volume'].apply(lambda x: process_df_trades(x, int))

            df = df.replace(['-nan', 'nan', np.nan], 0)
            return df
        except:
            try:
                trades_list, num_col = [], 7
                with open(filepath, 'r') as f:
                    for fl in f.readlines():
                        i_trades_list = [fli.strip() for fli in fl.split(',')]
                        len_fl = len(i_trades_list)
                        if (len_fl == 7 or len_fl == 9 or len_fl == 13 or len_fl == 14) and len(i_trades_list[2]) > 0:
                            trades_list.append(i_trades_list)
                            num_col = len(i_trades_list)
                if num_col == 7:
                    df_trades = pd.DataFrame(trades_list, columns=columns_list[:7]).dropna()
                elif num_col in [9, 13, 14]:
                    df_trades = pd.DataFrame(trades_list, columns=columns_list[:num_col]).dropna()
                    df_trades['status'] = df_trades['status'].astype('int')
                    df_trades['signal'] = df_trades['signal'].astype('float')

                df_trades['LongShort'] = df_trades['LongShort'].astype('int')
                try:
                    df_trades['Price'] = df_trades['Price'].astype('float')
                except:
                    df_trades['Price'] = df_trades['Price'].apply(lambda x: process_df_trades(x, float))
                try:
                    df_trades['Volume'] = df_trades['Volume'].astype('int')
                except:
                    df_trades['Volume'] = df_trades['Volume'].apply(lambda x: process_df_trades(x, int))

                df_trades = df_trades.dropna(axis=0)
                df_trades['SecuCode'] = df_trades['SecuCode'].apply(lambda x: x[2:])
                df_trades['Date'] = df_trades['Date'].astype('str')
                return df_trades
            except:
                try:
                    trades_list = []
                    with open(filepath, 'r') as f:
                        for fl in f.readlines():
                            i_trades_list = [fli.strip() for fli in fl.split(',')]
                            if len(i_trades_list) >= 7 and len(i_trades_list[0]) == 8 and len(i_trades_list[2]) == 8:
                                try:
                                    i_trades_list[0] = int(i_trades_list[0])
                                    i_trades_list[1] = int(i_trades_list[1])
                                    i_trades_list[3] = int(i_trades_list[3])
                                    i_trades_list[4] = int(i_trades_list[4])
                                    i_trades_list[5] = float(i_trades_list[5])
                                    i_trades_list[6] = int(i_trades_list[6])
                                    trades_list.append(i_trades_list[:7])
                                except:
                                    pass

                    df_trades = pd.DataFrame(trades_list, columns=columns_list[:7]).dropna()
                    df_trades['LongShort'] = df_trades['LongShort'].astype('int')
                    try:
                        df_trades['Price'] = df_trades['Price'].astype('float')
                    except:
                        df_trades['Price'] = df_trades['Price'].apply(lambda x: process_df_trades(x, float))
                    try:
                        df_trades['Volume'] = df_trades['Volume'].astype('int')
                    except:
                        df_trades['Volume'] = df_trades['Volume'].apply(lambda x: process_df_trades(x, int))

                    df_trades = df_trades.dropna(axis=0)
                    df_trades['SecuCode'] = df_trades['SecuCode'].apply(lambda x: x[2:])
                    df_trades['Date'] = df_trades['Date'].astype('str')
                    return df_trades
                except:
                    print(f"{cur_date} {Production} assert False, '这异常不应该出现的'")
                    return pd.DataFrame([], columns=columns_list)
    else:
        return pd.DataFrame([], columns=columns_list)


def get_trades_bar_mode_new(cur_date, Production, filepath=None):
    columns_list = Trades_Columns_List_New

    if filepath is None:
        filepath = f'{PLATFORM_PATH_DICT["z_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{cur_date}/'
        filepath = filepath + f'{cur_date}_T0_trades-{Production}_new.txt'

        if not os.path.exists(filepath):
            filepath = f'{PLATFORM_PATH_DICT["t_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{cur_date}/'
            filepath = filepath + f'{cur_date}_T0_trades-{Production}_new.txt'

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            df = pd.read_table(filepath, sep=',', header=None).dropna()
            df = df[[i for i in range(min(len(df.columns), len(columns_list)))]]
            df.columns = columns_list[:len(df.columns)]
            df['SecuCode'] = df['SecuCode'].apply(lambda x: expand_stockcode(x))
            df['LongShort'] = df['LongShort'].astype('int')
            try:
                df['Price'] = df['Price'].astype('float')
            except:
                df['Price'] = df['Price'].apply(lambda x: process_df_trades(x, float))
            try:
                df['Volume'] = df['Volume'].astype('int')
            except:
                df['Volume'] = df['Volume'].apply(lambda x: process_df_trades(x, int))

            df = df.dropna(axis=0).reset_index(drop=True)
            return df
        except:
            try:
                trades_list, num_col = [], 6
                with open(filepath, 'r') as f:
                    for fl in f.readlines():
                        i_trades_list = [fli.strip() for fli in fl.split(',')]
                        len_lines = len(i_trades_list)
                        if len_lines in [6, 8, 12, 13] and len(i_trades_list[2]) > 0:
                            trades_list.append(i_trades_list)
                            num_col = len(i_trades_list)
                if num_col == 6:
                    df_trades = pd.DataFrame(trades_list, columns=columns_list[:num_col]).dropna()
                elif num_col in [8, 12, 13]:
                    df_trades = pd.DataFrame(trades_list, columns=columns_list[:num_col]).dropna()
                    df_trades['status'] = df_trades['status'].astype('int')
                    df_trades['signal'] = df_trades['signal'].astype('float')
                df_trades['LongShort'] = df_trades['LongShort'].astype('int')
                try:
                    df_trades['Price'] = df_trades['Price'].astype('float')
                except:
                    df_trades['Price'] = df_trades['Price'].apply(lambda x: process_df_trades(x, float))
                try:
                    df_trades['Volume'] = df_trades['Volume'].astype('int')
                except:
                    df_trades['Volume'] = df_trades['Volume'].apply(lambda x: process_df_trades(x, int))

                df_trades = df_trades.dropna(axis=0)
                df_trades['SecuCode'] = df_trades['SecuCode'].apply(lambda x: expand_stockcode(x))
                return df_trades
            except:
                assert False, '不应该出现这个错误'

    else:
        return pd.DataFrame([], columns=columns_list)


def get_trading_daily_alpha(start_date, end_date, product=None):
    engine = create_engine("mssql+pymssql://mluo:Cfi888_@dbs.cfi:1433/TRDB")
    if product is None:
        data_sql = '''
                    select Date, Production, Daily_alpha
                    from Production_FullSummary
                    where (Date between '%s' and '%s')
                    order by Date, Production;
                    '''
        data_str = data_sql % (start_date, end_date)
    else:
        data_sql = '''
                select Date, Production, Daily_alpha
                from Production_FullSummary
                where (Date between '%s' and '%s') and (Production = '%s')
                order by Date, Production;
                '''
        data_str = data_sql % (start_date, end_date, product)
    df_daily_alpha = pd.read_sql(data_str, engine)
    df_daily_alpha = df_daily_alpha.drop_duplicates(subset=['Date', 'Production'], keep='first')

    df_daily_alpha['Daily_alpha'] = df_daily_alpha['Daily_alpha'].apply(
        lambda x: float(x.replace('%', '')) / 100)

    return df_daily_alpha


def get_production_summary_db(start_date, end_date, product=None, table_name='simple', date_str='Date', prod_str='产品'):
    """
        'simple': 'Production_SimpleSummary',
        'full': 'Production_FullSummary',
        'ret': 'Production_RetSummary',
        'open_short': 'Production_OpenShortRecord',
        'future_trades': 'Production_FutureTradesSummary',
        'cancel_ratio': 'Production_CancelRatioSummary',
    """
    table_name = {
        'simple': 'Production_SimpleSummary',
        'full': 'Production_FullSummary',
        'ret': 'Production_RetSummary',
        'ipo': 'Production_IPO_NewPnLSummary',
        'open_short': 'Production_OpenShortRecord',
        'future_trades': 'Production_FutureTradesPnLSummary',
        'cancel_ratio': 'Production_CancelRatioSummary',
    }.get(table_name, table_name)
    engine = create_engine("mssql+pymssql://mluo:Cfi888_@dbs.cfi:1433/TRDB")
    if product is None:
        data_sql = f'''
                    select *
                    from {table_name}
                    where ({date_str} between '%s' and '%s')
                    order by {date_str}, {prod_str};
                    '''
        data_str = data_sql % (start_date, end_date)
    else:
        data_sql = f'''
                select *
                from {table_name}
                where ({date_str} between '%s' and '%s') and ({prod_str} = '%s')
                order by {date_str}, {prod_str};
                '''
        data_str = data_sql % (start_date, end_date, product)
    df = pd.read_sql(data_str, engine).drop_duplicates(subset=[date_str, prod_str], keep='first')
    return df


def get_netvalue_unit_from_db_simple(start_date, end_date, product):
    engine = create_engine("mssql+pymssql://mluo:Cfi888_@dbs.cfi:1433/TRDB")
    if product is None:
        data_sql = '''
                    select Date, 
                    产品 as Product, 
                    昨日净值 as PreNetValue
                    from Production_SimpleSummary
                    where (Date between '%s' and '%s')
                    order by Date, 产品;
                    '''
        data_str = data_sql % (start_date, end_date)
    else:
        data_sql = '''
                select Date, 
                产品 as Product, 
                昨日净值 as PreNetValue
                from Production_SimpleSummary
                where (Date between '%s' and '%s') and (产品 = '%s')
                order by Date, 产品;
                '''
        data_str = data_sql % (start_date, end_date, product)
    df = pd.read_sql(data_str, engine)
    return df


def wechat_bot_msg(msg, mention=None, type_api='check'):
    if mention is None:
        mention = []

    if WECHAT_BOT_KEY_DICT.get(type_api, None) is not None:
        key = WECHAT_BOT_KEY_DICT[type_api]
        url = f'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}'
    else:
        raise 'ValueError'
    # 企业微信建群后在群里创建机器人，将页面的hook地址复制过来
    data = {
        "msgtype": "text",
        "text": {
            "content": msg,
            "mentioned_list": mention,
        }
    }
    a = requests.post(url, json=data)
    print(a)


def wechat_bot_msg_check(msg, mention=None, type_api='check'):
    if mention is None:
        mention = []

    if WECHAT_BOT_KEY_DICT.get(type_api, None) is not None:
        key = WECHAT_BOT_KEY_DICT[type_api]
        url = f'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}'
    else:
        raise 'ValueError'
    # 企业微信建群后在群里创建机器人，将页面的hook地址复制过来
    data = {
        # "touser": '|'.join(mention),
        "msgtype": "text",
        "text": {
            "content": msg,
        }
    }
    a = requests.post(url, json=data)
    print(a)


def wechat_bot_image(image_path, type_api='check'):
    import base64
    import hashlib

    with open(image_path, 'rb') as file:
        data = file.read()
        encodestr = base64.b64encode(data)
        image_data = str(encodestr, 'utf-8')

    with open(image_path, 'rb') as file:
        md = hashlib.md5()
        md.update(file.read())
        image_md5 = md.hexdigest()

    if WECHAT_BOT_KEY_DICT.get(type_api, None) is not None:
        key = WECHAT_BOT_KEY_DICT[type_api]
        url = f'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}'
    else:
        raise 'ValueError'

    data = {
        "msgtype": "image",
        "image": {
            "base64": image_data,
            "md5": image_md5,
        }
    }
    a = requests.post(url, json=data)
    print(a)


def wechat_bot_markdown(content, type_api='check'):
    if WECHAT_BOT_KEY_DICT.get(type_api, None) is not None:
        key = WECHAT_BOT_KEY_DICT[type_api]
        url = f'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}'
    else:
        raise 'ValueError'

    data = {
        "msgtype": "markdown",
        "markdown": {
            "content": content
        }
    }
    a = requests.post(url, json=data)
    print(a)


def wechat_bot_file(file_path, type_api='check'):
    if WECHAT_BOT_KEY_DICT.get(type_api, None) is not None:
        upload_url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={WECHAT_BOT_KEY_DICT[type_api]}&type=file"
        msg_url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={WECHAT_BOT_KEY_DICT[type_api]}&type=file"
    else:
        raise 'ValueError'

    if not isinstance(file_path, list):
        file_path = [file_path]

    for file_path_i in file_path:
        data = {"file": open(file_path_i, "rb")}
        response = requests.post(url=upload_url, files=data)
        json_resp = response.json()
        media_id = json_resp["media_id"]

        msg = {
            "msgtype": "file",
            "file": {
                "media_id": media_id
            },
            "mentioned_list": [],
        }
        r = requests.post(url=msg_url, json=msg)
        print(r)


def get_pta_report(start_date, end_date):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/DataSupply")

    data_sql = '''
        select * from
        dbo.stock_pta_report
        where date between '%s' and '%s'
    '''

    data_str = data_sql % (start_date, end_date)
    df = pd.read_sql(data_str, engine)
    return df


def get_1mindata(cur_date):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/DataSupply")

    data_sql = '''
         select * from dbo.Interval1min
         where date = '%s';
        '''

    data_str = data_sql % (cur_date)
    df = pd.read_sql(data_str, engine)
    return df


def get_5mindata(cur_date):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/DataSupply")

    data_sql = '''
         select * from dbo.Interval5min
         where date = '%s';
        '''

    data_str = data_sql % (cur_date)
    df = pd.read_sql(data_str, engine)
    return df


def get_30mindata(cur_date):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/DataSupply")

    data_sql = '''
         select * from dbo.Interval30min
         where date = '%s';
        '''

    data_str = data_sql % (cur_date)
    df = pd.read_sql(data_str, engine)
    return df


def get_30min0935_data(cur_date):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/DataSupply")

    data_sql = '''
         select * from dbo.Interval30min_V2
         where date = '%s';
        '''

    data_str = data_sql % (cur_date)
    df = pd.read_sql(data_str, engine)
    return df


def get_price_marketing_5min(curdate, paras_name='close'):
    filepath = f'{PLATFORM_PATH_DICT["z_path"]}mdsync/binIntervalRefresh/data/{curdate}/'
    code_list = [expand_stockcode(code) for code in
                 pd.read_csv(f'{PLATFORM_PATH_DICT["z_path"]}homes/wenzan/instrument.csv', header=None)[0].tolist()]

    outputdir = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/realtime_quota_vwap/{curdate}/'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    columns_list = ['093000',
                    '093500', '094000', '094500', '095000', '095500', '100000',
                    '100500', '101000', '101500', '102000', '102500', '103000',
                    '103500', '104000', '104500', '105000', '105500', '110000',
                    '110500', '111000', '111500', '112000', '112500', '113000',
                    '130500', '131000', '131500', '132000', '132500', '133000',
                    '133500', '134000', '134500', '135000', '135500', '140000',
                    '140500', '141000', '141500', '142000', '142500', '143000',
                    '143500', '144000', '144500', '145000', '145500', '150000']

    df_5min_price = pd.DataFrame(np.memmap(
        filepath + f'{paras_name}.bin', dtype=np.float64, shape=(49, len(code_list))).T,
                                  index=code_list, columns=columns_list).dropna(axis=1, how='all')
    return df_5min_price


def get_market_value(curdate):
    engine_wddb = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/WDDB")
    data_sql_wddb = '''
            select 
            S_INFO_WINDCODE as SecuCode,
            S_VAL_MV*10000 as cap,
            S_DQ_MV*10000 as negcap
            from ASHAREEODDERIVATIVEINDICATOR
            
            where TRADE_DT = '%s'
         '''

    data_sql_wddb = data_sql_wddb % curdate

    df_wddb = pd.read_sql(data_sql_wddb, engine_wddb)
    df_wddb['SecuCode'] = df_wddb['SecuCode'].apply(lambda x: x.split('.')[0])
    df_wddb['QueryDate'] = curdate

    return df_wddb


def get_price(start_date, end_date, source='jy'):
    curdate = datetime.datetime.now().strftime('%Y%m%d')
    if (curdate == end_date) and (start_date == end_date):
        source = 'wd'
    else:
        source = source

    if source == 'wd':
        engine_wddb = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/WDDB")
        data_sql_wddb = '''
         select 
         S_DQ_PRECLOSE as PreClosePrice, 
         S_DQ_CLOSE as ClosePrice, 
         S_DQ_OPEN as OpenPrice,
         S_DQ_HIGH as HighPrice,
         S_DQ_LOW as LowPrice, 
         S_DQ_AVGPRICE as VWAP, 
         S_DQ_PCTCHANGE as CloseChange,
         S_DQ_VOLUME as MVolume,
         S_DQ_AMOUNT as Turnover,
         convert(varchar(10),TRADE_DT,112) as QueryDate,
         LEFT(S_INFO_WINDCODE,6) as SecuCode
         from AShareEODPrices 
         where TRADE_DT between '%s' and '%s' 
         order by TRADE_DT;
         '''
        data_str_wddb = data_sql_wddb % (start_date, end_date)
        df_price = pd.read_sql(data_str_wddb, engine_wddb, index_col=['QueryDate', 'SecuCode'])
        df_price['MVolume'] *= 100
        df_price['Turnover'] *= 1000
    elif source == 'jy':
        engine = create_engine("mssql+pymssql://yangwh:yangwenhui888_@dbs.cfi:1433/JYDB")
        data_sql = '''
            select 
        	    convert(varchar(10),QueryDate,112) as QueryDate,
				PrevClosePrice as PreClosePrice, 
                TurnoverVolume as MVolume,
                TurnoverValue as Turnover,
				ClosePrice, OpenPrice, HighPrice, LowPrice, SecuCode
            from Table_StockDailyQuote
        		where QueryDate between '%s' and '%s' 
        	order by QueryDate;
            '''
        data_sql = data_sql % (start_date, end_date)
        df_price = pd.read_sql(data_sql, engine, index_col=['QueryDate', 'SecuCode'])
        df_price['VWAP'] = df_price['Turnover'] / df_price['MVolume'].replace(0, np.nan)
        df_price['CloseChange'] = (df_price['ClosePrice'] - df_price['PreClosePrice']) / df_price['PreClosePrice'] * 100
    else:
        raise 'ValueError'

    engine_jydb = "mssql+pymssql://ht:huangtao@dbs.cfi:1433/JYDB?charset=GBK"
    data_sql_jydb = '''
    select convert(varchar(10),QueryDate,112)
    as QueryDate, SecuCode, STFlag, Uplimit, Downlimit 
    from Table_StockTradePriceLimitation
    where QueryDate between '%s' and '%s' 
    order by QueryDate,SecuCode;
    '''
    data_str_jydb = data_sql_jydb % (start_date, end_date)
    df_jydb = pd.read_sql(data_str_jydb, engine_jydb, index_col=['QueryDate', 'SecuCode'])

    df_price = pd.merge(df_price, df_jydb, how='left', left_index=True, right_index=True)
    df_price['TradeState'] = df_price['MVolume'] != 0

    return df_price


def get_1minindex(cur_date):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/DataSupply")

    data_sql = '''
         select * from dbo.Index1min_V3
         where date = '%s'
         '''

    data_str = data_sql % (cur_date)
    df = pd.read_sql(data_str, engine)
    return df


def get_indexprice(start_date, end_date, IndexName='ZZ500'):
    if IndexName_2_IndexSecuCode.get(IndexName, None) is not None:
        SecuCode = IndexName_2_IndexSecuCode[IndexName]
    else:
        raise 'Incorrect Index Name!'
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi:1433/JYDB?charset=GBK")
    data_sql = '''
        declare @QuerySecuCode varchar(10) 
        declare @SttDate smalldatetime 
        declare @EndDate smalldatetime 
        set @QuerySecuCode = '%s' 
        set @SttDate = '%s' 
        set @EndDate = '%s'
        select T1.SecuCode, T1.SecuAbbr, convert(varchar(10),QT.TradingDay,112) 
        as QueryDate, PrevClosePrice, OpenPrice, HighPrice, LowPrice, ClosePrice 
        from QT_IndexQuote QT, 
        (select InnerCode, SecuCode, SecuAbbr 
        from SecuMain 
        SM  
        where SM.SecuCode = @QuerySecuCode and SM.SecuCategory = 4) 
        T1 
        where QT.InnerCode = T1.InnerCode and QT.TradingDay
        between @SttDate and @EndDate 
        order by QT.TradingDay
        '''
    data_str = data_sql % (SecuCode, start_date, end_date)
    df = pd.read_sql(data_str, engine, index_col=['QueryDate'])
    return df


def get_indexprice_all(start_date, end_date):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi:1433/JYDB?charset=GBK")
    data_sql = '''
        declare @QuerySecuCode varchar(10) 
        declare @SttDate smalldatetime 
        declare @EndDate smalldatetime 
        set @SttDate = '%s' 
        set @EndDate = '%s'
        select T1.SecuCode, T1.SecuAbbr, convert(varchar(10),QT.TradingDay,112) 
        as QueryDate, PrevClosePrice, OpenPrice, HighPrice, LowPrice, ClosePrice 
        from QT_IndexQuote QT, 
        (select InnerCode, SecuCode, SecuAbbr 
        from SecuMain 
        SM  
        where SM.SecuCategory = 4) 
        T1 
        where QT.InnerCode = T1.InnerCode and QT.TradingDay
        between @SttDate and @EndDate 
        order by QT.TradingDay
        '''
    data_str = data_sql % (start_date, end_date)
    df = pd.read_sql(data_str, engine, index_col=['QueryDate'])
    df = df[df['SecuCode'].isin(IndexSecuCode_2_IndexName_FutureName.keys())]
    df['IndexName'] = df['SecuCode'].apply(lambda x: IndexSecuCode_2_IndexName_FutureName[x][0])
    return df


def get_futureprice(startdate, enddate):
    """
    获取指定日期，期货价格
    :param InputDate: 查询日期
    :param IndexName: 指数名称
    :return:
    """
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi:1433/WDDB")
    data_sql = '''
    select 
        S_INFO_WINDCODE as SecuCode, 
        convert(int,TRADE_DT) as QueryDate, 
        S_DQ_PRESETTLE as PreSettlePrice, 
        S_DQ_OPEN as OpenPrice,
        S_DQ_HIGH as HighPrice,
        S_DQ_LOW as LowPrice,
        S_DQ_CLOSE as ClosePrice,
        S_DQ_SETTLE as SettlePrice
    from CINDEXFUTURESEODPRICES 
    where convert(int,TRADE_DT) between %s and %s 
    order by S_INFO_WINDCODE;
    '''
    data_str = data_sql % (startdate, enddate)

    df = pd.read_sql(data_str, engine)
    # df = df[df['SecuCode'].apply(lambda x: x[:2] == index_2_futurecode[IndexName])].reset_index(drop=True)
    df['SecuCode'] = df['SecuCode'].apply(lambda x: x.split('.')[0])

    return df


def get_securities_lending():
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi:1433/WDDB")
    data_sql = """
        select 
        T1.S_INFO_WINDCODE as SecuCode, 
        T1.S_MARGIN_MARGINRATE as MarginRate, 
        T1.S_MARGIN_ELIMINDATE as EliminDate, 
        T2.S_MARGIN_CONVERSIONRATE as ConversionRate
        from
        (select S_INFO_WINDCODE, S_MARGIN_MARGINRATE,S_MARGIN_CONVERSIONRATE,S_MARGIN_ELIMINDATE
        from AShareMarginSubject
        where S_IS_NEW = 1
        and S_MARGIN_SHARETYPE = 244000002
        and (S_INFO_WINDCODE like '6%' or S_INFO_WINDCODE like '3%' or S_INFO_WINDCODE like '0%')) T1
        left join 
        (select S_INFO_WINDCODE, S_MARGIN_MARGINRATE,S_MARGIN_CONVERSIONRATE,S_MARGIN_ELIMINDATE
        from AShareMarginSubject
        where S_IS_NEW = 1
        and S_MARGIN_SHARETYPE = 244000003
        and (S_INFO_WINDCODE like '6%' or S_INFO_WINDCODE like '3%' or S_INFO_WINDCODE like '0%')) T2
        on T1.S_INFO_WINDCODE = T2.S_INFO_WINDCODE
    """

    df = pd.read_sql(data_sql, engine)
    df['SecuCode'] = df['SecuCode'].apply(lambda x: x.split('.')[0])
    df = df[df['EliminDate'].isnull()]
    print(df.head())

    return df


def get_ipo_ret_details(start_date, end_date):
    engine = create_engine("mssql+pymssql://mluo:Cfi888_@dbs.cfi:1433/TRDB?charset=GBK")
    sql = """
        select *
        from Production_IPO_NewPnLSummary
        where date between %s and %s
    """

    df = pd.read_sql(sql % (start_date, end_date), engine)
    # df['SecuCode'] = df['SecuCode'].apply(lambda x: x.split('.')[0])
    return df


def get_ipo_code_list(start_date, end_date):
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi:1433/JYDB?charset=GBK")
    sql = """
        select 
        SecuCode, SecuAbbr,SecuMarket, SecuCategory,
        convert(varchar(10),ListedDate,112) as ListedDate
        from SecuMain
        where ((SecuMarket = 83) or (SecuMarket = 90)) and ((SecuCategory = 1) or (SecuCategory = 41)) and (ListedDate between '%s' and '%s')
        order by ListedDate
    """

    df = pd.read_sql(sql % (start_date, end_date), engine)
    return {date: df['SecuCode'].to_list() for date, df in df.groupby('ListedDate')}


def get_secucode_id_info():
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi:1433/JYDB?charset=GBK")
    sql = """
        select  
        SecuCode, ISIN, EngName
        from SecuMain
        where ((SecuMarket = 83) or (SecuMarket = 90)) and ((SecuCategory = 1) or (SecuCategory = 41))
    """

    df = pd.read_sql(sql, engine)
    return df


