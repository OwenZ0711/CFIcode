# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:48:28 2022

@author: rtchg
"""
import os.path

import pandas as pd
import numpy as np
import datetime
import calendar
import shutil
import sys
import json
import warnings
from toolmodules.modules_constant import *

if 'win' in sys.platform:
    PLATFORM_PATH_DICT = {
        'v_path': 'V:/',
        'z_path': 'Z:/',
        'u_path': 'U:/',
        't_path': 'T:/',
        'w_path': 'W:/',
        'y_path': 'Y:/',
        'm_path': 'Z:/webmonitor_data/MarketPath/',
        's_path': 'S:/',
        'x_path': 'X:/',
    }
else:
    PLATFORM_PATH_DICT = {
        'v_path': '/mnt/nas-3/',
        'z_path': '/mnt/nas-3/',
        'u_path': '/mnt/nas-v/',
        't_path': '/mnt/nas-3.old/',
        'w_path': '/mnt/nas-1-market/',
        'y_path': '/mnt/nas-1-market/',
        'm_path': '/mnt/nas-3/webmonitor_data/MarketPath/',
        's_path': '/mnt/nas-6/',
        'x_path': '//marketing.cfi/',
    }

if not os.path.exists(PLATFORM_PATH_DICT["y_path"]): PLATFORM_PATH_DICT['y_path'] = PLATFORM_PATH_DICT['m_path']

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

CURDATE = datetime.datetime.now().strftime('%Y%m%d')
MARKET_SUMMARY_PATH = f'{PLATFORM_PATH_DICT["y_path"]}监控/%s/Summary.xlsx'
SWAP_DETAILS_PATH = f'{PLATFORM_PATH_DICT["y_path"]}22、收益互换成本汇总统计&黑名单/'
VALUATION_STATEMENT_PATH = f'{PLATFORM_PATH_DICT["y_path"]}下载/估值表/'
TRUSTEESHIP_PATH = f'{PLATFORM_PATH_DICT["y_path"]}托管/'
DATA_PATH_SELF = f'{PLATFORM_PATH_DICT["z_path"]}webmonitor_data/'

CALENDAR_CSV_PATH = f'{DATA_PATH_SELF}calendar.csv'
REPORT_PATH = f'{DATA_PATH_SELF}report.xls'

AUTOEXEC_LOG_PATH = f'{PLATFORM_PATH_DICT["v_path"]}StockData/Trading/AutoExecLog/'
LOG_TEMP_PATH = f'{PLATFORM_PATH_DICT["v_path"]}StockData/Trading/TempLogDir/{CURDATE}/'
BACKTEST_PATH = f'{PLATFORM_PATH_DICT["s_path"]}StockData/Backtest_Data_wf/post-trade/'

if not os.path.exists(LOG_TEMP_PATH):
    try: os.makedirs(LOG_TEMP_PATH)
    except: pass


try:
    with open(f'{PLATFORM_PATH_DICT["v_path"]}stock_root/StockData/Trading/AnalysisTrading/config/adjust_paras_special.json', 'r', encoding='utf-8') as f:
        __config_json = json.load(f)
        MIX_PROPORTION_CHANGE_DICT = __config_json['mix_proportion_change_dict']

    with open(f'{PLATFORM_PATH_DICT["v_path"]}stock_root/StockData/Trading/AnalysisTrading/config/test.json', 'r', encoding='utf-8') as f:
        SimuDict = json.load(f)

    with open(f'{PLATFORM_PATH_DICT["v_path"]}stock_root/StockData/Trading/AnalysisTrading/config/setting.json', 'r', encoding='utf-8') as f:
        __config_json_setting = json.load(f)
        MIX_INDEX_PROPORTION_DEFAULT_DICT = __config_json_setting['proportion_default']
        MIX_INDEX_VERSION_DICT = __config_json_setting['mix_index_version_dict']
        Dict_FutureNameReplace = __config_json_setting['product_future_name_replace']
        Production_NetValueMatch = __config_json_setting['product_netvalue_name_replace']
        Test_ProductionList = __config_json_setting['product_test']
        ProductionList_AlphaShort = __config_json_setting['product_ls']
        ProductionList_AlphaShort_Short = __config_json_setting['product_ls_short']
        Third_Exec_ProductionList = __config_json_setting['product_exec_third']
        CDR_SECUCODE_LIST = __config_json_setting['cdr_secucode_list']
        Production_OwnList_Swap = __config_json_setting['product_swap']
        Production_OptionList = __config_json_setting['product_option']
        Production_OwnList = __config_json_setting['product_own']
        Production_HedgingList = __config_json_setting['product_hedging']
        Production_DMAList_CapitalNotEqual = __config_json_setting['product_dma_capital_neq']
        Production_SpecialList_Delete = __config_json_setting['product_clear']
        SZ_SH_Transfer = __config_json_setting['hs_trans_broker']
        NonAuto_TransIn_BrokerList = __config_json_setting['non_auto_tranin_broker']

        REPORT_REPLACE_DICT = __config_json_setting['report_formatted_dict']
        IndexName_Long_2_Short = __config_json_setting['indexname_long_2_short']
        FutureNameList = __config_json_setting['future_name_list']
        IndexNameList = __config_json_setting['index_name_list']

        MIX_INDEX_VERSION_DICT_NAME_2_PROPORTION = {MIX_INDEX_VERSION_DICT[porp]: porp for porp in MIX_INDEX_VERSION_DICT.keys()}
        Dict_ProductionName_Replace = {Dict_FutureNameReplace[__key]: __key for __key in Dict_FutureNameReplace}
        Production_OwnList += Production_OwnList_Swap + ProductionList_AlphaShort

    with open(f'{PLATFORM_PATH_DICT["v_path"]}stock_root/StockData/Trading/AnalysisTrading/config/bank_info.json', 'r', encoding='utf-8') as f:
        __config_json_bank_info = json.load(f)
        BankName_2_SimpleBankName_CT = __config_json_bank_info['bank_2_simple_bankname_ct']
        BankName_2_SimpleBankName_ZS = __config_json_bank_info['bank_2_simple_bankname_zs']
        BankName_2_SimpleBankName_GS = __config_json_bank_info['bank_2_simple_bankname_gs']
        BankName_2_SimpleBankName_HA = __config_json_bank_info['bank_2_simple_bankname_ha']

        Production_2_BankPwd = __config_json_bank_info['production_2_bankpwd']

        BankCode_2_BankName = __config_json_bank_info['bankcode_2_bankname_csc']
        BankCode_2_BankName_CT = __config_json_bank_info['bankcode_2_bankname_ct']
        BankName_2_BankCode = {value: key for key, value in BankCode_2_BankName.items()}
        BankName_2_BankCode_CT = {value: key for key, value in BankCode_2_BankName_CT.items()}
except json.JSONDecodeError:
    assert False, 'JSON文件填写问题，请注意双引号和行末逗号的使用！'


T0_LendProductionList = []
T0_SwapProductionList = []
T0_ProductionList = T0_SwapProductionList + T0_LendProductionList


def get_production_2_index_production_2_strategy_dict(fpath=REPORT_PATH, return_df=False):
    print('正在读取', fpath)
    df_report = pd.read_excel(fpath, dtype='str').rename({'佣金（包含经手费/证管费）': '佣金'}, axis='columns')

    MulAcc_Main_Dict = Production_MulAccDC_Main_Dict
    MulAcc_Main_Dict.update(Production_MulAccZQ_Main_Dict)
    MulAcc_Main_Dict.update(Production_MulAccYX_Main_Dict)

    df_check_futacc = df_report.copy(deep=True)[['内部代码', '期货账户']].rename(
        {'内部代码': 'Product', '期货账户': 'FutAcc'}, axis='columns')
    df_check_futacc['MainFut'] = df_check_futacc['Product'].apply(lambda x: MulAcc_Main_Dict.get(x, x))
    df_check_futacc['FutAcc'] = df_check_futacc['FutAcc'].fillna('nan')
    df_check_futacc = df_check_futacc[df_check_futacc['Product'].isin(WinterFallProductionList)]
    df_check_futacc['FutAccNum'] = df_check_futacc['FutAcc'].apply(
        lambda x: df_check_futacc['FutAcc'].value_counts().to_dict()[x])
    df_check_futacc['MainFutNum'] = df_check_futacc['MainFut'].apply(
        lambda x: df_check_futacc['MainFut'].value_counts().to_dict()[x])
    df_check_futacc = df_check_futacc[~df_check_futacc['Product'].isin(Production_OwnList_Swap)]

    df_check_futacc_neq = df_check_futacc[
        (df_check_futacc['FutAccNum'] != df_check_futacc['MainFutNum']) & (~ df_check_futacc['FutAcc'].isin(['nan', '开立中']))]
    check_futacc_nan = df_check_futacc[
        (df_check_futacc['FutAcc'] == 'nan') &
        (~ df_check_futacc['Product'].isin(Production_OwnList_Swap + Production_OptionList + ProductionList_AlphaShort))]['Product'].to_list()

    check_futacc_nan = list(set(check_futacc_nan) - set(T0_ProductionList))
    print(f'没有期货户产品： {check_futacc_nan}')
    if not df_check_futacc_neq.empty: print(df_check_futacc_neq)

    if fpath == REPORT_PATH: assert df_check_futacc_neq.empty, '期货户多账户，存在问题，请查看！'

    df_report = df_report.replace(REPORT_REPLACE_DICT)
    df_report['混合比例'] = df_report['混合比例'].fillna('').str.strip()
    df_report['对标指数'] = df_report.apply(lambda row: MIX_INDEX_VERSION_DICT.get(row['混合比例'], row['对标指数']), axis=1)
    df_report['佣金'] = df_report['佣金'].fillna('万1.2')
    df_report['佣金'] = df_report['佣金'].apply(
        lambda x: round(float(x.replace('万', '').replace('..', '.').replace('，', ',').split(',')[0].strip()) / 10000, 7))
    df_report['打新'] = df_report['打新'].fillna(0).astype('float')

    df_report = df_report.replace([np.nan, -np.nan], '')

    if return_df: return df_report

    production_report_list = list(df_report['内部代码'].unique())

    production_2_proportion = {}
    production_2_feeratio = {}
    production_2_bankname = {}
    production_2_product_name = {}
    production_2_product_name_simple = {}
    production_2_hosting = {}
    production_2_valuation_sheet_path = {}
    production_2_product_name_ipo = {}
    production_2_product_name_sec = {}
    production_2_fee_min_value = {}
    production_2_order_fee_mode = {}

    if '最低收费' not in df_report.columns.to_list(): df_report['最低收费'] = '0'
    if '流量费' not in df_report.columns.to_list(): df_report['流量费'] = '0'

    production_2_index = {}
    if 900 < int(datetime.datetime.now().strftime('%H%M')) < 1600:
        mix_change_proportion_dict = MIX_PROPORTION_CHANGE_DICT.get(CURDATE, {})
    else:
        mix_change_proportion_dict = {}
        # mix_change_proportion_dict = MIX_PROPORTION_CHANGE_DICT.get(CURDATE, {})
    for name, index, prop, fee_ratio, bank_name, product_name_sec, product_name, prod_hosting, product_name_simple, \
            is_ipo, valuation_sheet_path, fee_min_value, order_fee_mode in df_report[
        ['内部代码', '对标指数', '混合比例', '佣金', '托管银行', '产品名称', '原始产品名称', '托管', '产品简称', '打新',
         '估值表目录', '最低收费', '流量费']].values:
        bank_name = bank_name.replace('中国', '')
        if len(index) == 0: continue

        prop = mix_change_proportion_dict.get(prop.replace('.0', ''), prop)
        prop = mix_change_proportion_dict.get(name, prop)
        index = MIX_INDEX_VERSION_DICT.get(prop, index)

        prop_list = prop.split('/')
        len_prop_list = len(prop_list)
        if len_prop_list == 1:
            production_2_proportion[name] = {index: 1} if IndexName_2_IndexSecuCode.get(
                index) is not None else MIX_INDEX_PROPORTION_DEFAULT_DICT
        elif len_prop_list >= 3:
            prop_list = list(map(float, prop_list))
            production_2_proportion[name] = {index_name: proportion for index_name, proportion in
                                             zip(IndexNameList[:len_prop_list], prop_list)}
        else:
            raise ValueError(f'{name} 混合比例{prop}有误，请检查！')

        production_2_feeratio[name] = fee_ratio
        production_2_bankname[name] = bank_name.strip() if bank_name.strip() else '-'
        production_2_hosting[name] = prod_hosting
        production_2_product_name_simple[name] = product_name_simple
        production_2_product_name_sec[name] = product_name_sec
        production_2_valuation_sheet_path[name] = valuation_sheet_path

        if (str(fee_min_value) == 'nan') or (str(fee_min_value) == ''):
            production_2_fee_min_value[name] = 0
        else:
            production_2_fee_min_value[name] = float(fee_min_value.strip())

        if (str(order_fee_mode) == 'nan') or (str(order_fee_mode) == ''):
            production_2_order_fee_mode[name] = 0
        else:
            production_2_order_fee_mode[name] = float(order_fee_mode.strip())

        if name in Production_OwnList_Swap:
            production_2_product_name[name] = product_name_sec
        elif name in ['XHYX1A', 'XHYX1B', 'XHYX1C']:
            production_2_product_name['XHYX1A'] = product_name
        elif name in ['XHYX1B1', 'XHYX1B2', 'XHYX1B3']:
            production_2_product_name['XHYX1B1'] = product_name
        elif name in ['XHYX1C1']:
            production_2_product_name['XHYX1C1'] = product_name
        else:
            if product_name_sec == product_name: production_2_product_name[name] = product_name

        if int(is_ipo) == 1: production_2_product_name_ipo[name] = product_name
        production_2_index[name] = str(index)

    production_2_strategy = {prod: str(strategy) for prod, strategy in df_report[['内部代码', '策略']].values}
    production_2_feeratio['SHORTMSLS1'] = 0.00025
    for simu_prod in SimuDict:
        if SimuDict[simu_prod].get('proportion', None) is not None:
            production_2_proportion[simu_prod] = SimuDict[simu_prod]['proportion']

    for simu_prod in SimuDict:
        if SimuDict[simu_prod].get('index', None) is not None:
            production_2_index[simu_prod] = SimuDict[simu_prod]['index']

    for simu_prod in SimuDict:
        if SimuDict[simu_prod].get('feeratio', None) is not None:
            production_2_feeratio[simu_prod] = SimuDict[simu_prod]['feeratio']

    for prod in ProductionList_AlphaShort:
        production_2_index[prod], production_2_strategy[prod] = 'LS', 'LS'

    for prod in T0_ProductionList:
        production_2_index[prod], production_2_strategy[prod] = 'T0', 'T0'

    production_2_product_name.update(
        {
            'HAIJING': '世纪前沿海镜1号私募证券投资基金',
            'ZJDC12': '世纪前沿量化对冲12号私募证券投资基金',
            'ZJDC8': '世纪前沿量化对冲8号私募证券投资基金',
            'ZHUXI2': '世纪前沿竹溪2号私募证券投资基金',
            'ZQ9': '世纪前沿指数增强9号私募证券投资基金',
            'ZS9B': '世纪前沿量化对冲专享9号私募证券投资基金B',
            'HLYX5': '世纪前沿红利优选5号私募证券投资基金'
        }
    )

    production_2_index['LX1'] = 'ZZ2000'
    production_2_strategy['LX1'] = 'ZQ'
    production_2_proportion['LX1'] = {'ZZ2000': 1}

    production_name_2_production = {production_2_product_name[prod_]: prod_ for prod_ in production_2_product_name.keys()}
    production_name_2_production['世纪前沿量化对冲21号私募证券投资基金'] = 'HXDC21'

    production_name_ipo_2_production = {production_2_product_name_ipo[prod_]: prod_ for prod_ in production_2_product_name_ipo.keys()}

    return production_2_index, production_2_strategy, \
        production_2_proportion, production_2_feeratio, production_2_bankname, \
        production_2_product_name, production_2_hosting, production_2_product_name_simple, \
        production_2_valuation_sheet_path, production_2_product_name_ipo, production_2_product_name_sec, \
        production_2_fee_min_value, production_2_order_fee_mode, production_name_2_production, production_name_ipo_2_production, production_report_list, df_report


class GetProductionParas():
    def __init__(self, curdate: str=None):
        if curdate is None:
            self.config_dir = f'{PLATFORM_PATH_DICT["z_path"]}webmonitor_data/trading-accounts.cfg'
            self.report_path = f'{PLATFORM_PATH_DICT["z_path"]}webmonitor_data/report.xls'
        else:
            self.config_dir = f'{PLATFORM_PATH_DICT["z_path"]}webmonitor_data/{curdate}/trading-accounts.cfg'
            self.report_path = f'{PLATFORM_PATH_DICT["z_path"]}webmonitor_data/{curdate}/report.xls'

    def __get_trading_accs(self):
        cfg = {}
        with open(self.config_dir) as f:
            for fi in f.readlines():
                if fi.strip():
                    key, value = fi.strip().split('=')
                    cfg[key] = tuple(value.strip()[1:-1].split())

        prod_list = list(cfg['productions'] + cfg['dualcenter_productions'])
        acc_list = list(cfg['accounts'] + cfg['dualcenter_accounts'])

        assert len(prod_list) == len(acc_list), 'trading-accounts.cfg has errors!'

        production_2_acc = {key: value for key, value in zip(prod_list, acc_list) if ('SIMU' not in key) and ('NEWCOMBO' not in key)}
        acc_2_production = {value: key for key, value in production_2_acc.items()}
        dualcenter_production_dict = {key: value for key, value in zip(cfg['dualcenter_productions'], cfg['dualcenter_accounts'])}
        product_list = list(production_2_acc.keys())
        dict_info = {
            'product_list': product_list,
            'product_list_lower': [pl.lower() for pl in product_list],
            'product_2_acc': production_2_acc,
            'acc_2_product': acc_2_production,
            'dual_product': cfg['dualcenter_productions'],
            'product_2_acc_dual': dualcenter_production_dict,
            'product_alpha_list': [prod for prod in product_list if (prod not in T0_ProductionList + Test_ProductionList + ProductionList_AlphaShort)]
        }
        return dict_info

    def get_product_accounts_list_infor(self, return_mode='arg_list'):
        dict_info = self.__get_trading_accs()
        if return_mode == 'arg_dict': return dict_info
        return dict_info['product_list'], dict_info['product_list_lower'], dict_info['product_alpha_list'], dict_info['product_2_acc'], dict_info['acc_2_product'], dict_info['dual_product'], dict_info['product_2_acc_dual']

    def get_product_info(self):
        (production_2_index, production_2_strategy,
        production_2_proportion, production_2_feeratio, production_2_bankname,
        production_2_product_name, production_2_hosting, production_2_product_name_simple,
        production_2_valuation_sheet_path, production_2_product_name_ipo, production_2_product_name_second,
        production_2_fee_min_value, production_2_order_fee_mode, production_name_2_production,
        production_name_ipo_2_production, production_report_list, df_report) = get_production_2_index_production_2_strategy_dict(self.report_path)
        return {
            'index_dict': production_2_index,
            'strategy_dict': production_2_strategy,
            'proportion_dict': production_2_proportion,
            'fee_ratio_dict': production_2_feeratio,
            'fee_min_value_dict': production_2_fee_min_value,
            'order_fee_mode_dict': production_2_order_fee_mode,

            'hosting_dict': production_2_hosting,
            'bank_name_dict': production_2_bankname,
            'valuation_sheet_path_dict': production_2_valuation_sheet_path,

            'product_name_simple_dict': production_2_product_name_simple,
            'product_name_second_dict': production_2_product_name_second,
            'product_name_dict': production_2_product_name,
            'product_name_reverse_dict': production_name_2_production,
            'product_name_ipo_dict': production_2_product_name_ipo,
            'product_name_ipo_reverse_dict': production_name_ipo_2_production,

            'product_report_list': production_report_list,
            'report_df': df_report
        }

__get_product_paras = GetProductionParas()
(WinterFallProductionList, WinterFallProductionList_Lower, WinterFallProductionList_Alpha,
 PRODUCTION_2_ACCOUNT, ACCOUNT_2_PRODUCTION, DUALCENTER_PRODUCTION, DUALCENTER_PRODUCTION_DICT) = __get_product_paras.get_product_accounts_list_infor()

(PRODUCTION_2_INDEX, PRODUCTION_2_STRATEGY, PRODUCTION_2_PROPORTION, PRODUCTION_2_FEE_RATIO, PRODUCTION_2_BANKNAME,
PRODUCTION_2_PRODUCT_NAME, PRODUCTION_2_HOSTING, PRODUCTION_2_PRODUCT_NAME_SIMPLE,
PRODUCTION_2_VALUATION_SHEET_PATH, PRODUCTION_2_PRODUCT_NAME_IPO, PRODUCTION_2_PRODUCT_NAME_ORIGIN,
PRODUCTION_2_FEE_MIN_VALUE, PRODUCTION_2_ORDER_FEE_MODE, ProductionName_2_Production, ProductionNameIsIPO_2_Production,
PRODUCTION_REPORT_LIST, PRODUCTION_REPORT_DF) = get_production_2_index_production_2_strategy_dict()

def get_production_2_colo_dict():
    df_acc_table = pd.read_csv(f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/AccountTable.csv').dropna()
    # df_acc_table_300 = pd.read_csv(f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/AccountTable_300.csv').dropna()
    df_acc_table_ls = pd.read_csv(f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/AccountTable_ls.csv').dropna()
    df_acc_table = pd.concat([df_acc_table, df_acc_table_ls], axis=0)
    df_acc_table = df_acc_table[df_acc_table['HostName'] != 'xxx']

    target_dict = {prod: hostname for prod, hostname in zip(df_acc_table['Product'], df_acc_table['HostName'])}

    with open(f'{DATA_PATH_SELF}colo_list_trading.txt', 'r') as f: colo_list = list(set(f.read().split(',')) - set(Temp_Colo_Mechine_Dict['simulist'] + ['---']))

    return target_dict, colo_list, {
        'SHORTZJJT1': 'ciccsc-sz-2',
        'ZJJT1': 'ciccsc-sz-2',
        'MSLS1': 'cf-sz-1',
        'SHORTMSLS1': 'cf-sz-1',
        'MSLS2': 'cf-sz-1',
        'SHORTMSLS2': 'cf-sz-1',
    }


PRODUCTION_2_COLO, COLO_HOSTNAME_LIST, COLO_HOSTNAME_LIST_OTHERS = get_production_2_colo_dict()
def production_2_colo(production): return PRODUCTION_2_COLO.get(production, COLO_HOSTNAME_LIST_OTHERS.get(production, '---'))
def production_2_colo_sh(production, colo_sz=None): return (production_2_colo(production) if colo_sz is None else colo_sz).replace('sz', 'sh').replace('dg', 'sh')


def get_predate(cur_date, n, trade_cal=None, ret_df=False):
    assert n != 0, 'n == 0!'
    
    if trade_cal is None:
        trade_cal = pd.read_csv(CALENDAR_CSV_PATH, dtype={'cal_date': 'str'}).sort_values('cal_date')
        trade_cal['DateCount'] = np.cumsum(trade_cal['is_open'])
    else: trade_cal = trade_cal.copy(deep=True)

    if ret_df: return trade_cal
    
    date_2_is_open = {date: (bool(is_open), dcount) for date, is_open, dcount in zip(trade_cal['cal_date'], trade_cal['is_open'], trade_cal['DateCount'])}
    assert date_2_is_open.get(cur_date, None) is not None, f'当前天日期({cur_date})没有找到！'
    
    while not date_2_is_open[cur_date][0]: cur_date = (datetime.datetime.strptime(cur_date, '%Y%m%d') + datetime.timedelta(days=int(np.sign(n)))).strftime('%Y%m%d')

    NCount = date_2_is_open[cur_date][1] - n
    
    assert NCount >= 1, '超出索引范围！'
    PreDate = trade_cal.iloc[np.where(trade_cal['DateCount'] == NCount)[0][0]]['cal_date']

    if n > 0: assert int(PreDate) < int(cur_date), 'int(predate) < int(curdate) = False'
    if n < 0: assert int(PreDate) > int(cur_date), 'int(predate) > int(curdate) = False'

    return PreDate

