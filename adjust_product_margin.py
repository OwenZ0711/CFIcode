import json
import os.path

import numpy as np
import xlwt
from toolmodules.modules import *
from generate_colo_cfg_parameters import Production_D0_Bar_5M


def generate_cfg_config(dict_info, file_name, cfg_output_dir):
    if not os.path.exists(cfg_output_dir): os.makedirs(cfg_output_dir)

    time_flag=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    with open(cfg_output_dir + file_name, 'wb') as f:
        cfg_cont = ''
        for key in dict_info.keys(): cfg_cont += key + '=(' + ' '.join([str(value) for value in dict_info[key]]) + ')\n'
        cfg_cont += f'cfg_time_flag={time_flag}'
        f.write(cfg_cont.encode('utf-8'))

def read_cfg_config(file_path):
    cfg = {}
    if not os.path.exists(file_path): return cfg

    with open(file_path, 'r') as f:
        for fi in f.readlines():
            if fi.strip() and (not fi.startswith('cfg_time_flag')):
                key, value = fi.strip().split('=')
                cfg[key] = tuple(value.strip().replace('(', '').replace(')', '').split())
    return cfg



def calculate_dual_center_cash_ratio_shift(curdate, bar, bar_5m, df_monitor=None, wait_bar=True):
    predate = get_predate(curdate, 1)
    df_price = get_price(predate, predate).reset_index()[['SecuCode', 'ClosePrice']].rename(
        {'SecuCode': 'code'}, axis='columns')

    outputdir = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/TransferDualCenter/{curdate}/'
    if not os.path.exists(outputdir): os.makedirs(outputdir)

    df_accsumm = get_production_list_trading(curdate, ret_df_data=True)
    product_2_bar_mode = {prod: bar_mode for prod, bar_mode in zip(df_accsumm['Account'], df_accsumm['bar'])}
    product_2_holdmv_dict = {prod: holdmv for prod, holdmv in zip(df_accsumm['Account'], df_accsumm['HoldMV'])}
    conlist = []
    for product in WinterFallProductionList:
        if product in T0_ProductionList + Test_ProductionList + ProductionList_AlphaShort: continue

        holdmv = product_2_holdmv_dict.get(product)
        if holdmv is None: continue
        print(product)
        df_pre_close = get_position(predate, product)[['SecuCode', 'Volume']].rename(
            {'SecuCode': 'code', 'Volume': 'volume_bar_0'}, axis='columns')
        df_pre_close['code'] = df_pre_close['code'].apply(lambda x: expand_stockcode(x))
        df_pre_close = df_pre_close.set_index('code')

        bar_mode = product_2_bar_mode.get(product, 8)
        if bar_mode == 8: file_path = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/d0_destpos/{curdate}/{bar}/{curdate}-{product}-destpos.csv'
        elif bar_mode == 48: file_path = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/d0_destpos_5m/{curdate}/{bar_5m}/{curdate}-{product}-destpos.csv'
        else: raise ValueError
        while wait_bar:
            if not os.path.exists(file_path):
                print(f'{file_path} 不存在!')
                time.sleep(2)
            else: break
        df_trade_volume = pd.read_csv(file_path)[['InstrumentID', 'yshort']]
        df_trade_volume.columns = ['code', f'volume_bar']
        df_trade_volume['code'] = df_trade_volume['code'].apply(lambda x: x[2:])
        df_trade_volume = df_trade_volume.set_index('code')

        df_quota_diff = pd.concat([df_pre_close, df_trade_volume], axis=1).fillna(0)
        df_quota_diff[f'Weight_Diff'] = df_quota_diff[f'volume_bar'] - df_quota_diff[f'volume_bar_0']

        df_quota_diff = df_quota_diff.rename({'Weight_Diff': product}, axis='columns').reset_index()
        df_quota_diff = pd.merge(df_quota_diff, df_price, on='code', how='left')
        df_quota_diff[product] *= df_quota_diff['ClosePrice'] / holdmv
        df_quota_diff = df_quota_diff.set_index('code')[[product]]
        conlist.append(df_quota_diff)

    df_exch_chg = pd.concat(conlist, axis=1).fillna(0).reset_index()
    df_exch_chg['Exchange'] = df_exch_chg['code'].apply(lambda x: 'SH-Shift' if x[0] == '6' else 'SZ-Shift')
    df_exch_chg = df_exch_chg.set_index('code').groupby('Exchange').sum().T.reset_index().rename(
        {'index': 'Product'}, axis='columns')

    df_longshort = pd.read_csv(
        f'{PLATFORM_PATH_DICT["v_path"]}StockData/IndexPortfolioFile/Operation/OpenCloseAmount_{curdate}.csv')
    df_longshort = df_longshort[['Account', 'Amount', 'Future']].rename({'Account': 'Product'}, axis='columns')
    df_longshort = df_longshort[~df_longshort['Amount'].isna()]
    if not df_longshort.empty:
        product_2_targetmv = {prod: amount for prod, amount in df_longshort[['Product', 'Amount']].values}
    else:
        product_2_targetmv = {}

    if df_monitor is None:
        df_monitor = GetTradingDependData().get_monitor_stocks_data(curdate, down_json_mode=True)

    df_monitor = df_monitor[['Product', 'MV_net', 'Capital']].rename(
        {"NAME": 'Product'}, axis='columns')
    df_monitor['MV_net'] = df_monitor['MV_net'].astype('float')
    df_monitor['Capital'] = df_monitor['Capital'].astype('float')
    df_monitor = df_monitor[~df_monitor['Product'].isin(T0_ProductionList + Test_ProductionList + ProductionList_AlphaShort)]
    df_monitor = df_monitor[df_monitor['Product'].isin(DUALCENTER_PRODUCTION)]

    df_monitor = df_monitor.groupby('Product').agg({'Capital': 'sum', 'MV_net': 'mean'}).reset_index()
    df_monitor = df_monitor[df_monitor['Product'].isin(product_2_holdmv_dict.keys())]
    df_monitor['Position'] = df_monitor['Product'].apply(lambda x: product_2_holdmv_dict[x])
    df_monitor['IniCapitalR'] = (df_monitor['Capital'] + df_monitor['MV_net']) / df_monitor['Position']
    df_monitor = df_monitor[['Product', 'Position', 'IniCapitalR']]

    df_check = pd.merge(df_monitor, df_exch_chg, on='Product', how='left')
    df_check['IniCapitalR'] *= 100
    df_check['SH-Shift'] *= 100
    df_check['SZ-Shift'] *= 100
    df_check['SH-Left'] = df_check['IniCapitalR'] / 2 - df_check['SH-Shift']
    df_check['SZ-Left'] = df_check['IniCapitalR'] / 2 - df_check['SZ-Shift']
    df_check['Colo'] = df_check['Product'].apply(lambda x: production_2_colo(x))
    df_check['Class'] = df_check['Product'].apply(lambda x: f'{production_2_index(x)}{production_2_strategy(x)}'.replace('-', ''))
    df_check['TradeColo/Account/Class'] = df_check['Product'].apply(lambda x: str(production_2_account(x))) + '/' + \
                                          df_check['Colo'].astype('str') + '/' + df_check['Class'].astype('str')
    df_check['AutoTrans'] = df_check['Colo'].apply(
        lambda x: np.sum([str(x).startswith(colo_) for colo_ in SZ_SH_Transfer]) > 0)

    df_check_lack = df_check[(df_check['SZ-Left'] < 0.0) & (df_check['SH-Left'] < 0.0) & (~ df_check['AutoTrans'])][
        ['Product', 'TradeColo/Account/Class', 'IniCapitalR',
         'SH-Shift', 'SZ-Shift', 'SH-Left', 'SZ-Left']].set_index(['Product']).copy(deep=True)

    lack_product_list = df_check_lack.index.to_list()
    df_check = df_check[
        ((df_check['SZ-Left'] < 0.0) | (df_check['SH-Left'] < 0.0)) &
        (~ df_check['Product'].isin(lack_product_list)) & (~ df_check['AutoTrans'])].copy(deep=True)

    if df_check_lack.empty and df_check.empty:
        msg_capital_shift = f'{curdate}-Ti{bar} DualCenter-Capital-Check: 无异常'
        wechat_bot_msg_check(msg_capital_shift)

        with open(outputdir + f'{curdate}_{bar}_TransferDualCenter_flag.txt', 'w') as f: f.write('')
        return None

    df_check_lack['IniCapitalR'] = np.round(df_check_lack['IniCapitalR'], 2)
    df_check_lack['SH-Shift'] = np.round(df_check_lack['SH-Shift'], 2)
    df_check_lack['SZ-Shift'] = np.round(df_check_lack['SZ-Shift'], 2)
    df_check_lack['SH-Left'] = np.round(df_check_lack['SH-Left'], 2)
    df_check_lack['SZ-Left'] = np.round(df_check_lack['SZ-Left'], 2)

    df_check['Direct'] = (df_check['SH-Left'] < 0).astype('int').apply(
        lambda x: {1: 'to SH', 0: 'to SZ'}[x])
    df_check['TargetMV'] = df_check['Product'].apply(lambda x: product_2_targetmv.get(x, ''))
    df_check['SHL<0'] = df_check['SH-Left'] < 0
    df_check['TransM/R'] = df_check['SHL<0'] * df_check['SZ-Left'] + (1 - df_check['SHL<0']) * df_check['SH-Left'] - \
                           (df_check['SH-Left'] + df_check['SZ-Left']) / 2
    df_check['TransM/R'] = (np.round(df_check['TransM/R'] * df_check['Position'] / 1000000)
                            ).astype('int').astype('str') + 'W/' + np.round(df_check['TransM/R'], 2).astype('str') + '%'

    df_check = df_check[['Product', 'TradeColo/Account/Class', 'TargetMV', 'IniCapitalR', 'TransM/R',
                         'SH-Shift', 'SZ-Shift', 'Direct', 'SH-Left', 'SZ-Left']].set_index('Product')

    df_check['IniCapitalR'] = np.round(df_check['IniCapitalR'], 2)
    df_check['SH-Shift'] = np.round(df_check['SH-Shift'], 2)
    df_check['SZ-Shift'] = np.round(df_check['SZ-Shift'], 2)
    df_check['SH-Left'] = np.round(df_check['SH-Left'], 2)
    df_check['SZ-Left'] = np.round(df_check['SZ-Left'], 2)

    df_check = pd.concat([df_check, df_check_lack], axis=0)
    df_check['LeftLack'] = np.minimum(df_check['SH-Left'], df_check['SZ-Left'])
    df_check = df_check.sort_values('LeftLack', ascending=True).drop('LeftLack', axis=1).reset_index()

    df_check.to_csv(outputdir + f'{curdate}_{bar}_TransferDualCenter.csv', encoding='utf-8-sig', index=False)

    df_check = df_check.fillna('').astype('str').style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]).set_table_styles(
        [{'selector': 'th', 'props': [('border', '1px solid black')]},
         {'selector': 'td', 'props': [('border', '1px solid black')]}]).apply(
            lambda x: ['background-color: {0}'.format('red') if float(v) < -1 else '' for v in x], axis=0,
            subset=['SH-Left', 'SZ-Left'])
    # dfi.export(df_check, filename=outputdir + f'{curdate}_{bar}_TransferDualCenter.png', dpi=300, fontsize=8, max_cols=-1, max_rows=-1, table_conversion='selenium')
    dfi.export(df_check, filename=outputdir + f'{curdate}_{bar}_TransferDualCenter.png', dpi=300, fontsize=8, max_cols=-1, max_rows=-1, table_conversion='chr')
    wechat_bot_image(outputdir + f'{curdate}_{bar}_TransferDualCenter.png', type_api='check')

    with open(outputdir + f'{curdate}_{bar}_TransferDualCenter_flag.txt', 'w') as f: f.write('')


def whether_rebuy(curdate):
    next_trading_day = get_predate(curdate, -1)
    next_2_trading_day = get_predate(curdate, -2)
    next_delta_days = (datetime.datetime.strptime(str(next_2_trading_day), '%Y%m%d') -
                       datetime.datetime.strptime(str(next_trading_day), '%Y%m%d')).days
    if next_delta_days >= 5:
        isrebuy = 1
    else:
        isrebuy = 0

    # if isrebuy == 1:
    #     while int(datetime.datetime.now().strftime('%H%M%S')) <= 150000:
    #         time.sleep(3)
    # isrebuy = 1
    return isrebuy


def get_settle_price_correction_factor(curdate):
    # settle_predict_json = f'{PLATFORM_PATH_DICT["v_path"]}StockData/Trading/TradingJSON/{curdate}/settle_predict/'
    # conlist = []
    # for path_csv in list(Path(settle_predict_json).glob('future_price_realtime_*.csv')):
    #     df_settle = pd.read_csv(path_csv)
    #     conlist.append(df_settle)
    #
    # if not conlist: return 0
    # df_settle = pd.concat(conlist, axis=0)

    settle_predict_json = f'{PLATFORM_PATH_DICT["v_path"]}StockData/Trading/TradingJSON/{curdate}/'
    tsl = TransferServerLocal(ip='120.24.92.78', port=22, username='monitor', password='ExcellenceCenter0507')
    tsl.download_file(
        server_path=f'/home/monitor/MonitorCode/HelloWorld/static/file/settle_price/{curdate}_future_price_realtime.csv',
        local_path=f'{settle_predict_json}{curdate}_future_price_realtime.csv')

    df_settle = pd.read_csv(f'{settle_predict_json}{curdate}_future_price_realtime.csv')
    df_settle = df_settle[(df_settle['Time'].astype('int') > 140000) & (df_settle['Time'].astype('int') < 150000)]

    df_settle['Time'] = (np.round(df_settle['Time'] / 100) * 100).astype('int')
    df_settle['Time'] = df_settle['Time'].apply(
        lambda x: datetime.datetime.strptime(f'{curdate}-{str(x)}', '%Y%m%d-%H%M%S'))
    df_settle = df_settle.groupby('Time').mean()

    time_index = pd.date_range(start=curdate + ' 14:00', end=curdate + ' 14:57', freq='T')

    df_settle = df_settle.reindex(time_index).resample(
        '1min', closed='right', label='right').interpolate(method='linear').fillna(method='bfill')
    df_twap = df_settle.mean().to_frame().reset_index().rename({'index': 'SecuCode', 0: 'Twap'}, axis='columns')
    df_close = df_settle.iloc[-1].to_frame().reset_index().rename({'index': 'SecuCode', 0: 'Twap'}, axis='columns')
    df_close.columns = ['SecuCode', 'Close']

    df_settle = pd.merge(df_twap, df_close, on='SecuCode', how='left')
    df_settle = df_settle[df_settle['SecuCode'].astype('str').apply(lambda x: len(x) == 6) &
                          (~ df_settle['SecuCode'].str.endswith('_MAR'))]

    df_settle['TwapRet'] = df_settle['Twap'] / df_settle['Close'] - 1
    settle_correction_factor = np.maximum(df_settle['TwapRet'].fillna(0), 0).max()

    assert settle_correction_factor < 0.02, f'结算价修复结果raito={settle_correction_factor} > 0.02! '

    return settle_correction_factor


def generate_extra_trans_and_rebuy_config(
        curdate,
        config_dict_list,
        extra_mode_dict=None,
        reverse_repo_price_low=None,
        rebuy_code_dict=None,
        wechat_and_email=False):
    """
        mode:
            'rebuycash': 6,
                kwargs = {
                    'product_list': [],
                    'money': [10000],
                    'rebuycode': [],
                    'rebuymoney': [],
                }
            'rebuystock': 5,
                kwargs = {
                    'curdate': '20240116',
                    'product_list': [],
                    'secucode': '000049',
                    'rebuystock': '080049',
                    'retio': 0.3,
                }
            'fast2fast': 3,
    `           kwargs = {
                    'product_list': [],
                    'money': [10000],
                    'out_exch': ['SH'],
                }
            'normal2fast': 2,
                kwargs = {
                    'product_list': [],
                    'money': [10000],
                    'in_exch': ['SH'],
                }
    """
    predate = get_predate(curdate, 1)
    if rebuy_code_dict is None:
        rebuy_code_dict = {'SH': '204001', 'SZ': '131810'}

    if extra_mode_dict is None:
        extra_mode_dict = {
            'rebuycash': 6,
            'rebuystock': 5,
            'fast2fast': 3,
            'normal2fast': 2,
        }

    if reverse_repo_price_low is None:
        reverse_repo_price_low = {
            'SZ': '0.005',
            'SH': '0.005',
        }

    if not isinstance(config_dict_list, list):
        config_dict_list = [config_dict_list]

    conlist = []
    for config_dict in config_dict_list:
        if not config_dict:
            continue

        mode = extra_mode_dict[config_dict['mode']]
        if mode == 3 or mode == 2:
            df_trans = pd.DataFrame.from_dict(config_dict).rename(
                {'in_exch': 'exch', 'out_exch': 'exch'}, axis='columns')
            df_trans['mode'] = mode
            df_trans = df_trans.astype('str')
            conlist.append(df_trans)
        elif mode == 5:
            if config_dict['curdate'] != str(curdate):
                continue

            product_list = config_dict['production']
            secucode = config_dict['secucode']
            exch = 'SH' if secucode[0] == '6' else 'SZ'
            rebuy_code = config_dict['rebuystock']
            ratio = config_dict['ratio']
            infor_list = []
            for product in product_list:
                df_position = get_position(curdate, product)
                if df_position.empty: df_position = get_position(predate, product)

                if not df_position.empty:
                    volume = df_position[df_position['SecuCode'] == secucode]['Volume'].sum()
                    rebuyamount = int(np.floor(volume * ratio / 100) * 10000)
                    if rebuyamount > 0: infor_list.append([mode, product, exch, rebuy_code, rebuyamount])
            df_rebuy = pd.DataFrame(infor_list, columns=['mode', 'production', 'exch', 'security_id', 'amount']).astype('str')

            conlist.append(df_rebuy)
        elif mode == 6:
            if config_dict.get('rebuy_df_infor', None) is not None:
                df_rebuy = config_dict['rebuy_df_infor'].copy(deep=True)
            else:
                df_rebuy = pd.DataFrame.from_dict(config_dict)
                df_rebuy['exch'] = 'SH'
                df_rebuy = df_rebuy[df_rebuy['production'].isin(DUALCENTER_PRODUCTION)]
                df_rebuy_sz = df_rebuy.copy(deep=True)
                df_rebuy_sz['exch'] = 'SZ'
                df_rebuy = pd.concat([df_rebuy, df_rebuy_sz], axis=0)
                df_rebuy['rebuycode'] = df_rebuy['exch'].apply(lambda x: rebuy_code_dict[x])

            if not df_rebuy.empty:
                df_rebuy['mode'] = mode
                df_rebuy['price'] = df_rebuy['exch'].apply(lambda x: reverse_repo_price_low[x])
                df_rebuy = df_rebuy.astype('str')
                df_rebuy = df_rebuy[~df_rebuy['production'].isin(Production_OwnList_Swap + ProductionList_AlphaShort)].rename(
                    {'rebuycode': 'security_id', 'rebuymoney': 'amount'}, axis='columns')
                conlist.append(df_rebuy)
        else:
            raise 'Paras Error!'

    if not conlist:
        return None

    df_config = pd.concat(conlist, axis=0).fillna('-').sort_values(['mode', 'exch', 'production'])
    df_config['colo'] = df_config.apply(
        lambda row: production_2_colo(row['production'])
        if row['exch'] == 'SZ' else production_2_colo_sh(row['production']), axis=1)

    cfg_output_dir = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/AutoTransferCfg/{curdate}/'
    if not os.path.exists(cfg_output_dir):
        os.makedirs(cfg_output_dir)

    path_acc_cfg = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/StockConfigAccount/%s_stock_account_config.csv'
    if os.path.exists(path_acc_cfg % curdate):
        df_trader_info = pd.read_csv(path_acc_cfg % curdate, encoding='GBK', dtype='str').rename(
            {'Product': 'production', 'Account': 'account', 'api': 'api_type', 'traderid': 'trader_id'}, axis='columns')
    else:
        df_trader_info = pd.read_csv(path_acc_cfg % predate, encoding='GBK', dtype='str').rename(
            {'Product': 'production', 'Account': 'account', 'api': 'api_type', 'traderid': 'trader_id'}, axis='columns')
    df_trader_info['api_type'] = df_trader_info['api_type'].apply(lambda x: x.replace('-', '_'))

    if 'account' in df_config.columns.to_list(): df_config = df_config.drop('account', axis=1)
    if 'trader_id' in df_config.columns.to_list(): df_config = df_config.drop('trader_id', axis=1)

    conlist = []
    for exch, df_extra_exch in df_config.groupby('exch'):
        df_extra_exch = pd.merge(
            df_extra_exch,
            df_trader_info[['production', 'account', 'api_type', 'trader_id']],
            on='production',
            how='left').fillna('-').drop('exch', axis=1)
        df_extra_exch['account'] = df_extra_exch['account'].str[3:]

        exch_lower = str(exch).lower()
        generate_cfg_config(df_extra_exch.to_dict(orient='list'),f'{curdate}_bank_transfer_extra_{exch_lower}.cfg', cfg_output_dir)
        conlist.append(df_extra_exch)

    df_config = pd.concat(conlist, axis=0)
    df_config.to_excel(cfg_output_dir + f'{curdate}_Extra_Transfer.xlsx', index=False)

    if wechat_and_email:
        msg_ret = f'{curdate}-ExtraTrans:'
        for cmd_scp in [
            f'ssh jumper "[ ! -d ~/rtchg/bank_trans/{curdate}/ ] && mkdir ~/rtchg/bank_trans/{curdate}/"',
            f'"C:/Program Files/Git/usr/bin/scp.exe" -r {cfg_output_dir}{curdate}_bank_transfer_extra_*.cfg jumper:~/rtchg/bank_trans/{curdate}/',
        ]:
            status, log = subprocess.getstatusoutput(cmd_scp)
            msg_ret += f'\n\tstatus={status}\n\tlog={log}'

        wechat_bot_msg_check(msg_ret)


class GetCashPurposeData(GetProductionInformation):
    """
        现金用途数据
    """

    def __init__(self):
        super().__init__()

    def get_short_data(self, curdate, return_oper_df=False):
        short_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/ShortConfig/'
        if os.path.exists(f'{short_path}Short_{curdate}.csv') and (not return_oper_df):
            df_short = pd.read_csv(f'{short_path}Short_{curdate}.csv')
            if 'Exchange' not in df_short.columns.to_list():
                df_short['Exchange'] = 0

            df_short['Amount'] = - np.abs(df_short['Amount'])
            df_short['Exchange'] *= df_short['Amount']
            short_dict = df_short.groupby('Production')['Amount'].sum().to_dict()
            exchange_dict = df_short[df_short['Exchange'] != 0].groupby('Production')['Exchange'].sum().to_dict()
            return short_dict, exchange_dict, []
        else:
            short_path = f'{PLATFORM_PATH_DICT["y_path"]}产品开平记录/Operation_{curdate}.xlsx'
            if not os.path.exists(short_path):
                return {}, {}

            df_mannul = pd.read_excel(short_path)
            if 'RepayTime' not in df_mannul.columns.to_list():
                df_mannul['RepayTime'] = df_mannul['ShortTime']
            if 'Exchange' not in df_mannul.columns.to_list():
                df_mannul['Exchange'] = 0

            df_mannul = df_mannul[['ShortTime', 'CashoutTime', 'Production', 'Amount', 'RepayTime', 'Exchange']]
            df_mannul['Exchange'] = df_mannul['Exchange'].fillna(0).astype('int')
            df_mannul['Amount'] = - np.abs(df_mannul['Amount'])
            df_mannul['Exchange'] *= df_mannul['Amount']
            df_mannul = df_mannul.dropna(axis=0)

            if df_mannul.empty:
                if return_oper_df:
                    return df_mannul

                return {}, {}, []
            try:
                df_mannul['ShortTime'] = df_mannul['ShortTime'].apply(lambda x: format_date_2_str(x))
                df_mannul['CashoutTime'] = df_mannul['CashoutTime'].apply(lambda x: format_date_2_str(x))
                df_mannul['RepayTime'] = df_mannul['RepayTime'].apply(lambda x: format_date_2_str(x))
            except:
                print('日期格式填写错误', traceback.format_exc())

            assert (df_mannul['ShortTime'].astype('int') <= df_mannul['CashoutTime'].astype('int')).sum() > 0, \
                '出金日期小于平仓日期'
            assert df_mannul['CashoutTime'].apply(lambda x: get_whether_trading_day(x)).mean() == 1, \
                '出金日期非交易日'
            assert df_mannul['ShortTime'].apply(lambda x: get_whether_trading_day(x)).mean() == 1, \
                '赎回日期非交易日'

            series_num = df_mannul['Production'].value_counts()
            repeat_dict = series_num[series_num > 1].to_dict()
            assert not repeat_dict, f'产品名重复:{repeat_dict}, Warning, 请检查{short_path}！'

            if return_oper_df: return df_mannul
            short_dict = df_mannul.groupby('Production')['Amount'].sum().to_dict()
            exchange_dict = df_mannul[df_mannul['Exchange'] != 0].groupby('Production')['Exchange'].sum().to_dict()
            pre_date_repay = df_mannul[
                df_mannul['RepayTime'].astype('int') <= df_mannul['ShortTime'].astype('int')]['Production'].to_list()

            return short_dict, exchange_dict, pre_date_repay

    def get_waiting_repay_data(self, curdate):
        short_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/ShortConfig/'
        if os.path.exists(f'{short_path}Waiting_{curdate}.csv'):
            df_waiting = pd.read_csv(f'{short_path}Waiting_{curdate}.csv')

            if 'Exchange' not in df_waiting.columns.to_list():
                df_waiting['Exchange'] = 0

            df_waiting['Amount'] = - np.abs(df_waiting['Amount'])
            df_waiting['Exchange'] *= df_waiting['Amount']
            short_dict = df_waiting.groupby('Production')['Amount'].sum().to_dict()
            exchange_dict = df_waiting[df_waiting['Exchange'] != 0].groupby('Production')['Exchange'].sum().to_dict()

            return short_dict, exchange_dict
        else:
            return {}, {}

    def get_repay_data(self, curdate):
        repay_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/RepayConfig/'
        if os.path.exists(f'{repay_path}Repay_{curdate}.csv'):
            df_repay = pd.read_csv(f'{repay_path}Repay_{curdate}.csv')

            if 'Exchange' not in df_repay.columns.to_list():
                df_repay['Exchange'] = 0

            df_repay['Amount'] = - np.abs(df_repay['Amount'])
            df_repay['Exchange'] *= df_repay['Amount']
            repay_dict = df_repay.groupby('Production')['Amount'].sum().to_dict()
            exchange_dict = df_repay[df_repay['Exchange'] != 0].groupby('Production')['Exchange'].sum().to_dict()
            return repay_dict, exchange_dict
        else:
            return {}, {}

    def get_cashout_extra_plan(self, curdate):
        repay_plan_out = f'{PLATFORM_PATH_DICT["y_path"]}产品开平记录/计划之外出金配置/{curdate}_Repay.xlsx'
        repay_dict = {}
        if os.path.exists(repay_plan_out):
            df_repay = pd.read_excel(repay_plan_out)
            if not df_repay.empty:
                df_repay = df_repay.groupby('Product')[['Repay']].sum().reset_index()
                for prod, amount in df_repay[['Product', 'Repay']].values:
                    repay_dict[prod] = - abs(amount)

        return repay_dict

    def get_rebuy_data(self, curdate):
        path_file = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/AutoTransferCfg/{curdate}/'
        path_file += f'{curdate}_Extra_Transfer.xlsx'
        if os.path.exists(path_file):
            df_rebuy = pd.read_excel(path_file)
            df_rebuy = df_rebuy[df_rebuy['mode'].astype('int') == 6]
            df_rebuy = df_rebuy.groupby('production')[['amount']].sum().reset_index()
            df_rebuy['amount'] = - df_rebuy['amount'] / 100
            rebuy_money_dict = df_rebuy.set_index('production')['amount'].to_dict()
        else:
            rebuy_money_dict = {}
        return rebuy_money_dict

    def get_call_margin_morning(self, curdate):
        path_call_margin_morning = f'{PLATFORM_PATH_DICT["v_path"]}Trading/AutoCapitalManager/AutoTransferCfg/' \
                                   f'{curdate}/{curdate}_hedge_market_open_details.xlsx'
        if os.path.exists(path_call_margin_morning):
            df_call_morning = pd.read_excel(path_call_margin_morning)
            df_call_morning = df_call_morning[['Product', 'SZ出金额', 'SH出金额']].rename(
                {'SZ出金额': 'SZ', 'SH出金额': 'SH'}, axis='columns')
            df_call_morning['CashOut'] = df_call_morning['SZ'].fillna('0W').str[:-1].astype('float') + \
                                         df_call_morning['SH'].fillna('0W').str[:-1].astype('float')
            dict_call_margin = {
                product: - abs(call_money) for product, call_money in df_call_morning[['Product', 'CashOut']].values
            }
        else:
            dict_call_margin = {}

        return dict_call_margin

    def get_repay_amount_data_history(self, curdate):
        predate_5 = get_predate(curdate, 5)
        date_list = get_trading_days(predate_5, curdate)
        infor_list = []
        for date in date_list: infor_list.append(self.get_short_data(date, return_oper_df=True)[['Production', 'Amount', 'RepayTime']])

        df = pd.concat(infor_list, axis=0)
        df['Production'] = df['Production'].apply(lambda x: self.get_product_trading_main_name(x))
        df = df.groupby(['RepayTime', 'Production'])['Amount'].sum().reset_index()
        df = pd.pivot_table(df, index='RepayTime', columns='Production', values='Amount').reindex(date_list, fill_value=0).fillna(0)
        df = df.astype('int').T
        df.index.name = None

        return df

    def get_apply_amount_data(self, curdate, return_exchange=False, return_df=False):
        dict_config_path = {
            'hedge_open_index_fix':
                f'{PLATFORM_PATH_DICT["y_path"]}产品开平记录/早盘补开股指/{curdate}_OpenIndexFuture.csv',
            'hedge_open_index':
                f'{PLATFORM_PATH_DICT["y_path"]}产品开平记录/开股指/{curdate}_OpenIndexFuture.csv'
        }
        dict_res = {}
        exchange_dict_res = {}
        infor_list = []
        for mode in dict_config_path:
            path_cur = dict_config_path[mode]
            if not os.path.exists(path_cur): continue

            df_open_index = pd.read_csv(path_cur, encoding='GBK')
            if df_open_index.empty: continue
            if 'Exchange' not in df_open_index.columns.to_list(): df_open_index['Exchange'] = 0
            df_open_index['Exchange'] = df_open_index['Exchange'].fillna(0).astype('int')
            
            try: df_open_index['ApplyDate'] = df_open_index['ApplyDate'].apply(lambda x: format_date_2_str(x))
            except: print('日期格式填写错误', traceback.format_exc())

            series_num = df_open_index['Account'].value_counts()
            repeat_dict = series_num[series_num > 1].to_dict()
            assert not repeat_dict, f'产品名重复:{repeat_dict}, Warning, 请检查{path_cur}！'

            df_open_index['Account'] = df_open_index['Account'].apply(lambda x: self.get_product_trading_main_name(x))
            infor_list.append(df_open_index)
            dict_res[mode] = df_open_index.groupby('Account')['Money'].sum().to_dict()
            exchange_dict_res.update(df_open_index[df_open_index['Exchange'] == 1].groupby('Account')['Exchange'].mean().to_dict())
        
        if infor_list: df_open_index = pd.concat(infor_list, axis=0)
        else: df_open_index = pd.DataFrame()
        
        if return_df: return df_open_index
        if return_exchange: return dict_res, exchange_dict_res
        else: return dict_res

    def get_apply_amount_data_history(self, curdate, pren=5):
        predate_n = get_predate(curdate, pren)
        date_list = get_trading_days(predate_n, curdate)
        infor_list = []
        for date in date_list:
            df = self.get_apply_amount_data(date, return_df=True)
            if not df.empty: infor_list.append(df)
        
        if infor_list:
            df = pd.concat(infor_list, axis=0).rename({'Account': 'Product', 'ApplyDate': 'Date'}, axis='columns')
            df['Product'] = df['Product'].apply(lambda x: self.get_product_trading_main_name(x))
            df = df.groupby(['Date', 'Product'])['Money'].sum().reset_index()
            df = pd.pivot_table(df, index='Date', columns='Product', values='Money').reindex(date_list, fill_value=0).fillna(0)
            df = df.astype('int').T
            df.index.name = None
        else: df = pd.DataFrame()
        return df

    def get_apply_amount_data_dc(self, curdate, check_mode=True, return_df=False):
        path_cur = f'{PLATFORM_PATH_DICT["y_path"]}产品开平记录/对冲申购/{curdate}_DC_ApplyAmount.csv'
        if not os.path.exists(path_cur):
            return {} if not return_df else None

        df_apply_dc = pd.read_csv(path_cur, encoding='GBK').drop_duplicates()
        try: df_apply_dc['ApplyDate'] = df_apply_dc['ApplyDate'].apply(lambda x: format_date_2_str(x))
        except: print('日期格式填写错误', traceback.format_exc())

        if check_mode:
            series_num = df_apply_dc['Account'].value_counts()
            repeat_dict = series_num[series_num > 1].to_dict()

            assert not repeat_dict, f'产品名重复:{repeat_dict}, Warning, 请检查{path_cur}！'

        df_apply_dc['Account'] = df_apply_dc['Account'].apply(lambda x: self.get_product_trading_main_name(x))
        dict_res = df_apply_dc.groupby('Account')['Money'].sum().to_dict()
        if return_df: return df_apply_dc
        return dict_res

    def get_apply_amount_data_dc_history(self, curdate, n_pre=1):
        predate_1 = get_predate(curdate, n_pre)
        date_list = get_trading_days(predate_1, curdate)
        infor_list = []
        for date in date_list:
            df = self.get_apply_amount_data_dc(date, return_df=True)
            if not df.empty: infor_list.append(df)

        if infor_list:
            df = pd.concat(infor_list, axis=0).rename({'Account': 'Product', 'ApplyDate': 'Date'}, axis='columns')
            df['Product'] = df['Product'].apply(lambda x: self.get_product_trading_main_name(x))
            df = df.groupby(['Date', 'Product'])['Money'].sum().reset_index()
            df = pd.pivot_table(df, index='Date', columns='Product', values='Money').reindex(date_list, fill_value=0).fillna(0)
            df = df.astype('int').T
            df.index.name = None
        else: df = pd.DataFrame()

        return df

    def get_all_cash_status_data(self, curdate, predate=None, temp_cash_deduct=None):
        if predate is None:
            predate = get_predate(curdate, 1)

        short_dict, exchange_dict, pre_date_repay = self.get_short_data(curdate)
        wait_repay_dict, wait_exchange_dict = self.get_waiting_repay_data(curdate)
        repay_dict, repay_exchange_dict = self.get_repay_data(curdate)
        cashout_extra_dict = self.get_cashout_extra_plan(curdate)
        rebuy_data_dict = self.get_rebuy_data(predate)
        call_margin_morning = self.get_call_margin_morning(curdate)
        apply_dict, exchange_apply_dict = self.get_apply_amount_data(curdate, return_exchange=True)

        if temp_cash_deduct is None:
            temp_cash_deduct = {}

        short_dict = {
            key: short_dict.get(key, 0) - abs(temp_cash_deduct.get(key, 0))
            for key in set(short_dict) | set(temp_cash_deduct)
        }

        exchange_dict = {
            key: exchange_dict.get(key, 0) + wait_exchange_dict.get(key, 0) + repay_exchange_dict.get(key, 0)
            for key in set(exchange_dict) | set(wait_exchange_dict) | set(repay_exchange_dict)}

        dict_cash_out = {
            key: wait_repay_dict.get(key, 0) + repay_dict.get(key, 0) + cashout_extra_dict.get(key, 0)
            for key in set(wait_repay_dict) | set(repay_dict) | set(cashout_extra_dict)}

        df_all = pd.concat([
            pd.DataFrame.from_dict(short_dict, orient='index').rename({0: '当日赎回'}, axis='columns'),
            pd.DataFrame.from_dict(exchange_dict, orient='index').rename({0: '赎回换仓在途'}, axis='columns'),
            pd.DataFrame.from_dict(wait_repay_dict, orient='index').rename({0: '赎回预付'}, axis='columns'),
            pd.DataFrame.from_dict(repay_dict, orient='index').rename({0: '赎回当日付'}, axis='columns'),
            pd.DataFrame.from_dict(cashout_extra_dict, orient='index').rename({0: '额外出金'}, axis='columns'),
            pd.DataFrame.from_dict(rebuy_data_dict, orient='index').rename({0: '昨日逆回购'}, axis='columns'),
            pd.DataFrame.from_dict(call_margin_morning, orient='index').rename({0: '当日早盘出金'}, axis='columns'),
            pd.DataFrame(apply_dict).rename({'hedge_open_index_fix': '早盘补开股指', 'hedge_open_index': '开股指'},
                                            axis='columns')
        ], axis=1).fillna(0).astype('int').reset_index().rename({'index': 'Product'}, axis='columns')
        if not short_dict:
            df_all['当日赎回'] = 0
            df_all['赎回换仓在途'] = 0
        if not repay_dict:
            df_all['赎回当日付'] = 0

        if not cashout_extra_dict:
            df_all['额外出金'] = 0

        df_all['Date'] = curdate

        return dict_cash_out, short_dict, exchange_dict, apply_dict, exchange_apply_dict, rebuy_data_dict, call_margin_morning, df_all, pre_date_repay

    def get_next_repay_cashout(self, curdate):
        nextdate = get_predate(curdate, -1)
        path_next_out = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/RepayConfig/' \
                        f'Repay_{nextdate}.csv'
        path_next_out_others = f'{PLATFORM_PATH_DICT["y_path"]}产品开平记录/计划之外出金配置/{nextdate}_Repay.xlsx'
        if os.path.exists(path_next_out):
            df_next_co = pd.read_csv(path_next_out)[['Production', 'Amount']].rename(
                {'Production': 'Product', 'Amount': 'NextCo'}, axis='columns')
        else:
            df_next_co = pd.DataFrame(columns=['Product', 'NextCo'])

        if os.path.exists(path_next_out_others):
            df_next_co_others = pd.read_excel(path_next_out_others)[['Product', 'Repay']].rename({'Repay': 'NextCo'}, axis='columns')
        else:
            df_next_co_others = pd.DataFrame(columns=['Product', 'NextCo'])

        df_next_co = pd.concat([df_next_co, df_next_co_others], axis=0).groupby('Product')[['NextCo']].sum().reset_index()
        if df_next_co.empty: return pd.DataFrame(columns=['Product', 'NextCo'])
        
        df_next_co['NextCo'] = - np.abs(df_next_co['NextCo']) * 10000
        return df_next_co

    def get_daily_repay_cashout(self, curdate):
        path_cashout = f'{PLATFORM_PATH_DICT["y_path"]}产品开平记录/证券户当日赎回出金/{curdate}_Repay.xlsx'
        if not os.path.exists(path_cashout):
            df_repay = pd.DataFrame(columns=['Product', 'RepayMoney'])
        else:
            df_repay = pd.read_excel(path_cashout).rename({'Repay': 'RepayMoney'}, axis='columns')
            if not df_repay.empty:
                df_repay = df_repay[['Product', 'RepayMoney']]
                df_repay['Product'] = df_repay['Product'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x.strip()))
                df_repay['RepayMoney'] = df_repay['RepayMoney'].apply(
                    lambda x: int(x)
                    if isinstance(x, int) or isinstance(x, float) else re.sub(r'[^0-9]', '', x.strip()))
                df_repay['RepayMoney'] = - np.abs(df_repay['RepayMoney'] * 10000)
            else:
                df_repay = pd.DataFrame(columns=['Product', 'RepayMoney'])
        return df_repay
    
    def get_daily_apply_cashin(self, curdate):
        path_apply = f'{PLATFORM_PATH_DICT["y_path"]}产品开平记录/当日入金当日建仓需求/{curdate}_TransIn_Origin.csv'
        if not os.path.exists(path_apply): return None
        
        df_apply = pd.read_csv(path_apply).groupby('Account')['Money'].sum().reset_index()
        return df_apply



class GetTradingDependData(GetCashPurposeData):
    def __init__(self):
        super().__init__()

    def __get_auto_transfer_result(self, curdate, **kwargs):
        path_file = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/AutoTransferResults/{curdate}/'
        if not os.path.exists(path_file): os.makedirs(path_file)

        down_trans_json = kwargs.get('down_trans_json', False)
        deducted_trans = kwargs.get('deducted_trans', False)
        deducted_trans_drop_colo_list = kwargs.get('deducted_trans_drop_colo_list', [])
        deducted_trans_drop_product_list = kwargs.get('deducted_trans_drop_product_list', [])

        if deducted_trans_drop_colo_list is None: deducted_trans_drop_colo_list = []
        if deducted_trans_drop_product_list is None: deducted_trans_drop_product_list = []

        date_cur, time_flag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S').split('-')
        down_flag = (curdate == date_cur) & (int(time_flag) < 160000)
        if down_trans_json and down_flag and deducted_trans:
            cmd_scp = f'"C:/Program Files/Git/usr/bin/scp.exe" jumper:~/rtchg/bank_trans_log/{curdate}_jl/* {path_file}'
            status, log = subprocess.getstatusoutput(cmd_scp)
            print(f'down {curdate}_*.jsonl: ({status})')

        infor_list = []
        for file in list(Path(path_file).glob(f'{curdate}_*.jsonl')):
            host_name = file.stem[9:]
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    json_obj = json.loads(line)
                    json_obj['colo'] = host_name
                    if json_obj.get('mkt') is None: json_obj['mkt'] = 'sz' if 'sh' not in host_name else 'sh'
                    if 1 <= json_obj['mode'] <= 2:
                        if json_obj['dir'] == 'out':
                            infor_list.append(json_obj)
        if infor_list:
            df = pd.DataFrame(infor_list)
            df['NAME'] = df['account'].apply(lambda x: ACCOUNT_2_PRODUCTION.get(x, x)).astype('str')
            df = df[(~df['colo'].isin(deducted_trans_drop_colo_list)) & (~df['NAME'].isin(deducted_trans_drop_product_list))]
            df['NAME'] += np.where(df['NAME'].isin(DUALCENTER_PRODUCTION), '-' + df['mkt'].str.upper(), '')
            df['amount'] *= -1
            df = df[df['result'].astype('str') == 'success']
            df = df.groupby(['NAME', 'colo'])[['amount']].sum().reset_index()

            return df.set_index('NAME')['amount'].to_dict()
        else: return {}

    def __get_monitor_data(self, json_name, capital_path_monitor):
        # tsl = TransferServerLocal(ip='120.24.92.78', port=22, username='monitor', password='ExcellenceCenter0507')
        # tsl.download_file(server_path=f'/home/monitor/MonitorCode/HelloWorld/static/file/{json_name}', local_path=capital_path_monitor + json_name)
        # return 0
        cmd_scp = f'"C:/Program Files/Git/usr/bin/scp.exe" monitor:~/MonitorCode/HelloWorld/static/file/{json_name} {capital_path_monitor}'
        status, log = subprocess.getstatusoutput(cmd_scp)
        print(f'down {json_name}: ({status}, {log})')

        return status

    def __read_monitor_data(self, json_name, capital_path_monitor):
        with open(capital_path_monitor + json_name, 'r') as f:
            print('正在读取--->', capital_path_monitor + json_name)
            dict_monitor_data = json.load(f)
        return dict_monitor_data

    def get_monitor_data(self, curdate, json_name, down_json_mode, capital_path_monitor=None):
        if capital_path_monitor is None:
            capital_path_monitor = f'{PLATFORM_PATH_DICT["v_path"]}StockData/Trading/TradingJSON/{curdate}/'

        if not os.path.exists(capital_path_monitor): os.makedirs(capital_path_monitor)

        date_cur, time_flag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S').split('-')
        down_flag = (curdate == date_cur) & (int(time_flag) < 160000)
        if down_json_mode and down_flag:
            while True:
                try:
                    status = self.__get_monitor_data(json_name, capital_path_monitor)
                    if status != 0: continue
                    dict_monitor_data = self.__read_monitor_data(json_name, capital_path_monitor)
                    break
                except:
                    print('读取失败，重新下载--->', capital_path_monitor + '')
        else:
            try:
                dict_monitor_data = self.__read_monitor_data(json_name, capital_path_monitor)
            except:
                while True:
                    try:
                        status = self.__get_monitor_data(json_name, capital_path_monitor)
                        if status != 0: continue
                        dict_monitor_data = self.__read_monitor_data(json_name, capital_path_monitor)
                        break
                    except:
                        print('读取失败，重新下载--->', capital_path_monitor + '')

        return dict_monitor_data

    def get_monitor_future_position_data(self, curdate, down_json_mode, data_type='df', df_pos=None):
        if df_pos is None:
            df_hedge_position = pd.DataFrame(self.get_monitor_data(curdate, 'Position.json', down_json_mode))
            df_hedge_position = df_hedge_position.rename(
                {'Pos_Long': 'Long', 'Pos_Short': 'Short'}, axis='columns')
            df_hedge_position['Product'] = df_hedge_position.apply(
                lambda row: self.future_account_2_product.get(str(row['Account']), row['Product'])
                if (row['Product'] in ['xxx', 'XXX']) else row['Product'], axis=1).replace(Dict_FutureNameReplace)
            df_hedge_position['Long'] = df_hedge_position['Long'].astype('int')
            df_hedge_position['Short'] = df_hedge_position['Short'].astype('int')
        else:
            df_hedge_position = df_pos.copy(deep=True)

        df_hedge_position['FutureName'] = df_hedge_position['Instrument'].str[:2]
        if data_type == 'df':
            return df_hedge_position

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

    def get_monitor_future_margin_data(self, curdate, down_json_mode, filename='Capital.json'):
        df_hedge_capital = pd.DataFrame(self.get_monitor_data(curdate, filename, down_json_mode))

        df_hedge_capital = df_hedge_capital.rename(
            {'Pos_Long': 'Long', 'Pos_Short': 'Short'}, axis='columns')
        df_hedge_capital['Product'] = df_hedge_capital.apply(
            lambda row: self.future_account_2_product.get(str(row['Account']), row['Product'])
            if (row['Product'] in ['xxx', 'XXX']) else row['Product'], axis=1).replace(Dict_FutureNameReplace)
        df_hedge_capital['Capital'] = df_hedge_capital['Capital'].astype('float')
        df_hedge_capital['Margin'] = df_hedge_capital['Margin'].astype('float')
        df_hedge_capital['Available'] = df_hedge_capital['Available'].astype('float')
        df_hedge_capital['WithdrawQuota'] = df_hedge_capital['WithdrawQuota'].replace('none', 0).astype('float').fillna(0)
        df_hedge_capital['Short'] = df_hedge_capital['Short'].apply(
            lambda x: int(str(x).split('.')[0] if str(x).split('.')[0] != '' else 0))
        df_hedge_capital['Long'] = df_hedge_capital['Long'].apply(
            lambda x: int(str(x).split('.')[0] if str(x).split('.')[0] != '' else 0))

        return df_hedge_capital

    def get_monitor_real_price_data(self, curdate, down_json_mode):
        dict_monitor_price_data = self.get_monitor_data(curdate, 'test4.json', down_json_mode)[0]
        dict_monitor_price_data = {str(_key).upper(): float(dict_monitor_price_data[_key]) for _key in dict_monitor_price_data}

        df_monitor_price_data = pd.DataFrame.from_dict(dict_monitor_price_data, orient='index').reset_index().rename({0: 'Value'}, axis='columns')
        df_monitor_price_data['Value'] = df_monitor_price_data['Value'].replace('', np.nan)
        if df_monitor_price_data['Value'].isna().sum() > 0:
            print(f'{curdate}价格存在nan值:')
            print(df_monitor_price_data[df_monitor_price_data['Value'].isna()])
            predate = get_predate(curdate, 1)
            df_future_price = get_futureprice(predate, predate)
            df_future_price['SecuCode'] = df_future_price['SecuCode'].astype('str') + '_PRICE'
            dict_future_price_pre = df_future_price.set_index('SecuCode')['SettlePrice'].to_dict()
            dict_index_price_pre = get_indexprice_all(curdate, curdate).replace({'ZZA500': 'A500'}).set_index('IndexName')['ClosePrice'].to_dict()
            dict_index_price_pre.update(dict_future_price_pre)

            df_monitor_price_data['Value'] = np.where(
                ~ df_monitor_price_data['Value'].isna(),
                df_monitor_price_data['Value'],
                df_monitor_price_data['index'].apply(lambda x: dict_index_price_pre.get(x, np.nan)))

        dict_monitor_price_data = df_monitor_price_data.set_index('index')['Value'].to_dict()
        df_price_data = pd.DataFrame.from_dict(dict_monitor_price_data, orient='index').reset_index().rename(
            {"index": 'SecuCode', 0: 'Price'}, axis='columns')
        df_price_data = df_price_data[df_price_data['SecuCode'].str.endswith('_PRICE')]

        df_price_data = df_price_data[~ df_price_data['Price'].isna()]
        df_price_data['FutureName'] = df_price_data['SecuCode'].str[:2]
        instrument_list = sorted(list(df_price_data['SecuCode'].str[2:6].unique()))

        dict_monitor_price_data.update(df_price_data.groupby('FutureName')['Price'].mean().to_dict())

        dict_price_data, dict_value_data = {}, {}
        for _key in dict_monitor_price_data:
            if _key.endswith('_PRICE'):
                dict_price_data[_key[:6]] = round(dict_monitor_price_data[_key], 4)
                dict_value_data[_key[:6]] = round(dict_monitor_price_data[_key] * Future_Value_Multiplier[_key[:2]], 2)
            elif Future_Value_Multiplier.get(_key, None) is not None:
                dict_price_data[_key] = round(dict_monitor_price_data[_key], 4)
                dict_value_data[_key] = round(dict_monitor_price_data[_key] * Future_Value_Multiplier[_key], 2)

        for index_name in self.index_list:
            if (dict_price_data.get(index_name, None) is None) or (dict_value_data.get(index_name, None) is None):
                assert False, f'{index_name} 没有价格数据！！'

        return dict_price_data, dict_value_data, instrument_list

    def get_monitor_stocks_data(
            self, curdate, down_json_mode, deducted_trans=True, down_trans_json=False,
            deducted_trans_drop_colo_list=None, deducted_trans_drop_product_list=None):
        conlist = []
        for filename in Winterfell_Monitor_JsonData_List:
            df_monitor = pd.DataFrame(self.get_monitor_data(curdate, filename, down_json_mode))
            if not df_monitor.empty:
                conlist.append(df_monitor)

        df_monitor = pd.concat(conlist, axis=0).drop_duplicates(subset=['NAME'])
        df_monitor['Capital'] = df_monitor['Capital'].astype('float')
        if deducted_trans:
            trans_dict = self.__get_auto_transfer_result(
                curdate, down_trans_json=down_trans_json, deducted_trans=deducted_trans,
                deducted_trans_drop_colo_list=deducted_trans_drop_colo_list,
                deducted_trans_drop_product_list=deducted_trans_drop_product_list)

            print(f'\n{curdate}-盘中调拨扣除:')
            for prd, amount in trans_dict.items():
                print(f'\t{prd}: {amount}')
            df_monitor['TransDeduct'] = df_monitor['NAME'].apply(lambda x: trans_dict.get(x, 0))
            df_monitor['Capital'] += df_monitor['TransDeduct']

        df_monitor['Quota_DiffRatio'] = df_monitor['Quota_DiffRatio'].astype('float')
        df_monitor['MV_net'] = df_monitor['MV_net'].astype('float')
        df_monitor['PnLratio'] = df_monitor['PnLratio'].astype('float')
        df_monitor['MV_normal'] = df_monitor['MV_normal'].astype('float')
        df_monitor['Quota_MV'] = df_monitor['Quota_MV'].astype('float')
        df_monitor['MV_collateral'] = df_monitor['MV_collateral'].astype('float')
        df_monitor['MV_shortSell'] = df_monitor['MV_shortSell'].astype('float')
        df_monitor['Exchange'] = df_monitor['NAME'].apply(lambda x: x.split('-')[1] if '-' in x else 'SZ')
        df_monitor['NAME'] = df_monitor['NAME'].apply(lambda x: x.split('-')[0])
        df_monitor = df_monitor[df_monitor['NAME'].isin(WinterFallProductionList)].rename(
            {'NAME': 'Product'}, axis='columns').reset_index(drop=True)

        return df_monitor

    def get_monitor_future_basis_data(self, curdate, down_json_mode=False):
        dict_monitor_price_data = self.get_monitor_data(curdate, 'test4.json', down_json_mode)[0]
        dict_monitor_price_data = {str(_key).upper(): float(dict_monitor_price_data[_key]) for _key in dict_monitor_price_data}
        dict_basis_rate_data = {}
        for _key in dict_monitor_price_data:
            if _key.endswith('_RATE'):
                dict_basis_rate_data[_key] = round(dict_monitor_price_data[_key], 4)

        df = pd.DataFrame([dict_basis_rate_data]).T.reset_index()
        df.columns = ['Contract', 'RetAnn']
        df['Contract'] = df['Contract'].str[:6]
        df['FutureName'] = df['Contract'].str[:2]

        return df

    def __check_close_cash_data(self, curdate, df_capital_monitor):
        df_data = df_capital_monitor.copy(deep=True)
        """check 数据时间有效性"""
        df_invalid_data = df_data.copy(deep=True)
        df_invalid_data = df_invalid_data[
            ~((df_invalid_data['Date'].astype('int') >= int(curdate)) & (
                    df_invalid_data['Time'].astype('float') > 150000))]
        df_invalid_data['Colo'] = df_invalid_data['Product'].apply(lambda x: production_2_colo(x))
        if not df_invalid_data.empty:
            print(df_invalid_data)
        """check 数据完整度"""
        lack_product = set(WinterFallProductionList) - set(df_data['Product'].unique())
        print("缺少产品: ", lack_product)
        df_dual = df_data[df_data['Product'].isin(DUALCENTER_PRODUCTION)]
        df_dual['Value'] = df_dual['Product'].apply(lambda x: df_dual['Product'].value_counts().to_dict()[x])
        df_dual_lack = df_dual[df_dual['Value'] != 2]
        if not df_dual_lack.empty:
            df_dual_lack['Exchange'] = df_dual_lack['Exchange'].apply(lambda x: 'SH' if x == 'SZ' else 'SZ')
            print(df_dual_lack)


    def get_close_cash_data(self, curdate):
        capital_path = f'{DATA_PATH_SELF}{curdate}/{curdate}_ProductCloseStocksCash.csv'
        if not os.path.exists(capital_path):
            OriginPath = Path(f'{DATA_PATH_SELF}{curdate}_end/')
            conlist = []
            for capital_path in list(OriginPath.glob('*-capital.csv')):
                df_capital_end = pd.read_csv(capital_path, header=None, dtype={1: str})
                df_capital_end['Exchange'] = 'SZ'
                conlist.append(df_capital_end)

            for capital_path in list(OriginPath.glob('*-capital_sz.csv')):
                df_capital_end = pd.read_csv(capital_path, header=None, dtype={1: str})
                df_capital_end['Exchange'] = 'SZ'
                conlist.append(df_capital_end)

            for capital_path in list(OriginPath.glob('*-capital_sh.csv')):
                df_capital_end = pd.read_csv(capital_path, header=None, dtype={1: str})
                df_capital_end['Exchange'] = 'SH'
                conlist.append(df_capital_end)

            df_capital_end = pd.concat(conlist, axis=0)
            df_capital_end.columns = [
                'Product', 'Account', 'Date', 'tmp1', 'tmp2', 'Stocks', 'Extra_Money',
                'Position', 'Time', 'RealMoney', 'UseMoney', 'MinDiff', 'Exchange']

            self.__check_close_cash_data(curdate, df_capital_end)
        else:
            df_capital_end = pd.read_csv(capital_path)

        df_capital_end = df_capital_end.groupby('Product')[['RealMoney']].sum().reset_index().rename(
            {'RealMoney': 'StockMoney'}, axis='columns')
        df_capital_end = df_capital_end[
            df_capital_end['Product'].isin(WinterFallProductionList) &
            (~df_capital_end['Product'].isin(ProductionList_AlphaShort))]

        return df_capital_end

    def get_close_position_data(self, curdate):
        pos_path = f'{DATA_PATH_SELF}{curdate}/{curdate}_ProductCloseHoldMV.csv'
        if not os.path.exists(pos_path):
            df_price = get_price(curdate, curdate).reset_index()
            infor_list = []
            for product in WinterFallProductionList:
                position_values = get_position_values(
                    curdate, pricedate=curdate, Production=product, df_price=df_price)
                infor_list.append([product, position_values])

            df_position = pd.DataFrame(infor_list, columns=['Product', 'HoldMV'])
            df_position.to_csv(pos_path, index=False)
        else:
            df_position = pd.read_csv(pos_path)

        df_position['QuotaMV'] = df_position['HoldMV']
        df_position['RealTHoldMV'] = df_position['HoldMV']

        return df_position

    def get_close_price_db_data(self, curdate, expect_price_rise=1.0):
        df_index_data = get_indexprice_all(curdate, curdate)
        df_future_data = get_futureprice(curdate, curdate)

        dict_price, dict_value_dict = {}, {}
        for index, closeprice in df_index_data[['IndexName', 'ClosePrice']].values:
            dict_price[index] = round(closeprice * (1 + expect_price_rise), 4)
            dict_value_dict[index] = \
                round(closeprice * (1 + expect_price_rise) * Future_Value_Multiplier.get(index, np.nan), 2)

        df_future_data = df_future_data[df_future_data['SecuCode'].apply(lambda x: len(x) == 6)]
        for secucode, settleprice in df_future_data[['SecuCode', 'SettlePrice']].values:
            dict_price[secucode] = round(settleprice * (1 + expect_price_rise), 4)
            dict_value_dict[secucode] = \
                round(settleprice * (1 + expect_price_rise) * Future_Value_Multiplier.get(secucode[:2], np.nan), 2)

        df_future_data['FutureName'] = df_future_data['SecuCode'].str[:2]
        future_name_price_dict = df_future_data.groupby('FutureName')['SettlePrice'].mean().to_dict()
        for fut_nm in future_name_price_dict:
            settleprice = future_name_price_dict[fut_nm]
            dict_price[fut_nm] = round(settleprice * (1 + expect_price_rise), 4)
            dict_value_dict[fut_nm] = \
                round(settleprice * (1 + expect_price_rise) * Future_Value_Multiplier.get(fut_nm[:2], np.nan), 2)

        instrument_list = sorted(list(df_future_data['SecuCode'].str[2:].unique()))

        return dict_price, dict_value_dict, instrument_list
    
    def get_close_future_basis_data(self, curdate, close_basis_all=False):
        conlist = []
        for file_path in Path(f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/FutureBasisInformation/{curdate}/').glob(f'{curdate}_Interval_Basis_*.csv'):
            conlist.append(pd.read_csv(file_path))
        df = pd.concat(conlist, axis=0)
        if close_basis_all:
            return df

        df = df[['Interval', 'FutureBasisRetAnn']].rename({'Interval': 'Contract', 'FutureBasisRetAnn': 'RetAnn'}, axis='columns')
        df = df[df['Contract'].apply(lambda x: len(x) == 6)]
        df['FutureName'] = df['Contract'].str[:2]
        return df
    
    def get_future_margin_fix_data(self, curdate):
        target_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/MarginFix/{curdate}_future_margin_fix.csv'
        if os.path.exists(target_path):
            df = pd.read_csv(target_path)
        else:
            df = pd.DataFrame(columns=['Product', 'MarginFix'])
            df.to_csv(target_path, index=False)
        
        dict_margin_fix = df.set_index('Product')['MarginFix'].to_dict()
        return dict_margin_fix

    def get_netvalue_dict(self, date=None, product_name_2_code=None, ret_mode='dict', market_path=None):
        """
        用于盘中 Market Value Process 流程
        """
        if date is None: date = datetime.datetime.now().strftime('%Y%m%d')
        try:
            if market_path is None:
                if not os.path.exists(MARKET_SUMMARY_PATH % date):
                    try:
                        predate = get_predate(date, 1)
                        df_netvalue = pd.read_excel(MARKET_SUMMARY_PATH % predate)
                    except:
                        pre2date = get_predate(date, 2)
                        df_netvalue = pd.read_excel(MARKET_SUMMARY_PATH % pre2date)
                else:
                    try:
                        df_netvalue = pd.read_excel(MARKET_SUMMARY_PATH % date)
                    except:
                        try:
                            predate = get_predate(date, 1)
                            df_netvalue = pd.read_excel(MARKET_SUMMARY_PATH % predate)
                        except:
                            pre2date = get_predate(date, 2)
                            df_netvalue = pd.read_excel(MARKET_SUMMARY_PATH % pre2date)
            else:
                df_netvalue = pd.read_excel(market_path)

            if ret_mode != 'dict': return df_netvalue
        except:
            print(traceback.format_exc())
            return {}

        df_netvalue = df_netvalue[['产品名称', '资产净值', '单位净值', '银行存款', '日期']]
        if product_name_2_code is None:
            product_name_2_code = ProductionName_2_Production

        if isinstance(product_name_2_code, list):
            product_name_2_code = {
                prod: ProductionName_2_Production[prod]
                for prod in ProductionName_2_Production.keys() if
                ProductionName_2_Production[prod] in product_name_2_code}

        df_netvalue = df_netvalue[df_netvalue['产品名称'].isin(product_name_2_code.keys())]
        df_netvalue['Production'] = df_netvalue['产品名称'].apply(lambda x: Production_NetValueMatch.get(product_name_2_code[x], product_name_2_code[x]))
        df_netvalue['资产净值'] = df_netvalue['资产净值'].astype('float')
        df_netvalue['单位净值'] = df_netvalue['单位净值'].astype('float')
        df_netvalue['日期'] = df_netvalue['日期'].fillna(1).astype('int').astype('str').replace('1', np.nan)
        
        netvalue_dict = df_netvalue.set_index('Production').to_dict(orient='index')
        
        return netvalue_dict

    def get_quota_position_expose_data(self, curdate):
        output_dir = f'{PLATFORM_PATH_DICT["v_path"]}StockTrading/StockData/InterdayAlpha/ProductionInfo/Trading/{curdate}/'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        target_path = f'{output_dir}{curdate}_ProductQuotaPositionExpose.csv'

        if not os.path.exists(target_path):
            df_accsumm, dict_alpha, dict_holdmv, dict_bar_mode, accsumm_product_list = get_account_summary_info(curdate)
            df_price = get_price(curdate, curdate).reset_index()
            infor_list = []
            for product in WinterFallProductionList:
                bar_mode = dict_bar_mode.get(product, 8)
                holdmv = get_position_values(curdate, pricedate=curdate, Production=product, df_price=df_price)
                quota_mv = get_position_values(curdate, pricedate=curdate, Production=product, df_price=df_price, mode='quota', quota_bar=bar_mode)
                infor_list.append({
                    'Date': curdate,
                    'Product': product,
                    'QuotaMV': quota_mv,
                    'HoldMV': holdmv
                })

            df_quota_position_expose = pd.DataFrame(infor_list)
            df_quota_position_expose['MVExpose'] = df_quota_position_expose['HoldMV'] - df_quota_position_expose['QuotaMV']
            df_quota_position_expose['MVExposeRatio'] = np.round(df_quota_position_expose['MVExpose'] / np.maximum(
                np.abs(df_quota_position_expose['HoldMV']), np.abs(df_quota_position_expose['QuotaMV'])), 3)
            df_quota_position_expose.to_csv(target_path, index=False)
        else:
            df_quota_position_expose = pd.read_csv(target_path, dtype={'Date': 'str'})

        return df_quota_position_expose


class ProcessTradingDependData(GetTradingDependData):
    """
        数据获取与处理
    """

    def __init__(self, curdate, paras_dict):
        super().__init__()
        self.curdate = curdate
        self.predate = get_predate(curdate, 1)
        self.nextdate = get_predate(curdate, -1)

        self.down_json_mode = paras_dict.get('down_json_mode', False)
        self.run_price_mode = paras_dict.get('run_price_mode', 'realtime')
        self.adjust_price_ratio = paras_dict.get('adjust_price_ratio', False)
        self.temp_cashout_deduct_dict = paras_dict.get('temp_cashout_deduct_dict', {})
        self.margin_base_ratio = 0.12

    def load_price_data(self):
        date_cur, time_flag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S').split('-')
        pre_market_flag = (self.curdate == date_cur) & (int(time_flag) < 93000)
        if self.run_price_mode == 'realtime':
            if pre_market_flag: 
                dict_price_data, dict_value_data, instrument_list = self.get_monitor_real_price_data(self.curdate, self.down_json_mode)
                dict_price_data, dict_value_data, instrument_list = self.get_close_price_db_data(self.predate, self.adjust_price_ratio)
            else:
                dict_price_data, dict_value_data, instrument_list = self.get_monitor_real_price_data(self.curdate, self.down_json_mode)
        elif self.run_price_mode == 'preclose':
            dict_price_data, dict_value_data, instrument_list = self.get_close_price_db_data(self.predate, self.adjust_price_ratio)
        elif self.run_price_mode == 'close':
            dict_price_data, dict_value_data, instrument_list = self.get_close_price_db_data(self.curdate, self.adjust_price_ratio)
        else:
            raise 'ValueError'
        
        print(dict_price_data)
        return dict_price_data, dict_value_data, instrument_list

    def load_cash_flow_data(self, cashout_date, rebuy_date):
        cash_out_dict, short_dict, exchange_dict, apply_dict, exchange_apply_dict, rebuy_money_dict, call_margin_morning, df_cash_out_plan, pre_date_repay = \
            self.get_all_cash_status_data(cashout_date, rebuy_date, self.temp_cashout_deduct_dict)
        
        inter_res = short_dict.keys() & apply_dict.get('hedge_open_index', {}).keys()
        assert not inter_res, f"{inter_res} 既有申购又有赎回！"

        return cash_out_dict, short_dict, exchange_dict, apply_dict, exchange_apply_dict, rebuy_money_dict, df_cash_out_plan, pre_date_repay

    def load_future_data(self, curdate, down_json_mode, instrument_month, pos_mode='pre-close', nt_ret_capital=False):
        df_hedge_capital = self.get_monitor_future_margin_data(curdate, down_json_mode)
        df_hedge_position = self.get_monitor_future_position_data(curdate, down_json_mode, data_type='Instrument')

        swap_pos_dict = get_swap_product_future_pos(curdate, pos_mode=pos_mode, resp_mode='Instrument')
        option_pos_dict = get_option_product_future_pos(curdate, pos_mode=pos_mode, resp_mode='dict', instrument=instrument_month)
        swap_pos_dict.update(option_pos_dict)

        df_swap_position = pd.DataFrame(
            [[prod, swap_pos_dict[prod]] for prod in swap_pos_dict], columns=['Product', 'HedgePosDict'])
        df_hedge_position = pd.concat([df_swap_position, df_hedge_position], axis=0)
        if nt_ret_capital: return df_hedge_position

        return df_hedge_capital, df_hedge_position

    def load_stocks_data(self, curdate, down_json_mode, **kwargs):
        deducted_trans = kwargs.get('deducted_trans', False)
        down_trans_json = kwargs.get('down_trans_json', False)
        deducted_trans_drop_colo_list = kwargs.get('deducted_trans_drop_colo_list', [])
        deducted_trans_drop_product_list = kwargs.get('deducted_trans_drop_product_list', [])

        df_stock_data = self.get_monitor_stocks_data(
            curdate, down_json_mode, deducted_trans, down_trans_json,
            deducted_trans_drop_colo_list, deducted_trans_drop_product_list)

        df_stock_data = df_stock_data.groupby('Product').agg(
            {'MV_normal': 'mean', 'Quota_MV': 'mean', 'MV_collateral': 'mean', 'MV_shortSell': 'mean',
             'MV_net': 'mean', 'Capital': 'sum'}).reset_index().rename(
            {'MV_normal': 'HoldMV', 'Quota_MV': 'QuotaMV', 'Capital': 'StockRealTime'}, axis='columns')
        df_stock_data['RealTHoldMV'] = df_stock_data['MV_collateral'] + df_stock_data['MV_shortSell']
        df_stock_data['StockMoney'] = np.maximum(df_stock_data['StockRealTime'] + df_stock_data['MV_net'], 0)

        return df_stock_data

    def load_stocks_query_data(self, curdate):
        df_capital = self.get_close_cash_data(curdate)
        df_position = self.get_close_position_data(curdate)

        df_stock_data = pd.merge(df_position, df_capital, on='Product', how='outer')

        df_stock_data['StockRealTime'] = df_stock_data['StockMoney']
        return df_stock_data

    def load_operation_product_records(self):
        df = pd.read_csv(f'{PLATFORM_PATH_DICT["y_path"]}产品开平记录/其他/转托管记录.csv', dtype={'Date': 'str'})
        return {date: df_prd['Product'].to_list() for date, df_prd in df.groupby('Date')}

    def format_result_excel_columns(self, df_res, format_type='', columns_list=None):
        df_res = df_res.copy(deep=True)
        for columns in columns_list:
            if format_type == 'W':
                df_res[columns] = np.round(df_res[columns].fillna(0) / 1e4).astype('int').astype('str') + 'W'
            elif format_type == '+W':
                df_res[columns] = np.round(df_res[columns].fillna(0) / 1e4).astype('int')
                df_res[columns] = df_res[columns].apply(
                    lambda x: '{:+}W'.format(x)).replace(['+0W', '-0W'], '')
            elif format_type == 'W2int':
                df_res[columns] = df_res[columns].replace(['', 'nan'], '0W').str[:-1].astype('int') * 10000
            elif format_type == '%':
                df_res[columns] = np.round(df_res[columns] * 100, 1)
            elif format_type == 'ListW':
                df_res[columns] = df_res[columns].fillna('').apply(
                    lambda x: '' if (str(x) == 'nan') or (str(x) == '')
                    else ['' if (str(_x) == 'nan') or (str(_x) == '') else str(int(round(_x / 1e4))) + 'W' for _x in x])
            elif format_type == 'ExecPos':
                df_res[columns] = df_res[columns].fillna('').apply(
                    lambda x: '' if (str(x) == 'nan') or (str(x) == '')
                    else self.format_future_operation_list_2_str(x, self.future_list))
            elif format_type == 'ExecPosLong':
                df_res[columns] = df_res[columns].fillna('').apply(
                    lambda x: '' if (str(x) == 'nan') or (str(x) == '')
                    else self.format_future_operation_list_2_str(x, self.future_list, 'LONG'))
            elif format_type == 'HedgePosList':
                df_res[columns] = df_res[columns].fillna('').apply(
                    lambda x: '' if (str(x) == 'nan') or (str(x) == '') else list([int(_x) for _x in x]))
            elif format_type == 'array2list':
                df_res[columns] = df_res[columns].fillna('').apply(
                    lambda x: '' if (str(x) == 'nan') or (str(x) == '') else list(x))

        return df_res

    def format_future_operation_list_2_str(self, fut_exec, future_mix_list, future_dir='SHORT', reverse=False, rvrs_type='arr'):
        if reverse:
            if 'I' not in str(fut_exec):
                return np.array([0] * len(future_mix_list))

            cont_dict = {}
            for contract in fut_exec.split(','):
                fut, num = contract.split(':')
                if 'SHORT' in num:
                    num = - int(num.replace('SHORT ', ''))
                else:
                    num = int(num.replace('LONG ', ''))
                cont_dict[fut] = num
            if rvrs_type == 'dict': return cont_dict
            elif rvrs_type == 'arr': return np.array([cont_dict.get(fut, 0) for fut in future_mix_list])
            else: raise ValueError
        else:
            adjust_fut_format = []
            for ifut, fut_adj in enumerate(fut_exec):
                if fut_adj > 0:
                    if future_dir == 'SHORT':
                        adjust_fut_format.append(f'{future_mix_list[ifut]}:-SHORT {int(abs(fut_adj))}')
                    else:
                        adjust_fut_format.append(f'{future_mix_list[ifut]}:+LONG {int(abs(fut_adj))}')
                elif fut_adj < 0:
                    if future_dir == 'SHORT':
                        adjust_fut_format.append(f'{future_mix_list[ifut]}:+SHORT {int(abs(fut_adj))}')
                    else:
                        adjust_fut_format.append(f'{future_mix_list[ifut]}:-LONG {int(abs(fut_adj))}')

            adjust_fut_format = ','.join(adjust_fut_format) if adjust_fut_format else ''
            return adjust_fut_format

    def format_future_position_dict_2_list(self, fut_pos, future_mix_list):
        if (not str(fut_pos)) or (str(fut_pos) == 'nan'):
            return np.array([0] * len(future_mix_list))

        if isinstance(fut_pos, str):
            fut_pos = eval(fut_pos)

        if isinstance(fut_pos, dict):
            return np.array([fut_pos.get(fut, 0) for fut in future_mix_list])
        else:
            return np.array([0] * len(future_mix_list))

    def format_future_position_list_2_dict(self, *args, **kwargs):
        """
        format_mode='series_target'
            fut_pos_list, origin_fut_pos_list, future_mix_list = args
        format_mode='series'
            fut_pos_list, future_mix_list = args
        """
        format_mode = kwargs.get('format_mode', 'series_target')
        if format_mode == 'series_target':
            fut_pos, origin_fut_pos, future_mix_list = args
            return pd.Series([
                self.format_future_position_list_2_dict(
                    fut_pos_list, origin_fut_pos_list, future_mix_list, format_mode='list_target')
                for fut_pos_list, origin_fut_pos_list in zip(fut_pos, origin_fut_pos)
            ])
        elif format_mode == 'series':
            fut_pos, future_mix_list = args
            return pd.Series([
                self.format_future_position_list_2_dict(
                    fut_pos_list, future_mix_list, format_mode='list')
                for fut_pos_list in fut_pos
            ])
        elif format_mode == 'list_target':
            fut_pos, origin_fut_pos, future_mix_list = args

            if (not str(fut_pos)) or (str(fut_pos).lower() == 'nan'):
                return {}

            if isinstance(fut_pos, str):
                eval(fut_pos)

            if isinstance(origin_fut_pos, str):
                eval(origin_fut_pos)

            if isinstance(fut_pos, Iterable) and isinstance(origin_fut_pos, Iterable):
                return {
                    fut: int(num)
                    for num, origin_num, fut in zip(fut_pos, origin_fut_pos, future_mix_list)
                    if (num != 0) or (origin_num != 0)
                }
            else:
                assert False, f"{fut_pos}, {origin_fut_pos} 不是nan与空字符,不可解析为list, 且不可迭代"
        elif format_mode == 'list':
            fut_pos, future_mix_list = args
            if (not str(fut_pos)) or (str(fut_pos).lower() == 'nan'):
                return {}

            if isinstance(fut_pos, str):
                eval(fut_pos)

            if isinstance(fut_pos, Iterable):
                return {fut: int(num) for num, fut in zip(fut_pos, future_mix_list) if num != 0}
            else:
                assert False, f"{fut_pos} 不是nan与空字符,不可解析为list, 且不可迭代"

    def format_future_pos_2_future_name(self, pos_dict):
        new_pos_dict = {}
        for key, value in pos_dict.items():
            new_pos_dict[key[:2]] = value + new_pos_dict.get(key[:2], 0)

        return new_pos_dict

    def output_config_adjust_expose(self, df_result, expose_bm_name, output_dir_process, output_dir):
        df_res = df_result.copy(deep=True)
        df_res = df_res[['Product', 'ExecPos', 'ExpectPos']]
        df_res['Product'] = df_res['Product'].replace(Dict_ProductionName_Replace)
        df_res = df_res[df_res['ExecPos'].astype('str').replace('nan', '') != '']
        infor_list = []
        for product, exec_pos, expect_pos in df_res[['Product', 'ExecPos', 'ExpectPos']].values:
            for exec_contract in exec_pos.split(','):
                cont_name, action = exec_contract.split(':')
                infor_list.append([product, cont_name, action, expect_pos])
        df_res = pd.DataFrame(infor_list, columns=['Account', 'Tag', 'Future', 'ExpectPos'])
        df_res['Amount'] = ''
        df_res['Active'] = 998
        df_res['Distribution'] = '0/0/0/0/0/0/0/0'
        df_res = df_res[['Account', 'Amount', 'Future', 'Tag', 'Active', 'Distribution', 'ExpectPos']]

        df_res.to_csv(output_dir + f'{self.curdate}_{expose_bm_name}.csv', index=False)
        df_res.to_csv(f'{output_dir_process}{expose_bm_name}.csv', index=False)

    def output_config_trans_to_future(self, df_res, output_dir, paras_dict, suffix_name='', positive_mode=False):
        min_transfer_unit = paras_dict.get('min_transfer_unit', 5e4)
        df_res = df_res[['Product', 'Main', 'Transfer', 'EstmNVR', 'NetValue', 'StockMoney', 'ExcMgRTrs', 'HoldMVR']]
        df_res = self.format_result_excel_columns(df_res, format_type='W2int', columns_list=['Transfer', 'NetValue', 'StockMoney'])

        df_res['Transfer'] = np.floor(np.abs(df_res['Transfer']) / min_transfer_unit) * min_transfer_unit * np.sign(df_res['Transfer']).astype('int')
        if positive_mode:
            df_res = df_res[df_res['Transfer'] > 0]
        else:
            df_res = df_res[df_res['Transfer'] != 0]
        df_res = df_res[['Product', 'Main', 'Transfer']].sort_values(['Main', 'Product'])
        df_res['Transfer'] = df_res['Transfer'].astype('int')

        df_res.to_csv(output_dir + f'{self.curdate}_allocation_capital{suffix_name}.csv', index=False)

    def output_config_targetmv(self, df_result, adj_paras_dict, output_dir_process, output_dir):
        df_result = df_result[
            (df_result['LngShtM'].fillna('').astype('str') != '') |
            (df_result['ExecPos'].fillna('').astype('str') != '') |
            ((~ df_result['Class'].str.endswith('DC')) &
             (np.abs(df_result['ExpMR'] / 100) > adj_paras_dict.get('apply_expose_recognition_min_ratio', 0.02)))]
        df_result['StartBar'] = 8
        df_result = df_result[adj_paras_dict['formated_columns_process']]

        df_result.to_csv(f'{output_dir}{self.curdate}_LS_config.csv', index=False)
        df_result.to_csv(f'{output_dir_process}LS_config.csv', index=False)

    def output_change_cfg_paras_config(self, df_result, adj_paras_dict, close_t0_list):
        df_result = df_result.copy(deep=True)
        df_result['LngSht%'] = df_result['LngSht%'] / 100
        flag = int(datetime.datetime.now().strftime('%H%M%S'))

        close_flag = (np.abs(df_result['LngSht%']) > adj_paras_dict['close_t0_repay_ratio']) & (df_result['LngSht%'] < 0)
        df_result = df_result[close_flag]

        if df_result.empty and not (close_t0_list): return print(f'{self.curdate}-当日没有大额平仓')

        # flag_lsr = (adj_paras_dict['close_t0_repay_ratio'] < np.abs(df_result['LngSht%'])) & (np.abs(df_result['LngSht%']) < adj_paras_dict['kill_repay_ratio'])
        # df_result['action'] = np.where(flag_lsr, 'close_t0', 'kill')
        df_result['action'] = 'close_t0'
        dict_res = {action: sorted(df['Product'].to_list()) for action, df in df_result.groupby('action')}
        if close_t0_list:
            if dict_res.get('close_t0') is not None: dict_res['close_t0'] = sorted(list(set(dict_res['close_t0'] + close_t0_list)))
            else: dict_res['close_t0'] = close_t0_list

        print(f'{self.curdate}-参数调整: {dict_res}')
        gcpc = GenerateChangeParasConfig(self.curdate)
        change_config_path = gcpc.cfg_output_dir + f'{self.curdate}_change_cfg_paras.json'
        if not os.path.exists(change_config_path):
            process_continue = True
        else:
            with open(change_config_path, 'r') as jf: origin_dict_res = json.load(jf)
            if origin_dict_res == dict_res: process_continue = False
            else: process_continue = True

        if not process_continue: return print(f'{self.curdate}-参数已经传输!')
        with open(change_config_path, 'w') as jf: json.dump(dict_res, jf, indent=2)

        gcpc.generate_change_cfg([
            {
                'all_prod': False,
                'class': [],
                'class_drop': [],
                'add_plist': prd_list,
                'drop_plist': [],
                'exch_list': ['sz', 'sh'],
                'add_priority': True,
                'cmd': action,
                'cmd_paras_dict': {
                }
            } for action, prd_list in dict_res.items()
        ])
        gcpc.upload_change_cfg()

    def custom_round_future_position(self, arr, threshold=0.5):
        """
        自定义舍入函数，支持动态阈值和多种输入格式。

        参数:
            arr (float, pd.Series): 输入数据，可以是单个浮点数或 pd.Series。
            threshold (float, pd.Series): 阈值，可以是单个浮点数或 pd.Series。

        返回:
            float 或 pd.Series: 舍入后的结果，格式与 arr 相同。
        """
        # 将 arr 转换为 pd.Series（如果还不是）
        if not isinstance(arr, pd.Series):
            arr = pd.Series([arr]) if np.isscalar(arr) else pd.Series(arr)

        # 将 threshold 转换为 pd.Series（如果还不是）
        if np.isscalar(threshold):
            threshold = pd.Series(threshold, index=arr.index, dtype=float)
        elif not isinstance(threshold, pd.Series):
            threshold = pd.Series(threshold, index=arr.index)

        # 确保 threshold 的长度与 arr 相同
        if len(threshold) != len(arr):
            raise ValueError("threshold 的长度必须与 arr 的长度相同")

        # 如果 arr 的元素是 np.array，则合并为一个大的 np.array
        if isinstance(arr.iloc[0], np.ndarray):
            # 合并所有 np.array
            combined_arr = np.concatenate(arr.values)
            # 扩展 threshold
            expanded_threshold = np.concatenate([np.full_like(a, t) for a, t in zip(arr, threshold)])

            # 计算舍入结果
            integer_part = np.floor(np.abs(combined_arr))
            decimal_part = np.abs(combined_arr) - integer_part
            positive_mask = (combined_arr >= 0)
            positive_round = np.where(decimal_part < expanded_threshold, integer_part, integer_part + 1)
            negative_round = np.where(decimal_part < expanded_threshold, -integer_part, -(integer_part + 1))
            rounded_combined = np.where(positive_mask, positive_round, negative_round)

            # 将结果拆分为原始结构
            split_indices = np.cumsum([len(a) for a in arr])[:-1]
            rounded_arrays = np.split(rounded_combined, split_indices)
            return pd.Series(rounded_arrays, index=arr.index)
        else:
            # 如果 arr 的元素是标量，则直接计算
            integer_part = np.floor(np.abs(arr))
            decimal_part = np.abs(arr) - integer_part
            positive_mask = (arr >= 0)
            positive_round = np.where(decimal_part < threshold, integer_part, integer_part + 1)
            negative_round = np.where(decimal_part < threshold, -integer_part, -(integer_part + 1))
            rounded_arr = np.where(positive_mask, positive_round, negative_round)
            return pd.Series(rounded_arr, index=arr.index)
    
    def custom_ls_mv_match_trade_accounts(self, df_stock_prod, longshort_mv, target_holdmv, stock_money_long, stock_money, stock_mv_t0_forbid):
        if longshort_mv > 0:
            target_cash_r = (stock_money_long - longshort_mv) / target_holdmv
            df_stock_prod['LngShtM'] = ((df_stock_prod['StockMLong'] - target_cash_r * df_stock_prod['HoldMV']) / (1 + target_cash_r))
            if stock_mv_t0_forbid:
                short_mv = np.minimum(df_stock_prod['LngShtM'], 0)
                long_mv = np.maximum(df_stock_prod['LngShtM'], 0)
                df_stock_prod['LngShtM'] = (long_mv + short_mv.sum() * long_mv / long_mv.sum())
        else:
            target_cash_r = (stock_money - longshort_mv) / target_holdmv
            df_stock_prod['LngShtM'] = ((df_stock_prod['StockMoney'] - target_cash_r * df_stock_prod['HoldMV']) / (1 + target_cash_r))
            if stock_mv_t0_forbid:
                short_mv = np.minimum(df_stock_prod['LngShtM'], 0)
                long_mv = np.maximum(df_stock_prod['LngShtM'], 0)
                df_stock_prod['LngShtM'] = (short_mv + long_mv.sum() * short_mv / short_mv.sum())

        return df_stock_prod

    def custom_compare_array_in_series(self, s1, s2, mode='lt'):
        stacked_s1 = np.stack(s1.values)
        stacked_s2 = np.stack(s2.values)
        if mode == 'lt': comparison_result = stacked_s1 <= stacked_s2
        else: comparison_result = stacked_s1 >= stacked_s2
        return pd.Series(list(comparison_result), index=s1.index)

    def custom_target_ratio_class_special_ls(self, mv, nav, flag, M):
        r = (mv / nav).replace([np.inf, -np.inf, np.nan], 0)
        r = np.array(r[flag].to_list())
        mv = np.array(mv[flag].to_list())
        nav = np.array(nav[flag].to_list())

        indices = np.argsort(r)
        r_sorted = r[indices]
        mv_sorted = mv[indices]
        nav_sorted = nav[indices] 

        total_mv = np.sum(mv_sorted)
        total_nav = np.sum(nav_sorted)
        n = len(r_sorted)
        if M == 0: return np.nan
        
        if M < 0:
            if r_sorted[0] * total_nav - total_mv >= M:
                return (total_mv + M) / total_nav
            else:
                left, right = 0, n - 1
                for _ in range(n):
                    if right - left <= 1:
                        mv_sum = np.sum(mv_sorted[right:])
                        nav_sum = np.sum(nav_sorted[right:])
                        return (mv_sum + M) / nav_sum
                    
                    mid = (left + right) // 2
                    money = np.sum(np.minimum((r_sorted[mid] - r_sorted) * nav_sorted, 0))
                    if money == M:
                        return r_sorted[mid]
                    elif money > M:
                        right = mid
                    else:
                        left = mid
        else:
            if r_sorted[0] * total_nav - total_mv <= M:
                return (total_mv + M) / total_nav
            else:
                left, right = 0, n - 1
                for _ in range(n):
                    if right - left <= 1:
                        mv_sum = np.sum(mv_sorted[right:])
                        nav_sum = np.sum(nav_sorted[right:])
                        return (mv_sum + M) / nav_sum
                    
                    mid = (left + right) // 2
                    money = np.sum(np.maximum((r_sorted[mid] - r_sorted) * nav_sorted, 0))
                    if money == M:
                        return r_sorted[mid]
                    elif money > M:
                        right = mid
                    else:
                        left = mid
        return np.nan

    def generate_bar_process_amount(self, amount, linear_mode, start_bar, bar_mode):
        if not isinstance(amount, str):
            amount = '0W'
        if linear_mode != 0:
            if linear_mode == 1:
                end_bar = 8 if bar_mode == 8 else 47
            else:
                end_bar = linear_mode
                
            format_array = np.array([
                int((min(bar, end_bar) - start_bar + 1) / (end_bar - start_bar + 1) * int(amount[:-1]))
                if bar >= start_bar else 0 for bar in range(1, bar_mode + 1)])
        else:
            format_array = np.array([int(amount[:-1]) if bar >= start_bar else 0 for bar in range(1, bar_mode + 1)])
            
        return "/".join([str(amount) for amount in format_array])

    def statistic_future_operation_num(self, operation_list, is_str=False, future_dir='SHORT'):
        num_dict = {}
        for operation in operation_list:
            if (str(operation) == 'nan') or str(operation) == '':
                continue

            if not is_str:
                operation = self.format_future_operation_list_2_str(operation, self.future_list, future_dir=future_dir)
            for contract in operation.split(','):
                if 'I' not in str(contract):
                    continue

                name, temp_con = contract.strip().split(':')
                direction = temp_con[0]
                fut_dir, num = temp_con[1:].split()
                key_name = f'{direction}{name}_{fut_dir}'
                if num_dict.get(key_name, None) is None:
                    num_dict[key_name] = int(num)
                else:
                    num_dict[key_name] += int(num)
        return num_dict


class AdjustProfile():
    def __init__(self, config_path=None, platform=None):
        if config_path is None:
            config_path = f'{PLATFORM_PATH_DICT["v_path"]}StockData/Trading/AnalysisTrading/config/'

        if platform is None:
            platform = 'win'
        self.platform = platform
        self.config_path = config_path

        with open(os.path.join(config_path, "adjust_paras.json"), "r") as f: self.adjust_paras = json.loads(f.read())

        with open(os.path.join(config_path, "adjust_paras_special.json"), "r") as f: self.adjust_paras_special = json.loads(f.read())

        with open(os.path.join(config_path, "setting.json"), "r", encoding='utf-8') as f: self.setting_paras = json.loads(f.read())

        with open(os.path.join(config_path, "formation.json"), "r", encoding='utf-8') as f: self.format_paras = json.loads(f.read())

        with open(os.path.join(config_path, "hs_ratio.json"), "r") as f: self.hs_ratio_thres = json.loads(f.read())

        with open(os.path.join(config_path, "adjust_trans_deducted.json"), "r") as f: self.adjust_trans_deducted = json.loads(f.read())

        with open(os.path.join(config_path, "product_type.json"), "r") as f: self.prod_type_dict = json.loads(f.read())

    def get_setting_paras(self):
        return self.setting_paras

    def get_run_mode_paras(self, run_mode='marketing'):
        run_mode_list = list(self.adjust_paras['run_mode'].keys())
        run_mode_list.remove('uni_var')

        assert self.adjust_paras['run_mode'].get(
            run_mode, None) is not None, f"{run_mode} 不在已有配置之内: {run_mode_list}"
        paras_dict = deepcopy(self.adjust_paras['run_mode']['uni_var'])
        paras_dict.update(deepcopy(self.adjust_paras['run_mode'][run_mode]))
        return paras_dict

    def get_adj_mode_paras(self, adjust_mode='index_steady'):
        adj_mode_list = list(self.adjust_paras['adj_mode'].keys())
        adj_mode_list.remove('uni_var')

        assert self.adjust_paras['adj_mode'].get(
            adjust_mode, None) is not None, f"{adjust_mode} 不在已有配置之内: {adj_mode_list}"
        paras_dict = deepcopy(self.adjust_paras['adj_mode']['uni_var'])
        paras_dict.update(deepcopy(self.adjust_paras['adj_mode'][adjust_mode]))
        return paras_dict

    def get_trans_adj_mode_paras(self):
        with open(os.path.join(self.config_path, "adjust_paras_trans.json"), "r") as f:
            adjust_paras = json.loads(f.read())

        adjust_paras.update(self.setting_paras)
        adjust_paras.update(self.prod_type_dict)

        return adjust_paras

    def get_adj_mode_all_paras(self, adjust_mode='index_steady'):
        adj_paras_dict = self.get_adj_mode_paras(adjust_mode=adjust_mode)
        adj_paras_dict.update(self.setting_paras)
        adj_paras_dict.update(self.prod_type_dict)
        adj_paras_dict.update(self.format_paras)
        adj_paras_dict.update(self.adjust_paras_special)

        return adj_paras_dict

    def get_adj_special_config(self):
        return self.adjust_paras_special

    def get_hs_ratio_thres_paras(self):
        return self.hs_ratio_thres

    def get_output_dir(self, curdate):
        output_dir = self.setting_paras['path'][self.platform]['output_path'] + f'{curdate}/'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        return output_dir

    def get_config_dir(self):
        output_dir = self.setting_paras['path'][self.platform]['config_path']
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        return output_dir

    def get_config_dir_prss(self):
        output_dir = self.setting_paras['path'][self.platform]['config_prss_path']
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        return output_dir

    def get_process_dir(self):
        output_dir = self.setting_paras['path'][self.platform]['process_path']
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        return output_dir

    def get_trans_data_dir(self):
        output_dir = self.setting_paras['path'][self.platform]['trans_data_path']
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        return output_dir

    def get_downstream_dir(self):
        output_dir = self.setting_paras['path'][self.platform]['downstream_path']
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        return output_dir

    def get_target_cash_ratio(self):
        product_2_target_ratio = {}
        for prod in WinterFallProductionList:
            colo = production_2_colo(prod)
            prod_class = production_2_index(prod) + production_2_strategy(prod)

            is_hs_trans_ready = ((np.sum([colo.startswith(colo_) for colo_ in self.setting_paras['hs_trans_broker']]) > 0) or (prod not in DUALCENTER_PRODUCTION))
            is_open_index = (prod_class not in self.setting_paras['class_non_open_index']) and (prod not in self.setting_paras['product_non_open_index'])
            is_d0_bar_5m = prod in Production_D0_Bar_5M
            if self.adjust_paras_special['product_cash_special'].get(prod) is not None:
                product_2_target_ratio[prod] = self.adjust_paras_special['product_cash_special'][prod]
                continue

            if prod in self.setting_paras['product_swap'] + self.setting_paras['product_option']:
                if is_hs_trans_ready: product_2_target_ratio[prod] = 0.03
                else:
                    if is_d0_bar_5m: product_2_target_ratio[prod] = 0.05
                    else: product_2_target_ratio[prod] = 0.03
                continue

            if prod_class.endswith('DC'):
                product_2_target_ratio[prod] = 0.25
                continue

            if is_hs_trans_ready: product_2_target_ratio[prod] = 0.035
            else:
                if is_d0_bar_5m: product_2_target_ratio[prod] = 0.05
                else:
                    if is_open_index: product_2_target_ratio[prod] = 0.04
                    else: product_2_target_ratio[prod] = 0.035

        return product_2_target_ratio


class AdjustPositionSwap(ProcessTradingDependData):
    def __init__(self, curdate, run_paras_dict):
        super().__init__(curdate=curdate, paras_dict=run_paras_dict)
        self.process_check = ProcessCheck()

    def adjust_add_check(self, df_result, paras_dict, exec_mode):
        mini_product_nav_thres = paras_dict.get('mini_product_nav_thres', 3e7)

        df_result_check = df_result.copy(deep=True)
        long_holdmv = np.maximum(df_result_check['LngShtM'], 0).sum()
        short_holdmv = np.minimum(df_result_check['LngShtM'], 0).sum()
        fut_oper_dict = self.statistic_future_operation_num(df_result_check['ExecPos'].to_list())

        # print(f"""'加减仓：总加仓市值：', {long_holdmv}""")
        # print(f"""'加减仓：总平仓市值：', {short_holdmv}""")
        # print(f"""'加减仓：期货操作张数: ', {fut_oper_dict}""")

        mini_product_list = df_result_check[
            (df_result_check['EstmNAV'] <= mini_product_nav_thres)]['Product'].to_list()
        not_enough_money_prod = df_result_check[
            ((df_result_check['StockMoney'] -
              np.maximum(df_result_check['LngShtM'], 0)) <= 0) & (df_result_check['LngShtM'] > 0)]['Product'].to_list()

        df_result = self.format_result_excel_columns(
            df_result, format_type='W', columns_list=[
                'RealTHoldMV', 'HoldMV', 'QuotaMV', 'StockMoney', 'Capital', 'Margin', 'Transfer', 'NetValue',
                'StockMLong', 'ExpTargetM', 'WithdrawQuota', 'EstmNAV', 'ExpectHoldMV', 'ExpectMargin', 'BankCash',
                'StockRealTime', 'StockOutAvail', 'ExpAdjM', 'EpctFutValue', 'ExpM', 'QExpM', 'RTExpM', 'FtrValue', 'FtrVLSMx', 'FtrIdxValue'])
        df_result = self.format_result_excel_columns(
            df_result, format_type='ListW', columns_list=['IdxValueArr', 'FtrValueArr', 'ExpAdjMV', 'ExpMV'])
        df_result = self.format_result_excel_columns(
            df_result, format_type='+W', columns_list=['LngShtM'])
        df_result = self.format_result_excel_columns(
            df_result, format_type='ExecPos', columns_list=['ExecPos'])
        df_result = self.format_result_excel_columns(
            df_result, format_type='array2list', columns_list=['HedgePos', 'Proportion', 'ExpMRV'])

        if exec_mode == 'trans':
            df_result = df_result.reindex(columns=paras_dict['formated_columns_trans'], fill_value=np.nan).dropna(how='all', axis=1)
        else:
            df_result = df_result.reindex(columns=paras_dict['formated_columns'], fill_value=np.nan).dropna(how='all', axis=1)
        df_result = df_result.sort_values(['Class', 'ExpMR', 'Main'], ascending=False)
        df_result = df_result.reset_index(drop=True)

        product_list_adj = df_result['Product'].to_list()
        df_result_style = df_result.fillna('0').replace('', '0').style.set_properties(
            **{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]).set_table_styles([
            {'selector': 'th', 'props': [('border', '1px solid black')]},
            {'selector': 'td', 'props': [('border', '1px solid black')]}]
        ).apply(lambda x: self.process_check.flag_product(x, mini_product_list, paras_dict), axis=0,
                subset=['Product', 'Main']).apply(
            lambda x: ['background-color: {0}'.format('#5B9B00')
                       if self.process_check.flag_cash_ratio(v, 3) else '' for v in x], axis=0,
            subset=['Left', 'StkCshR']).apply(
            lambda x: ['background-color: {0}'.format('#FFB2DE') if float(v) > 1 else '' for v in x], axis=0,
            subset=['BankCashRatio']).apply(
            lambda x: ['background-color: {0}'.format('#FFB2DE')
                       if p in not_enough_money_prod else '' for p in product_list_adj], axis=0,
            subset=['LngShtM', 'StockMoney']).background_gradient(
            cmap='RdYlGn_r', subset=['LngSht%'], vmin=-20, vmax=20).background_gradient(
            cmap='Reds', subset=['HoldMVR', 'EpctMVR', 'EstmNVR']).background_gradient(
            cmap='RdYlGn_r', subset=['MExpMR', 'ExpMR', 'RTExpMR', 'QExpMR', 'AdjNVR'], vmin=-3, vmax=3)

        return df_result_style, df_result

    def adjust_match_trade_accs(self, df_total_data, df_stocks, exec_mode, paras_dict):
        subacc_position_dvt_tolerate_max_ratio = paras_dict.get('subacc_position_dvt_tolerate_max_ratio', 0.002)
        df_stocks = df_stocks.copy(deep=True)
        df_data = df_total_data.copy(deep=True).reset_index(drop=True)
        df_data['ExpectMargin'] = df_data['EpctFutValue'] * self.margin_base_ratio

        """计算期望暴露情况"""
        df_data[f'ExpAdjMV'] = (df_data[f'ExpectPos'] * df_data[f'IdxValueArr'] + df_data[f'ExpectHoldMV'] * df_data['Proportion'])
        df_data[f'ExpAdjM'] = pd.Series(np.sum(np.stack(df_data['ExpAdjMV'].values), axis=1), index=df_data.index)
        df_data[f'AdjNVR'] = np.round(df_data[f'ExpAdjM'] / df_data['EstmNAV'] * 100, 1)

        df_data[f'ExpectPos'] = self.format_future_position_list_2_dict(df_data['ExpectPos'], df_data['HedgePos'], self.future_list)

        """调整前后多头占净资产比例"""
        df_data['TransNVR'] = np.round(df_data['Transfer'] / df_data['EstmNAV'] * 100, 1)
        df_data['HoldMVR'] = np.round(df_data['HoldMV'] / df_data['EstmNAV'] * 100, 1)
        df_data['HoldMVR'] = np.where((df_data['HoldMVR'] < 0) | (df_data['HoldMVR'] > 200), 100, df_data['HoldMVR'])
        df_data['FtrMVR'] = np.round(df_data['FtrValue'] / df_data['EstmNAV'] * 100, 1)
        df_data['EpctMVR'] = np.round(df_data['ExpectHoldMV'] / df_data['EstmNAV'] * 100, 1)
        df_data['EstmNVR'] = np.round(((df_data['EstmNAV'] - df_data['ShortMain'] * 10000) / df_data['NetValue'] * 100).fillna(100), 1)

        """计算股票户剩余现金情况"""
        df_data[f'Left'] = np.round((df_data['StockMoney'] - df_data['LngShtM']) / df_data[f'ExpectHoldMV'] * 100, 1)

        conlist = []
        for product, df_product in df_data.groupby('Product'):
            df_stock_prod = df_stocks[df_stocks['Main'] == df_product['Main'].iloc[0]].copy(deep=True)
            longshort_mv = df_product['LngShtM'].iloc[0]
            trans_money = df_product['Transfer'].iloc[0]
            target_holdmv = df_product['ExpectHoldMV'].iloc[0]
            stock_money = df_product['StockMoney'].iloc[0]
            stock_money_long = df_product['StockMLong'].iloc[0]

            if exec_mode.startswith('adj_exp') or exec_mode.startswith('report'):
                df_stock_prod['LngShtM'] = 0
            else:
                if longshort_mv > 0:
                    target_cash_r = (stock_money_long - longshort_mv) / target_holdmv
                    df_stock_prod['LngShtM'] = ((df_stock_prod['StockMLong'] - target_cash_r * df_stock_prod['HoldMV']) / (1 + target_cash_r))
                else:
                    target_cash_r = (stock_money - longshort_mv) / target_holdmv
                    df_stock_prod['LngShtM'] = ((df_stock_prod['StockMoney'] - target_cash_r * df_stock_prod['HoldMV']) / (1 + target_cash_r))

            adj_mv_series = np.where(df_stock_prod['HoldMV'] != 0, df_stock_prod['HoldMV'], np.abs(df_stock_prod['LngShtM']))
            df_stock_prod['LngSht%'] = np.round((df_stock_prod['LngShtM'] / adj_mv_series).replace([np.inf, -np.inf], 1).fillna(1) * 100, 1)
            df_stock_prod['LngShtM'] *= np.abs(df_stock_prod['LngSht%']) > (subacc_position_dvt_tolerate_max_ratio * 100)

            df_stock_prod['Transfer'] = trans_money
            df_stock_prod['ExpMR'] = df_product['ExpMR'].iloc[0]
            df_stock_prod['MExpMR'] = df_product['MExpMR'].iloc[0]
            df_stock_prod['QExpMR'] = df_product['QExpMR'].iloc[0]
            df_stock_prod['RTExpMR'] = df_product['RTExpMR'].iloc[0]

            df_stock_prod['EpctMVR'] = df_product['EpctMVR'].iloc[0]
            df_stock_prod['HoldMVR'] = df_product['HoldMVR'].iloc[0]
            df_stock_prod['EstmNVR'] = df_product['EstmNVR'].iloc[0]

            df_stock_prod['AdjNVR'] = df_product['AdjNVR'].iloc[0]

            df_stock_prod['Left'] = (
                    (df_stock_prod['StockMoney'] - df_stock_prod['LngShtM']) / (df_stock_prod['HoldMV'] + df_stock_prod['LngShtM']) * 100)
            df_stock_prod['Left'] = \
                np.round(df_stock_prod['Left'], 1).astype('str') + '/' + str(round(df_product['Left'].iloc[0], 1))
            df_stock_prod['StkCshR'] = \
                np.round(df_stock_prod['StkCshR'], 1).astype('str') + '/' + str(round(df_product['StkCshR'].iloc[0], 1))
            df_stock_prod['CptlRatio'] = df_product['CptlRatio'].iloc[0]
            conlist.append(df_stock_prod)

        df_stock_oper = pd.concat(conlist, axis=0)

        df_stock_oper['LngShtM'] = (np.round((df_stock_oper['LngShtM']) / 1e4) * 1e4).astype('int')
        adj_mv_series = np.where(df_stock_oper['HoldMV'] != 0, df_stock_oper['HoldMV'], np.abs(df_stock_oper['LngShtM']))
        df_stock_oper['LngSht%'] = np.round((df_stock_oper['LngShtM'] / adj_mv_series).replace([np.inf, -np.inf], 1).fillna(1) * 100, 1)

        df_data = df_data.rename(
            {'Transfer': 'TtlTrans', 'LngShtM': 'TtlLS'}, axis='columns').set_index(['Product', 'Main'])
        drop_columns_list = list(set(df_stock_oper.columns.to_list()).intersection(list(df_data.columns.to_list())))
        df_data = df_data.drop(drop_columns_list, axis=1).reset_index()

        df_result = pd.merge(df_stock_oper, df_data, on=['Product', 'Main'], how='outer')

        df_result['TFlag'] = datetime.datetime.now().strftime('%H%M%S')
        df_result['How'] = 'TargetMV'

        return self.adjust_add_check(df_result, paras_dict, exec_mode)

    def adjust(self, *args):
        df_total_data, df_stocks, exec_mode, paras_dict = args
        mini_product_nav_thres = paras_dict.get('mini_product_nav_thres', 3e7)

        position_adj_min_ratio = paras_dict.get('position_adj_min_ratio', 0.015)
        expose_adj_min_ratio = paras_dict.get('expose_adj_min_ratio', 0.015)
        adj_min_ratio_filter = paras_dict.get('adj_min_ratio_filter', False)

        stocks_reserve_min_ratio = paras_dict.get('stocks_reserve_min_ratio', 0.02)
        position_dvt_tolerate_max_ratio = paras_dict.get('position_dvt_tolerate_max_ratio', 0.002)
        future_t0_forbid = paras_dict.get('future_t0_forbid', False)

        cur_special_ls_dict = paras_dict.get('product_special_ls', {}).get(self.curdate, {})

        df_total = df_total_data.copy(deep=True)
        df_total = df_total[df_total['Product'].isin(paras_dict['product_option'] + paras_dict['product_swap'] + paras_dict['product_adj_special'])]
        df_stocks = df_stocks[df_stocks['Product'].isin(paras_dict['product_option'] + paras_dict['product_swap'] + paras_dict['product_adj_special'])]

        df_total['EpctMVR'] = 1 - df_total['CashTargetRatio']
        special_flag = df_total['Product'].apply(lambda x: paras_dict['product_special'].get(x, None) is not None)
        special_pos_r = df_total.apply(
            lambda row: paras_dict['product_special'].get(row['Product'], {}).get('PositionRatio', row['EpctMVR']), axis=1)
        special_adj_p = df_total['Product'].apply(
            lambda x: paras_dict['product_special'].get(x, {}).get('adjust_posr_active', False))
        special_openi = df_total['Product'].apply(
            lambda x: paras_dict['product_special'].get(x, {}).get('open_index_active', False))

        special_ls = df_total['Product'].apply(lambda x: cur_special_ls_dict.get(x, 0))
        special_ls = np.where(np.abs(special_ls) > 1, special_ls * 1e4, np.round(special_ls * df_total['HoldMV'] / 1e4) * 1e4).astype('int')
        special_ls += df_total['ShortMain'] * 1e4
        df_total['ShortMain'] = special_ls / 1e4

        df_total['EpctMVR'] = df_total['EpctMVR'] * (~special_flag) + special_flag * special_pos_r

        """计算初始暴露情况"""
        df_total['ExpM'] = df_total['FtrIdxValue'] + df_total['HoldMV']
        df_total['ExpMR'] = np.round(df_total['ExpM'] / df_total['EstmNAVOpn'] * 100, 1)

        df_total['ExpMV'] = df_total[f'HedgePos'] * df_total[f'IdxValueArr'] + df_total['HoldMV'] * df_total['Proportion']
        df_total['ExpMRV'] = (df_total['ExpMV'] / df_total['EstmNAVOpn'] * 100).apply(lambda x: np.round(x, 1))

        df_total['RTExpM'] = df_total['FtrIdxValue'] + df_total['RealTHoldMV']
        df_total['RTExpMR'] = np.round(df_total['RTExpM'] / df_total['EstmNAVOpn'] * 100, 1)

        df_total['QExpM'] = df_total['FtrIdxValue'] + df_total['QuotaMV']
        df_total['QExpMR'] = np.round(df_total['QExpM'] / df_total['EstmNAVOpn'] * 100, 1)

        df_total['MExpMR'] = np.round(df_total['ExpMRV'].apply(lambda x: np.max(np.abs(x))), 1)
        
        if exec_mode.startswith('report'):
            df_total['ExpectHoldMV'] = df_total['HoldMV']
            df_total['LngShtM'] = 0
            df_total['Transfer'] = 0
            df_total['ExpTargetM'] = 0
            df_total['EpctFutValue'] = df_total['FtrValue']
            df_total['ExpectPos'] = df_total['HedgePos']
            df_total['ExecPos'] = df_total['ExpectPos'] - df_total['HedgePos']
            return self.adjust_match_trade_accs(df_total, df_stocks, exec_mode, paras_dict)

        if not exec_mode.startswith('adj_exp'):
            df_total['ExpectHoldMV'] = np.where((~special_flag) + special_flag * special_adj_p, df_total['EstmNAV'] * df_total['EpctMVR'], df_total['HoldMV'])
            df_total['ExpectHoldMV'] = np.where(special_ls == 0, df_total['ExpectHoldMV'], df_total['HoldMV'] + special_ls)
            df_total['LngShtM'] = df_total['ExpectHoldMV'] - df_total['HoldMV']
        else:
            df_total['ExpectHoldMV'] = df_total[exec_mode.split('-')[1]]
            df_total['LngShtM'] = 0

        df_total['Transfer'] = 0
        df_total['ExpTargetM'] = 0
        df_total['EpctFutValue'] = df_total['ExpectHoldMV'] * df_total['Idx2FtrR']
        df_total['EpctFutValue'] = (((~special_flag) + special_flag * special_openi) * df_total['EpctFutValue'] + special_flag * (~special_openi) *  (df_total['FtrValue']))

        round_thres_flag = (df_total['Proportion'] * df_total[f'IdxValueArr'])
        round_thres_flag = position_adj_min_ratio * df_total['EstmNAV'] / pd.Series(np.sum(np.stack(round_thres_flag.values), axis=1), index=round_thres_flag.index)
        round_thres_flag = 1 - np.minimum(round_thres_flag, 0.5)

        if future_t0_forbid:
            fut_mv_dir = df_total['FtrValue'] > df_total['EpctFutValue']
            cur_mv_future = df_total['HedgePos'] * df_total['FtrValueArr']
            target_mv_future = - df_total['EpctFutValue'] * df_total['Proportion']
            flag_short_pos = self.custom_compare_array_in_series(cur_mv_future, target_mv_future)

            flag_freeze = flag_short_pos * (~ fut_mv_dir) + (~ flag_short_pos) * fut_mv_dir
            new_allocate_r = (~ flag_freeze) * df_total['Proportion']
            new_allocate_r_sum = pd.Series(np.sum(np.stack(new_allocate_r.values), axis=1), index=new_allocate_r.index)
            new_allocate_r /= np.where(new_allocate_r_sum == 0, 1, new_allocate_r_sum)

            fix_target_mv_fut = cur_mv_future * flag_freeze
            fix_target_mv_fut += (- df_total['EpctFutValue'] - pd.Series(np.sum(np.stack(fix_target_mv_fut.values), axis=1), index=fix_target_mv_fut.index)) * new_allocate_r
            expect_pos_init = fix_target_mv_fut * df_total['Ftr2IdxR'] / df_total[f'IdxValueArr']
        else:
            expect_pos_init = - df_total['EpctFutValue'] * df_total['Ftr2IdxR'] * df_total['Proportion'] / df_total[f'IdxValueArr']

        expect_pos_init_sum = pd.Series(np.sum(np.stack(expect_pos_init.values), axis=1), index=expect_pos_init.index)
        expect_pos_init_round = self.custom_round_future_position(expect_pos_init, round_thres_flag) * (expect_pos_init_sum != 0)

        expect_round_diff = expect_pos_init - expect_pos_init_round
        expect_round_diff_sum = pd.Series(np.sum(np.stack(expect_round_diff.values), axis=1), index=expect_round_diff.index)

        expect_round_diff_round = (self.custom_round_future_position(expect_round_diff_sum, round_thres_flag) * (expect_round_diff_sum < 0) * expect_round_diff.apply(lambda x: x == np.min(x)) +
                                   self.custom_round_future_position(expect_round_diff_sum, 1 - round_thres_flag) * (expect_round_diff_sum > 0) * expect_round_diff.apply(lambda x: x == np.max(x)))
        # 计算调整后期货仓位
        expect_pos_round = expect_pos_init_round + expect_round_diff_round

        df_total['ExpectPos'] = (((~special_flag) + special_flag * special_openi) * expect_pos_round + special_flag * (~special_openi) *  df_total['HedgePos'])
        expect_fix_round_expm = ((expect_pos_round - expect_pos_init) * df_total['IdxValueArr'])
        expect_fix_round_expm = pd.Series(np.sum(np.stack(expect_fix_round_expm.values), axis=1), index=expect_fix_round_expm.index)
        if not exec_mode.startswith('adj_exp'):
            df_total['LngShtM'] -= expect_fix_round_expm * (1 - special_flag * ((~ special_adj_p) & (special_ls == 0)))
            df_total['ExpectHoldMV'] = df_total['HoldMV'] + df_total['LngShtM']

        df_total['EpctFutValue'] = pd.Series(np.sum(np.stack((df_total[f'ExpectPos'] * df_total[f'FtrValueArr']).values), axis=1), index=df_total.index)
        df_total['ExecPos'] = (df_total[f'ExpectPos'] - df_total['HedgePos']).apply(lambda x: np.round(x).astype('int'))

        df_total['LngShtM'] *= (np.abs(df_total['LngShtM']) >= (position_dvt_tolerate_max_ratio * df_total['EstmNAV'])) | (special_ls != 0)
        df_total['LngShtM'] = (np.round((df_total['LngShtM'].fillna(0)) / 1e4) * 1e4).astype('int')
        df_total['ExpectHoldMV'] = df_total['HoldMV'] + df_total['LngShtM']

        if not df_total.empty:
            df_style, df = self.adjust_match_trade_accs(df_total, df_stocks, exec_mode, paras_dict)
        else:
            df = pd.DataFrame(columns=paras_dict['formated_columns'])
            df_style = df.style

        return df_style, df

    def process(self, *args):
        df_total_origin, df_stocks, paras_dict = args
        df_total = df_total_origin.copy(deep=True)

        """盘中调整仓位-执行调拨"""
        df_adj_style, df_adj = self.adjust(
            df_total, df_stocks, 'trans', paras_dict)

        """HoldMV暴露与调节"""
        df_adj_exp_holdmv_style, df_adj_exp_holdmv = self.adjust(
            df_total, df_stocks, 'adj_exp-HoldMV', paras_dict)
        df_adj_exp_quota_style, df_adj_exp_quota = self.adjust(
            df_total, df_stocks, 'adj_exp-QuotaMV', paras_dict)
        df_adj_exp_real_time_style, df_adj_exp_real_time = self.adjust(
            df_total, df_stocks, 'adj_exp-RealTHoldMV', paras_dict)

        dict_adj_res = dict()
        if 'df_adj_style' in locals():
            dict_adj_res['调整'] = eval('df_adj_style')
        if 'df_adj_exp_holdmv_style' in locals():
            dict_adj_res['adj_exp-HoldMV'] = eval('df_adj_exp_holdmv_style')
        if 'df_adj_exp_quota_style' in locals():
            dict_adj_res['adj_exp-QuotaMV'] = eval('df_adj_exp_quota_style')
        if 'df_adj_exp_real_time_style' in locals():
            dict_adj_res['adj_exp-RealTHoldMV'] = eval('df_adj_exp_real_time_style')

        output_dir = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/LongShort/{self.curdate}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        time_flag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        path_total_res = f'{output_dir}{self.curdate}_Total_AdjPositionSwap_{time_flag}.xlsx'

        with pd.ExcelWriter(path_total_res, engine='openpyxl') as writer:
            for adj_type in dict_adj_res:
                dict_adj_res[adj_type].to_excel(writer, sheet_name=adj_type, index=False)
            df_total_origin.to_excel(writer, sheet_name='账户原始数据', index=False)
            self.df_cash_out_plan.to_excel(writer, sheet_name='出金原始数据', index=False)

        # self.output_config_targetmv(df_not_trans)


class AdjustPositionHedge(ProcessTradingDependData):
    def __init__(self, curdate, run_paras_dict):
        super().__init__(curdate=curdate, paras_dict=run_paras_dict)
        self.process_check = ProcessCheck()

    def adjust_add_check(self, df_result, paras_dict, exec_mode):
        margin_ratio_short_target = paras_dict.get('margin_ratio_short_target', 0.27)
        margin_exec_min_ratio = paras_dict.get('margin_exec_min_ratio', 0.14)
        margin_base_ratio = paras_dict.get('margin_base_ratio', 0.12)
        mini_product_nav_thres = paras_dict.get('mini_product_nav_thres', 3e7)
        
        df_result_check = df_result.copy(deep=True)
        long_holdmv = np.maximum(df_result_check['LngShtM'], 0).sum()
        short_holdmv = np.minimum(df_result_check['LngShtM'], 0).sum()
        fut_oper_dict = self.statistic_future_operation_num(df_result_check['ExecPos'].to_list())

        # print(f"""'加减仓：总加仓市值：', {long_holdmv}""")
        # print(f"""'加减仓：总平仓市值：', {short_holdmv}""")
        # print(f"""'加减仓：期货操作张数: ', {fut_oper_dict}""")

        mini_product_list = df_result_check[
            (df_result_check['EstmNAV'] <= mini_product_nav_thres)]['Product'].to_list()
        not_enough_money_prod = df_result_check[
            (((df_result_check['StockMoney'] - df_result_check['Transfer']) -
             np.maximum(df_result_check['LngShtM'], 0)) <= 0) & (df_result_check['LngShtM'] > 0)]['Product'].to_list()

        df_result = self.format_result_excel_columns(
            df_result, format_type='W', columns_list=[
                'RealTHoldMV', 'HoldMV', 'QuotaMV', 'StockMoney', 'Capital', 'Margin', 'Transfer', 'NetValue',
                'StockMLong', 'ExpTargetM', 'WithdrawQuota', 'EstmNAV', 'ExpectHoldMV', 'ExpectMargin', 'BankCash',
                'StockRealTime', 'StockOutAvail', 'ExpAdjM', 'EpctFutValue', 'ExpM', 'QExpM', 'RTExpM', 'FtrValue', 'FtrVLSMx', 'FtrIdxValue'])
        df_result = self.format_result_excel_columns(
            df_result, format_type='ListW', columns_list=['IdxValueArr', 'FtrValueArr', 'ExpAdjMV', 'ExpMV'])
        df_result = self.format_result_excel_columns(
            df_result, format_type='+W', columns_list=['LngShtM'])
        df_result = self.format_result_excel_columns(
            df_result, format_type='ExecPos', columns_list=['ExecPos'])
        df_result = self.format_result_excel_columns(
            df_result, format_type='array2list', columns_list=['HedgePos', 'Proportion', 'ExpMRV'])

        if exec_mode == 'trans':
            df_result = df_result.reindex(columns=paras_dict['formated_columns_trans'], fill_value=np.nan).dropna(how='all', axis=1)
        else:
            df_result = df_result.reindex(columns=paras_dict['formated_columns'], fill_value=np.nan).dropna(how='all', axis=1)
        df_result = df_result.sort_values(['Class', 'HoldMVR', 'Main'], ascending=False)
        df_result = df_result.reset_index(drop=True)

        product_list_adj = df_result['Product'].to_list()
        df_result_style = df_result.fillna('0').replace('', '0').style.set_properties(
            **{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]).set_table_styles([
            {'selector': 'th', 'props': [('border', '1px solid black')]},
            {'selector': 'td', 'props': [('border', '1px solid black')]}]
        ).apply(lambda x: self.process_check.flag_product(x, mini_product_list, paras_dict), axis=0,
                subset=['Product', 'Main']).apply(
            lambda x: ['background-color: {0}'.format('#5B9B00')
                       if self.process_check.flag_cash_ratio(v, 3) else '' for v in x], axis=0,
            subset=['Left', 'StkCshR']).apply(
            lambda x: ['background-color: {0}'.format('#FFB2DE') if float(v) > 1 else '' for v in x], axis=0,
            subset=['BankCashRatio']).apply(
            lambda x: ['background-color: {0}'.format('#FFB2DE')
                       if p in not_enough_money_prod else '' for p in product_list_adj], axis=0,
            subset=['LngShtM', 'StockMoney']).background_gradient(
            cmap='RdYlGn_r', subset=['LngSht%'], vmin=-20, vmax=20).background_gradient(
            cmap='Reds', subset=['FtrMVR', 'HoldMVR', 'EpctMVR']).background_gradient(
            cmap='RdYlGn_r', subset=['EstmNVR'], vmin=98, vmax=102).background_gradient(
            cmap='RdYlGn_r', subset=['MExpMR', 'ExpMR', 'RTExpMR', 'QExpMR', 'AdjNVR'], vmin=-10, vmax=10).background_gradient(
            cmap='RdYlGn', subset=['MxDayMgR', 'IniMaxMgR', 'MxNxtMgR', 'TrsExcMgR', 'MgRTrsExc'],
            vmin=margin_exec_min_ratio * 100, vmax=margin_ratio_short_target * 100).apply(
            lambda x: ['background-color: {0}'.format('#5B9B00')
                       if float(v) <= (margin_base_ratio * 100) else ('background-color: {0}'.format('#FFB2DE')
                       if float(v) <= (margin_exec_min_ratio * 100) else '') for v in x], axis=0, subset=['ExcMgRTrs'])

        return df_result_style, df_result

    def adjust_match_trade_accs(self, df_total_data, df_stocks, exec_mode, paras_dict):
        min_transfer_unit = paras_dict.get('min_transfer_unit', 5e4)
        subacc_position_dvt_tolerate_max_ratio = paras_dict.get('subacc_position_dvt_tolerate_max_ratio', 0.002)
        stock_mv_t0_forbid = paras_dict.get('stock_mv_t0_forbid', False)

        df_stocks = df_stocks.copy(deep=True)
        df_data = df_total_data.copy(deep=True).reset_index(drop=True)
        df_data['ExpectMargin'] = df_data['EpctFutValue'] * self.margin_base_ratio

        """计算期望暴露情况"""
        df_data[f'ExpAdjMV'] = df_data[f'ExpectPos'] * df_data[f'IdxValueArr'] + df_data['ExpectHoldMV'] * df_data['Proportion']
        df_data[f'ExpAdjM'] = pd.Series(np.sum(np.stack(df_data[f'ExpAdjMV'].values), axis=1), index=df_data.index)
        df_data[f'AdjNVR'] = np.round(df_data[f'ExpAdjM'] / df_data['EstmNAV'] * 100, 1)

        df_data[f'ExpectPos'] = self.format_future_position_list_2_dict(
            df_data['ExpectPos'], df_data['HedgePos'], self.future_list)

        """调整前后多头占净资产比例"""
        df_data['TransNVR'] = np.round(df_data['Transfer'] / df_data['EstmNAV'] * 100, 1)
        df_data['HoldMVR'] = np.round(df_data['HoldMV'] / df_data['EstmNAV'] * 100, 1)
        df_data['HoldMVR'] = np.where((df_data['HoldMVR'] < 0) | (df_data['HoldMVR'] > 200), 100, df_data['HoldMVR'])
        df_data['FtrMVR'] = np.round(df_data['FtrValue'] / df_data['EstmNAV'] * 100, 1)
        df_data['EpctMVR'] = np.round(df_data['ExpectHoldMV'] / df_data['EstmNAV'] * 100, 1)
        df_data['EstmNVR'] = np.round(((df_data['EstmNAV'] - df_data['ShortMain'] * 10000) / df_data['NetValue'] * 100).fillna(100), 1)

        """计算保证金比例情况"""
        df_data['MgRTrsExc'] = np.round(df_data['Capital'] / np.abs(df_data[f'FtrVLSMx']) * 100, 1).replace([-np.inf, np.inf], 0)
        df_data['TrsExcMgR'] = np.round((df_data['Capital'] + df_data['Transfer']) / np.abs(df_data[f'EpctFutValue']) * 100, 1)
        df_data['ExcMgRTrs'] = np.round(df_data['Capital'] / np.abs(df_data[f'EpctFutValue']) * 100, 1)
        df_data['IniMaxMgR'] = np.round(
            (df_data['Capital'] + df_data['StockOutAvail']) / np.abs(df_data[f'FtrValue']) * 100, 1)
        df_data['MxDayMgR'] = np.round(
            (df_data['Capital'] + df_data['StockOutAvail'] -
             np.maximum(df_data['LngShtM'], 0)) / np.abs(df_data[f'EpctFutValue']) * 100, 1)
        df_data['MxNxtMgR'] = np.round(
            (df_data['Capital'] + df_data['StockMoney'] - df_data['LngShtM']) /
            np.abs(df_data[f'EpctFutValue']) * 100, 1)

        """计算股票户剩余现金情况"""
        df_data[f'Left'] = np.round(
            (df_data['StockMoney'] - df_data['Transfer'] - df_data['LngShtM']) / df_data[f'ExpectHoldMV'] * 100, 1)

        conlist = []
        for product, df_product in df_data.groupby('Product'):
            df_stock_prod = df_stocks[df_stocks['Main'] == df_product['Main'].iloc[0]].copy(deep=True)
            longshort_mv = df_product['LngShtM'].iloc[0]
            trans_money = df_product['Transfer'].iloc[0]
            target_holdmv = df_product['ExpectHoldMV'].iloc[0]
            stock_money = df_product['StockMoney'].iloc[0]
            stock_money_long = df_product['StockMLong'].iloc[0]

            if exec_mode.startswith('adj_exp') or exec_mode.startswith('report'):
                df_stock_prod['LngShtM'] = 0
            else:
                df_stock_prod = self.custom_ls_mv_match_trade_accounts(
                    df_stock_prod, longshort_mv, target_holdmv, stock_money_long, stock_money, stock_mv_t0_forbid)
                
            df_stock_prod['LngShtM'] = df_stock_prod['LngShtM'].replace([np.inf, -np.inf], 0).fillna(0)
            adj_mv_series = np.where(df_stock_prod['HoldMV'] != 0, df_stock_prod['HoldMV'], np.abs(df_stock_prod['LngShtM']))
            df_stock_prod['LngSht%'] = np.round((df_stock_prod['LngShtM'] / adj_mv_series).replace([np.inf, -np.inf], 1).fillna(1) * 100, 1)
            df_stock_prod['LngShtM'] *= np.abs(df_stock_prod['LngSht%']) > (subacc_position_dvt_tolerate_max_ratio * 100)

            expect_holdmv = df_stock_prod['LngShtM'] + df_stock_prod['HoldMV']
            expect_holdmv_total = expect_holdmv.sum()
            temp_avail = np.maximum(np.minimum(df_stock_prod['StockOutAvail'], df_stock_prod['StockMoney'] - np.maximum(df_stock_prod['LngShtM'], 0)), 0)

            total_avail = temp_avail.sum()
            if trans_money > 0:
                trans_money = min(trans_money, total_avail)
                target_cash_r = (total_avail - trans_money) / expect_holdmv_total
                avail_cash_r = temp_avail / expect_holdmv
                avail_cash = temp_avail * (avail_cash_r > target_cash_r)
                df_stock_prod['Transfer'] = avail_cash / avail_cash.sum() * trans_money
            else:
                df_stock_prod['Transfer'] = expect_holdmv / expect_holdmv_total * trans_money

            df_stock_prod['Transfer'] = np.floor(np.abs(df_stock_prod['Transfer']) / min_transfer_unit) * min_transfer_unit * np.sign(df_stock_prod['Transfer'])
            df_stock_prod['ExpMR'] = df_product['ExpMR'].iloc[0]
            df_stock_prod['MExpMR'] = df_product['MExpMR'].iloc[0]
            df_stock_prod['QExpMR'] = df_product['QExpMR'].iloc[0]
            df_stock_prod['RTExpMR'] = df_product['RTExpMR'].iloc[0]

            df_stock_prod['TransNVR'] = df_product['TransNVR'].iloc[0]
            df_stock_prod['FtrMVR'] = df_product['FtrMVR'].iloc[0]
            df_stock_prod['EpctMVR'] = df_product['EpctMVR'].iloc[0]
            df_stock_prod['HoldMVR'] = df_product['HoldMVR'].iloc[0]
            df_stock_prod['EstmNVR'] = df_product['EstmNVR'].iloc[0]

            df_stock_prod['AdjNVR'] = df_product['AdjNVR'].iloc[0]

            df_stock_prod['MgRTrsExc'] = df_product['MgRTrsExc'].iloc[0]
            df_stock_prod['ExcMgRTrs'] = df_product['ExcMgRTrs'].iloc[0]
            df_stock_prod['TrsExcMgR'] = df_product['TrsExcMgR'].iloc[0]
            df_stock_prod['IniMaxMgR'] = df_product['IniMaxMgR'].iloc[0]
            df_stock_prod['MxDayMgR'] = df_product['MxDayMgR'].iloc[0]
            df_stock_prod['MxNxtMgR'] = df_product['MxNxtMgR'].iloc[0]

            df_stock_prod['Left'] = (df_stock_prod['StockMoney'] - df_stock_prod['Transfer'] - df_stock_prod['LngShtM']) / (df_stock_prod['HoldMV'] + df_stock_prod['LngShtM']) * 100
            df_stock_prod['Left'] = np.round(df_stock_prod['Left'], 1).astype('str') + '/' + str(round(df_product['Left'].iloc[0], 1))
            df_stock_prod['StkCshR'] = np.round(df_stock_prod['StkCshR'], 1).astype('str') + '/' + str(round(df_product['StkCshR'].iloc[0], 1))
            df_stock_prod['CptlRatio'] = df_product['CptlRatio'].iloc[0]
            conlist.append(df_stock_prod)

        df_stock_oper = pd.concat(conlist, axis=0)

        df_stock_oper['LngShtM'] = (np.round((df_stock_oper['LngShtM']) / 1e4) * 1e4).astype('int')
        adj_mv_series = np.where(df_stock_oper['HoldMV'] != 0, df_stock_oper['HoldMV'], np.abs(df_stock_oper['LngShtM']))
        df_stock_oper['LngSht%'] = np.round((df_stock_oper['LngShtM'] / adj_mv_series).replace([np.inf, -np.inf], 1).fillna(1) * 100, 1)

        df_data = df_data.rename({'Transfer': 'TtlTrans', 'LngShtM': 'TtlLS'}, axis='columns').set_index(['Product', 'Main'])
        drop_columns_list = list(set(df_stock_oper.columns.to_list()).intersection(list(df_data.columns.to_list())))
        df_data = df_data.drop(drop_columns_list, axis=1).reset_index()
        df_result = pd.merge(df_stock_oper, df_data, on=['Product', 'Main'], how='outer')

        df_result['TFlag'] = datetime.datetime.now().strftime('%H%M%S')
        df_result['How'] = 'TargetMV'

        return self.adjust_add_check(df_result, paras_dict, exec_mode)

    def adjust(self, *args, **kwargs):
        df_total_data, df_stocks, exec_mode, paras_dict = args
        adjust_var = paras_dict.get('adjust_var', 'margin_ratio')

        # day_broken_not_allow = paras_dict.get('day_broken_not_allow', [])
        margin_ratio_short_target = paras_dict.get('margin_ratio_short_target', 0.27)
        margin_ratio_long_target = max(paras_dict.get('margin_ratio_long_target', 0.27), margin_ratio_short_target)
        margin_ratio_trans_target = paras_dict.get('margin_ratio_trans_target', 0.27)

        pos_ratio_short_target = paras_dict.get('pos_ratio_short_target', 0.8)
        pos_ratio_long_target = min(paras_dict.get('pos_ratio_long_target', 0.8), pos_ratio_short_target)
        pos_ratio_trans_target = paras_dict.get('pos_ratio_trans_target', 0.8)

        class_ratio_target_dict = paras_dict.get('class_ratio_target_dict', {})
        margin_exec_min_ratio = paras_dict.get('margin_exec_min_ratio', 0.14)

        equity_dc_mctr = paras_dict.get('equity_dc_mctr', 0.2)
        hybrid_dc_mctr = paras_dict.get('hybrid_dc_mctr', 0.3)

        margin_call_dc_target_ratio = paras_dict.get('margin_call_dc_target_ratio', 0.22)
        margin_call_dc_target_buffer_ratio = paras_dict.get('margin_call_dc_target_buffer_ratio', 0.18)
        stocks_reserve_min_money = paras_dict.get('stocks_reserve_min_money', 100000)
        stocks_reserve_min_ratio = paras_dict.get('stocks_reserve_min_ratio', 0.02)
        min_transfer_unit = paras_dict.get('min_transfer_unit', 5e4)

        position_adj_min_ratio = paras_dict.get('position_adj_min_ratio', 0.025)
        expose_adj_min_ratio = paras_dict.get('expose_adj_min_ratio', 0.015)
        adj_min_ratio_filter = kwargs.get('adj_min_ratio_filter', paras_dict.get('adj_min_ratio_filter', False))

        expose_target_ratio = paras_dict.get('expose_target_ratio', 0.0)
        long_exp_target_ratio = paras_dict.get('long_exp_target_ratio', 0.0)
        short_exp_target_ratio = paras_dict.get('short_exp_target_ratio', 0.0)
        exp_same_direction_add = paras_dict.get('exp_same_direction_add', False)
        expose_target_benchmark = paras_dict.get('expose_target_benchmark', 'NetValue')
        # mini_product_nav_thres = paras_dict.get('mini_product_nav_thres', 3e7)
        position_dvt_tolerate_max_ratio = paras_dict.get('position_dvt_tolerate_max_ratio', 0.002)

        future_t0_forbid = paras_dict.get('future_t0_forbid', False)
        cur_special_ls_dict = paras_dict.get('product_special_ls', {}).get(self.curdate, {})
        cur_class_special_ls_list = paras_dict.get('class_special_ls', {}).get(self.curdate, {})

        stock_positon_adjust_mode = paras_dict.get('stock_positon_adjust_mode', 'all')
        df_total = df_total_data.copy(deep=True)
        df_stocks = df_stocks.copy(deep=True)

        df_total = df_total[
            df_total['Class'].str.endswith('DC') &
            (~ df_total['Product'].isin(paras_dict['product_swap'] + paras_dict['product_option'] + paras_dict['product_adj_special']))]
        df_stocks = df_stocks[
            df_stocks['Class'].str.endswith('DC') &
            (~ df_stocks['Product'].isin(paras_dict['product_swap'] + paras_dict['product_option'] + paras_dict['product_adj_special']))]

        assert expose_target_benchmark in ['HoldMV', 'NetValue', 'ExpectHoldMV', 'EstmNAV'], f"{expose_target_benchmark} not in ['HoldMV', 'NetValue', 'ExpectHoldMV', 'EstmNAV']!"
        if expose_target_benchmark == 'NetValue': expose_target_benchmark = 'EstmNAV'

        """计算初始暴露情况"""
        df_total['ExpM'] = df_total['HoldMV'] + df_total['FtrIdxValue']
        df_total['ExpMR'] = np.round(df_total['ExpM'] / df_total['EstmNAVOpn'] * 100, 1)

        df_total['ExpMV'] = df_total[f'HedgePos'] * df_total[f'IdxValueArr'] + df_total['HoldMV'] * df_total['Proportion']
        df_total['ExpMRV'] = (df_total['ExpMV'] / df_total['EstmNAVOpn'] * 100).apply(lambda x: np.round(x, 1))

        df_total['RTExpM'] = df_total['RealTHoldMV'] + df_total['FtrIdxValue']
        df_total['RTExpMR'] = np.round(df_total['RTExpM'] / df_total['EstmNAVOpn'] * 100, 1)

        df_total['QExpM'] = df_total['QuotaMV'] + df_total['FtrIdxValue']
        df_total['QExpMR'] = np.round((df_total['QExpM']) / df_total['EstmNAVOpn'] * 100, 1)

        df_total['MExpMR'] = df_total['ExpMRV'] - df_total['Proportion'] * expose_target_ratio * 100
        df_total['MExpMR'] = np.round(df_total['MExpMR'].apply(lambda x: np.max(np.abs(x))), 1)

        if exec_mode.startswith('report'):
            df_total['ExpectHoldMV'] = df_total['HoldMV']
            df_total['LngShtM'] = 0
            df_total['Transfer'] = 0
            df_total['ExpTargetM'] = 0
            df_total['EpctFutValue'] = df_total['FtrValue']
            df_total['ExpectPos'] = df_total['HedgePos']
            df_total['ExecPos'] = df_total['ExpectPos'] - df_total['HedgePos']
            return self.adjust_match_trade_accs(df_total, df_stocks, exec_mode, paras_dict)

        flag_above_80 = df_total['Product'].isin(paras_dict.get('PrdType_Above80', []))
        flag_below_80 = df_total['Product'].isin(paras_dict.get('PrdType_Below80', []))
        flag_hybrid = df_total['Product'].isin(paras_dict.get('PrdType_Hybrid', []))
        if adjust_var == 'position_ratio':
            if exec_mode.startswith('trans'):
                pos_ratio_long_adj_target = pos_ratio_trans_target
                pos_ratio_short_adj_target = pos_ratio_trans_target
            else:
                pos_ratio_long_adj_target = df_total['Class'].apply(lambda x: class_ratio_target_dict.get(x, {}).get('pos_ratio_long_target', pos_ratio_long_target))
                pos_ratio_short_adj_target = df_total['Class'].apply(lambda x: class_ratio_target_dict.get(x, {}).get('pos_ratio_short_target', pos_ratio_short_target))
            
            posr_below80_inv = class_ratio_target_dict['Below80Inv']['pos_ratio_target']
            posr_below80 = class_ratio_target_dict['Below80']['pos_ratio_target']
            posr_above80 = class_ratio_target_dict['Above80']['pos_ratio_target']
            
            pos_ratio_long_adj_target = np.where(~ flag_hybrid, pos_ratio_long_adj_target, np.minimum(pos_ratio_long_adj_target, posr_below80_inv))
            pos_ratio_long_adj_target = np.where(~ flag_below_80, pos_ratio_long_adj_target, np.minimum(pos_ratio_long_adj_target, posr_below80))
            pos_ratio_long_adj_target = np.where(~ flag_above_80, pos_ratio_long_adj_target, np.maximum(pos_ratio_long_adj_target, posr_above80))

            pos_ratio_short_adj_target = np.where(~ flag_hybrid, pos_ratio_short_adj_target, np.minimum(pos_ratio_short_adj_target, posr_below80_inv))
            pos_ratio_short_adj_target = np.where(~ flag_below_80, pos_ratio_short_adj_target, np.minimum(pos_ratio_short_adj_target, posr_below80))
            pos_ratio_short_adj_target = np.where(~ flag_above_80, pos_ratio_short_adj_target, np.maximum(pos_ratio_short_adj_target, posr_above80))
                
            margin_long_adj_target_series = (1 - pos_ratio_long_adj_target) / ((pos_ratio_long_adj_target - expose_target_ratio) * df_total['Idx2FtrR'])
            margin_short_adj_target_series = (1 - pos_ratio_short_adj_target) / ((pos_ratio_short_adj_target - expose_target_ratio) * df_total['Idx2FtrR'])
        elif adjust_var == 'margin_ratio':
            if exec_mode.startswith('trans'):
                margin_long_adj_target_series = margin_ratio_trans_target
                margin_short_adj_target_series = margin_ratio_trans_target
            else:
                margin_long_adj_target_series = df_total['Class'].apply(lambda x: class_ratio_target_dict.get(x, {}).get('margin_ratio_long_target', margin_ratio_long_target))
                margin_short_adj_target_series = df_total['Class'].apply(lambda x: class_ratio_target_dict.get(x, {}).get('margin_ratio_short_target', margin_ratio_short_target))
            
            marginr_below80_inv = class_ratio_target_dict['Below80Inv']['margin_ratio_target']
            marginr_below80 = class_ratio_target_dict['Below80']['margin_ratio_target']
            marginr_above80 = class_ratio_target_dict['Above80']['margin_ratio_target']
            
            margin_long_adj_target_series = np.where(~ flag_hybrid, margin_long_adj_target_series, np.maximum(margin_long_adj_target_series, marginr_below80_inv))
            margin_long_adj_target_series = np.where(~ flag_below_80, margin_long_adj_target_series, np.maximum(margin_long_adj_target_series, marginr_below80))
            margin_long_adj_target_series = np.where(~ flag_above_80, margin_long_adj_target_series, np.minimum(margin_long_adj_target_series, marginr_above80))

            margin_short_adj_target_series = np.where(~ flag_hybrid, margin_short_adj_target_series, np.maximum(margin_short_adj_target_series, marginr_below80_inv))
            margin_short_adj_target_series = np.where(~ flag_below_80, margin_short_adj_target_series, np.maximum(margin_short_adj_target_series, marginr_below80))
            margin_short_adj_target_series = np.where(~ flag_above_80, margin_short_adj_target_series, np.minimum(margin_short_adj_target_series, marginr_above80))

            pos_ratio_long_adj_target = (1 + margin_long_adj_target_series * df_total['Idx2FtrR'] * expose_target_ratio) / (1 + margin_long_adj_target_series * df_total['Idx2FtrR'])
            pos_ratio_short_adj_target = (1 + margin_short_adj_target_series * df_total['Idx2FtrR'] * expose_target_ratio) / (1 + margin_short_adj_target_series * df_total['Idx2FtrR'])
        else: raise ValueError
        
        if exec_mode.startswith('not_trans'): 
            flag_class_special_ls_total = df_total['Product'].apply(lambda x: cur_special_ls_dict.get(x, 0) != 0)
            special_pos_r = df_total['Product'].apply(lambda x: paras_dict['product_special'].get(x, {}).get('PositionRatio', 0))
            for dict_clss_ls in cur_class_special_ls_list:
                flag_class_special_ls = (df_total['Class'].isin(dict_clss_ls.get('ClassList', [])) | df_total['Product'].isin(dict_clss_ls.get('ProductList', []))) & (~df_total['Product'].isin(dict_clss_ls.get('ProductDropList', [])))
                special_amount = dict_clss_ls.get('Amount', 0) * 1e4 + (flag_class_special_ls * df_total['ShortMain']).sum() * 1e4
                target_r_clss_ls = self.custom_target_ratio_class_special_ls(df_total['HoldMV'], df_total['EstmNAV'], flag_class_special_ls, special_amount)
                
                print(target_r_clss_ls, special_amount)
                flag_class_special_ls_total = flag_class_special_ls_total | flag_class_special_ls
                special_pos_r += (flag_class_special_ls * target_r_clss_ls) * (special_pos_r == 0)

            special_pos_r = special_pos_r.replace(0, np.nan)
        else:
            special_pos_r = pd.Series(np.nan, index=df_total.index)
            flag_class_special_ls_total = False
            
        special_margin_r = (1 - special_pos_r) / ((special_pos_r - expose_target_ratio) * df_total['Idx2FtrR'])

        df_total['MarginLongAdjRatio'] = margin_long_adj_target_series * special_margin_r.isna() + special_margin_r.fillna(0)
        df_total['MarginShortAdjRatio'] = margin_short_adj_target_series * special_margin_r.isna() + special_margin_r.fillna(0)

        long_adj_diff = pos_ratio_long_adj_target - (df_total['HoldMVR'] / 100)
        short_adj_diff = pos_ratio_short_adj_target - (df_total['HoldMVR'] / 100)
        normal_interval_flag = ((~ (((np.abs(long_adj_diff) >= position_adj_min_ratio) & (long_adj_diff >= 0)) |
                                ((np.abs(short_adj_diff) >= position_adj_min_ratio) & (short_adj_diff <= 0)))) &
                                (df_total['MExpMR'] <= (expose_adj_min_ratio * 100)) &
                                (np.abs(df_total['ExpMR'] - expose_target_ratio * 100) <= (expose_adj_min_ratio * 100)) &
                                (~ flag_class_special_ls_total))
        if adj_min_ratio_filter:
            if not exec_mode.startswith('adj_exp'): 
                abnormal_flag = (~ normal_interval_flag) | (df_total['ShortMain'] != 0)
            else: 
                abnormal_flag = ~ normal_interval_flag
            
            df_total = df_total[abnormal_flag]
            special_pos_r = special_pos_r[abnormal_flag]
            flag_above_80 = flag_above_80[abnormal_flag]
            flag_below_80 = flag_below_80[abnormal_flag]
            flag_hybrid = flag_hybrid[abnormal_flag]
        
        if not exec_mode.startswith('adj_exp'):
            # 计算理想保证金水平和暴露水平下的
            target_expose_amount = expose_target_ratio * df_total[expose_target_benchmark]
            df_total['templong'] = \
                (((df_total['Capital'] + df_total['StockMoney']) - df_total['MarginLongAdjRatio'] * df_total['Idx2FtrR'] *
                 (df_total['HoldMV'] - target_expose_amount)) / (1 + df_total['MarginLongAdjRatio'] * df_total['Idx2FtrR']))
            df_total['tempshort'] = \
                (((df_total['Capital'] + df_total['StockMoney']) - df_total['MarginShortAdjRatio'] * df_total['Idx2FtrR'] *
                  (df_total['HoldMV'] - target_expose_amount)) / (1 + df_total['MarginShortAdjRatio'] * df_total['Idx2FtrR']))

            if stock_positon_adjust_mode == 'only-long':
                df_total['LngShtM'] = df_total['templong'] * (df_total['templong'] >= 0)
            elif stock_positon_adjust_mode == 'only-short':
                df_total['LngShtM'] = df_total['tempshort'] * (df_total['tempshort'] < 0)
            elif stock_positon_adjust_mode == 'all':
                df_total['LngShtM'] = df_total['templong'] * (df_total['templong'] >= 0) + df_total['tempshort'] * (df_total['tempshort'] < 0)
            else:
                df_total['LngShtM'] = 0

            if exec_mode == 'not_trans':
                special_ls = df_total['Product'].apply(lambda x: cur_special_ls_dict.get(x, 0))
                special_ls += (special_ls == 0) * df_total['Class'].apply(lambda x: cur_special_ls_dict.get(x, 0))
                special_ls += (special_ls == 0) * cur_special_ls_dict.get('ALLPROD', 0)
                
                special_ls = np.where(np.abs(special_ls) > 1, special_ls * 1e4, np.round(special_ls * df_total['HoldMV'] / 1e4) * 1e4).astype('int')
            else: 
                special_ls = 0
            special_adj_p = df_total['Product'].apply(lambda x: paras_dict['product_special'].get(x, {'adjust_posr_active': True}).get('adjust_posr_active', False)) | (special_ls != 0) | (~ special_pos_r.isna())
            special_openi = df_total['Product'].apply(lambda x: paras_dict['product_special'].get(x, {'open_index_active': True}).get('open_index_active', True)) | (special_ls != 0) | (~ special_pos_r.isna())
            df_total['LngShtM'] = np.where((special_ls == 0) | ((df_total['ShortMain'] * 1e4 < special_ls) & (df_total['ShortMain'] != 0)), df_total['LngShtM'], special_ls) * special_adj_p

            df_total['ExpectHoldMV'] = df_total['HoldMV'] + df_total['LngShtM']
            df_total['EpctFutValue'] = ((df_total['ExpectHoldMV'] - expose_target_ratio * df_total[expose_target_benchmark]) * df_total['Idx2FtrR'])
            df_total['EpctFutValue'] = df_total['EpctFutValue'] * special_openi + df_total['FtrValue'] * (~ special_openi)

            df_total['tempReserve'] = np.maximum(stocks_reserve_min_money, stocks_reserve_min_ratio * df_total['ExpectHoldMV']) * (df_total['LngShtM'] > 0)

            if exec_mode == 'trans':
                flag_equity = df_total['Product'].isin(paras_dict.get('PrdType_Equity', []))
                except_capital = np.where(flag_equity, equity_dc_mctr * df_total['RealTHoldMV'], margin_call_dc_target_ratio * df_total['EpctFutValue'])
                except_capital = np.where(flag_hybrid, hybrid_dc_mctr * df_total['RealTHoldMV'], except_capital)

                df_total['Transfer'] = except_capital - df_total['Capital']
                df_total['Transfer'] *= np.abs(df_total['Transfer'] / df_total['EpctFutValue']) >= margin_call_dc_target_buffer_ratio
                df_total['Transfer'] = \
                    np.maximum(np.minimum(df_total['Transfer'], df_total['StockMoney'] - np.maximum(df_total['LngShtM'], 0) - df_total['tempReserve']), 0) * (df_total['Transfer'] > 0) + \
                    np.maximum(df_total['Transfer'], - df_total['WithdrawQuota']) * (df_total['Transfer'] <= 0)
                df_total['Transfer'] = np.floor(np.abs(df_total['Transfer']) / min_transfer_unit) * min_transfer_unit * np.sign(df_total['Transfer'])
            else:
                df_total['Transfer'] = 0

            if exec_mode == 'not_trans':
                """期货户根据保证金情况，得到最小市值"""
                df_total['EpctFutValue'] = np.minimum(
                    df_total['EpctFutValue'],
                    np.maximum(df_total['Capital'] / margin_exec_min_ratio * (df_total['EpctFutValue'] > df_total['FtrValue']), df_total['FtrValue']) +
                    df_total['EpctFutValue'] * (df_total['EpctFutValue'] <= df_total['FtrValue']))

                """股票户根据股票户资金情况，得到最小市值"""
                df_total['ExpectHoldMV'] = np.minimum(
                    df_total['ExpectHoldMV'],
                    df_total['ExpectHoldMV'] * (df_total['LngShtM'] <= 0) +
                    (df_total['HoldMV'] + np.minimum(
                        df_total['StockMoney'] - df_total['tempReserve'], df_total['StockMLong'])) * (df_total['LngShtM'] > 0))

                """股票户根据期货户市值调整市值"""
                df_total['ExpectHoldMV'] = np.minimum(
                    df_total['ExpectHoldMV'], df_total['EpctFutValue'] * df_total['Ftr2IdxR'] + expose_target_ratio * df_total[expose_target_benchmark])

                """期货户根据股票户市值调整市值"""
                df_total['EpctFutValue'] = np.minimum(
                    df_total['EpctFutValue'], (df_total['ExpectHoldMV'] - expose_target_ratio * df_total[expose_target_benchmark]) * df_total['Idx2FtrR'])

                df_total['LngShtM'] = df_total['ExpectHoldMV'] - df_total['HoldMV']

            df_total['ExpTargetM'] = expose_target_ratio * df_total[expose_target_benchmark]
            df_total = df_total.drop(['templong', 'tempshort', 'tempReserve'], axis=1)
        else:
            df_total['ExpectHoldMV'] = df_total[exec_mode.split('-')[1]]
            df_total['Transfer'] = 0
            df_total['LngShtM'] = 0

            if exp_same_direction_add:
                df_total['ExpTargetM'] = \
                    long_exp_target_ratio * df_total[expose_target_benchmark] * (df_total['ExpM'] > 0) + \
                    short_exp_target_ratio * df_total[expose_target_benchmark] * (df_total['ExpM'] <= 0)
            else:
                df_total['ExpTargetM'] = \
                    np.minimum(long_exp_target_ratio * df_total[expose_target_benchmark], df_total['ExpM']) * (df_total['ExpM'] > 0) + \
                    np.maximum(short_exp_target_ratio * df_total[expose_target_benchmark], df_total['ExpM']) * (df_total['ExpM'] <= 0)

            df_total['EpctFutValue'] = np.maximum(df_total['ExpectHoldMV'] - df_total['ExpTargetM'], 0)
            df_total['ExpTargetM'] = df_total['ExpectHoldMV'] - df_total['EpctFutValue']
            df_total['EpctFutValue'] = df_total['EpctFutValue'] * df_total['Idx2FtrR']

        # 期货round 造成的pos 偏离小于 position_adj_min_ratio
        round_thres_flag = (df_total['Proportion'] * df_total[f'IdxValueArr'])
        round_thres_flag = position_adj_min_ratio * df_total['EstmNAV'] / pd.Series(np.sum(np.stack(round_thres_flag.values), axis=1), index=round_thres_flag.index)
        round_thres_flag = 1 - np.minimum(round_thres_flag, 0.5)

        if future_t0_forbid:
            fut_mv_dir = df_total['FtrValue'] > df_total['EpctFutValue']
            cur_mv_future = df_total['HedgePos'] * df_total['FtrValueArr']
            target_mv_future = - df_total['EpctFutValue'] * df_total['Proportion']
            flag_short_pos = self.custom_compare_array_in_series(cur_mv_future, target_mv_future)

            flag_freeze = flag_short_pos * (~ fut_mv_dir) + (~ flag_short_pos) * fut_mv_dir
            new_allocate_r = (~ flag_freeze) * df_total['Proportion']
            new_allocate_r_sum = pd.Series(np.sum(np.stack(new_allocate_r.values), axis=1), index=new_allocate_r.index)
            new_allocate_r /= np.where(new_allocate_r_sum == 0, 1, new_allocate_r_sum)

            fix_target_mv_fut = cur_mv_future * flag_freeze
            fix_target_mv_fut += (- df_total['EpctFutValue'] - pd.Series(np.sum(np.stack(fix_target_mv_fut.values), axis=1), index=fix_target_mv_fut.index)) * new_allocate_r
            expect_pos_init = fix_target_mv_fut * df_total['Ftr2IdxR'] / df_total[f'IdxValueArr']
        else:
            expect_pos_init = - df_total['EpctFutValue'] * df_total['Ftr2IdxR'] * df_total['Proportion'] / df_total[f'IdxValueArr']

        expect_pos_init_sum = pd.Series(np.sum(np.stack(expect_pos_init.values), axis=1), index=expect_pos_init.index)
        expect_pos_init_round = self.custom_round_future_position(expect_pos_init, round_thres_flag) * (expect_pos_init_sum != 0)

        expect_round_diff = expect_pos_init - expect_pos_init_round
        expect_round_diff_sum = pd.Series(np.sum(np.stack(expect_round_diff.values), axis=1), index=expect_round_diff.index)

        expect_round_diff_round = (self.custom_round_future_position(expect_round_diff_sum, round_thres_flag) * (expect_round_diff_sum < 0) * expect_round_diff.apply(lambda x: x == np.min(x)) +
                                   self.custom_round_future_position(expect_round_diff_sum, 1 - round_thres_flag) * (expect_round_diff_sum > 0) * expect_round_diff.apply(lambda x: x == np.max(x)))
        # 计算调整后期货仓位
        expect_pos_round = expect_pos_init_round + expect_round_diff_round

        df_total['ExpectPos'] = expect_pos_round.apply(lambda x: x.astype('int'))
        expect_fix_round_expm = ((expect_pos_round - expect_pos_init) * df_total['IdxValueArr'])
        expect_fix_round_expm = pd.Series(np.sum(np.stack(expect_fix_round_expm.values), axis=1), index=expect_fix_round_expm.index)
        if not exec_mode.startswith('adj_exp'):
            df_total['LngShtM'] -= expect_fix_round_expm
            df_total['ExpectHoldMV'] = df_total['HoldMV'] + df_total['LngShtM']

        df_total['ExecPos'] = df_total[f'ExpectPos'] - df_total['HedgePos']

        df_total['LngShtM'] *= (~ normal_interval_flag) & adj_min_ratio_filter + (~ adj_min_ratio_filter)
        df_total['ExecPos'] *= (~ normal_interval_flag) & adj_min_ratio_filter + (~ adj_min_ratio_filter)

        df_total['ExpectPos'] = df_total['HedgePos'] + df_total[f'ExecPos']
        df_total['EpctFutValue'] = (df_total[f'ExpectPos'] * df_total[f'FtrValueArr'])
        df_total['EpctFutValue'] = pd.Series(np.sum(np.stack(df_total['EpctFutValue'].values), axis=1), index=df_total.index)

        df_total['LngShtM'] *= np.abs(df_total['LngShtM']) >= (position_dvt_tolerate_max_ratio * df_total['EstmNAV'])
        df_total['LngShtM'] = (np.round((df_total['LngShtM'].fillna(0)) / 1e4) * 1e4).astype('int')
        df_total['ExpectHoldMV'] = df_total['HoldMV'] + df_total['LngShtM']

        if not df_total.empty:
            df_style, df = self.adjust_match_trade_accs(df_total, df_stocks, exec_mode, paras_dict)
        else:
            df = pd.DataFrame(columns=paras_dict['formated_columns'])
            df_style = df.style

        return df_style, df

    def process(self, *args):
        df_total_origin, df_stocks, paras_dict = args
        df_total = df_total_origin.copy(deep=True)

        """盘中调整仓位-执行调拨"""
        df_trans_style, df_trans = self.adjust(
            df_total, df_stocks, 'trans', paras_dict)

        """盘中调整仓位-不执行调拨"""
        df_not_trans_style, df_not_trans = self.adjust(
            df_total, df_stocks, 'not_trans', paras_dict)

        """HoldMV暴露与调节"""
        df_adj_exp_holdmv_style, df_adj_exp_holdmv = self.adjust(
            df_total, df_stocks, 'adj_exp-HoldMV', paras_dict)
        df_adj_exp_quota_style, df_adj_exp_quota = self.adjust(
            df_total, df_stocks, 'adj_exp-QuotaMV', paras_dict)
        df_adj_exp_real_time_style, df_adj_exp_real_time = self.adjust(
            df_total, df_stocks, 'adj_exp-RealTHoldMV', paras_dict)

        dict_adj_res = dict()
        if 'df_trans_style' in locals():
            dict_adj_res['trans'] = eval('df_trans_style')
        if 'df_not_trans_style' in locals():
            dict_adj_res['not_trans'] = eval('df_not_trans_style')
        if 'df_adj_exp_holdmv_style' in locals():
            dict_adj_res['adj_exp-HoldMV'] = eval('df_adj_exp_holdmv_style')
        if 'df_adj_exp_quota_style' in locals():
            dict_adj_res['adj_exp-QuotaMV'] = eval('df_adj_exp_quota_style')
        if 'df_adj_exp_real_time_style' in locals():
            dict_adj_res['adj_exp-RealTHoldMV'] = eval('df_adj_exp_real_time_style')

        output_dir = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/LongShort/{self.curdate}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        time_flag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        path_total_res = f'{output_dir}{self.curdate}_Total_AdjMargin_{time_flag}.xlsx'

        with pd.ExcelWriter(path_total_res, engine='openpyxl') as writer:
            for adj_type in dict_adj_res:
                dict_adj_res[adj_type].to_excel(writer, sheet_name=adj_type, index=False)
            df_total_origin.to_excel(writer, sheet_name='账户原始数据', index=False)
            self.df_cash_out_plan.to_excel(writer, sheet_name='出金原始数据', index=False)

        # self.output_config_targetmv(df_not_trans)


class AdjustPosition(ProcessTradingDependData):
    def __init__(self, curdate, run_paras_dict, apply_zq_history_list=None):
        super().__init__(curdate=curdate, paras_dict=run_paras_dict)

        self.apply_zq_history_list = apply_zq_history_list if apply_zq_history_list is not None else []
        self.process_check = ProcessCheck()

    def adjust_add_check(self, df_result, paras_dict, exec_mode):
        margin_ratio_short_target = paras_dict.get('margin_ratio_short_target', 0.27)
        margin_exec_min_ratio = paras_dict.get('margin_exec_min_ratio', 0.14)
        margin_base_ratio = paras_dict.get('margin_base_ratio', 0.12)
        mini_product_nav_thres = paras_dict.get('mini_product_nav_thres', 3e7)

        df_result_check = df_result.copy(deep=True)
        long_holdmv = np.maximum(df_result_check['LngShtM'], 0).sum()
        short_holdmv = np.minimum(df_result_check['LngShtM'], 0).sum()
        fut_oper_dict = self.statistic_future_operation_num(df_result_check['ExecPos'].to_list(), future_dir='LONG')

        # print(f"""'加减仓：总加仓市值：', {long_holdmv}""")
        # print(f"""'加减仓：总平仓市值：', {short_holdmv}""")
        # print(f"""'加减仓：期货操作张数: ', {fut_oper_dict}""")

        mini_product_list = df_result_check[
            (df_result_check['EstmNAV'] <= mini_product_nav_thres)]['Product'].to_list()
        not_enough_money_prod = df_result_check[
            ((df_result_check['StockMoney'] -
              np.maximum(df_result_check['LngShtM'], 0)) <= 0) & (df_result_check['LngShtM'] > 0)]['Product'].to_list()

        df_result = self.format_result_excel_columns(
            df_result, format_type='W', columns_list=[
                'RealTHoldMV', 'HoldMV', 'QuotaMV', 'StockMoney', 'Capital', 'Margin', 'Transfer', 'NetValue',
                'StockMLong', 'ExpTargetM', 'WithdrawQuota', 'EstmNAV', 'ExpectHoldMV', 'ExpectMargin', 'BankCash',
                'StockRealTime', 'StockOutAvail', 'ExpAdjM', 'EpctFutValue', 'ExpM', 'QExpM', 'RTExpM', 'FtrValue', 'FtrVLSMx', 'FtrIdxValue'])
        df_result = self.format_result_excel_columns(df_result, format_type='ListW', columns_list=['IdxValueArr', 'FtrValueArr', 'ExpAdjMV', 'ExpMV'])
        df_result = self.format_result_excel_columns(df_result, format_type='+W', columns_list=['LngShtM'])
        df_result = self.format_result_excel_columns(df_result, format_type='ExecPosLong', columns_list=['ExecPos'])
        df_result = self.format_result_excel_columns(df_result, format_type='array2list', columns_list=['HedgePos', 'Proportion', 'ExpMRV'])

        if exec_mode == 'trans':
            df_result = df_result.reindex(columns=paras_dict['formated_columns_trans'], fill_value=np.nan).dropna(how='all', axis=1)
        else:
            df_result = df_result.reindex(columns=paras_dict['formated_columns'], fill_value=np.nan).dropna(how='all', axis=1)
        df_result = df_result.sort_values(['Class', 'HoldMVR', 'Main'], ascending=False)
        df_result = df_result.reset_index(drop=True)

        product_list_adj = df_result['Product'].to_list()
        df_result_style = df_result.fillna('0').replace('', '0').style.set_properties(
            **{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]).set_table_styles([
            {'selector': 'th', 'props': [('border', '1px solid black')]},
            {'selector': 'td', 'props': [('border', '1px solid black')]}]
        ).apply(lambda x: self.process_check.flag_product(x, mini_product_list, paras_dict), axis=0,
                subset=['Product', 'Main']).apply(
            lambda x: ['background-color: {0}'.format('#5B9B00')
                        if self.process_check.flag_cash_ratio(v, 3) else '' for v in x], axis=0,
            subset=['Left', 'StkCshR']).apply(
            lambda x: ['background-color: {0}'.format('#FFB2DE') if float(v) > 1 else '' for v in x], axis=0,
            subset=['BankCashRatio', 'CptlRatio']).apply(
            lambda x: ['background-color: {0}'.format('#FFB2DE')
                       if p in not_enough_money_prod else '' for p in product_list_adj], axis=0,
            subset=['LngShtM', 'StockMoney']).background_gradient(
            cmap='RdYlGn_r', subset=['LngSht%'], vmin=-20, vmax=20).background_gradient(
            cmap='Reds', subset=['HoldMVR', 'EpctMVR', 'EstmNVR']).background_gradient(
            cmap='RdYlGn_r', subset=['MExpMR', 'ExpMR', 'RTExpMR', 'QExpMR', 'AdjNVR'], vmin=-3, vmax=3).background_gradient(
            cmap='RdYlGn', subset=['MxDayMgR', 'IniMaxMgR', 'MxNxtMgR', 'TrsExcMgR', 'MgRTrsExc'],
            vmin=margin_exec_min_ratio * 100, vmax=margin_ratio_short_target * 100).apply(
            lambda x: ['background-color: {0}'.format('#5B9B00')
                       if float(v) <= (margin_base_ratio * 100) else ('background-color: {0}'.format('#FFB2DE')
                       if float(v) <= (margin_exec_min_ratio * 100) else '') for v in x], axis=0, subset=['ExcMgRTrs'])

        return df_result_style, df_result

    def adjust_match_trade_accs(self, df_total_data, df_stocks, exec_mode, paras_dict):
        subacc_position_dvt_tolerate_max_ratio = paras_dict.get('subacc_position_dvt_tolerate_max_ratio', 0.002)
        stock_mv_t0_forbid = paras_dict.get('stock_mv_t0_forbid', False)
        min_transfer_unit = paras_dict.get('min_transfer_unit', 1e4)

        df_stocks = df_stocks.copy(deep=True)
        df_data = df_total_data.copy(deep=True).reset_index(drop=True)
        df_data['ExpectMargin'] = df_data['EpctFutValue'] * self.margin_base_ratio

        expect_idx_value = (df_data[f'ExpectPos'] * df_data[f'IdxValueArr'])
        expect_idx_value = pd.Series(np.sum(np.stack(expect_idx_value.values), axis=1), index=expect_idx_value.index)
        """计算期望暴露情况"""
        df_data[f'ExpAdjM'] = (
                (expect_idx_value - (df_data['EstmNAV'] - df_data['ExpectHoldMV'])) * df_data['IsOpenIdx'] +
                (expect_idx_value + df_data['ExpectHoldMV'] - df_data['EpctMVR'] * df_data['EstmNAV']) * (1 - df_data['IsOpenIdx']) +
                df_data['IsExchange'] * 1e4)
        df_data['ExpAdjMV'] = df_data[f'ExpAdjM'] * df_data['Proportion']
        df_data[f'AdjNVR'] = np.round(df_data[f'ExpAdjM'] / df_data['EstmNAV'] * 100, 1)

        df_data[f'ExpectPos'] = self.format_future_position_list_2_dict(
            df_data['ExpectPos'], df_data['HedgePos'], self.future_list)

        """调整前后多头占净资产比例"""
        df_data['TransNVR'] = np.round(df_data['Transfer'] / df_data['EstmNAV'] * 100, 1)
        df_data['HoldMVR'] = np.round(df_data['HoldMV'] / df_data['EstmNAV'] * 100, 1)
        df_data['HoldMVR'] = np.where((df_data['HoldMVR'] < 0) | (df_data['HoldMVR'] > 200), 100, df_data['HoldMVR'])
        df_data['FtrMVR'] = np.round(df_data['FtrValue'] / df_data['EstmNAV'] * 100, 1)
        df_data['EpctMVR'] = np.round(df_data['ExpectHoldMV'] / df_data['EstmNAV'] * 100, 1)
        df_data['EstmNVR'] = np.round(((df_data['EstmNAV'] - df_data['ShortMain'] * 10000) / df_data['NetValue'] * 100).fillna(100), 1)

        """计算股票户剩余现金情况"""
        df_data[f'Left'] = np.round((df_data['StockMoney'] - df_data['LngShtM']) / df_data[f'ExpectHoldMV'] * 100, 1)

        """计算保证金比例情况"""
        df_data['MgRTrsExc'] = np.round(df_data['Capital'] / np.abs(df_data[f'FtrVLSMx']) * 100, 1).replace([-np.inf, np.inf, np.nan], 100)
        df_data['TrsExcMgR'] = np.round((df_data['Capital'] + df_data['Transfer']) / np.abs(df_data[f'EpctFutValue']) * 100, 1).replace([-np.inf, np.inf, np.nan], 100)
        df_data['ExcMgRTrs'] = np.round(df_data['Capital'] / np.abs(df_data[f'EpctFutValue']) * 100, 1).replace([-np.inf, np.inf, np.nan], 100)
        df_data['IniMaxMgR'] = np.round(
            (df_data['Capital'] + df_data['StockOutAvail']) / np.abs(df_data[f'FtrValue']) * 100, 1).replace([-np.inf, np.inf, np.nan], 100)
        df_data['MxDayMgR'] = np.round(
            (df_data['Capital'] + df_data['StockOutAvail'] -
             np.maximum(df_data['LngShtM'], 0)) / np.abs(df_data[f'EpctFutValue']) * 100, 1).replace([-np.inf, np.inf, np.nan], 100)
        df_data['MxNxtMgR'] = np.round(
            (df_data['Capital'] + df_data['StockMoney'] - df_data['LngShtM']) / np.abs(df_data[f'EpctFutValue']) * 100, 1).replace([-np.inf, np.inf, np.nan], 100)

        conlist = []
        for product, df_product in df_data.groupby('Product'):
            df_stock_prod = df_stocks[df_stocks['Main'] == df_product['Main'].iloc[0]].copy(deep=True)
            longshort_mv = df_product['LngShtM'].iloc[0]
            trans_money = df_product['Transfer'].iloc[0]
            target_holdmv = df_product['ExpectHoldMV'].iloc[0]
            stock_money = df_product['StockMoney'].iloc[0]
            stock_money_long = df_product['StockMLong'].iloc[0]

            if exec_mode.startswith('adj_exp') or exec_mode.startswith('report'):
                df_stock_prod['LngShtM'] = 0
            else:
                df_stock_prod = self.custom_ls_mv_match_trade_accounts(
                    df_stock_prod, longshort_mv, target_holdmv, stock_money_long, stock_money, stock_mv_t0_forbid)

            adj_mv_series = np.where(df_stock_prod['HoldMV'] != 0, df_stock_prod['HoldMV'], np.abs(df_stock_prod['LngShtM']))
            df_stock_prod['LngSht%'] = np.round((df_stock_prod['LngShtM'] / adj_mv_series).replace([np.inf, -np.inf], 1).fillna(1) * 100, 1)
            df_stock_prod['LngShtM'] *= np.abs(df_stock_prod['LngSht%']) > (subacc_position_dvt_tolerate_max_ratio * 100)

            expect_holdmv = df_stock_prod['LngShtM'] + df_stock_prod['HoldMV']
            expect_holdmv_total = expect_holdmv.sum()
            temp_avail = np.maximum(np.minimum(df_stock_prod['StockOutAvail'], df_stock_prod['StockMoney'] - np.maximum(df_stock_prod['LngShtM'], 0)), 0)

            total_avail = temp_avail.sum()
            if trans_money > 0:
                trans_money = min(trans_money, total_avail)
                target_cash_r = (total_avail - trans_money) / expect_holdmv_total
                avail_cash_r = temp_avail / expect_holdmv
                avail_cash = temp_avail * (avail_cash_r > target_cash_r)
                df_stock_prod['Transfer'] = avail_cash / avail_cash.sum() * trans_money
            else:
                df_stock_prod['Transfer'] = expect_holdmv / expect_holdmv_total * trans_money

            df_stock_prod['Transfer'] = np.floor(np.abs(df_stock_prod['Transfer']) / min_transfer_unit) * min_transfer_unit * np.sign(df_stock_prod['Transfer'])

            df_stock_prod['ExpMR'] = df_product['ExpMR'].iloc[0]
            df_stock_prod['MExpMR'] = df_product['MExpMR'].iloc[0]
            df_stock_prod['QExpMR'] = df_product['QExpMR'].iloc[0]
            df_stock_prod['RTExpMR'] = df_product['RTExpMR'].iloc[0]

            df_stock_prod['EpctMVR'] = df_product['EpctMVR'].iloc[0]
            df_stock_prod['HoldMVR'] = df_product['HoldMVR'].iloc[0]
            df_stock_prod['EstmNVR'] = df_product['EstmNVR'].iloc[0]

            df_stock_prod['MgRTrsExc'] = df_product['MgRTrsExc'].iloc[0]
            df_stock_prod['ExcMgRTrs'] = df_product['ExcMgRTrs'].iloc[0]
            df_stock_prod['TrsExcMgR'] = df_product['TrsExcMgR'].iloc[0]
            df_stock_prod['IniMaxMgR'] = df_product['IniMaxMgR'].iloc[0]
            df_stock_prod['MxDayMgR'] = df_product['MxDayMgR'].iloc[0]
            df_stock_prod['MxNxtMgR'] = df_product['MxNxtMgR'].iloc[0]

            df_stock_prod['AdjNVR'] = df_product['AdjNVR'].iloc[0]

            df_stock_prod['Left'] = (df_stock_prod['StockMoney'] - df_stock_prod['LngShtM']) / (df_stock_prod['HoldMV'] + df_stock_prod['LngShtM']) * 100
            df_stock_prod['Left'] = \
                np.round(df_stock_prod['Left'], 1).astype('str') + '/' + str(round(df_product['Left'].iloc[0], 1))
            df_stock_prod['StkCshR'] = \
                np.round(df_stock_prod['StkCshR'], 1).astype('str') + '/' + str(round(df_product['StkCshR'].iloc[0], 1))
            df_stock_prod['CptlRatio'] = df_product['CptlRatio'].iloc[0]
            conlist.append(df_stock_prod)

        df_stock_oper = pd.concat(conlist, axis=0)

        df_stock_oper['LngShtM'] = (np.round((df_stock_oper['LngShtM']) / 1e4) * 1e4).astype('int')
        adj_mv_series = np.where(df_stock_oper['HoldMV'] != 0, df_stock_oper['HoldMV'], np.abs(df_stock_oper['LngShtM']))
        df_stock_oper['LngSht%'] = np.round((df_stock_oper['LngShtM'] / adj_mv_series).replace([np.inf, -np.inf], 1).fillna(1) * 100, 1)

        df_data = df_data.rename({'Transfer': 'TtlTrans', 'LngShtM': 'TtlLS'}, axis='columns').set_index(['Product', 'Main'])
        drop_columns_list = list(set(df_stock_oper.columns.to_list()).intersection(list(df_data.columns.to_list())))
        df_data = df_data.drop(drop_columns_list, axis=1).reset_index()

        df_result = pd.merge(df_stock_oper, df_data, on=['Product', 'Main'], how='outer')

        df_result['TFlag'] = datetime.datetime.now().strftime('%H%M%S')
        df_result['How'] = 'TargetMV'

        return self.adjust_add_check(df_result, paras_dict, exec_mode)

    def adjust(self, *args):
        df_total_data, df_stocks, exec_mode, paras_dict, flag_match_apply = args

        position_zq_adj_min_ratio = paras_dict.get('position_zq_adj_min_ratio', 0.01)
        expose_zq_adj_min_ratio = paras_dict.get('expose_zq_adj_min_ratio', 0.02)
        adj_min_ratio_filter = paras_dict.get('adj_min_ratio_filter', False)
        stocks_reserve_min_money = paras_dict.get('stocks_reserve_min_money', 2e4)
        stocks_reserve_min_ratio = paras_dict.get('stocks_reserve_min_ratio', 0.01)
        position_dvt_tolerate_max_ratio = paras_dict.get('position_dvt_tolerate_max_ratio', 0.01)

        apply_expose_recognition_min_ratio = paras_dict.get('apply_expose_recognition_min_ratio', 0.02)

        margin_call_zq_target_ratio = paras_dict.get('margin_call_zq_target_ratio', 0.22)
        margin_call_zq_target_buffer_ratio = paras_dict.get('margin_call_zq_target_buffer_ratio', 0.04)
        min_transfer_unit = paras_dict.get('min_transfer_unit', 5e4)

        margin_exec_min_ratio = paras_dict.get('margin_exec_min_ratio', 0.14)
        apply_exchange_short_multi = paras_dict.get('apply_exchange_short_multi', 1.16)
        
        cur_special_ls_dict = paras_dict.get('product_special_ls', {}).get(self.curdate, {})

        margin_fix_dict = paras_dict.get('margin_fix_dict', {})

        df_total = df_total_data.copy(deep=True)
        df_stocks = df_stocks.copy(deep=True)

        df_total = df_total[~ df_total['Class'].str.endswith('DC')]
        df_stocks = df_stocks[~ df_stocks['Class'].str.endswith('DC')]

        df_total['EpctMVR'] = 1 - df_total['CashTargetRatio'] #刨去预留现金比例
        special_flag = df_total['Product'].apply(lambda x: paras_dict['product_special'].get(x, None) is not None)
        special_pos_r = df_total.apply(
            lambda row: paras_dict['product_special'].get(row['Product'], {}).get('PositionRatio', row['EpctMVR']), axis=1)
        special_adj_p = df_total['Product'].apply(
            lambda x: paras_dict['product_special'].get(x, {'adjust_posr_active': True}).get('adjust_posr_active', False))
        special_openi = df_total['Product'].apply(
            lambda x: paras_dict['product_special'].get(x, {'open_index_active': True}).get('open_index_active', False))

        df_total['IsOpenIdx'] *= special_openi
        df_total['EpctMVR'] = df_total['EpctMVR'] * (~special_flag) + special_flag * special_pos_r

        """计算初始暴露情况"""
        temp_expm = (df_total['FtrIdxValue'] - df_total['Capital'] + df_total['ShortMain'] * 1e4)
        temp_expm_non_oi = (df_total['FtrIdxValue'] - df_total['EpctMVR'] * df_total['EstmNAVOpn']) 
        df_total['ExpM'] = (
                (temp_expm - df_total['StockMoney']) * df_total['IsOpenIdx'] +
                (temp_expm_non_oi + df_total['HoldMV']) * (1 - df_total['IsOpenIdx']))
        df_total['ExpMR'] = np.round(df_total['ExpM'] / df_total['EstmNAVOpn'] * 100, 1)

        df_total['ExpMV'] = df_total['ExpM'] * df_total['Proportion']
        df_total['ExpMRV'] = df_total['ExpMR'] * df_total['Proportion']

        df_total['RTExpM'] = (
                (temp_expm - df_total['StockRealTime']) * df_total['IsOpenIdx'] +
                (temp_expm_non_oi + df_total['RealTHoldMV']) * (1 - df_total['IsOpenIdx']))
        df_total['RTExpMR'] = np.round(df_total['RTExpM'] / df_total['EstmNAVOpn'] * 100, 1)

        df_total['QExpM'] = (
                (temp_expm - df_total['StockMoney'] - (df_total['HoldMV'] - df_total['QuotaMV'])) * df_total['IsOpenIdx'] +
                (temp_expm_non_oi + df_total['QuotaMV']) * (1 - df_total['IsOpenIdx']))
        df_total['QExpMR'] = np.round(df_total['QExpM'] / df_total['EstmNAVOpn'] * 100, 1)

        df_total['MExpMR'] = np.round(df_total['ExpMRV'].apply(lambda x: np.max(np.abs(x))), 1)

        if exec_mode.startswith('report'):
            df_total['ExpectHoldMV'] = df_total['HoldMV']
            df_total['LngShtM'] = 0
            df_total['Transfer'] = 0
            df_total['ExpTargetM'] = 0
            df_total['EpctFutValue'] = df_total['FtrValue']
            df_total['ExpectPos'] = df_total['HedgePos']
            df_total['ExecPos'] = df_total['ExpectPos'] - df_total['HedgePos']

            return self.adjust_match_trade_accs(df_total, df_stocks, exec_mode, paras_dict)

        normal_interval_flag = ((np.abs(df_total['EpctMVR'] - df_total['HoldMVR'] / 100) < position_zq_adj_min_ratio) &
                                (np.abs(df_total['ExpMR'] * special_openi) < (expose_zq_adj_min_ratio * 100)) &
                                (~ df_total['Main'].isin(df_stocks[df_stocks['StkCshR'] < (stocks_reserve_min_ratio * 100)]['Main'].unique())))
        normal_interval_flag = normal_interval_flag & (df_total['ShortMain'] == 0) & (df_total['ApplyMain'] == 0) & (df_total['NApplyMain'] == 0)
        if adj_min_ratio_filter:
            if not exec_mode.startswith('trans'):
                df_total = df_total[~ normal_interval_flag]
                special_flag = special_flag[~ normal_interval_flag]
                special_adj_p = special_adj_p[~ normal_interval_flag]
                special_openi = special_openi[~ normal_interval_flag]

        special_ls = df_total['Product'].apply(lambda x: cur_special_ls_dict.get(x, 0))
        special_ls += (special_ls == 0) * df_total['Class'].apply(lambda x: cur_special_ls_dict.get(x, 0))
        special_ls += (special_ls == 0) * cur_special_ls_dict.get('ALLPROD', 0)

        special_ls = np.where(np.abs(special_ls) > 1, special_ls * 1e4, np.round(special_ls * df_total['HoldMV'] / 1e4) * 1e4).astype('int')

        df_total['ExpectHoldMV'] = np.where(((~special_flag) | (special_flag & special_adj_p)), df_total['EstmNAV'] * df_total['EpctMVR'], df_total['HoldMV'])
        df_total['LngShtM'] = np.where((special_ls == 0) | ((df_total['ShortMain'] * 1e4 < special_ls) & (df_total['ShortMain'] != 0)), df_total['ExpectHoldMV'] - df_total['HoldMV'], special_ls) * special_adj_p
        df_total['ExpectHoldMV'] = df_total['HoldMV'] + df_total['LngShtM']
        
        reserve_long = np.maximum(stocks_reserve_min_money, stocks_reserve_min_ratio * df_total['EstmNAV'])
        reserve_long = np.minimum(np.maximum(df_total['StockMoney'] - reserve_long, 0), df_total['StockMLong'])
        df_total['LngShtM'] = np.minimum(df_total['LngShtM'], np.where(df_total['LngShtM'] <= 0, df_total['LngShtM'], reserve_long))
        df_total['ExpectHoldMV'] = df_total['LngShtM'] + df_total['HoldMV']
        df_total['ExpTargetM'] = 0

        if exec_mode.startswith('adj_exp'):
            df_total['Transfer'] = 0
            df_total['LngShtM'] = 0
            df_total['ExpectHoldMV'] = df_total['HoldMV']
        
        df_total['EpctFutValue'] = np.where(
            df_total['IsOpenIdx'], df_total['EstmNAV'] - df_total['ExpectHoldMV'],
            np.minimum(df_total['FtrIdxValue'], df_total['EpctMVR'] * df_total['EstmNAV'] - df_total['ExpectHoldMV']))
        df_total['EpctFutValue'] = np.where(((~special_flag) + special_flag * special_openi), df_total['EpctFutValue'], df_total['FtrIdxValue']) - df_total['IsExchange'] * 1e4
        
        if flag_match_apply: apply_route = df_total['AlyOnRt'] * 1e4
        else: apply_route = (df_total['Product'].isin(self.apply_zq_history_list) * df_total['ExpM'] * (df_total['ExpMR'] > apply_expose_recognition_min_ratio * 100))        
        df_total['EpctFutValue'] += apply_route

        if not exec_mode.startswith('adj_exp'):
            apply_total = (df_total['ApplyMain'] + df_total['NApplyMain']) * df_total['Idx2FtrR'] * 1e4
            flag_exchange_apply = apply_total != 0
            exchange_apply = (apply_total * margin_exec_min_ratio - (df_total['Capital'] + df_total['StockMoney'] - df_total['LngShtM'] - df_total['EpctFutValue'] * margin_exec_min_ratio))
            exchange_apply = np.maximum(exchange_apply * apply_exchange_short_multi, 0) * flag_exchange_apply
            df_total['LngShtM'] -= exchange_apply
            df_total['EpctFutValue'] += exchange_apply
            df_total['EpctFutValue'] = np.maximum(df_total['EpctFutValue'], 0)

            adj_total_ratio = (np.abs(df_total['LngShtM']) + np.abs(df_total['EpctFutValue'] - df_total['FtrIdxValue'])) / df_total['EstmNAV']
            flag_adj = flag_exchange_apply | (adj_total_ratio > expose_zq_adj_min_ratio)
            df_total['LngShtM'] *= flag_adj
            df_total['EpctFutValue'] = np.where(flag_adj, df_total['EpctFutValue'], df_total['FtrIdxValue'])
            df_total['ExpectHoldMV'] = df_total['LngShtM'] + df_total['HoldMV']
        else:
            flag_exchange_apply = False

        if exec_mode == 'trans':
            series_value = df_total['FtrValue'] + df_total['Product'].apply(lambda x: margin_fix_dict.get(x, 0)) * 1e4 * df_total['Idx2FtrR']
            df_total['Transfer'] = series_value * margin_call_zq_target_ratio - df_total['Capital']
            df_total['Transfer'] *= np.abs(df_total['Transfer'] / series_value).replace([np.inf, -np.inf, np.nan], 0) >= margin_call_zq_target_buffer_ratio
            df_total['Transfer'] = \
                np.maximum(np.minimum(df_total['Transfer'], df_total['StockMoney']), 0) * (df_total['Transfer'] > 0) + \
                np.maximum(df_total['Transfer'], - df_total['WithdrawQuota']) * (df_total['Transfer'] <= 0)
            df_total['Transfer'] = np.floor(np.abs(df_total['Transfer']) / min_transfer_unit) * min_transfer_unit * np.sign(df_total['Transfer'])
            df_total['LngShtM'] = 0
            df_total['ExpectHoldMV'] = df_total['HoldMV']
            df_total['EpctFutValue'] = df_total['FtrIdxValue']
        else:
            df_total['Transfer'] = 0

        # # 期货round 造成的pos 偏离小于 position_zq_adj_min_ratio
        round_thres_flag = (df_total['Proportion'] * df_total[f'IdxValueArr'])
        round_thres_flag = position_zq_adj_min_ratio * df_total['EstmNAV'] / pd.Series(np.sum(np.stack(round_thres_flag.values), axis=1), index=round_thres_flag.index)
        round_thres_flag = 1 - np.minimum(round_thres_flag, 0.5)

        expect_pos_init = df_total['EpctFutValue'] * df_total['Proportion'] / df_total[f'IdxValueArr']
        expect_pos_init_sum = pd.Series(np.sum(np.stack(expect_pos_init.values), axis=1), index=expect_pos_init.index)

        expect_pos_init_round = self.custom_round_future_position(expect_pos_init, round_thres_flag) * (expect_pos_init_sum != 0)
        expect_round_diff = expect_pos_init - expect_pos_init_round
        expect_round_diff_sum = pd.Series(np.sum(np.stack(expect_round_diff.values), axis=1), index=expect_round_diff.index)
        expect_pos_init_round_sum = pd.Series(np.sum(np.stack(expect_pos_init_round.values), axis=1), index=expect_pos_init_round.index)

        flag_future_pos_neq_zero = (df_total['IsOpenIdx'].astype('bool') & (expect_pos_init_round_sum == 0)) | flag_exchange_apply
        expect_round_diff_round = (self.custom_round_future_position(expect_round_diff_sum, 1 - round_thres_flag) * (expect_round_diff_sum < 0) * expect_round_diff.apply(lambda x: x == np.min(x)) +
                                   self.custom_round_future_position(expect_round_diff_sum, round_thres_flag * (~ flag_future_pos_neq_zero)) * (expect_round_diff_sum > 0) * expect_round_diff.apply(lambda x: x == np.max(x)))
        # 计算调整后期货仓位
        expect_pos_round = expect_pos_init_round + expect_round_diff_round

        df_total['ExpectPos'] = expect_pos_round.apply(lambda x: x.astype('int'))
        expect_fix_round_expm = ((expect_pos_round - expect_pos_init) * df_total['IdxValueArr'])
        expect_fix_round_expm = pd.Series(np.sum(np.stack(expect_fix_round_expm.values), axis=1), index=expect_fix_round_expm.index)
        res_cash_ratio = (df_total['Capital'] + df_total['StockMoney'] - df_total['LngShtM'] + expect_fix_round_expm) / df_total['EstmNAV']
        """
        1. 产品市值太小，最小1张多头，现金占比较大；
        2. 产品的资金有部分在期货户或者双中心调拨的原因导致，现金占比较大；
        3. 产品持仓较大，暴露配平时，舍入问题造成的偏离目标资金比例；
        """
        df_total['LngShtM'] += - expect_fix_round_expm + np.where(
            (res_cash_ratio <= df_total['CashTargetRatio']) | (df_total['AlyOnRt'] != 0) | flag_exchange_apply | (round_thres_flag == 0.5), 0,
            np.minimum(np.minimum((res_cash_ratio - df_total['CashTargetRatio']) * df_total['EstmNAV'], expose_zq_adj_min_ratio * df_total['EstmNAV']),
                       np.maximum(reserve_long - np.maximum(df_total['LngShtM'] - expect_fix_round_expm, 0), 0)))
        
        df_total['LngShtM'] = np.minimum(df_total['LngShtM'], np.where(df_total['LngShtM'] <= 0, df_total['LngShtM'], reserve_long))
        df_total['ExpectHoldMV'] = df_total['HoldMV'] + df_total['LngShtM']

        df_total['ExecPos'] = df_total[f'ExpectPos'] - df_total['HedgePos']

        df_total['LngShtM'] *= (~ normal_interval_flag) & adj_min_ratio_filter + (~ adj_min_ratio_filter)
        df_total['ExecPos'] *= (~ normal_interval_flag) & adj_min_ratio_filter + (~ adj_min_ratio_filter)

        df_total['ExpectPos'] = df_total['HedgePos'] + df_total[f'ExecPos']
        df_total['EpctFutValue'] = (df_total[f'ExpectPos'] * df_total[f'FtrValueArr'])
        df_total['EpctFutValue'] = pd.Series(np.sum(np.stack(df_total['EpctFutValue'].values), axis=1), index=df_total.index)

        df_total['LngShtM'] *= (np.abs(df_total['LngShtM']) >= (position_dvt_tolerate_max_ratio * df_total['EstmNAV'])) | flag_exchange_apply
        df_total['LngShtM'] = (np.round((df_total['LngShtM'].fillna(0)) / 1e4) * 1e4).astype('int')
        df_total['ExpectHoldMV'] = df_total['HoldMV'] + df_total['LngShtM']
        
        if not df_total.empty:
            df_style, df = self.adjust_match_trade_accs(df_total, df_stocks, exec_mode, paras_dict)
        else:
            df = pd.DataFrame(columns=paras_dict['formated_columns'])
            df_style = df.style

        return df_style, df

    def adjust_open_index(self, *args):
        df_total_data, paras_dict, open_index, dict_oi, hedge_cfgout = args
        margin_exec_min_ratio = paras_dict.get('margin_exec_min_ratio', 0.014)
        min_transfer_unit = paras_dict.get('min_transfer_unit', 5e4)

        df_total = df_total_data.copy(deep=True)

        df_total['ApplyAmount'] = (df_total['ApplyMain'] if open_index == 'OpenIndex' else df_total['ApplyFixMain']) * 1e4
        df_total = df_total[df_total['ApplyAmount'] != 0].reset_index(drop=True)
        infor_list_cfg_out = []
        hedge_cfgout_dict = hedge_cfgout.set_index('Product').to_dict(orient='index')
        for oi_prd, oi_amount in dict_oi.items():
            if oi_prd not in df_total['Main'].to_list():
                class_prd = production_2_index(oi_prd) + production_2_strategy(oi_prd)
                hedgepos_dict = hedge_cfgout_dict.get(oi_prd, {}).get('HedgePosDict', {})
                hedgepos_dict = {} if str(hedgepos_dict) in ['nan', ''] else hedgepos_dict
                hedgepos_fn_dict = self.format_future_pos_2_future_name(hedgepos_dict)

                proportion = paras_dict['class_proportion_fix'].get(class_prd, production_2_proportion(oi_prd))
                infor_list_cfg_out.append({
                    'Product': oi_prd,
                    'Main': oi_prd,
                    'Class': class_prd,
                    'StockMoney': 0,
                    'HoldMV': 0,
                    'ApplyAmount': oi_amount * 1e4,
                    'Capital': hedge_cfgout_dict.get(oi_prd, {}).get('Capital', 0),
                    'Margin': hedge_cfgout_dict.get(oi_prd, {}).get('Margin', 0),
                    'WithdrawQuota': hedge_cfgout_dict.get(oi_prd, {}).get('WithdrawQuota', 0),
                    'HedgePosDict': hedgepos_dict,
                    'HedgePos': np.array([hedgepos_fn_dict.get(ind_nm, 0) for ind_nm in self.future_list]),
                    'Proportion': np.array([proportion.get(ind_nm, 0) for ind_nm in self.index_list]),
                    'IdxValueArr': deepcopy(df_total_data['IdxValueArr'].iloc[0]),
                    'FtrValueArr': deepcopy(df_total_data['FtrValueArr'].iloc[0]),
                })
        df_cfg_out = pd.DataFrame(infor_list_cfg_out)
        df_total = pd.concat([df_total, df_cfg_out], axis=0).reset_index(drop=True)

        df_total['EstmNAV'] = df_total['StockMoney'] + df_total['Capital'] + df_total['HoldMV'] + df_total['ApplyAmount']
        df_total['ExecPos'] = (df_total['ApplyAmount'] * df_total['Proportion'] / df_total['IdxValueArr']).apply(lambda x: np.round(x))
        df_total['ExpectPos'] = df_total['ExecPos'] + df_total['HedgePos']

        df_total['ApplyExpM'] = (pd.Series(np.sum(np.stack((df_total['ExecPos'] * df_total['IdxValueArr']).values), axis=1), index=df_total.index) - df_total['ApplyAmount'])
        df_total['ApplyExpMR'] = np.round(df_total['ApplyExpM'] / df_total['EstmNAV'] * 100, 1)

        # 计算保证金
        df_total['EpctFutValue'] = (df_total[f'ExpectPos'] * df_total[f'FtrValueArr'])
        df_total['EpctFutValue'] = pd.Series(np.sum(np.stack(df_total['EpctFutValue'].values), axis=1), index=df_total.index)
        df_total['ExcMgRTrs'] = np.round((df_total['Capital'] / np.abs(df_total[f'EpctFutValue'])).replace([np.inf, -np.inf], 1) * 100, 1)
        df_total['Transfer'] = - (df_total['Capital'] - df_total['EpctFutValue'] * margin_exec_min_ratio)
        df_total['Transfer'] = np.ceil(np.maximum(df_total['Transfer'], 0) / min_transfer_unit) * min_transfer_unit

        df_total[f'ExpectPos'] = self.format_future_position_list_2_dict(df_total['ExpectPos'], df_total['HedgePos'], self.future_list)

        df_total = self.format_result_excel_columns(
            df_total, format_type='W', columns_list=["ApplyAmount", "StockOutAvail", "EstmNAV", "Transfer"])
        df_total = self.format_result_excel_columns(
            df_total, format_type='ExecPosLong', columns_list=['ExecPos'])
        df_total['Colo'] = df_total['Product'].apply(lambda x: production_2_colo(x))

        df_total = df_total.reindex(columns=paras_dict['formated_columns_oi'], fill_value=0)

        df_total = df_total.reset_index(drop=True)
        df_io_proces = df_total.copy(deep=True).rename({'ApplyAmount': 'Apply'}, axis='columns')

        df_io_proces['LngShtM'] = np.nan
        df_io_proces['How'] = open_index
        df_io_proces['PlanIn'] = 1
        df_io_proces = df_io_proces.reindex(columns=paras_dict['formated_columns_process'], fill_value=0)

        return df_total, df_io_proces

    def process(self, *args):
        df_total_origin, df_stocks, paras_dict = args
        df_total = df_total_origin.copy(deep=True)

        """盘中调整仓位-执行调拨"""
        df_adj_style, df_adj = self.adjust(
            df_total, df_stocks, 'not_trans', paras_dict)

        """HoldMV暴露与调节"""
        df_adj_exp_holdmv_style, df_adj_exp_holdmv = self.adjust(
            df_total, df_stocks, 'adj_exp-HoldMV', paras_dict)
        df_adj_exp_quota_style, df_adj_exp_quota = self.adjust(
            df_total, df_stocks, 'adj_exp-QuotaMV', paras_dict)
        df_adj_exp_real_time_style, df_adj_exp_real_time = self.adjust(
            df_total, df_stocks, 'adj_exp-RealTHoldMV', paras_dict)

        dict_adj_res = dict()
        if 'df_adj_style' in locals():
            dict_adj_res['调整'] = eval('df_adj_style')
        if 'df_adj_exp_holdmv_style' in locals():
            dict_adj_res['adj_exp-HoldMV'] = eval('df_adj_exp_holdmv_style')
        if 'df_adj_exp_quota_style' in locals():
            dict_adj_res['adj_exp-QuotaMV'] = eval('df_adj_exp_quota_style')
        if 'df_adj_exp_real_time_style' in locals():
            dict_adj_res['adj_exp-RealTHoldMV'] = eval('df_adj_exp_real_time_style')

        output_dir = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/LongShort/{self.curdate}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        time_flag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        path_total_res = f'{output_dir}{self.curdate}_Total_AdjPosition_{time_flag}.xlsx'

        with pd.ExcelWriter(path_total_res, engine='openpyxl') as writer:
            for adj_type in dict_adj_res:
                dict_adj_res[adj_type].to_excel(writer, sheet_name=adj_type, index=False)


class ProcessCheck(GetProductionInformation):
    def __init__(self):
        super().__init__()

    def statistics_future_position_nums(self, curdate, df_total_data, output_dir):
        df_total_data = df_total_data.copy(deep=True)
        df_total_data['Swap'] = np.where(~ df_total_data['Main'].isin(Production_OwnList_Swap + Production_OptionList), '--', df_total_data['Main'])
        df_total_data['Number'] = 1
        infor_list = []
        for (clss, isswap), df in df_total_data.groupby(['Class', 'Swap']):
            df_hedge_pos = pd.DataFrame(df['HedgePosDict'].to_list()).sum(axis=0).to_frame().T
            if not df_hedge_pos.empty:
                df_hedge_pos['Class'] = clss
                df_hedge_pos['Swap'] = isswap
                df_hedge_pos['Number'] = df['Number'].sum()
                df_hedge_pos['HoldMV'] = df['RealTHoldMV'].sum()
                df_hedge_pos['EstmNAV'] = df['EstmNAV'].sum()
                df_hedge_pos['NetValue'] = df['NetValue'].sum()
                infor_list.append(df_hedge_pos)
            else:
                infor_list.append(pd.DataFrame([{
                    'Class': clss,
                    'Swap': isswap,
                    'Number': df['Number'].sum(),
                    'HoldMV': df['RealTHoldMV'].sum(),
                    'NetValue': df['NetValue'].sum(),
                    'EstmNAV': df['EstmNAV'].sum()}
                ]))
        
        df_hedge_pos_total = pd.DataFrame(df_total_data['HedgePosDict'].to_list()).sum(axis=0).to_frame().T
        df_hedge_pos_total['Class'] = ''
        df_hedge_pos_total['Swap'] = ''
        df_hedge_pos_total['Number'] = df_total_data['Number'].sum()
        df_hedge_pos_total['HoldMV'] = df_total_data['RealTHoldMV'].sum()
        df_hedge_pos_total['EstmNAV'] = df_total_data['EstmNAV'].sum()
        df_hedge_pos_total['NetValue'] = df_total_data['NetValue'].sum()
        infor_list.append(df_hedge_pos_total)

        df_hedge_pos = pd.concat(infor_list).fillna(0)
        df_hedge_pos['MVR'] = np.round(df_hedge_pos['HoldMV'] / df_hedge_pos['NetValue'] * 100, 1)
        df_hedge_pos['EMVR'] = np.round(df_hedge_pos['HoldMV'] / df_hedge_pos['EstmNAV'] * 100, 1)
        df_hedge_pos['HoldMV'] = np.round(df_hedge_pos['HoldMV'] / 1e8, 1)
        df_hedge_pos['EstmNAV'] = np.round(df_hedge_pos['EstmNAV'] / 1e8, 1)
        df_hedge_pos['NetValue'] = np.round(df_hedge_pos['NetValue'] / 1e8, 1)
        df_hedge_pos['Number'] = df_hedge_pos['Number'].astype('int')
        df_hedge_pos = df_hedge_pos.set_index(['Swap', 'Class', 'Number', 'HoldMV', 'EstmNAV', 'NetValue', 'MVR', 'EMVR']).astype('int').T.sort_index().T

        future_name_list = df_hedge_pos.columns.to_list()
        df_hedge_pos = df_hedge_pos.reset_index().sort_values(['Swap', 'EMVR'], ascending=True).astype('str').reset_index(drop=True)
        df_hedge_pos = df_hedge_pos.style.set_properties(
            **{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]).set_table_styles([
                {'selector': 'th', 'props': [('border', '1px solid black')]},
            {'selector': 'td', 'props': [('border', '1px solid black')]}]).background_gradient(
            cmap='Reds', subset=['MVR', 'EMVR']).apply(
            lambda x: ['background-color: red' if float(v) > 0 else ('background-color: green' if float(v) < 0 else '') for v in x], axis=0, subset=future_name_list)
        excel_path = output_dir + f'statistics_future_position_{curdate}.xlsx'
        img_path = output_dir + f'statistics_future_position_{curdate}.png'
        df_hedge_pos.to_excel(excel_path, index=False)
        try:
            dfi.export(df_hedge_pos, filename=img_path, dpi=300, fontsize=10, max_cols=-1, max_rows=-1, table_conversion='chrome')
            wechat_bot_image(img_path, type_api='analysis-report')
        except:
            wechat_bot_file(output_dir + f'statistics_future_position_{curdate}.xlsx', type_api='check')

    def check_future_data(self, date, df_capital, product_list):
        date_now = datetime.datetime.now()
        date_cur, time_flag = date_now.strftime('%Y%m%d-%H%M%S').split('-')

        df_capital['UpdateTime'] = df_capital['UpdateTime'].fillna(date_cur + time_flag).apply(
            lambda x: datetime.datetime.strptime(str(x).replace('T', ''), '%Y%m%d%H%M%S'))
        df_capital['TimeDiff'] = (date_now - df_capital['UpdateTime']).apply(lambda x: x.seconds if x.days >= 0 else 0)

        df_capital_delay = df_capital[(df_capital['TimeDiff'] > 60) & df_capital['Product'].isin(product_list)]

        df_capital_nonfnd = df_capital[(df_capital['Margin'] > 0) & (~ df_capital['Product'].isin(product_list))]
        df_capital_nonfnd['Position'] = df_capital_nonfnd['Long'] + df_capital_nonfnd['Short']
        dict_nonfnd = df_capital_nonfnd.set_index(['Product', 'Account'])['Position'].to_dict()

        msg_nonfnd = '\n\t\t'.join([f'{key_}: {dict_nonfnd[key_]};' for key_ in dict_nonfnd])
        msg_delay = f"\n\t未匹配到股票户但有持仓的产品[{time_flag}]:\n\t\t{msg_nonfnd}"
        msg_delay += f"\n\t期货更新时间异常情况[{time_flag}]:\n"
        time_flag = int(time_flag)

        future_data_except = False
        if (time_flag < 150000) and (time_flag > 83000) and (date == date_cur):
            for product, account, update_time in df_capital_delay[['Product', 'Account', 'UpdateTime']].values:
                msg_delay += f'\t\t{update_time.strftime("%H%M%S")}-{product},{account}\n'
            if not df_capital_delay.empty:
                future_data_except = True

        return msg_delay, future_data_except

    def check_product_repeat(self, series):
        series_num = series.value_counts()
        repeat_dict = series_num[series_num > 1].to_dict()
        assert not repeat_dict, f'产品名重复:{repeat_dict}, Warning, 请检查！'

    def check_instrument_direction(self, series_class, series_fut, series_prod):
        series_class = series_class.copy(deep=True)
        series_fut = series_fut.copy(deep=True)
        dc_series = series_class.fillna('').str.endswith('DC')
        long_exec = series_fut.fillna('').apply(lambda x: 'LONG' in x)
        short_exec = series_fut.fillna('').apply(lambda x: 'SHORT' in x)
        flag_long_in_dc = series_prod[long_exec & dc_series].to_list()
        flag_short_in_zq = series_prod[short_exec & (~ dc_series)].to_list()

        assert not flag_long_in_dc, f"DC 策略开 LONG 期货: {flag_long_in_dc}"
        assert not flag_short_in_zq, f"ZQ 策略开 SHORT 期货: {flag_short_in_zq}"

    def flag_product(self, obj, mini_product_list, paras_dict):
        if not isinstance(obj, Iterable):
            obj = [obj]

        flag_list = []
        for __obj in obj:
            _temp = ''
            if self.MulAcc_Main_dict.get(__obj, None) is not None:
                _temp += 'font-weight: bold; '
            elif __obj in mini_product_list:
                _temp += 'font-style: italic; color: #A52A2A; '

            if __obj in paras_dict.get('PrdType_Above80', []):
                _temp += 'background-color: yellow'
            elif __obj in paras_dict.get('PrdType_Below80', []):
                _temp += 'background-color: red'
            elif __obj in paras_dict.get('PrdType_Equity', []):
                _temp += 'background-color: #B9D3EE'
            elif __obj in paras_dict.get('PrdType_Hybrid', []):
                _temp += 'background-color: #E066FF'
            elif __obj in paras_dict.get('PrdType_FutDeri', []):
                _temp += 'background-color: #CD950C'
            
            flag_list.append(_temp)

        return flag_list

    def flag_cash_ratio(self, cr, thres):
        if not isinstance(cr, str):
            return True

        if cr in ['', '-', 'nan', 'nan/nan']:
            return True

        return float(cr.split('/')[0]) < thres

    def check_target_mv(self, obj, curdate, df_process_origin, df_origin, df_origin_stocks, oi_dict, paras_dict):
        margin_exec_min_ratio = paras_dict.get('margin_exec_min_ratio', 0.14)
        min_transfer_unit = paras_dict.get('min_transfer_unit', 5e4)

        df_process = df_process_origin.copy(deep=True)
        df_process['LngShtM'] = df_process['LngShtM'].fillna('0W').str[:-1].astype('int') * 1e4

        ls_amount = df_process['LngShtM']
        msg = (f'{curdate}-TargetMV:\n'
               f'\tLong 金额： {int(np.maximum(ls_amount, 0).sum() / 1e4)}W;\n'
               f'\tShort金额： {int(np.minimum(ls_amount, 0).sum() / 1e4)}W;\n'
               f'\t期货操作 ：  {obj.statistic_future_operation_num(df_process["ExecPos"].to_list(), is_str=True)}')

        df_process['ExecPos'] = df_process['ExecPos'].fillna('').apply(
            lambda x: obj.format_future_operation_list_2_str(x, self.future_list, reverse=True))

        df_stk_process = df_process.groupby(['Main', 'How'])['LngShtM'].sum().reset_index()
        df_ftr_process = df_process[df_process['Main'] == df_process['Product']][['Main', 'How', 'ExecPos', 'ExpectPos']]
        df_process_main = pd.merge(df_ftr_process, df_stk_process, on=['Main', 'How'], how='outer')
        df_process_main['ExpectPos'] = df_process_main['ExpectPos'].fillna('').apply(lambda x: list(obj.format_future_position_dict_2_list(x, self.future_list)))
        df_process_main['TransExpectPos'] = df_process_main['ExpectPos']

        df_ls_oi = df_process_main[df_process_main['How'] == 'OpenIndex'].set_index('Main')
        if not df_ls_oi.empty:
            df_ls_targetmv = df_process_main[df_process_main['How'] == 'TargetMV'].set_index('Main')
            df_ls_oi_fix = df_process_main[df_process_main['How'] == 'OpenIndexFix'].set_index('Main')
            tmv_prd_list = df_ls_targetmv.index.to_list()
            for main in df_ls_oi.index.to_list():
                if main not in tmv_prd_list: continue

                oi_exec_pos = df_ls_oi.loc[main, 'ExecPos']
                if str(oi_exec_pos) in ['nan', '']: oi_exec_pos = np.array([0, 0, 0, 0])

                targetmv_expect_pos = df_ls_targetmv.loc[main, 'TransExpectPos']
                if str(targetmv_expect_pos) in ['nan', '']: targetmv_expect_pos = np.array([0, 0, 0, 0])

                new_except_pos = np.array(oi_exec_pos) + np.array(targetmv_expect_pos)

                df_ls_oi.loc[main, 'TransExpectPos'] = str(list(new_except_pos))
                df_ls_targetmv.loc[main, 'TransExpectPos'] = str(list(new_except_pos))

            df_process_main = pd.concat([df_ls_oi, df_ls_oi_fix, df_ls_targetmv], axis=0).reset_index()
        df_process_main['TransExpectPos'] = df_process_main['TransExpectPos'].astype('str').apply(lambda x: np.array(eval(x)))

        infor_list = []
        for (prod_main, how_value), df_main in df_process_main.groupby(['Main', 'How']):
            df_ori_main = df_origin[df_origin['Main'] == prod_main]
            if df_ori_main.empty:
                print(prod_main)
                infor_list.append({'Main': prod_main, 'How': how_value, 'FtrExT0R': 0.01, 'TgtPosR': 1, })
                continue
            df_check = pd.merge(df_main, df_ori_main, on='Main', how='outer')
            
            df_check['ShortMain'] *= (df_check['How'] == 'TargetMV')
            df_check['ApplyMain'] *= (df_check['How'] == 'OpenIndex')
            df_check['ApplyFixMain'] *= (df_check['How'] == 'OpenIndexFix')
            
            df_check['ApplyMain'] += np.maximum(df_check['ApplyFixMain'], 0)
            
            target_ind_value = (df_check['ExpectPos'] * df_check['IdxValueArr'])
            target_ind_value = pd.Series(np.sum(np.stack(target_ind_value.values), axis=1), index=df_check.index)

            target_fut_value = (df_check['TransExpectPos'] * df_check['FtrValueArr']).sum().sum()
            exe_fut_value = (df_check['ExecPos'].fillna(0) * df_check['IdxValueArr']).sum().sum()
            abs_exe_fut_value = (np.abs(df_check['ExecPos'].fillna(0)) * df_check['IdxValueArr']).sum().sum()
            
            df_check['EstmNAV'] = df_check['StockMoney'] + df_check['Capital'] + df_check['HoldMV']
            target_nav = df_check['EstmNAV'].sum()
            target_holdmv = (df_check['HoldMV'] + df_check['LngShtM']).sum()
            if df_check['Class'].iloc[0].endswith('DC'):
                expm = (df_check['FtrIdxValue'] + df_check['HoldMV']).sum() #对冲产品暴露为股票多头市值-期货空头市值
                expm_target = (target_ind_value + target_holdmv).sum()
                expm_exec = (exe_fut_value + df_check['LngShtM']).sum()
            else:
                expm = (df_check['EstmNAV'] * df_check['CashTargetRatio'] * (1 - df_check['IsOpenIdx']) + df_check['FtrIdxValue'] - 
                        (df_check['StockMoney'] + df_check['Capital'] - df_check['ShortMain'] * 1e4)).sum()
                expm_target = (df_check['EstmNAV'] * df_check['CashTargetRatio'] * (1 - df_check['IsOpenIdx']) + target_ind_value -
                               (df_check['StockMoney'] + df_check['Capital'] - df_check['LngShtM'] + df_check['ApplyMain'] * 1e4)).sum()
                expm_exec = (exe_fut_value + df_check['LngShtM'] - df_check['ShortMain'] * 1e4 - df_check['ApplyMain'] * 1e4).sum()

            expm_r = expm / target_nav
            expm_target_r = expm_target / target_nav
            expm_exec_r = expm_exec / target_nav
            target_position_ratio = target_holdmv / target_nav
            fut_t0_ratio = abs_exe_fut_value / abs(exe_fut_value) if exe_fut_value != 0 else 1
            transfer = np.ceil((np.abs(target_fut_value * margin_exec_min_ratio) - df_check['Capital'].sum()) / min_transfer_unit) * min_transfer_unit
            infor_list.append({
                'Main': prod_main,
                'How': how_value,
                'Trans': transfer * (how_value != 'OpenIndexFix'),
                'ExpM': expm,
                'TgtExpM': expm_target,
                'ExeExpM': expm_exec,
                'ExpMR': expm_r,
                'ExeExpMR': expm_exec_r,
                'TgtExpMR': expm_target_r,
                'FtrExT0R': fut_t0_ratio / 100,

                'TgtMV': target_holdmv,
                'TgtPosR': target_position_ratio,
            })
        df_check_ori = pd.DataFrame(infor_list).fillna(0)
        swap_and_option_list = paras_dict['product_swap'] + paras_dict['product_option']
        infor_list_new = []
        for prod_main, df_main in df_check_ori.groupby('Main'):
            if prod_main in swap_and_option_list: df_main['Trans'] = 0
            else: 
                total_trans = df_main[df_main['How'] != 'OpenIndexFix']['Trans'].mean()
                df_main['Trans'] = max(total_trans, 0) * pd.Series(df_main.index == pd.Series(df_main.index).iloc[0], index=df_main.index)
            infor_list_new.append(df_main)
        
        df_check = pd.concat(infor_list_new, axis=0)
        df_check = obj.format_result_excel_columns(
            df_check, format_type='W', columns_list=['ExpM', 'TgtExpM', 'ExeExpM', 'TgtMV', 'Trans'])
        df_check = obj.format_result_excel_columns(
            df_check, format_type='%', columns_list=['ExpMR', 'TgtExpMR', 'ExeExpMR', 'FtrExT0R', 'TgtPosR'])
        
        df_process = pd.merge(df_process_origin[['Product', 'Main', 'How', 'LngShtM', 'ExecPos', 'ExpectPos']], df_origin_stocks[['Product', 'Main', 'Short', 'Apply', 'ApplyFix']], on=['Product', 'Main'], how='left')
        df_check = pd.merge(df_process, df_check, on=['Main', 'How'], how='outer')

        df_check['Apply'] = df_check['Main'].apply(lambda x: oi_dict.get('hedge_open_index', {}).get(x, 0))
        df_check['ApplyFix'] = df_check['Main'].apply(lambda x: oi_dict.get('hedge_open_index_fix', {}).get(x, 0))

        df_check['Class'] = df_check['Product'].apply(lambda x: production_2_strategy(x) + ('SWAP' if x in swap_and_option_list else ''))
        df_check['Trans'] = df_check['Trans'].replace('0W', '')
        df_check['FtrExT0R'] = df_check['FtrExT0R'].fillna(1)
        df_check['ExpMR'] = df_check['ExpMR'].fillna(0)
        df_check['TgtExpMR'] = df_check['TgtExpMR'].fillna(0)
        df_check['ExeExpMR'] = df_check['ExeExpMR'].fillna(0)
        
        df_check = df_check.fillna('').sort_values(['How', 'Class', 'Main'])
        return df_check, msg

    def check_stock_data_abnormal_data(self, curdate, down_json_mode):
        df_close = self.load_stocks_query_data(curdate)[['Product', 'HoldMV']]
        df_monitor_data = self.get_monitor_stocks_data(curdate, down_json_mode)
        df_monitor_data['RealTHoldMV'] = \
            df_monitor_data['MV_collateral'].astype('float') + df_monitor_data['MV_shortSell'].astype('float')
        df_monitor_mv = df_monitor_data.groupby('Product')[['RealTHoldMV']].mean().reset_index()

        df_compare_holdmv = pd.merge(df_close, df_monitor_mv, on='Product', how='outer')
        df_compare_holdmv['MVDiffR'] = np.round(
            (df_compare_holdmv['RealTHoldMV'] - df_compare_holdmv['HoldMV']) / df_compare_holdmv['HoldMV'] * 100, 2)
        df_compare_holdmv['absMVDiffR'] = np.abs(df_compare_holdmv['MVDiffR'])
        df_compare_holdmv = df_compare_holdmv[~df_compare_holdmv['Product'].isin(ProductionList_AlphaShort)]
        df_compare_holdmv = df_compare_holdmv.sort_values('absMVDiffR', ascending=False)
        df_compare_holdmv['Colo'] = df_compare_holdmv['Product'].apply(lambda x: production_2_colo(x))
        print(df_compare_holdmv[df_compare_holdmv['absMVDiffR'] > 1])

        df_monitor_data = df_monitor_data[df_monitor_data['Product'].isin(DUALCENTER_PRODUCTION)]
        df_monitor_data = pd.melt(
            df_monitor_data,
            id_vars=['Product', 'Exchange'],
            value_vars=['MV_normal', 'Quota_MV', 'MV_collateral', 'MV_shortSell', 'MV_net', 'Quota_DiffRatio'])
        df_monitor_data = pd.pivot_table(
            df_monitor_data, index=['Product', 'variable'], columns='Exchange', values='value').reset_index()
        df_monitor_data['DiffR'] = np.round((df_monitor_data['SH'] - df_monitor_data['SZ']) / (
                df_monitor_data['SH'] + df_monitor_data['SZ']) / 2 * 100, 2)
        df_monitor_data['absDiffR'] = np.abs(df_monitor_data['DiffR'])
        df_monitor_data['Flag'] = (df_monitor_data['variable'] == 'Quota_DiffRatio') * \
                                  (np.abs(df_monitor_data['SH']) < 0.1) * (np.abs(df_monitor_data['SZ']) < 0.1) + \
                                  (df_monitor_data['variable'] == 'MV_net') * \
                                  (np.abs(df_monitor_data['SH']) < 5e5) * (np.abs(df_monitor_data['SZ']) < 5e5)
        df_monitor_data['DiffR'] *= (1 - df_monitor_data['Flag'])
        df_monitor_data['absDiffR'] *= (1 - df_monitor_data['Flag'])
        df_monitor_data = df_monitor_data.sort_values('absDiffR', ascending=False).drop('Flag', axis=1)
        df_monitor_data['SH'] = np.round(df_monitor_data['SH'] / 1e4).astype('int').astype('str') + 'W'
        df_monitor_data['SZ'] = np.round(df_monitor_data['SZ'] / 1e4).astype('int').astype('str') + 'W'

        df_monitor_data.index.name = None
        df_monitor_data.columns.name = None
        print(df_monitor_data[df_monitor_data['absDiffR'] > 1])


class ProcessAdjust(ProcessTradingDependData):
    def __init__(self, curdate, **kwargs):

        self.run_mode = kwargs.get('run_mode', 'marketing')
        self.adjust_mode = kwargs.get('adjust_mode', 'index_steady')
        self.refresh_match_apply = kwargs.get('refresh_match_apply', False)
        self.check_future_status = kwargs.get('check_future_status', True)
        self.check_future_num = kwargs.get('check_future_num', True)
        self.short_deduct = kwargs.get('short_deduct', True)
        self.temp_cashout_deduct_dict = kwargs.get('temp_cashout_deduct_dict', {})
        self.new_prdt_info = kwargs.get('new_prdt_info', {}).get(curdate, [])
        self.config_path = kwargs.get('config_path', None)

        self.fut_pos_mode = 'pre-close' if self.run_mode.startswith('marketing') else 'close'

        adj_profile = AdjustProfile(config_path=self.config_path)
        self.run_paras_dict = adj_profile.get_run_mode_paras(run_mode=self.run_mode)
        self.run_paras_dict.update({'temp_cashout_deduct_dict': self.temp_cashout_deduct_dict})

        self.adj_paras_dict = adj_profile.get_adj_mode_all_paras(adjust_mode=self.adjust_mode)
        self.adj_paras_dict['min_hs_ratio_dict'] = adj_profile.get_hs_ratio_thres_paras()
        self.adj_paras_dict['margin_fix_dict'] = self.get_future_margin_fix_data(curdate)

        self.deducted_trans_drop_colo_list = adj_profile.adjust_trans_deducted.get(curdate, {}).get('colo', [])
        self.deducted_trans_drop_product_list = adj_profile.adjust_trans_deducted.get(curdate, {}).get('prd', [])

        self.cash_ratio_dict = adj_profile.get_target_cash_ratio()

        self.output_dir = adj_profile.get_output_dir(curdate)
        self.config_dir = adj_profile.get_config_dir()
        self.config_dir_prss = adj_profile.get_config_dir_prss()
        self.process_dir = adj_profile.get_process_dir()
        self.trans_data_dir = adj_profile.get_trans_data_dir()
        self.downstream_dir = adj_profile.get_downstream_dir()

        self.process_check = ProcessCheck()

        df_accsumm = get_production_list_trading(curdate, ret_df_data=True)
        self.product_2_bar_mode = {prod: bar_mode for prod, bar_mode in zip(df_accsumm['Account'], df_accsumm['bar'])}
        self.trading_alpha_product_list = df_accsumm['Account'].to_list()
        self.trading_alpha_product_list += Production_SpecialList_Delete
        super().__init__(
            curdate=curdate,
            paras_dict=self.run_paras_dict
        )
        self.down_json_mode = kwargs.get('down_json_mode', self.run_paras_dict.get('down_json_mode', False))
        self.deducted_trans = kwargs.get('deducted_trans', self.run_paras_dict.get('deducted_trans', False))
        self.down_trans_json = kwargs.get('down_trans_json', self.run_paras_dict.get('down_trans_json', False))

        self.run_price_mode = self.run_paras_dict.get('run_price_mode', 'realtime')
        self.adjust_price_ratio = self.run_paras_dict.get('adjust_price_ratio', False)
        self.adjust_position_ratio = self.run_paras_dict.get('adjust_position_ratio', False)
        self.run_position_mode = self.run_paras_dict.get('run_position_mode', 'realtime')
        self.temp_cashout_deduct_dict = self.run_paras_dict.get('temp_cashout_deduct_dict', {})
        self.temp_drop_product_list = self.run_paras_dict.get('temp_drop_product_list', [])

        self.path_total_res = f'{self.output_dir}{self.curdate}_Total_Process.xlsx'

        self.adjust_expose_mode_list = ['adj_exp-HoldMV', 'adj_exp-RealTHoldMV', 'adj_exp-QuotaMV']
        self.priority_move_list = ['Product', 'Main', 'Class', 'Cls', 'HoldMVR', 'EstmNVR', 'NetValue', 'NetValueDate', 'BankCash', 'BankCashRatio']
        
        self.dict_netvalue_data = self.get_netvalue_dict()
        (self.cash_out_plan_dict, self.cur_short_dict, self.exchange_dict, self.apply_dict, self.apply_next_dict,
         self.exchange_apply_dict, self.rebuy_money_dict, self.df_cash_out_plan, self.predate_repay_list) = self.load_adjust_money_data()

        (apply_repay_history, self.apply_zq_history_list, self.repay_history, self.apply_history, 
         self.apply_simple_dict, self.apply_matched_dict, self.flag_match_apply) = self.load_apply_repay_history_data(curdate)

        self.dict_price_data, self.dict_value_data, self.instrument_list = self.load_price_data()
        self.df_total_data, self.df_stocks, self.df_hedge_cfgout = self.load_all_trading_data(short_deduct=self.short_deduct)

        self.apply_repay_history = self.check_netvalue(apply_repay_history)

    def calculate_new_product_future_operation(self, product, money, cash_ratio=0.06, new_product=None):
        cash = money / (1 + cash_ratio) * 10000
        proportion = production_2_proportion(product)
        print(product, proportion)
        proportion = np.array([proportion.get(index_name, 0) for index_name in self.index_list])
        index_price_array = np.array([self.dict_value_data[index] for index in self.index_list])

        adjust_fut = np.round(cash * proportion / index_price_array).astype('int')
        stock_value = int(round(cash / 10000))
        future_value = - int(round(np.sum(adjust_fut * index_price_array) / 10000))
        adjust_fut_format = []
        for ifut, fut_adj in enumerate(adjust_fut):
            if fut_adj > 0:
                adjust_fut_format.append(f'{self.future_list[ifut]}:+SHORT {int(abs(fut_adj))}')
            elif fut_adj < 0:
                adjust_fut_format.append(f'{self.future_list[ifut]}:-SHORT {int(abs(fut_adj))}')
            else:
                pass

        if new_product is None: new_product = product
        print(f'{new_product}-多头市值： {stock_value}W')
        print(f'{new_product}-空头市值： {future_value}W')
        print(f'{new_product}-空头仓位： {";".join(adjust_fut_format)}\n')
    
    def match_apply_on_route(self, apply_his_df):
        apply_his_df = apply_his_df.copy(deep=True)
        apply_simple_dict = {
            prd: "/".join([str(num) for num in df.set_index('Product').dropna(axis=1).astype('int').values[0]])
            for prd, df in apply_his_df.replace(0, np.nan).groupby('Product')}
        
        flag_match_apply = os.path.exists(self.path_total_res)
        if flag_match_apply:
            cur_match_flag = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/Flag/{self.curdate}_match_flag.txt'
            cur_match_path = f'{self.config_dir_prss}apply_match.csv'
            if (not os.path.exists(cur_match_flag)) or self.refresh_match_apply:
                try: df_res = pd.read_excel(self.path_total_res, sheet_name='zq')
                except: return apply_simple_dict, {}, False
                
                df_res = df_res[df_res['Main'] == df_res['Product']]
                df_res = df_res[['Main', 'ExpM']].set_index('Main')
                df_res['ExpM'] = df_res['ExpM'].str[:-1].astype('int')
                
                apply_his_df = apply_his_df.set_index('Product')
                columns_list = apply_his_df.columns.to_list()
                merge_his_df = np.cumsum(apply_his_df[columns_list[::-1]], axis=1).join(df_res, how='inner').fillna(0)
                diff_his_df = pd.DataFrame(np.array(merge_his_df) - np.tile(merge_his_df[['ExpM']], merge_his_df.shape[1]), index=merge_his_df.index, columns=merge_his_df.columns).abs().drop('ExpM', axis=1)
                flag_min = diff_his_df == pd.DataFrame(np.tile(diff_his_df.min(axis=1).to_frame(), diff_his_df.shape[1]), index=diff_his_df.index, columns=diff_his_df.columns)
                apply_matched_dict = np.maximum(merge_his_df[flag_min].fillna(method='bfill', axis=1)[columns_list[-1]], 0).astype('int').to_dict()

                df_matched = pd.DataFrame.from_dict(apply_matched_dict, orient='index').reset_index().rename({'index': 'Product', 0: 'Amount'}, axis='columns')
                df_matched.to_csv(cur_match_path, index=False)

                with open(cur_match_flag, 'w') as jf: jf.write('')
            else:
                df_matched = pd.read_csv(cur_match_path)
                apply_matched_dict = {prd: amount for prd, amount in zip(df_matched['Product'], df_matched['Amount'])}
        else:
            apply_matched_dict = {}

        return apply_simple_dict, apply_matched_dict, flag_match_apply

    def load_apply_repay_history_data(self, curdate):
        apply_zq_history = self.get_apply_amount_data_history(curdate)
        apply_dc_history = self.get_apply_amount_data_dc_history(curdate, 5)

        apply_zq_history_list = apply_zq_history.index.to_list()

        apply_history = pd.concat([apply_zq_history, apply_dc_history], axis=0)
        repay_history = self.get_repay_amount_data_history(curdate)

        union_index = apply_history.index.union(repay_history.index)

        apply_repay_history = apply_history.reindex(union_index, fill_value=0) + repay_history.reindex(union_index, fill_value=0)
        apply_repay_history = apply_repay_history.reset_index().rename({'index': 'Product'}, axis='columns')
        apply_history = apply_history.reset_index().rename({'index': 'Product'}, axis='columns')

        apply_history_check = apply_history.copy(deep=True)

        if curdate in apply_history_check.columns.to_list(): apply_history_check = apply_history_check.drop(curdate, axis=1)

        apply_simple_dict, apply_matched_dict, flag_match_apply = self.match_apply_on_route(apply_history_check)

        return apply_repay_history, apply_zq_history_list, repay_history, apply_history, apply_simple_dict, apply_matched_dict, flag_match_apply

    def load_adjust_money_data(self):
        if self.run_price_mode == 'close':
            cashout_date, rebuy_date = self.nextdate, self.curdate
        else:
            cashout_date, rebuy_date = self.curdate, self.predate

        cash_out_dict, cur_short_dict, exchange_dict, apply_dict, exchange_apply_dict, rebuy_money_dict, df_cash_out_plan, pre_date_repay = (
            self.load_cash_flow_data(cashout_date, rebuy_date))
        apply_next_dict = self.get_apply_amount_data(self.nextdate).get('hedge_open_index', {})

        for apply_key in apply_dict:
            error_product = list(set(list(apply_dict[apply_key].keys())) - set(PRODUCTION_REPORT_LIST))
            assert not error_product, f'{apply_key} 类型申购 产品名填写错误:{error_product}！'
        error_product = list(set(list(cur_short_dict.keys())) - set(self.trading_alpha_product_list))
        assert not error_product, f'赎回 产品名填写错误:{error_product}！'

        with open(f'{self.output_dir}{self.curdate}_cash_deduct.json', 'w') as jf:
            json.dump(df_cash_out_plan[['Product', 'Date', '当日赎回', '赎回当日付']].rename(
                {'当日赎回': 'Short', '赎回当日付': 'CashOut'}, axis='columns').to_dict(orient='records'), jf, indent=2)
        tsl = TransferServerLocal(ip='120.24.92.78', port=22, username='monitor', password='ExcellenceCenter0507')
        tsl.upload_file(
            local_path=f'{self.output_dir}{self.curdate}_cash_deduct.json',
            server_path='/home/monitor/MonitorCode/HelloWorld/static/file/CashDeduct.json')

        return cash_out_dict, cur_short_dict, exchange_dict, apply_dict, apply_next_dict, exchange_apply_dict, rebuy_money_dict, df_cash_out_plan, pre_date_repay

    def load_all_trading_data(self, short_deduct=True):
        load_start_time = time.time()
        df_hedge_capital, df_hedge_position = self.load_future_data(
            self.curdate, self.down_json_mode, instrument_month=self.instrument_list[-2], pos_mode=self.fut_pos_mode)

        if self.run_position_mode == 'realtime':
            df_stocks = self.load_stocks_data(
                self.curdate, self.down_json_mode,
                deducted_trans=self.deducted_trans,
                down_trans_json=self.down_trans_json,
                deducted_trans_drop_colo_list=self.deducted_trans_drop_colo_list,
                deducted_trans_drop_product_list=self.deducted_trans_drop_product_list)
            if self.new_prdt_info:
                df_new_stocks = pd.DataFrame(self.new_prdt_info)
                df_new_stocks['StockRealTime'] = df_new_stocks['StockMoney']
                df_new_stocks = df_new_stocks.reindex(columns=df_stocks.columns, fill_value=0)
                df_stocks = pd.concat([df_stocks, df_new_stocks], axis=0)
        elif self.run_position_mode == 'query':
            df_stocks = self.load_stocks_query_data(self.curdate)
        elif self.run_position_mode == 'pre-query':
            df_stocks = self.load_stocks_query_data(self.predate)
        else:
            raise 'ValueError'

        df_hedge = self.formatting_future_data(df_hedge_capital, df_hedge_position)
        df_stocks = self.formatting_stocks_data(df_stocks, short_deduct=short_deduct)
        df_total_data, df_stocks, df_hedge_cfgout = self.formatting_all_data(df_hedge, df_stocks)

        msg_delay, future_data_except = self.process_check.check_future_data(
            self.curdate, df_hedge_capital, df_stocks['Product'].to_list())
        
        flag = datetime.datetime.now().strftime('%H%M%S')
        msg = f"""{self.curdate}-{flag}:\n\t当日补开股指：{self.apply_dict.get('hedge_open_index_fix', {})}"""
        msg += f"""\n\t当日开股指：{self.apply_dict.get('hedge_open_index', {})}"""
        msg += f"""\n\t次日开股指：{self.apply_next_dict}"""
        msg += f"""\n\t当日赎回：{self.cur_short_dict}"""
        msg += f"""\n\t赎回换仓在途：{self.exchange_dict}"""
        msg += f"""\n\t当日出金：{self.cash_out_plan_dict}\n{msg_delay}"""
        if 90000 < int(flag) < 150000:
            wechat_bot_msg_check(msg)
        else:
            print(msg)

        if self.check_future_status: assert (not future_data_except), '期货数据更新有断掉产品, 请查看！'
        if self.check_future_num and (int(flag) < 160000): self.process_check.statistics_future_position_nums(self.curdate, df_total_data, self.output_dir)

        load_end_time = time.time()
        print(f'\n{self.curdate}-{flag}-加载数据: {round(load_end_time - load_start_time, 2)}s')

        return df_total_data, df_stocks, df_hedge_cfgout

    def formatting_stocks_data(self, df_stock_data, short_deduct=True):
        df_stock_data['CashOut'] = df_stock_data['Product'].apply(lambda x: self.cash_out_plan_dict.get(x, 0)).astype('int')
        df_stock_data['Short'] = df_stock_data['Product'].apply(lambda x: self.cur_short_dict.get(x, 0)).astype('int')
        df_stock_data['IsExchange'] = df_stock_data['Product'].apply(lambda x: self.exchange_dict.get(x, 0)).astype('int')

        df_stock_data['ApplyFix'] = df_stock_data['Product'].apply(
            lambda x: self.apply_dict.get('hedge_open_index_fix', {}).get(x, 0))
        df_stock_data['Apply'] = df_stock_data['Product'].apply(
            lambda x: self.apply_dict.get('hedge_open_index', {}).get(x, 0))
        df_stock_data['NextApply'] = df_stock_data['Product'].apply(lambda x: self.apply_next_dict.get(x, 0))
        df_stock_data['RebuyMoney'] = df_stock_data['Product'].apply(lambda x: self.rebuy_money_dict.get(x, 0))

        df_stock_data['StockOutAvail'] = np.maximum(
            df_stock_data['StockMoney'] + df_stock_data['RebuyMoney'], 0) + df_stock_data['CashOut'] * 1e4
        df_stock_data['StockMoney'] += df_stock_data['CashOut'] * 1e4 + df_stock_data['Short'] * 1e4 * short_deduct
        df_stock_data['StockRealTime'] += df_stock_data['CashOut'] * 1e4 + df_stock_data['Short'] * 1e4 * short_deduct
        df_stock_data['BarMode'] = df_stock_data['Product'].apply(lambda x: self.product_2_bar_mode.get(x, 8))

        df_stock_data['Main'] = df_stock_data['Product'].apply(lambda x: self.get_product_trading_main_name(x))

        df_stock_data = df_stock_data[
            ['Product', 'Main', 'HoldMV', 'QuotaMV', 'RealTHoldMV',
             'StockMoney', 'StockRealTime', 'StockOutAvail', 'CashOut', 'Short', 'Apply', 'ApplyFix', 'NextApply',
             'IsExchange', 'BarMode']]
        return df_stock_data

    def formatting_future_data(self, df_hedge_capital, df_hedge_position):
        df_hedge_capital = df_hedge_capital.copy(deep=True).groupby('Product')[
            ['Capital', 'Margin', 'WithdrawQuota']].sum().reset_index()
        df_hedge = pd.merge(df_hedge_capital, df_hedge_position, on='Product', how='outer')

        df_hedge['Capital'] = df_hedge['Capital'].fillna(0)
        df_hedge['Margin'] = df_hedge['Margin'].fillna(0)
        df_hedge['WithdrawQuota'] = df_hedge['WithdrawQuota'].fillna(0)
        df_hedge['Main'] = df_hedge['Product'].apply(lambda x: self.get_product_trading_main_name(x))

        return df_hedge

    def formatting_all_data(self, df_hedge, df_stocks):
        drop_product_list = (
                    self.adj_paras_dict['product_drop'] + self.adj_paras_dict['product_ls'] + self.adj_paras_dict[
                'product_clear'])
        df_stocks = df_stocks.copy(deep=True)
        df_hedge = df_hedge.copy(deep=True)
        df_stocks = df_stocks[(~ df_stocks['Product'].isin(drop_product_list))]
        df_hedge = df_hedge[(~ df_hedge['Product'].isin(drop_product_list))]

        df_stocks['HoldMV'] = df_stocks['HoldMV'] * (1 + self.adjust_position_ratio)
        df_stocks['QuotaMV'] = df_stocks['QuotaMV'] * (1 + self.adjust_position_ratio)
        df_stocks['RealTHoldMV'] = df_stocks['RealTHoldMV'] * (1 + self.adjust_position_ratio)

        df_stocks['Class'] = df_stocks['Product'].apply(lambda x: production_2_index(x) + production_2_strategy(x))
        df_stocks['Colo'] = df_stocks['Product'].apply(lambda x: production_2_colo(x))
        
        df_stocks['HST'] = np.maximum(
            df_stocks['Colo'].apply(lambda x: np.sum([x.startswith(colo_) for colo_ in SZ_SH_Transfer]) > 0).astype(
                'int'), 1 - df_stocks['Product'].isin(list(set(DUALCENTER_PRODUCTION) - set(self.adj_paras_dict['product_swap']))).astype('int'))

        df_stocks['HSRatio'] = df_stocks['Class'].apply(lambda x: self.adj_paras_dict['min_hs_ratio_dict'].get(x, 0.5)) * 2
        df_stocks['StockMLong'] = \
            df_stocks['StockMoney'] * (df_stocks['HSRatio'] * (1 - df_stocks['HST']) + df_stocks['HST'])

        adj_mv_series = np.where(df_stocks['HoldMV'] != 0, df_stocks['HoldMV'], np.abs(df_stocks['StockMoney']))
        df_stocks['StkCshR'] = (df_stocks['StockMoney'] / adj_mv_series).replace([np.inf, -np.inf], 1).fillna(1) * 100
        df_stocks['CashTargetRatio'] = df_stocks['Product'].apply(
            lambda x: self.cash_ratio_dict.get(x, self.adj_paras_dict.get('cash_ratio', 0.05)) +
                      self.adj_paras_dict.get('cash_extra_ratio', 0))

        df_multi_sum = df_stocks.groupby(['Main', 'Class']).agg({
            'HoldMV': 'sum', 'QuotaMV': 'sum', 'RealTHoldMV': 'sum', 'StockMoney': 'sum', 'StockRealTime': 'sum',
            'StockOutAvail': 'sum', 'StockMLong': 'sum', 'CashOut': 'sum', 'CashTargetRatio': 'max',
            'Short': 'sum', 'Apply': 'sum', 'ApplyFix': 'sum', 'NextApply': 'sum', 'IsExchange': 'sum'}).rename(
            {'Short': 'ShortMain', 'Apply': 'ApplyMain', 'ApplyFix': 'ApplyFixMain', 'NextApply': 'NApplyMain'},
            axis='columns').reset_index()

        df_hedge_except = df_hedge[
            (~df_hedge['Main'].isin(df_multi_sum['Main'].to_list())) & (~df_hedge['HedgePosDict'].isna())]
        if not df_hedge_except.empty: print('没有实盘但有期货仓位的产品：', df_hedge_except)

        df_total_data = pd.merge(df_multi_sum, df_hedge, on='Main', how='left').fillna(0)

        df_hedge_cfgout = df_hedge[~ df_hedge['Main'].isin(df_total_data['Main'].to_list())].copy(deep=True)

        df_total_data['Product'] = df_total_data['Main']
        df_total_data['IsOpenIdx'] = (~ (df_total_data['Class'].isin(self.adj_paras_dict['class_non_open_index']) |
                                      df_total_data['Product'].isin(self.adj_paras_dict['product_non_open_index']))).astype('int')

        df_total_data['WithdrawQuota'] = df_total_data['WithdrawQuota']
        df_total_data['StkCshR'] = np.round(df_total_data['StockMoney'] / df_total_data['HoldMV'] * 100, 2)
        df_total_data['HedgePosDict'] = df_total_data['HedgePosDict'].apply(lambda x: x if isinstance(x, dict) else {})
        df_total_data['HedgePos'] = df_total_data['HedgePosDict'].apply(
            lambda x: self.format_future_pos_2_future_name(x))
        df_total_data['HedgePos'] = df_total_data['HedgePos'].apply(
            lambda x: np.array([np.round(x.get(fut_nm, 0), 2) for fut_nm in self.future_list]))
        df_total_data['FtrValue'] = np.abs(df_total_data['HedgePosDict'].apply(
            lambda x: np.sum([x[fut_nm] * self.dict_value_data.get(fut_nm, np.nan) for fut_nm in x])))
        df_total_data[f'FtrVLSMx'] = df_total_data['HedgePosDict'].apply(
            lambda x: np.array([x[fut_nm] * self.dict_value_data.get(fut_nm, np.nan) for fut_nm in x]))
        df_total_data[f'FtrVLSMx'] = df_total_data[f'FtrVLSMx'].apply(
            lambda x: max(np.abs(np.sum(x * (x > 0))), np.abs(np.sum(x * (x <= 0)))))

        product_num = len(df_total_data)
        df_total_data[f'IdxValueArr'] = \
            [np.array([self.dict_value_data[ind_nm] for ind_nm in self.index_list])] * product_num
        df_total_data[f'FtrValueArr'] = \
            [np.array([self.dict_value_data[fut_nm] for fut_nm in self.future_list])] * product_num

        df_total_data['Proportion'] = df_total_data['Product'].apply(lambda x: production_2_proportion(x))
        df_total_data['Proportion'] = df_total_data.apply(
            lambda row: self.adj_paras_dict['class_proportion_fix'].get(row['Class'], row['Proportion']), axis=1)
        df_total_data['Proportion'] = df_total_data['Proportion'].apply(
            lambda x: np.array([x.get(ind_nm, 0) for ind_nm in self.index_list]))

        df_total_data['Idx2FtrR'] = df_total_data['Proportion'] / df_total_data['IdxValueArr'] * df_total_data[
            'FtrValueArr']
        df_total_data['Idx2FtrR'] = pd.Series(np.sum(np.stack(df_total_data['Idx2FtrR'].values), axis=1),
                                              index=df_total_data.index)
        df_total_data['Ftr2IdxR'] = df_total_data['Proportion'] / df_total_data['FtrValueArr'] * df_total_data[
            'IdxValueArr']
        df_total_data['Ftr2IdxR'] = pd.Series(np.sum(np.stack(df_total_data['Ftr2IdxR'].values), axis=1),
                                              index=df_total_data.index)

        df_total_data['FtrIdxValue'] = df_total_data['HedgePos'] * df_total_data['IdxValueArr']
        df_total_data['FtrIdxValue'] = pd.Series(np.sum(np.stack(df_total_data['FtrIdxValue'].values), axis=1),
                                                 index=df_total_data.index)

        df_total_data['Capital'] = df_total_data['Capital'] - df_total_data['FtrValue'] * self.adjust_position_ratio
        df_total_data['EstmNAV'] = df_total_data['StockMoney'].fillna(0) + df_total_data['Capital'].fillna(0) + df_total_data['HoldMV'].fillna(0)
        df_total_data['EstmNAVOpn'] = df_total_data['EstmNAV'] - df_total_data['ShortMain'].fillna(0) * 1e4
        
        df_total_data['InvMVR'] = np.round(df_total_data['HoldMV'].fillna(0) / (df_total_data['HoldMV'].fillna(0) + df_total_data['Capital'].fillna(0)) * 100, 1)
        df_total_data['InvMVR'] = np.where((df_total_data['InvMVR'] < 0) | (df_total_data['InvMVR'] > 200), 100, df_total_data['InvMVR'])

        df_total_data['HoldMVR'] = np.round(df_total_data['HoldMV'] / df_total_data['EstmNAV'] * 100, 1)
        df_total_data['HoldMVR'] = np.where((df_total_data['HoldMVR'] < 0) | (df_total_data['HoldMVR'] > 200), 100, df_total_data['HoldMVR'])

        df_total_data['BankCash'] = df_total_data['Product'].apply(
            lambda x: self.dict_netvalue_data.get(Dict_ProductionName_Replace.get(x, x), {}).get('银行存款', 0)).fillna(0)
        df_total_data['NetValue'] = df_total_data['Product'].apply(
            lambda x: self.dict_netvalue_data.get(Dict_ProductionName_Replace.get(x, x), {}).get('资产净值', np.nan))
        df_total_data['NetValue'] = np.where(df_total_data['NetValue'].isna(), df_total_data['EstmNAV'], df_total_data['NetValue'])
        df_total_data['NetValueDate'] = df_total_data['Product'].apply(
            lambda x: self.dict_netvalue_data.get(Dict_ProductionName_Replace.get(x, x), {}).get('日期',
                                                                                                 np.nan)).fillna('-')
        df_total_data['BankCashRatio'] = np.round(df_total_data['BankCash'] / df_total_data['EstmNAV'] * 100, 2)
        df_total_data['Cls'] = np.where(df_total_data['Class'].str.endswith('DC'), 1, 0) + df_total_data[
            'Product'].isin(self.adj_paras_dict['product_swap'] + self.adj_paras_dict['product_option'])
        df_total_data['EstmNVR'] = np.round(
            ((df_total_data['EstmNAV'] - df_total_data['ShortMain'] * 10000) / df_total_data['NetValue'] * 100).fillna(100), 1)
        df_total_data['CptlRatio'] = np.round(df_total_data['Capital'] / df_total_data['EstmNAV'] * 100, 2)
        df_total_data['ApplyHL'] = df_total_data['Product'].apply(lambda x: self.apply_simple_dict.get(x, ''))
        df_total_data['AlyOnRt'] = df_total_data['Product'].apply(lambda x: self.apply_matched_dict.get(x, 0))
        
        df_total_data = df_total_data.sort_values(['Cls', 'Product'], ascending=True)

        df_total_data = df_total_data[self.priority_move_list + [col for col in df_total_data.columns.to_list() if
                                                                 col not in self.priority_move_list]]

        return df_total_data, df_stocks, df_hedge_cfgout

    def process_report(self):
        df_total_data, df_stocks = self.df_total_data, self.df_stocks
        adj_dc = AdjustPositionHedge(self.curdate, self.run_paras_dict)
        adj_zq = AdjustPosition(self.curdate, self.run_paras_dict)
        adj_swap = AdjustPositionSwap(self.curdate, self.run_paras_dict)

        df_swap_style, df_swap = adj_swap.adjust(df_total_data, df_stocks, 'report', self.adj_paras_dict)
        df_zq_style, df_zq = adj_zq.adjust(df_total_data, df_stocks, 'report', self.adj_paras_dict, self.flag_match_apply)
        df_dc_style, df_dc = adj_dc.adjust(df_total_data, df_stocks, 'report', self.adj_paras_dict)

        self.process_report_swap(df_swap)

        df_swap = df_swap[df_swap['Product'] == df_swap['Main']].reindex(columns=self.adj_paras_dict['report_dc_col'], fill_value=0)
        df_dc = df_dc[df_dc['Product'] == df_dc['Main']].reindex(columns=self.adj_paras_dict['report_dc_col'], fill_value=0)
        df_zq = df_zq[df_zq['Product'] == df_zq['Main']].reindex(columns=self.adj_paras_dict['report_zq_col'], fill_value=0)

        repay_dict_list = self.repay_history.T.to_dict(orient='list')
        repay_date_list = self.repay_history.columns.to_list()
        repay_date_index_dict = {d_: i_ for i_, d_ in enumerate(repay_date_list)}

        df_swap['累计赎回'] = df_swap.apply(
            lambda row: sum(repay_dict_list.get(row['Product'], [])[(repay_date_index_dict.get(str(row['NetValueDate']), len(repay_date_list)) + 1):]), axis=1)
        df_dc['累计赎回'] = df_dc.apply(
            lambda row: sum(repay_dict_list.get(row['Product'], [])[(repay_date_index_dict.get(str(row['NetValueDate']), len(repay_date_list)) + 1):]), axis=1)
        df_zq['累计赎回'] = df_zq.apply(
            lambda row: sum(repay_dict_list.get(row['Product'], [])[(repay_date_index_dict.get(str(row['NetValueDate']), len(repay_date_list)) + 1):]), axis=1)
        df_swap['净资产校对比例'] = np.round(
            1 / (df_swap['NetValue'].replace('', '0W').str[:-1].astype('float') + df_swap['累计赎回']) *
            df_swap['EstmNAV'].replace('', '0W').str[:-1].astype('float') * 100, 2).replace([np.inf, -np.inf], 0)
        df_dc['净资产校对比例'] = np.round(
            1 / (df_dc['NetValue'].replace('', '0W').str[:-1].astype('float') + df_dc['累计赎回']) *
            df_dc['EstmNAV'].replace('', '0W').str[:-1].astype('float') * 100, 2).replace([np.inf, -np.inf], 0)
        df_zq['净资产校对比例'] = np.round(
            1 / (df_zq['NetValue'].replace('', '0W').str[:-1].astype('float') + df_zq['累计赎回']) *
            df_zq['EstmNAV'].replace('', '0W').str[:-1].astype('float') * 100, 2).replace([np.inf, -np.inf], 0)

        check_ratio_list = sorted(df_dc['净资产校对比例'].to_list() + df_zq['净资产校对比例'].to_list())
        df_swap['累计赎回'] = (df_swap['累计赎回'].astype('str') + 'W').replace('0W', '')
        df_dc['累计赎回'] = (df_dc['累计赎回'].astype('str') + 'W').replace('0W', '')
        df_zq['累计赎回'] = (df_zq['累计赎回'].astype('str') + 'W').replace('0W', '')

        df_dc.to_csv(f'{self.output_dir}{self.curdate}_Expose_Calculate_Mix.csv', index=False, encoding='GBK')
        df_swap.to_csv(f'{self.output_dir}{self.curdate}_Expose_Calculate_Swap.csv', index=False, encoding='GBK')
        df_zq.to_csv(f'{self.output_dir}{self.curdate}_Expose_Calculate.csv', index=False, encoding='GBK')

        return df_dc, df_zq, df_swap

    def process_report_swap(self, df_swap):
        df_swap = df_swap.reindex(columns=self.adj_paras_dict['report_swap_col'], fill_value=0)
        
        caption_html = f"""
        <br><h2 style="margin: 0;">{self.curdate} SWAP PRODUCT MARGIN CHECK</h2><br>
        """
        df_swap = df_swap.astype('str').style.set_properties(
            **{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]).set_table_styles([
            {'selector': 'th', 'props': [('border', '1px solid black')]},
            {'selector': 'td', 'props': [('border', '1px solid black')]}]).apply(
            lambda x: ['background-color: red' if abs(float(v)) > 2 else '' for v in x],
              axis=0, subset=['MExpMR', 'ExpMR']).set_caption(caption_html)
        
        excel_path = f'{self.output_dir}check_swap_{self.curdate}.xlsx'
        img_path = f'{self.output_dir}check_swap_{self.curdate}.png'
        df_swap.to_excel(excel_path, index=False)
        try:
            dfi.export(df_swap, filename=img_path, dpi=300, fontsize=10, max_cols=-1, max_rows=-1, table_conversion='chrome')
            wechat_bot_image(img_path, type_api='report')
        except:
            wechat_bot_file(excel_path, type_api='report')

    def process_hedge_open_index(self, trans_type='hedge_open_index', trans_mode=True, to_downstream=True, oi_tmv_combine=False):
        open_index = 'OpenIndex' if trans_type == 'hedge_open_index' else 'OpenIndexFix'
        df_total_data = self.df_total_data.copy(deep=True)

        if self.apply_dict.get(trans_type, {}):
            adj_pos = AdjustPosition(self.curdate, self.run_paras_dict)
            df_res, df_io_process = adj_pos.adjust_open_index(df_total_data, self.adj_paras_dict, open_index,
                                                              self.apply_dict.get(trans_type, {}), self.df_hedge_cfgout)
            df_res.to_excel(self.output_dir + f'{self.curdate}_{trans_type}.xlsx', index=False)

            try:
                # dfi.export(df_res, filename=self.output_dir + f'{self.curdate}_{trans_type}.png', dpi=300, fontsize=10, max_cols=-1, max_rows=-1, table_conversion='selenium')
                dfi.export(df_res, filename=self.output_dir + f'{self.curdate}_{trans_type}.png', dpi=300, fontsize=10,
                           max_cols=-1, max_rows=-1, table_conversion='chrome')
                wechat_bot_image(self.output_dir + f'{self.curdate}_{trans_type}.png', type_api='capital-config')
            except:
                wechat_bot_file(self.output_dir + f'{self.curdate}_{trans_type}.xlsx', type_api='check')

            assert not df_io_process.empty, '开股指返回文件为空！'
            
            df_io_process = df_io_process[~df_io_process['Product'].isin(self.adj_paras_dict['product_apply_non_open_index'])]
            if not df_io_process.empty:
                self.process_config_wr(mode='oi', df_process=df_io_process)

                if to_downstream: self.process_to_downstream(update_pkl=False, new_mode=True, oi_tmv_combine=oi_tmv_combine)

                df_res = df_res.rename({'Trans': 'Transfer'}, axis='columns')
                df_res['Transfer'] = df_res['Transfer'].fillna('0W').replace('', '0W').str[:-1].astype('int') * 10000
                df_res = df_res.groupby('Main')[['Transfer']].sum().reset_index()
                df_res['Product'] = df_res['Main']
                df_res = df_res[df_res['Transfer'] > 0][['Product', 'Main', 'Transfer']]
                df_res.to_csv(f'{self.trans_data_dir}{self.curdate}_{trans_type}_origin.csv', index=False)

                if trans_mode: self.process_transfer(trans_type)
        else:
            print(f'当日没有{trans_type} 类型开股指开股指！')

    def process_transfer(self, trans_type):
        df_transfer = pd.read_csv(f'{self.trans_data_dir}{self.curdate}_{trans_type}_origin.csv')
        if not df_transfer.empty:
            print(df_transfer)
            try:
                gtc = GenerateTransferConfig(self.curdate, self.run_paras_dict, self.adj_paras_dict)
                gtc.run(
                    curdate=self.curdate,
                    down_json_mode=False,
                    wechat_and_email=True,
                    trans_type=trans_type,
                    paras_dict=self.adj_paras_dict,
                    open_index_df=df_transfer
                )
            except:
                msg = traceback.format_exc()
                print(msg)
                wechat_bot_msg_check(msg)

    def process_config_wr(self, mode='r', df_process=None, start_bar=None, start_bar_5m=None):
        if mode == 'r':
            file_path_process = f'{self.process_dir}{self.curdate}_LS.csv'
            if os.path.exists(file_path_process):
                df_process = pd.read_csv(file_path_process)
            else:
                df_process = pd.DataFrame(columns=self.adj_paras_dict['formated_columns_process'])

            return df_process
        elif mode == 'w':
            process_file_path = self.process_dir + f'{self.curdate}_LS.csv'
            if os.path.exists(process_file_path):
                time_flag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                file_path_bk = self.process_dir + f'backup/{self.curdate}_LS_{time_flag}.csv'
                shutil.copy(process_file_path, file_path_bk)
            df_process = df_process[self.adj_paras_dict['formated_columns_process']]

            df_process.to_csv(process_file_path, index=False)
        elif mode == 'replace':
            config_product_list = df_process['Main'].to_list()
            df_origin = self.process_config_wr(mode='r')
            df_open_index = df_origin[df_origin['How'].str.startswith('OpenIndex')]
            df_origin = df_origin[((df_origin['BarMode'] != 8) | (df_origin['StartBar'] != start_bar)) &
                                  ((df_origin['BarMode'] != 48) | (df_origin['StartBar'] != start_bar_5m)) &
                                  (~df_origin['How'].str.startswith('OpenIndex')) &
                                  (~df_origin['Main'].isin(config_product_list))]

            df_origin['ExecPos'] = np.nan

            df_new = pd.concat([df_origin, df_process, df_open_index], axis=0)
            self.process_config_wr(mode='w', df_process=df_new)
        elif mode == 'oi':
            open_index_type = df_process['How'].iloc[0]
            df_origin = self.process_config_wr(mode='r')
            df_open_index = df_origin[(df_origin['How'] == open_index_type) & df_origin['Main'].isin(
                list(self.exchange_apply_dict.keys()))].copy(deep=True)
            df_origin = df_origin[df_origin['How'] != open_index_type]
            for exchange_apply in self.exchange_apply_dict:
                if exchange_apply in df_open_index['Main'].to_list(): df_process = df_process[
                    df_process['Main'] != exchange_apply]

            df_new = pd.concat([df_origin, df_open_index, df_process], axis=0)
            self.process_config_wr(mode='w', df_process=df_new)

    def process_monitor_cfg(self, df_ls):
        df_targetmv = df_ls.copy(deep=True)[['Account', 'Main', 'Amount', 'Future', 'ExpectPos', 'Active']].rename(
            {'Future': 'ExecPos', 'Amount': 'LngShtM', 'Account': 'Product'}, axis='columns')
        df_targetmv['LngShtM'] = df_targetmv['LngShtM'].fillna('0W').replace('', '0W').str[:-1].astype('int')
        df_targetmv['Date'] = self.curdate

        conlist = []
        for product, df_prod in df_targetmv.groupby('Product'):
            if len(df_prod) == 1:
                conlist.append(df_prod)
                continue

            df_prod = df_prod[df_prod['Active'].astype('int') != 2]
            if len(df_prod) == 1:
                conlist.append(df_prod)
                continue

            df_prod['ExecPos'] = df_prod['ExecPos'].fillna('')
            df_prod_tmv = df_prod[df_prod['Active'].astype('int') == 1]
            df_prod_oi = df_prod[df_prod['Active'].astype('int') == 0]
            df_prod_tmv['ExecPos'] = df_prod_tmv['ExecPos'].iloc[0] + df_prod_oi['ExecPos'].iloc[0]
            df_prod_tmv['ExpectPos'] = df_prod_oi['ExpectPos'].iloc[0]
            conlist.append(df_prod_tmv)

        df_targetmv_new = pd.concat(conlist, axis=0).replace('', np.nan)

        with open(f'{self.output_dir}{self.curdate}_TargetMV.json', 'w') as jf:
            json.dump(df_targetmv_new.to_dict(orient='records'), jf, indent=2)
        tsl = TransferServerLocal(ip='120.24.92.78', port=22, username='monitor', password='ExcellenceCenter0507')
        tsl.upload_file(local_path=f'{self.output_dir}{self.curdate}_TargetMV.json',
                        server_path='/home/monitor/MonitorCode/HelloWorld/static/file/TargetMV.json')

    def process_to_downstream(self, update_pkl=True, new_mode=True, oi_tmv_combine=True):
        df_ls = self.process_config_wr(mode='r').rename(
            {'Product': 'Account', 'LngShtM': 'Amount', 'ExecPos': 'Future', 'How': 'Active'}, axis='columns')
        
        if oi_tmv_combine:
            # 无股票操作时，期货仓位之间合并
            df_oi_fix = df_ls[df_ls['Active'] == 'OpenIndexFix']
            df_ls = df_ls[df_ls['Active'] != 'OpenIndexFix']
            conlist = [df_oi_fix]
            for _, df_main in df_ls.groupby('Main'):
                type_num = len(df_main['Active'].value_counts())
                amount_sum = np.abs(df_main['Amount'].fillna('0W').str[:-1].astype('int').sum())
                if (type_num == 1) or (amount_sum > 0):
                    conlist.append(df_main)
                    continue
                
                df_main_oi = df_main[df_main['Active'] == 'OpenIndex']
                df_main_tmv = df_main[df_main['Active'] == 'TargetMV']
                tmv_fut_opr = df_main_tmv['Future'].iloc[0]
                future_dir = 'LONG' if 'LONG' in tmv_fut_opr else 'SHORT'
                tmv_fut_opr = self.format_future_operation_list_2_str(tmv_fut_opr, self.future_list, future_dir=future_dir, reverse=True, rvrs_type='dict')
                oi_fut_opr = df_main_oi['Future'].iloc[0]
                oi_fut_opr = self.format_future_operation_list_2_str(oi_fut_opr, self.future_list, future_dir=future_dir, reverse=True, rvrs_type='dict')
                expect_pos = str(df_main_oi['ExpectPos'].iloc[0]).strip()
                expect_pos = eval(expect_pos) if str(expect_pos).startswith('{') else {}

                oi_fut_opr = {key: oi_fut_opr.get(key, 0) + tmv_fut_opr.get(key, 0) for key in set(oi_fut_opr) | set(tmv_fut_opr)}
                expect_pos = {key: expect_pos.get(key, 0) + tmv_fut_opr.get(key, 0) for key in set(expect_pos) | set(tmv_fut_opr)}
                
                oi_fut_opr = [oi_fut_opr.get(fut, 0) for fut in self.future_list]
                df_main_oi['Future'] = self.format_future_operation_list_2_str(oi_fut_opr, self.future_list, future_dir=future_dir, reverse=False)
                df_main_oi['ExpectPos'] = str(expect_pos)
                conlist.append(df_main_oi)
            df_ls = pd.concat(conlist, axis=0)

        # 开股指目标仓位 ——> targetmv 后
        df_ls_oi = df_ls[df_ls['Active'] == 'OpenIndex']
        if not df_ls_oi.empty:
            df_ls_targetmv = df_ls[df_ls['Active'] == 'TargetMV']
            df_ls_oi_fix = df_ls[df_ls['Active'] == 'OpenIndexFix']
            for index in df_ls_oi.index.to_list():
                df_tmv = df_ls_targetmv[df_ls_targetmv['Account'] == df_ls_oi.loc[index, 'Account']]
                if not df_tmv.empty:
                    targetmv_except_pos = df_tmv['ExpectPos'].iloc[0]
                    if isinstance(targetmv_except_pos, str): targetmv_except_pos = eval(targetmv_except_pos)
                    if isinstance(targetmv_except_pos, float): targetmv_except_pos = {}
                    operation_pos = self.format_future_operation_list_2_str(
                        df_ls_oi.loc[index, 'Future'], self.future_list, future_dir='LONG', reverse=True, rvrs_type='dict')
                    new_except_pos = {key: targetmv_except_pos.get(key, 0) + operation_pos.get(key, 0) for key in
                                      set(targetmv_except_pos) | set(operation_pos)}
                    df_ls_oi.loc[index, 'ExpectPos'] = str(new_except_pos)
            df_ls = pd.concat([df_ls_oi, df_ls_oi_fix, df_ls_targetmv], axis=0)

        df_ls['Active'] = df_ls['Active'].replace({'TargetMV': 1, 'OpenIndex': 0, 'OpenIndexFix': 2})
        df_ls['Active'] = np.where(df_ls['Active'] == 0,
                                   df_ls['Account'].apply(lambda x: self.exchange_apply_dict.get(x, 0)),
                                   df_ls['Active']).astype('int').astype('str')

        df_ls_dup = df_ls[df_ls.duplicated(subset=['Account', 'Active'])]
        assert df_ls_dup.empty, f'设置存在重复: {df_ls_dup}'

        self.process_check.check_instrument_direction(df_ls['Class'], df_ls['Future'], df_ls['Account'])

        # 期货主账户映射
        conlist = []
        for (active_, trading_acc), df_ls_line in df_ls.groupby(['Active', 'Account']):
            hedge_acc = Dict_ProductionName_Replace.get(trading_acc)
            if hedge_acc is None:
                conlist.append(df_ls_line)
                continue

            if isinstance(df_ls_line['Future'].iloc[0], str) or isinstance(df_ls_line['ExpectPos'].iloc[0], str):
                df_ls_line_hedge = df_ls_line.copy(deep=True)
                df_ls_line_hedge['Account'] = hedge_acc
                df_ls_line_hedge['Amount'] = np.nan

                df_ls_line['Future'] = np.nan
                df_ls_line['ExpectPos'] = np.nan

                conlist.append(df_ls_line_hedge)
                conlist.append(df_ls_line)
            else:
                conlist.append(df_ls_line)

        df_ls = pd.concat(conlist, axis=0).reset_index(drop=True).sort_values(['Active', 'Linear', 'Main'])
        df_ls = df_ls.dropna(subset=['Amount', 'Future', 'ExpectPos'], how='all', axis=0)

        self.process_monitor_cfg(df_ls)

        df_ls['Amount'] = df_ls['Amount'].apply(
            lambda x: x.strip().replace(' ', '').upper() if isinstance(x, str) else x)
        df_ls['Future'] = df_ls['Future'].apply(
            lambda x: x.strip().replace(' ', '').upper().replace('SHORT', 'SHORT ').replace('LONG', 'LONG ') if isinstance(x, str) else x)
        df_ls['Distribution'] = df_ls.apply(
            lambda row: self.generate_bar_process_amount(row['Amount'], row['Linear'], row['StartBar'], row['BarMode']), axis=1)

        if new_mode: df_ls = df_ls[['Account', 'Amount', 'Future', 'Active', 'Distribution', 'ExpectPos']]
        else: df_ls = df_ls[['Account', 'Amount', 'Future', 'Active', 'Distribution']]
        
        # 混合对冲 按照 FutureName 拆成行
        ls_mix_infor = []
        msg_swap_list = [f'{self.curdate}:']
        for dict_ls_mix in df_ls.to_dict(orient='records'):
            future_str = dict_ls_mix['Future']
            account = dict_ls_mix['Account']
            if account in self.adj_paras_dict['product_swap'] + self.adj_paras_dict['product_option']:
                dict_ls_mix['Future'] = np.nan
                dict_ls_mix['Tag'] = np.nan
                ls_mix_infor.append(deepcopy(dict_ls_mix))
                msg_swap_list.append(f'{account}: {dict_ls_mix["Amount"]}|{future_str}|{dict_ls_mix["ExpectPos"]}')
                continue

            if not isinstance(future_str, str):
                dict_ls_mix['Future'] = np.nan
                dict_ls_mix['Tag'] = np.nan
                ls_mix_infor.append(deepcopy(dict_ls_mix))
                continue

            for fut_oper in future_str.split(','):
                future_name, future_oper = fut_oper.split(':')
                dict_ls_mix['Future'] = future_oper
                dict_ls_mix['Tag'] = future_name
                ls_mix_infor.append(deepcopy(dict_ls_mix))
        df_ls = pd.DataFrame(ls_mix_infor)

        msg_swap = '\n\t'.join(msg_swap_list)
        wechat_bot_msg_check(msg_swap, type_api='capital-config')

        if new_mode: df_ls = df_ls[['Account', 'Amount', 'Future', 'Tag', 'Active', 'Distribution', 'ExpectPos']]
        else: df_ls = df_ls[['Account', 'Amount', 'Future', 'Tag', 'Active', 'Distribution']]

        target_file_path = self.downstream_dir + f'OpenCloseAmount_{self.curdate}.csv'
        if not os.path.exists(target_file_path):
            df_ls.to_csv(target_file_path, index=False)
        else:
            df_ls_pre = pd.read_csv(target_file_path)
            if not df_ls_pre.empty:
                time_flag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                shutil.copy(target_file_path, f'{self.downstream_dir}backup/OpenCloseAmount_{time_flag}.csv')
            df_ls_pre = df_ls_pre[~ df_ls_pre['Active'].astype('str').isin(['0', '1', '2'])]
            df_ls = pd.concat([df_ls, df_ls_pre], axis=0)
            df_ls.to_csv(target_file_path, index=False)

        if update_pkl: 
            time_flag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            
            status, log = subprocess.getstatusoutput(f'"C:/Program Files/Git/usr/bin/bash.exe" {PLATFORM_PATH_DICT["v_path"]}StockData/IndexPortfolioFile/Code/update_openclose_amount.sh')
            path_pkl = f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/{self.curdate}/PreLoadPara.pkl'
            path_pkl_5m = f'{PLATFORM_PATH_DICT["v_path"]}StockData/ConfigDailyStore/{self.curdate}/PreLoadPara_5m.pkl'
            
            mtime_format = datetime.datetime.fromtimestamp(os.path.getmtime(path_pkl)).strftime("%Y%m%d-%H%M%S")
            mtime_format_5m = datetime.datetime.fromtimestamp(os.path.getmtime(path_pkl_5m)).strftime("%Y%m%d-%H%M%S")
            msg = f'{time_flag}[update pkl]: ({status}, {log})\n\t{mtime_format}—>PreLoadPara.pkl;\n\t{mtime_format_5m}—>PreLoadPara_5m.pkl;'
            
            wechat_bot_msg_check(msg)
    
    def process_close_t0(self):
        close_t0_exec_ratio = self.adj_paras_dict.get('close_t0_exec_ratio', 0.05)
        holdmv_dict_pre = self.df_stocks.set_index('Product')['HoldMV'].to_dict()

        df_pcss = self.process_config_wr(mode='r')
        df_pcss = df_pcss[df_pcss['How'] == 'TargetMV']
        df_pcss['LngShtM'] = df_pcss['LngShtM'].fillna('0W').str[:-1].astype('float') * 1e4
        df_pcss['LngShtR'] = df_pcss['LngShtM'] / df_pcss['Product'].apply(lambda x: holdmv_dict_pre.get(x, 1))
        ct0_list = df_pcss[np.abs(df_pcss['LngShtR']) > close_t0_exec_ratio]['Product'].to_list()

        gcpc = GenerateChangeParasConfig(self.curdate)
        change_config_path = gcpc.cfg_output_dir + f'{self.curdate}_change_cfg_paras.json'
        if not os.path.exists(change_config_path):
            new_dict_res = {'close_t0': ct0_list}
            process_continue = True
        else:
            with open(change_config_path, 'r') as jf: origin_dict_res = json.load(jf)
            new_dict_res = deepcopy(origin_dict_res)
            if origin_dict_res.get('close_t0') is None: new_dict_res['close_t0'] = ct0_list
            else: new_dict_res['close_t0'] = sorted(list(set(new_dict_res['close_t0'] + ct0_list)))

            if new_dict_res == origin_dict_res: process_continue = False
            else: process_continue = True

        if not process_continue: return print(f'{self.curdate}-参数已经传输!')
        with open(change_config_path, 'w') as jf: json.dump(new_dict_res, jf, indent=2)
        print(new_dict_res)

        gcpc.generate_change_cfg([
            {
                'all_prod': False,
                'class': [],
                'class_drop': [],
                'add_plist': prd_list,
                'drop_plist': [],
                'exch_list': ['sz', 'sh'],
                'add_priority': True,
                'cmd': action,
                'cmd_paras_dict': {
                }
            } for action, prd_list in new_dict_res.items()
        ])
        gcpc.upload_change_cfg()


    def process_generate_swap_future_operation(self, product_list=None):
        price_mode = 'realtime' if self.run_mode.startswith('marketing') else 'close'

        if price_mode == 'realtime': df_basis = self.get_monitor_future_basis_data(self.curdate, down_json_mode=True)
        else: df_basis = self.get_close_future_basis_data(self.predate, close_basis_all=False)

        if product_list is not None: print(df_basis)
        basis_ascending_list = df_basis.sort_values(['FutureName', 'RetAnn']).groupby('FutureName').agg(
            {'Contract': lambda x: x.to_list()})['Contract'].to_dict()
        for futname, contract_list in basis_ascending_list.items(): print(futname, contract_list)
        print('\n')

        df_swap = pd.read_excel(f'{self.output_dir}{self.curdate}_Total_Process.xlsx', sheet_name='swap').set_index('Product')
        swap_product_list = df_swap.index.to_list()
        df_swap_prss = pd.read_csv(f'{self.config_dir_prss}LS_config.csv')
        if product_list is None: product_list = df_swap_prss[df_swap_prss['PlanIn'] == 1]['Product'].to_list()
        for product in product_list:
            if product not in swap_product_list: continue

            hedge_dict = eval(df_swap.loc[product, "HedgePosDict"])
            hedge_dict = {key: int(value) for key, value in hedge_dict.items()}
            future_operation = df_swap.loc[product, "ExecPos"]
            if str(future_operation) in ['0', 'nan', '']: continue

            future_operation_dict = {fut_oper.split(':')[0]: int(fut_oper.split(':')[1].replace('SHORT ', '')) for
                                     fut_oper in future_operation.split(',')}
            future_operation_list = []
            for fut_nm, oper_num in future_operation_dict.items():
                if oper_num >= 0:
                    contract_nm = basis_ascending_list[fut_nm][0]
                    future_operation_list.append(f'开 {abs(oper_num)}张 SHORT {contract_nm}')
                else:
                    for fut_nm_pos in basis_ascending_list[fut_nm][::-1]:
                        pos_num = hedge_dict.get(fut_nm_pos, 0)
                        if pos_num == 0: continue
                        if oper_num == 0: continue
                        if oper_num < pos_num:
                            future_operation_list.append(f'平 {abs(pos_num)}张 SHORT {fut_nm_pos}')
                            oper_num -= pos_num
                        else:
                            future_operation_list.append(f'平 {abs(oper_num)}张 SHORT {fut_nm_pos}')
                            oper_num = 0

            future_operation_details = '; '.join(future_operation_list)

            msg = f'\033[33m{product}\033[0m:\n'
            msg += f'\t\033[33m期货持仓: {hedge_dict}\033[0m\n'
            msg += f'\t\033[33m股票操作: {df_swap.loc[product, "LngShtM"]}\033[0m\n'
            msg += f'\t\033[33m期货操作: {future_operation}\033[0m\n'
            msg += f'\t\033[33m期货操作细节: {future_operation_details}\033[0m\n'
            msg += f'\t\033[33m执行时间:  \033[0m'
            print(msg)

    def process_df_2_style(self, df, columns_list=None, flag='', thres=2):
        if flag == 'apply_repay':
            df = df.copy(deep=True).astype('str').style.set_properties(
                **{'text-align': 'center'}).set_table_styles(
                [dict(selector='th', props=[('text-align', 'center')])]).set_table_styles([
                {'selector': 'th', 'props': [('border', '1px solid black')]},
                {'selector': 'td', 'props': [('border', '1px solid black')]}]).apply(
                lambda x: ['background-color: #FFB2DE' if float(v) > (100 + thres)
                           else ('background-color: #5B9B00' if float(v) < (100 - thres) else '') for v in x], axis=0,
                subset=['EstmNVR']).apply(
                lambda x: ['background-color: #FFB2DE' if float(v) > 0
                           else ('background-color: #5B9B00' if float(v) < 0 else '') for v in x], axis=0, subset=columns_list)
        elif flag == 'origin_data':
            df = df.copy(deep=True).astype('str').style.set_properties(
                **{'text-align': 'center'}).set_table_styles(
                [dict(selector='th', props=[('text-align', 'center')])]).set_table_styles([
                {'selector': 'th', 'props': [('border', '1px solid black')]},
                {'selector': 'td', 'props': [('border', '1px solid black')]}]).background_gradient(
                cmap='Reds', subset=['HoldMVR']).apply(
                lambda x: ['background-color: #FFB2DE' if float(v) > (100 + thres)
                           else ('background-color: #5B9B00' if float(v) < (100 - thres) else '') for v in x], axis=0,
                subset=['EstmNVR'])
        else:
            raise ValueError
        return df
    
    def check(self, df_target=None, redown_mode=True):
        mini_product_nav_thres = self.adj_paras_dict.get('mini_product_nav_thres', 3e7)
        mini_product_list = self.df_total_data[(self.df_total_data['EstmNAV'] <= mini_product_nav_thres)]['Product'].to_list()

        if redown_mode:
            self.dict_price_data, self.dict_value_data, self.instrument_list = self.load_price_data()
            self.df_total_data, self.df_stocks, self.df_hedge_cfgout = self.load_all_trading_data()

        if df_target is None: df_target = self.process_config_wr(mode='r')

        df_check, check_msg = self.process_check.check_target_mv(
            self, self.curdate, df_process_origin=df_target, df_origin=self.df_total_data,
            df_origin_stocks=self.df_stocks, oi_dict=self.apply_dict, paras_dict=self.adj_paras_dict)

        df_check_except = df_check[df_check['FtrExT0R'] != 1]
        if not df_check_except.empty:
            print('如下产品期货操作存在回转，请确认！')
            print(df_check_except)
            wechat_bot_msg_check('如下产品期货操作存在回转，请确认！请输入 1 表示继续:')
            if input('请输入 1 表示继续:').strip() != '1': raise ValueError

        img_path = self.output_dir + f'check_target_mv_{self.curdate}.png'
        df_check_style = df_check.fillna('').astype('str').style.set_properties(
            **{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]).set_table_styles([
            {'selector': 'th', 'props': [('border', '1px solid black')]},
            {'selector': 'td', 'props': [('border', '1px solid black')]}]).apply(
            lambda x: ['background-color: {0}'.format('red') if abs(float(v)) > 2 else '' for v in x],
            axis=0, subset=['ExpMR', 'TgtExpMR', 'ExeExpMR']).apply(
            lambda x: ['background-color: {0}'.format('yellow') if str(v) == 'TargetMV' else '' for v in x], axis=0,
            subset=['How']).apply(
            lambda x: self.process_check.flag_product(x, mini_product_list, self.adj_paras_dict), axis=0,
            subset=['Product', 'Main']).background_gradient(
            cmap='Reds', vmin=1, vmax=3, subset=['FtrExT0R']).background_gradient(cmap='Reds', subset=['TgtPosR'])
        df_check_style.to_excel(self.output_dir + f'check_target_mv_{self.curdate}.xlsx', index=None)

        try:
            # dfi.export(df_check_style, filename=img_path, dpi=300, fontsize=10, max_cols=-1, max_rows=-1, table_conversion='selenium')
            dfi.export(df_check_style, filename=img_path, dpi=300, fontsize=10, max_cols=-1, max_rows=-1, table_conversion='chrome')
            wechat_bot_image(img_path)
        except:
            wechat_bot_file(self.output_dir + f'check_target_mv_{self.curdate}.xlsx', type_api='check')

        df_check = df_check.rename({'Trans': 'Transfer'}, axis='columns')
        df_check['Transfer'] = df_check['Transfer'].fillna('0W').replace('', '0W').str[:-1].astype('int') * 10000
        df_check = df_check.groupby('Main')[['Transfer']].sum().reset_index()
        df_check['Product'] = df_check['Main']
        df_check = df_check[df_check['Transfer'] > 0][['Product', 'Main', 'Transfer']]
        df_check.to_csv(f'{self.trans_data_dir}{self.curdate}_hedge_open_index_origin.csv', index=False)

        wechat_bot_msg_check(check_msg)

    def check_netvalue(self, df_apply_repay):
        df_origin = self.df_total_data.copy(deep=True)
        df_origin = df_origin[df_origin['Cls'] <= 1]
        non_netvalue_product_list = df_origin[df_origin['NetValueDate'] == '-']['Product'].to_list()
        print(f'\n{self.curdate}-没有匹配到净值产品：{non_netvalue_product_list}\n')
        df_origin = df_origin[self.priority_move_list]
        df_origin['MissValue(W)'] = np.round(df_origin['NetValue'] * (1 - df_origin['EstmNVR'] / 100) / 1e4).fillna(
            0).astype('int').astype('str') + 'W'
        df_origin.insert(0, 'BankCashRatio', df_origin.pop('BankCashRatio'))
        df_origin.insert(0, 'BankCash', df_origin.pop('BankCash'))
        df_origin.insert(0, 'MissValue(W)', df_origin.pop('MissValue(W)'))

        df_origin = pd.merge(df_apply_repay.astype('str'), df_origin, on='Product', how='outer').fillna(0)

        df_origin['Flag'] = np.abs(df_origin['EstmNVR'] - df_origin['EstmNVR'].median())
        df_origin = self.format_result_excel_columns(df_origin, format_type='W', columns_list=['NetValue', 'BankCash'])
        df_origin = df_origin.sort_values('Flag', ascending=False).drop('Flag', axis=1)
        return df_origin

    def calculate(self, end_bar_default=0, end_bar_5m_default=43, close_t0_list={}):
        df_total_data, df_stocks = self.df_total_data, self.df_stocks
        df_total_data.to_csv(f'{self.output_dir}{self.curdate}_origin_data.csv', index=False)

        adj_dc = AdjustPositionHedge(self.curdate, self.run_paras_dict)
        adj_zq = AdjustPosition(self.curdate, self.run_paras_dict, self.apply_zq_history_list)
        adj_swap = AdjustPositionSwap(self.curdate, self.run_paras_dict)

        df_style_swap, df_swap = adj_swap.adjust(df_total_data, df_stocks, 'not_trans', self.adj_paras_dict)
        df_style_zq, df_zq = adj_zq.adjust(df_total_data, df_stocks, 'not_trans', self.adj_paras_dict, self.flag_match_apply)
        df_style_dc, df_dc = adj_dc.adjust(df_total_data, df_stocks, 'not_trans', self.adj_paras_dict)
        
        df_ar = self.process_df_2_style(
            self.apply_repay_history,
            [col for col in self.apply_repay_history.columns.to_list() if col.startswith('2')], 
            flag='apply_repay')
        
        with pd.ExcelWriter(self.path_total_res, engine='openpyxl') as writer:
            df_style_swap.to_excel(writer, sheet_name=f'swap', index=False)
            df_style_zq.to_excel(writer, sheet_name=f'zq', index=False)
            df_style_dc.to_excel(writer, sheet_name=f'dc', index=False)

            df_ar.to_excel(writer, sheet_name='申购赎回原始数据', index=False)

            self.apply_history.to_excel(writer, sheet_name='申购数据', index=False)
            self.df_cash_out_plan.to_excel(writer, sheet_name='出金原始数据', index=False)
            self.process_df_2_style(df_total_data, [], flag='origin_data').to_excel(writer, sheet_name='账户原始数据', index=False)

            config_product_list = self.process_config_wr(mode='r')['Main'].to_list()
            
            df_swap['PlanIn'] = (~ df_swap['Main'].isin(config_product_list)) & (df_swap['ShortMain'] != 0)
            df_dc['PlanIn'] = (~ df_dc['Main'].isin(config_product_list)).astype('int')
            df_zq['PlanIn'] = ((~ df_zq['Main'].isin(config_product_list)) & 
                               ((~df_zq['Main'].isin(self.adj_paras_dict['zq_not_adjust_list'])) | (df_zq['Short'] != 0))).astype('int')

            df_total = pd.concat(objs=[df_swap, df_zq, df_dc], axis=0)
            df_total['Class'] = df_total['Class'].astype('str')
            df_total['Linear'] = ((~df_total['Class'].str.endswith('DC')) & (df_total['Short'] != 0)).astype('int')
            df_total['Linear'] = df_total['Linear'].replace({0: end_bar_default, 1: 8}) * (df_total['BarMode'] == 8) + (
                df_total['BarMode'] != 8) * df_total['Linear'].replace({0: end_bar_5m_default - 1, 1: 47})

            self.output_config_targetmv(df_total, self.adj_paras_dict, self.config_dir_prss, self.config_dir)
            self.output_change_cfg_paras_config(df_total, self.adj_paras_dict, close_t0_list.get(self.curdate, []))

        self.process_generate_swap_future_operation()

    def calculate_trans_target(self):
        df_total_data, df_stocks = self.df_total_data, self.df_stocks
        adj_dc = AdjustPositionHedge(self.curdate, self.run_paras_dict)
        adj_zq = AdjustPosition(self.curdate, self.run_paras_dict, self.apply_zq_history_list)
        
        path_total_res = f'{self.output_dir}{self.curdate}_Total_Process_Trans.xlsx'
        with pd.ExcelWriter(path_total_res, engine='openpyxl') as writer:
            df_style, df_dc_trans = adj_dc.adjust(df_total_data, df_stocks, 'trans', self.adj_paras_dict, adj_min_ratio_filter=False)
            df_style.to_excel(writer, sheet_name=f'dc-trans', index=False)

            df_style_zq, df_zq_trans = adj_zq.adjust(df_total_data, df_stocks, 'trans', self.adj_paras_dict, self.flag_match_apply)
            df_style_zq.to_excel(writer, sheet_name=f'zq-trans', index=False)

            self.output_config_trans_to_future(df_dc_trans, self.output_dir, self.adj_paras_dict)
            self.output_config_trans_to_future(df_zq_trans, self.output_dir, self.adj_paras_dict, suffix_name='_zq', positive_mode=True)

    def calculate_adj_exp(self, adj_exp_class_list=None, adj_exp_product_list=None):
        df_total_data, df_stocks = self.df_total_data, self.df_stocks
        all_class_list = sorted(list(df_total_data['Class'].unique()))
        print(f'\n{self.curdate}-所有产品策略分类: \n\t{all_class_list}')

        if not adj_exp_class_list: adj_exp_class_list = all_class_list
        if adj_exp_product_list is None: adj_exp_product_list = []
        
        target_product_list = adj_exp_product_list + self.adj_paras_dict['product_option'] + self.adj_paras_dict['product_swap']
        df_total_data = df_total_data[
            df_total_data['Class'].isin(adj_exp_class_list) | 
            df_total_data['Product'].isin(target_product_list) |
            (~ df_total_data['Class'].str.endswith('DC'))]

        path_total_res = f'{self.output_dir}{self.curdate}_Total_Process_AdjExp.xlsx'
        adj_swap = AdjustPositionSwap(self.curdate, self.run_paras_dict)
        adj_dc = AdjustPositionHedge(self.curdate, self.run_paras_dict)
        adj_zq = AdjustPosition(self.curdate, self.run_paras_dict, self.apply_zq_history_list)
        
        with pd.ExcelWriter(path_total_res, engine='openpyxl') as writer:
            for exec_mode in self.adjust_expose_mode_list:
                df_style, df = adj_dc.adjust(df_total_data, df_stocks, exec_mode, self.adj_paras_dict)
                df_style.to_excel(writer, sheet_name=exec_mode, index=False)

                df_style_zq, df_zq = adj_zq.adjust(df_total_data, df_stocks, exec_mode, self.adj_paras_dict, self.flag_match_apply)
                df_style_zq.to_excel(writer, sheet_name=f'zq-{exec_mode}', index=False)

                df_style_swap, df_swap = adj_swap.adjust(df_total_data, df_stocks, exec_mode, self.adj_paras_dict)
                df_style_swap.to_excel(writer, sheet_name=f'swap-{exec_mode}', index=False)

                df = pd.concat(objs=[df_swap, df, df_zq], axis=0)
                self.output_config_adjust_expose(df, exec_mode, self.config_dir_prss, self.config_dir)

    def process(self, start_bar=7, start_bar_5m=37, update_pkl=True, new_mode=True, oi_mode=True, check_mode=True, trans_mode=True, oi_tmv_combine=True):
        df_config = pd.read_csv(self.config_dir_prss + 'LS_config.csv')
        df_config['StartBar'] = start_bar * (df_config['BarMode'] == 8) + start_bar_5m * (df_config['BarMode'] != 8)
        df_config = df_config[
            (df_config['PlanIn'].astype('int') == 1) &
            ((df_config['LngShtM'].fillna('').astype('str') != '') | (df_config['ExecPos'].fillna('').astype('str') != ''))]

        self.process_config_wr(mode='replace', df_process=df_config, start_bar=start_bar, start_bar_5m=start_bar_5m)
        if oi_mode: self.process_hedge_open_index(trans_type='hedge_open_index', trans_mode=False, to_downstream=False)
        if check_mode: self.check()
        self.process_to_downstream(update_pkl=update_pkl, new_mode=new_mode, oi_tmv_combine=oi_tmv_combine)
        self.process_close_t0()

        if trans_mode: self.process_transfer(trans_type='hedge_open_index')


class GenerateTransferConfig(ProcessTradingDependData, AdjustProfile):
    def __init__(self, curdate, run_paras_dict, adj_paras_dict):
        super().__init__(curdate, run_paras_dict)
        AdjustProfile.__init__(self)

        self.curdate = curdate
        self.predate = get_predate(curdate, 1)
        self.nextdate = get_predate(curdate, -1)
        self.margin_base_ratio = 0.12

        self.rebuy_code_dict = {'SH': '204001', 'SZ': '131810'}
        self.trans_type_2_title = {
            'hedge_market_close': '当日赎回追保出入金情况',
            'hedge_open_index': '当日开股指出入金情况',
            'hedge_open_index_fix': '当日早盘补开股指出入金情况',
            'hedge_market_open': '当日盘前追保出入金情况'
        }
        self.future_trans_products_dict, self.auto_future_acc = self.get_future_accounts_auto_infor(adj_paras_dict['non_auto_trans_future_products'])
        trans_broker_records = self.load_operation_product_records()
        self.trans_broker_records_list = trans_broker_records.get(curdate, [])

    def get_settle_price_fix_factor(self, paras_dict):
        curdate = paras_dict.get('curdate', self.curdate)
        trans_type = paras_dict.get('trans_type', False)

        if trans_type == 'hedge_market_close':
            try:
                settle_price_fix_ratio = get_settle_price_correction_factor(curdate)
            except:
                print(traceback.format_exc())
                settle_price_fix_ratio = 0
        else:
            settle_price_fix_ratio = 0

        return settle_price_fix_ratio

    def get_trader_config_stocks(self, paras_dict):
        path_cfg = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/StockConfigAccount/'
        path_acc_cfg = f'{path_cfg}{self.curdate}_stock_account_config.csv'
        if not os.path.exists(path_acc_cfg): path_acc_cfg = f'{path_cfg}{self.predate}_stock_account_config.csv'

        df_trader_info = pd.read_csv(path_acc_cfg, encoding='GBK', dtype='str')
        value_trader_info = df_trader_info['Product'].value_counts()
        duplicates_trader_product = value_trader_info[value_trader_info > 1].index.to_list()

        df_trader_info['Account'] = df_trader_info['Account'].str[3:]
        df_trader_info['api'] = df_trader_info['api'].apply(lambda x: x.replace('-', '_'))
        df_trader_info['Broker'] = df_trader_info['Colo'].apply(
            lambda x: str(x).split('-')[0] if not str(x).startswith('cfipa') else 'pa')

        trans_bank_flag = (df_trader_info['Broker'].isin(paras_dict['auto_trans_broker_bank']) &
                           (~ df_trader_info['Bank'].isin(paras_dict['nonauto_trans_bank_list'])))
        df_trader_info['trans_mode'] = (1 * trans_bank_flag + 2 * (1 - trans_bank_flag)).astype('int').astype('str')

        df_trader_info['side'] = 'bank-out'

        df_trader_info = df_trader_info.rename(
            {'Product': 'production', 'Account': 'account', 'BankSimple': 'bank', 'BankCode': 'bank_id',
             'traderid': 'trader_id', 'Password': 'pwd', 'BankPwd': 'bank_pwd', 'api': 'api_type',
             'Bank': 'BankName'}, axis='columns')[[
            'production', 'account', 'bank', 'bank_id', 'trader_id', 'side', 'pwd', 'bank_pwd',
            'api_type', 'trans_mode', 'BankName', 'Broker']]

        return df_trader_info, duplicates_trader_product

    def get_future_capital_reserve_list(self, reserve_future_capital_list):
        apply_dict = self.get_apply_amount_data(self.curdate)
        apply_next_dict = self.get_apply_amount_data(self.nextdate)
        apply_dc_dict = self.get_apply_amount_data_dc(self.curdate, False)
        for key, key_dict in apply_dict.items(): reserve_future_capital_list += list(key_dict.keys())
        for key, key_dict in apply_next_dict.items(): reserve_future_capital_list += list(key_dict.keys())
        reserve_future_capital_list += list(apply_dc_dict.keys())
        reserve_future_capital_list = [self.get_product_future_main_name(prd) for prd in set(reserve_future_capital_list)]
        return reserve_future_capital_list

    def get_capital_and_trans_future(self, paras_dict):
        curdate = paras_dict.get('curdate', self.curdate)
        mrgcll_posr_joint_ctrl_mode = paras_dict.get('mrgcll_posr_joint_ctrl_mode', False)

        margin_call_target_ratio = paras_dict.get('margin_call_target_ratio', 0.22)
        margin_call_min_ratio = paras_dict.get('margin_call_min_ratio', 0.15)
        margin_call_max_ratio = paras_dict.get('margin_call_max_ratio', 0.24)
        margin_call_target_ratio_zq = paras_dict.get('margin_call_target_ratio_zq', 0.22)
        margin_call_min_ratio_zq = paras_dict.get('margin_call_min_ratio_zq', 0.13)
        margin_call_max_ratio_zq = paras_dict.get('margin_call_max_ratio_zq', 0.30)

        down_json_mode = paras_dict.get('down_json_mode', False)
        min_transfer_money = paras_dict.get('min_transfer_money', 200000)
        min_transfer_unit = paras_dict.get('min_transfer_unit', 50000)
        settle_price_fix_ratio = paras_dict.get('settle_price_fix_ratio', 0)
        only_call_margin = paras_dict.get('only_call_margin', False)
        future_trans_active = paras_dict.get('future_trans_active', True)
        
        reserve_future_capital_list = self.get_future_capital_reserve_list(paras_dict.get('reserve_future_capital_list', []))
        
        df_fut_trans = self.get_monitor_future_margin_data(curdate, down_json_mode)
        df_fut_trans = df_fut_trans[df_fut_trans['Product'].isin(WinterFallProductionList)]
        df_fut_trans['Main'] = df_fut_trans['Product'].apply(lambda x: self.get_product_trading_main_name(x))
        if not future_trans_active:
            df_fut_trans['FutCashIn'] = 0
            df_fut_trans['FutCashOut'] = 0
            df_fut_trans = df_fut_trans.groupby(['Product', 'Main']).agg({
                'Account': lambda x: x.head(1),
                'Capital': 'sum',
                'Margin': 'sum',
                'WithdrawQuota': 'sum', 'FutCashIn': 'sum', 'FutCashOut': 'sum'}).reset_index()
        elif paras_dict.get('open_index_df') is not None:
            df_open_index = paras_dict['open_index_df'].copy(deep=True).rename({"Transfer": 'FutCashIn'}, axis='columns')
            df_open_index['FutCashIn'] = (- np.ceil(np.abs(df_open_index['FutCashIn']) / min_transfer_unit) * min_transfer_unit)
            df_open_index['FutCashOut'] = 0
            df_fut_trans = pd.merge(df_open_index, df_fut_trans, on=['Product', 'Main'], how='left')
            df_fut_trans = df_fut_trans.groupby('Product').head(1)
        elif paras_dict.get('morning_tran_df') is not None:
            df_market_open = paras_dict['morning_tran_df'].copy(deep=True).rename({"Transfer": 'FutCashIn'}, axis='columns')
            df_market_open = df_market_open.groupby('Main')[['FutCashIn']].sum().reset_index().rename({"Main": 'Product'}, axis='columns')
            df_market_open = df_market_open[df_market_open['FutCashIn'] > 0]
            df_market_open['FutCashIn'] *= -1
            df_market_open['FutCashOut'] = 0
            df_fut_trans = pd.merge(df_market_open, df_fut_trans, on='Product', how='left')
            df_fut_trans = df_fut_trans.groupby('Product').head(1)
        else:
            dc_product_flag = df_fut_trans['Product'].apply(lambda x: production_2_strategy(x)) == 'DC'
            df_fut_trans['FutTrans'] = (df_fut_trans['Capital'] - df_fut_trans['Margin'] / 0.12 * (1 + settle_price_fix_ratio) *
                                        (dc_product_flag * margin_call_target_ratio + (~ dc_product_flag) * margin_call_target_ratio_zq))

            to_future_flag = (df_fut_trans['Capital'] - df_fut_trans['Margin'] / 0.12 * (1 + settle_price_fix_ratio) *
                              (dc_product_flag * margin_call_min_ratio + (~ dc_product_flag) * margin_call_min_ratio_zq) < 0)
            to_stock_flag = (df_fut_trans['Capital'] - df_fut_trans['Margin'] / 0.12 * (1 + settle_price_fix_ratio) *
                             (dc_product_flag * margin_call_max_ratio + (~ dc_product_flag) * margin_call_max_ratio_zq) > 0)

            df_fut_trans['FutCashIn'] = df_fut_trans['FutTrans'] * to_future_flag
            df_fut_trans['FutCashOut'] = df_fut_trans['FutTrans'] * to_stock_flag
            if mrgcll_posr_joint_ctrl_mode:
                df_trans = pd.read_csv(f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/LongShort/{curdate}/{curdate}_allocation_capital.csv')
                df_fut_trans['AccNum'] = 1
                joint_ctrl_trans = df_trans.groupby('Main')['Transfer'].sum().to_dict()
                margin_acc_sum = df_fut_trans.groupby('Main')['Margin'].sum().to_dict()
                acc_number = df_fut_trans.groupby('Main')['AccNum'].sum().to_dict()

                df_fut_trans['JointCtrlTrans'] = - df_fut_trans['Main'].apply(lambda x: joint_ctrl_trans.get(x, 0))
                df_fut_trans['MarginTotal'] = df_fut_trans['Main'].apply(lambda x: margin_acc_sum.get(x, 0))
                df_fut_trans['AccNumTotal'] = df_fut_trans['Main'].apply(lambda x: acc_number.get(x, 1))

                df_fut_trans['JointCtrlTrans'] *= np.where(
                    df_fut_trans['MarginTotal'] != 0, df_fut_trans['Margin'] / df_fut_trans['MarginTotal'], df_fut_trans['AccNum'] / df_fut_trans['AccNumTotal'])

                df_fut_trans['FutCashIn'] = np.where(dc_product_flag, df_fut_trans['JointCtrlTrans'] * (df_fut_trans['JointCtrlTrans'] < 0), df_fut_trans['FutCashIn'])
                df_fut_trans['FutCashOut'] = np.where(dc_product_flag, df_fut_trans['JointCtrlTrans'] * (df_fut_trans['JointCtrlTrans'] > 0), df_fut_trans['FutCashOut'])

            df_fut_trans['FutCashIn'] = np.round(df_fut_trans['FutCashIn'] / min_transfer_unit) * min_transfer_unit
            df_fut_trans['FutCashOut'] = np.floor(np.minimum(df_fut_trans['FutCashOut'], df_fut_trans['WithdrawQuota']) / min_transfer_unit) * min_transfer_unit
            df_fut_trans['FutCashOut'] *= (df_fut_trans['FutCashOut'] > min_transfer_money) * (~ df_fut_trans['Product'].isin(reserve_future_capital_list)) * (not only_call_margin)

        df_fut_trans['Main'] = df_fut_trans['Product'].apply(lambda x: self.get_product_trading_main_name(x))
        df_fut_trans = df_fut_trans[['Product', 'Main', 'Account', 'Capital', 'Margin', 'WithdrawQuota', 'FutCashIn', 'FutCashOut']]
        df_fut_trans = df_fut_trans.sort_values(['Product', 'Margin'], ascending=False)

        return df_fut_trans

    def get_capital_and_trans_stocks(self, paras_dict):
        curdate = paras_dict.get('curdate', self.curdate)
        predate = paras_dict.get('predate', self.predate)
        down_json_mode = paras_dict.get('down_json_mode', False)
        deducted_trans = paras_dict.get('deducted_trans', False)
        down_trans_json = paras_dict.get('down_trans_json', False)
        dc_rebuy_left = paras_dict.get('dc_rebuy_left', 0.10)
        zq_rebuy_left = paras_dict.get('zq_rebuy_left', 0.01)
        min_transfer_unit = paras_dict.get('min_transfer_unit', 50000)

        equity_prd_list = paras_dict.get('PrdType_Equity', [])

        df_stock_money = self.get_monitor_stocks_data(
            curdate, down_json_mode, deducted_trans, down_trans_json,
            deducted_trans_drop_colo_list=self.adjust_trans_deducted.get(curdate, {}).get('colo', []),
            deducted_trans_drop_product_list=self.adjust_trans_deducted.get(curdate, {}).get('prd', []))
        
        dict_rebuy = self.get_rebuy_data(predate)

        df_stock_money['HoldMV'] = df_stock_money['MV_collateral'] + df_stock_money['MV_shortSell']
        df_exch_cash = pd.pivot_table(df_stock_money, index='Product', columns='Exchange', values='Capital').fillna(0).reset_index()
        df_stock_money = df_stock_money.groupby('Product').agg({'MV_net': 'mean', 'HoldMV': 'mean', 'Capital': 'sum'}).reset_index()
        df_stock_money['PreRebuyM'] = df_stock_money['Product'].apply(lambda x: dict_rebuy.get(x, 0)) * 1e4 - min_transfer_unit
        df_stock_money['CashAvailOut'] = np.maximum(df_stock_money['Capital'] + np.minimum(df_stock_money['MV_net'] + df_stock_money['PreRebuyM'], 0), 0)
        df_stock_money = df_stock_money[['Product', 'HoldMV', 'Capital', 'CashAvailOut']].rename({"Capital": 'StockTotal'}, axis='columns')
        df_stock_money = pd.merge(df_stock_money, df_exch_cash, on='Product', how='outer').fillna(0)
        df_stock_money['Colo'] = df_stock_money['Product'].apply(lambda x: production_2_colo(x))
        df_stock_money['Colo_SH'] = df_stock_money['Colo'].apply(lambda x: production_2_colo_sh('', x))
        df_stock_money['Strategy'] = df_stock_money['Product'].apply(lambda x: production_2_strategy(x))

        df_stock_money['RebuyReserve'] = df_stock_money['HoldMV'] * df_stock_money['Strategy'].apply(lambda x: dc_rebuy_left if str(x) == 'DC' else zq_rebuy_left)
        df_stock_money['Main'] = df_stock_money['Product'].apply(lambda x: self.get_product_trading_main_name(x))
        df_stock_money['IsEquityPrd'] = df_stock_money['Product'].isin(equity_prd_list).astype('int')
        return df_stock_money

    def generate_trans_config_stocks(self, df_transfer, zq_product_list, paras_dict):
        curdate = paras_dict.get('curdate', self.curdate)
        trading_day = paras_dict.get('trading_day', curdate)
        trans_type = paras_dict.get('trans_type', 'hedge_market_close')
        min_transfer_unit = paras_dict.get('min_transfer_unit', 50000)
        rebuy_code_dict = paras_dict.get('rebuy_code_dict', self.rebuy_code_dict)
        extra_trans_infor = paras_dict.get('extra_trans_infor', [])

        rebuy_drop_list_extra = paras_dict.get('rebuy_drop_list_extra', [])

        rebuy_test_money = paras_dict.get('rebuy_test_money', None)
        only_call_margin = paras_dict.get('only_call_margin', False)
        cfg_output_dir = paras_dict['cfg_output_dir']
        wechat_and_email = paras_dict.get('wechat_and_email', False)
        trans_flag = paras_dict.get('trans_flag', "".join(trans_type.split('_')[2:]) if trans_type.count('_') >= 2 else trans_type)
        
        df_trans_fix = df_transfer.copy(deep=True)

        df_trans_bank = df_trans_fix[df_trans_fix['trans_mode'] == '1'].copy(deep=True)
        df_trans_bank_fix = df_trans_bank.copy(deep=True)
        df_trans_not_bank = df_trans_fix[df_trans_fix['trans_mode'] != '1'].copy(deep=True)
        fix_single_flag = (((df_trans_bank['StkSZOut'] > 0) & (df_trans_bank['StkSHOut'] == 0)) |
                           ((df_trans_bank['StkSZOut'] == 0) & (df_trans_bank['StkSHOut'] < 0))) + 1
        df_trans_bank['StkSZOut'] = ((df_trans_bank['StkSZOut'] - (df_trans_bank['StkSZOut'] > 0) *
                                      min_transfer_unit / 2 * fix_single_flag).astype('int'))
        df_trans_bank['StkSHOut'] = (df_trans_bank['StkSHOut'] - (df_trans_bank['StkSHOut'] > 0) *
                                     min_transfer_unit / 2 * fix_single_flag).astype('int')

        df_trans_bank_fix['StkSZOut'] = (min_transfer_unit / 2 * (df_trans_bank['StkSZOut'] > 0)).astype('int')
        df_trans_bank_fix['StkSHOut'] = (min_transfer_unit / 2 * (df_trans_bank['StkSHOut'] > 0)).astype('int')
        df_trans_bank_fix['trans_mode'] = '2'

        df_trans_bank_fix['trans_flag'] = 'fix'
        df_trans_bank['trans_flag'] = trans_flag
        df_trans_not_bank['trans_flag'] = trans_flag

        df_trans_bank['StkSZOut'] += df_trans_bank['StkSHOut'] * df_trans_bank['Colo'].apply(lambda x: np.sum([brkr in x for brkr in paras_dict['auto_trans_broker_only_sz']]) > 0)
        df_trans_not_bank['StkSZOut'] += df_trans_not_bank['StkSHOut'] * df_trans_not_bank['Colo'].apply(lambda x: np.sum([brkr in x for brkr in paras_dict['auto_trans_broker_only_sz']]) > 0)

        df_trans_fix = pd.concat([df_trans_bank_fix, df_trans_bank, df_trans_not_bank], axis=0).reset_index(drop=True)
        df_trans_fix['trans_mode'] = df_trans_fix['trans_mode'].replace('-', '2')
        trans_sz = df_trans_fix[
            ['production', 'account', 'Colo', 'StkSZOut', 'bank', 'bank_id',
             'trader_id', 'side', 'pwd', 'bank_pwd', 'api_type', 'trans_mode', 'trans_flag']].rename(
            {'Colo': 'colo', 'StkSZOut': 'money'}, axis='columns').copy(deep=True).reset_index(drop=True)
        trans_sh = df_trans_fix[
            ['production', 'account', 'Colo_SH', 'StkSHOut', 'bank', 'bank_id',
             'trader_id', 'side', 'pwd', 'bank_pwd', 'api_type', 'trans_mode', 'trans_flag']].rename(
            {'Colo_SH': 'colo', 'StkSHOut': 'money'}, axis='columns').copy(deep=True).reset_index(drop=True)

        only_sz_out_flag = trans_sh['colo'].apply(lambda x: np.sum([brkr in x for brkr in paras_dict['auto_trans_broker_only_sz']]) > 0)
        trans_sh['trans_mode'] = (only_sz_out_flag * 3 + (~ only_sz_out_flag) * trans_sh['trans_mode'].astype('int')).astype('str')

        trans_sz = trans_sz[trans_sz['money'] != 0]
        trans_sh = trans_sh[trans_sh['money'] != 0]
        trans_sz['money'] = (- trans_sz['money']).astype('int')
        trans_sh['money'] = (- trans_sh['money']).astype('int')

        with pd.ExcelWriter(f'{cfg_output_dir}{trading_day}_bank_transfer_cfg_{trans_type}.xlsx', engine='openpyxl') as writer:
            trans_sz.to_excel(writer, sheet_name=f'SZ', index=False)
            trans_sh.to_excel(writer, sheet_name=f'SH', index=False)

        transfer_info_sz, transfer_info_sh = trans_sz.to_dict(orient='list'), trans_sh.to_dict(orient='list')

        rebuy_sz = df_transfer[
            ['production', 'account', 'Colo', 'SZRebuy', 'trader_id']].rename(
            {'Colo': 'colo', 'SZRebuy': 'rebuymoney'}, axis='columns').copy(deep=True)
        rebuy_sh = df_transfer[
            ['production', 'account', 'Colo_SH', 'SHRebuy', 'trader_id']].rename(
            {'Colo_SH': 'colo', 'SHRebuy': 'rebuymoney'}, axis='columns').copy(deep=True)
        rebuy_sz['exch'] = 'SZ'
        rebuy_sh['exch'] = 'SH'
        rebuy_sz = rebuy_sz[rebuy_sz['rebuymoney'] != 0]
        rebuy_sh = rebuy_sh[rebuy_sh['rebuymoney'] != 0]
        rebuy_sz['rebuymoney'] = (rebuy_sz['rebuymoney'] / 100).astype('int')
        rebuy_sh['rebuymoney'] = (rebuy_sh['rebuymoney'] / 100).astype('int')

        rebuy_df = pd.concat([rebuy_sz, rebuy_sh], axis=0)[['production', 'colo', 'account', 'trader_id', 'rebuymoney', 'exch']]
        rebuy_df['rebuycode'] = rebuy_df['exch'].apply(lambda x: rebuy_code_dict[x])
    
        if not isinstance(extra_trans_infor, list): extra_trans_infor = [extra_trans_infor]
        if not rebuy_df.empty:
            extra_trans_infor.append({'mode': 'rebuycash', 'rebuy_df_infor': rebuy_df})
            rebuy_plan_product_list = list(rebuy_df['production'].unique())
        else:
            rebuy_plan_product_list = []

        if (rebuy_test_money is not None) and (0 < rebuy_test_money < 10000):
            extra_trans_infor.append({
                'mode': 'rebuycash', 'rebuymoney': int(rebuy_test_money / 100),
                'production': list(set(zq_product_list) - set(rebuy_plan_product_list + rebuy_drop_list_extra + Production_OwnList_Swap + ProductionList_AlphaShort))})
        
        if extra_trans_infor and (not only_call_margin):
            generate_extra_trans_and_rebuy_config(trading_day, config_dict_list=extra_trans_infor, rebuy_code_dict=rebuy_code_dict, wechat_and_email=wechat_and_email)

        generate_cfg_config(transfer_info_sz, f'{trading_day}_bank_transfer_{trans_type}_sz.cfg', cfg_output_dir)
        generate_cfg_config(transfer_info_sh, f'{trading_day}_bank_transfer_{trans_type}_sh.cfg', cfg_output_dir)

    def generate_trans_config_future(self, df_transfer, paras_dict):
        curdate = paras_dict.get('curdate', self.curdate)
        trading_day = paras_dict.get('trading_day', curdate)
        trans_type = paras_dict.get('trans_type', 'hedge_market_close')
        cfg_output_dir = paras_dict['cfg_output_dir']

        cmd_fut_cash_out_in = ['userid,action,number']
        for product, hedge_acc, transfer_in, transfer_out in df_transfer[['Product', 'Account', 'FutCashIn', 'FutCashOut']].values:
            if product in paras_dict['non_auto_trans_future_products']: continue
            if not self.auto_future_acc.get(str(hedge_acc), False): continue
            if str(hedge_acc) in ['nan', '']: continue

            if transfer_in < 0: cmd_fut_cash_out_in.append(f'{hedge_acc},i,{abs(int(transfer_in))}')
            if transfer_out > 0: cmd_fut_cash_out_in.append(f'{hedge_acc},o,{abs(int(transfer_out))}')

        cmd_cash = '\n'.join(cmd_fut_cash_out_in)
        with open(f'{cfg_output_dir}{trading_day}_{trans_type}.txt', 'wb') as f: f.write(cmd_cash.encode('utf-8'))

    def generate_summary(self, df_stock_trans, paras_dict):
        curdate = paras_dict.get('curdate', self.curdate)
        trading_day = paras_dict.get('trading_day', curdate)
        trans_type = paras_dict.get('trans_type', 'hedge_market_close')
        cfg_output_dir = paras_dict['cfg_output_dir']
        rebuy_code_dict = paras_dict.get('rebuy_code_dict', self.rebuy_code_dict)

        df_stock_trans = df_stock_trans[
            (np.abs(df_stock_trans['StkCashIn']) +
             np.abs(df_stock_trans['StkCashOut']) +
             np.abs(df_stock_trans['StkSZOut']) +
             np.abs(df_stock_trans['StkSHOut']) +
             np.abs(df_stock_trans['RepayMoney']) +
             np.abs(df_stock_trans['SZRebuy']) +
             np.abs(df_stock_trans['SHRebuy']) +
             np.abs(df_stock_trans['RepayMoney'])) > 0]

        df_config = df_stock_trans[[
            'Product', 'Main', 'Colo', 'NextCo', 'Strategy', 'HoldMV', 'RebuyReserve', 'StkOutOrigin', 'StkCashOut', 'CashGap',
            'StkSZOut', 'StkSHOut', 'SZRebuy', 'SHRebuy', 'SZ', 'SH']]
        df_config_sz = df_config[[
            'Product', 'Main', 'Colo', 'NextCo', 'Strategy', 'HoldMV', 'RebuyReserve', 'StkSZOut', 'SZRebuy', 'SZ']].rename(
            {'StkSZOut': 'CurOut', 'SZRebuy': 'Rebuy', 'SZ': 'Capital'}, axis='columns')
        df_config_sz['Exchange'] = 'SZ' + rebuy_code_dict['SZ']
        df_config_sh = df_config[[
            'Product', 'Main', 'Colo', 'NextCo', 'Strategy', 'HoldMV', 'RebuyReserve', 'StkSHOut', 'SHRebuy', 'SH']].rename(
            {'StkSHOut': 'CurOut', 'SHRebuy': 'Rebuy', 'SH': 'Capital'}, axis='columns')
        df_config_sh['Exchange'] = 'SH' + rebuy_code_dict['SH']
        df_config_exch = pd.concat([df_config_sz, df_config_sh], axis=0).sort_values('Product')
        df_config_exch['RebuyRatio'] = np.round(df_config_exch['Rebuy'] / df_config_exch['HoldMV'], 3)

        df_config_gap = df_config.copy(deep=True)[['Product', 'Main', 'CashGap']].rename({'CashGap': 'Transfer'}, axis='columns')
        df_config_gap = df_config_gap[df_config_gap['Transfer'] > 0]
        if not df_config_gap.empty:
            df_config_gap.to_csv(f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/Transfer/{trading_day}_{trans_type}_gap.csv', index=False)

        df_config = self.format_result_excel_columns(
            df_config, format_type='W',
            columns_list=['HoldMV', 'RebuyReserve', 'NextCo', 'StkOutOrigin', 'StkCashOut', 'CashGap', 'StkSZOut', 'StkSHOut', 'SZRebuy', 'SHRebuy', 'SZ', 'SH'])

        df_config = df_config.rename(
            {'NextCo': '次日出金', 'RebuyReserve': '回购预留', 'StkOutOrigin': '原始需求额', 'StkCashOut': '总需求额', 'CashGap': '资金缺口',
             'StkSZOut': 'SZ出金额', 'StkSHOut': 'SH出金额', 'SZRebuy': 'SZRebuy', 'SHRebuy': 'SHRebuy', 'SZ': 'SZ总资金', 'SH': 'SH总资金'}, axis='columns')

        path_details = f'{cfg_output_dir}{trading_day}_{trans_type}_details.xlsx'
        with pd.ExcelWriter(path_details, engine='openpyxl') as writer:
            df_config.to_excel(writer, sheet_name=f'details', index=False)
            df_config_exch.to_excel(writer, sheet_name=f'details_exch', index=False)

        print(f'{trading_day} Summary: {path_details} 生产完成！')

    def upload_trans_config(self, paras_dict):
        curdate = paras_dict.get('curdate', self.curdate)
        trading_day = paras_dict.get('trading_day', curdate)
        trans_type = paras_dict.get('trans_type', 'hedge_market_close')
        cfg_output_dir = paras_dict['cfg_output_dir']

        cmd_scp = f'"C:/Program Files/Git/usr/bin/scp.exe" -r {cfg_output_dir}{trading_day}_{trans_type}.txt product:~/operation/{trading_day}_{trans_type}.txt'
        print(cmd_scp)
        status, log = subprocess.getstatusoutput(cmd_scp)
        msg_transfer_ret = f'[{trading_day}-期货]:\n\tstatus={status}\n\tlog={log}'
        wechat_bot_msg_check(msg_transfer_ret)

        msg_transfer_ret = f'[{trading_day}-股票]:'
        for cmd_scp in [
            f'ssh jumper "[ ! -d ~/rtchg/bank_trans/{trading_day}/ ] && mkdir ~/rtchg/bank_trans/{trading_day}/"',
            f'"C:/Program Files/Git/usr/bin/scp.exe" '
            f'-r {cfg_output_dir}{trading_day}_bank_transfer_{trans_type}_sz.cfg '
            f'jumper:~/rtchg/bank_trans/{trading_day}/{trading_day}_bank_transfer_repay_sz.cfg',
            f'"C:/Program Files/Git/usr/bin/scp.exe" '
            f'-r {cfg_output_dir}{trading_day}_bank_transfer_{trans_type}_sh.cfg '
            f'jumper:~/rtchg/bank_trans/{trading_day}/{trading_day}_bank_transfer_repay_sh.cfg',
        ]:
            print(cmd_scp)
            status, log = subprocess.getstatusoutput(cmd_scp)
            msg_transfer_ret += f'\n\tstatus={status}\n\tlog={log}'
        wechat_bot_msg_check(msg_transfer_ret)

    def allocation(self, df_stock_data, df_fut_data, zq_product_list, paras_dict, duplicates_trader_product):
        margin_ratio_base = paras_dict.get('margin_ratio_base', 0.12)
        min_reserve_mcr = paras_dict.get('min_reserve_mcr', 0.1)
        max_rebuy_navr = paras_dict.get('max_rebuy_navr', 0.35)

        min_transfer_unit = paras_dict.get('min_transfer_unit', 50000)
        trans_type = paras_dict.get('trans_type', 'hedge_market_close')
        repay_trans_active = paras_dict.get('repay_trans_active', True)
        sz_sh_equal = paras_dict.get('sz_sh_equal', False)

        isrebuy = bool(paras_dict.get('isrebuy', 0))
        dc_rebuy_forbid = paras_dict.get('dc_rebuy_forbid', False)
        zq_rebuy_forbid = paras_dict.get('zq_rebuy_forbid', False)
        whether_dc_rebuy = paras_dict.get('whether_dc_rebuy', False)
        whether_zq_rebuy = paras_dict.get('whether_zq_rebuy', False)

        rebuy_list_extra = paras_dict.get('rebuy_list_extra', [])
        rebuy_drop_list_extra = paras_dict.get('rebuy_drop_list_extra', [])

        if not repay_trans_active:
            df_stock_data['RepayMoney'] = 0
            df_stock_data['NextCo'] = 0

        conlist, conlist_stk = [], []
        for main_prod in set(list(df_fut_data['Main'].unique()) + list(df_stock_data['Main'].unique())):
            df_main_stk = df_stock_data[df_stock_data['Main'] == main_prod].copy(deep=True)
            df_main_fut = df_fut_data[df_fut_data['Main'] == main_prod].copy(deep=True)
            
            # if df_main_stk.empty or df_main_fut.empty: continue
            nav_estm = (df_main_stk['HoldMV'] + df_main_stk['StockTotal'] + df_main_stk['RepayMoney'] + df_main_stk['NextCo']).sum() + df_main_fut['Capital'].sum()
            df_main_stk['StkCashRatio'] = df_main_stk['StockTotal'] / df_main_stk['HoldMV']
            df_main_stk['StkMVRatio'] = df_main_stk['HoldMV'] / df_main_stk['HoldMV'].sum()
            fut_out_money = df_main_fut['FutCashOut'].sum()
            fut_in_money = df_main_fut['FutCashIn'].sum()
            fut_trans = fut_out_money + fut_in_money
            is_equity_prd = df_main_stk['IsEquityPrd'].sum() > 0

            repay_out = df_main_stk['RepayMoney'].sum()
            stock_trans = repay_out + fut_trans
            
            len_stk, len_fut = len(df_main_stk), len(df_main_fut)
            if stock_trans >= 0:
                df_main_stk['StkCashIn'] = (
                        np.round((df_main_stk['StkMVRatio'] == np.min(
                            df_main_stk['StkMVRatio'] + df_main_stk['Product'].isin(self.trans_broker_records_list + duplicates_trader_product).astype('int'))) *
                                 stock_trans / min_transfer_unit) * min_transfer_unit)
                df_main_stk['StkOutOrigin'] = 0
                df_main_stk['StkCashOut'] = 0
                df_main_stk['StkSZOut'] = 0
                df_main_stk['StkSHOut'] = 0
            else:
                df_main_stk['StkCashIn'] = 0
                if paras_dict.get('morning_tran_df') is not None:
                    df_market_open = paras_dict['morning_tran_df'][
                        paras_dict['morning_tran_df']['Main'] == main_prod].copy(deep=True).rename({"Transfer": 'StkCashOut'}, axis='columns')
                    df_main_stk = pd.merge(df_main_stk, df_market_open, on=['Product', 'Main'], how='right')
                    df_main_stk['StkOutOrigin'] = df_main_stk['StkCashOut']
                else:
                    if fut_trans >= 0: repay_fut_out_ratio = df_main_stk['RepayMoney'] / repay_out
                    else:
                        total_cash_avail_out = df_main_stk['CashAvailOut'].sum()
                        if total_cash_avail_out > 0: repay_fut_out_ratio = df_main_stk['CashAvailOut'] / df_main_stk['CashAvailOut'].sum()
                        else: repay_fut_out_ratio = pd.Series(1 / len_stk, index=df_main_stk.index) if len_stk != 0 else 0
                    df_main_stk['StkOutOrigin'] = - (df_main_stk['RepayMoney'] + np.round(repay_fut_out_ratio * fut_trans / min_transfer_unit) * min_transfer_unit)
                    df_main_stk['StkCashOut'] = np.minimum(df_main_stk['StkOutOrigin'], np.floor(df_main_stk['CashAvailOut'] / min_transfer_unit - 1) * min_transfer_unit)

                    stock_out_lack_repay = - df_main_stk['StkCashOut'].sum() - repay_out - fut_out_money
                    stock_out_lack_total = - df_main_stk['StkCashOut'].sum() - repay_out - fut_trans
                    if (stock_out_lack_total > 0) and (stock_out_lack_repay > 0):
                        total_margin = df_main_fut['Margin'].sum()
                        total_capital = df_main_fut['Capital'].sum()
                        if total_margin > 0: distr_ratio = df_main_fut['Margin'] / total_margin
                        elif total_capital > 0:  distr_ratio = df_main_fut['Capital'] / total_capital
                        else: distr_ratio = 1 / max(len_fut, 1)

                        flag_fix_lack = np.round(stock_out_lack_total * distr_ratio / min_transfer_unit) * min_transfer_unit
                        flag_fix_lack += df_main_fut['FutCashIn'] + df_main_fut['FutCashOut']
                        df_main_fut['FutCashIn'] = flag_fix_lack * (flag_fix_lack < 0)
                        df_main_fut['FutCashOut'] = np.minimum(flag_fix_lack * (flag_fix_lack >= 0), np.floor(df_main_fut['WithdrawQuota'] / min_transfer_unit) * min_transfer_unit)
                    elif (stock_out_lack_total > 0) and (stock_out_lack_repay <= 0):
                        if fut_in_money != 0: distr_ratio = df_main_fut['FutCashIn'] / fut_in_money
                        else: distr_ratio = 1 / max(len_fut, 1)

                        df_main_fut['FutCashIn'] += np.round(stock_out_lack_total * distr_ratio / min_transfer_unit) * min_transfer_unit

                if sz_sh_equal:
                    dual_flag = df_main_stk['Product'].isin(DUALCENTER_PRODUCTION).astype('int')
                    dual_flag_sh = dual_flag / 2
                    dual_flag_sz = dual_flag_sh + 1 - dual_flag
                    df_main_stk['StkSZOut'] = dual_flag_sz * (df_main_stk['StkCashOut'] + min_transfer_unit)
                    df_main_stk['StkSHOut'] = dual_flag_sh * (df_main_stk['StkCashOut'] + min_transfer_unit)
                else:
                    df_main_stk['StkSZOut'] = np.round(
                        df_main_stk['SZ'] / df_main_stk['StockTotal'] * (df_main_stk['StkCashOut'] != 0) *
                        (df_main_stk['StkCashOut'] / min_transfer_unit + 1)) * min_transfer_unit
                    df_main_stk['StkSHOut'] = np.round(
                        df_main_stk['SH'] / df_main_stk['StockTotal'] * (df_main_stk['StkCashOut'] != 0) *
                        (df_main_stk['StkCashOut'] / min_transfer_unit + 1)) * min_transfer_unit

            df_main_stk['StkSZOut'] = df_main_stk['StkSZOut'].fillna(0).astype('int')
            df_main_stk['StkSHOut'] = df_main_stk['StkSHOut'].fillna(0).astype('int')

            if len_fut > 0:
                df_main_fut['Product'] += ('-' + pd.Series(np.array(list(range(1, len_fut + 1))), index=df_main_fut.index).astype('str'))
                df_main_fut['Product'] = df_main_fut['Product'].apply(lambda x: x.replace('-1', ''))
            
            sign_strategy = (df_main_stk['Strategy'] == 'DC').max() * 2 - 1
            min_reserve_mc_money = (df_main_fut['Margin'] * (1 + sign_strategy * min_reserve_mcr + min_reserve_mcr / margin_ratio_base)).sum()
            min_reserve_mc_money = max(nav_estm * (1 - max_rebuy_navr) - df_main_stk['HoldMV'].sum(), min_reserve_mc_money)
            rebuy_reserve = max(min_reserve_mc_money - df_main_fut['Capital'].sum() - np.abs(df_main_stk['StkCashIn']).sum() + df_main_fut['FutCashOut'].sum() + df_main_fut['FutCashIn'].sum(), 0)
            
            if df_main_stk['StockTotal'].sum() == 0: rebuy_reserve_allocr = df_main_stk['HoldMV'] / df_main_stk['HoldMV'].sum()
            else: rebuy_reserve_allocr = df_main_stk['StockTotal'] / df_main_stk['StockTotal'].sum()
            
            df_main_stk['RebuyReserve'] = np.maximum(rebuy_reserve_allocr * rebuy_reserve, df_main_stk['RebuyReserve'])
            df_main_stk['RebuyMoney'] = (df_main_stk['StockTotal'] - df_main_stk['RebuyReserve'] - df_main_stk['StkSZOut'] - df_main_stk['StkSHOut'] - np.abs(df_main_stk['NextCo'])) * (trans_type == 'hedge_market_close')
            if is_equity_prd:
                rebuy_max_money = max(df_main_stk['HoldMV'].sum() / 0.81 - df_main_stk['HoldMV'].sum() - (df_main_fut['Capital'] - df_main_fut['FutCashIn'] - df_main_fut['FutCashOut']).sum(), 0)
                if df_main_stk['RebuyMoney'].sum() > 0:
                    df_main_stk['RebuyMoney'] *= min(rebuy_max_money / df_main_stk['RebuyMoney'].sum(), 1)

            df_main_stk['SZRebuy'] = np.maximum(np.floor(
                df_main_stk['SZ'] / df_main_stk['StockTotal'] *
                df_main_stk['RebuyMoney'] / min_transfer_unit) * min_transfer_unit, 0).replace([np.inf, -np.inf], 0).fillna(0).astype('int')
            df_main_stk['SHRebuy'] = np.maximum(np.floor(
                df_main_stk['SH'] / df_main_stk['StockTotal'] *
                df_main_stk['RebuyMoney'] / min_transfer_unit) * min_transfer_unit, 0).replace([np.inf, -np.inf], 0).fillna(0).astype('int')
            df_main_stk['RebuyMoney'] = df_main_stk['SZRebuy'] + df_main_stk['SHRebuy']

            df_main_stk['CashGap'] = (df_main_stk['StkOutOrigin'] - df_main_stk['StkCashOut']).replace([np.inf, -np.inf], 0).fillna(0).astype('int')

            conlist.append(df_main_fut)
            conlist_stk.append(df_main_stk)

        df_trans_fut = pd.concat(conlist, axis=0)
        df_trans_stk = pd.concat(conlist_stk, axis=0)

        flag_rebuy = (~ df_trans_stk['production'].isin(rebuy_drop_list_extra)) & (
            df_trans_stk['production'].isin(rebuy_list_extra) |
            np.where(df_trans_stk['production'].isin(zq_product_list), whether_zq_rebuy, whether_dc_rebuy) |
            (np.where(df_trans_stk['production'].isin(zq_product_list), not zq_rebuy_forbid, not dc_rebuy_forbid) & isrebuy))

        df_trans_stk['SZRebuy'] *= flag_rebuy
        df_trans_stk['SHRebuy'] *= flag_rebuy

        return df_trans_stk, df_trans_fut

    def run_process(self, paras_dict):
        curdate = paras_dict.get('curdate', self.curdate)
        trading_day = paras_dict.get('trading_day', curdate)
        trans_type = paras_dict.get('trans_type', 'hedge_market_close')
        wechat_and_email = paras_dict.get('wechat_and_email', False)

        cfg_output_dir = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/AutoTransferCfg/{trading_day}/'
        if not os.path.exists(cfg_output_dir): os.makedirs(cfg_output_dir)

        paras_dict['cfg_output_dir'] = cfg_output_dir
        paras_dict['settle_price_fix_ratio'] = self.get_settle_price_fix_factor(paras_dict)

        df_trader_info, duplicates_trader_product = self.get_trader_config_stocks(paras_dict)
        if paras_dict.get('market_open_repay_df') is not None:  df_repay = paras_dict.get('market_open_repay_df')
        else: df_repay = self.get_daily_repay_cashout(curdate)
        
        df_repay_next = self.get_next_repay_cashout(curdate)
        df_stock_money = self.get_capital_and_trans_stocks(paras_dict)
        df_fut_trans = self.get_capital_and_trans_future(paras_dict)

        zq_product_list = list(df_stock_money[df_stock_money['Strategy'] != 'DC']['Product'].unique())

        df_stock_money = pd.merge(df_stock_money, df_trader_info, left_on='Product', right_on='production', how='left').fillna('-')
        df_stk_trans = pd.merge(df_stock_money, df_repay, on='Product', how='left').fillna(0)
        df_stk_trans = pd.merge(df_stk_trans, df_repay_next, on='Product', how='left').fillna(0)

        df_stk_trans.to_csv(f'{cfg_output_dir}{trading_day}_{trans_type}_stk_trans.csv', index=False, encoding='GBK')
        df_fut_trans.to_csv(f'{cfg_output_dir}{trading_day}_{trans_type}_fut_trans.csv', index=False, encoding='GBK')

        df_stock_trans, df_future_trans = self.allocation(df_stk_trans, df_fut_trans, zq_product_list, paras_dict, duplicates_trader_product)
        self.generate_trans_config_future(df_future_trans, paras_dict)
        self.generate_trans_config_stocks(df_stock_trans, zq_product_list, paras_dict)

        if wechat_and_email: self.upload_trans_config(paras_dict)

        self.generate_summary(df_stock_trans, paras_dict)
        
        self.check_transfer_and_send_email(df_stock_trans, df_future_trans, paras_dict)

    def check_transfer_and_send_email(self, df_stock_trans, df_future_trans, paras_dict):
        curdate = paras_dict.get('curdate', self.curdate)
        trading_day = paras_dict.get('trading_day', curdate)
        trans_type = paras_dict.get('trans_type', 'hedge_market_close')
        wechat_and_email = paras_dict.get('wechat_and_email', False)
        cfg_output_dir = paras_dict['cfg_output_dir']

        df_transfer = pd.merge(df_future_trans, df_stock_trans, on=['Product', 'Main'], how='outer').fillna(0)
        df_transfer = df_transfer.sort_values('Main', ascending=True)
        df_transfer['pwd'] = df_transfer['pwd'].apply(lambda x: '******' if len(str(x)) > 8 else x)
        df_transfer['Time'] = datetime.datetime.now().strftime('%H%M%S')
        df_transfer['产品名称'] = df_transfer['Product'].apply(lambda x: PRODUCTION_2_PRODUCT_NAME_SIMPLE.get(str(x).split('-')[0], '-'))
        df_transfer['托管'] = df_transfer['Product'].apply(lambda x: PRODUCTION_2_HOSTING.get(str(x).split('-')[0], '-'))
        df_transfer['当前M比例'] = (np.round((df_transfer['Capital'] / df_transfer['Margin'].replace(0, np.nan) * self.margin_base_ratio).fillna(0) * 100, 1).astype('str') + '%').replace('0.0%', '')
        df_transfer['目标M比例'] = (np.round(((df_transfer['Capital'] - df_transfer['FutCashIn'] - df_transfer['FutCashOut']) / df_transfer['Margin'].replace(0, np.nan) * self.margin_base_ratio).fillna(0) * 100, 1).astype('str') + '%').replace('0.0%', '')

        df_transfer['FutCashIn'] = np.abs(df_transfer['FutCashIn'])
        df_transfer['FutCashOut'] = np.abs(df_transfer['FutCashOut'])
        df_transfer['StkCashIn'] = np.abs(df_transfer['StkCashIn'])
        df_transfer['StkCashOut'] = np.abs(df_transfer['StkCashOut'])
        df_transfer['StkSZOut'] = np.abs(df_transfer['StkSZOut'])
        df_transfer['StkSHOut'] = np.abs(df_transfer['StkSHOut'])
        df_transfer['RepayMoney'] = np.abs(df_transfer['RepayMoney'])

        df_transfer = df_transfer[(
            np.abs(df_transfer['FutCashIn']) +
            np.abs(df_transfer['FutCashOut']) +
            np.abs(df_transfer['StkCashIn']) +
            np.abs(df_transfer['StkCashOut']) +
            np.abs(df_transfer['StkSZOut']) +
            np.abs(df_transfer['StkSHOut']) +
            np.abs(df_transfer['RepayMoney'])) > 0]
        df_transfer = df_transfer.rename(
            {'Account': 'HedgeAcc', 'WithdrawQuota': '期货可出', 'account': 'StockAcc', 'pwd': 'Trader',
             'FutCashIn': '期货入金', 'FutCashOut': '期货出金', 'StkCashIn': '证券入金', 'StkCashOut': '证券出金',
             'StkSZOut': '证券SZ出金', 'StkSHOut': '证券SH出金', 'RepayMoney': '赎回金额', 'BankName': '银行'}, axis='columns')

        df_transfer = self.format_result_excel_columns(
            df_transfer, format_type='W',
            columns_list=['期货可出', '期货入金', '期货出金', '证券入金', '证券出金', '证券SZ出金', '证券SH出金', '赎回金额'])
        df_transfer = df_transfer.fillna('').astype('str').replace(['0W', '0', '0.0'], '')

        def flag_stocks_trans_status(broker_, prod_, trans_mode_, stocks_out_):
            if (str(stocks_out_) == '') or (str(stocks_out_) == 'nan'):
                return ''

            if prod_ in paras_dict['non_auto_trans_products']:
                return '待手动'

            if str(trans_mode_) == '1':
                return 'ok'

            if broker_ in paras_dict['auto_trans_broker']:
                return '15:04'
            elif broker_ in paras_dict['auto_trans_broker_delay']:
                return '15:12'
            elif broker_ in paras_dict['auto_trans_broker_bank']:
                return '待调拨'
            else:
                return '待处理'

        df_transfer['证银'] = df_transfer.apply(
            lambda row: flag_stocks_trans_status(row['Broker'], row['Product'], row['trans_mode'], row['证券出金']),
            axis=1)
        df_transfer['银期'] = df_transfer.apply(
            lambda row: 'ok' if self.auto_future_acc.get(row['HedgeAcc'], False) and
                                ((row['期货出金'] != '') or (row['期货入金'] != '')) else '', axis=1)
        df_transfer['盘前时间'] = df_transfer['Colo'].apply(
            lambda x: np.sum([str(x).startswith(colo_) for colo_ in PreMarket_Start_Delay_Colo]) > 0).astype('int')
        df_transfer['盘前时间'] = df_transfer['盘前时间'].apply(lambda x: {0: '8:52', 1: '9:02'}[x])

        df_transfer = df_transfer[
            ['Product', 'Main', 'Colo', 'HedgeAcc', 'StockAcc', 'Time', 'Trader', 'Broker',
             '赎回金额', '期货出金', '证券出金', '证券入金', '期货入金', '产品名称', '托管',
             '证银', '银期', '银行', '盘前时间', '证券SZ出金', '证券SH出金', '期货可出', '当前M比例', '目标M比例']]

        trans_product_list = df_transfer['Product'].to_list()
        trans_hedge_acc_list = df_transfer['HedgeAcc'].to_list()
        trans_broker_list = df_transfer['Broker'].to_list()
        result_stocks_out_list = df_transfer['证银'].to_list()

        df_transfer = df_transfer.astype('str').style.set_properties(
            **{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]).set_table_styles([
            {'selector': 'th', 'props': [('border', '1px solid black')]},
            {'selector': 'td', 'props': [('border', '1px solid black')]}])

        def get_color_hedge_market_close(xi_, v_):
            if len(str(v_).strip()) > 0:
                if self.auto_future_acc.get(trans_hedge_acc_list[xi_], False):
                    return 'background-color: {0}'.format('#FFB2DE')
                else:
                    return 'background-color: {0}'.format('#5B9B00')
            else:
                return ''

        def get_color_hedge_market_close_broker(xi_, v_):
            if len(str(v_).strip()) > 0:
                if ((trans_broker_list[xi_] in paras_dict['auto_trans_broker'] +
                     paras_dict['auto_trans_broker_delay'] + paras_dict['auto_trans_broker_bank']) and
                        (trans_product_list[xi_] not in paras_dict['non_auto_trans_products'])):
                    if str(result_stocks_out_list[xi_]).strip() == 'ok':
                        return 'background-color: {0}'.format('#FFB2DE')
                    else:
                        return 'background-color: {0}'.format('#FFD25F')
                else:
                    return 'background-color: {0}'.format('#5B9B00')
            else:
                return ''

        df_transfer = df_transfer.apply(
            lambda x: [get_color_hedge_market_close(xi, v) for xi, v in enumerate(x)],
            axis=0, subset=['期货出金', '期货入金', 'Product']).apply(
            lambda x: ['background-color: {0}'.format('#5B9B00')
                       if str(v).strip() else '' for xi, v in enumerate(x)], axis=0, subset=['证券入金']).apply(
            lambda x: [get_color_hedge_market_close_broker(xi, v) for xi, v in enumerate(x)], axis=0,
            subset=['证券出金'])

        excel_path = f'{cfg_output_dir}{trading_day}_{trans_type}_report.xlsx'
        df_transfer.to_excel(excel_path, index=False)
        if wechat_and_email:
            wechat_bot_file(excel_path, type_api='capital-config')

        attchmentdir = []
        for summary_path in [
            excel_path,
            f'{cfg_output_dir}{trading_day}_bank_transfer_cfg_{trans_type}.xlsx',
            f'{cfg_output_dir}{trading_day}_{trans_type}_details.xlsx',
        ]:
            if os.path.exists(summary_path): attchmentdir.append(summary_path)

        if wechat_and_email:
            ase = AutoSendEmail(
                curdate=trading_day,
                content_dict={},
                subject=f'{trading_day}-{self.trans_type_2_title.get(trans_type, "当日出入金情况")} Summary',
                receivers=EMAIL_RECEIVER_LIST,
                attchmentdir=attchmentdir)
            ase.send_email()

    def run(self, **kwargs):
        trans_type = kwargs.get('trans_type', 'hedge_market_close')
        trading_day = self.nextdate if trans_type == 'hedge_market_open' else self.curdate
        wechat_and_email = kwargs.get('wechat_and_email', False)

        paras_dict = kwargs.get('paras_dict', {})
        paras_dict['curdate'] = self.curdate
        paras_dict['trading_day'] = trading_day
        paras_dict['predate'] = self.predate
        paras_dict['trans_type'] = trans_type
        paras_dict['wechat_and_email'] = wechat_and_email
        paras_dict['down_json_mode'] = kwargs.get('down_json_mode', False)

        if trans_type == 'hedge_market_close':
            paras_dict['isrebuy'] = kwargs.get('isrebuy', whether_rebuy(self.curdate))
            paras_dict['only_call_margin'] = False
            paras_dict['sz_sh_equal'] = False
            paras_dict['future_trans_active'] = True
            paras_dict['repay_trans_active'] = True
        elif trans_type == 'hedge_market_open':
            paras_dict['isrebuy'] = 0
            paras_dict['only_call_margin'] = kwargs.get('only_call_margin', True)
            paras_dict['sz_sh_equal'] = kwargs.get('sz_sh_equal', True)
            paras_dict['future_trans_active'] = kwargs.get('future_trans_active', True)
            paras_dict['repay_trans_active'] = kwargs.get('repay_trans_active', False)
        else:
            paras_dict['isrebuy'] = 0
            paras_dict['only_call_margin'] = True # 不做逆回购，期货不转出
            paras_dict['sz_sh_equal'] = False
            paras_dict['future_trans_active'] = True
            paras_dict['repay_trans_active'] = False

        paras_dict['extra_trans_infor'] = kwargs.get('extra_trans_infor', [])
        paras_dict['rebuy_test_money'] = kwargs.get('rebuy_test_money', None)

        paras_dict['open_index_df'] = kwargs.get('open_index_df', None)
        paras_dict['morning_tran_df'] = kwargs.get('morning_tran_df', None)
        paras_dict['market_open_repay_df'] = kwargs.get('market_open_repay_df', None)

        self.run_process(paras_dict)

        gtbc = GenerateTransferBrokerConfig(trading_day, trans_type=trans_type)
        gtbc.generate(wechat_and_email=wechat_and_email)

    def run_hedge_market_open(self, adj_paras_dict, **kwargs):
        down_json_mode = kwargs.get('down_json_mode', False)
        wechat_and_email = kwargs.get('wechat_and_email', True)
        trans_plan = kwargs.get('trans_plan', 'morning_margin_call')
        repay_trans_plan = kwargs.get('repay_trans_plan', 'morning_repay')
        oifix_trans_plan = kwargs.get('oifix_trans_plan', 'hedge_open_index_gap')

        repay_trans_path = f"{adj_paras_dict['path']['win']['trans_data_path']}{self.nextdate}_{repay_trans_plan}.csv"
        if os.path.exists(repay_trans_path): 
            open_repay_df = pd.read_csv(repay_trans_path)
            if open_repay_df.empty: open_repay_df = None
        else: open_repay_df = None

        trans_plan_path = f"{adj_paras_dict['path']['win']['trans_data_path']}{self.nextdate}_{trans_plan}.csv"
        if os.path.exists(trans_plan_path): 
            df_trans_open = pd.read_csv(trans_plan_path)
            if df_trans_open.empty: df_trans_open = None
        else: df_trans_open = None

        open_index_fix_path = f"{adj_paras_dict['path']['win']['trans_data_path']}{self.curdate}_{oifix_trans_plan}.csv"
        if os.path.exists(open_index_fix_path):
            df_trans_open_index_fix = pd.read_csv(open_index_fix_path)
            if not df_trans_open_index_fix.empty:
                if df_trans_open is None:
                    df_trans_open = df_trans_open_index_fix
                else:
                    df_trans_open = pd.concat([df_trans_open, df_trans_open_index_fix], axis=0)
                    df_trans_open = df_trans_open.groupby(['Product', 'Main'])[['Transfer']].sum().reset_index()

        future_trans_active = True if df_trans_open is not None else False
        repay_trans_active = True if open_repay_df is not None else False
        if (df_trans_open is not None) or (open_repay_df is not None):
            self.run(
                curdate=self.curdate,
                down_json_mode=down_json_mode,
                wechat_and_email=wechat_and_email,
                trans_type='hedge_market_open',
                paras_dict=adj_paras_dict,
                future_trans_active=future_trans_active,
                repay_trans_active=repay_trans_active,
                morning_tran_df=df_trans_open,
                market_open_repay_df=open_repay_df,
            )

    def run_hedge_market_close(self, adj_paras_dict, **kwargs):
        down_json_mode = kwargs.get('down_json_mode', False)
        wechat_and_email = kwargs.get('wechat_and_email', True)
        rebuy_test_money = kwargs.get('rebuy_test_money', None)
        self.run(
            curdate=self.curdate,
            down_json_mode=down_json_mode,
            wechat_and_email=wechat_and_email,
            trans_type='hedge_market_close',
            paras_dict=adj_paras_dict,
            rebuy_test_money=rebuy_test_money,
            open_index_df=None,
            morning_tran_df=None,
            market_open_repay_df=None,
        )


class GenerateTransferBrokerConfig():
    def __init__(self, curdate, trans_type='hedge_market_close'):
        self.curdate = curdate
        self.predate = get_predate(curdate, 1)
        self.trans_type = trans_type
        self.broker_2_template_name = {
            '招行': ['招行', 'zsyh'],
            '招商': ['招商证券', 'zszq'],
            '国泰君安': ['国君', 'gtja'],
            '广发': ['广发', 'gf'],
            '中信': ['中信', 'citic'],
            '华泰': ['华泰', 'ht'],
        }
        self.colo_account_2_account_id = ['ciccsc']

        self.path_template = f'{PLATFORM_PATH_DICT["y_path"]}产品开平记录/批量指令模板/'
        self.res_path = f'{PLATFORM_PATH_DICT["y_path"]}产品开平记录/批量指令/{curdate}/{trans_type}/'
        if not os.path.exists(self.res_path): os.makedirs(self.res_path)
        self.account_2_account_id = self.get_account_2_account_id()

    def get_account_2_account_id(self):
        path_acc = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/StockConfigAccount/'
        path_acc_cfg = f'{path_acc}{self.curdate}_stock_account_config.csv'
        if not os.path.exists(path_acc_cfg): path_acc_cfg = f'{path_acc}{self.predate}_stock_account_config.csv'

        df_trader_info = pd.read_csv(path_acc_cfg, encoding='GBK', dtype='str')
        df_trader_info['Account'] = df_trader_info['Account'].str[3:]
        df_trader_info['account_id'] = df_trader_info['account_id'].str[3:]
        df_trader_info = df_trader_info[df_trader_info['Colo'].apply(lambda x: np.sum([x.startswith(_colo) for _colo in self.colo_account_2_account_id]) > 0)]
        df_trader_info = df_trader_info[df_trader_info['account_id'] != '-']
        account_2_account_id = {account: account_id for account, account_id in zip(df_trader_info['Account'], df_trader_info['account_id'])}
        return account_2_account_id

    def get_trans_data(self, curdate, trans_type):
        trans_report_path = f'{PLATFORM_PATH_DICT["z_path"]}stock/Trading/AutoCapitalManager/AutoTransferCfg/{curdate}/{curdate}_{trans_type}_report.xlsx'
        df_trans = pd.read_excel(trans_report_path, dtype='str')[['Product', 'Main', 'HedgeAcc', 'StockAcc', '托管', '期货出金', '证券出金', '证券入金', '期货入金', '证银', '银期']]
        df_trans = df_trans.set_index(['Product', 'Main', 'HedgeAcc', 'StockAcc', '托管']).reset_index()
        df_trans['期货出金'] = df_trans['期货出金'].fillna('0W').str[:-1].astype('int')
        df_trans['证券出金'] = df_trans['证券出金'].fillna('0W').str[:-1].astype('int')
        df_trans['期货入金'] = df_trans['期货入金'].fillna('0W').str[:-1].astype('int')
        df_trans['证券入金'] = df_trans['证券入金'].fillna('0W').str[:-1].astype('int')
        df_trans['StockAcc'] = df_trans['StockAcc'].apply(lambda x: self.account_2_account_id.get(x, x))

        trans_data_dict = {}
        for broker, df_broker in df_trans.groupby('托管'):
            trans_df_list = []
            for main_prod, df in df_broker.groupby('Main'):
                df_fut_out = df[['Product', 'Main', 'HedgeAcc', '期货出金', '银期']].rename({'期货出金': 'Transfer', 'HedgeAcc': 'Account', '银期': 'Flag'}, axis='columns')
                df_fut_out['Type'] = 'future-out'
                df_stock_out = df[['Product', 'Main', 'StockAcc', '证券出金', '证银']].rename({'证券出金': 'Transfer', 'StockAcc': 'Account', '证银': 'Flag'}, axis='columns')
                df_stock_out['Type'] = 'stock-out'
                df_fut_in = df[['Product', 'Main', 'HedgeAcc', '期货入金', '银期']].rename({'期货入金': 'Transfer', 'HedgeAcc': 'Account', '银期': 'Flag'}, axis='columns')
                df_fut_in['Type'] = 'future-in'
                df_stock_in = df[['Product', 'Main', 'StockAcc', '证券入金', '证银']].rename({'证券入金': 'Transfer', 'StockAcc': 'Account', '证银': 'Flag'}, axis='columns')
                df_stock_in['Type'] = 'stock-in'
                
                df = pd.concat([df_fut_out, df_stock_out, df_fut_in, df_stock_in], axis=0)
                df['Flag'] = df['Flag'].fillna('-')
                df = df[(df['Flag'].astype('str') != 'ok') & (df['Transfer'] != 0)]
                df = df[['Product', 'Main', 'Account', 'Transfer', 'Type']].replace(Dict_ProductionName_Replace)
                df['ProdName'] = df['Main'].apply(lambda x: PRODUCTION_2_PRODUCT_NAME.get(x, ''))
                df['Transfer'] *= 1e4
                df['Account'] = df['Account'].astype('str')
                if not df.empty:
                    trans_df_list.append(df)
            
            if trans_df_list:
                trans_data_dict[broker] = pd.concat(trans_df_list, axis=0)
        
        return trans_data_dict
    
    def match_account(self, product_name, trans_dict, broker, error_list, pn2ca):
        account = pn2ca.get(product_name)
        if (account is None):
            error_list.append(f"{broker}-{trans_dict['Product']}-{trans_dict['Account']}-{trans_dict['ProdName']}")
            account = trans_dict['Account']
        else:
            account = None
            for acc in pn2ca[product_name]:
                if not str(acc): continue
                if trans_dict['Account'].startswith(acc) or trans_dict['Account'].endswith(acc) or acc.endswith(trans_dict['Account']) or acc.startswith(trans_dict['Account']):
                    account = acc
            if account is None:
                error_list.append(f"{broker}-{trans_dict['Product']}-没有找到{trans_dict['Account']}-{trans_dict['ProdName']}")
                account = trans_dict.get('Account', 'nan')
        return account, error_list
        
    def generate_config_zsyh(self, broker, template_name, trans_list, row_num):
        df = pd.read_excel(f'{self.path_template}托管户&经纪商资金账户信息.xlsx', sheet_name=f'{template_name}账户信息', dtype='str').astype('str')
        df = df.fillna('')
        product_name_2_account_broker = {prod.split('-', maxsplit=1)[-1].strip(): acc.split('/')[0].strip() for prod, acc in zip(df['项目名称'], df['托管账号/户名/开户行'])}
        product_name_2_account_list = {prod.split('-', maxsplit=1)[-1].strip(): acc.split() for prod, acc in zip(df['项目名称'], df['三方存管保证金账户'])}
        
        df_res = pd.read_excel(f'{self.path_template}【{template_name}】指令批量导入模板.xls', nrows=row_num)
        error_list = []
        for i, trans_dict in enumerate(trans_list):
            if i <= (len(df_res) - 1): df_res.iloc[i, :] = np.nan
            product_name = trans_dict['ProdName']
            account, error_list = self.match_account(product_name, trans_dict, broker, error_list, product_name_2_account_list)
            account_broker = product_name_2_account_broker.get(product_name, 'nan')

            df_res.loc[i, '付款账号'] = account_broker
            df_res.loc[i, '指令类型'] = 'YZ' if trans_dict['Type'].startswith('stock') else 'YQ'
            df_res.loc[i, '收方金额'] = trans_dict['Transfer']
            df_res.loc[i, '收方账号'] = account
            
            df_res.loc[i, '划款用途'] = '划款用途'
            
            df_res.loc[i, '划款类型'] = 10 if trans_dict['Type'].startswith('stock') else np.nan
            df_res.loc[i, '资金方向'] = {'stock-out': 19, 'stock-in': 18, 'future-out': 24, 'future-in': 23}[trans_dict['Type']]
            
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet('Sheet1')
        for row_idx, row in df_res.T.reset_index().T.reset_index(drop=True).fillna('').iterrows():
            for col_idx, value in enumerate(row):
                worksheet.write(row_idx, col_idx, value)
        workbook.save(f'{self.res_path}{self.curdate}_{broker}.xls')

        return error_list
    
    def generate_config_zszq(self, broker, template_name, trans_list, row_num):
        df = pd.read_excel(f'{self.path_template}托管户&经纪商资金账户信息.xlsx', sheet_name=f'{template_name}账户信息', dtype='str')
        df = df.fillna('')
        product_name_2_account = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['关联托管户'])}
        product_name_2_account_name = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['关联托管户户名'])}
        product_name_2_account_bank = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['关联托管户开户行'])}
        product_name_2_product_code = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['产品代码'])}
        product_name_2_future = {prod.strip(): acc.split('+')[0].strip() for prod, acc in zip(df['产品名称'], df['经纪商'])}
        account_2_broker_name = {account.strip(): broker_name.strip() for account, broker_name in zip(df['资金账户'], df['经纪商'])}
        
        workbook = openpyxl.load_workbook(f'{self.path_template}【{template_name}】指令批量导入模板.xlsx')
        sheet = workbook.active
        error_list = []
        start_row = 4
        for _ in range(10):
            sheet.delete_rows(start_row)    
        for trans_dict in trans_list:
            if account_2_broker_name.get(trans_dict['Account']) is None: 
                error_list.append(f"{broker}-{trans_dict['Product']}-{trans_dict['Account']}-{trans_dict['ProdName']}")
            
            account_broker = product_name_2_account.get(trans_dict['ProdName'], 'nan')
            account_name = product_name_2_account_name.get(trans_dict['ProdName'], 'nan')
            account_bank = product_name_2_account_bank.get(trans_dict['ProdName'], 'nan')
            product_code = product_name_2_product_code.get(trans_dict['ProdName'], 'nan')
            broker_name = account_2_broker_name.get(trans_dict['Account'], 'nan')
            
            sheet[f'A{start_row}'] = product_code
            
            sheet[f'B{start_row}'] = np.nan
            sheet[f'C{start_row}'] = np.nan
            sheet[f'D{start_row}'] = broker_name
            sheet[f'E{start_row}'] = trans_dict['Account']
            
            sheet[f'F{start_row}'] = account_bank
            sheet[f'G{start_row}'] = {'stock-out': '证转银', 'stock-in': '银转证', 'future-out': '期转银', 'future-in': '银转期'}[trans_dict['Type']]
            
            sheet[f'H{start_row}'] = round(trans_dict['Transfer'], 2)
            sheet[f'H{start_row}'].number_format = '0.00'
            
            sheet[f'I{start_row}'] = datetime.datetime.strptime(self.curdate, '%Y%m%d')
            sheet[f'I{start_row}'].number_format = 'yyyy/m/d'
            sheet[f'J{start_row}'] = '摘要'
            
            start_row += 1
        workbook.save(f'{self.res_path}{self.curdate}_{broker}.xlsx')
        return error_list
    
    def generate_config_gtja(self, broker, template_name, trans_list, row_num):
        df = pd.read_excel(f'{self.path_template}托管户&经纪商资金账户信息.xlsx', sheet_name=f'{template_name}账户信息', dtype='str', header=1)
        df = df.fillna('')
        
        product_name_2_product_code = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['产品代码'])}
        product_name_2_product_name_full = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['账户名称'])}
        product_name_2_account_bank = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['开户行'])}
        product_name_2_account = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['账号'])}
        account_2_broker_name = {account.strip(): broker_name.strip() for account, broker_name in zip(df['资金账户号码'], df['经纪商名称'])}
        product_name_2_capital_account = {prod.strip(): df_prod['资金账户号码'].to_list() for prod, df_prod in df.groupby('产品名称')}

        workbook = openpyxl.load_workbook(f'{self.path_template}【{template_name}】指令批量导入模板.xlsx')
        sheet = workbook.active
        error_list = []
        start_row = 4
        for _ in range(10):
            sheet.delete_rows(start_row)    
        for trans_dict in trans_list:
            product_name = trans_dict['ProdName']
            
            account, error_list = self.match_account(product_name, trans_dict, broker, error_list, product_name_2_capital_account)

            product_code = product_name_2_product_code.get(product_name, 'nan')
            account_broker = product_name_2_account.get(product_name, 'nan')

            broker_name = account_2_broker_name.get(account, 'nan')
            account_bank = product_name_2_account_bank.get(product_name, 'nan')
            product_name_full = product_name_2_product_name_full.get(product_name, 'nan')

            sheet[f'A{start_row}'] = product_code
            sheet[f'B{start_row}'] = None
            sheet[f'C{start_row}'] = {'stock-out': '证转银', 'stock-in': '银转证', 'future-out': '期转银', 'future-in': '银转期'}[trans_dict['Type']]
        
            sheet[f'D{start_row}'] = account if trans_dict['Type'].endswith('out') else account_broker
            sheet[f'E{start_row}'] = account_broker if trans_dict['Type'].endswith('out') else account
            
            sheet[f'F{start_row}'] = product_name_full if trans_dict['Type'].endswith('out') else product_name
            sheet[f'G{start_row}'] = account_bank if trans_dict['Type'].endswith('out') else broker_name
            
            sheet[f'H{start_row}'] = datetime.datetime.strptime(self.curdate, '%Y%m%d')
            sheet[f'H{start_row}'].number_format = 'yyyy/m/d'
            sheet[f'I{start_row}'] = round(trans_dict['Transfer'], 2)
            sheet[f'I{start_row}'].number_format = '0.00'
            
            start_row += 1
        workbook.save(f'{self.res_path}{self.curdate}_{broker}.xlsx')
        return error_list
    
    def generate_config_gf(self, broker, template_name, trans_list, row_num):
        df = pd.read_excel(f'{self.path_template}托管户&经纪商资金账户信息.xlsx', sheet_name=f'{template_name}账户信息', dtype='str')
        df = df.fillna('')
        df = df.rename({col: col.strip() for col in df.columns.to_list()}, axis='columns')
        
        product_name_2_product_code = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['产品代码'])}
        product_name_2_account_bank = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['关联托管户开户行'])}
        product_name_2_account = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['关联托管户'])}
        
        workbook = openpyxl.load_workbook(f'{self.path_template}【{template_name}】指令批量导入模板.xlsx')
        sheet = workbook.active
        error_list = []
        start_row = 2
        for _ in range(10):
            sheet.delete_rows(start_row)    
        for trans_dict in trans_list:
            product_name = trans_dict['ProdName']
            if product_name_2_account.get(product_name) is None: error_list.append(f"{broker}-{trans_dict['Product']}-{trans_dict['ProdName']}")
            
            product_code = product_name_2_product_code.get(product_name, 'nan')
            account_broker = product_name_2_account.get(product_name, 'nan')
            account_bank = product_name_2_account_bank.get(product_name, 'nan')
            
            sheet[f'A{start_row}'] = {'stock-out': '证转银', 'stock-in': '银转证', 'future-out': '期转银', 'future-in': '银转期'}[trans_dict['Type']]            
            sheet[f'B{start_row}'] = round(trans_dict['Transfer'], 2)
            sheet[f'B{start_row}'].number_format = '0.00'
            
            sheet[f'C{start_row}'] = product_name
            sheet[f'D{start_row}'] = trans_dict['Account']
            sheet[f'E{start_row}'] = account_broker
            
            sheet[f'F{start_row}'] = datetime.datetime.strptime(self.curdate, '%Y%m%d')
            sheet[f'F{start_row}'].number_format = 'yyyy-m-d'
            sheet[f'G{start_row}'] = '摘要'

            start_row += 1
        workbook.save(f'{self.res_path}{self.curdate}_{broker}.xlsx')
        return error_list
    
    def generate_config_citic(self, broker, template_name, trans_list, row_num):
        df = pd.read_excel(f'{self.path_template}托管户&经纪商资金账户信息.xlsx', sheet_name=f'{template_name}账户信息', dtype='str')
        df = df.fillna('')
        df = df[df['账户类型'] == '托管账户']
        
        product_name_2_account = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['账号'])}
        product_name_2_account_name = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['账户名称'])}
        product_name_2_account_bank = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['开户行/经纪商'])}
        product_name_2_product_code = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['产品代码'])}
        product_name_2_more_pay_num = {prod.strip(): acc.split('+')[0].strip() for prod, acc in zip(df['产品名称'], df['大额支付号']) if str(acc) not in ['nan', '']}
        
        
        workbook = openpyxl.load_workbook(f'{self.path_template}【{template_name}】指令批量导入模板.xlsx')
        sheet = workbook.active
        error_list = []
        start_row = 4
        
        for _ in range(10):
            sheet.delete_rows(start_row)    
        for trans_dict in trans_list:
            prod_name = str(trans_dict['ProdName']).strip()
            if product_name_2_account.get(prod_name) is None:
                error_list.append(f"{broker}-{trans_dict['Product']}-{trans_dict['Account']}-{prod_name}")
            
            account_broker = product_name_2_account.get(prod_name, 'nan')
            account_name = product_name_2_account_name.get(prod_name, 'nan')
            account_bank = product_name_2_account_bank.get(prod_name, 'nan')
            product_code = product_name_2_product_code.get(prod_name, 'nan')
            more_pay_num = product_name_2_more_pay_num.get(prod_name, 'nan')
            
            sheet[f'A{start_row}'] = product_code
            
            sheet[f'C{start_row}'] = trans_dict['Account']
            sheet[f'D{start_row}'] = account_bank
            
            sheet[f'E{start_row}'] = {'stock-out': '证银转账', 'stock-in': '银证转账', 'future-out': '期银转账', 'future-in': '银期转账'}[trans_dict['Type']]
            sheet[f'F{start_row}'] = round(trans_dict['Transfer'], 2)
            sheet[f'F{start_row}'].number_format = '0.00'
            # sheet[f'G{start_row}'] = datetime.datetime.strptime(self.curdate, '%Y%m%d')
            # sheet[f'G{start_row}'].number_format = 'yyyy/m/d'
            start_row += 1
        workbook.save(f'{self.res_path}{self.curdate}_{broker}.xlsx')
        return error_list
    
    def generate_config_ht(self, broker, template_name, trans_list, row_num):
        df = pd.read_excel(f'{self.path_template}托管户&经纪商资金账户信息.xlsx', sheet_name=f'{template_name}账户信息', dtype='str', header=[0, 1])
        df = df.fillna('')
        df_js = df['资金账户']
        df_jz = df['托管户'].rename({'账户名称': '托管户账户名称', '账户号': '托管户账户号'}, axis='columns')
        df = pd.concat([df_js, df_jz], axis=1)
        
        product_name_2_account_broker = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['托管户账户号'])}
        product_name_2_account_name_broker = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['托管户账户名称'])}
        product_name_2_product_code = {prod.strip(): acc.strip() for prod, acc in zip(df['产品名称'], df['产品代码'])}
        account_2_account_name = {account.strip(): broker_name.strip() for account, broker_name in zip(df['账户号'], df['账户名称'])}
        
        workbook = openpyxl.load_workbook(f'{self.path_template}【{template_name}】指令批量导入模板.xlsx')
        sheet = workbook.active
        error_list = []
        start_row = 8
        for _ in range(10): sheet.delete_rows(start_row)    
        
        squence = 1
        for trans_dict in trans_list:
            if account_2_account_name.get(trans_dict['Account']) is None: error_list.append(f"{broker}-{trans_dict['Product']}-{trans_dict['Account']}-{trans_dict['ProdName']}")
            
            product_code = product_name_2_product_code.get(trans_dict['ProdName'], 'nan')
            account_name_broker = product_name_2_account_name_broker.get(trans_dict['ProdName'], 'nan')
            account_name = account_2_account_name.get(trans_dict['Account'], 'nan')
            account_broker = product_name_2_account_broker.get(trans_dict['ProdName'], 'nan')
            account = trans_dict['Account']
            
            sheet[f'A{start_row}'] = squence
            sheet[f'B{start_row}'] = datetime.datetime.strptime(self.curdate, '%Y%m%d')
            sheet[f'B{start_row}'].number_format = 'yyyy/m/d'
            sheet[f'C{start_row}'] = product_code
            sheet[f'D{start_row}'] = {'stock-out': '证转银', 'stock-in': '银转证', 'future-out': '期转银', 'future-in': '银转期'}[trans_dict['Type']]
            sheet[f'E{start_row}'] = round(trans_dict['Transfer'], 2)
            sheet[f'E{start_row}'].number_format = '0.00'
            
            sheet[f'F{start_row}'] = account_name if trans_dict['Type'].endswith('out') else account_name_broker
            sheet[f'G{start_row}'] = account if trans_dict['Type'].endswith('out') else account_broker
            sheet[f'H{start_row}'] = account_name if trans_dict['Type'].endswith('in') else account_name_broker
            sheet[f'I{start_row}'] = account if trans_dict['Type'].endswith('in') else account_broker
            
            start_row += 1
            squence += 1
        workbook.save(f'{self.res_path}{self.curdate}_{broker}.xlsx')
        return error_list
    
    def generate(self, wechat_and_email=False):
        dict_trans_data = self.get_trans_data(self.curdate, self.trans_type)
        
        error_matched_list, non_auto_list, run_error = [], [], []
        for broker, df_trans in dict_trans_data.items():
            print(broker)
            print(df_trans)
            if self.broker_2_template_name.get(broker) is None:
                non_auto_list.append(broker)
                continue
            
            template_name, func_name = self.broker_2_template_name[broker]
            gene_func = f'self.generate_config_{func_name}'
        
            try:
                error_matched_list += eval(gene_func)(broker, template_name, df_trans.to_dict(orient='records'), len(df_trans))
            except:
                run_error.append(broker)
                check_msg = traceback.format_exc()
                print(check_msg)
                wechat_bot_msg_check(check_msg)
                
        msg_error_matched = ','.join(error_matched_list)
        msg_non_auto = ','.join(non_auto_list)
        msg_run_error = ','.join(run_error)

        if wechat_and_email:
            msg = f'# {self.curdate}-批量指令文件:\n'
            msg += f"## 无自动指令: \n\t{msg_non_auto}\n"
            msg += f"## 匹配失败:\n\t{msg_error_matched}\n"
            msg += f"## 运行失败:\n\t{msg_run_error}\n"
            print(msg)
            # wechat_bot_markdown(msg, type_api='check')
            wechat_bot_markdown(msg, type_api='capital-config')


class GenerateChangeParasConfig(GetProductionInformation):
    def __init__(self, curdate):
        super().__init__()

        self.curdate = curdate
        self.cfg_output_dir = f'{PLATFORM_PATH_DICT["z_path"]}Trading/AutoCapitalManager/AutoChangeCfgParas/{curdate}/'
        if not os.path.exists(self.cfg_output_dir): os.makedirs(self.cfg_output_dir)

        self.index_2_prd_lst, self.class_2_prd_lst, self.colo_2_prd_lst = self.get_product_trading_class_info()

    def output_check_info(self):
        msg = f'# {self.curdate}-CfgParasCheck:'
        for exch in ['sz', 'sh']:
            dict_cfg = read_cfg_config(self.cfg_output_dir + f'{self.curdate}_change_cfg_paras_{exch}.cfg')
            df_cfg = pd.DataFrame(dict_cfg).astype('str').replace('-', np.nan).dropna(axis=1, how='all').fillna('-')
            key_columns = df_cfg.columns.to_list()[1:]
            for key_list, df in df_cfg.groupby(key_columns):
                if isinstance(key_list, str): key_list = [key_list]
                product_list = sorted(df['exec_prd_list'].to_list())
                msg += f'\n### {exch.upper()}[{",".join(key_columns)}={",".join(key_list)}({len(product_list)})]:\n\t{",".join(product_list)}'

        wechat_bot_markdown(msg, type_api='check')

    def generate_change_cfg(self, paras_list):
        infor_list = []
        for paras_dict in paras_list:
            target_prod_list = self.match_product_list_with_class_info(
                self.curdate, {self.curdate: paras_dict}, self.class_2_prd_lst, paras_dict.get('add_priority', True))
            cmd_paras_dict = paras_dict.get('cmd_paras_dict', {})
            for exch in paras_dict.get('exch_list', ['sz', 'sh']):
                dict_chg = {
                    'exec_prd_list': target_prod_list,
                    'exec_cmd': paras_dict['cmd'],
                    'exec_exch': exch.lower(),
                }
                for cmd_paras, value in cmd_paras_dict.items():
                    dict_chg[cmd_paras] = str(value)
                infor_list.append(pd.DataFrame(dict_chg))

        df_chg_cfg = pd.concat(infor_list, axis=0).fillna('-')
        for exch, df_exch in df_chg_cfg.groupby('exec_exch'):
            generate_cfg_config(
                dict_info=df_exch.drop('exec_exch', axis=1).to_dict(orient='list'),
                file_name=f'{self.curdate}_change_cfg_paras_{exch}.cfg',
                cfg_output_dir=self.cfg_output_dir)

        self.output_check_info()

    def upload_change_cfg(self):
        msg_ret = f'{self.curdate}-CfgParasUpload:'
        for cmd_scp in [
            f'ssh jumper "[ ! -d ~/rtchg/bank_trans/{self.curdate}/ ] && mkdir ~/rtchg/bank_trans/{self.curdate}/"',
            f'"C:/Program Files/Git/usr/bin/scp.exe" -r {self.cfg_output_dir}{self.curdate}_change_cfg_paras_*.cfg jumper:~/rtchg/bank_trans/{self.curdate}/',
        ]:
            status, log = subprocess.getstatusoutput(cmd_scp)
            msg_ret += f'\n\tstatus={status}\n\tlog={log}'

        wechat_bot_msg_check(msg_ret)


class POBGenerate():
    def __init__(self) -> None:
        pass

    def generate_test_quota_average_weight(self, curdate, product, holdmv, code_num=60):
        df_price = get_price(curdate, curdate).reset_index()[['SecuCode', 'ClosePrice']]
        print(production_2_index(product) + production_2_strategy(product))

        weight_df = get_alpha_quota_bar(curdate, product, 8).rename({"Volume": 'Ti8_Weight'}, axis='columns')
        weight_df = pd.merge(weight_df, df_price, on='SecuCode', how='left')
        weight_df['Ti8_Weight'] *= weight_df['ClosePrice']
        weight_df['Ti8_Weight'] /= weight_df['Ti8_Weight'].sum()
        weight_df = weight_df[['SecuCode', 'Ti8_Weight', 'ClosePrice']]

        weight_df['Exchange'] = weight_df['SecuCode'].apply(lambda x: expend_market(x).split('.')[1])
        conlist = []
        for exch, weight_df_exch in weight_df.groupby('Exchange'):
            weight_df_exch = weight_df_exch.head(int(code_num / 2))
            weight_df_exch['Weight'] = 1 / int(code_num / 2)
            conlist.append(weight_df_exch)
        weight_df = pd.concat(conlist, axis=0)

        weight_df['Volume'] = weight_df['Weight'] * holdmv / 2 * 10000 / weight_df['ClosePrice']
        weight_df['POB'] = weight_df['Volume'].apply(lambda x: int(round(x / 200) * 200)) * weight_df['SecuCode'].apply(
            lambda x: x[:3] == '688') + weight_df['Volume'].apply(lambda x: int(round(x / 100) * 100)) * \
                        weight_df['SecuCode'].apply(lambda x: x[:3] != '688')
        weight_df['SecuCode'] = weight_df['SecuCode'].apply(lambda x: expend_market(x, 3))
        weight_df['POBValues'] = weight_df['POB'] * weight_df['ClosePrice']
        weight_df['tmp'] = 0
        hprint(weight_df)

        real_holdmv = str(round(weight_df['POBValues'].sum() / 10000, 1))
        sh_real_holdmv = str(round(weight_df[weight_df['SecuCode'].str[0] == '6']['POBValues'].sum() / 10000, 1))
        sz_real_holdmv = str(round(weight_df[weight_df['SecuCode'].str[0] != '6']['POBValues'].sum() / 10000, 1))
        print(f'{code_num}只票, 市值为：{real_holdmv}W, SH市值为 {sh_real_holdmv}W, SZ市值为 {sz_real_holdmv}W')

        weight_df = weight_df[['SecuCode', 'POB', 'tmp']]
        hprint(weight_df)
        path_pob = f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/'
        if not os.path.exists(path_pob):
            os.makedirs(path_pob)
        path_pob_file = f'{path_pob}POB-AlphaT0-{holdmv}W-{real_holdmv}W-CodeNum-{code_num}-{curdate}-quota.csv'
        print(path_pob_file)
        weight_df.to_csv(path_pob_file, header=False, index=False)
        tsl = TransferServerLocal(
            ip='139.196.57.140', port=22, username='jumper', password='jump2cfi888')
        tsl.upload_file(
            local_path=path_pob_file,
            server_path=f'/home/jumper/pob_files/{product}-POB-{curdate}.csv')

    def generate_pob_use_position(self, curdate, Production, short_value, fraction_share=True, long_mode=False):
        predate = get_predate(curdate, 1)
        tsl = TransferServerLocal()
        if not long_mode:
            local_path = f'{LOG_TEMP_PATH}{curdate}_position-{Production}.txt'
            try:
                try:
                    tsl.download_file(server_path=f'/home/jumper/{curdate}_position-{Production}.txt', local_path=local_path)
                    print(f'/home/jumper/{curdate}_position-{Production}.txt')
                except:
                    tsl.download_file(server_path=f'/home/jumper/post-trade/{curdate}_position-{Production}.txt', local_path=local_path)
                    print(f'/home/jumper/post-trade/{curdate}_position-{Production}.txt')
            except:
                tsl.download_file(server_path=f'/home/jumper/rtchg/{curdate}_position-{Production}.txt', local_path=local_path)
                print(f'/home/jumper/rtchg/{curdate}_position-{Production}.txt')
            # tsl.download_file(server_path=f'/home/jumper/rtchg/{curdate}_position-{Production}.txt',
            #                   local_path=local_path)
            df_pos = pd.read_table(f'{LOG_TEMP_PATH}/{curdate}_position-{Production}.txt', sep=',', header=None)
            df_pos.columns = ['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume']
            df_pos['SecuCode'] = df_pos['SecuCode'].apply(lambda x: expand_stockcode(x))
        else:
            df_pos = get_position(curdate, Production)
            
        if long_mode:
            df_pos['PreCloseVolume'] = df_pos['Volume']
            df_pos['OpenVolume'] = 0

        df_pos['yShort'] = df_pos['Volume'] - df_pos['OpenVolume']

        df_price = get_price(predate, predate).reset_index()[['SecuCode', 'ClosePrice']]

        price_dict = {code: price for code, price in df_price[['SecuCode', 'ClosePrice']].values}
        holdmv_pre = sum([vol * price_dict[code] for code, vol in df_pos[['SecuCode', 'PreCloseVolume']].values])
        holdmv_volume = sum([vol * price_dict[code] for code, vol in df_pos[['SecuCode', 'Volume']].values])
        holdmv_diff = holdmv_volume - holdmv_pre
        holdmv_pre = int(round(holdmv_pre / 10000))
        holdmv_volume = int(round(holdmv_volume / 10000))
        holdmv_diff = int(round(holdmv_diff / 10000))
        print(f"{Production}: 昨收市值：{holdmv_pre}W, 可卖市值：{holdmv_volume}W, 市值差: {holdmv_diff}W")

        df_merge = pd.merge(df_pos, df_price, on='SecuCode', how='left').reset_index(drop=True)
        df_merge['SecuCode'] = df_merge.apply(lambda row: row['SecuCode'] + '.' + row['Exchange'], axis=1)

        df_merge = df_merge[['SecuCode', 'PreCloseVolume', 'yShort', 'ClosePrice']].rename(
            {'PreCloseVolume': 'PreVolume', 'ClosePrice': 'Price'}, axis='columns')
        holdmv = (df_merge['PreVolume'] * df_merge['Price']).sum()

        if isinstance(short_value, int) or isinstance(short_value, float):
            if isinstance(short_value, float):
                short_ratio = short_value
                short_value *= 10000
                short_value = round(short_ratio * holdmv / 10000) * 10000
            else:
                short_value *= 10000
                short_ratio = short_value / holdmv

            df_merge['POB'] = df_merge['PreVolume'].apply(
                lambda row: int(round(row * short_ratio / 100) * 100))

            df_merge['POB'] = df_merge.apply(
                lambda row: row['POB'] if row['POB'] <= row['yShort']
                else int(np.floor(row['yShort'] / 100) * 100), axis=1)

            df_merge['POB'] = df_merge.apply(
                lambda row: 0
                if row['SecuCode'][:3] == '688' and row['POB'] == 100 and row['yShort'] < 200
                else row['POB'], axis=1)

            pob_mv = (df_merge['POB'] * df_merge['Price']).sum()
            
            init_direction, direction_not_change = pob_mv < short_value, True
            price_dict = {code: price for code, price in df_merge[['SecuCode', 'Price']].values}
            df_merge = df_merge.set_index('SecuCode')
            code_list = df_merge[df_merge['POB'] < df_merge['yShort']].index.to_list()

            while abs(pob_mv / short_value - 1) > 0.001 and direction_not_change:
                rand_code = random.choice(code_list)
                if rand_code[:3] == '688':
                    if pob_mv < short_value:
                        if df_merge.loc[rand_code, 'POB'] == 0 and \
                                df_merge.loc[rand_code, 'POB'] + 200 <= df_merge.loc[rand_code, 'yShort']:
                            df_merge.loc[rand_code, 'POB'] += 200
                            pob_mv += 200 * price_dict[rand_code]
                        elif df_merge.loc[rand_code, 'POB'] != 0 and \
                                df_merge.loc[rand_code, 'POB'] + 100 <= df_merge.loc[rand_code, 'yShort']:
                            df_merge.loc[rand_code, 'POB'] += 100
                            pob_mv += 100 * price_dict[rand_code]
                    else:
                        if df_merge.loc[rand_code, 'POB'] == 200:
                            df_merge.loc[rand_code, 'POB'] -= 200
                            pob_mv -= 200 * price_dict[rand_code]
                        elif 300 >= df_merge.loc[rand_code, 'POB'] > 200:
                            df_merge.loc[rand_code, 'POB'] = 200
                            pob_mv -= (df_merge.loc[rand_code, 'POB'] - 200) * price_dict[rand_code]
                        elif df_merge.loc[rand_code, 'POB'] > 300:
                            df_merge.loc[rand_code, 'POB'] -= 100
                            pob_mv -= 100 * price_dict[rand_code]
                else:
                    if pob_mv < short_value:
                        if df_merge.loc[rand_code, 'POB'] + 100 <= df_merge.loc[rand_code, 'yShort']:
                            df_merge.loc[rand_code, 'POB'] += 100
                            pob_mv += 100 * price_dict[rand_code]
                    else:
                        if df_merge.loc[rand_code, 'POB'] >= 100:
                            df_merge.loc[rand_code, 'POB'] -= 100
                            pob_mv -= 100 * price_dict[rand_code]
                direction_not_change = init_direction == (pob_mv < short_value)

            df_merge['POB'] = np.minimum(df_merge['POB'], df_merge['yShort'])
            if not long_mode:
                df_merge['POB'] *= -1
            df_merge['tmp'] = 0
            
            pob_mv = (df_merge['POB'] * df_merge['Price']).sum()
            pob_mv_sh = (df_merge.index.str.startswith('6') * df_merge['POB'] * df_merge['Price']).sum()
            pob_mv_sz = ((~ df_merge.index.str.startswith('6')) * df_merge['POB'] * df_merge['Price']).sum()
            
            df_merge = df_merge.reset_index()
            if fraction_share:
                df_merge = df_merge[~(((np.abs(df_merge['POB']) < 200) & (df_merge['SecuCode'].str[:3] == '688')) |
                                    ((np.abs(df_merge['POB']) < 100) & (df_merge['SecuCode'].str[:3] != '688')))]
            df_merge = df_merge.reset_index()[['SecuCode', 'POB', 'tmp']]
            if not long_mode:
                df_merge = df_merge[df_merge['POB'] < 0]
            print(f'ratio: {round(short_ratio * 100, 1)}%', f'pob_mv: {pob_mv // 10000}W', f'pob_mv_sh: {pob_mv_sh // 10000}W', f'pob_mv_sz: {pob_mv_sz // 10000}W', f'holdmv: {holdmv // 10000}W')

            hprint(df_merge)
            if not os.path.exists(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/'):
                os.makedirs(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/')
            pob_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/POB-AlphaT0-{Production}-{curdate}-Short_-{short_value // 10000}w.csv'
            df_merge.to_csv(pob_path, header=None, index=None)
        elif isinstance(short_value, str):
            df_merge = df_merge[['SecuCode', 'yShort', 'Price']]
            df_merge['tmp'] = 0
            # df_merge['yShort'] = df_merge['yShort'].apply(lambda x: - (x // 100) * 100)
            if not long_mode:
                df_merge['yShort'] *= -1
                df_merge = df_merge[df_merge['yShort'] < 0]
            if fraction_share:
                df_merge = df_merge[~(((np.abs(df_merge['yShort']) < 200) & (df_merge['SecuCode'].str[:3] == '688')) |
                                    ((np.abs(df_merge['yShort']) < 100) & (df_merge['SecuCode'].str[:3] != '688')))]
            pob_mv = sum((df_merge['yShort'] * df_merge['Price']).values) // 10000
            print(f'ratio: 100%', f'pob_mv: {pob_mv}W')
            df_merge = df_merge[['SecuCode', 'yShort', 'tmp']]
            hprint(df_merge)
            if not os.path.exists(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/'):
                os.makedirs(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/')
            pob_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/POB-AlphaT0-{Production}-{curdate}-Short_all_{pob_mv}w.csv'
            df_merge.to_csv(pob_path, header=None, index=None)
        else:
            assert False, '参数错误请输入平仓市值或者 all 全平！'

        tsl = TransferServerLocal(
            ip='139.196.57.140', port=22, username='jumper', password='jump2cfi888')
        tsl.upload_file(
            local_path=pob_path,
            server_path=f'/home/jumper/pob_files/{Production}-POB-{curdate}.csv')

    def generate_pob_use_position_short_all(self, curdate, Production):
        predate = get_predate(curdate, 1)
        tsl = TransferServerLocal()

        local_path = f'{LOG_TEMP_PATH}{curdate}_position-{Production}.txt'
        try:
            try:
                tsl.download_file(server_path=f'/home/jumper/{curdate}_position-{Production}.txt', local_path=local_path)
                print(f'/home/jumper/{curdate}_position-{Production}.txt')
            except:
                tsl.download_file(server_path=f'/home/jumper/post-trade/{curdate}_position-{Production}.txt', local_path=local_path)
                print(f'/home/jumper/post-trade/{curdate}_position-{Production}.txt')
        except:
            tsl.download_file(server_path=f'/home/jumper/rtchg/{curdate}_position-{Production}.txt', local_path=local_path)
            print(f'/home/jumper/rtchg/{curdate}_position-{Production}.txt')

        df_pos = pd.read_table(f'{LOG_TEMP_PATH}/{curdate}_position-{Production}.txt', sep=',', header=None)
        df_pos.columns = ['SecuCode', 'Exchange', 'PreCloseVolume', 'OpenVolume', 'Volume']
        df_pos['SecuCode'] = df_pos['SecuCode'].apply(lambda x: expand_stockcode(x))

        df_price = get_price(predate, predate).reset_index()[['SecuCode', 'ClosePrice']].rename({'ClosePrice': 'Price'}, axis='columns')

        df_merge = pd.merge(df_pos, df_price, on='SecuCode', how='left').reset_index(drop=True)
        df_merge['SecuCode'] = df_merge.apply(lambda row: row['SecuCode'] + '.' + row['Exchange'], axis=1)
        df_merge['Volume'] *= -1
        df_merge = df_merge[df_merge['Volume'] < 0]
        df_merge = df_merge[['SecuCode', 'Volume', 'Price']]
        df_merge['tmp'] = 0

        pob_mv = sum((df_merge['Volume'] * df_merge['Price']).values) // 10000
        print(f'pob_mv: {pob_mv}W')
        df_merge = df_merge[['SecuCode', 'Volume', 'tmp']]
        hprint(df_merge)
        if not os.path.exists(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/'): os.makedirs(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/')
        pob_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/POB-AlphaT0-{Production}-{curdate}-Short_all_{pob_mv}w.csv'
        df_merge.to_csv(pob_path, header=None, index=None)

        tsl = TransferServerLocal(ip='139.196.57.140', port=22, username='jumper', password='jump2cfi888')
        tsl.upload_file(local_path=pob_path, server_path=f'/home/jumper/pob_files/{Production}-POB-{curdate}.csv')

    def generate_pob_use_pob_and_position_diff(self, curdate, product, pob_file='POB-AlphaT0-YPA-20221130-quota', mode='long_pob'):
        df_pob = pd.read_csv(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/{pob_file}.csv', header=None)
        df_pob.columns = ['SecuCode', 'Quota', 'tmp']
        local_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/{curdate}_position-{product}.txt'

        tsl = TransferServerLocal()
        try:
            try:
                tsl.download_file(server_path=f'/home/jumper/{curdate}_position-{product}.txt', local_path=local_path)
                print(f'/home/jumper/{curdate}_position-{product}.txt')
            except:
                tsl.download_file(server_path=f'/home/jumper/post-trade/{curdate}_position-{product}.txt', local_path=local_path)
                print(f'/home/jumper/post-trade/{curdate}_position-{product}.txt')
        except:
            tsl.download_file(server_path=f'/home/jumper/rtchg/{curdate}_position-{product}.txt', local_path=local_path)
            print(f'/home/jumper/rtchg/{curdate}_position-{product}.txt')

        df_position = get_position(curdate, product, local_path)
        df_position = df_position[['SecuCode', 'Volume']]
        df_position['SecuCode'] = df_position['SecuCode'].apply(lambda x: expend_market(x, 3))
        df_pob_sec = pd.merge(df_pob, df_position, on='SecuCode', how='outer').fillna(0)
        df_pob_sec['POB'] = df_pob_sec['Quota'] - df_pob_sec['Volume']
        df_pob_sec['Quota'] = df_pob_sec['Quota'].astype('int')
        df_pob_sec['POB'] = df_pob_sec['POB'].astype('int')
        df_pob_sec['tmp'] = df_pob_sec['tmp'].astype('int')
        df_pob_sec = df_pob_sec[['SecuCode', 'POB', 'tmp', 'Quota', 'Volume']]

        df_pob_sec.to_csv(
            f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/second-{pob_file}-merged-all.csv',
            header=False, index=False)

        hprint(df_pob_sec)
        if mode == 'long_pob':
            df_pob_sec = df_pob_sec[df_pob_sec['POB'] > 100]
            df_pob_sec = df_pob_sec[df_pob_sec.apply(
                lambda row: row['POB'] > 200 if row['SecuCode'].startswith('688') else row['POB'] > 100, axis=1)]
        else:
            df_pob_sec = df_pob_sec[df_pob_sec['POB'] != 0]
            df_pob_sec = df_pob_sec[df_pob_sec.apply(
                lambda row: (row['POB'] > 200 if row['SecuCode'].startswith('688') else row['POB'] > 100)
                if (row['POB'] > 0)
                else ((row['POB'] + row['Volume'] == 0)
                    if (row['POB'] > -200
                        if row['SecuCode'].startswith('688')
                        else row['POB'] > -100) else True), axis=1)]
        df_pob_sec.to_csv(
            f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/second-{pob_file}-merged.csv',
            header=False, index=False)
        df_pob_sec = df_pob_sec[['SecuCode', 'POB', 'tmp']]
        hprint(df_pob_sec)
        df_pob_sec.to_csv(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/second-{pob_file}.csv', header=False, index=False)

        predate = get_predate(curdate, 1)
        df_price = get_price(predate, predate).reset_index()
        df_pob_sec['SecuCode'] = df_pob_sec['SecuCode'].apply(lambda x: x.split('.')[0])
        df_pob_sec = pd.merge(df_pob_sec, df_price, on='SecuCode', how='left')
        df_pob_sec['Value'] = df_pob_sec['POB'] * df_pob_sec['ClosePrice']
        sz_value = df_pob_sec[df_pob_sec['SecuCode'].str[0] != '6']['Value'].sum()
        sh_value = df_pob_sec[df_pob_sec['SecuCode'].str[0] == '6']['Value'].sum()
        total_value = np.abs(df_pob_sec['Value']).sum()
        print(f'SZ: {sz_value}, SH: {sh_value} | {total_value}')

        tsl = TransferServerLocal(
            ip='139.196.57.140', port=22, username='jumper', password='jump2cfi888')
        tsl.upload_file(
            local_path=f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/second-{pob_file}.csv',
            server_path=f'/home/jumper/pob_files/{product}-POB-{curdate}.csv')

    def generate_pob_use_destpos_and_position_diff(self, curdate, product, quota_product, pre_pos_mode=False):
        """
        if quota_product is not None: 筛选 curdate 最新的 quota 为目标
        else: 使用 pob_file 为目标

        if pre_pos_mode: 筛选 predate 日的收盘仓位
        else: 筛选 curdate 日的收盘仓位
        """
        predate = get_predate(curdate, 1)
        if pre_pos_mode: df_position = get_position(predate, product)[['SecuCode', 'Volume']]
        else: df_position = get_position(curdate, product)[['SecuCode', 'Volume']]

        for bar in range(1, 9)[::-1]:
            df_pob = get_alpha_quota_bar(curdate, quota_product, bar, short_mode=True)
            if not df_pob.empty:
                print(f'use quota {bar}')
                break

        df_pob = df_pob.rename({'Volume': 'POB'}, axis='columns')
        df_pob['SecuCode'] = df_pob['SecuCode'].apply(lambda x: expend_market(x, 3))

        df_position['SecuCode'] = df_position['SecuCode'].apply(lambda x: expend_market(x, 3))
        df_pob_new = pd.merge(df_pob, df_position, on='SecuCode', how='outer').fillna(0)
        df_pob_new['NewPob'] = (df_pob_new['POB'] - df_pob_new['Volume']).astype('int')
        df_pob_new['NewPob'] = df_pob_new.apply(
            lambda row: int(round(row['NewPob'] / 200) * 200)
            if row['SecuCode'][:3] == '688' and row['NewPob'] < 200
            else int(round(row['NewPob'] / 100) * 100), axis=1)
        df_pob_new['tmp'] = 0

        hprint(df_pob_new)

        df_pob_new = df_pob_new[['SecuCode', 'NewPob', 'tmp']]
        df_pob_new = df_pob_new[df_pob_new['NewPob'] != 0]

        if not os.path.exists(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/'):
            os.makedirs(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/')

        pob_path_file = f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/{curdate}-{product}-POB.csv'
        df_pob_new.to_csv(pob_path_file, header=False, index=False)
        tsl = TransferServerLocal()
        tsl.upload_file(local_path=pob_path_file, server_path=f'/home/jumper/pob_files/{product}-POB-{curdate}.csv')

        df_price = get_price(predate, predate)
        df_pob_new['Exchange'] = df_pob_new['SecuCode'].apply(lambda x: x.split('.')[1])
        df_pob_new['SecuCode'] = df_pob_new['SecuCode'].apply(lambda x: x.split('.')[0])
        df_pob_new = pd.merge(df_pob_new, df_price, on='SecuCode', how='left')
        df_pob_new['Value'] = df_pob_new['ClosePrice'] * df_pob_new['NewPob']
        df_pob_new['LS'] = (df_pob_new['NewPob'] < 0).astype('int')
        df_pob_new = df_pob_new.replace({'SSE': 'SH', 'SZE': 'SZ'})

        print(df_pob_new.groupby(['Exchange', 'LS'])[['Value']].sum().reset_index())
        print(df_pob_new.groupby('LS')[['Value']].sum().reset_index())

    def generate_pob_use_destpos_and_position_diff_short(self, curdate, product='SHORTZJJT1', quota_prd=None, bar=8, quota_pre_mode=False, pre_pos_mode=False):
        if quota_prd is None:
            quota_prd = product

        tsl = TransferServerLocal()
        local_path = f'{LOG_TEMP_PATH}{curdate}_position-{product}.txt'

        try:
            tsl.download_file(
                server_path=f'/home/jumper/post-trade/{curdate}_position-{product}.txt', local_path=local_path)
            print(f'/home/jumper/post-trade/{curdate}_position-{product}.txt')
        except:
            try:
                tsl.download_file(
                    server_path=f'/home/jumper/{curdate}_position-{product}.txt', local_path=local_path)
                print(f'/home/jumper/{curdate}_position-{product}.txt')
            except: pass
        
        predate = get_predate(curdate, 1)
        if quota_pre_mode:
            df_quota = get_alpha_quota_bar(predate, quota_prd, bar=bar, short_mode=True).rename(
                {'InstrumentID': 'SecuCode', 'Volume': 'DestPos'}, axis='columns')
        else:
            df_quota = get_alpha_quota_bar(curdate, quota_prd, bar=bar, short_mode=True).rename(
                {'InstrumentID': 'SecuCode', 'Volume': 'DestPos'}, axis='columns')
        
        if pre_pos_mode:
            df_pos_pre = get_position(predate, product, filepath=local_path)[['SecuCode', 'Volume']].rename(
                {'Volume': 'PreCloseVolume'}, axis='columns')
        else:
            df_pos_pre = get_position(curdate, product, filepath=local_path)[['SecuCode', 'Volume']].rename(
                {'Volume': 'PreCloseVolume'}, axis='columns')

        df_pos_pre['PreCloseVolume'] *= -1
        df_quota = pd.merge(df_quota, df_pos_pre, on='SecuCode', how='outer').fillna(0)
        df_quota['Volume'] = df_quota['DestPos'] - df_quota['PreCloseVolume']
        df_price = get_price(predate, predate).reset_index()[['SecuCode', 'ClosePrice']]

        df_quota = pd.merge(df_quota, df_price, on='SecuCode', how='left')
        df_quota['SecuCode'] = df_quota['SecuCode'].apply(lambda x: expend_market(x))
        df_quota['POB'] = 0

        net_value = (df_quota['Volume'] * df_quota['ClosePrice']).sum()
        total_value = np.abs(df_quota['Volume'] * df_quota['ClosePrice']).sum()
        short_value = np.round(net_value - (total_value + net_value) / 2, 2)
        long_value = np.round((total_value + net_value) / 2, 2)
        print(f'{product}_bar={bar}_空头市值：', short_value)
        print(f'{product}_bar={bar}_多头市值：', long_value)
        print(f'{product}_bar={bar}_净市值：', long_value + short_value)

        hprint(df_quota)
        df_quota = df_quota[['SecuCode', 'Volume', 'POB']]
        df_quota['Volume'] = df_quota['Volume'].astype('int')
        df_quota = df_quota[df_quota['Volume'] != 0]
        if not os.path.exists(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/'):
            os.makedirs(f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/')

        pob_path = f'{PLATFORM_PATH_DICT["z_path"]}Trading/POB_file/{curdate}/' \
                f'POB-AlphaT0-{product}-{curdate}-Short.csv'
        df_quota.to_csv(pob_path, header=False, index=False)

        tsl = TransferServerLocal(
            ip='139.196.57.140', port=22, username='jumper', password='jump2cfi888')
        tsl.upload_file(
            local_path=pob_path,
            server_path=f'/home/jumper/pob_files/{product}-POB-{curdate}.csv')


if __name__ == '__main__':
    curdate, flag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S').split('-')

    # curdate = '20250415'
    # gtbc = GenerateTransferBrokerConfig(curdate, trans_type='hedge_open_index_fix')
    # gtbc.generate(wechat_and_email=True)

    # curdate = '20250416'
    # gtbc = GenerateTransferBrokerConfig(curdate, trans_type='hedge_market_close')
    # gtbc.generate(wechat_and_email=True)

    # curdate = '20241210'
    # gtbc = GenerateTransferBrokerConfig(curdate, trans_type='hedge_market_open')
    # gtbc.generate()

    
    """
    from decimal import Decimal, ROUND_HALF_UP
    
    number = Decimal('3.14159')
    rounded_number = number.quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)
    print(rounded_number) 
    
    #%%
        margin_ratio = capital / (margin / 0.12) = capital / margin * 0.12
        ratio = margin / capital = 0.12 / margin_ratio
        0.80 -- 0.15
        0.90 -- 0.133
        0.75 -- 0.16
        0.50 -- 0.24
    """
    
    """
    1. 可控参数：
        当日总体保证金比例： margin_ratio_target
        当日最小保证金比例： margin_ratio_target_short
        期货户最低保证金比例： margin_exec_min_ratio
        最终期望暴露水平: expose_target_ratio
        多头暴露产品，目标多头暴露比例：long_exp_target_ratio
        空头暴露产品，目标空头暴露比例：short_exp_target_ratio 
        
    1. 调拨资金：
        同时调整暴露、风险度、持仓占比     # 调拨失败应该怎么处理
    2. 不调拨资金：
        1. 同时调整暴露、风险度，持仓占比
            1. 同时调整暴露、风险度，只降低持仓占比
            2. 同时调整暴露、风险度、只增加持仓占比
        3. 只调整暴露，不调整仓位
            1. 风险度合理，期货调整不受影响
            2. 风险度不合理，期货只调整空头尽量匹配
        
    """