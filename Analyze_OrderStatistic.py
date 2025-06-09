# 
"""
(一)得出某个产品在当日单秒内申报最高的次数并汇总成表就好了
(二)频繁瞬时撤单，是指日内频繁出现申报后迅速撤单、全日撤单比例较高的异常交易行为。竟价交易阶段同时存在下列情形的，予以重点监控:
    (1)1秒钟内申报又撤销申报;
    (2)满足情形(1)的行为在单只或者多只股票上发生500 次以上;
    (3)全日累计撤销申报金额占累计申报金额的 50%以上。

SH000905-20250418
SH000852-20250418
SH000300-20250418
CSI932000-20250418

上证综合指数
科创50指数

深证成分指数
深证综合指数
创业板指

"""
import sys
sys.path.append(r"V:\StockData\Trading\AnalysisTrading")
from toolmodules.modules import GetProductionInformation
from collections import ChainMap
from operation_others2 import *
from toolmodules.modules_constvars import *
import Analyze_OrderFreq
import pandas as pd
import os

today_date = "20250528"
IndexName_2_IndexinteriorCode_own = {
    'ZZ500': '4978',
    'ZZA500': '636661',
    'HS300': '3145',
    'SZ50': '46',
    'ZZ1000': '39144',
    'ZZ2000': '561230',
    'ZZ800': '4982',
    'ZZHL': '6973',
    'ZZQZ': '14110',
    'SSEINDEX': '1',
    'KC50' : "303968",
    'KCB' : '11089'
}


# def compute_exclusive_rolling_sum(df,value_col):
#     # Compute cumulative sum of value_col
#     #df['cumsum'] = df[value_col].cumsum()
    
#     # Calculate the time 60 seconds before each row
#     prod_ticker_df = df.copy(deep=True)
#     prod_ticker_df.set_index("timestamp",inplace=True)
#     prod_ticker_df["last_min"] = np.zeros(len(prod_ticker_df))
#     prod_ticker_df["last_min"] = prod_ticker_df[value_col].rolling(window="60s").sum()
#     prod_ticker_df.reset_index(inplace=True)
#     print(prod_ticker_df["last_min"])
#     return prod_ticker_df["last_min"]


def get_stockdaily_indexweight(QueryDate, index_name='ZZ500', OrderStyle='ListedDate'):
    if (int(QueryDate) < 20230831) and (index_name == 'ZZ2000'): QueryDate = '20230831'

    end_date = QueryDate
    cur_date = datetime.datetime.strptime(QueryDate, '%Y%m%d')
    one_month_ago = cur_date + datetime.timedelta(weeks=-5)
    start_date = one_month_ago.strftime('%Y%m%d')
    engine = create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/JYDB")

    if IndexName_2_IndexinteriorCode_own.get(index_name, None) is not None:
        index_code = IndexName_2_IndexinteriorCode_own[index_name]
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

KC50_ticker_weight = get_stockdaily_indexweight(today_date, index_name='KC50').reset_index()
SZ50_ticker_weight = get_stockdaily_indexweight(today_date, index_name='SZ50').reset_index()


def rolling_turnover_by_code(today_date,source_df):#,turnover_df):
    '''
    计算所有产品的个股rolling 1min 成交额，更改localtime使其可以与未来对应
    return order_df columns: "production", 'flag','code','timestamp','localtime','trade_turnover','rolling_sum_trade_turnover'
    '''
    order_df = source_df[["production","flag","code","trade_turnover"]].copy(deep=True)
    order_df['code'] = order_df['code'].str[-6:]
    # 不分产品
    order_df = combine_all_productions(order_df)
    print("Start calculating trader rolling turnover ...")
    #order_df = order_df.groupby("code","flag")
    #order_df = order_df[(order_df["production"]== "A500YX1") | (order_df["production"]== "DC29")]
    order_df["timestamp"] = pd.to_datetime(order_df["flag"], format="%H:%M:%S")
    order_df = order_df.sort_values("flag",ascending=True)
    # order_df['send_turnover_rolling_sum'] = (
    #     order_df.groupby(["production","code"])[['timestamp','send_turnover']].rolling(window='60s',on="timestamp")["send_turnover"].sum()
    #     #.apply(lambda x: compute_exclusive_rolling_sum(x, 'send_turnover'))
    # )
    # rolling_sum_order_df = order_df.groupby(["production","code"]).sort_values("flag",ascending = True)[['timestamp','trade_turnover']].rolling(window='60s',on="timestamp")["trade_turnover"].sum().reset_index()
    rolling_sum_order_df = order_df.groupby(["code"])[['timestamp','trade_turnover']].rolling(window='60s',on="timestamp")["trade_turnover"].sum().reset_index()
    rolling_sum_order_df.rename(columns={"trade_turnover":"rolling_sum_trade_turnover"},inplace=True)
    order_df = order_df[::-1]
    rolling_sum_order_df_forward = order_df.groupby(["code"])[['timestamp','trade_turnover']].rolling(window='60s',on="timestamp")["trade_turnover"].sum().reset_index()
    rolling_sum_order_df_forward.rename(columns={"trade_turnover":"rolling_sum_trade_turnover_forward"},inplace=True)
    order_df = order_df[::-1]
    print(rolling_sum_order_df.columns)
    order_df = pd.merge(order_df, rolling_sum_order_df, how = "left", on=["code","timestamp"])
    order_df = pd.merge(order_df, rolling_sum_order_df_forward, how = "left", on=["code","timestamp"])
    order_df["localtime"] = order_df["timestamp"].apply(lambda x: (x.strftime('%H:%M:%S')[:2] + x.strftime('%H:%M:%S')[3:5] + x.strftime('%H:%M:%S')[-2:])).astype(int)
    return order_df


def combine_prod_code_trade_at_fill(today_date,source_df):#,turnover_df):
    '''
    计算所有产品的个股rolling 1min 成交额，更改localtime使其可以与未来对应
    return order_df columns: "production", 'time','SecuCode','timestamp','localtime','trade_turnover','rolling_sum_trade_turnover'
    '''
    order_df = source_df[["production","time","SecuCode","trade_turnover"]].copy(deep=True)
    order_df = order_df.rename(columns={"SecuCode" : 'code'})
    order_df['flag'] = order_df['time'].astype(str).str.zfill(6)
    order_df['code'] = order_df['code'].str[-6:]
    # 不分产品
    order_df = combine_all_productions(order_df)
    order_df["localtime"] = order_df['flag'].astype(int)
    return order_df


def rolling_turnover_by_code_at_fill(today_date,source_df):
    print("Start calculating trader rolling turnover ...")
    order_df = source_df
    order_df['trade_turnover'] = order_df['trade_turnover'].fillna(0)
    order_df["timestamp"] = pd.to_datetime(order_df["localtime"].astype(str).str.zfill(6), format="%H%M%S")
    order_df = order_df.sort_values("localtime",ascending=True)
    # order_df['send_turnover_rolling_sum'] = (
    #     order_df.groupby(["production","code"])[['timestamp','send_turnover']].rolling(window='60s',on="timestamp")["send_turnover"].sum()
    #     #.apply(lambda x: compute_exclusive_rolling_sum(x, 'send_turnover'))
    # )
    # rolling_sum_order_df = order_df.groupby(["production","code"]).sort_values("flag",ascending = True)[['timestamp','trade_turnover']].rolling(window='60s',on="timestamp")["trade_turnover"].sum().reset_index()
    rolling_sum_order_df = order_df.groupby(["code"])[['timestamp','trade_turnover']].rolling(window='60s',on="timestamp")["trade_turnover"].sum().reset_index()
    rolling_sum_order_df.rename(columns={"trade_turnover":"rolling_sum_trade_turnover"},inplace=True)
    order_df = order_df[::-1]
    rolling_sum_order_df_forward = order_df.groupby(["code"])[['timestamp','trade_turnover']].rolling(window='60s',on="timestamp")["trade_turnover"].sum().reset_index()
    rolling_sum_order_df_forward.rename(columns={"trade_turnover":"rolling_sum_trade_turnover_forward"},inplace=True)
    order_df = order_df[::-1]
    print(rolling_sum_order_df.columns)
    order_df = pd.merge(order_df, rolling_sum_order_df, how = "left", on=["code","timestamp"])
    order_df = pd.merge(order_df, rolling_sum_order_df_forward, how = "left", on=["code","timestamp"])
    return order_df


def trade_rolling_index_turnover(code_trade_turnover, index_weight_df,index_sum_turnover):
    index_trade_turnover_account = code_trade_turnover[code_trade_turnover['code'].str[2:].isin(index_weight_df['SecuCode'])].copy()
    index_trade_turnover_account = index_trade_turnover_account[['localtime','rolling_sum_trade_turnover']].groupby("localtime").sum().reset_index()
    index_trade_turnover_account = index_trade_turnover_account.merge(index_sum_turnover, on = 'localtime',how='left')
    index_trade_turnover_account['指数交易量市场占比'] = index_trade_turnover_account['rolling_sum_trade_turnover']/index_trade_turnover_account['market_turnover_rolling']
    return index_trade_turnover_account


def get_all_ticker_df(date):
    '''
    获取所有个股秒级交易数据，算出rolling 1min 成交额，以及中间价，分钟内最高价，分钟内最低价
    ['localtime', 'code', 'market_turnover_rolling', 'mid_price','max_price', 'min_price']
    '''
    """return a df containing all ticker's 1min sum, reset at 1300"""
    filenames = os.listdir(rf"V:\stock_root\StockData\d0_file\CodeSecondData\{date}")
    filenames = filenames[:1000] # for test
    #filenames = ["SH600050-20250528.csv"]
    df_list = []
    i = 0
    start = time.time()
    end = time.time()
    while i<len(filenames): 
        if len(filenames[i]) < 10 or filenames[i][-12:-4]!=date:
            removed_element = filenames.pop(i)
        else:
            ticker_df = pd.read_csv(os.path.join(rf"V:\stock_root\StockData\d0_file\CodeSecondData\{date}",filenames[i]))[["localtime","turnover","bp1","ap1"]]
            ticker_df['localtime_str'] = ticker_df['localtime'].astype(str)
            ticker_df['localtime_str'] = ticker_df['localtime_str'].str.zfill(6)
            ticker_df["timestamp"] = pd.to_datetime(ticker_df['localtime_str'], format="%H%M%S")
            ticker_df["mid_price"] = (ticker_df["bp1"] + ticker_df["ap1"])/2
            ticker_df["turnover_cumsum"] = ticker_df["turnover"].cumsum()
            ticker_df["market_turnover_rolling"] = np.zeros(len(ticker_df))
            ticker_df["market_turnover_rolling"] = ticker_df[["timestamp",'turnover']].rolling(window="60s",on="timestamp", min_periods=1)["turnover"].sum()
            ticker_df["max_price"] = ticker_df[["timestamp",'mid_price']].rolling(window="60s",on="timestamp", min_periods=1)["mid_price"].max()
            ticker_df["min_price"] = ticker_df[["timestamp",'mid_price']].rolling(window="60s",on="timestamp", min_periods=1)["mid_price"].min()
            ticker_df = ticker_df.sort_values(by="timestamp",ascending=True)
            ticker_df = ticker_df[::-1]
            ticker_df["market_turnover_rolling_forward"] = ticker_df[["timestamp",'turnover']].rolling(window="60s",on="timestamp", min_periods=1)["turnover"].sum()
            ticker_df["max_price_forward"] = ticker_df[["timestamp",'mid_price']].rolling(window="60s",on="timestamp", min_periods=1)["mid_price"].max()
            ticker_df["min_price_forward"] = ticker_df[["timestamp",'mid_price']].rolling(window="60s",on="timestamp", min_periods=1)["mid_price"].min()
            ticker_df = ticker_df[::-1]
            ticker_df['leastsqr_x'] = ticker_df["localtime_str"].str[-2:].astype(int)
            ticker_df['minute_group'] = ticker_df['localtime_str'].str[:4]
            ticker_df_leastsqr = ticker_df[['minute_group','leastsqr_x','mid_price']].sort_values(by='leastsqr_x',ascending = True)
            ticker_df_leastsqr["X_times_Y"] = ticker_df_leastsqr['leastsqr_x'] * ticker_df_leastsqr['mid_price']
            ticker_df_leastsqr_group = ticker_df_leastsqr.groupby('minute_group')
            # std_x = ticker_df_leastsqr_group['leastsqr_x'].agg(np.std)
            # std_y = ticker_df_leastsqr_group['mid_price'].agg(np.std)
            # rho = ticker_df_leastsqr_group.apply(lambda g:find_corr(g))
            rho_process_df = ticker_df_leastsqr_group.agg({
                "leastsqr_x":["sum",'std'],
                "mid_price":["sum",'std'],
                "X_times_Y" : 'sum',
                'minute_group':'count'
                })
            std_x = rho_process_df[('leastsqr_x', 'std')]
            std_y = rho_process_df[('mid_price', 'std')]
            rho_process_df['cov_X_Y'] = (
                (rho_process_df[('X_times_Y', 'sum')] - 
                 (rho_process_df[('leastsqr_x', 'sum')] * rho_process_df[('mid_price', 'sum')]) / rho_process_df[('minute_group', 'count')]) /
                (rho_process_df[('minute_group', 'count')] - 1)
            )
            rho = (
                rho_process_df['cov_X_Y'] / 
                (rho_process_df[('leastsqr_x', 'std')] * rho_process_df[('mid_price', 'std')])
            )
            mean_x = ticker_df_leastsqr_group['leastsqr_x'].agg(np.mean)
            mean_y = ticker_df_leastsqr_group['mid_price'].agg(np.mean)
            # ticker_df = ticker_df.groupby('minute_group')
            ticker_df['std_x'] = ticker_df['minute_group'].map(std_x)
            ticker_df['std_y'] = ticker_df['minute_group'].map(std_y)
            ticker_df['rho'] = ticker_df['minute_group'].map(rho)
            ticker_df['mean_x'] = ticker_df['minute_group'].map(mean_x)
            ticker_df['mean_y'] = ticker_df['minute_group'].map(mean_y)
            # ticker_df = ticker_df.reset_index()
            ticker_df['beta'] = ticker_df['rho'] * ticker_df["std_y"]/ticker_df['std_x']
            ticker_df['alpha'] = ticker_df['mean_y'] - ticker_df['beta']*ticker_df['mean_x']
            ticker_df['epsilon'] = (ticker_df['mid_price'] - ticker_df['alpha'] - ticker_df['beta'] * ticker_df['leastsqr_x']).abs().fillna(0)
            ticker_df['beta'] = (ticker_df['beta']*60).fillna(0)
            #cumsum_7199 = ticker_df.iloc[7200]["turnover_cumsum"] if 7200 < len(ticker_df) else 0
            #mask = (ticker_df.index > 7200) & (ticker_df.index < 7260)
            #ticker_df.loc[mask, "market_turnover_rolling"] = ticker_df.loc[mask, "turnover_cumsum"] - cumsum_7199
            ticker_df["code"] = filenames[i][:8]
            #print(ticker_df.iloc[:10])
            """
            if turnover_df.empty:
                # turnover_df = ticker_df[["localtime","turnover_cumsum"]]
                turnover_df = ticker_df[["localtime","code","turnover_rolling"]]
            else:
                #turnover_df = pd.merge(turnover_df, ticker_df[["localtime","code","turnover_rolling"]],how="left",on="localtime")
                turnover_df = pd.concat([turnover_df,ticker_df[["localtime","code","turnover_rolling"]]],copy=False)
                # turnover_df = pd.merge(turnover_df, ticker_df[["localtime","turnover_cumsum"]],how="left",on="localtime")
            """
            df_list.append(ticker_df[["localtime","code","market_turnover_rolling","mid_price","max_price","min_price","market_turnover_rolling_forward","max_price_forward","min_price_forward","beta",'epsilon']])#,"turnover_cumsum","turnover"]])
            if i%100 == 0:
                end = time.time()
                print(f"{i} stocks read,用时{end-start}")
                start = time.time()
            i+=1
    turnover_df = pd.concat(df_list)
    
    print(f'提取个股数据总用时{end-start}')
    return turnover_df

def find_corr(df):
    return np.corrcoef(df['leastsqr_x'],df['mid_price'])[0,1]


def one_percent_up_or_down_column(source_df,original_ticker_df):
    '''
    在每个（产品，秒钟时间，股票代码）里面查找当前秒钟和一分钟前的秒钟内的价格变动信息
    '''
    print("Calculating if large order come with more than 1% drop on ticker price in past 1 min...")
    start = time.time()
    def calculate_prior_localtime(localtime):
        if localtime < 93100:
            return 93000  # Before 9:31:00, use 9:30:00
        elif 130000 < localtime < 130100:
            return 113000  # Around lunch break, use 11:30:00
        else:
            # Convert to datetime, subtract one minute, and format back to float
            localtime_str = f"{int(localtime):06d}"
            localtime_dt = pd.to_datetime(localtime_str, format="%H%M%S")
            prior_dt = localtime_dt - pd.Timedelta(minutes=1)
            return float(prior_dt.strftime("%H%M%S"))
    merged_df = source_df[["localtime","code"]].copy(deep=True)
    mask = original_ticker_df["code"].isin(merged_df["code"])
    ticker_df = original_ticker_df.loc[mask].copy(deep=True)
    merged_df['prior_localtime'] = merged_df['localtime'].apply(calculate_prior_localtime)
    print("Merging based on two times")
    merged_df = pd.merge(merged_df, ticker_df[['code', 'localtime', 'mid_price','max_price','min_price','max_price_forward','min_price_forward']], 
                     on=['code', 'localtime'], how='left', suffixes=('', '_current'))
    merged_df = pd.merge(merged_df, ticker_df[['code', 'localtime', 'mid_price']], 
                     left_on=['code', 'prior_localtime'], right_on=['code', 'localtime'], 
                     how='left', suffixes=('', '_prior'))
    # print("calculating minute max...")
    # merged_df['range_max'] = merged_df.apply(lambda row: max(ticker_df.loc[(ticker_df['code']==row["code"]) & (ticker_df['localtime'] <= row["localtime"])
    #                                        & (ticker_df['localtime'] >= row["localtime_prior"])]["mid_price"]), axis = 1)
    # print("calculating minute min...")
    # merged_df['range_min'] = merged_df.apply(lambda row: min(ticker_df.loc[(ticker_df['code']==row["code"]) & (ticker_df['localtime'] <= row["localtime"])
    #                                        & (ticker_df['localtime'] >= row["localtime_prior"])]["mid_price"]), axis = 1)
    print(merged_df.columns)
    merged_df['percent_change'] = abs((merged_df['mid_price'] - merged_df['mid_price_prior']) / merged_df['mid_price_prior'])
    merged_df['percent_change'] = merged_df["percent_change"].fillna(0)
    merged_df["分钟内最大波动范围"] = np.maximum(abs((merged_df['max_price'] - merged_df['min_price']) / merged_df['min_price']), abs((merged_df['max_price_forward'] - merged_df['min_price_forward']) / merged_df['min_price_forward']))
    merged_df["分钟内最大波动范围"] = merged_df["分钟内最大波动范围"].fillna(0)
    merged_df['分钟内百1以上波动'] = (merged_df['分钟内最大波动范围'] > 0.01).astype(int)
    merged_df['分钟内百1以上波动'] = merged_df['分钟内百1以上波动'].fillna(0)
    end = time.time()
    print(f"Time took to calculate is {end-start}")
    return merged_df[["localtime","code",'percent_change','分钟内百1以上波动',"分钟内最大波动范围"]]


"""每个ticker快照位置V:\StockData\d0_file\CodeSecondData\20250428"""
def single_stock_cancel_500_in_one_sec(source_df):
    cancel_one_sec_df = source_df[["production",'cancel_num_in_1s',"exch"]].copy(deep=True)
    cancel_one_sec_df = cancel_one_sec_df.groupby(["production","exch"])[["cancel_num_in_1s"]].sum().reset_index()
    return cancel_one_sec_df

def combine_all_productions(source_df):
    '''
    按照code来算每秒总成交量
    '''
    order_df = source_df[["production","flag","code","trade_turnover"]].copy(deep=True)
    print(order_df.columns)
    order_df = order_df.groupby(["flag","code"])['trade_turnover'].sum().reset_index()
    return order_df

def only_keep_main_acc(source_df, command = "2"):
    assert command in ["1","2","2.3"],\
        "command = '1' 是监管(一), command = '2' 是监管(二), command = '2.3'是监管(二)下面的(3)条"
    df = source_df.copy(deep=True)
    prod_info = GetProductionInformation()
    df['main_prd'] = df['production'].apply(lambda prd: prod_info.get_product_trading_main_name(prd))
    if command == "1":
        assert all(col in source_df.columns for col in ["production",'colo',"exch",'flag','order_cnt','cancel_cnt']),\
            "监管第一条需要参数 production,colo,flag,order_cnt,cancel_cnt,cum_cnt"
        df = df.groupby(['main_prd',"flag","exch"])[['order_cnt','cancel_cnt',"order_cnt_shift","cancel_cnt_shift","send_turnover","cancel_turnover"]].sum().reset_index()
        df["cum_cnt"] = df["order_cnt"] + df["cancel_cnt"]
        df['cum_cnt_shift'] = df['order_cnt_shift'] + df["cancel_cnt_shift"]
        df["max_cum_cnt"] = np.maximum(df['cum_cnt'],df['cum_cnt_shift'])
        df['cum_cancel_ratio'] = df['cancel_turnover']/df['send_turnover']
        df = df.rename(columns={"main_prd":"production"}).sort_values(by="max_cum_cnt",ascending=False)
        df = df.loc[(df["max_cum_cnt"] >= float(150)) & (df['cum_cancel_ratio'] > 0.1)]
        df = df[["production","exch",'flag','order_cnt','cancel_cnt',"max_cum_cnt",'cum_cancel_ratio']]
        df_1_res = df.style.background_gradient(cmap="Oranges", axis=0, subset=['cum_cancel_ratio']) \
        .background_gradient(cmap="Blues", axis=0, subset=['max_cum_cnt']) \
        .set_table_styles([
        dict(selector='td', props=[('border', '1px solid black')]),
        dict(selector='caption', props=[('padding', '1px'), ('margin', '1px 0'), ('text-align', 'center')])
        ]).set_caption(f'''
        <div style="font-size: 10px">
            <strong>发撤单不得超过600且总撤单率不能超过总发单50%</strong><br>
        </div>
        ''')
        dfi.export(df_1_res,filename = rf"V:\StockData\Trading\owen_code\7月新规\监管第一条,png",dpi=500, fontsize=8,max_cols = -1,max_rows=-1,table_conversion='chrome')
        return df
    elif command == "2":
        # 
        assert all(col in source_df.columns for col in ["production","cancel_num_in_1s"]),\
            "监管第二条需要参数 production, cancel_num_in_1s"
        df = df.groupby(['main_prd',"exch"])[['cancel_num_in_1s']].sum().reset_index()
        df = df.rename(columns={"main_prd":"production"}).sort_values(by="cancel_num_in_1s",ascending=False)
        return df.loc[df["cancel_num_in_1s"] > 0]
    elif command == "2.3":
        assert all(col in source_df.columns for col in ["production",'send_turnover','trade_turnover']),\
            "监管第二.3条需要参数 production, send_turnover, trade_turnover"
        df = df.groupby(['main_prd',"exch"])[['send_turnover','trade_turnover']].sum().reset_index()
        df['cancel_rate'] = df.apply(lambda x: 1-(float(x["trade_turnover"])/float(x["send_turnover"])), axis=1)
        df = df.rename(columns={"main_prd":"production"}).sort_values(by="cancel_rate",ascending=False)
        return df
    
def get_large_send_and_cancel_order(source_df):
    """generate a dataframe that contains the total send, cancel or cumulative order count of a product in a colo machine.
        group by flag(which stands for time flag), and sum every product within the flag"""
    assert all(col in source_df.columns for col in ["production","exch",'colo','flag','order_cnt','cancel_cnt',"send_turnover","trade_turnover"]), \
        "Double Check order_freq_1s_details_new file, possibly modified"
    acc_cum_sec = source_df[["production",'colo',"exch",'flag','order_cnt','cancel_cnt','order_cnt_shift','cancel_cnt_shift']].copy(deep =True)
    acc_cum_sec_turnover = source_df[['production','colo','exch','send_turnover','trade_turnover']].copy(deep=True)
    acc_cum_sec_turnover['cancel_turnover'] = acc_cum_sec_turnover['send_turnover'] - acc_cum_sec_turnover['trade_turnover']
    acc_cum_sec = acc_cum_sec.groupby(["production","colo","flag","exch"])[["order_cnt","cancel_cnt",'order_cnt_shift',"cancel_cnt_shift"]].sum().reset_index()
    acc_cum_sec_turnover = acc_cum_sec_turnover.groupby(["production","colo","exch"])[['trade_turnover',"send_turnover","cancel_turnover"]].sum().reset_index()
    acc_cum_sec = acc_cum_sec.merge(acc_cum_sec_turnover,on=['production','colo','exch'],how='left')
    return acc_cum_sec

def get_fifty_percent_amount_cancel(source_df):
    assert all(col in source_df.columns for col in ["production",'colo',"code","exch",'send_turnover','trade_turnover']), \
        "Double Check order_freq_1s_details_new file, possibly modified"
    turnover_total_df = source_df[["production",'colo',"exch","code",'flag','send_turnover','trade_turnover']].copy(deep=True)
    turnover_total_df = turnover_total_df.groupby(["production","colo","code","exch"])[['send_turnover','trade_turnover']].sum().reset_index()
    return turnover_total_df


# def color_df(original_df):

def get_index_ticker_turnover_data(date,index_component):
    '''
    获取指数内个股秒级交易数据，算出rolling 1min 成交额，以及中间价，分钟内最高价，分钟内最低价
    ['localtime', 'code', 'market_turnover_rolling', 'mid_price','max_price', 'min_price']
    '''
    """return a df containing all ticker's 1min sum, reset at 1300"""
    filenames = os.listdir(rf"V:\stock_root\StockData\d0_file\CodeSecondData\{date}")
    df_list = []
    i = 0
    while i<len(filenames):
        if len(filenames[i]) < 10 or filenames[i][-12:-4]!=date or filenames[i][2:8] not in index_component['SecuCode'].to_list():
            removed_element = filenames.pop(i)
        else:
            ticker_df = pd.read_csv(os.path.join(rf"V:\stock_root\StockData\d0_file\CodeSecondData\{date}",filenames[i]))[["localtime","turnover","bp1","ap1"]]
            ticker_df['localtime_str'] = ticker_df['localtime'].astype(str)
            ticker_df['localtime_str'] = ticker_df['localtime_str'].str.zfill(6)
            ticker_df["timestamp"] = pd.to_datetime(ticker_df['localtime_str'], format="%H%M%S")
            ticker_df["mid_price"] = (ticker_df["bp1"] + ticker_df["ap1"])/2
            ticker_df["turnover_cumsum"] = ticker_df["turnover"].cumsum()
            ticker_df["market_turnover_rolling"] = np.zeros(len(ticker_df))
            ticker_df["market_turnover_rolling"] = ticker_df[["timestamp",'turnover']].rolling(window="60s",on="timestamp", min_periods=1)["turnover"].sum()
            ticker_df["max_price"] = ticker_df[["timestamp",'mid_price']].rolling(window="60s",on="timestamp", min_periods=1)["mid_price"].max()
            ticker_df["min_price"] = ticker_df[["timestamp",'mid_price']].rolling(window="60s",on="timestamp", min_periods=1)["mid_price"].min()
            #cumsum_7199 = ticker_df.iloc[7200]["turnover_cumsum"] if 7200 < len(ticker_df) else 0
            #mask = (ticker_df.index > 7200) & (ticker_df.index < 7260)
            #ticker_df.loc[mask, "market_turnover_rolling"] = ticker_df.loc[mask, "turnover_cumsum"] - cumsum_7199
            ticker_df["code"] = filenames[i][:8]
            #print(ticker_df.iloc[:10])
            """
            if turnover_df.empty:
                # turnover_df = ticker_df[["localtime","turnover_cumsum"]]
                turnover_df = ticker_df[["localtime","code","turnover_rolling"]]
            else:
                #turnover_df = pd.merge(turnover_df, ticker_df[["localtime","code","turnover_rolling"]],how="left",on="localtime")
                turnover_df = pd.concat([turnover_df,ticker_df[["localtime","code","turnover_rolling"]]],copy=False)
                # turnover_df = pd.merge(turnover_df, ticker_df[["localtime","turnover_cumsum"]],how="left",on="localtime")
            """
            df_list.append(ticker_df[["localtime","code","market_turnover_rolling","mid_price","max_price","min_price"]])#,"turnover_cumsum","turnover"]])
            if i%10 == 0:
                print(i,"stocks read")
            i+=1
    turnover_df = pd.concat(df_list)
    return turnover_df

def get_index_turnover_data_combined(index_df, date):
    index_ticker_df = get_index_ticker_turnover_data(date, index_df)
    print(index_ticker_df.head(10))
    # index_ticker_df = ticker_df.merge(index_df, left_on = "code",right_on = 'SecuCode',how = "left")
    index_ticker_df = index_ticker_df[['localtime','market_turnover_rolling']].groupby(["localtime"]).sum().reset_index()
    return index_ticker_df


def filter_market_reaction(original_df, beta_threshold,epsilon_threshold):
    selected_df_up_down = original_df[(original_df['beta'].abs() > beta_threshold) & (original_df[epsilon] < epsilon_threshold)]
    selected_df_move = original_df[original_df['分钟内百1以上波动'] == 1]
    return pd.concat([selected_df_up_down,selected_df_move]).copy().drop_duplicates()
    
def forward_rolling_mean(series, window=5):
    # Extract the values as a numpy array
    values = series.values
    # Reverse the values to simulate forward-looking window
    reversed_values = values[::-1]
    # Compute rolling mean on reversed values
    rolling_mean_reversed = pd.Series(reversed_values).rolling(window=window, min_periods=1).mean().values
    # Reverse back and return as a Series with original index
    return pd.Series(rolling_mean_reversed[::-1], index=series.index)


"""
individual_stock_all_path = "V:\stock_root\StockData\d0_file\CodeSecondData"
target_date = "20250423"
if os.path.exists(os.path.join(individual_stock_all_path,target_date)):
    for stock_file in os.listdir(os.path.join(individual_stock_all_path,target_date)):
        if stock_file == f"SH600660-{target_date}.csv":
            curr_stock_df = pd.read_csv(os.path.join(individual_stock_all_path,target_date,f"SH600660-{target_date}.csv"))
            curr_stock_df["mp"] = (curr_stock_df['bp1'] + curr_stock_df["ap1"])/2
            #print(curr_stock_df["localtime"][:100])
            print("\n\n")
            print(curr_stock_df["localtime"].iloc[0], curr_stock_df["localtime"].iloc[-1])
            print(curr_stock_df.loc[curr_stock_df["localtime"] == "120000"])
            print(curr_stock_df.head(10))
            
            

detail_order_freq = r"V:\StockData\d0_file\IntraAnalysisResults\20250417\OrderFreq\20250417_order_freq_1s_details.csv"
order_freq = r"V:\StockData\d0_file\IntraAnalysisResults\20250417\OrderFreq\20250417_order_freq_1s.csv"
summary = r"V:\StockData\d0_file\IntraAnalysisResults\20250417\OrderFreq\20250417_order_freq_summary.csv"
colo = "hto-sz-1"
date = "20250422"
colo_data = pd.read_csv(rf"Z:\ProcessedData\PTA\parsed_log\{colo}\{date}\{colo}-worker-{date}.csv")

op_date = '20250422'
trade_cfg_df = Analyze_OrderFreq.read_trading_account_from_db(op_date)
colo_list = Analyze_OrderFreq.get_colo_list(op_date)
trader_list_cfg_df = Analyze_OrderFreq.get_trader_cfg(op_date)
cfg_df = pd.merge(trader_list_cfg_df, trade_cfg_df[["colo",  "stid", "trade_acc", "production"]], on=["colo", "trade_acc"], how="inner")
cfg_df = cfg_df[cfg_df['colo'].isin(colo_list)]

#pta_task_profile = Analyze_OrderFreq.PtaTaskProfile()
# colo_2_machine_cpu_freq = {
#     colo: pta_task_profile.machine_cpu_freq(colo, op_date) for colo in cfg_df['colo'].unique()}
# cfg_df['cpu_freq'] = cfg_df['colo'].apply(lambda x: colo_2_machine_cpu_freq[x])


df_detail_order_freq = pd.read_csv(detail_order_freq)
df_order_freq = pd.read_csv(order_freq)
df_summary = pd.read_csv(summary,encoding="latin1")
df_colo = pd.read_csv(colo_data)
"""

if __name__ == "__main__":
    '''
    op_date = '20250428'
    trade_cfg_df = Analyze_OrderFreq.read_trading_account_from_db(op_date)
    colo_list = Analyze_OrderFreq.get_colo_list(op_date)
    trader_list_cfg_df = Analyze_OrderFreq.get_trader_cfg(op_date)
    cfg_df = pd.merge(trader_list_cfg_df, trade_cfg_df[["colo",  "stid", "trade_acc", "production"]], on=["colo", "trade_acc"], how="inner")
    cfg_df = cfg_df[cfg_df['colo'].isin(colo_list)]

    pta_task_profile = Analyze_OrderFreq.PtaTaskProfile()
    colo_2_machine_cpu_freq = {
        colo: pta_task_profile.machine_cpu_freq(colo, op_date) for colo in cfg_df['colo'].unique()}
    # cfg_df['cpu_freq'] = cfg_df['colo'].apply(lambda x: colo_2_machine_cpu_freq[x])
    # cfg_df.to_csv(rf"V:\StockData\Trading\owen_code\trial_record\colo_cpu_freq.csv",index=False)
    cfg_df = pd.read_csv(rf"V:\StockData\Trading\owen_code\trial_record\colo_cpu_freq.csv")
    '''
    
    
    
    print("This file processed multiple send/cancel order regulation violation checks. \n\n")
    order_data_source = pd.read_parquet(rf"Z:\StockData\d0_file\IntraAnalysisResults\{today_date}\OrderFreq\{today_date}_order_freq_1s_details_new.parquet")
    """
    part 1
    (一)得出某个产品在当日单秒内申报最高的次数并汇总成表
    """
    print("Processing Part One Violation Check File...")
    large_cancel_send_order_df = get_large_send_and_cancel_order(order_data_source)
    acc_agg_cancel_send_violation = only_keep_main_acc(large_cancel_send_order_df, "1")
    """
    part 2
    (二)频繁瞬时撤单，是指日内频繁出现申报后迅速撤单、全日撤单比例较高的异常交易行为。竟价交易阶段同时存在下列情形的，予以重点监控:
        (1)1秒钟内申报又撤销申报;
        (2)满足情形(1)的行为在单只或者多只股票上发生500 次以上;
        (3)全日累计撤销申报金额占累计申报金额的 50%以上。
    """
    print("Processing Part Two Violation Check File...")
    send_and_cancel_in_1_sec = only_keep_main_acc(order_data_source,"2")
    
    daily_total_turnover = get_fifty_percent_amount_cancel(order_data_source)
    cancel_send_ratio = only_keep_main_acc(daily_total_turnover,"2.3")
    """
    part 3
    (三)频繁拉抬打压，是指日内多次出现股票小幅拉抬打压的异常交易行为。连续竞价阶段在单只或者多只股票同时存在下列情形15 次以上的，予以重点监控
        任意1分钟内，
        (1)单只股票上买入成交价呈上升趋势或者卖出成交价呈下降趋势; 趋势目前没有具体判断方式所以没有做
        (2)期间买入(卖出)成交数量占市场成交总量的10%以上;
        (3)期间股票涨(跌)幅 1%以上。
    """
    print("Processing Part Three Violation Check File...")
    # 合成交割订单数据
    today_all_products = set(large_cancel_send_order_df["production"])

    trade_fill_info_lst = []
    no_trade_ticker_list = []
    for prod_name in today_all_products:
        prod_trade_execute_df = get_trades(today_date,prod_name)
        if len(prod_trade_execute_df) == 0:
            print(f"{prod_name}未输出交易数据")
            no_trade_ticker_list.append(prod_trade_execute_df)
        else:
            if prod_name in today_all_products:
                prod_trade_execute_df['production'] = prod_name
                prod_trade_execute_df['trade_turnover'] = prod_trade_execute_df['Price'] * prod_trade_execute_df["Volume"]
                trade_fill_info_lst.append(prod_trade_execute_df)
    trade_fill_info_df = pd.concat(trade_fill_info_lst)
    trade_fill_info_long = trade_fill_info_df[trade_fill_info_df['LongShort'] == 1]
    trade_fill_info_short = trade_fill_info_df[trade_fill_info_df['LongShort'] == 0]
    
    
    
    all_ticker_turnover_df= get_all_ticker_df(today_date)
    all_ticker_turnover_df = all_ticker_turnover_df.rename(columns={'code':'code_center'})
    all_ticker_turnover_df['code_num_str'] = all_ticker_turnover_df['code_center'].str[-6:]
    grouped_all_ticker_turnover_df = all_ticker_turnover_df[['market_turnover_rolling','code_center']].groupby('code_center')
    all_ticker_turnover_df['smooth_market_turnover_rolling'] = grouped_all_ticker_turnover_df['market_turnover_rolling'].transform(forward_rolling_mean)
    all_ticker_turnover_df['code'] = all_ticker_turnover_df['code_num_str']
    all_ticker_code_selected = set(all_ticker_turnover_df['code'])
    trade_fill_info_df = trade_fill_info_df[trade_fill_info_df['SecuCode'].isin(all_ticker_code_selected)]

    trade_code_turnover_df_fill = combine_prod_code_trade_at_fill(today_date,trade_fill_info_df)
    print("combine trade and market turnover data then select...")
    res_df_fill = all_ticker_turnover_df.merge(trade_code_turnover_df_fill[["code","localtime","trade_turnover"]],how="left",on=["code","localtime"])
    res_df_fill = rolling_turnover_by_code_at_fill(today_date,res_df_fill)
    res_df_fill = res_df_fill[(res_df_fill['rolling_sum_trade_turnover'] != 0)|(res_df_fill['rolling_sum_trade_turnover_forward'] != 0)]
    try:
        res_df_fill["trade_ratio"] = np.where(
            (res_df_fill["rolling_sum_trade_turnover"] == 0) & (res_df_fill["market_turnover_rolling"] == 0),
            0,  # Case 1: Both are 0, return 0
            np.where(
                (res_df_fill["rolling_sum_trade_turnover"] != 0) & ((res_df_fill["market_turnover_rolling"] == 0) | (res_df_fill["market_turnover_rolling"] < 100* res_df_fill["mid_price"])),
                res_df_fill["rolling_sum_trade_turnover"] / res_df_fill["smooth_market_turnover_rolling"],  # Case 2
                np.maximum(res_df_fill["rolling_sum_trade_turnover"] / res_df_fill["market_turnover_rolling"] , res_df_fill["rolling_sum_trade_turnover_forward"] / res_df_fill["market_turnover_rolling_forward"])  # Case 3
            )
        )
    except Exception as e:
        print(res_df_fill.loc[(res_df_fill['rolling_sum_trade_turnover']!=0)&(res_df_fill['market_turnover_rolling'] == 0)])
        print("potentially this error,cannot be product rolling filled order not equals to zero while market 1min rolling turnover is zero \n 不能产品显示有成交记录但是市场上没有成交记录，这不合理")
    res_df_fill = res_df_fill.loc[res_df_fill["trade_ratio"] >= 0.1] 
    res_df_fill_merged = one_percent_up_or_down_column(res_df_fill, all_ticker_turnover_df)
    res_df_fill = res_df_fill.merge(res_df_fill_merged,how="left",on=["localtime","code"])
    # res_df_fill = res_df_fill[["code","localtime","mid_price","min_price","max_price","trade_turnover","rolling_sum_trade_turnover","market_turnover_rolling",'smooth_market_turnover_rolling',"trade_ratio",
    #                 "percent_change","分钟内最大波动范围","分钟内百1以上波动",'beta','epsilon']]
    res_df_fill = res_df_fill[res_df_fill['分钟内百1以上波动'] == 1]
    '''
    res_df = trade_code_turnover_df.merge(all_ticker_turnover_df,how="left",left_on=["code","localtime"],right_on=['code_num_str','localtime'])
    try:
        res_df["trade_ratio"] = np.where(
            (res_df["rolling_sum_trade_turnover"] == 0) & ((res_df["market_turnover_rolling"] == 0) | (res_df["market_turnover_rolling"] < 100* res_df["mid_price"])),
            0,  # Case 1: Both are 0, return 0
            np.where(
                (res_df["rolling_sum_trade_turnover"] != 0) & (res_df["market_turnover_rolling"] == 0),
                res_df["rolling_sum_trade_turnover"] / res_df["smooth_market_turnover_rolling"],  # Case 2
                res_df["rolling_sum_trade_turnover"] / res_df["market_turnover_rolling"]  # Case 3
            )
        )
    except Exception as e:
        print(res_df.loc[(res_df['trade_ratio']!=0)&(res_df['market_turnover_rolling'] == 0)])
        print("potentially this error,cannot be product rolling filled order not equals to zero while market 1min rolling turnover is zero \n 不能产品显示有成交记录但是市场上没有成交记录，这不合理")
    res_df = res_df.loc[res_df["trade_ratio"] >= 0.1]
    res_df["上一分钟价格变化"],res_df["分钟内百1以上波动"],res_df['分钟内最大波动范围'] = one_percent_up_or_down_column(res_df, all_ticker_turnover_df)
    res_df = res_df[["code","flag","trade_turnover","rolling_sum_trade_turnover","market_turnover_rolling","trade_ratio",
                    "分钟内百1以上波动","mid_price","min_price","max_price","上一分钟价格变化","分钟内最大波动范围",'beta','epsilon']]
    """再筛选一下有多少违规的就可以了"""
    small_residual = res_df[res_df['beta'] != 0]['epsilon'].quantile(0.25)
    large_beta = min(abs(res_df['beta'].quantile(0.9)),abs(res_df['beta'].quantile(0.1)))
    res_df_filtered = filter_market_reaction(res_df,large_beta,small_residual)
    
    
    """成交时订单计算方法"""
    res_df_fill = trade_code_turnover_df_fill.merge(all_ticker_turnover_df,how="left",left_on=["code","localtime"],right_on=['code_num_str','localtime'])
    try:
        res_df_fill["trade_ratio"] = np.where(
            (res_df_fill["rolling_sum_trade_turnover"] == 0) & (res_df_fill["market_turnover_rolling"] == 0),
            0,  # Case 1: Both are 0, return 0
            np.where(
                (res_df_fill["rolling_sum_trade_turnover"] != 0) & ((res_df_fill["market_turnover_rolling"] == 0) | (res_df_fill["market_turnover_rolling"] < 100* res_df_fill["mid_price"])),
                res_df_fill["rolling_sum_trade_turnover"] / res_df_fill["smooth_market_turnover_rolling"],  # Case 2
                res_df_fill["rolling_sum_trade_turnover"] / res_df_fill["market_turnover_rolling"]  # Case 3
            )
        )
    except Exception as e:
        print(res_df_fill.loc[(res_df_fill['trade_ratio']!=0)&(res_df_fill['market_turnover_rolling'] == 0)])
        print("potentially this error,cannot be product rolling filled order not equals to zero while market 1min rolling turnover is zero \n 不能产品显示有成交记录但是市场上没有成交记录，这不合理")
    res_df_fill = res_df_fill.loc[res_df_fill["trade_ratio"] >= 0.1] 
    res_df_fill["上一分钟价格变化"],res_df_fill["分钟内百1以上波动"],res_df_fill['分钟内最大波动范围'] = one_percent_up_or_down_column(res_df_fill, all_ticker_turnover_df)
    res_df_fill = res_df_fill[["code","flag","trade_turnover","rolling_sum_trade_turnover","market_turnover_rolling","trade_ratio",
                    "分钟内百1以上波动","mid_price","min_price","max_price","上一分钟价格变化","分钟内最大波动范围",'beta','epsilon']]
    """再筛选一下有多少违规的就可以了"""
    small_residual_fill = res_df_fill[res_df_fill['beta'] != 0]['epsilon'].quantile(0.25)
    large_beta_fill = min(abs(res_df_fill['beta'].quantile(0.9)),abs(res_df_fill['beta'].quantile(0.1)))
    res_df_fill_filtered = filter_market_reaction(res_df_fill,large_beta_fill,small_residual_fill)
    '''
    
    
    
    """
    part4 指数拉踩打压
    (四)短时间大额成交，是指短时间内买入(卖出)金额特别巨大，加剧本所主要指数波动的异常交易行为。连续竞价阶段任意1分钟内同时存在下列情形的，予以重点监控:
        (1)主动买入(卖出)上证综合指数成份股金额1亿元以上，或主动买入(卖出)科创 50 指数成份股金额 2000万元以上;
        (2)主动买入(卖出)金额占期间全市场上证综合指数成份股或科创 50 指数成份股买入(卖出)成交金额的 3%以上;
        (3)上证综合指数涨(跌)幅 0.2%以上，或科创 50 指数涨(跌)幅 0.4%以上。
    """
    # 先合成指数成分每秒价格，成分交易量
    KC50_component_turnover = get_index_turnover_data_combined(KC50_ticker_weight,today_date)
    SZ50_component_turnover = get_index_turnover_data_combined(SZ50_ticker_weight,today_date)
    # 占市场交易量百分比 以及成分股交易金额
    KC50_turnover_res_percentage = trade_rolling_index_turnover(trade_code_turnover_df,KC50_ticker_weight,KC50_component_turnover)
    SZ50_turnover_res_percentage = trade_rolling_index_turnover(trade_code_turnover_df,SZ50_ticker_weight,KC50_component_turnover)

    
    """
    example_df = pd.read_csv(r"V:\stock_root\StockData\d0_file\CodeSecondData\20250428\SZ002137-20250428.csv")
    example_df = example_df[["localtime",'turnover']]
    example_df["cum_turnover"] = example_df['turnover'].cumsum()
    new_df = all_ticker_turnover_df.merge(example_df[["localtime","cum_turnover"]], how="left",on="localtime")
    """
    


        


"""
输入数据
1. 发单数据
    
2. 指数快照
    V:\ProcessedData/OPTIONS/index-alltick.v2/20250417/SH000905-20250417.csv
3. 个股的快照数据
    V:\stock_root\StockData\d0_file\CodeSecondData\{date}
    
    V:\StockData\d0_file\IntraAnalysisResults\20250417\OrderFreq
4. 券商数据
    Z:\ProcessedData\PTA\parsed_log\{colo}\{date}\{colo}-worker-{date}.csv


输出数据:
    产品数 * len(code_list), 239 * 60, 发单数，发单金额，其他变量
    沪深分开
    
    指数快照
        成交金额 1s
        价格 1s
    个股快照
        成交金额 1s
        价格 1s

rolling 1min

"""