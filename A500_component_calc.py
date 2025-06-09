# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 13:00:48 2025

@author: zhangziyao
"""
import sys
sys.path.append(rf'V:\stock_root\StockData\Trading\AnalysisTrading')

from operation_others2 import *
from toolmodules.modules_constvars import *

def get_300_not_ZZA500(vol_df, price_df, dict_index,total):
    vol_df_temp = vol_df.loc[vol_df["SecuCode"].isin(dict_index["HS300"]["SecuCode"]) & ~vol_df["SecuCode"].isin(dict_index["ZZA500"]["SecuCode"])]
    price_df_temp = price_df.loc[price_df["SecuCode"].isin(dict_index["HS300"]["SecuCode"]) & ~price_df["SecuCode"].isin(dict_index["ZZA500"]["SecuCode"])]
    merged_df = vol_df_temp.merge(price_df_temp, on="SecuCode", how="inner")
    merged_df["Product"] = merged_df["Volume"] * merged_df["ClosePrice"]
    sum_portion = merged_df["Product"].sum()/total
    return sum_portion


def get_ZZA500(vol_df, price_df, dict_index,total):
    vol_df_temp = vol_df.loc[vol_df["SecuCode"].isin(dict_index["ZZA500"]["SecuCode"])]
    price_df_temp = price_df.loc[price_df["SecuCode"].isin(dict_index["ZZA500"]["SecuCode"])]
    merged_df = vol_df_temp.merge(price_df_temp, on="SecuCode", how="inner")
    merged_df["Product"] = merged_df["Volume"] * merged_df["ClosePrice"]
    sum_portion = merged_df["Product"].sum()/total
    return sum_portion


def get_500_not_ZZA500(vol_df, price_df, dict_index,total):
    vol_df_temp = vol_df.loc[vol_df["SecuCode"].isin(dict_index["ZZ500"]["SecuCode"]) & ~vol_df["SecuCode"].isin(dict_index["ZZA500"]["SecuCode"])]
    price_df_temp = price_df.loc[price_df["SecuCode"].isin(dict_index["ZZ500"]["SecuCode"]) & ~price_df["SecuCode"].isin(dict_index["ZZA500"]["SecuCode"])]
    merged_df = vol_df_temp.merge(price_df_temp, on="SecuCode", how="inner")
    merged_df["Product"] = merged_df["Volume"] * merged_df["ClosePrice"]
    sum_portion = merged_df["Product"].sum()/total
    return sum_portion


def get_1000_not_ZZA500(vol_df, price_df, dict_index,total):
    vol_df_temp = vol_df.loc[vol_df["SecuCode"].isin(dict_index["ZZ1000"]["SecuCode"]) & ~vol_df["SecuCode"].isin(dict_index["ZZA500"]["SecuCode"])]
    price_df_temp = price_df.loc[price_df["SecuCode"].isin(dict_index["ZZ1000"]["SecuCode"]) & ~price_df["SecuCode"].isin(dict_index["ZZA500"]["SecuCode"])]
    merged_df = vol_df_temp.merge(price_df_temp, on="SecuCode", how="inner")
    merged_df["Product"] = merged_df["Volume"] * merged_df["ClosePrice"]
    sum_portion = merged_df["Product"].sum()/total
    return sum_portion


def get_1000(vol_df, price_df, dict_index,total):
    vol_df_temp = vol_df.loc[vol_df["SecuCode"].isin(dict_index["ZZ1000"]["SecuCode"])]
    price_df_temp = price_df.loc[price_df["SecuCode"].isin(dict_index["ZZ1000"]["SecuCode"])]
    merged_df = vol_df_temp.merge(price_df, on="SecuCode", how="inner")
    merged_df["Product"] = merged_df["Volume"] * merged_df["ClosePrice"]
    sum_portion = merged_df["Product"].sum()/total
    return sum_portion


def get_not_1800(vol_df, price_df, dict_index,total):
    vol_df_temp = vol_df.loc[~vol_df["SecuCode"].isin(dict_index["ZZ1000"]["SecuCode"]) & ~vol_df["SecuCode"].isin(dict_index["ZZ500"]["SecuCode"]) & ~vol_df["SecuCode"].isin(dict_index["HS300"]["SecuCode"])]
    price_df_temp = price_df.loc[~price_df["SecuCode"].isin(dict_index["ZZ1000"]["SecuCode"]) & ~price_df["SecuCode"].isin(dict_index["ZZ500"]["SecuCode"]) & ~price_df["SecuCode"].isin(dict_index["HS300"]["SecuCode"])]
    merged_df = vol_df_temp.merge(price_df_temp, on="SecuCode", how="inner")
    len(merged_df)
    merged_df["Product"] = merged_df["Volume"] * merged_df["ClosePrice"]
    sum_portion = merged_df["Product"].sum()/total
    return sum_portion


def get_total_amount(vol_df, price_df):
    merged_df = vol_df.merge(price_df,on="SecuCode",how="inner")
    merged_df["Product"] = merged_df["Volume"] * merged_df["ClosePrice"]
    total_amount = merged_df["Product"].sum()
    return total_amount


def calculate_index_weight_ratio_daily(curdate, product):
    df_weight = get_position(curdate, product)[['SecuCode', 'Volume']]
    if df_weight.empty:
        return None
    df_price = get_price(curdate, curdate).reset_index()[['SecuCode', 'ClosePrice']]
    df_weight = pd.merge(df_weight, df_price, on='SecuCode', how='left')
    df_weight['Value'] = df_weight['Volume'] * df_weight['ClosePrice']
    df_weight['Weight'] = df_weight['Value'] / df_weight['Value'].sum()

    dict_index_stock_list = get_stockdaily_indexweight_dict(curdate)
    # df_weight = dict_index_stock_list['ZZA500'].rename({'IndexWeight': 'Weight'}, axis='columns')
    dic_index_weight = {'Product': product, 'Date': curdate}
    dic_index_weight['Nums'] = len(df_weight)

    dic_index_weight['HS300'] = df_weight[
        df_weight['SecuCode'].isin(dict_index_stock_list['HS300']['SecuCode'].to_list())]['Weight'].sum()
    dic_index_weight['HS300-ZZA500'] = df_weight[
        df_weight['SecuCode'].isin(dict_index_stock_list['HS300']['SecuCode'].to_list()) & 
        (~ df_weight['SecuCode'].isin(dict_index_stock_list['ZZA500']['SecuCode'].to_list()))]['Weight'].sum()
    dic_index_weight['ZZ500'] = df_weight[
        df_weight['SecuCode'].isin(dict_index_stock_list['ZZ500']['SecuCode'].to_list())]['Weight'].sum()
    dic_index_weight['ZZ500-ZZA500'] = df_weight[
        df_weight['SecuCode'].isin(dict_index_stock_list['ZZ500']['SecuCode'].to_list()) & 
        (~ df_weight['SecuCode'].isin(dict_index_stock_list['ZZA500']['SecuCode'].to_list()))]['Weight'].sum()
    dic_index_weight['ZZA500'] = df_weight[
        df_weight['SecuCode'].isin(dict_index_stock_list['ZZA500']['SecuCode'].to_list())]['Weight'].sum()
    dic_index_weight['ZZ1000'] = df_weight[
        df_weight['SecuCode'].isin(dict_index_stock_list['ZZ1000']['SecuCode'].to_list())]['Weight'].sum()
    dic_index_weight['ZZ1000-ZZA500'] = df_weight[
        df_weight['SecuCode'].isin(dict_index_stock_list['ZZ1000']['SecuCode'].to_list()) & 
        (~ df_weight['SecuCode'].isin(dict_index_stock_list['ZZA500']['SecuCode'].to_list()))]['Weight'].sum()
    dic_index_weight['NotIn1800'] = df_weight[
        ~ df_weight['SecuCode'].isin(dict_index_stock_list['HS300']['SecuCode'].to_list() + 
                                     dict_index_stock_list['ZZ500']['SecuCode'].to_list() + 
                                     dict_index_stock_list['ZZ1000']['SecuCode'].to_list())]['Weight'].sum()
    dic_index_weight['ZZ2000'] = df_weight[
        df_weight['SecuCode'].isin(dict_index_stock_list['ZZ2000']['SecuCode'].to_list())]['Weight'].sum()

    dic_index_weight['科创板'] = df_weight[df_weight['SecuCode'].str.startswith('688')]['Weight'].sum()
    dic_index_weight['创业板'] = df_weight[df_weight['SecuCode'].str.startswith('3')]['Weight'].sum()
    dic_index_weight['深圳主板'] = df_weight[df_weight['SecuCode'].str.startswith('0')]['Weight'].sum()
    print(dic_index_weight)

    return dic_index_weight




if __name__ == '__main__':
    curdate, flag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S').split('-')
    # from Analyze_intradaily import *
    curdate = '20250331'
    df_price = get_price(curdate, curdate).reset_index() 
    df = get_position(curdate, 'A500ZQ1')
    
    total_amount = get_total_amount(df[["SecuCode","Volume"]], df_price[["SecuCode","ClosePrice"]])
    
    
    HS300_not_ZZA500 = get_300_not_ZZA500(df[["SecuCode","Volume"]], df_price[["SecuCode","ClosePrice"]],dict_index,
                                          total_amount)
    
    ZZ500_not_ZZA500 = get_500_not_ZZA500(df[["SecuCode","Volume"]], df_price[["SecuCode","ClosePrice"]],dict_index,
                                          total_amount)
    
    ZZA500 = get_ZZA500(df[["SecuCode","Volume"]], df_price[["SecuCode","ClosePrice"]],dict_index,
                                          total_amount)
    
    ZZ1000 = get_1000(df[["SecuCode","Volume"]], df_price[["SecuCode","ClosePrice"]],dict_index,
                                         total_amount)
    
    not_1800 = get_not_1800(df[["SecuCode","Volume"]], df_price[["SecuCode","ClosePrice"]],dict_index,
                                          total_amount)
    



