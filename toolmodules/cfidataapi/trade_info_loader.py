#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

sys.path.append('./')
sys.path.append('../')

from toolmodules.cfidataapi.data_api import load_mat
from toolmodules.cfidataapi.logger import CfiLogger

logger = CfiLogger.get_logger(__file__)

# alpha_order_md_root = "/mnt/nas-6/ProcessedData/PTA/alpha_order_md"
# alpha_trade_root = "/mnt/nas-6/ProcessedData/PTA/alpha_trade"
#
# production_dir_map = {
#     "SZ": "/mnt/nas-v/ProcessedData/stock_tick_gen_ss/guotai/full_mat_dp5_diff",
#     "SH": "/mnt/nas-v/ProcessedData/stock_sh_tick_gen/full_dp5_diff"
# }

if 'win' in sys.platform:
    # alpha_order_md_root = "S:/ProcessedData/PTA/alpha_order_md"
    # alpha_trade_root = "S:/ProcessedData/PTA/alpha_trade"
    # alpha_order_md_ready_flag_dir = "S:/ProcessedData/PTA/alpha_order_md/ReadyFlag"
    # alpha_trade_ready_flag_dir = "S:/ProcessedData/PTA/alpha_trade/ReadyFlag"
    
    alpha_order_md_root = "V:/ProcessedData/PTA/alpha_order_md"
    alpha_trade_root = "V:/ProcessedData/PTA/alpha_trade"
    alpha_order_md_ready_flag_dir = "V:/ProcessedData/PTA/alpha_order_md/ReadyFlag"
    alpha_trade_ready_flag_dir = "V:/ProcessedData/PTA/alpha_trade/ReadyFlag"

    production_dir_map = {
        "SZ": "V:/ProcessedData/stock_tick_gen_ss/guotai/full_mat_dp5_diff",
        "SH": "V:/ProcessedData/stock_sh_tick_gen/full_dp5_diff"
    }
else:
    # alpha_order_md_root = "/mnt/nas-6/ProcessedData/PTA/alpha_order_md"
    # alpha_trade_root = "/mnt/nas-6/ProcessedData/PTA/alpha_trade"
    # alpha_order_md_ready_flag_dir = "/mnt/nas-6/ProcessedData/PTA/alpha_order_md/ReadyFlag"
    # alpha_trade_ready_flag_dir = "/mnt/nas-6/ProcessedData/PTA/alpha_trade/ReadyFlag"
    
    alpha_order_md_root = "/mnt/nas-3/ProcessedData/PTA/alpha_order_md"
    alpha_trade_root = "/mnt/nas-3/ProcessedData/PTA/alpha_trade"
    alpha_order_md_ready_flag_dir = "/mnt/nas-3/ProcessedData/PTA/alpha_order_md/ReadyFlag"
    alpha_trade_ready_flag_dir = "/mnt/nas-3/ProcessedData/PTA/alpha_trade/ReadyFlag"

    production_dir_map = {
        "SZ": "/mnt/nas-v/ProcessedData/stock_tick_gen_ss/guotai/full_mat_dp5_diff",
        "SH": "/mnt/nas-v/ProcessedData/stock_sh_tick_gen/full_dp5_diff"
    }


def load_alpha_order(alpha_path):
    """
        load alpha_order
    """
    #cols date <= 20221111
    #alpha_order_column = ["localtime", "order_exchtime", "insid",
    #        "wf-md_time", "wf-order_time", "direction", "volume",
    #        "price", "wf-orderid", "weight_ref"]
    alpha_order_column = ["localtime", "order_exchtime", "insid", "wf-md_time",
            "md_decode_time", "strategy_time", "wf-order_time", "direction", "volume",
            "price", "wf-orderid", "pre_quota", "target_smooth", "weight_ref", "signal", "t0-status"]

    alpha_order_df = pd.read_csv(alpha_path, header=None, names=alpha_order_column)
    alpha_order_df.sort_values(["order_exchtime", "localtime"], inplace=True)
    alpha_order_df.reset_index(inplace=True, drop=True)
    return alpha_order_df


def get_trade_info(insid, date, colo_list=None):
    engine = create_engine(f"mssql+pymssql://public_data:public_data@dbs.cfi/DataSupply?charset=utf8")
    query_account_info_sql = f"select * from trading_account where [date]={date}"

    account_df = pd.read_sql(query_account_info_sql, con=engine)
    if colo_list is None:
        colo_list = list(account_df["colo"].unique())

    alpha_order_md_ready_flag_path = os.path.join(alpha_order_md_ready_flag_dir, date, "DataReadyFlag.txt")
    if int(date) > 20221123 and not os.path.exists(alpha_order_md_ready_flag_path):
        raise RuntimeError(f"alpha_order_md of {date} not ready")

    alpha_trade_ready_flag_path = os.path.join(alpha_trade_ready_flag_dir, date, "DataReadyFlag.txt")
    if int(date) > 20221123 and not os.path.exists(alpha_trade_ready_flag_path):
        raise RuntimeError(f"alpha_trade of {date} not ready")

    trade_info_list = []
    for colo in colo_list:
        order_md_path = os.path.join(alpha_order_md_root, colo, date, f"{insid}-{date}.csv")
        trade_path = os.path.join(alpha_trade_root, colo, date, f"{insid}-{date}.csv")

        if os.path.exists(order_md_path):
            order_md_df = pd.read_csv(order_md_path)
            order_md_df.rename({"insid_x": "insid"}, axis=1, inplace=True)
            order_md_df = order_md_df[~order_md_df["wf-md_id"].isnull()].copy()
        else:
            continue

        alpha_trade_cols = [
            "trade_localtime", "trade_exchtime", "insid", "wf-orderid", "direction", "trade_volume", "trade_price"]
        trade_df = pd.DataFrame(columns=alpha_trade_cols)
        if os.path.exists(trade_path):
            trade_df = pd.read_csv(trade_path, header=None, names=alpha_trade_cols)
            trade_df = trade_df[(trade_df["trade_price"] > 0) & (trade_df["trade_volume"] > 0)].copy()

        trade_df = pd.merge(trade_df, order_md_df[["wf-orderid", "localtime"]], on="wf-orderid", how="left")

        trade_localtime = pd.to_timedelta(trade_df["trade_localtime"])
        order_localtime = pd.to_timedelta(trade_df["localtime"]) 

        # filter_mask = (~order_localtime.isnull()) & \
        #               (order_localtime < trade_localtime) & \
        #               (order_localtime + pd.Timedelta(20, unit="s") > trade_localtime)
        filter_mask = (~order_localtime.isnull()) & \
                      (order_localtime < trade_localtime) & \
                      (order_localtime + pd.Timedelta(1830, unit="s") > trade_localtime)
        trade_df = trade_df[filter_mask]

        order_md_trade_df = pd.merge(order_md_df, trade_df, on=["localtime", "wf-orderid", "insid", "direction"], how="left")
        order_md_trade_df["colo"] = colo
        trade_info_list.append(order_md_trade_df)
    if trade_info_list:
        trade_info_df = pd.concat(trade_info_list, axis=0)
        if trade_info_df.empty:
            return trade_info_df

        trade_info_df["stid"] = trade_info_df["wf-orderid"].apply(lambda t: int(t) >> 52)
        trade_info_df = pd.merge(trade_info_df, account_df[["colo", "stid", "production"]], on=["colo", "stid"])
        trade_info_df.reset_index(inplace=True, drop=True)
        return trade_info_df
    else:
        return pd.DataFrame()


def trade_agg_func(order_md_df):
    order_md_list = []
    if order_md_df.empty:
        return pd.DataFrame()

    for (wf_orderid, colo, wf_order_time), orderided_df in order_md_df.groupby(["wf-orderid", "colo", "wf-order_time"]):
        order_md_list.append(
            {
                "wf-orderid": wf_orderid,
                "insid": orderided_df["insid"].iloc[0],
                "localtime": orderided_df["localtime"].iloc[0],
                "order_exchtime": orderided_df["order_exchtime"].iloc[0],
                "wf-order_time": wf_order_time,
                "direction": orderided_df["direction"].iloc[0],
                "volume": orderided_df["volume"].iloc[0],
                "price": orderided_df["price"].iloc[0],
                "weight_ref": orderided_df["weight_ref"].iloc[0],

                "wf-md_id": orderided_df["wf-md_id"].iloc[0],
                "volume_acc_seq": orderided_df["volume_acc_seq"].iloc[0],
                "wf-md_time": orderided_df["wf-md_time"].iloc[0],
                "md_exchtime": orderided_df["md_exchtime"].iloc[0],
                "last": orderided_df["last"].iloc[0],
                "volume_acc": orderided_df["volume_acc"].iloc[0],
                "turnover_acc": orderided_df["turnover_acc"].iloc[0],
                "ap1": orderided_df["ap1"].iloc[0],
                "bp1": orderided_df["bp1"].iloc[0],
                "av1": orderided_df["av1"].iloc[0],
                "bv1": orderided_df["bv1"].iloc[0],
                "signal": orderided_df["signal"].iloc[0],
                "t0-status": orderided_df["t0-status"].iloc[0],
                "trade_volume": np.nansum(orderided_df["trade_volume"]),
                "trade_turnover": np.nansum(orderided_df["trade_volume"] * orderided_df["trade_price"]),
                "production": orderided_df["production"].iloc[0],
                "colo": orderided_df["colo"].iloc[0],
          }
        )

    return pd.DataFrame(order_md_list)


def backtest_md_loader(insid, date):
    exch = insid[:2]
    backtest_md_path = os.path.join(production_dir_map[exch], date, f"{insid}-{date}.mat")

    column_name = ["localtime", "bp1", "ap1", "bv1", "av1", "last", "volume", "turnover", "tag"]
    last_cols45 = [item for j in [(f"bp{i}", f"bv{i}", f"ap{i}", f"av{i}") for i in range(2, 11)] for item in j]
    column_name_45 = column_name + last_cols45

    exch = os.path.basename(backtest_md_path)[:2]

    if exch == "SZ":
        cols = ["localtime", "bp1", "bv1", "ap1", "av1"] + column_name_45[5:]
    elif exch == "SH":
        cols = ["localtime", "bp1", "bv1", "ap1", "av1"] + column_name_45[5:] + ["ns_time"]
    # cols = ["localtime", "bp1", "bv1", "ap1", "av1"] + column_name_45[5:] + ["ns_time"]

    if not os.path.exists(backtest_md_path):
        return pd.DataFrame()

    backtest_md_df = load_mat(backtest_md_path, columns=cols)
    backtest_md_df["volume_acc"] = backtest_md_df["volume"].cumsum()
    backtest_md_df["turnover_acc"] = backtest_md_df["turnover"].cumsum()

    volume_acc_seq = pd.Series(range(len(backtest_md_df)))
    volume_acc_inc_mask = (backtest_md_df["volume_acc"] - backtest_md_df["volume_acc"].shift(1, fill_value=0)) > 0 

    volume_ladder = pd.Series([np.nan] * len(backtest_md_df))
    volume_ladder[volume_acc_inc_mask] = volume_acc_seq[volume_acc_inc_mask]
    volume_ladder.fillna(method="ffill", inplace=True)
    volume_ladder.fillna(0, inplace=True)
    backtest_md_df["volume_acc_seq"] = volume_acc_seq - volume_ladder
    backtest_md_df["volume_acc_seq"] = np.uint(backtest_md_df["volume_acc_seq"])
    backtest_md_df["index"] = backtest_md_df.index
  
    return backtest_md_df 


def wf_backtest_md_match(insid, date):
    backtest_md_df = backtest_md_loader(insid, date)
    wf_md_order_df = get_trade_info(insid, date)

    if backtest_md_df.empty or wf_md_order_df.empty:
        return pd.DataFrame()

    ask_bid_cols = [item for j in [(f"bp{i}", f"bv{i}", f"ap{i}", f"av{i}") for i in range(1, 6)] for item in j]
    trading_cols = ["last", "volume_acc", "volume_acc_seq"]

    merge_key_list = ask_bid_cols + trading_cols
    backtest_md_use_cols = merge_key_list + ["index"]
    merged_df = pd.merge(wf_md_order_df, backtest_md_df[backtest_md_use_cols], how="left", on=merge_key_list)
    match_mask = ~(merged_df["index"].isnull())

    match_volume_acc_seq_len = 0
    no_match_len = 0
    if (~match_mask).any():
        merge_key_list = ["volume_acc", "volume_acc_seq"]
        backtest_md_use_cols = merge_key_list + ["index"]

        no_match_wf_md_order_index = wf_md_order_df[~match_mask].index
        merged_volume_acc_seq_df = pd.merge(wf_md_order_df[~match_mask], backtest_md_df[backtest_md_use_cols], 
                how="left", on=merge_key_list)
        merged_volume_acc_seq_df.index = no_match_wf_md_order_index

        volume_acc_seq_match_mask = ~(merged_volume_acc_seq_df["index"].isnull())
        no_match_len = np.sum(~volume_acc_seq_match_mask)
        match_volume_acc_seq_len = np.sum(volume_acc_seq_match_mask)

        merged_df.loc[~match_mask, "index"] = merged_volume_acc_seq_df["index"]

    match_ask_bid_len = np.sum(match_mask)
    if match_volume_acc_seq_len > 0:
        logger.info(f"{insid} match_bid_ask_len: {match_ask_bid_len}, match_volume_acc_seq_len {match_volume_acc_seq_len}, no_match_volume_seq_len: {no_match_len}")

    volume_acc_seq_no_match_mask = merged_df["index"].isnull()
    for no_match_volume_acc in merged_df[volume_acc_seq_no_match_mask]["volume_acc"].unique():
        no_greater_mask = backtest_md_df["volume_acc"] <= no_match_volume_acc
        no_less_mask = backtest_md_df["volume_acc"] >= no_match_volume_acc
        equ_mask = no_greater_mask & no_less_mask
        if equ_mask.any():
            merged_df.loc[merged_df["volume_acc"] == no_match_volume_acc, "index"] = backtest_md_df.loc[equ_mask, "index"].iloc[-1]
        elif no_less_mask.any():
            merged_df.loc[merged_df["volume_acc"] == no_match_volume_acc, "index"] = backtest_md_df.loc[no_less_mask, "index"].iloc[0]

    index_null_mask = merged_df["index"].isnull()
    if index_null_mask.any():
        logger.info(f"{insid} no match count {np.nansum(index_null_mask)}")
    else:
        merged_df["index"] = np.uint(merged_df["index"])

    merged_df.reset_index(inplace=True, drop=True)
    return merged_df



if __name__ == "__main__":
    pass
    #df = get_trade_info("SH600000", "20220829")
    #print(df.shape)
