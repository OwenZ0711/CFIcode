#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd
import io
import libconf
import json
import subprocess
import datetime
import pymssql
from sqlalchemy import create_engine

from toolmodules.utils.parallelism import Parallelism
from toolmodules.utils.classifier_base import GeneralClassifier
from toolmodules.utils.logger import CfiLogger
from toolmodules.modules_constvars import Temp_Colo_Mechine_Dict
logger = CfiLogger.get_logger(__file__)


def pre_trading_day(day):
    conn = pymssql.connect('dbs.cfi:1433', 'DataSupply', 'Cfi888_', 'JYDB', charset="utf8", as_dict=True)
    if conn is None:
        raise Exception('db connected error')

    sql = "DECLARE  @QueryDate DATETIME" \
            "\nSET @QueryDate = '%s'" \
            "\nSELECT TOP 1 convert(varchar(10), TradingDate,112) AS TradingDate" \
            "\nFROM" \
            "\n(" \
            "\n        SELECT DISTINCT TradingDate" \
            "\n        FROM QT_TradingDayNew A" \
            "\n        LEFT JOIN CT_SystemConst B ON A.SecuMarket = B.DM" \
            "\n        WHERE B.DM IN (90, 83)" \
            "\n        AND B.LB=201" \
            "\n        AND A.IfTradingDay=1" \
            "\n        AND A.TradingDate < @QueryDate" \
            "\n        ) AA" \
            "\nORDER BY TradingDate DESC;" % day

    cursor = conn.cursor()
    cursor.execute(sql)
    row = cursor.fetchone()

    if row is None:
        raise Exception('SQL excuted error')

    return row["TradingDate"]


def next_trading_day(day):
    conn = pymssql.connect('dbs.cfi:1433', 'DataSupply', 'Cfi888_', 'JYDB', charset="utf8", as_dict=True)
    if conn is None:
        raise Exception('db connected error')

    sql = "DECLARE @QueryDate DATETIME" \
           "\nSET @QueryDate = '%s'" \
           "\nSELECT TOP 1 convert(varchar(10), TradingDate,112) AS TradingDate" \
           "\nFROM" \
           "\n(" \
           "\n        SELECT DISTINCT TradingDate" \
           "\n        FROM QT_TradingDayNew A" \
           "\n        LEFT JOIN CT_SystemConst B ON A.SecuMarket = B.DM" \
           "\n        WHERE B.DM IN (90, 83)" \
           "\n        AND B.LB=201" \
           "\n        AND A.IfTradingDay=1" \
           "\n        AND A.TradingDate > @QueryDate" \
           "\n        ) AA" \
           "\nORDER BY TradingDate ASC;" % day
  
    cursor = conn.cursor()
    cursor.execute(sql)
    row = cursor.fetchone()
  
    if row is None:
        raise Exception('SQL excuted error')
  
  #conn.close()
    return row["TradingDate"]


def get_trader_cfg(date):
    colo_list = get_colo_list(date)
    account_df_list = []
    for colo in colo_list:
        trader_list_path = f"/mnt/nas-3/ProcessedData/PTA/parsed_log/{colo}/{date}/trader-list.cfg"
        if not os.path.exists(trader_list_path):
            account_df_list.append(
                pd.DataFrame(columns=["trade_acc", "remote_colo", "exchange", "colo"])
            )
        else:
            with io.open(trader_list_path) as f:
                logger.info(trader_list_path)
                config = libconf.load(f)
                account_df = pd.DataFrame(
                    [
                        {
                            "trade_acc": t["name"],
                            "remote_colo": t["remote_colo"] if "remote_colo" in t else "",
                            "exchange": t["exchange"] if "exchange" in t else "SZSE"
                        } for t in config["trader"]
                    ]
                )

                if account_df.empty:
                    logger.info(f"{trader_list_path} empty")
                    continue

                account_df["exchange"] = account_df["exchange"].apply(lambda t: "SZ" if t == "SZSE" else t)
                account_df["exchange"] = account_df["exchange"].apply(lambda t: "SH" if t == "SHSE" else t)
                account_df["remote_colo"].replace("", np.nan, inplace=True)

                account_df["colo"] = colo
                account_df_list.append(account_df)

    return pd.concat(account_df_list, names=["trade_acc", "remote_colo", "exchange", "colo"], axis=0)


def get_colo_list(date):
    """
    """
    engine = create_engine(f"mssql+pymssql://datasupply:Cfi888_@dbs.cfi/DataSupply?charset=utf8")
    colo_list = pd.read_sql(f"select distinct [colo] from trading_account where [date] = {date}", con=engine)["colo"].tolist()

    if len(colo_list) == 0:
        colo_list = open("/mnt/nas-3/webmonitor_data/colo_list_trading.txt", "r").read().strip().split(",")
        colo_list = list(set(colo_list) - set(Temp_Colo_Mechine_Dict['simulist']))

        extra_colo_list = []
        black_list = []
        colo_list = [t for t in colo_list if t not in black_list] + extra_colo_list

    normal_colo_list = [t for t in colo_list if t.count("-") == 2]
    abnormal_colo_list = sorted([t for t in set(colo_list) if t.count("-") != 2])

    colo_list = normal_colo_list
    colo_array = sorted(["-".join(np.array(t.split("-"))[[0,2,1]]) if t.count("-") > 1 else t for t in colo_list])
    colo_list = ["-".join(np.array(str(t).split("-"))[[0,2,1]]) if str(t).count("-") > 1 else t for t in colo_array]
    return colo_list + abnormal_colo_list


def read_trading_account_from_db(date):
    engine = create_engine(f"mssql+pymssql://datasupply:Cfi888_@dbs.cfi/DataSupply?charset=utf8")
    trading_account_df = pd.read_sql(f"select * from trading_account where [date]='{date}'", engine)
    return trading_account_df


def run_cmd(cmd, host=None, port=None, **kwargs):
    run_args = {
        "check": True,
    }
    run_args.update(kwargs)
    if host:
        cmd_arg = ["ssh", "-q", str(host)]
        if port:
            cmd_arg.extend(["-p", str(int(port))])
        cmd_arg.append(cmd)
        logger.info(f"Running cmd:{cmd_arg}")
        logger.info(f"Running cmd," + " ".join(f"'{_x}'" for _x in cmd_arg))
        return subprocess.run(cmd_arg, **run_args)
    else:
        cmd_arg = cmd 
        if isinstance(cmd, str):
            run_args["shell"] = True
            logger.info(f"Running cmd,{cmd_arg}")
            return subprocess.run(cmd_arg, **run_args)
        else:
            logger.info(f"Running cmd," + " ".join(f"'{_x}'" for _x in cmd_arg))
            return subprocess.run(cmd_arg, **run_args)


class PtaTaskProfile(object):
    def __init__(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_logs_profile.json"), "r") as f:
            self.trade_log_profile = json.loads(f.read())

    def special_care_production(self):
        return self.trade_log_profile["SpecialCare"]["types"]

    def special_care_display_columns(self):
        return self.trade_log_profile["SpecialCare"]["DisplayColumns"]

    def local_report_dir(self):
        return self.trade_log_profile["local_path"]["report"]

    def local_parsed_dir(self):
        return self.trade_log_profile["local_path"]["parsed_log"]

    def local_chart_dir(self):
        return self.trade_log_profile["local_path"]["chart"]

    def nas_report_dir(self):
        return self.trade_log_profile["nas_path"]["report"]

    def nas_parsed_dir(self):
        return self.trade_log_profile["nas_path"]["parsed_log"]

    def email_server(self):
        return self.trade_log_profile["notify"]["email"]["smtp_server"]

    def email_account(self):
        return self.trade_log_profile["notify"]["email"]["account"]

    def email_passwd(self):
        return self.trade_log_profile["notify"]["email"]["passwd"]

    def email_send_to(self):
        return self.trade_log_profile["notify"]["email"]["sendto"]

    def err_email_send_to(self):
        return self.trade_log_profile["err_notify"]

    def machines_keys(self):
        return list(self.trade_log_profile["machines"].keys())

    def cpu_name(self, colo_name, date):
        machine_info_file_path = os.path.join(self.local_parsed_dir(), colo_name, date, "machine_info.txt")
        if not os.path.exists(machine_info_file_path):
            machine_info_file_path = os.path.join(self.nas_parsed_dir(), colo_name, date, "machine_info.txt")

        cmd = f"cat {machine_info_file_path} | awk -F '[:：]' '/^型号名称：|^Model name:/ {{print $2$4}}' | sed 's/^ *//g'"
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
        cpu_name = str(ret.stdout, encoding="utf-8").strip()
        return cpu_name

    def machine_cpu_freq(self, colo_name, date):
        machine_info_file_path = os.path.join(self.local_parsed_dir(), colo_name, date, "machine_info.txt")
        if not os.path.exists(machine_info_file_path):
            machine_info_file_path = os.path.join(self.nas_parsed_dir(), colo_name, date, "machine_info.txt")

        cmd = f"""awk '/Refined TSC clocksource calibration:/ {{ if ($9 == "MHz") print $8}}' {machine_info_file_path}""" 
        tsc_ret = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
        tsc_str = str(tsc_ret.stdout, encoding="utf-8").strip()
        if tsc_str != "":
            return float(tsc_str) * 1e3

        logger.info(f"{colo_name} {date} cpuinfo not found")
        cpu_name = self.cpu_name(colo_name, date)

        if cpu_name not in self.trade_log_profile["cpuinfo"]:
            logger.info(colo_name, cpu_name)

        return self.trade_log_profile["cpuinfo"][cpu_name] * 1e6

    def jumper_info(self, hostname):
        if hostname in self.trade_log_profile["machines"] and \
            "jumper_info" in self.trade_log_profile["machines"][hostname]:
            return self.trade_log_profile["machines"][hostname]["jumper_info"]
        else:
            return self.trade_log_profile["jumper_info"]

    def machine_user(self, hostname):
        return self.jumper_info(hostname)["username"]

    def machine_ip(self, hostname):
        return self.jumper_info(hostname)["host"]

    def machine_port(self, hostname):
        return self.jumper_info(hostname)["port"]

    def machine_parsed_log(self, hostname):
        return self.jumper_info(hostname)["parsed_log"]

    def colo_worker_download(self):
        return self.trade_log_profile["worker_download"]

    def nas_raw_dir(self):
        return self.trade_log_profile["nas_raw_dir"]

    def exception(self):
        exception_dict = self.trade_log_profile["exception"]
        return exception_dict 

    def pta_local_root(self):
        return self.trade_log_profile["PTA_ROOT"]

    def download_dir(self, category):
        return os.path.join(self.pta_local_root(), "RawData", category)

    def extract_dir(self, category):
        return os.path.join(self.pta_local_root(), "ExtractRawData", category)

    def intermediate_dir(self, category):
        return os.path.join(self.pta_local_root(), "ProcessedData", category)

    def clean_up_remote_dag(self):
        def _handle(**context):
            task_date = context["tomorrow_ds_nodash"]
            today = datetime.datetime.now().strftime("%Y%m%d")
            machines = self.machines_keys()
            cmd_list = []
            for m_key in machines:
                host_ip = self.machine_ip(m_key)
                user = self.machine_user(m_key)
                port = self.machine_port(m_key)
                data_dir = self.machine_parsed_log(m_key)
                if today == task_date:
                    cmd = f"rm -rf {data_dir}"
                else:
                    cmd = f"find {data_dir} -mindepth 3 -type d | grep -v {today} | xargs rm -rf"

                cmd_list.append((cmd, f"{user}@{host_ip}", port))

            for cmd_para in set(cmd_list):
                run_cmd(cmd_para[0], cmd_para[1], cmd_para[2])

        return _handle

pta_task_profile = PtaTaskProfile()


def load_alpha_order_md(op_date, production):
    order_md_dir = f"/data2/andrew/market_impact/data/{op_date}/arranged/alpha_order_md/{op_date}/{production}"
    if not os.path.exists(order_md_dir):
        return pd.DataFrame(columns=["insid", "wf-orderid"])

    para_list = [(os.path.join(order_md_dir, t), ) for t in os.listdir(order_md_dir)]
    pl =  Parallelism()
    pl.run(pd.read_csv, para_list)
    order_md_df = pd.concat(pl.get_results(), axis=0)
    return order_md_df 


def freq_analysis_monthly(op_date):
    morning_timeindex_idx = pd.timedelta_range(start="09:30:00", end="11:30:00", freq="1s")
    after_timeindex_idx = pd.timedelta_range(start="13:00:00", end="15:00:00", freq="1s")
    timeindex_idx = morning_timeindex_idx.append(after_timeindex_idx)

    time_dict = {
        str(t)[7:15]: t for t in timeindex_idx 
    }

    classifier = GeneralClassifier(time_dict)
    trade_cfg_df = read_trading_account_from_db(op_date)

    colo_list = get_colo_list(op_date)
    trader_list_cfg_df = get_trader_cfg(op_date)
    cfg_df = pd.merge(trader_list_cfg_df, trade_cfg_df[["colo",  "stid", "trade_acc", "production"]], on=["colo", "trade_acc"], how="inner")

    order_list = []
    for colo in colo_list:
        nas_parsed_dir = pta_task_profile.nas_parsed_dir()
        csv_path = os.path.join(nas_parsed_dir, colo, op_date, f"{colo}-worker-{op_date}.csv")
        order_df = pd.read_csv(csv_path)
        print(colo, colo_list.index(colo))
        order_df["colo"] = colo
        order_list.append(order_df)


    order_df = pd.concat(order_list, axis=0)
    order_df = order_df[order_df["send_err_time"].isnull()]
    #order_df = order_df[order_df["send_err_time"].isnull()]
    order_df = pd.merge(order_df, cfg_df[["colo", "stid", "production", "remote_colo", "exchange"]], on=["colo", "stid"], how="left")
    order_df["exch"] = order_df["insid"].apply(lambda t: t[:2])

    colo_filter_mask = order_df["remote_colo"].isnull() | \
                       (order_df["exchange"] == order_df["exch"])
    order_df = order_df[colo_filter_mask]

    order_df.rename({"orderid": "wf-orderid"}, inplace=True, axis=1)

    r_df = pd.read_excel("report 202310.xls")
    r_df.rename({"内部代码": "production"}, axis=1, inplace=True)
    print(r_df.head(10))
    order_df = pd.merge(r_df, order_df, on="production", how="inner")


  #  order_list = []
  #  cnt = 0
  #  for p, p_order_df in order_df.groupby("production"):
  #      cnt = cnt + 1
        #print(p, cnt)
  #      alpha_order_md_df = load_alpha_order_md(op_date, p) 
  #      alpha_order_md_df.drop_duplicates(["insid", "wf-orderid"], inplace=True)
  #      o_df = pd.merge(p_order_df, alpha_order_md_df[["insid", "wf-orderid"]], on=["insid", "wf-orderid"], how="inner")
  #      order_list.append(o_df)
  #  order_df = pd.concat(order_list, axis=0)

    order_df.reset_index(inplace=True, drop=True)
    order_df["md_exchtime"] = pd.to_timedelta(order_df["md_exchtime"])
    order_df["flag"] = classifier.classify(order_df["md_exchtime"])

    print(order_df.head(10))
    order_freq_df = pd.DataFrame(
        [
            {
                "production": p,
                "flag": flag,
                "exch": exch,
                "order_cnt": len(flag_df),
                "cancel_cnt": np.nansum(~flag_df["cancel_time"].isnull())
            } for (p, flag, exch), flag_df in order_df.groupby(["原始产品名称", "flag", "exch"])
            #} for (p, flag, exch), flag_df in order_df.groupby(["production", "flag", "exch"])
        ]
    )

    #print(order_freq_df)

    order_freq_summary_df = pd.DataFrame([
        {
            "production": p,

            "order_freq": np.nanmax(p_df["order_cnt"]),
            "cancel_freq": np.nanmax(p_df["cancel_cnt"]),
            "order_and_cancel_freq": np.nanmax(p_df["order_cnt"] + p_df["cancel_cnt"]),

            "order_daily_sum": np.nansum(p_df["order_cnt"]),
            "cancel_daily_sum": np.nansum(p_df["cancel_cnt"]),
            "order_and_cancel_daily_sum": np.nansum(p_df["order_cnt"] + p_df["cancel_cnt"]),
        }
        for p, p_df in order_freq_df.groupby(["production"])
    ]
    )
    return order_freq_summary_df


def order_freq_analysis(start_date, end_date):
    start_trading_day = next_trading_day(pre_trading_day(start_date))
    trading_day = start_trading_day

    para_list = []
    while trading_day <= end_date:
        para_list.append((trading_day,))
        trading_day = next_trading_day(trading_day)

    pl = Parallelism(processes=8)
    pl.run(freq_analysis_monthly, para_list)
    order_freq_df = pd.concat(pl.get_results(), axis=0)

    p_list = []
    for p, p_df in order_freq_df.groupby(["production"]):
        print(np.nanargmax(p_df["order_and_cancel_daily_sum"]))
        p_list.append(
            {
                "production": p,
                "max_order_and_cancel_daily_sum": np.nanmax(p_df["order_and_cancel_daily_sum"]),
                "order_and_cancel_freq": np.nanmax(p_df["order_and_cancel_freq"])
                #"max_date": p_df["date"].iloc[np.nanargmax(p_df["order_and_cancel_daily_sum"])]
            }
        )

    p_df = pd.DataFrame(p_list)
    p_df.to_excel(f"{start_date}-{end_date}.xlsx", index=False)


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print(f"usage {sys.argv[0]} <start_date> <end_date>")
    #     sys.exit(-1)
    #
    # start_date, end_date = sys.argv[1], sys.argv[2]

    start_date, end_date = '20231101', '20231130'
    order_freq_analysis(start_date, end_date)
    #load_alpha_order_md(op_date, "DC9")
