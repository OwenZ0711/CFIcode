#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys

sys.path.append('./')
sys.path.append('../')

import h5py
import pandas as pd
import numpy as np
import math
import datetime
from toolmodules.cfidataapi.logger import CfiLogger


_LOGGER = CfiLogger.get_logger(__file__)


def load_mat(fpath, columns, key="BarData"):
    with h5py.File(fpath, "r") as fh5:
        key_list = [key for key in fh5.keys()]
        if key not in key_list:
            errinfo = f"key {key} is not in {fpath} available keys {key_list}"
            _LOGGER.error(errinfo)
            raise RuntimeError(errinfo)

        return pd.DataFrame(np.transpose(fh5[key]), columns=columns)


def timestamp_to_datenum(stamp):
    return datetime.datetime.toordinal(stamp + datetime.timedelta(days=366)) + (
            stamp.hour * 3600 + stamp.minute * 60 + stamp.second + stamp.microsecond / 1000000) / 24 / 3600


def detenum_to_timestamp(datenum):
    decim, days = math.modf(datenum)
    date = datetime.date.fromordinal(int(days)) - datetime.timedelta(days=366)
    seconds = decim * 24 * 3600

    return pd.to_datetime(f"{date}", format="%Y-%m-%d") + datetime.timedelta(seconds=seconds)


def get_interval1min_index(index_file_path, interval_time):
    index_columns = ["localtime", "idx"]
    df = load_mat(index_file_path, index_columns, key="Time1minIdx")
    df["localtime"] = df["localtime"].apply(lambda t: detenum_to_timestamp(t))
    df["localtime"] = df["localtime"].apply(
        lambda t:
        datetime.datetime.strptime(
            (t + datetime.timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M"),
            "%Y-%m-%d %H:%M"
        )
    )

    df.set_index("localtime", inplace=True, drop=False)
    index_row = df.loc[interval_time]

    if index_row.empty:
        return None

    idx = int(index_row["idx"])
    if idx <= 0:
        return None
    return idx - 1


if __name__ == "__main__":
    idx = get_interval1min_index('/mnt/nas-3/ProcessedData/stock_interval1min_index/20210521/SH600000-20210521.mat',
                           '2021-05-21 09:16:00')
    print(idx)