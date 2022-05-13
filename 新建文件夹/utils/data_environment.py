import numpy as np
import pandas as pd
import os
import sys
from glob import glob

import warnings

from importlib import reload


warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

from . import config
from . import kiwi_operators as kwo

# ==========================================================================================
# status_filter
trade_status = pd.read_pickle(f"{config.dir_data_day}/tradeStatus_1e9_MainAdj.pkl")
status_filter = pd.DataFrame(1, index=trade_status.index, columns=trade_status.columns)
status_filter[~trade_status] = np.nan

# ==========================================================================================
# daily data
data_day = dict()

data_day["close"] = pd.read_pickle(f"{config.dir_data_day}/Close_MainAdj.pkl")
data_day["high"] = pd.read_pickle(f"{config.dir_data_day}/High_MainAdj.pkl")
data_day["low"] = pd.read_pickle(f"{config.dir_data_day}/Low_MainAdj.pkl")
data_day["open"] = pd.read_pickle(f"{config.dir_data_day}/Open_MainAdj.pkl")

data_day["openinterest"] = pd.read_pickle(
    f"{config.dir_data_day}/OpenInterest_MainAdj.pkl"
)
data_day["settle"] = pd.read_pickle(f"{config.dir_data_day}/Settle_MainAdj.pkl")
data_day["amount"] = pd.read_pickle(f"{config.dir_data_day}/Amount_MainAdj.pkl")
data_day["volume"] = pd.read_pickle(f"{config.dir_data_day}/Volume_MainAdj.pkl")
data_day["size"] = pd.read_pickle(f"{config.dir_data_day}/Size_MainAdj.pkl")

data_day["ocr"] = data_day["open"] / data_day["close"] - 1
data_day["hlr"] = data_day["high"] / data_day["low"] - 1
data_day["vwap"] = data_day["amount"] / data_day["volume"]

# ==========================================================================================
# minute data
data_min = dict()
data_min["open"] = pd.read_pickle(f"{config.dir_data_min}/open.pkl")
data_min["high"] = pd.read_pickle(f"{config.dir_data_min}/high.pkl")
data_min["low"] = pd.read_pickle(f"{config.dir_data_min}/low.pkl")
data_min["close"] = pd.read_pickle(f"{config.dir_data_min}/close.pkl")
data_min["volume"] = pd.read_pickle(f"{config.dir_data_min}/volume.pkl")
data_min["amount"] = pd.read_pickle(f"{config.dir_data_min}/amount.pkl")
data_min["vwap"] = data_min["amount"] / data_min["volume"]

for elem in data_min.keys():
    data_min[elem] = data_min[elem].reindex(columns=data_day["open"].columns)

# ==========================================================================================
# daily forward return, open-to-open
fwd_ret = data_day["open"].pct_change(config.fwd_day).shift(-(config.fwd_day + 1))
fwd_one = data_day["open"].pct_change(1).shift(-2)
fwd_ret = fwd_ret * status_filter
fwd_one = fwd_one * status_filter


data = dict()
data["day"] = data_day
data["min"] = data_min

