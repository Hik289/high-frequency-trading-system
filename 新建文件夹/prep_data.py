import numpy as np
import pandas as pd
import os
import sys
from glob import glob
import warnings
from importlib import reload

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

from utils import config
from utils import kiwi_operators as kwo

os.makedirs(f"{config.dir_root}", exist_ok=True)
os.makedirs(f"{config.dir_feat}", exist_ok=True)
os.makedirs(f"{config.dir_selected_feat}", exist_ok=True)
os.makedirs(f"{config.dir_rolling_feat}", exist_ok=True)
os.makedirs(f"{config.dir_rolling_selected_feat}", exist_ok=True)
os.makedirs(f"{config.dir_rolling_res}", exist_ok=True)
os.makedirs(f"{config.dir_cache}", exist_ok=True)


# specially structured data should be prepared
# for daily data:
#   read data in dir_raw_data_day
#   transform index type from str to int
#   example: '2015-01-01' to 20150101
#   prepared data saved in dir_data_day
# for minute data:
#   read data in dir_raw_data_min
#   transform index type from datetime to int
#   example: 2015-01-01 09:00 to 201501010900
#   varieties will be joint together, i.e. varities will be treated as columns
# data preparation is time consuming

# ==========================================================================================
# optionally transform daily data

print("preparing daily data")
os.makedirs(f"{config.dir_data_day}", exist_ok=True)
item_lst = [
    "tradeStatus_1e9_MainAdj",
    "Close_MainAdj",
    "High_MainAdj",
    "Low_MainAdj",
    "Open_MainAdj",
    "Settle_MainAdj",
    "Amount_MainAdj",
    "Volume_MainAdj",
    "Size_MainAdj",
    "OpenInterest_MainAdj",
]
for item in item_lst:
    cur_data = pd.read_csv(f"{config.dir_raw_data_day}/{item}.csv", index_col=0)
    cur_data = kwo.transform_daily_index_str_to_int(cur_data)
    cur_data.to_pickle(f"{config.dir_data_day}/{item}.pkl")
print("daily data generated")
print()

print("preparing minute data")

# ==========================================================================================
# optionally transform minute data
os.makedirs(f"{config.dir_data_min}", exist_ok=True)

# read in data and transform idx from datetime to int
item_lst = ["open", "high", "low", "close", "volume"]
dt_data = dict()
for item in item_lst:
    cur_data = pd.read_hdf(f"{config.dir_raw_data_min}/{item}.h5")
    cur_data = kwo.transform_minute_index_datetime_to_int(cur_data)
    dt_data[item] = cur_data
print("raw data loaded")

# post-reinstatement
df_gap = pd.read_hdf(f"{config.dir_raw_data_min}/pricegap.h5").fillna(0)
df_gap = kwo.transform_minute_index_datetime_to_int(df_gap)

for item in ["open", "high", "low", "close"]:
    df_adj = df_gap / dt_data[item] + 1
    df_adj = df_adj.reindex(dt_data[item].index)
    df_adj = df_adj.fillna(1)
    df_adj = df_adj.cumprod()
    for col in df_adj.columns:
        df_adj[col] = df_adj[col] / df_adj[col].iloc[-1]
    dt_data[item] = dt_data[item] * df_adj
print("reinstatement finished")

# get amount
df_size = pd.read_pickle(f"{config.dir_data_day}/Size_MainAdj.pkl")
close_dates = pd.Series(dt_data["close"].index // 10000).unique()
df_size = df_size.reindex(index=close_dates).fillna(method="ffill")
df_size.index = df_size.index * 10000 + 900
close_idx = set(dt_data["close"].index)
size_idx = set(df_size.index)
union_idx = close_idx.union(size_idx)
union_idx = sorted(list(union_idx))
df_size = df_size.reindex(union_idx).fillna(method="ffill")
df_size = df_size.reindex(dt_data["close"].index)
df_amount = df_size * dt_data["close"] * dt_data["volume"]
dt_data["amount"] = df_amount
print("amount generated")

# save data
for item in dt_data.keys():
    dt_data[item].to_pickle(f"{config.dir_data_min}/{item}.pkl")
print("minute data generated")
print()
