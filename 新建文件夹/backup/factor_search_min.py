#%%
import numpy as np
import pandas as pd
import sys
from glob import glob
from bisect import bisect_left
from multiprocessing import Pool
from minepy import MINE as NMI
import os
import itertools
import warnings
from time import time

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"

import factor_operators_day as fco

sys.path.append("../")
from utils import consts
from utils import kiwi_operators as kwo


import importlib as imp

imp.reload(consts)
imp.reload(kwo)
imp.reload(fco)

train_begin = 20160101
valid_begin = 20180101
test_begin = 20190701
fwd_day = 4
min_train_ic = 0.025
min_valid_ic = 0.025

# ==============================================================================
# prepare day bar data

env_day = dict()

env_day["close"] = pd.read_pickle(f"{consts.dir_data_day}/Close_MainAdj.pkl")
env_day["high"] = pd.read_pickle(f"{consts.dir_data_day}/High_MainAdj.pkl")
env_day["low"] = pd.read_pickle(f"{consts.dir_data_day}/Low_MainAdj.pkl")
env_day["open"] = pd.read_pickle(f"{consts.dir_data_day}/Open_MainAdj.pkl")

env_day["openinterest"] = pd.read_pickle(
    f"{consts.dir_data_day}/OpenInterest_MainAdj.pkl"
)
env_day["settle"] = pd.read_pickle(f"{consts.dir_data_day}/Settle_MainAdj.pkl")
env_day["amount"] = pd.read_pickle(f"{consts.dir_data_day}/Amount_MainAdj.pkl")
env_day["volume"] = pd.read_pickle(f"{consts.dir_data_day}/Volume_MainAdj.pkl")

# extended items
with np.errstate(divide="ignore", invalid="ignore"):
    env_day["ocr"] = env_day["open"] / env_day["close"] - 1
    env_day["hlr"] = env_day["high"] / env_day["low"] - 1

day_status = pd.read_pickle(f"{consts.dir_data_day}/tradeStatus_1e9_MainAdj.pkl")
day_filter = pd.DataFrame(1.0, index=day_status.index, columns=day_status.columns)
day_filter[~day_status] = np.nan

for key in env_day.keys():
    env_day[key] = env_day[key] * day_filter
    env_day[key] = env_day[key].replace([np.inf, -np.inf], np.nan)

# ==============================================================================
# prepare min bar data

env_min = dict()

env_min["close"] = pd.read_pickle(f"{consts.dir_data_min}/CloseAdj.pkl")
env_min["open"] = pd.read_pickle(f"{consts.dir_data_min}/OpenAdj.pkl")
env_min["high"] = pd.read_pickle(f"{consts.dir_data_min}/HighAdj.pkl")
env_min["low"] = pd.read_pickle(f"{consts.dir_data_min}/LowAdj.pkl")

env_min["openyield"] = pd.read_pickle(f"{consts.dir_data_min}/openyield.pkl")
env_min["closeyield"] = pd.read_pickle(f"{consts.dir_data_min}/closeyield.pkl")

env_min["volume"] = pd.read_pickle(f"{consts.dir_data_min}/Vol.pkl")
env_min["vwap"] = pd.read_pickle(f"{consts.dir_data_min}/VWAPAdj.pkl")
env_min["twap"] = pd.read_pickle(f"{consts.dir_data_min}/TWAPAdj.pkl")
env_min["openinterest"] = pd.read_pickle(f"{consts.dir_data_min}/OpenInterest.pkl")
env_min["amount"] = env_min["vwap"] * env_min["volume"]

min_status = pd.read_pickle(f"{consts.dir_data_min}/trade_status.pkl")
min_filter = pd.DataFrame(1.0, index=min_status.index, columns=min_status.columns)
min_filter[~min_status] = np.nan

for key in env_min.keys():
    env_min[key] = env_min[key] * min_filter
    env_min[key] = env_min[key].replace([np.inf, -np.inf], np.nan)


# ==============================================================================
# define day bar forward return and functions

fwd_ret = pd.read_pickle(f"{consts.dir_fwd_rets}/fwd_ret_{str(fwd_day)}.pkl")
fwd_ret = fwd_ret * day_filter


def get_train_dataframe(dataframe: pd.DataFrame):
    days = list(dataframe.index)
    split_1 = bisect_left(days, train_begin)
    split_2 = bisect_left(days, valid_begin)
    df_res = dataframe.iloc[split_1:split_2]
    return df_res


def get_valid_dataframe(dataframe: pd.DataFrame):
    days = list(dataframe.index)
    split_1 = bisect_left(days, valid_begin)
    split_2 = bisect_left(days, test_begin)
    df_res = dataframe.iloc[split_1:split_2]
    return df_res


def get_test_dataframe(dataframe: pd.DataFrame):
    days = list(dataframe.index)
    split_1 = bisect_left(days, test_begin)
    df_res = dataframe.iloc[split_1:]
    return df_res


def filter_frozen(dataframe: pd.DataFrame):
    df_res = dataframe.copy()
    df_res[env_day["amount"] < 0] = np.nan
    df_res[env_day["volume"] < 0] = np.nan
    return df_res


train_fwd_ret = get_train_dataframe(fwd_ret)
valid_fwd_ret = get_valid_dataframe(fwd_ret)
test_fwd_ret = get_test_dataframe(fwd_ret)


def get_train_ic(in_feat: pd.DataFrame):
    res = kwo.calc_ic_mean(in_feat, train_fwd_ret)
    return res


def get_valid_ic(in_feat: pd.DataFrame):
    res = kwo.calc_ic_mean(in_feat, valid_fwd_ret)
    return res


def get_test_ic(in_feat: pd.DataFrame):
    res = kwo.calc_ic_mean(in_feat, test_fwd_ret)
    return res


# def get_train_ic(in_feat: pd.DataFrame):
#     res = kwo.calc_ic(in_feat, train_fwd_ret, stack_dataframe=True)
#     return res


# def get_valid_ic(in_feat: pd.DataFrame):
#     res = kwo.calc_ic(in_feat, valid_fwd_ret, stack_dataframe=True)
#     return res


def get_train_sharpe(in_feat: pd.DataFrame):
    res = kwo.calc_sharpe_from_feat(in_feat, train_fwd_ret)
    return res / np.sqrt(fwd_day)


def get_valid_sharpe(in_feat: pd.DataFrame):
    res = kwo.calc_sharpe_from_feat(in_feat, valid_fwd_ret)
    return res / np.sqrt(fwd_day)


def get_test_sharpe(in_feat: pd.DataFrame):
    res = kwo.calc_sharpe_from_feat(in_feat, test_fwd_ret)
    return res / np.sqrt(fwd_day)


#%%
# ================================================================================
# 06 slide_occur_again
def calc_index_mismatch_corr(df_1: pd.DataFrame, df_2: pd.DataFrame):
    tmp_1 = df_1
    tmp_2 = df_2.copy()
    tmp_2.index = tmp_1.index
    se_res = tmp_1.corrwith(tmp_2)
    return se_res


def calc_slide_win_corr(df_long: pd.DataFrame, df_short: pd.DataFrame, slide: int = 1):
    len_long = len(df_long)
    len_short = len(df_short)
    idx_lst = range(0, len_long - len_short + 1, slide)
    se_lst = list()
    for idx in idx_lst:
        df_1 = df_long.iloc[idx : idx + len_short]
        df_2 = df_short
        se_tmp = calc_index_mismatch_corr(df_1, df_2)
        se_tmp.name = df_1.index[-1] // 10000
        se_lst.append(se_tmp)
    df_res = pd.concat(se_lst, axis=1).transpose()
    return df_res


def get_ranked_indices(arg_df):
    se_lst = list()
    for col in arg_df.columns:
        se_tmp = arg_df[col].rank(ascending=False)
        se_tmp.index.name = col
        se_tmp.name = "rank"
        se_tmp = se_tmp.reset_index()
        se_tmp = se_tmp.sort_values("rank").reset_index()
        se_lst.append(se_tmp[col])
    df_res = pd.concat(se_lst, axis=1)
    return df_res


def get_daily_min(date: int):
    min_lst = list()
    min_lst += list(range(901, 960))
    min_lst += list(range(1000, 1016))
    min_lst += list(range(1031, 1060))
    min_lst += list(range(1100, 1131))
    min_lst += list(range(1331, 1360))
    min_lst += list(range(1400, 1460))
    min_lst += [1500]

    res_lst = [elem + date * 10000 for elem in min_lst]
    return res_lst


def get_night_min(date: int):
    min_lst = list()
    min_lst += list(range(2101, 2160))
    min_lst += list(range(2200, 2260))
    min_lst += list(range(2300, 2360))
    min_lst += list(range(0, 60))
    min_lst += list(range(100, 160))
    min_lst += list(range(200, 231))
    res_lst = [elem + date * 10000 for elem in min_lst]
    return res_lst


def reindex_dataframe_daily_min(dataframe):
    dates = sorted(list(set(dataframe.index // 10000)))
    idx_lst = list()
    for elem in dates:
        idx_lst += get_daily_min(elem)
    df_res = dataframe.reindex(idx_lst).sort_index()
    return df_res


def add_2400(time_lst):
    res_lst = list()
    res_lst.append(time_lst[0])
    for idx in range(1, len(time_lst)):
        if time_lst[idx] % 10000 < 231:
            res_lst.append(
                res_lst[idx - 1] // 10000 * 10000 + 2400 + time_lst[idx] % 10000
            )
        else:
            res_lst.append(time_lst[idx])
    return res_lst


def reindex_dataframe_night_min(dataframe):
    dates = sorted(list(set(dataframe.index // 10000)))
    idx_lst = list()
    for elem in dates:
        idx_lst += get_night_min(elem)
    df_res = dataframe.reindex(idx_lst).sort_index()
    return df_res


def reindex_dataframe_full_min(dataframe):
    df_1 = reindex_dataframe_daily_min(dataframe)
    df_2 = reindex_dataframe_night_min(dataframe)
    df_res = pd.concat([df_1, df_2])
    return df_res.sort_index()


def calc_slide_occur_again(
    item: str,
    win_roll: int = 252,
    win_day: int = 2,
    win_min: int = 225,
    fwd_day: int = 4,
    num_top: int = 4,
    mode: int = 0,
):
    full_min = win_day * win_min
    df_item = reindex_dataframe_daily_min(env_min[item])
    if mode == 1:
        df_item = df_item.pct_change(1)
    df_item.index.name = None
    df_item["date"] = df_item.index // 10000
    date_lst = sorted(list(set(df_item["date"])))
    groups = df_item.groupby("date")
    df_lst = list()
    for date in date_lst:
        df_tmp = groups.get_group(date)
        # df_tmp = df_tmp[df_tmp.index % 10000 >= 900]
        df_lst.append(df_tmp.iloc[:win_min])
    df_item = pd.concat(df_lst).sort_index()
    del df_item["date"]

    day_open = env_day["open"]
    day_fwd = day_open.pct_change(fwd_day).shift(-(fwd_day + 1))
    day_fwd = day_fwd.reindex(date_lst)

    def calc_one_date(date: int):
        df_long = df_item[df_item.index < date * 10000 + 9999]
        df_short = df_long.iloc[-full_min:]
        df_long = df_long.iloc[: -(fwd_day + 1) * win_min]
        if len(df_long) > win_roll * win_min:
            df_long = df_long.iloc[-win_roll * win_min :]

        df_corr = calc_slide_win_corr(df_long, df_short, slide=win_min)
        ranked = get_ranked_indices(df_corr).iloc[:num_top]
        ret_lst = list()
        for col in ranked.columns:
            ret_tmp = day_fwd.loc[list(ranked[col]), col].reset_index()
            ret_lst.append(ret_tmp[col])
        df_ret = pd.concat(ret_lst, axis=1)
        se_res = df_ret.mean(axis=0)
        se_res.name = date
        return se_res

    dates = sorted(list(set(day_fwd.index).intersection(set(date_lst))))[
        max([win_roll, 22, fwd_day + 1]) :
    ]
    se_lst = list()
    for elem in dates:
        # if item == "open" and num_top == 1 and win_day == 2:
        #     print(elem)
        se_lst.append(calc_one_date(elem))
    df_res = pd.concat(se_lst, axis=1).transpose()
    return df_res


items = [
    "open",
    "high",
    "low",
    "close",
    "openinterest",
    "amount",
    "volume",
    "vwap",
    "twap",
]

win_roll_lst = [252]
win_min_lst = [30, 225]

days = [8, 15]
tops = [10, 30]

mode_lst = [0, 1]

params = list()

for i1 in items:
    for i2 in win_roll_lst:
        for i3 in win_min_lst:
            for i4 in days:
                for i5 in tops:
                    for i6 in mode_lst:
                        param = [i1, i2, i3, i4, i5, i6]
                        params.append(param)


def func_feat(param):
    df_feat = calc_slide_occur_again(
        item=param[0],
        win_roll=param[1],
        win_min=param[2],
        win_day=param[3],
        num_top=param[4],
        mode=param[5],
    )
    df_feat.to_pickle(
        f"{consts.dir_search_factors}/06_slide_occur_again/{param[0]}_{param[1]}_{param[2]}_{param[3]}_{param[4]}_{param[5]}.pkl"
    )
    return df_feat


pool = Pool(142)
pool.map(func_feat, params)
pool.close()

#%%
df_old = pd.read_hdf("/mnt/lustre/group/ftluo/AllSample/hf_data/close_yields.h5")
df_old.index.name = "DateTime"
df_old = df_old.replace([np.inf, -np.inf], np.nan)
df_old = df_old.reset_index()
df_old["DateTime"] = pd.to_datetime(df_old["DateTime"]).apply(
    lambda x: int(x.strftime("%Y%m%d%H%M"))
)
df_old = df_old.set_index("DateTime")
df_old = df_old.sort_index(axis=0).sort_index(axis=1)
df_old.to_pickle(f"{consts.dir_data_min}/closeyield.pkl")


df_old = pd.read_hdf("/mnt/lustre/group/ftluo/AllSample/hf_data/open_yields.h5")
df_old.index.name = "DateTime"
df_old = df_old.replace([np.inf, -np.inf], np.nan)
df_old = df_old.reset_index()
df_old["DateTime"] = pd.to_datetime(df_old["DateTime"]).apply(
    lambda x: int(x.strftime("%Y%m%d%H%M"))
)
df_old = df_old.set_index("DateTime")
df_old = df_old.sort_index(axis=0).sort_index(axis=1)
df_old.to_pickle(f"{consts.dir_data_min}/openyield.pkl")
#%%
df_tmp = pd.read_pickle(
    f"{consts.dir_search_factors}/06_slide_occur_again/vwap_5_5.pkl"
)
print(df_tmp)

#%%
paths = glob(f"{consts.dir_search_factors}/06_slide_occur_again/*.pkl")
feats = [pd.read_pickle(elem) for elem in paths]
corrs_train = [get_train_ic(elem) for elem in feats]
corrs_valid = [get_valid_ic(elem) for elem in feats]
corrs_test = [get_test_ic(elem) for elem in feats]
df_corrs = pd.DataFrame([corrs_train, corrs_valid, corrs_test]).transpose()
df_corrs.columns = ["train_ic", "valid_ic", "test_ic"]
print(df_corrs)

#%%
train_sign = np.sign(df_corrs["train_ic"])
for col in df_corrs.columns:
    df_corrs[col] = df_corrs[col] * train_sign

df_corrs.to_csv(f"{consts.dir_workspace}/ic.csv")
#%%
final_ic = df_corrs.copy()
final_ic = final_ic[final_ic["train_ic"] > 0.025]
final_ic = final_ic[final_ic["valid_ic"] > 0.025]
print(final_ic)
#%%
paths = glob(f"{consts.dir_search_factors}/06_slide_occur_again/*.pkl")
feats = [pd.read_pickle(elem) for elem in paths]
sharpe_train = [get_train_sharpe(elem) for elem in feats]
sharpe_valid = [get_valid_sharpe(elem) for elem in feats]
sharpe_test = [get_test_sharpe(elem) for elem in feats]
df_sharpe = pd.DataFrame([sharpe_train, sharpe_valid, sharpe_test]).transpose()
df_sharpe.columns = ["train_sharpe", "valid_sharpe", "test_sharpe"]
print(df_sharpe)
# df_sharpe.to_csv(f"{consts.dir_workspace}/sharpe.csv")
#%%
print(paths[127])
#%%
df_feat = feats[1]
df_pos = kwo.calc_pos(df_feat)
df_pos = df_pos.rolling(4).mean()
df_fwd_1 = pd.read_pickle(f"{consts.dir_fwd_rets}/fwd_ret_1.pkl")
df_fwd_1 = df_fwd_1 * day_filter
# se_ret = (df_pos * df_fwd_1).sum(axis=1)
# print(kwo.calc_ret_from_feat(df_pos,get_train_dataframe(df_fwd_1)))
old_ret = kwo.calc_ret_from_feat(df_feat, train_fwd_ret)
re_idx = list(range(1, len(old_ret), 4))
new_ret = old_ret.iloc[re_idx]
roll_ret = kwo.calc_ret_from_feat(df_pos, get_train_dataframe(df_fwd_1))
print(roll_ret.mean())
print(new_ret.mean())
print(roll_ret.std())
print(new_ret.std())
print(kwo.calc_sharpe_from_ret(roll_ret))
print(kwo.calc_sharpe_from_ret(new_ret) / 2)

# se_pnl = kwo.calc_pnl_from_ret_weight(se_ret, df_pos, 4e-4)
# print(se_pnl.dropna())
#%%
print(get_train_dataframe(df_fwd_1).rolling(4).sum()["A"].shift(-3))
print(train_fwd_ret["A"])

#%%
import shutil

final_idx_lst = list(final_ic.index)
final_paths = [paths[elem] for elem in final_idx_lst]
for elem in final_paths:
    shutil.copy(elem, "/home/kwsun/DATA/20220312_factors/search_factors/06_tmp/")

#%%
paths = glob(f"{consts.dir_search_factors}/06_tmp/*.pkl")
feats = [pd.read_pickle(elem) for elem in paths]
corrs_train = [get_train_ic(elem) for elem in feats]
corrs_valid = [get_valid_ic(elem) for elem in feats]
corrs_test = [get_test_ic(elem) for elem in feats]
df_corrs = pd.DataFrame([corrs_train, corrs_valid, corrs_test]).transpose()
df_corrs.columns = ["train_ic", "valid_ic", "test_ic"]
print(df_corrs)

#%%
def get_new_train_sharpe(in_feat: pd.DataFrame):
    tmp_fwd = train_fwd_ret.copy()
    re_idx = list(range(0, len(tmp_fwd), 4))
    tmp_fwd = tmp_fwd.iloc[re_idx]
    res = kwo.calc_sharpe_from_feat(in_feat, tmp_fwd)
    return res


print(get_new_train_sharpe(feats[1]))
print(get_train_sharpe(feats[1]))

#%%
tmp_fwd = train_fwd_ret.copy()
re_idx = list(range(3, len(tmp_fwd), 4))
tmp_fwd = tmp_fwd.iloc[re_idx]
# print(train_fwd_ret)
# print(tmp_fwd)

feat_tmp = feats[1]
old_ret = (train_fwd_ret * feat_tmp).sum(axis=1)
new_ret = (tmp_fwd * feat_tmp).sum(axis=1)
print(old_ret.dropna().mean())
print(new_ret.dropna().mean())

#%%
df_old = pd.read_hdf("/mnt/lustre/group/ftluo/AllSample/hf_data/close_yields.h5")
print(df_old.columns)

#%%
tmp_1 = pd.read_pickle(f"{consts.dir_data_min}/openyield.pkl").sort_index(axis=1)
tmp_2 = pd.read_pickle(f"{consts.dir_data_min}/OpenAdj.pkl").sort_index(axis=1)
print(tmp_1.index)
print(tmp_2.index)

#%%
tmp_feat = calc_slide_occur_again("openyield")

#%%
print(
    env_min["amount"].corrwith(env_min["openyield"].replace([np.inf, -np.inf], np.nan))
)

#%%
print(env_min["openyield"].columns)
#%%
print(env_min["openyield"].replace([np.inf, -np.inf], np.nan).corrwith(env_min["open"]))

#%%
df_tmp = env_min["openyield"]
df_tmp = df_tmp[df_tmp.index % 10000 >= 900]
df_tmp = df_tmp[df_tmp.index % 10000 <= 1500]
df_tmp["date"] = df_tmp.index // 10000
groups = df_tmp.groupby("date")
dates = sorted(list(set(df_tmp["date"])))
for elem in dates[:1]:
    print(groups.get_group(elem))

#%%
def get_daily_min(date: int):
    min_lst = list()
    min_lst += list(range(901, 960))
    min_lst += list(range(1000, 1016))
    min_lst += list(range(1031, 1060))
    min_lst += list(range(1100, 1131))
    min_lst += list(range(1331, 1360))
    min_lst += list(range(1400, 1460))
    min_lst += [1500]

    res_lst = [elem + date * 10000 for elem in min_lst]
    return res_lst


def reindex_dataframe_daily_min(dataframe):
    dates = sorted(list(set(dataframe.index // 10000)))
    idx_lst = list()
    for elem in dates:
        idx_lst += get_daily_min(elem)
    df_res = dataframe.reindex(idx_lst).sort_index()
    return df_res


print(reindex_dataframe_daily_min(env_min["open"]))

#%%
time_lst = sorted(list(set(env_min["open"].index % 10000)))
print(time_lst)

#%%
def get_night_min(date: int):
    min_lst = list()
    min_lst += list(range(2101, 2160))
    min_lst += list(range(2200, 2260))
    min_lst += list(range(2300, 2360))
    min_lst += list(range(0, 60))
    min_lst += list(range(100, 160))
    min_lst += list(range(200, 231))
    res_lst = [elem + date * 10000 for elem in min_lst]
    return res_lst


def reindex_dataframe_night_min(dataframe):
    dates = sorted(list(set(dataframe.index // 10000)))
    idx_lst = list()
    for elem in dates:
        idx_lst += get_night_min(elem)
    df_res = dataframe.reindex(idx_lst)
    df_res = df_res.set_index().sort_index()
    return df_res


def reindex_dataframe_full_min(dataframe):
    dates = sorted(list(set(dataframe.index // 10000)))
    idx_lst = list()
    for elem in dates:
        idx_lst += get_daily_min(elem)
        idx_lst += get_night_min(elem)
    df_res = dataframe.reindex(idx_lst)
    df_res = df_res.set_index().sort_index()
    return df_res


def add_2400(time_lst):
    res_lst = list()
    res_lst.append(time_lst[0])
    for idx in range(1, len(time_lst)):
        if time_lst[idx] % 10000 < 231:
            res_lst.append(
                time_lst[idx - 1] // 10000 * 10000 + 2400 + time_lst[idx] % 10000
            )
        else:
            res_lst.append(time_lst[idx])
    return res_lst


add_2400([20182359, 20190000])

