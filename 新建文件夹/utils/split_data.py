#%%
import numpy as np
import pandas as pd
import warnings
import os
from bisect import bisect_left

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")


def split_dataframe_plain(data: pd.DataFrame, by: str, ratio_lst):
    data = data.sort_values(by)
    len_data = len(data)
    ratios = np.array(ratio_lst)
    ratios = ratios / np.sum(ratios)
    target_num_lst = [int(elem * len_data) for elem in ratios]
    target_cum_lst = list()
    cur_num = 0
    for elem in target_num_lst:
        cur_num += elem
        target_cum_lst.append(cur_num)

    group_data = data.groupby(by)
    data_num_lst = [[name, len(group)] for name, group in group_data]
    df_num = pd.DataFrame(data_num_lst)
    df_num.columns = ["time", "num"]
    df_num = df_num.sort_values("time")
    df_num["cum_sum"] = df_num["num"].cumsum()
    src_cum_lst = list(df_num["cum_sum"])
    cut_lst = [bisect_left(src_cum_lst, elem) for elem in target_cum_lst]
    cut_lst = [src_cum_lst[elem] for elem in cut_lst]

    cut_lst.insert(0, 0)
    res_lst = [
        data.iloc[cut_lst[idx] : cut_lst[idx + 1]] for idx in range(len(ratio_lst))
    ]
    return res_lst


def split_dataframe_cv(
    data: pd.DataFrame, by: str, valid_ratio: float = 0.25, num_folds: int = 5
):
    data = data.dropna(how="all")
    total_len = len(data)
    valid_len = int(total_len / (num_folds + 1 / valid_ratio - 1))
    train_len = int(valid_len * (1 / valid_ratio - 1))
    res_lst = list()
    for fold_idx in range(num_folds):
        ratio_lst = [valid_len] * (num_folds + 1)
        ratio_lst[fold_idx] = train_len
        cur_res = split_dataframe_plain(data=data, by=by, ratio_lst=ratio_lst)
        res_lst.append(cur_res[fold_idx : fold_idx + 2])
    return res_lst


def resample_dataframe(data: pd.DataFrame, num_resample: int = 10, size_ratio=0.5):
    full_idx = list(range(len(data)))
    num_size = int(len(data) * size_ratio)
    res_lst = list()
    for resample_idx in range(num_resample):
        cur_idx = np.random.choice(full_idx, num_size)
        cur_res = data.iloc[cur_idx]
        res_lst.append(cur_res)
    return res_lst

