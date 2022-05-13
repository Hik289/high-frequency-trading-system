#%%
import pandas as pd
import numpy as np
import sys
import toad
import os

sys.path.append("..")
from .. import kiwi_operators as kwo
from importlib import reload

reload(kwo)

os.environ["OMP_NUM_THREADS"] = "1"

# ============================================================================
# Functions:
# 1) calc_win_rate
#   Calculates the winning rate of the factor on varieties.
#   This function avoids one of the contracts from overstaffing.
#   One may choose 'ic' or 'ret' mode.
# 2) calc_feat_psi
#   Calculate population stability index (psi).
#   Input data is a dataframe-form feature, which will be flatten into a 1d-array.
# 3) calc_joint_psi
#   Consider joint time-stability of feature and forward return.
#   Combine mode: substraction or division.
# ============================================================================


def calc_win_rate(df_feat: pd.DataFrame, df_fwd_ret: pd.DataFrame, mode: str = "ic"):
    """
    Input:
        df_feat: feature dataframe
        df_fwd_ret: target forward return
        mode: 'ic' or 'ret
    Return:
        winning ratio among varieties
    """
    data_1 = df_feat.copy().dropna(how="all")
    data_2 = df_fwd_ret.copy().dropna(how="all")
    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)
    if mode == "ret":
        df_ret = data_1 * data_2
        df_ret = df_ret.dropna(how="all")
        df_ret = df_ret.replace([np.inf, -np.inf], np.nan)

        se_ret = df_ret.mean(axis=1)
        ret = se_ret.mean()
        if ret < 0:
            se_ret = -se_ret
        se_ret[se_ret > 0] = 1
        se_ret[se_ret < 0] = 0
        return se_ret.mean()

    if mode == "ic":
        data_1 = data_1.replace([np.inf, -np.inf], np.nan)
        data_2 = data_2.replace([np.inf, -np.inf], np.nan)
        data_1 = data_1.dropna(how="all")
        data_2 = data_2.dropna(how="all")

        se_ic = data_1.corrwith(data_2, axis=1)
        if se_ic.mean() < 0:
            se_ic = -se_ic
        se_ic[se_ic > 0] = 1
        se_ic[se_ic < 0] = 0
        return se_ic.mean()


def calc_cs_win_rate(df_feat: pd.DataFrame, df_fwd_ret: pd.DataFrame, mode: str = "ic"):
    """
    Input:
        df_feat: feature dataframe
        df_fwd_ret: target forward return
        mode: 'ic' or 'ret
    Return:
        winning ratio among varieties
    """
    data_1 = df_feat.copy().dropna(how="all")
    data_2 = df_fwd_ret.copy().dropna(how="all")
    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)
    if mode == "ret":
        df_ret = data_1 * data_2
        df_ret = df_ret.dropna(how="all")
        df_ret = df_ret.replace([np.inf, -np.inf], np.nan)
        nanq = (df_ret != df_ret).sum(axis=0)
        for col in df_ret.columns:
            if nanq[col] > len(df_ret) / 2:
                df_ret[col] = np.nan

        se_ret = df_ret.mean(axis=0)
        ret = se_ret.mean()
        if ret < 0:
            se_ret = -se_ret
        se_ret[se_ret > 0] = 1
        se_ret[se_ret < 0] = 0
        return se_ret.mean()

    if mode == "ic":
        data_1 = data_1.replace([np.inf, -np.inf], np.nan)
        data_2 = data_2.replace([np.inf, -np.inf], np.nan)
        data_1 = data_1.dropna(how="all")
        data_2 = data_2.dropna(how="all")

        nanq = (data_1 != data_1).sum(axis=0)
        for col in data_1.columns:
            if nanq[col] > len(data_1) / 2:
                data_1[col] = np.nan
        for col in data_2.columns:
            if nanq[col] > len(data_2) / 2:
                data_2[col] = np.nan

        se_ic = kwo.calc_ic(data_1, data_2)
        if se_ic.mean() < 0:
            se_ic = -se_ic
        se_ic[se_ic > 0] = 1
        se_ic[se_ic < 0] = 0
        return se_ic.mean()


def calc_feat_psi(is_feat: pd.DataFrame, os_feat: pd.DataFrame):
    data_is = pd.DataFrame(np.array(is_feat).flatten(), columns=["feat"])
    data_os = pd.DataFrame(np.array(os_feat).flatten(), columns=["feat"])
    data_is = data_is.replace([np.inf, -np.inf], np.nan).dropna()
    data_os = data_os.replace([np.inf, -np.inf], np.nan).dropna()
    binner = toad.transform.Combiner()
    try:
        binner.fit(data_is, n_bins=10, method="quantile")
    except:
        return np.nan
    data_is = binner.transform(data_is)
    data_os = binner.transform(data_os)
    is_ratio = np.array(pd.value_counts(data_is["feat"]))
    os_ratio = np.array(pd.value_counts(data_os["feat"]))
    is_ratio = is_ratio / np.sum(is_ratio)
    os_ratio = os_ratio / np.sum(os_ratio)
    res_lst = [(ai - ei) * np.log(ai / ei) for ai, ei in zip(is_ratio, os_ratio)]
    return np.sum(res_lst)


def calc_joint_psi(
    is_feat: pd.DataFrame,
    os_feat: pd.DataFrame,
    is_fwd_ret: pd.DataFrame,
    os_fwd_ret: pd.DataFrame,
    mode: str = "div",
):
    if mode == "sub":
        stand_is_feat = kwo.calc_standardization(is_feat)
        stand_is_fwd_ret = kwo.calc_standardization(is_fwd_ret)
        stand_os_feat = kwo.calc_standardization(os_feat)
        stand_os_fwd_ret = kwo.calc_standardization(os_fwd_ret)
        data_is = stand_is_feat - stand_is_fwd_ret
        data_os = stand_os_feat - stand_os_fwd_ret
        data_is = data_is.replace([np.inf, -np.inf], np.nan)
        data_os = data_os.replace([np.inf, -np.inf], np.nan)
    elif mode == "div":
        data_is = is_feat / is_fwd_ret
        data_os = os_feat / os_fwd_ret
    else:
        raise ValueError(f"{mode} is not supported")

    return calc_feat_psi(data_is, data_os)


def get_noise_ic_std(data, num_sample: int = 100):
    if type(data) == pd.Series:
        se_data = data.copy().dropna()
        se_data = kwo.clip_series(se_data)
    elif type(data) == pd.DataFrame:
        se_data = data.stack().dropna()
        se_data = kwo.clip_series(se_data)
    else:
        raise ValueError(f"{type(data)} not supported")

    se_data = np.array(se_data)
    len_data = len(se_data)
    ic_lst = list()
    for idx in range(num_sample):
        se_rand = np.random.randn(len_data)
        ic_tmp = np.corrcoef(se_rand, se_data)[0, 1]
        ic_lst.append(ic_tmp)

    res_std = np.std(ic_lst)
    return res_std


def get_noise_cs_ic_std(data: pd.DataFrame, num_sample: int = 100):
    data = kwo.clip_dataframe(data)

    ic_lst = list()

    for idx in range(num_sample):
        df_rand = np.random.randn(data.shape[0], data.shape[1])
        df_rand = pd.DataFrame(df_rand, index=data.index, columns=data.columns)
        ic_lst.append(data.corrwith(df_rand, axis=1).mean())

    return np.std(ic_lst)


def get_long_short_cnt_ratio(data: pd.DataFrame):
    num_long = data[data > 0].count().sum() + 1e-8
    num_short = data[data < 0].count().sum() + 1e-8
    return num_long / num_short


def get_long_short_pos_ratio(data: pd.DataFrame):
    df_long = data.copy()
    df_long[df_long < 0] = 0
    df_short = data.copy()
    df_short[df_short > 0] = 0
    se_long = df_long.sum(axis=1)
    se_short = df_short.sum(axis=1).abs()
    se_res = se_long / se_short
    return se_res.mean()


def calc_exposure(dataframe: pd.DataFrame):
    df_pos = kwo.calc_pos(dataframe)
    df_pos[df_pos < 0] = 0
    se_res = df_pos.sum(axis=1) * 2 - 1
    return se_res.mean()


def calc_exposure_count(dataframe: pd.DataFrame):
    df_pos = kwo.calc_pos(dataframe)
    num_long = df_pos[df_pos > 0].count().sum() + 1e-8
    num_short = df_pos[df_pos < 0].count().sum() + 1e-8
    return (num_long - num_short) / (num_long + num_short)


def calc_single_exposure(dataframe: pd.DataFrame):
    df_tmp = dataframe.copy()
    df_tmp[df_tmp < 0] = 0
    df_tmp[df_tmp > 0] = 1
    se_res = df_tmp.mean(axis=0)
    se_res = se_res - 0.5
    se_res = se_res.abs()
    return se_res.mean()


def calc_pos_concentration(dataframe: pd.DataFrame):
    dataframe = dataframe.abs()
    se_as = dataframe.mean(axis=1)
    se_as[se_as < 1e-6] = np.nan
    dataframe = dataframe.div(se_as, axis=0)
    return dataframe.std(axis=1).mean()


def calc_ret_concentration(feat: pd.DataFrame, fwd_ret: pd.DataFrame):
    data_1 = feat.dropna(how="all")
    data_2 = fwd_ret.dropna(how="all")
    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)
    df_ret = data_1 * data_2

    se_ret = df_ret.sum(axis=0)
    try:
        res = se_ret.mean() / se_ret.std()
    except:
        res = np.nan
    return res

