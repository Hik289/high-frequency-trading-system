import numpy as np
import pandas as pd
import bottleneck as bk
import sys


def calc_diff(df_a: pd.DataFrame, df_b: pd.DataFrame):
    df_sum = df_a + df_b
    df_sum = df_sum.abs()
    df_sum[df_sum < 1e-6] = np.nan
    df_res = (df_a - df_b) / df_sum
    return df_res


def calc_standardization(df_fct: pd.DataFrame):
    df_res = df_fct.sub(df_fct.mean(axis=1), axis=0)
    df_std = df_res.std(axis=1)
    df_std[df_std < 1e-6] = np.nan
    df_res = df_res.div(df_std, axis=0)
    return df_res


def calc_demean(df_fct: pd.DataFrame):
    df_res = df_fct.sub(df_fct.mean(axis=1), axis=0)
    return df_res


def calc_cs_rank(dataframe: pd.DataFrame):
    df_res = dataframe.rank(axis=1) - 1
    df_cnt = df_res.copy()
    df_cnt[~np.isnan(df_cnt)] = 1
    se_cnt = df_cnt.sum(axis=1) - 1
    se_cnt[se_cnt < 1e-4] = np.nan
    df_res = df_res.div(se_cnt, axis=0)
    df_res = df_res - 0.5
    df_res = 2 * df_res
    return df_res


def calc_ts_rank(dataframe: pd.DataFrame, win: int = 10):
    arr_data = np.array(dataframe)
    arr_res = bk.move_rank(arr_data, window=win, min_count=3, axis=0)
    df_res = pd.DataFrame(arr_res, index=dataframe.index, columns=dataframe.columns)
    return df_res


def calc_ts_max(data: pd.DataFrame, win: int = 3):
    arr_data = np.array(data)
    arr_res = bk.move_max(arr_data, win, axis=0)
    df_res = pd.DataFrame(arr_res, index=data.index, columns=data.columns)
    return df_res


def calc_ts_min(data: pd.DataFrame, win: int = 3):
    arr_data = np.array(data)
    arr_res = bk.move_min(arr_data, win, axis=0)
    df_res = pd.DataFrame(arr_res, index=data.index, columns=data.columns)
    return df_res


def calc_rolling_corr(
    df_a: pd.DataFrame, df_b: pd.DataFrame, win: int, min_periods: int = 1
):
    df_res = pd.DataFrame(np.nan, index=df_a.index, columns=df_a.columns)
    for col in df_res.columns:
        df_res[col] = df_a[col].rolling(win, min_periods=min_periods).corr(df_b[col])
    return df_res


def calc_2d_arr_corr(data_1: np.array, data_2: np.array):
    arr_1 = data_1.copy()
    arr_2 = data_2.copy()
    arr_1[np.isnan(arr_2)] = np.nan
    arr_2[np.isnan(arr_1)] = np.nan
    mean_xx = np.nanmean(arr_1 * arr_1, axis=0)
    mean_xy = np.nanmean(arr_1 * arr_2, axis=0)
    mean_yy = np.nanmean(arr_2 * arr_2, axis=0)
    mean_x = np.nanmean(arr_1, axis=0)
    mean_y = np.nanmean(arr_2, axis=0)
    res_up = mean_xy - mean_x * mean_y
    res_down_x = np.sqrt(mean_xx - mean_x * mean_x)
    res_down_y = np.sqrt(mean_yy - mean_y * mean_y)
    res = res_up / res_down_x / res_down_y
    return res


def deform_reverse_half(dataframe: pd.DataFrame):
    se_mean = dataframe.mean(axis=1)
    df_sub = calc_demean(dataframe)
    df_sub = df_sub.abs()
    df_sub = calc_demean(df_sub)
    df_sub = df_sub.sub(-se_mean, axis=0)
    return df_sub


def deform_reverse_quantile(dataframe: pd.DataFrame, quantile: float = 0.3):
    q1 = quantile
    q2 = 1 - quantile
    se_1 = dataframe.quantile(q1, axis=1)
    se_2 = dataframe.quantile(q2, axis=1)
    df_1 = dataframe.sub(se_1, axis=0)
    df_2 = dataframe.sub(se_2, axis=0)
    df_res = dataframe.copy()
    df_res[df_1 < 0] = -df_res[df_1 < 0]
    df_res[df_2 > 0] = -df_res[df_2 > 0]
    return df_res


def deform_div_std(dataframe: pd.DataFrame):
    len_data = len(dataframe)
    df_std = dataframe.rolling(len_data, min_periods=1).std()
    df_res = dataframe / df_std
    return df_res


def deform_conserve_half(dataframe: pd.DataFrame, mode: int = 0):
    se_mean = dataframe.mean(axis=1)
    df_res = calc_demean(dataframe)
    if mode == 0:
        df_res[df_res < 0] = 0
    else:
        df_res[df_res > 0] = 0
    df_res = calc_demean(df_res)
    df_res = df_res.sub(-se_mean, axis=0)
    return df_res


def calc_deform(dataframe: pd.DataFrame, mode: int = 0):
    if mode == 0:
        return dataframe
    elif mode == 1:
        return deform_reverse_half(dataframe)
    elif mode == 2:
        return deform_reverse_quantile(dataframe, 0.15)
    elif mode == 3:
        return deform_reverse_quantile(dataframe, 0.05)
    elif mode == 4:
        return deform_div_std(dataframe)
    elif mode == 5:
        return deform_conserve_half(dataframe, mode=0)
    elif mode == 6:
        return deform_conserve_half(dataframe, mode=1)
    else:
        return dataframe

