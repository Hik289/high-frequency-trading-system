#%%
import numpy as np
import pandas as pd
import warnings
from minepy import MINE
import os
from bisect import bisect_left

os.environ["OMP_NUM_THREADS"] = "1"


def calc_sharpe_from_ret(arg_ret):
    if type(arg_ret) == pd.DataFrame:
        res_mean = arg_ret.mean(axis=0)
        res_std = arg_ret.std(axis=0)
        res_std[res_std.abs() < 1e-9] = np.nan
        return res_mean / res_std * np.sqrt(252)
    elif type(arg_ret) == pd.Series:
        res_mean = arg_ret.mean(axis=0)
        res_std = arg_ret.std(axis=0)
        if np.abs(res_std) < 1e-9:
            return np.nan
        return res_mean / res_std * np.sqrt(252)
    else:
        res_mean = np.mean(arg_ret, axis=0)
        res_std = np.std(arg_ret, axis=0)
        if np.abs(res_std) < 1e-9:
            return np.nan
        return res_mean / res_std * np.sqrt(252)


def calc_standardization(df_fct: pd.DataFrame):
    df_res = df_fct.sub(df_fct.mean(axis=1), axis=0)
    df_std = df_res.std(axis=1)
    df_std[df_std < 1e-6] = np.nan
    df_res = df_res.div(df_std, axis=0)
    return df_res


def calc_demean(df_fct: pd.DataFrame):
    df_res = df_fct.sub(df_fct.mean(axis=1), axis=0)
    return df_res


def calc_rolling_corr(
    df_a: pd.DataFrame, df_b: pd.DataFrame, win: int, min_periods: int = 1
):
    df_res = pd.DataFrame(np.nan, index=df_a.index, columns=df_a.columns)
    for col in df_res.columns:
        df_res[col] = df_a[col].rolling(win, min_periods=min_periods).corr(df_b[col])
    return df_res


def calc_pos(df_fct: pd.DataFrame):
    se_as = df_fct.abs().sum(axis=1)
    se_as[se_as < 1e-6] = np.nan
    df_res = df_fct.div(se_as, axis=0)
    return df_res


def calc_tvr(df_fct: pd.DataFrame):
    df_pos = calc_pos(df_fct).dropna(how="all")
    mv_pos = df_pos - df_pos.shift()
    with np.errstate(divide="ignore", invalid="ignore"):
        res = mv_pos.abs().sum(axis=1) / df_pos.shift().abs().sum(axis=1)
    res = res.replace([np.inf, -np.inf], np.nan)
    return res.mean()


def calc_pnl_from_ret_weight(
    arg_ret: pd.Series, arg_weight: pd.DataFrame = None, fee: float = 0
):
    se_ret = arg_ret + 1
    if arg_weight is not None:
        df_pos = calc_pos(arg_weight)
        mv_pos = df_pos - df_pos.shift()
        mv_pos = mv_pos.abs()
        se_cost = fee * mv_pos.sum(axis=1)
        se_ret = se_ret - se_cost
    se_pnl = se_ret.cumprod()
    return se_pnl


def calc_drawdown_from_ret(arg_ret: pd.Series):
    pnl = calc_pnl_from_ret_weight(arg_ret)
    df_max = pnl.cummax()
    df_down = (pnl - df_max) / df_max
    return df_down


def calc_max_drawdown_from_ret(arg_ret: pd.Series):
    df_down = calc_drawdown_from_ret(arg_ret)
    return df_down.min()


def clip_series(arg_se: pd.Series):
    upper = arg_se.quantile(0.997)
    lower = arg_se.quantile(0.003)
    res = arg_se.clip(lower=lower, upper=upper)
    return res


def clip_dataframe(arg_feat: pd.DataFrame):
    df_fct = arg_feat.replace([np.inf, -np.inf], np.nan)
    upper = df_fct.quantile(0.997, axis=0)
    lower = df_fct.quantile(0.003, axis=0)
    res = df_fct.clip(lower=lower, upper=upper, axis=1)
    return res


def clip_2d_arr(arg_arr: np.array):
    arg_feat = pd.DataFrame(arg_arr)
    res = clip_dataframe(arg_feat)
    return np.array(res)


def get_name_from_path(arg_path: str):
    res_name = arg_path.split(".")[0]
    res_name = res_name.split("/")[-1]
    return res_name


def get_names_from_paths(arg_paths):
    res = [get_name_from_path(elem) for elem in arg_paths]
    return res


def calc_corr(
    arg_1,
    arg_2,
    stack_dataframe: bool = False,
    clip_1: bool = True,
    clip_2: bool = True,
):
    data_1 = arg_1.copy()
    data_2 = arg_2.copy()
    data_1 = data_1.replace([np.inf, -np.inf], np.nan)
    data_2 = data_2.replace([np.inf, -np.inf], np.nan)

    if stack_dataframe:
        if type(data_1) == pd.DataFrame:
            data_1 = data_1.stack()
        if type(data_2) == pd.DataFrame:
            data_2 = data_2.stack()

    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)

    if clip_1:
        if type(data_1) == pd.Series:
            data_1 = clip_series(data_1)
        elif type(data_1) == pd.DataFrame:
            data_1 = clip_dataframe(data_1)
    if clip_2:
        if type(data_2) == pd.Series:
            data_2 = clip_series(data_2)
        elif type(data_2) == pd.DataFrame:
            data_2 = clip_dataframe(data_2)

    if type(data_1) == pd.Series and type(data_2) == pd.Series:
        return data_1.corr(data_2)
    elif type(data_1) == pd.DataFrame and type(data_2) == pd.DataFrame:
        return data_1.corrwith(data_2)


def calc_ic(
    arg_1,
    arg_2,
    stack_dataframe: bool = False,
    clip_1: bool = True,
    clip_2: bool = True,
):
    return calc_corr(
        arg_1=arg_1,
        arg_2=arg_2,
        stack_dataframe=stack_dataframe,
        clip_1=clip_1,
        clip_2=clip_2,
    )


def calc_rank_ic(
    arg_1,
    arg_2,
    stack_dataframe: bool = False,
    clip_1: bool = True,
    clip_2: bool = True,
):
    data_1 = arg_1.copy()
    data_2 = arg_2.copy()
    data_1 = data_1.replace([np.inf, -np.inf], np.nan)
    data_2 = data_2.replace([np.inf, -np.inf], np.nan)

    if stack_dataframe:
        if type(data_1) == pd.DataFrame:
            data_1 = data_1.stack()
        if type(data_2) == pd.DataFrame:
            data_2 = data_2.stack()

    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)

    if clip_1:
        if type(data_1) == pd.Series:
            data_1 = clip_series(data_1)
        elif type(data_1) == pd.DataFrame:
            data_1 = clip_dataframe(data_1)
    if clip_2:
        if type(data_2) == pd.Series:
            data_2 = clip_series(data_2)
        elif type(data_2) == pd.DataFrame:
            data_2 = clip_dataframe(data_2)

    if type(data_1) == pd.Series and type(data_2) == pd.Series:
        return data_1.corr(data_2, method="spearman")
    elif type(data_1) == pd.DataFrame and type(data_2) == pd.DataFrame:
        return data_1.corrwith(data_2, method="spearman")


def calc_corr_mean(
    arg_1: pd.DataFrame,
    arg_2: pd.DataFrame,
    drop_few: bool = True,
    clip_1: bool = True,
    clip_2: bool = True,
):
    data_1 = arg_1.copy()
    data_2 = arg_2.copy()
    data_1 = data_1.dropna(how="all")
    data_2 = data_2.dropna(how="all")
    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)
    if drop_few:
        len_data = len(co_indices)
        nanq_1 = (data_1 != data_1).sum(axis=0)
        nanq_2 = (data_2 != data_2).sum(axis=0)
        for col in data_1.columns:
            if nanq_1[col] > len_data / 2:
                data_1[col] = np.nan
        for col in data_2.columns:
            if nanq_2[col] > len_data / 2:
                data_2[col] = np.nan

    return calc_corr(data_1, data_2, clip_1, clip_2).mean()


def calc_ic_mean(
    arg_1: pd.DataFrame,
    arg_2: pd.DataFrame,
    drop_few: bool = True,
    clip_1: bool = True,
    clip_2: bool = True,
):
    return calc_corr_mean(
        arg_1=arg_1, arg_2=arg_2, drop_few=drop_few, clip_1=clip_1, clip_2=clip_2
    )


def calc_cross_section_ic_info(
    data_1: pd.DataFrame, data_2: pd.DataFrame, clip_1: bool = True, clip_2: bool = True
):
    idx_1 = set(data_1.index)
    idx_2 = set(data_2.index)
    co_idx = idx_1.intersection(idx_2)
    co_idx = sorted(list(co_idx))
    data_1 = data_1.reindex(co_idx)
    data_2 = data_2.reindex(co_idx)
    data_1 = clip_dataframe(data_1)
    data_2 = clip_dataframe(data_2)
    se_ic = data_1.corrwith(data_2, axis=1)
    ic_mean = se_ic.mean()
    ic_std = se_ic.std()
    res_dt = dict()
    res_dt["mean"] = ic_mean
    res_dt["std"] = ic_std
    return res_dt


def calc_ic_bin_info(
    arg_1, arg_2, bins: int = 6, clip_1: bool = True, clip_2: bool = True,
):
    data_1 = arg_1.copy()
    data_2 = arg_2.copy()
    data_1 = data_1.replace([np.inf, -np.inf], np.nan)
    data_2 = data_2.replace([np.inf, -np.inf], np.nan)
    data_1 = data_1.dropna(how="all")
    data_2 = data_2.dropna(how="all")

    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)

    if clip_1:
        if type(data_1) == pd.Series:
            data_1 = clip_series(data_1)
        elif type(data_1) == pd.DataFrame:
            data_1 = clip_dataframe(data_1)
    if clip_2:
        if type(data_2) == pd.Series:
            data_2 = clip_series(data_2)
        elif type(data_2) == pd.DataFrame:
            data_2 = clip_dataframe(data_2)

    bin_width = len(data_1) // bins

    data_lst_1 = [
        data_1.iloc[idx * bin_width : (idx + 1) * bin_width] for idx in range(bins)
    ]

    data_lst_2 = [
        data_2.iloc[idx * bin_width : (idx + 1) * bin_width] for idx in range(bins)
    ]

    if type(data_1) == pd.DataFrame:
        data_lst_1 = [elem.stack() for elem in data_lst_1]
        data_lst_2 = [elem.stack() for elem in data_lst_2]

    ic_lst = [calc_ic(i, j) for i, j in zip(data_lst_1, data_lst_2)]
    ic_mean = np.nanmean(ic_lst)
    ic_std = np.nanstd(ic_lst)

    res_dt = dict()
    res_dt["mean"] = ic_mean
    res_dt["std"] = ic_std
    return res_dt


def calc_ir(
    arg_1,
    arg_2,
    bins: int = 6,
    clip_1: bool = True,
    clip_2: bool = True,
    stack_dataframe: bool = False,
):
    data_1 = arg_1.copy()
    data_2 = arg_2.copy()
    data_1 = data_1.replace([np.inf, -np.inf], np.nan)
    data_2 = data_2.replace([np.inf, -np.inf], np.nan)
    data_1 = data_1.dropna(how="all")
    data_2 = data_2.dropna(how="all")

    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)

    if type(data_1) == pd.Series:
        data_1 = clip_series(data_1)
        data_2 = clip_series(data_2)

    if type(data_2) == pd.DataFrame:
        data_1 = clip_dataframe(data_1)
        data_2 = clip_dataframe(data_2)

    if clip_1:
        if type(data_1) == pd.Series:
            data_1 = clip_series(data_1)
        elif type(data_1) == pd.DataFrame:
            data_1 = clip_dataframe(data_1)
    if clip_2:
        if type(data_2) == pd.Series:
            data_2 = clip_series(data_2)
        elif type(data_2) == pd.DataFrame:
            data_2 = clip_dataframe(data_2)

    bin_width = len(data_1) // bins

    data_lst_1 = [
        data_1.iloc[idx * bin_width : (idx + 1) * bin_width] for idx in range(bins)
    ]

    data_lst_2 = [
        data_2.iloc[idx * bin_width : (idx + 1) * bin_width] for idx in range(bins)
    ]

    if stack_dataframe:
        data_lst_1 = [elem.stack() for elem in data_lst_1]
        data_lst_2 = [elem.stack() for elem in data_lst_2]

    if type(data_1) == pd.Series or stack_dataframe:
        ic_lst = [calc_ic(i, j) for i, j in zip(data_lst_1, data_lst_2)]
        ic_mean = np.nanmean(ic_lst)
        ic_std = np.nanstd(ic_lst)
        if ic_std < 1e-9:
            ic_std = np.nan
        return ic_mean / ic_std

    if type(data_1) == pd.DataFrame:
        ic_lst = [calc_ic(i, j) for i, j in zip(data_lst_1, data_lst_2)]
        df_ic = pd.concat(ic_lst, axis=1)
        ic_mean = df_ic.mean(axis=1)
        ic_std = df_ic.std(axis=1)
        ic_std[ic_std < 1e-9] = np.nan
        return ic_mean / ic_std


def calc_ir_mean(
    arg_1: pd.DataFrame,
    arg_2: pd.DataFrame,
    bins: int = 6,
    drop_few: bool = True,
    clip_1: bool = True,
    clip_2: bool = True,
):
    data_1 = arg_1.copy()
    data_2 = arg_2.copy()
    data_1 = data_1.dropna(how="all")
    data_2 = data_2.dropna(how="all")
    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)
    if drop_few:
        len_data = len(co_indices)
        nanq_1 = (data_1 != data_1).sum(axis=0)
        nanq_2 = (data_2 != data_2).sum(axis=0)
        for col in data_1.columns:
            if nanq_1[col] > len_data / 2:
                data_1[col] = np.nan
        for col in data_2.columns:
            if nanq_2[col] > len_data / 2:
                data_2[col] = np.nan

    return calc_ir(
        arg_1=arg_1, arg_2=arg_2, bins=bins, clip_1=clip_1, clip_2=clip_2
    ).mean()


def calc_sharpe_mean_from_rets(arg_rets: pd.DataFrame, drop_few: bool = True):
    df_ret = arg_rets.copy()
    df_ret = df_ret.replace([np.inf, -np.inf], np.nan)
    df_ret = df_ret.dropna(how="all")
    if drop_few:
        len_data = len(df_ret)
        nanq = (df_ret != df_ret).sum(axis=0)
        for col in df_ret.columns:
            if nanq[col] > len_data / 2:
                df_ret[col] = np.nan

    return calc_sharpe_from_ret(df_ret).mean()


def calc_sharpe_from_feat(
    arg_feat: pd.DataFrame, arg_fwd_ret: pd.DataFrame, clip_feat: bool = False
):
    data_1 = arg_feat.copy()
    data_2 = arg_fwd_ret.copy()
    data_1 = data_1.dropna(how="all")
    data_2 = data_2.dropna(how="all")
    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    if clip_feat:
        data_1 = clip_dataframe(data_1)
    data_2 = data_2.reindex(co_indices)
    len_data = len(co_indices)
    nanq_1 = (data_1 != data_1).sum(axis=0)
    nanq_2 = (data_2 != data_2).sum(axis=0)
    for col in data_1.columns:
        if nanq_1[col] > len_data * 3 / 5:
            data_1[col] = np.nan
    for col in data_2.columns:
        if nanq_2[col] > len_data * 3 / 5:
            data_2[col] = np.nan

    df_pos = calc_pos(data_1)
    se_ret = (df_pos * data_2).sum(axis=1)
    return calc_sharpe_from_ret(se_ret)


def calc_df_lst_mutual_corr(df_lst):
    indices = list()
    columns = list()
    for elem in df_lst:
        indices = indices + list(elem.index)
        columns = columns + list(elem.columns)
    indices = sorted(list(set(indices)))
    columns = sorted(list(set(columns)))

    in_dfs = [elem.reindex(index=indices, columns=columns) for elem in df_lst]
    in_arrs = [np.array(elem).flatten() for elem in in_dfs]
    in_arrs = np.array(in_arrs).transpose()
    df_all = pd.DataFrame(in_arrs)
    df_corr = df_all.corr()
    arr_corr = np.array(df_corr)
    return arr_corr


def calc_mic_array(arg_1: np.array, arg_2: np.array):
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(arg_1, arg_2)
    return mine.mic()


def calc_mic_series(
    arg_1: pd.Series, arg_2: pd.Series, clip_1: bool = True, clip_2: bool = True
):
    data_1 = arg_1.copy()
    data_2 = arg_2.copy()
    data_1 = data_1.dropna()
    data_2 = data_2.dropna()
    if len(data_1) < 5 or len(data_2) < 5:
        return np.nan
    if clip_1:
        data_1 = clip_series(data_1)
    if clip_2:
        data_2 = clip_series(data_2)
    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)

    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    return calc_mic_array(data_1, data_2)


def calc_mic_dataframe(
    arg_1: pd.DataFrame,
    arg_2: pd.DataFrame,
    stack_dataframe: bool = False,
    clip_1: bool = True,
    clip_2: bool = True,
):
    data_1 = arg_1.copy()
    data_2 = arg_2.copy()

    if stack_dataframe:
        data_1 = data_1.stack()
        data_2 = data_2.stack()
        return calc_mic_series(data_1, data_2, clip_1=clip_1, clip_2=clip_2)

    data_1 = data_1.dropna(how="all")
    data_2 = data_2.dropna(how="all")
    if clip_1:
        data_1 = clip_dataframe(data_1)
    if clip_2:
        data_2 = clip_dataframe(data_2)
    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)

    res_lst = list()
    for col in data_1.columns:
        if len(data_1[col].dropna()) < 5 or len(data_2[col].dropna()) < 5:
            res_lst.append(np.nan)
        else:
            res_lst.append(
                calc_mic_series(data_1[col], data_2[col], clip_1=False, clip_2=False)
            )
    se_res = pd.Series(res_lst, index=list(data_1.columns))
    return se_res


def calc_mic_mean(
    arg_1: pd.DataFrame,
    arg_2: pd.DataFrame,
    drop_few: bool = True,
    clip_1: bool = True,
    clip_2: bool = True,
):
    data_1 = arg_1.copy()
    data_2 = arg_2.copy()
    data_1 = data_1.dropna(how="all")
    data_2 = data_2.dropna(how="all")
    if clip_1:
        data_1 = clip_dataframe(data_1)
    if clip_2:
        data_2 = clip_dataframe(data_2)
    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)

    res_lst = list()
    for col in data_1.columns:
        if drop_few:
            jud_1 = len(data_1[col].dropna()) < (len(data_1) / 2)
            jud_2 = len(data_2[col].dropna()) < (len(data_2) / 2)
        else:
            jud_1 = False
            jud_2 = False
        if jud_1 or jud_2:
            res_lst.append(np.nan)
        else:
            res_lst.append(
                calc_mic_series(data_1[col], data_2[col], clip_1=False, clip_2=False)
            )
    return np.nanmean(res_lst)


def calc_1d_arr_corr(arr_1: np.array, arr_2: np.array):
    not_nan_1 = arr_1 == arr_1
    not_nan_2 = arr_2 == arr_2
    not_nan = not_nan_1 * not_nan_2
    res_1 = arr_1[not_nan]
    res_2 = arr_2[not_nan]
    return np.corrcoef(res_1, res_2)[0, 1]


def calc_2d_arr_corr(arr_1: np.array, arr_2: np.array):
    if arr_1.shape != arr_2.shape:
        return np.array(arr_1.shape[1], np.nan)
    res_lst = list()
    for idx in range(arr_1.shape[1]):
        cur_res = calc_1d_arr_corr(arr_1[:, idx], arr_2[:, idx])
        res_lst.append(cur_res)
    return np.array(res_lst)


def calc_ret_from_feat(
    arg_feat: pd.DataFrame,
    arg_fwd_ret: pd.DataFrame,
    clip_feat: bool = True,
    fee: float = 0,
):
    data_1 = arg_feat.copy()
    data_2 = arg_fwd_ret.copy()
    data_1 = data_1.dropna(how="all")
    data_2 = data_2.dropna(how="all")
    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    if clip_feat:
        data_1 = clip_dataframe(data_1)
    data_2 = data_2.reindex(co_indices)

    df_pos = calc_pos(data_1)
    se_ret = (df_pos * data_2).sum(axis=1)
    mv_pos = df_pos - df_pos.shift()
    mv_pos = mv_pos.abs()
    se_cost = fee * mv_pos.sum(axis=1)
    se_res = se_ret - se_cost
    return se_res


def transform_daily_index_int_to_str(dataframe: pd.DataFrame):
    df_res = dataframe.copy()
    df_res.index = pd.Series(df_res.index).apply(lambda x: str(x))
    df_res.index = pd.Series(df_res.index).apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}")
    return df_res


def transform_daily_index_int_to_datetime(dataframe: pd.DataFrame):
    df_res = dataframe.copy()
    df_res.index = pd.Series(df_res.index).apply(lambda x: str(x))
    df_res.index = pd.Series(df_res.index).apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}")
    df_res.index = pd.to_datetime(df_res.index)
    return df_res


def transform_daily_index_str_to_int(dataframe: pd.DataFrame):
    df_res = dataframe.copy()

    def get_int_from_str(arg_str: str):
        str_lst = arg_str.split("-")
        return 10000 * int(str_lst[0]) + 100 * int(str_lst[1]) + int(str_lst[2])

    df_res.index = pd.Series(df_res.index).apply(lambda x: get_int_from_str(x))
    return df_res


def transform_minute_index_datetime_to_int(dataframe: pd.DataFrame):
    df_res = dataframe.copy()
    cur_idx = pd.Series(df_res.index)
    cur_idx = pd.to_datetime(cur_idx)
    df_res.index = cur_idx.apply(lambda x: int(x.strftime("%Y%m%d%H%M")))
    return df_res


def transform_minute_index_int_to_datetime(dataframe: pd.DataFrame):
    df_res = dataframe.copy()
    cur_idx = pd.Series(df_res.index)
    cur_idx = cur_idx.apply(lambda x: str(x))
    cur_idx = pd.to_datetime(cur_idx, format="%Y%m%d%H%M")
    df_res.index = cur_idx
    return df_res


def get_partial_dataframe_by_date(
    dataframe: pd.DataFrame, date_beg: int, date_end: int
):
    days = list(dataframe.index)
    split_1 = bisect_left(days, date_beg)
    split_2 = bisect_left(days, date_end)
    df_res = dataframe.iloc[split_1:split_2]
    return df_res


def stack_feats(feats, feat_names=None):
    feats = [elem.replace([np.inf, -np.inf], np.nan) for elem in feats]
    for elem in feats:
        elem.index.name = "time"
        elem.columns.name = "investment"
    feats = [elem.stack() for elem in feats]
    if feat_names is not None:
        for idx in range(len(feats)):
            feats[idx].name = feat_names[idx]
    df_res = pd.concat(feats, axis=1).dropna(how="all")
    df_res = df_res.reset_index()
    df_res = df_res.dropna(how="all")
    return df_res


def stack_feats_target(feats, target: pd.DataFrame, feat_names=None):
    feats = [elem.replace([np.inf, -np.inf], np.nan) for elem in feats]
    target = target.replace([np.inf, -np.inf], np.nan)
    for elem in feats:
        elem.index.name = "time"
        elem.columns.name = "investment"
    target.index.name = "time"
    target.columns.name = "investment"
    se_feats = [elem.stack() for elem in feats]
    se_target = target.stack()
    if feat_names is not None:
        for idx in range(len(se_feats)):
            se_feats[idx].name = feat_names[idx]
    se_target.name = "target"
    se_lst = se_feats
    se_lst.append(se_target)
    df_res = pd.concat(se_lst, axis=1)
    df_res = df_res.reset_index()
    df_res = df_res.dropna(how="all")
    return df_res
