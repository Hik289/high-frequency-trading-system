#%%
from tkinter.font import BOLD
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import os
import tqdm
from glob import glob
from multiprocessing import Pool
from bisect import bisect_left
import sys
import scipy.stats as stats

from .. import kiwi_operators as kwo
from .. import split_data as sd
from importlib import reload

reload(kwo)
os.environ["OMP_NUM_THREADS"] = "1"

# =======================================================
# auxiliary funtions
# =======================================================

# given a dataframe, assign label to each row
def get_group_label(data, group_size):
    data = data.dropna()
    result = pd.Series(-1, index=data.index)
    data_len = len(data)
    data_unit = int(data_len / group_size)
    cur_iloc = 0
    for i in range(data_unit):
        result.iloc[cur_iloc : cur_iloc + group_size] = i
        cur_iloc += group_size

    return result, data_unit


# from range(0,data_unit, randomly choice data_unit/2)
# repeat num_boot times
def get_train_groups(data_unit, num_boot):
    n = data_unit
    result = []
    for i in range(num_boot):
        result.append(np.random.choice(np.arange(n), int(n / 2), replace=False))
    return result


def calc_single_pbo(
    all_groups, train_group, data, group_label, test_model, check_num="best"
):

    test_group = list(set(all_groups) - set(train_group))

    train_df = data[group_label.isin(train_group)]
    test_df = data[group_label.isin(test_group)]

    if test_model == "ret":
        train_best_param = (
            train_df.apply(lambda x: np.mean(x) * 252, axis=0).idxmax()
            if check_num == "best"
            else check_num
        )
        test_statistic = test_df.apply(lambda x: np.mean(x) * 252, axis=0)
    elif test_model == "sharpe":
        train_best_param = (
            train_df.apply(
                lambda x: (np.nanmean(x) * 252) / (np.std(x) * np.sqrt(252)), axis=0
            ).idxmax()
            if check_num == "best"
            else check_num
        )
        test_statistic = test_df.apply(
            lambda x: (np.mean(x) * 252) / (np.std(x) * np.sqrt(252)), axis=0
        )
    else:
        raise ValueError(f"{test_model} is not supported")

    train_best_param_oos_return = test_statistic.loc[train_best_param]

    relative_rank = (
        stats.percentileofscore(test_statistic.tolist(), train_best_param_oos_return)
        / 100
    )
    if relative_rank == 1:
        pbo = 1
    else:
        pbo = np.log(relative_rank / (1 - relative_rank))
    return pbo, train_best_param


def get_param_analysis(pbo_list, train_best_params, boot_num):
    data = pd.DataFrame({"pbo": pbo_list, "best_param": train_best_params})
    analysis = (
        data.groupby("best_param")["pbo"]
        .describe()
        .sort_values(by="count", ascending=False)
    )
    analysis_rank = (
        analysis[["25%", "50%", "75%"]].rank(ascending=True, axis=0).sum(1)
        * analysis["std"].rank(ascending=False)
        + analysis[["mean", "min"]].rank(ascending=True, axis=0).sum(1) * 2
    )
    result = analysis_rank.sort_values(ascending=False)[
        analysis["count"] > (boot_num / len(analysis))
    ]
    return list(result.index[:3])


def stationary_bootstrap(data, len_block_mean: int = 12, len_output=None):
    accept = 1 / len_block_mean
    len_data = len(data)

    if len_output is None:
        sample = [-1 for jdx in range(len_data)]
    else:
        sample = [-1 for jdx in range(len_output)]

    data_idx = np.random.randint(0, len_data)
    for res_idx in range(len(sample)):
        if np.random.uniform(0, 1) >= accept:
            data_idx = data_idx + 1
            if data_idx >= len_data:
                data_idx = 0
        else:
            data_idx = np.random.randint(0, len_data)

        sample[res_idx] = data[data_idx]

    return sample


def interval_bootstrap(data, len_output):
    len_start = len(data) - len_output
    start_idx = np.random.randint(0, len_start)
    return data[start_idx : start_idx + len_output]


# =======================================================
# selectors
# =======================================================


def select_parameters(
    data,
    mode: str = "ret",
    boot_num: int = 1000,
    group_size: int = 30,
    check_num: str = "best",
    pic_show: bool = False,
):
    """
    This function select parameters of one factor.
    'data' is a pd.DataFrame containing returns by each parameter
    data demo:
                            0         1         2       3        4
        dt
        2016-06-13      -0.07   -0.013    -0.04     0.01    0.04
        2016-06-14      -0.02   -0.013    -0.04     0.01    0.02
        2016-06-15      -0.03   -0.013    -0.05     0.02    0.01
    """

    data = data.dropna()  # data is a true matrix
    group_label, data_unit = get_group_label(data, group_size)
    all_groups = np.arange(data_unit)
    train_groups = get_train_groups(data_unit, boot_num)  # get train groups
    if len(all_groups) == 0:
        raise ValueError(f"data has not enough trade")

    results = []
    for train_group in train_groups:
        result = calc_single_pbo(
            all_groups, train_group, data, group_label, mode, check_num,
        )
        results.append(result)

    pbo_list = [i[0] for i in results]
    train_best_params = [i[1] for i in results]

    best_params = get_param_analysis(pbo_list, train_best_params, boot_num)

    if pic_show:
        if check_num == "best":
            params_id = pd.Series(data.columns).reset_index(drop=False)
            params_id.set_index(0, inplace=True)

            fig, axes = plt.subplots(1, 3, figsize=(20, 4))
            pd.Series(pbo_list).plot(
                kind="hist", bins=20, ax=axes[0], title="logit distribution"
            )
            axes[0].set_xlabel("logit")
            # train_best_params = []
            idmax_params = params_id.loc[train_best_params, "index"].tolist()
            pd.Series(idmax_params).plot(
                kind="hist", bins=20, ax=axes[1], title="best param distribution"
            )
            axes[1].set_xlabel("param idx")
            plt.scatter(pd.Series(idmax_params), pd.Series(pbo_list))
            axes[2].set_title("best param scatterplot")
            axes[2].set_xlabel("param idx")
            axes[2].set_ylabel("logit")

            fig, axes = plt.subplots(1, 3, figsize=(20, 4))
            for i in range(len(best_params)):
                pd.Series(
                    np.array(pbo_list)[np.array(train_best_params) == best_params[i]]
                ).plot(
                    kind="hist",
                    bins=10,
                    ax=axes[i],
                    title=f"hist of param {best_params[i]}",
                )
                axes[i].set_xlabel("logit")
            #
        else:
            pd.Series(pbo_list).plot(kind="hist", bins=20, title="logit distribution")

    grouped_pbo = [1 if x < 0 else 0 for x in pbo_list]
    result = sum(grouped_pbo) / len(pbo_list)
    # print("overfit prob:", result)
    dt_res = dict()
    dt_res["pbo_list"] = pbo_list
    dt_res["history_best_params"] = train_best_params
    dt_res["overfit_probability"] = result
    dt_res["best_params"] = best_params
    return dt_res


def get_boot_stat(
    data,
    fwd_ret=None,
    num_boot=5000,
    mode: str = "ret",
    boot_method: str = "stationary",
    len_boot=None,
    pic_show: bool = False,
):
    """
    mode = 'ic', 'ret', or 'sharpe'
    data can be pd.Series or pd.DataFrame
    If type(data) == pd.DataFrame, mean over columns is calculated.
    If mode='ic', fwd_ret should be assigned.
    Note that ic or is really time-consuming.
    boot_method can be 'simple' or 'stationary'. 
    """
    if type(data) == pd.Series:
        data = data.dropna()
    elif type(data) == pd.DataFrame:
        data = data.dropna(how="all")
    else:
        raise ValueError("Input data illegal.")

    if mode == "ic":
        if fwd_ret is None:
            raise ValueError("In ic mode, fwd_ret should be assigned.")

    if fwd_ret is not None:
        fwd_ret = fwd_ret.reindex(data.index)
        is_fwd_ret = fwd_ret
        os_fwd_ret = fwd_ret

    is_data = data
    os_data = data

    is_stat_lst = list()
    num_is = len(is_data)
    if len_boot is None:
        len_boot = num_is

    # for _ in tqdm.tqdm(range(num_boot)):
    for _ in range(num_boot):
        if boot_method == "interval":
            idx = list(interval_bootstrap(list(range(num_is)), len_output=len_boot))
        elif boot_method == "stationary":
            idx = list(
                stationary_bootstrap(
                    list(range(num_is)), len_block_mean=20, len_output=len_boot
                )
            )
        else:
            raise ValueError("Input boot_method invalid.")
        cur_is_data = is_data.iloc[idx]
        cur_is_data = cur_is_data.reset_index(drop=True)
        if mode == "ic":
            cur_is_fwd = is_fwd_ret.iloc[idx]
            cur_is_fwd = cur_is_fwd.reset_index(drop=True)

        if mode == "ret":
            if type(data) == pd.Series:
                is_statistic = cur_is_data.mean() * 252
            else:
                is_statistic = np.nanmean(np.array(cur_is_data).flatten()) * 252
        elif mode == "sharpe":
            if type(data) == pd.Series:
                is_statistic = kwo.calc_sharpe_from_ret(cur_is_data)
            else:
                is_statistic = kwo.calc_sharpe_mean_from_rets(cur_is_data)
        elif mode == "ic":
            if type(data) == pd.Series and type(fwd_ret) == pd.Series:
                is_statistic = kwo.calc_ic(cur_is_data, cur_is_fwd)
            elif type(data) == pd.DataFrame and type(fwd_ret) == pd.DataFrame:
                is_statistic = kwo.calc_ic(
                    cur_is_data, cur_is_fwd, stack_dataframe=True
                )
            else:
                raise ValueError("Input data for ic mode invalid.")

        else:
            raise ValueError(f"{mode} is not supported")
        is_stat_lst.append(is_statistic)

    if mode == "ret":
        if type(data) == pd.Series:
            os_stat = os_data.mean() * 252
        else:
            os_stat = np.nanmean(np.array(os_data).flatten()) * 252
    elif mode == "sharpe":
        if type(data) == pd.Series:
            os_stat = kwo.calc_sharpe_from_ret(os_data)
        else:
            os_stat = kwo.calc_sharpe_mean_from_rets(os_data)
    elif mode == "ic":
        if type(data) == pd.Series and type(fwd_ret) == pd.Series:
            os_stat = kwo.calc_ic(os_data, os_fwd_ret)
        elif type(data) == pd.DataFrame and type(fwd_ret) == pd.DataFrame:
            os_stat = kwo.calc_ic(os_data, os_fwd_ret, stack_dataframe=True)
        else:
            raise ValueError("Input data for ic mode invalid.")
    else:
        raise ValueError(f"{mode} is not supported")

    is_stat_lst = np.array(is_stat_lst)
    # if np.nanmean(is_stat_lst) < 0:
    #     is_stat_lst = is_stat_lst * (-1)
    #     os_stat = -os_stat

    if pic_show:
        pd.Series(is_stat_lst).plot(kind="hist", bins=num_boot // 25)
        plt.title(
            "boot stat dist (boot_mean {}   fulltime_stat {})".format(
                round(np.nanmean(is_stat_lst), 4), round(os_stat, 4)
            )
        )
        plt.axvline(os_stat, color="r", label="fulltime")
        plt.xlabel("statistic")

    is_stat_lst = sorted(is_stat_lst)
    os_location = bisect_left(is_stat_lst, os_stat)
    res_quan = os_location / len(is_stat_lst)
    res_boot_mean = np.mean(is_stat_lst)
    res_boot_std = np.std(is_stat_lst)
    res_dt = dict()
    res_dt["stat"] = os_stat
    res_dt["mean_quantile"] = res_quan
    res_dt["boot_mean"] = res_boot_mean
    res_dt["boot_std"] = res_boot_std
    res_dt["boot_skew"] = stats.skew(is_stat_lst)
    res_dt["boot_kurt"] = stats.kurtosis(is_stat_lst)
    bad_stats=np.sort(is_stat_lst)[:num_boot//10]
    res_dt['boot_bad']=np.mean(bad_stats)

    with np.errstate(divide="ignore", invalid="ignore"):
        res_dt["ratio"] = os_stat / res_boot_std
        res_dt["boot_ratio"] = res_boot_mean / res_boot_std

    return res_dt


def get_slide_stat(
    data,
    fwd_ret=None,
    window=None,
    slide_step: int = 1,
    mode: str = "ret",
    pic_show: bool = False,
):
    """
    mode = 'ic', 'ret', or 'sharpe'
    data can be pd.Series or pd.DataFrame
    If type(data) == pd.DataFrame, mean over columns is calculated.
    If mode='ic' , fwd_ret should be assigned.
    At least one of os_idx/os_date should be assigned. If os_idx is assigned, os_date will be screened.
    Note that ic mode is really time-consuming.
    """
    if type(data) == pd.Series:
        data = data.dropna()
    elif type(data) == pd.DataFrame:
        data = data.dropna(how="all")
    else:
        raise ValueError("Input data illegal.")

    if mode == "ic":
        if fwd_ret is None:
            raise ValueError("In ic mode, fwd_ret should be assigned.")

    if fwd_ret is not None:
        fwd_ret = fwd_ret.reindex(data.index)
        is_fwd_ret = fwd_ret
        os_fwd_ret = fwd_ret

    is_data = data
    os_data = data

    is_stat_lst = list()
    num_is = len(is_data)

    if window is None:
        window = num_is // 2

        # for _ in range(num_boot):
    num_slide = num_is - window + 1
    # for slide_idx in tqdm.tqdm(range(num_slide)):
    for slide_idx in range(0, num_slide, slide_step):
        cur_is_data = is_data.iloc[slide_idx : slide_idx + window]
        if mode == "ic":
            cur_is_fwd = is_fwd_ret.iloc[slide_idx : slide_idx + window]

        if mode == "ret":
            if type(data) == pd.Series:
                is_statistic = cur_is_data.mean() * 252
            else:
                is_statistic = np.nanmean(np.array(cur_is_data).flatten()) * 252
        elif mode == "sharpe":
            if type(data) == pd.Series:
                is_statistic = kwo.calc_sharpe_from_ret(cur_is_data)
            else:
                is_statistic = kwo.calc_sharpe_mean_from_rets(cur_is_data)
        elif mode == "ic":
            if type(data) == pd.Series and type(fwd_ret) == pd.Series:
                is_statistic = kwo.calc_ic(cur_is_data, cur_is_fwd)
            elif type(data) == pd.DataFrame and type(fwd_ret) == pd.DataFrame:
                is_statistic = kwo.calc_ic(
                    cur_is_data, cur_is_fwd, stack_dataframe=True
                )
            else:
                raise ValueError("Input data for ic mode invalid.")

        else:
            raise ValueError(f"{mode} is not supported")
        is_stat_lst.append(is_statistic)

    if mode == "ret":
        if type(data) == pd.Series:
            os_stat = os_data.mean() * 252
        else:
            os_stat = np.nanmean(np.array(os_data).flatten()) * 252
    elif mode == "sharpe":
        if type(data) == pd.Series:
            os_stat = kwo.calc_sharpe_from_ret(os_data)
        else:
            os_stat = kwo.calc_sharpe_mean_from_rets(os_data)
    elif mode == "ic":
        if type(data) == pd.Series and type(fwd_ret) == pd.Series:
            os_stat = kwo.calc_ic(os_data, os_fwd_ret)
        elif type(data) == pd.DataFrame and type(fwd_ret) == pd.DataFrame:
            os_stat = kwo.calc_ic(os_data, os_fwd_ret, stack_dataframe=True)
        else:
            raise ValueError("Input data for ic mode invalid.")
    else:
        raise ValueError(f"{mode} is not supported")

    if pic_show:
        pd.Series(is_stat_lst).plot(kind="hist", bins=num_slide // 25)
        plt.title(
            "slide stat dist (slide_mean {}   fulltime_stat {})".format(
                round(np.nanmean(is_stat_lst), 4), round(os_stat, 4)
            )
        )
        plt.axvline(os_stat, color="r", label="fulltime")
        plt.xlabel("statistic")

    is_stat_lst = sorted(is_stat_lst)
    os_location = bisect_left(is_stat_lst, os_stat)
    res_quan = os_location / len(is_stat_lst)
    res_boot_mean = np.mean(is_stat_lst)
    res_boot_std = np.std(is_stat_lst)
    res_dt = dict()
    res_dt["stat"] = os_stat
    res_dt["quantile"] = res_quan
    res_dt["slide_mean"] = res_boot_mean
    res_dt["slide_std"] = res_boot_std
    with np.errstate(divide="ignore", invalid="ignore"):
        res_dt["ratio"] = os_stat / res_boot_std
        res_dt["slide_ratio"] = res_boot_mean / res_boot_std

    return res_dt


def get_os_boot_quantile(
    data,
    fwd_ret=None,
    os_date=None,
    os_idx=None,
    num_boot=5000,
    mode: str = "ret",
    boot_method: str = "stationary",
    pic_show: bool = False,
):
    """
    mode = 'ic', 'ret', or 'sharpe'
    data can be pd.Series or pd.DataFrame
    If type(data) == pd.DataFrame, mean over columns is calculated.
    If mode='ic' , fwd_ret should be assigned.
    At least one of os_idx/os_date should be assigned. If os_idx is assigned, os_date will be screened.
    Note that ic mode is really time-consuming.
    boot_method can be 'simple' or 'stationary'. 
    """
    if type(data) == pd.Series:
        data = data.dropna()
    elif type(data) == pd.DataFrame:
        data = data.dropna(how="all")
    else:
        raise ValueError("Input data illegal.")

    if mode == "ic":
        if fwd_ret is None:
            raise ValueError("In ic mode, fwd_ret should be assigned.")

    if fwd_ret is not None:
        fwd_ret = fwd_ret.reindex(data.index)

    if os_idx is None:
        split_idx = bisect_left(list(data.index), os_date)
        is_data = data.iloc[:split_idx]
        os_data = data.iloc[split_idx:]
        if type(fwd_ret) == pd.Series or type(fwd_ret) == pd.DataFrame:
            is_fwd_ret = fwd_ret.iloc[:split_idx]
            os_fwd_ret = fwd_ret.iloc[split_idx:]

    else:
        is_data = data.iloc[:os_idx]
        os_data = data.iloc[os_idx:]
        if type(fwd_ret) == pd.Series or type(fwd_ret) == pd.DataFrame:
            is_fwd_ret = fwd_ret.iloc[:os_idx]
            os_fwd_ret = fwd_ret.iloc[os_idx:]

    is_stat_lst = list()
    num_is = len(is_data)
    num_os = len(os_data)
    if num_os < 0.8 * num_is:
        num_sample = num_os
    else:
        num_sample = num_is // 2

    # for _ in range(num_boot):
    # for _ in tqdm.tqdm(range(num_boot)):
    for _ in range(num_boot):
        if boot_method == "interval":
            idx = list(interval_bootstrap(list(range(num_is)), len_output=num_sample))
        elif boot_method == "stationary":
            idx = list(
                stationary_bootstrap(
                    list(range(num_is)), len_block_mean=20, len_output=num_sample
                )
            )
        else:
            raise ValueError("Input boot_method invalid.")

        cur_is_data = is_data.iloc[idx]
        cur_is_data = cur_is_data.reset_index(drop=True)
        if mode == "ic":
            cur_is_fwd = is_fwd_ret.iloc[idx]
            cur_is_fwd = cur_is_fwd.reset_index(drop=True)

        if mode == "ret":
            if type(data) == pd.Series:
                is_statistic = cur_is_data.mean() * 252
            else:
                is_statistic = np.nanmean(np.array(cur_is_data).flatten()) * 252
        elif mode == "sharpe":
            if type(data) == pd.Series:
                is_statistic = kwo.calc_sharpe_from_ret(cur_is_data)
            else:
                is_statistic = kwo.calc_sharpe_mean_from_rets(cur_is_data)
        elif mode == "ic":
            if type(data) == pd.Series and type(fwd_ret) == pd.Series:
                is_statistic = kwo.calc_ic(cur_is_data, cur_is_fwd)
            elif type(data) == pd.DataFrame and type(fwd_ret) == pd.DataFrame:
                is_statistic = kwo.calc_ic(
                    cur_is_data, cur_is_fwd, stack_dataframe=True
                )
            else:
                raise ValueError("Input data for ic mode invalid.")

        else:
            raise ValueError(f"{mode} is not supported")
        is_stat_lst.append(is_statistic)

    if mode == "ret":
        if type(data) == pd.Series:
            os_stat = os_data.mean() * 252
        else:
            os_stat = np.nanmean(np.array(os_data).flatten()) * 252
    elif mode == "sharpe":
        if type(data) == pd.Series:
            os_stat = kwo.calc_sharpe_from_ret(os_data)
        else:
            os_stat = kwo.calc_sharpe_mean_from_rets(os_data)
    elif mode == "ic":
        if type(data) == pd.Series and type(fwd_ret) == pd.Series:
            os_stat = kwo.calc_ic(os_data, os_fwd_ret)
        elif type(data) == pd.DataFrame and type(fwd_ret) == pd.DataFrame:
            os_stat = kwo.calc_ic(os_data, os_fwd_ret, stack_dataframe=True)
        else:
            raise ValueError("Input data for ic mode invalid.")
    else:
        raise ValueError(f"{mode} is not supported")

    if pic_show:
        pd.Series(is_stat_lst).plot(kind="hist", bins=num_boot // 25)
        plt.title(
            "is stat dist (is_mean {}   os_stat {})".format(
                round(np.nanmean(is_stat_lst), 4), round(os_stat, 4)
            )
        )
        plt.axvline(os_stat, color="r", label="os")
        plt.xlabel("statistic")

    is_stat_lst = sorted(is_stat_lst)
    os_location = bisect_left(is_stat_lst, os_stat)
    res = os_location / len(is_stat_lst)
    if np.mean(is_stat_lst) < 0:
        res = 1 - res

    return res


def get_os_slide_quantile(
    data,
    fwd_ret=None,
    os_date=None,
    os_idx=None,
    slide_step: int = 1,
    mode: str = "ret",
    pic_show: bool = False,
):
    """
    mode = 'ic', 'ret', or 'sharpe'
    data can be pd.Series or pd.DataFrame
    If type(data) == pd.DataFrame, mean over columns is calculated.
    If mode='ic' , fwd_ret should be assigned.
    At least one of os_idx/os_date should be assigned. If os_idx is assigned, os_date will be screened.
    Note that ic mode is really time-consuming.
    """
    if type(data) == pd.Series:
        data = data.dropna()
    elif type(data) == pd.DataFrame:
        data = data.dropna(how="all")
    else:
        raise ValueError("Input data illegal.")

    if mode == "ic":
        if fwd_ret is None:
            raise ValueError("In ic mode, fwd_ret should be assigned.")

    if fwd_ret is not None:
        fwd_ret = fwd_ret.reindex(data.index)

    if os_idx is None:
        split_idx = bisect_left(list(data.index), os_date)
        is_data = data.iloc[:split_idx]
        os_data = data.iloc[split_idx:]
        if type(fwd_ret) == pd.Series or type(fwd_ret) == pd.DataFrame:
            is_fwd_ret = fwd_ret.iloc[:split_idx]
            os_fwd_ret = fwd_ret.iloc[split_idx:]

    else:
        is_data = data.iloc[:os_idx]
        os_data = data.iloc[os_idx:]
        if type(fwd_ret) == pd.Series or type(fwd_ret) == pd.DataFrame:
            is_fwd_ret = fwd_ret.iloc[:os_idx]
            os_fwd_ret = fwd_ret.iloc[os_idx:]

    is_stat_lst = list()
    num_is = len(is_data)
    num_os = len(os_data)
    if num_os < 0.8 * num_is:
        window = num_os
    else:
        window = num_is // 2

        # for _ in range(num_boot):
    num_slide = num_is - window + 1
    # for slide_idx in tqdm.tqdm(range(num_slide)):
    for slide_idx in range(0, num_slide, slide_step):
        cur_is_data = is_data.iloc[slide_idx : slide_idx + window]
        if mode == "ic":
            cur_is_fwd = is_fwd_ret.iloc[slide_idx : slide_idx + window]

        if mode == "ret":
            if type(data) == pd.Series:
                is_statistic = cur_is_data.mean() * 252
            else:
                is_statistic = np.nanmean(np.array(cur_is_data).flatten()) * 252
        elif mode == "sharpe":
            if type(data) == pd.Series:
                is_statistic = kwo.calc_sharpe_from_ret(cur_is_data)
            else:
                is_statistic = kwo.calc_sharpe_mean_from_rets(cur_is_data)
        elif mode == "ic":
            if type(data) == pd.Series and type(fwd_ret) == pd.Series:
                is_statistic = kwo.calc_ic(cur_is_data, cur_is_fwd)
            elif type(data) == pd.DataFrame and type(fwd_ret) == pd.DataFrame:
                is_statistic = kwo.calc_ic(
                    cur_is_data, cur_is_fwd, stack_dataframe=True
                )
            else:
                raise ValueError("Input data for ic mode invalid.")

        else:
            raise ValueError(f"{mode} is not supported")
        is_stat_lst.append(is_statistic)

    if mode == "ret":
        if type(data) == pd.Series:
            os_stat = os_data.mean() * 252
        else:
            os_stat = np.nanmean(np.array(os_data).flatten()) * 252
    elif mode == "sharpe":
        if type(data) == pd.Series:
            os_stat = kwo.calc_sharpe_from_ret(os_data)
        else:
            os_stat = kwo.calc_sharpe_mean_from_rets(os_data)
    elif mode == "ic":
        if type(data) == pd.Series and type(fwd_ret) == pd.Series:
            os_stat = kwo.calc_ic(os_data, os_fwd_ret)
        elif type(data) == pd.DataFrame and type(fwd_ret) == pd.DataFrame:
            os_stat = kwo.calc_ic(os_data, os_fwd_ret, stack_dataframe=True)
        else:
            raise ValueError("Input data for ic mode invalid.")
    else:
        raise ValueError(f"{mode} is not supported")

    if pic_show:
        pd.Series(is_stat_lst).plot(kind="hist", bins=num_slide // 25)
        plt.title(
            "is stat dist (is_mean {}   os_stat {})".format(
                round(np.nanmean(is_stat_lst), 4), round(os_stat, 4)
            )
        )
        plt.axvline(os_stat, color="r", label="os")
        plt.xlabel("statistic")

    is_stat_lst = sorted(is_stat_lst)
    os_location = bisect_left(is_stat_lst, os_stat)
    res = os_location / len(is_stat_lst)

    return res


def get_valid_boot_quantile(
    data_train,
    data_valid,
    target_train=None,
    target_valid=None,
    num_boot: int = 1000,
    mode: str = "ret",
    pic_show: bool = False,
):
    """
    mode = 'ic', 'ret', or 'sharpe'
    data can be pd.Series or pd.DataFrame
    If type(data) == pd.DataFrame, mean over columns is calculated.
    If mode='ic' , target should be assigned.
    Note that ic mode is really time-consuming.
    Length of valid issues should be shorter than train issues.
    """
    if type(data_train) == pd.Series:
        data_train = data_train.dropna()
    elif type(data_train) == pd.DataFrame:
        data_train = data_train.dropna(how="all")
    else:
        raise ValueError("Input data illegal.")
    if type(data_valid) == pd.Series:
        data_valid = data_valid.dropna()
    elif type(data_valid) == pd.DataFrame:
        data_valid = data_valid.dropna(how="all")
    else:
        raise ValueError("Input data illegal.")

    if mode == "ic":
        if target_train is None or target_valid is None:
            raise ValueError("In ic mode, target should be assigned.")

    if target_train is not None:
        target_train = target_train.reindex(data_train.index)
    if target_valid is not None:
        target_valid = target_valid.reindex(data_valid.index)

    train_len = len(data_train)
    valid_len = len(data_valid)
    if valid_len >= train_len:
        raise ValueError("Valid length is shorter than train length.")

    is_stat_lst = list()
    for boot_idx in range(num_boot):
        cur_idx = list(
            stationary_bootstrap(
                list(range(train_len)), len_block_mean=20, len_output=valid_len
            )
        )
        cur_is_data = data_train.iloc[cur_idx]
        cur_is_data = cur_is_data.reset_index(drop=True)
        if mode == "ic":
            cur_is_fwd = target_train.iloc[cur_idx]
            cur_is_fwd = cur_is_fwd.reset_index(drop=True)

        if mode == "ret":
            if type(data_train) == pd.Series:
                is_stat = cur_is_data.mean() * 252
            else:
                is_stat = np.nanmean(np.array(cur_is_data).flatten()) * 252
        elif mode == "sharpe":
            if type(data_train) == pd.Series:
                is_stat = kwo.calc_sharpe_from_ret(cur_is_data)
            else:
                is_stat = kwo.calc_sharpe_mean_from_rets(cur_is_data)
        elif mode == "ic":
            if type(data_train) == pd.Series and type(target_train) == pd.Series:
                is_stat = kwo.calc_ic(cur_is_data, cur_is_fwd)
            elif (
                type(data_train) == pd.DataFrame and type(target_train) == pd.DataFrame
            ):
                is_stat = kwo.calc_ic(cur_is_data, cur_is_fwd, stack_dataframe=True)
            else:
                raise ValueError("Input data for ic mode invalid.")
        else:
            raise ValueError(f"{mode} is not supported")
        is_stat_lst.append(is_stat)

    if mode == "ret":
        if type(data_valid) == pd.Series:
            os_stat = data_valid.mean() * 252
        else:
            os_stat = np.nanmean(np.array(data_valid).flatten()) * 252
    elif mode == "sharpe":
        if type(data_valid) == pd.Series:
            os_stat = kwo.calc_sharpe_from_ret(data_valid)
        else:
            os_stat = kwo.calc_sharpe_mean_from_rets(data_valid)
    elif mode == "ic":
        if type(data_valid) == pd.Series and type(target_valid) == pd.Series:
            os_stat = kwo.calc_ic(data_valid, target_valid)
        elif type(data_valid) == pd.DataFrame and type(target_valid) == pd.DataFrame:
            os_stat = kwo.calc_ic(data_valid, target_valid, stack_dataframe=True)
        else:
            raise ValueError("Input data for ic mode invalid.")
    else:
        raise ValueError(f"{mode} is not supported")

    if pic_show:
        pd.Series(is_stat_lst).plot(kind="hist", bins=num_boot // 25)
        plt.title(
            "is stat dist (is_mean {}   os_stat {})".format(
                round(np.nanmean(is_stat_lst), 4), round(os_stat, 4)
            )
        )
        plt.axvline(os_stat, color="r", label="os")
        plt.xlabel("statistic")
        plt.show()

    is_stat_lst = sorted(is_stat_lst)
    os_location = bisect_left(is_stat_lst, os_stat)
    res = os_location / len(is_stat_lst)

    return res


def get_cv_os_boot_quantile(
    data,
    target=None,
    os_ratio: float = 0.3,
    num_folds: int = 5,
    num_boot: int = 1000,
    mode: str = "ret",
):
    if type(data) == pd.Series:
        data = data.dropna()
    elif type(data) == pd.DataFrame:
        data = data.dropna(how="all")
    else:
        raise ValueError("Input data illegal.")

    if mode == "ic":
        if target is None:
            raise ValueError("In ic mode, fwd_ret should be assigned.")

    if target is not None:
        target = target.reindex(data.index)

    split_lst = sd.time_cross_valid_split(
        data_x=data, data_y=target, valid_ratio=os_ratio, num_folds=num_folds
    )
    res_lst = list()
    if target is None:
        for split_elem in split_lst:
            x_train = split_elem[0]
            x_valid = split_elem[1]
            cur_res = get_valid_boot_quantile(
                x_train, x_valid, num_boot=num_boot, mode=mode, pic_show=False
            )
            res_lst.append(cur_res)
    else:
        for split_elem in split_lst:
            x_train = split_elem[0]
            x_valid = split_elem[1]
            y_train = split_elem[2]
            y_valid = split_elem[3]
            cur_res = get_valid_boot_quantile(
                x_train,
                x_valid,
                y_train,
                y_valid,
                num_boot=num_boot,
                mode=mode,
                pic_show=False,
            )
            res_lst.append(cur_res)
    return np.mean(res_lst)
