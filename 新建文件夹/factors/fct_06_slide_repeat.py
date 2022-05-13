import numpy as np
import pandas as pd
import sys
from glob import glob
from bisect import bisect_left
from multiprocessing import Pool
import os
import itertools
import warnings
import joblib
from functools import partial
from time import time

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append("../")
from utils import config
from utils import kiwi_operators as kwo
from utils import data_environment as env
from . import operators as ops
from . import search_pedestal as pedestal
from utils.feature_selection import packed_selectors as ps

from importlib import reload

# reload(kwo)
# reload(env)
# reload(config)


def calc_mismatch_corr(df_a: pd.DataFrame, df_b: pd.DataFrame):
    a_dates = pd.Series(df_a.index // 10000).unique()
    b_dates = pd.Series(df_b.index // 10000).unique()
    a_dates = np.sort(a_dates)
    b_dates = np.sort(b_dates)
    old_idx = list(df_b.index)
    for idx in range(len(old_idx)):
        old_date = old_idx[idx] // 10000
        new_date = a_dates[list(b_dates).index(old_date)]
        old_idx[idx] = new_date * 10000 + old_idx[idx] % 10000
    df_b.index = old_idx
    se_res = df_a.corrwith(df_b)
    se_res.name = b_dates[-1]
    return se_res


# ===================================================================
# define factor


def calc_slide_repeat(
    df_item: pd.DataFrame,
    df_fwd: pd.DataFrame,
    win_compare: int = 252,
    win_day: int = 4,
    win_smooth: int = 1,
    percentile: int = 10,
    m1: int = 0,
    m2: int = 0,
):
    num_mean_day = int(percentile * win_compare / 100)
    df_item = df_item.copy()
    if m1 == 1:
        df_item = df_item.pct_change(1)
    df_item["date"] = df_item.index // 10000
    groups = df_item.groupby("date")
    full_dates = sorted(df_item["date"].unique())
    fwd_mean = df_fwd.rolling(win_day).mean()

    def calc_one_date(arg_date: int):
        date_loc = full_dates.index(arg_date)
        if date_loc - config.fwd_day - 1 - win_compare - win_day < 0:
            sub_res = pd.Series(np.nan, index=df_item.columns)
            sub_res.name = arg_date
            return sub_res
        a_idx = full_dates[date_loc - win_day + 1 : date_loc + 1]
        b_lst = [date_loc - config.fwd_day - 1 - elem for elem in range(win_compare)]
        b_idx_lst = list()
        for elem in b_lst:
            b_idx_lst.append(full_dates[elem - win_day + 1 : elem + 1])
        a_feat = [groups.get_group(elem) for elem in a_idx]
        a_feat = pd.concat(a_feat, axis=0).sort_index(axis=0)
        del a_feat["date"]
        b_feat_lst = list()
        for b_idx in b_idx_lst:
            cur_feat = [groups.get_group(elem) for elem in b_idx]
            cur_feat = pd.concat(cur_feat, axis=0).sort_index(axis=0)
            del cur_feat["date"]
            b_feat_lst.append(cur_feat)
        corr_lst = list()
        for elem in b_feat_lst:
            corr_lst.append(calc_mismatch_corr(a_feat, elem))
        df_corr = pd.concat(corr_lst, axis=1).transpose().sort_index(axis=0)

        sub_res_lst = list()
        for col in df_corr.columns:
            se_corr = df_corr[col]
            if m2 == 0:
                se_corr = se_corr.sort_values(ascending=True)
            else:
                se_corr = se_corr.sort_values(ascending=False)
            good_dates = list(se_corr.index)[:num_mean_day]
            good_fwd = list()
            for elem in good_dates:
                try:
                    good_fwd.append(fwd_mean.loc[elem, col])
                except:
                    good_fwd.append(np.nan)
            sub_res_lst.append(np.nanmean(good_fwd))
        sub_res = pd.Series(sub_res_lst, index=df_corr.columns)
        sub_res.name = arg_date
        return sub_res

    res_lst = list()
    for cur_date in full_dates:
        cur_res = calc_one_date(cur_date)
        res_lst.append(cur_res)

    df_res = pd.concat(res_lst, axis=1).transpose().sort_index()
    if win_smooth > 1:
        df_res = df_res.rolling(win_smooth).mean()
    return df_res


def calc_feat(
    dt_data: dict,
    item: str,
    win_compare: int = 252,
    win_day: int = 4,
    win_smooth: int = 1,
    percentile: int = 10,
    m1: int = 0,
    m2: int = 0,
    m3: int = 0,
    deform: int = 0,
    sign: int = 1,
):
    df_item = dt_data["min"][item]
    df_fwd = (
        dt_data["day"]["open"].pct_change(config.fwd_day).shift(-(config.fwd_day + 1))
    )
    df_res = calc_slide_repeat(
        df_item=df_item,
        df_fwd=df_fwd,
        win_compare=win_compare,
        win_day=win_day,
        win_smooth=win_smooth,
        percentile=percentile,
        m1=m1,
        m2=m2,
    )
    df_res = df_res.replace([np.inf, -np.inf], np.nan)
    df_res = df_res * env.status_filter
    df_res = ops.calc_deform(df_res, deform)
    if m3 == 1:
        df_res = ops.calc_demean(df_res)
    df_res = kwo.calc_pos(df_res)
    return sign * df_res


def calc_feat_param(param):
    return calc_feat(
        dt_data=env.data,
        item=param[0],
        win_compare=param[1],
        win_day=param[2],
        win_smooth=param[3],
        percentile=param[4],
        m1=param[5],
        m2=param[6],
        m3=param[7],
        deform=param[8],
    )


def get_adjoint_full_params(full_param):
    param = full_param[1:5]
    param_lst = pedestal.get_adjoint_params(param)
    res_lst = [full_param[0:1] + elem + full_param[5:] for elem in param_lst]
    return res_lst


feat_prefix = "06_slide_repeat"

item_lst = [
    "amount",
    "volume",
    "open",
    "high",
    "close",
    "low",
]
w1_lst = [126, 252]
w2_lst = [6, 10]
w3_lst = [1, 16]
p_lst = [10, 30]
m1_lst = [0, 1]
m2_lst = [0, 1]
m3_lst = [0, 1]
deform_lst = [0, 1, 2, 3, 4, 5, 6]

params = list(
    itertools.product(
        item_lst, w1_lst, w2_lst, w3_lst, p_lst, m1_lst, m2_lst, m3_lst, deform_lst
    )
)
params = [list(elem) for elem in params]


class BaseFeat(object):
    def __init__(self, train_date_beg: int, train_date_end: int, n_jobs=None):
        self.train_date_beg = train_date_beg
        self.train_date_end = train_date_end
        self.params = params
        if n_jobs == None:
            self.n_jobs = os.cpu_count() * 2 // 3
        else:
            self.n_jobs = n_jobs

    def get_train_dataframe(self, dataframe: pd.DataFrame):
        return kwo.get_partial_dataframe_by_date(
            dataframe, date_beg=self.train_date_beg, date_end=self.train_date_end
        )

    def proc_param(self, param):
        feat = calc_feat_param(param)
        train_feat = self.get_train_dataframe(feat)
        train_fwd_ret = self.get_train_dataframe(env.fwd_ret)
        train_info = ps.get_basic_info(
            train_feat, train_fwd_ret, fwd_day=config.fwd_day
        )

        if train_info["sharpe"] > 0:
            feat_sign = 1
        else:
            feat_sign = -1
            feat = -feat
            train_feat = self.get_train_dataframe(feat)
            train_info = ps.get_basic_info(
                train_feat, train_fwd_ret, fwd_day=config.fwd_day
            )

        if not pedestal.judge_info(train_info):
            return None

        adjoint_param_lst = get_adjoint_full_params(param)
        adjoint_feat_lst = [
            feat_sign * calc_feat_param(elem) for elem in adjoint_param_lst
        ]
        adjoint_info_lst = [
            ps.get_basic_info(elem, train_fwd_ret, fwd_day=config.fwd_day)
            for elem in adjoint_feat_lst
        ]
        adjoint_judge = [
            pedestal.compare_info(train_info, elem) for elem in adjoint_info_lst
        ]
        if not np.prod(adjoint_judge):
            return None

        # output feat in data form
        fct_dir = f"{config.dir_feat}/data/{feat_prefix}"
        fct_name = "_".join([str(elem) for elem in param])
        feat.to_pickle(f"{fct_dir}__{fct_name}.pkl")

        # output feat in func form
        fct_dir = f"{config.dir_feat}/func/{feat_prefix}"
        fct_name = "_".join([str(elem) for elem in param])

        out_feat = partial(
            calc_feat,
            item=param[0],
            win_compare=param[1],
            win_day=param[2],
            win_smooth=param[3],
            percentile=param[4],
            m1=param[5],
            m2=param[6],
            m3=param[7],
            deform=param[8],
            sign=feat_sign,
        )
        joblib.dump(out_feat, f"{fct_dir}__{fct_name}.pkl")

        return feat

    def try_proc_param(self, param):
        try:
            self.proc_param(param)
        except:
            pass

    def search_params(self):
        pool = Pool(self.n_jobs)
        pool.map(self.try_proc_param, self.params)
        pool.close()
        pool.join()

