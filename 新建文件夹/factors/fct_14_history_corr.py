#%%
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

# import operators as ops
# import search_pedestal as pedestal
from utils.feature_selection import packed_selectors as ps


# ===================================================================
# define factor


def calc_history_corr(
    df_item: pd.DataFrame,
    df_fwd: pd.DataFrame,
    w1: int = 252,
    w2: int = 20,
    m1: int = 0,
    m2: int = 0,
):
    if m1 == 1:
        df_item = df_item.pct_change(1)
    idx_item = set(df_item.index)
    idx_fwd = set(df_fwd.index)
    co_idx = idx_item.intersection(idx_fwd)
    co_idx = sorted(list(co_idx))
    df_item = df_item.reindex(index=co_idx, columns=df_fwd.columns)
    df_fwd = df_fwd.reindex(index=co_idx)
    arr_item = np.array(df_item)
    arr_fwd = np.array(df_fwd)

    col_lst = list(df_item.columns)

    def calc_one_row(row_idx: int):
        if row_idx - config.fwd_day - w1 - w2 + 1 < 0:
            sub_res = [np.nan] * len(col_lst)
            return sub_res
        arr_long = arr_item[
            row_idx - config.fwd_day - w1 - w2 + 1 : row_idx - config.fwd_day
        ]
        cur_fwd = arr_fwd[row_idx - config.fwd_day - w1 : row_idx - config.fwd_day]
        arr_short = arr_item[row_idx - w2 + 1 : row_idx + 1]
        arr_corr = list()
        for cur_idx in range(w1):
            corr_1d = ops.calc_2d_arr_corr(arr_long[cur_idx : cur_idx + w2], arr_short)
            arr_corr.append(corr_1d)
        arr_corr = np.vstack(arr_corr)
        sub_res = np.nanmean(arr_corr * cur_fwd, axis=0)
        return sub_res

    res_lst = list()
    for row_idx in range(len(df_item)):
        res_lst.append(calc_one_row(row_idx))
    arr_res = np.vstack(res_lst)
    df_res = pd.DataFrame(arr_res, index=df_item.index, columns=df_item.columns)
    if m2 == 1:
        df_res = ops.calc_cs_rank(df_res)
    return df_res


def calc_feat(
    dt_data: dict,
    item: str,
    w1: int = 252,
    w2: int = 20,
    m1: int = 0,
    m2: int = 0,
    m3: int = 0,
    deform: int = 0,
    sign: int = 1,
):
    df_item = dt_data["day"][item]
    df_fwd = (
        dt_data["day"]["open"].pct_change(config.fwd_day).shift(-(config.fwd_day + 1))
    )
    df_res = calc_history_corr(
        df_item=df_item, df_fwd=df_fwd, w1=w1, w2=w2, m1=m1, m2=m2
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
        w1=param[1],
        w2=param[2],
        m1=param[3],
        m2=param[4],
        m3=param[5],
        deform=param[6],
    )


def get_adjoint_full_params(full_param):
    param = full_param[1:3]
    param_lst = pedestal.get_adjoint_params(param)
    res_lst = [full_param[0:1] + elem + full_param[3:] for elem in param_lst]
    return res_lst


# ===================================================================
# define searching space

feat_prefix = "14_history_corr"

item_lst = ["amount", "volume", "open", "high", "close", "low", "hlr", "ocr"]
w1_lst = [126, 252, 504, 756]
w2_lst = [8, 16, 32, 64, 128, 256]
m1_lst = [0, 1]
m2_lst = [0, 1]
m3_lst = [0, 1]
deform_lst = [0, 1, 2, 3, 4, 5, 6]

params = list(
    itertools.product(item_lst, w1_lst, w2_lst, m1_lst, m2_lst, m3_lst, deform_lst)
)
params = [list(elem) for elem in params]
full_params = params[:]
for elem in full_params:
    if elem[2] > elem[1] // 2:
        params.remove(elem)


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
            w1=param[1],
            w2=param[2],
            m1=param[3],
            m2=param[4],
            m3=param[5],
            deform=param[6],
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
