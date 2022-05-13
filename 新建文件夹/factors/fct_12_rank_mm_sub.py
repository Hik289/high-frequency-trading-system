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

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append("../")
from utils import config
from utils import kiwi_operators as kwo
from utils import data_environment as env
from . import operators as ops
from . import search_pedestal as pedestal
from utils.feature_selection import packed_selectors as ps

# ===================================================================
# define factor


def calc_rank_mm_sub(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    w1: int = 10,
    w2: int = 10,
    m1: int = 0,
    m2: int = 0,
):
    if m1 == 1:
        df_1 = df_1.pct_change(1)
        df_2 = df_2.pct_change(1)
    if w1 > 1:
        df_1 = df_1.rolling(w1).mean()
        df_2 = df_2.rolling(w2).mean()
    df_sub = ops.calc_diff(df_1, df_2)
    df_max = ops.calc_ts_max(df_sub, win=w2)
    df_min = ops.calc_ts_min(df_sub, win=w2)
    df_max = ops.calc_cs_rank(df_max)
    df_min = ops.calc_cs_rank(df_min)
    if m2 == 0:
        df_res = df_max + df_min
    else:
        df_res = df_max - df_min
    return df_res


def calc_feat(
    dt_data: dict,
    item_1: str,
    item_2: str,
    w1: int,
    w2: int,
    m1: int,
    m2: int,
    m3: int,
    deform: int = 0,
    sign: int = 1,
):
    df_1 = dt_data["day"][item_1].copy()
    df_2 = dt_data["day"][item_2].copy()
    df_res = calc_rank_mm_sub(df_1=df_1, df_2=df_2, w1=w1, w2=w2, m1=m1, m2=m2)
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
        item_1=param[0],
        item_2=param[1],
        w1=param[2],
        w2=param[3],
        m1=param[4],
        m2=param[5],
        m3=param[6],
        deform=param[7],
    )


def get_adjoint_full_params(full_param):
    param = full_param[2:4]
    param_lst = pedestal.get_adjoint_params(param)
    res_lst = [full_param[0:2] + elem + full_param[4:] for elem in param_lst]
    return res_lst


# ===================================================================
# define searching space

feat_prefix = "12_rank_mm_sub"

item_lst = [
    "open",
    "high",
    "close",
    "low",
]

w1_lst = [1, 5, 7, 15, 30, 60, 120]
w2_lst = [5, 7, 15, 30, 60]
m1_lst = [0, 1]
m2_lst = [0, 1]
m3_lst = [0, 1]
deform_lst = [0, 1, 2, 3, 4, 5, 6]

num_item = len(item_lst)
pair_lst = list()
for idx in range(num_item):
    for jdx in range(num_item):
        if idx < jdx:
            pair_lst.append([item_lst[idx], item_lst[jdx]])

params = list()
for pair in pair_lst:
    for w1 in w1_lst:
        for w2 in w2_lst:
            for m1 in m1_lst:
                for m2 in m2_lst:
                    for m3 in m3_lst:
                        for deform in deform_lst:
                            param = pair + [w1, w2, m1, m2, m3, deform]
                            params.append(param)


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
            item_1=param[0],
            item_2=param[1],
            w1=param[2],
            w2=param[3],
            m1=param[4],
            m2=param[5],
            m3=param[6],
            deform=param[7],
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

