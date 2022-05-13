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
def calc_mul_sub(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    df_3: pd.DataFrame,
    df_4: pd.DataFrame,
    w1: int = 10,
    w2: int = 1,
    m1: int = 0,
    m2: int = 0,
    m3: int = 0,
):
    if m1 == 1:
        df_1 = df_1.pct_change(1)
        df_2 = df_2.pct_change(1)
        df_3 = df_3.pct_change(1)
        df_4 = df_4.pct_change(1)
    if w1 > 0:
        df_1 = df_1.rolling(w1).mean()
        df_2 = df_2.rolling(w1).mean()
        df_3 = df_3.rolling(w1).mean()
        df_4 = df_4.rolling(w1).mean()
    if m2 == 0:
        df_a = ops.calc_diff(df_1, df_2)
        df_b = ops.calc_diff(df_3, df_4)
    else:
        df_a = df_1 / df_2 - 1
        df_b = df_3 / df_4 - 1
    if m3 == 1:
        df_a = ops.calc_cs_rank(df_a)
        df_b = ops.calc_cs_rank(df_b)

    df_res = df_a * df_b
    if w2 > 1:
        df_res = df_res.rolling(w2).mean()
    return df_res


def calc_feat(
    dt_data: dict,
    item_1: str,
    item_2: str,
    item_3: str,
    item_4: str,
    w1: int = 10,
    w2: int = 1,
    m1: int = 0,
    m2: int = 0,
    m3: int = 0,
    m4: int = 0,
    deform: int = 0,
    sign: int = 1,
):
    df_1 = dt_data["day"][item_1]
    df_2 = dt_data["day"][item_2]
    df_3 = dt_data["day"][item_3]
    df_4 = dt_data["day"][item_4]
    df_res = calc_mul_sub(
        df_1=df_1, df_2=df_2, df_3=df_3, df_4=df_4, w1=w1, w2=w2, m1=m1, m2=m2, m3=m3
    )
    df_res = df_res.replace([np.inf, -np.inf], np.nan)
    df_res = df_res * env.status_filter
    df_res = ops.calc_deform(df_res, deform)
    if m4 == 1:
        df_res = ops.calc_demean(df_res)
    df_res = kwo.calc_pos(df_res)
    return sign * df_res


def calc_feat_param(param):
    return calc_feat(
        dt_data=env.data,
        item_1=param[0],
        item_2=param[1],
        item_3=param[2],
        item_4=param[3],
        w1=param[4],
        w2=param[5],
        m1=param[6],
        m2=param[7],
        m3=param[8],
        m4=param[9],
        deform=param[10],
    )


def get_adjoint_full_params(full_param):
    param = full_param[4:6]
    param_lst = pedestal.get_adjoint_params(param)
    res_lst = [full_param[0:4] + elem + full_param[6:] for elem in param_lst]
    return res_lst


# ===================================================================
# define searching space
feat_prefix = "10_mul_sub"

item_lst = [
    "open",
    "high",
    "close",
    "low",
]
w1_lst = [3, 6, 12, 24, 48]
w2_lst = [1, 12, 24, 48]
m1_lst = [0, 1]
m2_lst = [0, 1]
m3_lst = [0, 1]
m4_lst = [0, 1]
deform_lst = [0, 1, 2, 3, 4, 5, 6]

params = list(
    itertools.product(
        item_lst,
        item_lst,
        item_lst,
        item_lst,
        w1_lst,
        w2_lst,
        m1_lst,
        m2_lst,
        m3_lst,
        m4_lst,
        deform_lst,
    )
)
params = [list(elem) for elem in params]
full_params = params[:]
params = list()
for elem in full_params:
    if item_lst.index(elem[0]) >= item_lst.index(elem[1]):
        continue
    if item_lst.index(elem[2]) >= item_lst.index(elem[3]):
        continue
    if item_lst.index(elem[0]) > item_lst.index(elem[2]):
        continue
    if elem not in params:
        params.append(elem)


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
            item_3=param[2],
            item_4=param[3],
            w1=param[4],
            w2=param[5],
            m1=param[6],
            m2=param[7],
            m3=param[8],
            m4=param[9],
            deform=param[10],
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
