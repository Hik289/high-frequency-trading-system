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


def calc_roll_corr_rank(df_a: pd.DataFrame, df_b: pd.DataFrame, win: int):
    df_a = ops.calc_cs_rank(df_a)
    df_b = ops.calc_cs_rank(df_b)
    df_a = kwo.calc_pos(df_a)
    df_b = kwo.calc_pos(df_b)
    df_res = pd.DataFrame(np.nan, index=df_a.index, columns=df_a.columns)
    for col in df_res.columns:
        df_res[col] = df_a[col].rolling(win).corr(df_b[col])
    return df_res


# win_1: rolling_mean
# win_2: roll_corr
# win_3: smooth
def calc_feat(
    dt_data: dict,
    item_1: str,
    item_2: str,
    win_1: int = 10,
    win_2: int = 10,
    win_3: int = 10,
    m1: int = 0,
    deform: int = 0,
    sign: int = 1,
):
    if item_1 == "hlr":
        df_1 = dt_data["day"]["high"] / dt_data["day"]["low"] - 1
    elif item_1 == "ocr":
        df_1 = dt_data["day"]["open"] / dt_data["day"]["close"] - 1
    else:
        df_1 = dt_data["day"][item_1].pct_change(1)

    if item_2 == "hlr":
        df_2 = dt_data["day"]["high"] / dt_data["day"]["low"] - 1
    elif item_2 == "ocr":
        df_2 = dt_data["day"]["open"] / dt_data["day"]["close"] - 1
    else:
        df_2 = dt_data["day"][item_2].pct_change(1)

    df_1 = df_1.rolling(win_1).mean()
    df_2 = df_2.rolling(win_2).mean()
    df_res = calc_roll_corr_rank(df_1, df_2, win=win_2)
    df_res = df_res.replace([np.inf, -np.inf], np.nan)
    if win_3 > 1:
        df_res = df_res.rolling(win_3).mean()
    df_res = df_res * env.status_filter
    df_res = ops.calc_deform(df_res, deform)
    if m1 == 1:
        df_res = ops.calc_demean(df_res)
    df_res = kwo.calc_pos(df_res)
    return sign * df_res


def calc_feat_param(param):
    return calc_feat(
        dt_data=env.data,
        item_1=param[0],
        item_2=param[1],
        win_1=param[2],
        win_2=param[3],
        win_3=param[4],
        m1=param[5],
        deform=param[6],
    )


def get_adjoint_full_params(full_param):
    param = full_param[2:5]
    param_lst = pedestal.get_adjoint_params(param)
    res_lst = [full_param[0:2] + elem + full_param[5:] for elem in param_lst]
    return res_lst


# ===================================================================
# define searching space
feat_prefix = "08_roll_corr_rank"

items = [
    "amount",
    "volume",
    "open",
    "high",
    "close",
    "low",
    "ocr",
    "hlr",
]

pair_lst = list()
for idx in range(len(items)):
    for jdx in range(len(items)):
        if idx < jdx:
            pair_lst.append([items[idx], items[jdx]])

w1_lst = [3, 6, 12, 24, 48]
w2_lst = [3, 6, 12, 24, 48]
w3_lst = [1, 6, 12, 24, 48]
m1_lst = [0, 1]
deform_lst = [0, 1, 2, 3, 4, 5, 6]

params = list()
for pair in pair_lst:
    for w1 in w1_lst:
        for w2 in w2_lst:
            for w3 in w3_lst:
                for m1 in m1_lst:
                    for deform in deform_lst:
                        param = pair + [w1] + [w2] + [w3] + [m1] + [deform]
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
        # ---------------------------------------
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
        # ---------------------------------------
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
            win_1=param[2],
            win_2=param[3],
            win_3=param[4],
            m1=param[5],
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
