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

from importlib import reload

# reload(kwo)
# reload(env)
# reload(config)


# ===================================================================
# define factor


def calc_plain_std(
    df_item: pd.DataFrame, win_roll: int = 10, win_smooth: int = 1, m1: int = 0
):
    if m1 == 0:
        df_std = df_item.rolling(win_roll).std()
        df_res = df_std / df_item
    else:
        df_item = df_item.pct_change(1)
        df_res = df_item.rolling(win_roll).std()
    if win_smooth > 1:
        df_res = df_res.rolling(win_smooth).mean()
    return df_res


def calc_feat(
    dt_data: dict,
    item: str,
    win_roll: int = 10,
    win_smooth: int = 1,
    m1: int = 0,
    m2: int = 0,
    deform: int = 0,
    sign: int = 1,
):
    if item == "hlr":
        df_item = dt_data["day"]["high"] / dt_data["day"]["low"] - 1
    elif item == "ocr":
        df_item = dt_data["day"]["open"] / dt_data["day"]["close"] - 1
    else:
        df_item = dt_data["day"][item].copy()

    if m1 == 1:
        df_item = df_item.pct_change(1)
    if item == "amount" or item == "volume":
        df_item = np.log(df_item)

    df_res = calc_plain_std(
        df_item=df_item, win_roll=win_roll, win_smooth=win_smooth, m1=m1
    )
    df_res = df_res.replace([np.inf, -np.inf], np.nan)
    df_res = df_res * env.status_filter
    df_res = ops.calc_deform(df_res, deform)
    if m2 == 1:
        df_res = ops.calc_demean(df_res)
    df_res = kwo.calc_pos(df_res)
    return sign * df_res


def calc_feat_param(param):
    return calc_feat(
        dt_data=env.data,
        item=param[0],
        win_roll=param[1],
        win_smooth=param[2],
        m1=param[3],
        m2=param[4],
        deform=param[5],
    )


def get_adjoint_full_params(full_param):
    param = full_param[1:4]
    param_lst = pedestal.get_adjoint_params(param)
    res_lst = [full_param[0:1] + elem + full_param[4:] for elem in param_lst]
    return res_lst


feat_prefix = "07_plain_std"

item_lst = ["amount", "volume", "open", "high", "close", "low", "hlr", "ocr"]
w1_lst = [3, 6, 12, 24, 48]
w2_lst = [1, 6, 12, 24, 48]
m1_lst = [0, 1]
m2_lst = [0, 1]
deform_lst = [0, 1, 2, 3, 4, 5, 6]
params = list(itertools.product(item_lst, w1_lst, w2_lst, m1_lst, m2_lst, deform_lst))
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
            win_roll=param[1],
            win_smooth=param[2],
            m1=param[3],
            m2=param[4],
            deform=param[5],
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

