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
def calc_one_roll_mom(dataframe: pd.DataFrame, win_i: int, win_j: int, mode: int):
    r_i = dataframe.rolling(win_i).mean()
    r_j = dataframe.rolling(win_j).mean()
    if mode == 0:
        return ops.calc_diff(r_i, r_j)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            df_res = r_i / r_j
        return df_res


def calc_feat(
    dt_data: dict,
    item: str,
    order,
    win_i: int,
    win_j: int,
    m1: int,
    m2: int,
    deform: int = 0,
    sign: int = 1,
):
    dataframe = dt_data["day"][item].copy()
    if order == 1:
        dataframe = dataframe.pct_change(1)
    df_res = calc_one_roll_mom(dataframe=dataframe, win_i=win_i, win_j=win_j, mode=m1)
    df_res = df_res.replace([np.inf, -np.inf], np.nan)
    df_res = df_res * env.status_filter
    df_res = ops.calc_deform(df_res, mode=deform)
    if m2 == 1:
        df_res = ops.calc_demean(df_res)
    df_res = kwo.calc_pos(df_res)
    return sign * df_res


def calc_feat_param(param):
    return calc_feat(
        dt_data=env.data,
        item=param[0],
        order=param[1],
        win_i=param[2],
        win_j=param[3],
        m1=param[4],
        m2=param[5],
        deform=param[6],
    )


def get_adjoint_full_params(full_param):
    param = full_param[2:4]
    param_lst = pedestal.get_adjoint_params(param)
    res_lst = [full_param[0:2] + elem + full_param[4:] for elem in param_lst]
    return res_lst


# ===================================================================
# define searching space
feat_prefix = "01_one_roll_mom"

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
orders = [0, 1]
win_1_lst = [3, 6, 12, 24, 48]
win_2_lst = [3, 6, 12, 24, 48]
m1_lst = [0, 1]
m2_lst = [0, 1]
deform_lst = [0, 1, 2, 3, 4]
params = list(
    itertools.product(items, orders, win_1_lst, win_2_lst, m1_lst, m2_lst, deform_lst)
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
            order=param[1],
            win_i=param[2],
            win_j=param[3],
            m1=param[4],
            m2=param[5],
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
