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


def calc_by_val_one(
    in_val: pd.DataFrame,
    in_by: pd.DataFrame,
    win_roll: int = 10,
    win_smooth: int = 1,
    quantile: int = 1,
    m1: int = 0,
    m2: int = 0,
):
    ratio = quantile * 0.1001
    df_val = in_val.copy()
    df_by = in_by.copy()

    set_val_index = set(df_val.index)
    set_by_index = set(df_by.index)
    co_index = sorted(list(set_val_index.union(set_by_index)))
    df_val = df_val.reindex(co_index)
    df_by = df_by.reindex(co_index)
    ar_val = np.array(df_val)
    ar_by = np.array(df_by)

    def _calc_sng_row(_idx):
        if _idx - win_roll + 1 < 0:
            return np.full(ar_val.shape[1], np.nan)

        part_by = ar_by[_idx - win_roll + 1 : _idx + 1]
        part_val = ar_val[_idx - win_roll + 1 : _idx + 1]
        not_nan_num = np.sum(~np.isnan(part_by), 0)

        if m1 == 0:
            top_idx = np.round(not_nan_num * ratio).astype(int)
            top_bound = -np.sort(-part_by, axis=0)[
                top_idx, list(range(part_by.shape[1]))
            ]
            top_boo = part_by >= top_bound
            ar_top = np.where(top_boo, part_val, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                top_mean = np.nanmean(ar_top, axis=0)
                all_mean = np.nanmean(part_val, axis=0)
            return [top_mean, all_mean]
        else:
            bot_idx = np.round(not_nan_num * ratio).astype(int)
            bot_bound = np.sort(part_by, axis=0)[bot_idx, list(range(part_by.shape[1]))]
            bot_boo = part_by <= bot_bound
            ar_bot = np.where(bot_boo, part_val, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                bot_mean = np.nanmean(ar_bot, axis=0)
                all_mean = np.nanmean(part_val, axis=0)
            return [bot_mean, all_mean]

    res_ext = np.full(ar_val.shape, np.nan)
    res_all = np.full(ar_val.shape, np.nan)
    for i in range(res_all.shape[0]):
        res_tmp = _calc_sng_row(i)
        res_ext[i] = res_tmp[0]
        res_all[i] = res_tmp[1]

    df_ext = pd.DataFrame(res_ext, index=df_val.index, columns=df_val.columns)
    df_all = pd.DataFrame(res_all, index=df_val.index, columns=df_val.columns)

    if m2 == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            df_fct = ops.calc_diff(df_ext, df_all)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            df_fct = df_ext / df_all

    if win_smooth > 1:
        df_fct = df_fct.rolling(window=win_smooth, min_periods=1).mean()

    return df_fct


def calc_feat(
    dt_data: dict,
    item_val: str,
    item_by: str,
    win_roll: int,
    win_smooth: int,
    quantile: int = 1,
    m1: int = 0,
    m2: int = 0,
    m3: int = 0,
    deform: int = 0,
    sign: int = 1,
):
    df_val = dt_data["day"][item_val].copy()
    df_by = dt_data["day"][item_by].copy()
    df_res = calc_by_val_one(
        in_val=df_val,
        in_by=df_by,
        win_roll=win_roll,
        win_smooth=win_smooth,
        quantile=quantile,
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
        item_val=param[0],
        item_by=param[1],
        win_roll=param[2],
        win_smooth=param[3],
        quantile=param[4],
        m1=param[5],
        m2=param[6],
        m3=param[7],
        deform=param[8],
    )


def get_adjoint_full_params(full_param):
    param = full_param[2:5]
    param_lst = pedestal.get_adjoint_params(param)
    res_lst = [full_param[0:2] + elem + full_param[5:] for elem in param_lst]
    return res_lst


# ===================================================================
# define searching space

feat_prefix = "04_by_val_one"

val_lst = [
    "amount",
    "volume",
    "open",
    "high",
    "close",
    "low",
    "ocr",
    "hlr",
]

by_lst = [
    "amount",
    "volume",
    "open",
    "high",
    "close",
    "low",
    "ocr",
    "hlr",
]

wr_lst = [3, 6, 12, 24]
ws_lst = [1, 6, 12, 24]
q_lst = [1, 3]
m1_lst = [0, 1]
m2_lst = [0, 1]
m3_lst = [0, 1]
deform_lst = [0, 1, 2, 3, 4]

params = list(
    itertools.product(
        val_lst, by_lst, wr_lst, ws_lst, q_lst, m1_lst, m2_lst, m3_lst, deform_lst
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
            item_val=param[0],
            item_by=param[1],
            win_roll=param[2],
            win_smooth=param[3],
            quantile=param[4],
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
