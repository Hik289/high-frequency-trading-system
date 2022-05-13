import numpy as np
import pandas as pd
import joblib

from importlib import reload
from glob import glob
import os
import sys
import shutil

sys.path.append("..")
from .. import kiwi_operators as kwo
from .. import config
from .. import data_environment as env

# reload(kwo)
# reload(config)
# reload(env)


def prep_feat(date_beg: int, date_end: int):
    path_lst = glob(f"{config.dir_selected_feat}/data/*.pkl")
    name_lst = kwo.get_names_from_paths(path_lst)
    feat_lst = [pd.read_pickle(elem) * env.status_filter for elem in path_lst]
    feat_lst = [
        kwo.get_partial_dataframe_by_date(elem, date_beg, date_end) for elem in feat_lst
    ]
    res_data = kwo.stack_feats(feat_lst, name_lst)
    return res_data


def prep_feat_target(
    date_beg: int,
    date_end: int,
    target_demean: bool = False,
    target_normalization: bool = True,
    target_standardization: bool = False,
):
    path_lst = glob(f"{config.dir_selected_feat}/data/*.pkl")
    name_lst = kwo.get_names_from_paths(path_lst)
    feat_lst = [pd.read_pickle(elem) * env.status_filter for elem in path_lst]
    feat_lst = [
        kwo.get_partial_dataframe_by_date(elem, date_beg, date_end) for elem in feat_lst
    ]
    cur_target = env.fwd_ret.copy()
    if target_standardization:
        se_std = cur_target.std(axis=0)
        cur_target = cur_target.div(se_std, axis=1)
    if target_demean:
        cur_target = kwo.calc_demean(cur_target)
    if target_normalization:
        cur_target = kwo.calc_pos(cur_target)

    cur_target = kwo.get_partial_dataframe_by_date(cur_target, date_beg, date_end)
    res_data = kwo.stack_feats_target(feat_lst, cur_target, name_lst)
    return res_data
