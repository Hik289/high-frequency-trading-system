import numpy as np
import pandas as pd
import joblib

from importlib import reload
from glob import glob
import os
import sys
import shutil

sys.path.append("..")
import kiwi_operators as kwo

import kiwi_operators as kwo
import config
import data_environment as env

reload(kwo)
reload(config)
reload(env)


def prep_feat(date_beg: int, date_end: int):
    path_lst = glob(f"{config.dir_selected_feat}/data/*.pkl")
    name_lst = kwo.get_names_from_paths(path_lst)
    feat_lst = [pd.read_pickle(elem) * env.status_filter for elem in path_lst]
    feat_lst = [
        kwo.get_partial_dataframe_by_date(elem, date_beg, date_end) for elem in feat_lst
    ]
    print(feat_lst[0])
    res_data = kwo.stack_feats(feat_lst, name_lst)
    return res_data.dropna(how="all")


def prep_feat_target(date_beg: int, date_end: int):
    path_lst = glob(f"{config.dir_selected_feat}/data/*.pkl")
    name_lst = kwo.get_names_from_paths(path_lst)
    feat_lst = [pd.read_pickle(elem) * env.status_filter for elem in path_lst]
    feat_lst = [
        kwo.get_partial_dataframe_by_date(elem, date_beg, date_end) for elem in feat_lst
    ]
    cur_target = kwo.get_partial_dataframe_by_date(env.fwd_ret, date_beg, date_end)
    res_data = kwo.stack_feats_target(feat_lst, cur_target, name_lst)
    return res_data.dropna(how="all")
