import numpy as np
import pandas as pd
import joblib

from importlib import reload
from glob import glob
import os
import sys
import shutil

from .models import model_pedestal as pedestal
from .models.lasso import ModelLasso
from .models.xgb import ModelXgb
from .models.lgb import ModelLgb
from .models.cat import ModelCat

from . import kiwi_operators as kwo
from . import config
from . import data_environment as env

# reload(kwo)
# reload(config)
# reload(env)


class Combiner(object):
    def __init__(self):
        self.model_01 = ModelLasso()
        self.model_02 = ModelXgb()
        self.model_03 = ModelLgb()
        self.model_04 = ModelCat()

    def train(self, train_date_beg: int, train_data_end: int, verbose: bool = False):
        self.model_01.train(train_date_beg, train_data_end)
        if verbose:
            print("model_01 trained")
        self.model_02.train(train_date_beg, train_data_end)
        if verbose:
            print("model_02 trained")
        self.model_03.train(train_date_beg, train_data_end)
        if verbose:
            print("model_03 trained")
        self.model_04.train(train_date_beg, train_data_end)
        if verbose:
            print("model_04 trained")

    def predict(self, date_beg: int, date_end: int):
        data = pedestal.prep_feat(date_beg, date_end).fillna(0)
        feat_cols = list(data.columns)
        feat_cols.remove("time")
        feat_cols.remove("investment")
        x_data = np.array(data[feat_cols])

        pred_01 = self.model_01.predict(x_data).reshape(-1,)
        pred_02 = self.model_02.predict(x_data).reshape(-1,)
        pred_03 = self.model_03.predict(x_data).reshape(-1,)
        pred_04 = self.model_04.predict(x_data).reshape(-1,)

        arr_pred = 2 * pred_01 + pred_02 + pred_03 + pred_04
        # arr_pred = pred_02
        df_res = data[["time", "investment"]]
        df_res["pred"] = arr_pred
        df_res = df_res.set_index(["time", "investment"])
        df_res = df_res["pred"]
        df_res.name = None
        df_res = df_res.unstack()
        df_res.index.name = None
        df_res.columns.name = None
        df_res = df_res * kwo.get_partial_dataframe_by_date(
            env.status_filter, date_beg=date_beg, date_end=date_end
        )
        df_res = kwo.calc_pos(df_res)
        df_res.index.name = None
        return df_res

    def evaluate_models(self, date_beg: int, date_end: int):
        data = pedestal.prep_feat_target(date_beg, date_end).dropna(how="all").fillna(0)
        feat_cols = list(data.columns)
        feat_cols.remove("time")
        feat_cols.remove("investment")
        feat_cols.remove("target")
        x_data = np.array(data[feat_cols])

        pred_01 = self.model_01.predict(x_data).reshape(-1,)
        pred_02 = self.model_02.predict(x_data).reshape(-1,)
        pred_03 = self.model_03.predict(x_data).reshape(-1,)
        pred_04 = self.model_04.predict(x_data).reshape(-1,)

        target = np.array(data["target"])

        res_dt = dict()
        res_dt["ic_01"] = np.corrcoef(target, pred_01)[0, 1]
        res_dt["ic_02"] = np.corrcoef(target, pred_02)[0, 1]
        res_dt["ic_03"] = np.corrcoef(target, pred_03)[0, 1]
        res_dt["ic_04"] = np.corrcoef(target, pred_04)[0, 1]

        return res_dt
