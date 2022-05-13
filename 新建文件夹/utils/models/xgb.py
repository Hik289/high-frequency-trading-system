import numpy as np
import pandas as pd
import joblib

from importlib import reload
from glob import glob
import os
import sys
import shutil
from . import model_pedestal as pedestal
from xgboost import XGBRegressor

from .. import kiwi_operators as kwo
from .. import config
from .. import data_environment as env

# reload(kwo)
# reload(config)
# reload(env)


class ModelXgb(object):
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            gamma=0.01,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.01,
            n_jobs=os.cpu_count() * 3 // 4,
        )

    def train(self, train_date_beg: int, train_date_end: int, verbose: bool = False):
        train_data = pedestal.prep_feat_target(train_date_beg, train_date_end)
        train_data = train_data.dropna()

        feat_cols = list(train_data.columns)
        feat_cols.remove("time")
        feat_cols.remove("investment")
        feat_cols.remove("target")
        x_train = np.array(train_data[feat_cols])
        y_train = np.array(train_data["target"])

        self.model.fit(x_train, y_train, verbose=verbose)

    def save_model(self, file_path: str):
        joblib.dump(self.model, file_path)

    def get_model(self):
        return self.model

    def predict(self, x_data: np.array):
        return self.model.predict(x_data)

