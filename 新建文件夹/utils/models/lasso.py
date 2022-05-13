import numpy as np
import pandas as pd
import joblib

from importlib import reload
from glob import glob
import os
import sys
import shutil
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from . import model_pedestal as pedestal

from .. import kiwi_operators as kwo
from .. import config
from .. import data_environment as env

# reload(kwo)
# reload(config)
# reload(env)


class ModelLasso(object):
    def __init__(self):
        self.model = LassoCV(fit_intercept=False)
        self.scaler = StandardScaler()

    def train(self, train_date_beg: int, train_date_end: int):
        train_data = pedestal.prep_feat_target(train_date_beg, train_date_end)
        train_data = train_data.dropna()

        feat_cols = list(train_data.columns)
        feat_cols.remove("time")
        feat_cols.remove("investment")
        feat_cols.remove("target")
        x_train = np.array(train_data[feat_cols])
        y_train = np.array(train_data["target"])
        x_train = self.scaler.fit_transform(x_train)

        self.model.fit(x_train, y_train)

    def save_model(self, file_path: str):
        joblib.dump(self.model, file_path)

    def get_model(self):
        return self.model

    def predict(self, x_data: np.array):
        x_data = self.scaler.transform(x_data)
        return self.model.predict(x_data)
