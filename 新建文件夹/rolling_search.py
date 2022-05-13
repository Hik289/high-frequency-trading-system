# ------------------------------------------
# data environment loading
import pandas as pd
import numpy as np
import joblib

# kwo: customized functions
# config: project configuration
# env: raw data storage
from utils import kiwi_operators as kwo
from utils import config
from utils import data_environment as env
from importlib import reload
from glob import glob
import os
import shutil

os.environ["OMP_NUM_THREADS"] = "1"

print("environment loaded")

from factors.factor_set import FactorSet
from utils.selector import FeatureSelector

for train_date_pair in config.train_date_pairs:
    train_date_beg = train_date_pair[0]
    train_date_end = train_date_pair[1]

    # ---------------------------------------------------------------------
    print(f"searching...")
    shutil.rmtree(f"{config.dir_feat}", ignore_errors=True)
    os.makedirs(f"{config.dir_feat}/data", exist_ok=True)
    os.makedirs(f"{config.dir_feat}/func", exist_ok=True)

    factors = FactorSet(train_date_beg, train_date_end)
    factors.search_params()

    old_dir = f"{config.dir_feat}"
    new_dir = f"{config.dir_rolling_feat}/{train_date_beg}_{train_date_end}"
    shutil.rmtree(new_dir, ignore_errors=True)
    shutil.copytree(old_dir, new_dir)

    # ---------------------------------------------------------------------
    print(f"selecting...")
    shutil.rmtree(f"{config.dir_selected_feat}", ignore_errors=True)
    os.makedirs(f"{config.dir_selected_feat}/data", exist_ok=True)
    os.makedirs(f"{config.dir_selected_feat}/func", exist_ok=True)

    try:
        train_selector = FeatureSelector(
            train_date_beg=train_date_beg, train_date_end=train_date_end
        )
        train_selector.filter_copy_features(verbose=True)
    except:
        pass

    old_dir = f"{config.dir_selected_feat}"
    new_dir = f"{config.dir_rolling_selected_feat}/{train_date_beg}_{train_date_end}"
    shutil.rmtree(new_dir, ignore_errors=True)
    shutil.copytree(old_dir, new_dir)

    print(f"task finished")
    print()

