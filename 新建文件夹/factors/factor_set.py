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
from time import time

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

reload(kwo)
# reload(env)
reload(config)

# import factors
from . import fct_02_two_roll_mom as fct_02
from . import fct_04_by_val_one as fct_04
from . import fct_05_by_val_two as fct_05
from . import fct_08_roll_corr_rank as fct_08
from . import fct_09_double_rank as fct_09
from . import fct_10_mul_sub as fct_10
from . import fct_12_rank_mm_sub as fct_12


class FactorSet(object):
    def __init__(self, train_date_beg: int, train_date_end: int):
        self.train_date_beg = train_date_beg
        self.train_date_end = train_date_end

        self.base_02 = fct_02.BaseFeat(self.train_date_beg, self.train_date_end)
        self.base_04 = fct_04.BaseFeat(self.train_date_beg, self.train_date_end)
        self.base_05 = fct_05.BaseFeat(self.train_date_beg, self.train_date_end)
        self.base_08 = fct_08.BaseFeat(self.train_date_beg, self.train_date_end)
        self.base_09 = fct_09.BaseFeat(self.train_date_beg, self.train_date_end)
        self.base_10 = fct_10.BaseFeat(self.train_date_beg, self.train_date_end)
        self.base_12 = fct_12.BaseFeat(self.train_date_beg, self.train_date_end)

    def search_params(self):
        st = time()
        self.base_02.search_params()
        print("fct_02 finished, cumulative time used %.2f minutes" % ((time() - st) / 60))
        self.base_04.search_params()
        print("fct_04 finished, cumulative time used %.2f minutes" % ((time() - st) / 60))
        self.base_05.search_params()
        print("fct_05 finished, cumulative time used %.2f minutes" % ((time() - st) / 60))
        self.base_08.search_params()
        print("fct_08 finished, cumulative time used %.2f minutes" % ((time() - st) / 60))
        self.base_09.search_params()
        print("fct_09 finished, cumulative time used %.2f minutes" % ((time() - st) / 60))
        self.base_10.search_params()
        print("fct_10 finished, cumulative time used %.2f minutes" % ((time() - st) / 60))
        self.base_12.search_params()
        print("fct_12 finished, cumulative time used %.2f minutes" % ((time() - st) / 60))

