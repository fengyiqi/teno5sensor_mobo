from __future__ import annotations

import math
from abc import ABC, abstractmethod
from math import pi
from typing import Optional

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.test_functions.base import (
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
from botorch.utils.sampling import sample_hypersphere, sample_simplex
from botorch.utils.transforms import unnormalize
from scipy.special import gamma
from torch import Tensor
from torch.distributions import MultivariateNormal
import numpy as np
import xml.etree.ElementTree as ET
import os
from boiles.objective.sodshocktube import Sod
from botorch.utils.sampling import draw_sobol_samples
from boiles.objective.tgv import TaylorGreenVortex
import pandas as pd

SILENT = True

class TENO5Objectives(MultiObjectiveTestProblem):
    r"""OPT problems.
    """

    dim = 5
    num_objectives = 2
    _bounds = [(-4, np.log10(1/3)), (1.0, 20.0), (1, 6), (0.2, 0.4), (0.2, 0.4)]
    _decimals = [5, 0, 0, 4, 4]
    _ref_point = [0.0, 0.0]
    # _max_hv = 59.36011874867746  # this is approximated using NSGA-II

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)
        self.sod_count, self.tgv_count = 0, 0
        self.sod_res, self.tgv_res = [], []
        self.sod_ref = Sod("baselines/sod_64_normal/domain/data_0.200*.h5")
        self.tgv_ref = TaylorGreenVortex("baselines/tgv_64_normal/domain/data_10.000*.h5")
        self.train_x = []
        if os.path.exists("data.csv"):
            self.ref_df = pd.read_csv("data.csv")
        else:
            self.ref_df = None

    def configure_param_xml(self, X: list, file: str = None):
        assert len(X) == 5, "shape of X must be 1 X d"
        # assert type(X) is list, "X should be a list"
        path = "scheme.xml" if file is None else file
        tree = ET.ElementTree(file=path)
        root = tree.getroot()
        for i, x in enumerate(X):
            x = round(10**x, 5) if i == 0 else x
            # root[1] is the parameters for teno5_sensor
            root[1][i].text = str(x)
        tree.write("scheme.xml")

    def evaluate_true(self, X: Tensor) -> Tensor:
        for x in X.numpy():
            self.train_x.append(x)
            self.sod_dispersion(x)
            self.tgv_iles(x)
            data = np.hstack((self.train_x, self.sod_res, self.tgv_res))
            df = pd.DataFrame(data=data, columns=["H_ct", "H_c", "H_q", "H_eta", "L_eta", "obj_disper", "obj_iles"])
            df.to_csv("data_runtime.csv", index=None)
        return torch.tensor(np.hstack((self.sod_res, self.tgv_res)))
    
    def run_tgv(self):
        os.system("ulimit -s unlimited")
        return os.system(f"mpiexec -n 4 ./ALPACA_3D_TENO5SENSOR ./tgv_64.xml")
    
    def tgv_iles_obj(self, n: int):
        tgv = TaylorGreenVortex(f"runtime_data/tgv_64_{n}/domain/data_10.000*.h5")
        if tgv.result_exit:
            return tgv.objective_spectrum() / self.tgv_ref.objective_spectrum() - 1
        else:
            return 1

    def tgv_iles(self, x):
        self.tgv_count += 1
        if self.ref_df is not None:
            tgv_iles_stored = self.ref_df.loc[(self.ref_df['H_ct'] == x[0]) & (self.ref_df['H_c'] == x[1]) & (self.ref_df['H_q'] == x[2]) & (self.ref_df['H_eta'] == x[3]) & (self.ref_df['L_eta'] == x[4]), 'obj_iles'].values
            if len(tgv_iles_stored) > 0:
                print("Find obj_iles in database!")
                self.tgv_res.append(tgv_iles_stored)
                return
        self.configure_param_xml(x)
        assert self.run_tgv() == 0, "tgv simulation is not successful"
        self.rename_folder("tgv_64", self.tgv_count)
        self.tgv_res.append([self.tgv_iles_obj(self.tgv_count)])
    
    def run_sod(self):
        return os.system("mpiexec -n 1 ./ALPACA_1D_TENO5SENSOR ./sod_64.xml")
    
    def rename_folder(self, folder: str, n: int):
        if not os.path.exists("runtime_data"):
            os.system("mkdir runtime_data")
        os.system(f"mv {folder} runtime_data/{folder}_{str(n)}")
        
    def sod_dispersion_obj(self, n: int):
        sod = Sod(f"runtime_data/sod_64_{n}/domain/data_0.200*.h5")
        return sod.highorder_reconstructed_rhs() / self.sod_ref.highorder_reconstructed_rhs() - 1

    def sod_dispersion(self, x):
        self.sod_count += 1
        if self.ref_df is not None:
            sod_disper_stored = self.ref_df.loc[(self.ref_df['H_ct'] == x[0]) & (self.ref_df['H_c'] == x[1]) & (self.ref_df['H_q'] == x[2]) & (self.ref_df['H_eta'] == x[3]) & (self.ref_df['L_eta'] == x[4]), 'obj_disper'].values
            if len(sod_disper_stored) > 0:
                print("Find obj_disper in database!")
                self.sod_res.append(sod_disper_stored)
                return
        self.configure_param_xml(x)
        assert self.run_sod() == 0, "sod simulation is not successfull"
        self.rename_folder("sod_64", self.sod_count)
        self.sod_res.append([self.sod_dispersion_obj(self.sod_count)])
            



if __name__ == "__main__":

    os.system("rm -rf runtime_data")
    torch.random.manual_seed(1)
    obj = TENO5Objectives().to(dtype=torch.float64)
    train_x = draw_sobol_samples(bounds=obj.bounds,n=2, q=1).squeeze(1)
    for i in range(train_x.shape[1]):
        train_x[:, i] = torch.round(train_x[:, i], decimals=obj._decimals[i])
    res = obj(train_x).numpy()
    print(res)
