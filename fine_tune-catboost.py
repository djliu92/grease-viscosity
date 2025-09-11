#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from utils import get_data, split_data, randomized_search#grid_search#
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# prepare data
X, y = get_data("1.xlsx", "Sheet1")
X_train, X_test, y_train, y_test = split_data(X, y)

# init models
nn = MLPRegressor(random_state=42)
pls = PLSRegression()
gpr = GaussianProcessRegressor()
knn = KNeighborsRegressor()
extra_tree = ExtraTreeRegressor(random_state=42)
gbdt = GradientBoostingRegressor(random_state=42)
lightgbm = LGBMRegressor(random_state=42)
catboost = CatBoostRegressor()



# catboost for SVR
catboost_param_grid = {
    'iterations': [500,1000,2000],
    'learning_rate': [0.05, 0.1, 0.2, 0.3],
    'depth':[4,6,10],#, 0.1, 1
    'l2_leaf_reg': [1, 3, 5, 10]  
}
catboost_best_params, catboost_best_score = randomized_search(
    X_train, y_train, catboost, catboost_param_grid
)
print(catboost_best_params)