#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from utils import get_data, split_data

X, y = get_data("1.xlsx", "Sheet1")
X_train, X_test, y_train, y_test = split_data(X, y)

catboost = CatBoostRegressor(
    iterations=500, 
    learning_rate=0.2,
    l2_leaf_reg=5, 
    depth=4, 
    verbose=0
)

catboost.fit(X_train, y_train)

pred_catboost = catboost.predict(X_test)

mse = mean_squared_error(y_test, pred_catboost)
mape = mean_absolute_percentage_error(y_test, pred_catboost) * 100
r2_score = r2_score(y_test, pred_catboost)

print("CatBoost Performance:")
print(f"MSE: {mse:.6f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2_score:.6f}")

results_df = pd.DataFrame({
    'Model': ['CatBoost'],
    'MSE': [mse],
    'MAPE (%)': [mape],
    'R2': [r2_score]
})

results_df.to_excel('catboost_performance.xlsx', index=False)

prediction_data = pd.DataFrame(X_test, columns=['fai', 'T', 'log(γ)'])
prediction_data['Actual Y'] = y_test
prediction_data['CatBoost'] = pred_catboost

with pd.ExcelWriter('catboost_performance.xlsx', mode='a', engine='openpyxl') as writer:
    prediction_data.to_excel(writer, sheet_name='Predictions', index=False)

print("Results are saved to catboost_performance.xlsx")