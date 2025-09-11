#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from utils import get_data, split_data
import scienceplots
import matplotlib
matplotlib.use('TkAgg')
plt.style.use("science")
plt.rcParams["text.usetex"] = False  

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

X, y = get_data("1.xlsx", "Sheet1")
X_train, X_test, y_train, y_test = split_data(X, y)

X_test = pd.DataFrame(X_test, columns=['fai', 'T', 'log_gamma'])

catboost = CatBoostRegressor(iterations=500, learning_rate=0.2,l2_leaf_reg=5, depth=4, verbose=0)
catboost.fit(X_train, y_train)

explainer = shap.Explainer(catboost, X_train)
shap_values = explainer(X_test)

shap_values_numpy = shap_values.values  

pdf_file = "SHAP_combined_with_top_line_corrected.pdf"
if os.path.exists(pdf_file):
    os.remove(pdf_file)

plt.style.use('default')

shap_importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean |SHAP|': np.abs(shap_values_numpy).mean(axis=0)
})

shap_values_df = pd.DataFrame(shap_values_numpy, columns=X_test.columns)
shap_values_df.insert(0, 'Sample_Index', np.arange(len(shap_values_df)))  

output_file = "shap_data.xlsx"
with pd.ExcelWriter(output_file) as writer:
    shap_importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
    shap_values_df.to_excel(writer, sheet_name="BeeSwarm_Data", index=False)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor='white')
ax.set_facecolor("white") 
shap.summary_plot(shap_values_numpy, X_test, feature_names=X_test.columns, plot_type="dot", show=False)
plt.title("SHAP Summary Plot (Bee Swarm) - CatBoost")
plt.savefig("shap_summary_beeswarm_catboost.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor='white')
ax.set_facecolor("white") 
shap.summary_plot(shap_values_numpy, X_test, feature_names=X_test.columns, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Bar) - CatBoost")
plt.savefig("shap_summary_bar_catboost.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.close()

fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1200, facecolor='white')
ax1.set_facecolor("white")  

shap.summary_plot(shap_values_numpy, X_test, feature_names=X_test.columns, plot_type="dot", show=False, color_bar=True)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  

ax1 = plt.gca()

ax2 = ax1.twiny()
fig.patch.set_facecolor("white")  
ax2.set_facecolor("white")  
shap.summary_plot(shap_values_numpy, X_test, feature_names=X_test.columns, plot_type="bar", show=False)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  

ax2.axhline(y=len(X_test.columns) - 1, color='gray', linestyle='-', linewidth=1)

bars = ax2.patches  
for bar in bars:
    bar.set_alpha(0.2)  

ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=12)
ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=12)

ax2.xaxis.set_label_position('top')  
ax2.xaxis.tick_top() 

ax1.set_ylabel('Features', fontsize=12)

plt.tight_layout()
plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')

plt.close('all')

shap_heatmap_df = pd.DataFrame(shap_values_numpy, columns=X_test.columns)
shap_heatmap_df.insert(0, 'Sample_Index', np.arange(len(shap_heatmap_df)))

fig, ax = plt.subplots(figsize=(12, 8), dpi=300, facecolor='white')
shap.plots.heatmap(shap_values, show=False)
plt.title("SHAP Heatmap - CatBoost")
plt.savefig("shap_heatmap_catboost.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

num_waterfall_samples = 5 

waterfall_data_list = []

for sample_idx in range(num_waterfall_samples):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor='white')
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    plt.title(f"SHAP Waterfall Plot (Sample {sample_idx}) - CatBoost")
    plt.savefig(f"shap_waterfall_sample_{sample_idx}_catboost.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    sample_data = pd.DataFrame({
        'Feature': X_test.columns,
        'SHAP Value': shap_values_numpy[sample_idx],
        'Feature Value': X_test.iloc[sample_idx].values
    })
    waterfall_data_list.append(sample_data)

with pd.ExcelWriter(output_file) as writer:
    shap_importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
    shap_values_df.to_excel(writer, sheet_name="BeeSwarm_Data", index=False)
    shap_heatmap_df.to_excel(writer, sheet_name="Heatmap_Data", index=False)
    
    for idx, df in enumerate(waterfall_data_list):
        df.to_excel(writer, sheet_name=f"Waterfall_Sample_{idx}", index=False)

shap.save_html(
    "shap_force_aggregated_catboost.html",
    shap.force_plot(
        explainer.expected_value, 
        shap_values.values, 
        X_test,
        plot_cmap="RdBu",
        feature_names=X_test.columns.tolist()
    )
)

num_samples = 5  
force_data_list = []

for i in range(num_samples):
    plt.figure()
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values.values[i,:],
        features=X_test.iloc[i,:],
        matplotlib=True,
        show=False,
        feature_names=X_test.columns.tolist()
    )
    plt.title(f"SHAP Force Plot - Sample {i}", y=1.05)
    plt.tight_layout()
    plt.savefig(f"shap_force_sample_{i}_catboost.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

print("SHAP finished：")
print("- shap_summary_beeswarm_catboost.png")
print("- shap_summary_bar_catboost.png")
print(f"- {pdf_file}")
print("- shap_heatmap_catboost.png")
print(f"- shap_waterfall_sample_0_catboost.png 至 shap_waterfall_sample_4_catboost.png")
print("- shap_force_aggregated_catboost.html")
print(f"- shap_force_sample_0_catboost.png 至 shap_force_sample_4_catboost.png")

print("\nSHAP Feature Importance ：")
print(shap_importance_df.head())  

print("\nSHAP Bee Swarm Data：")
print(shap_values_df.head())  

print(f"\nSHAP 相关数据已保存到 {output_file}，包含：")
print("- Feature_Importance")
print("- BeeSwarm_Data")
print("- Heatmap_Data")
print("- Waterfall_Sample_0 至 Waterfall_Sample_4")