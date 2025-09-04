#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:16:22 2022

@author: andres
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
from sklearn.model_selection import KFold, cross_val_score, cross_validate
import shap
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler

from catboost import CatBoostRegressor
import warnings

dpi_qua = 200

def plot_correlation(corr_name, column_name, plot_name, dpi, title):
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_name, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(column_name),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(column_name, rotation='vertical',fontsize=6)
    ax.set_yticklabels(column_name,fontsize=6)
    plt.title(label=title, fontsize=13)
    plt.tight_layout()
    plt.show()
    ax.figure.savefig(str(plot_name + '.png'), dpi=dpi)
    
def shap_beeswarm_plot(shap_values_mol, color_plot_shap, plot_name, dpi, title):
    shap.plots.beeswarm(shap_values_mol, show=False, color=plt.get_cmap(color_plot_shap))
    plt.xlim(-3,3)
    plt.title(label=title, fontsize=18)
    plt.tight_layout()
    plt.savefig('shap_' + plot_name + '.png', dpi=dpi)
    plt.show()
    
def shap_scatter_plot(shap_values_mol, color_plot_shap, plot_name, ref_color_sca, dpi):    
    shap.plots.scatter(shap_values_mol, color=ref_color_sca, show=False, cmap=color_plot_shap)
    plt.tight_layout()
    plt.savefig('shap_scatter_' + plot_name + '.png', dpi=dpi)
    plt.show()

def cluster_scatter_plot(s2d_values, color_plot_dbscan, plot_name, feature, column_feature, column_feature_stan, size_plt, alpha_plt, minx, maxx, dpi, title):    
    p_points = plt.scatter(s2d_values[:, 0], s2d_values[:, 1], c=column_feature_stan[feature], norm=plt.Normalize(), s=size_plt, alpha=alpha_plt, cmap=color_plot_dbscan)
    p_points.axes.axis('square')
    p_points.axes.set_xlim(minx, maxx)
    p_points.axes.set_ylim(minx, maxx)
    clb = plt.colorbar(p_points)
    clb.set_label(feature)
    indices_lab = np.round(np.linspace(column_feature[feature].min(), column_feature[feature].max(), num=5, endpoint=True), 2)
    indices = np.linspace(column_feature_stan[feature].min(), column_feature_stan[feature].max(), num=5, endpoint=True)
    clb.set_ticks(indices)
    clb.set_ticklabels(indices_lab)
    clb.solids.set(alpha=1)
    plt.title(label=title, fontsize=13)
    plt.tight_layout()
    plt.savefig('umap_dbscan_' + plot_name + '.png', dpi=dpi)
    plt.show()

color_plot = "viridis"
color_umad_1 = "viridis"
color_umad_2 = "tab20c"
color_umad_3 = "turbo"
model = 'catboost'

data = pd.read_pickle(r'database_total_final.pkl')
data = data.drop(columns=['Group_surf', 'SpecificHeat_surf'])

data = data[['atoms_surf', 'CN', 'gCN', 'CN_max',
              'FermiEnergy_surf', 'Total_charge_surf', 's_charge_surf', 'p_charge_surf', 'd_charge_surf', 'd_center', 'WorkFunction', 'stm_surf',
              'AtomicMass_surf', 'Electronegativity_surf', 'FirstIonization_surf', 'AtomicRadius_surf', 'MeltingPoint_surf', 'BoilingPoint_surf', 'NumberofShells_surf',
              'H_ads',
              'HOMO_ads', 'Total_charge_ads', 's_charge_ads', 'p_charge_ads', 'stm_ads',
              'AtomicMass_ads', 'Electronegativity_ads', 'Group_ads', 'FirstIonization_ads', 'AtomicRadius_ads', 'MeltingPoint_ads', 'BoilingPoint_ads', 'SpecificHeat_ads', 'NumberofShells_ads',
              'Adsorption_energy']]

data = data.astype(float)

data.drop_duplicates()

# data = data[data['Adsorption_energy'] <= 0] 

column_names = data.columns.values.tolist()

X = data.loc[:, data.columns != 'Adsorption_energy']
y = data['Adsorption_energy']

###Normalize with Robust Scaler

scaler = RobustScaler()
scaled = scaler.fit_transform(X)
scaled_robust_X = pd.DataFrame(scaled, columns=X.columns)
scaled_robust_Xy = pd.concat([scaled_robust_X, y], axis = 1)
scaled_robust_Xy_surf = pd.concat([scaled_robust_X.iloc[:, :19], y], axis = 1)
scaled_robust_Xy_ads = pd.concat([scaled_robust_X.iloc[:, 19:], y], axis = 1)

column_names = data.columns.values.tolist()
column_names_X = scaled_robust_X.columns.values.tolist()
column_names_surf = scaled_robust_Xy_surf.values.tolist()
column_names_ads = scaled_robust_Xy_ads.values.tolist()

correlations_scaled_robust_surf = scaled_robust_Xy_surf.corr()
correlations_scaled_robust_ads = scaled_robust_Xy_ads.corr()

plot_correlation(correlations_scaled_robust_surf, correlations_scaled_robust_surf, 'correlations_scaled_robust_surf', dpi_qua, 'Correlation between\nthe surface-related features')
plot_correlation(correlations_scaled_robust_ads, correlations_scaled_robust_ads, 'correlations_scaled_robust_ads', dpi_qua, 'Correlation between\nthe adsorbate-related features')

###Splitting dataset into test, validation and test sets

index_sort = shuffle(np.arange(len(X), dtype=int),
                     random_state=42).tolist()

X = X.reindex(index_sort)
y = y.reindex(index_sort)

len_data = len(X)
len_data_train = int(len_data*.70)
len_data_val = int((len_data - len_data_train)/2)
len_data_test = len_data - len_data_train - len_data_val

X_train = X.loc[:len_data_train]
X_val = X.loc[len_data_train:(len_data_train + len_data_val)]
X_test = X.loc[(len_data_train + len_data_val):]

y_train = y.loc[:len_data_train]
y_val = y.loc[len_data_train:(len_data_train + len_data_val)]
y_test = y.loc[(len_data_train + len_data_val):]

kfold = KFold(n_splits=10, shuffle=True, random_state=7)
cv = ShuffleSplit(n_splits=50, test_size=0.3, random_state=0)

cat_model = CatBoostRegressor(n_estimators=1300, learning_rate=0.12, depth=6, loss_function='LogCosh', silent=True, min_data_in_leaf=0, l2_leaf_reg=1.4)
cat_model.fit(X_train,y_train)


######Train
y_hat_train = cat_model.predict(X_train)

mae_train = mean_absolute_error(y_train, y_hat_train)
mse_train = mean_squared_error(y_train, y_hat_train)
r2_train = r2_score(y_train, y_hat_train)


######Validation
y_hat_val = cat_model.predict(X_val)

mae_val = mean_absolute_error(y_val, y_hat_val)
mse_val = mean_squared_error(y_val, y_hat_val)
r2_val = r2_score(y_val, y_hat_val)

######Test
y_hat_test = cat_model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_hat_test)
mse_test = mean_squared_error(y_test, y_hat_test)
r2_test = r2_score(y_test, y_hat_test)


print('')

print("Metrics train: mae %.5f" % mae_train, "mse %.5f" % mse_train, "R2 %.5f" % r2_train)
print("Metrics validation: mae %.5f" % mae_val, "mse %.5f" % mse_val, "R2 %.5f" % r2_val)
print("Metrics test: mae %.5f" % mae_test, "mse %.5f" % mse_test, "R2 %.5f" % r2_test)


lin_teo = np.arange(-10, 2, 0.1)
plt.scatter(lin_teo, lin_teo, linestyle='dashed', color='grey', s=2, alpha=0.5)
plt.scatter(y_train, y_hat_train, color='blue', label='Train set', s=8, alpha=0.4)
plt.scatter(y_test, y_hat_test, color='green', label='Test set', s=8, alpha=0.3)
plt.xlabel('DFT Energy [eV]')
plt.xlim([-11, 3])
plt.ylim([-11, 3])
plt.ylabel('ML predicted Energy [eV]')
plt.axis('square')
plt.legend()
plt.title(label='ML vs DFT adsorption energy\nfor the CatBoost model', fontsize=13)
plt.tight_layout()
plt.savefig('Predict_vs_Real_alloys.png', dpi=dpi_qua)
plt.show()