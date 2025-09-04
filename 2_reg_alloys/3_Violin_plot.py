#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 23:15:23 2025

@author: usuga
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import os

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_max_pool
from torch.nn import Linear
import torch.optim as optim

pwd = os.getcwd()

torch.manual_seed(0)

# # dpi_qua = 200

def sort_list(list_to_sort, index_max, index_list):
    tem_list = list(list_to_sort)
    sort_list = [tem_list[i] for i in index_list]
    sort_list = sort_list[:index_max]
    
    return sort_list

class GCN(torch.nn.Module):
    def __init__(self, num_features, dim, dropout):
        super(GCN, self).__init__()

        self.dropout = dropout

        self.conv1 = SAGEConv(num_features, dim)
        self.conv2 = SAGEConv(dim, dim)
        self.conv3 = SAGEConv(dim, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.fc = Linear(dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(
            x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(
            x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)

        x = global_max_pool(x, batch)

        x = self.fc(x)

        return x

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=False))
    model.eval()

def predict_y_hat(model, data):
    model.eval()

    y_hat = []

    with torch.no_grad():
        for batch in data:
            batch.to(device)
            y_hat_tem = model(batch.x, batch.edge_index, batch.batch)

            y_hat.append(y_hat_tem.cpu().numpy())

    return np.array(y_hat)


device = torch.device('cuda:0')

obj = pd.read_pickle(r"dict_keys_complete_alloys.pkl")
filename = "dict_keys_complete_alloys.pkl"

data_graph = list(obj['Graphs'])

index_sort = shuffle(np.arange(len(data_graph), dtype=int),
                      random_state=42).tolist()

data = [data_graph[i] for i in index_sort]

###Splitting dataset into test, validation and test sets
len_data = len(data)
len_data_train = int(len_data*.70)
len_data_val = int((len_data - len_data_train)/2)
len_data_test = len_data - len_data_train - len_data_val

data_train = data[:len_data_train]
data_val = data[len_data_train:(len_data_train + len_data_val)]
data_test = data[(len_data_train + len_data_val):]

version = 'v1'
sub_parameters = 'parameters_model_' + version
path_parameters = pwd + '/' + sub_parameters

###Definition of hyperparameters final
num_features = data_graph[0].num_features

filename = "dict_keys_complete_values_hyperparameters_v1.pkl"

batch_size_tem = 64
hidden_dim = 512      ### Number of units in hidden layer 1
dropout = 0.1
learning_rate = 0.0005


model = GCN(num_features, hidden_dim, dropout)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model = model.to(device)
model.cuda()
load_model(model, path=path_parameters)

y = list(obj['Bonding_energy'])

y_Er = [y[i] for i in index_sort]

y_train = np.array(y_Er[:len_data_train]) # 1,2,4,8,11,22,44,61,88,122,244,488,671,1342,2684,5368
y_val = np.array(y_Er[len_data_train:(len_data_train + len_data_val)])
y_test = np.array(y_Er[(len_data_train + len_data_val):])
dpi_qua = 600

######Train
# mse_train_tem, R2_train_tem = test(model, data_batch_train, device=device)
data_batch_train_tem = DataLoader(data_train, batch_size=1)
y_hat_train = predict_y_hat(model, data_batch_train_tem).reshape(1,-1)[0]

######Validation
# mse_test_tem, R2_test_tem = test(model, data_batch_test, device=device)
data_batch_val_tem = DataLoader(data_val, batch_size=1)
y_hat_val = predict_y_hat(model, data_batch_val_tem).reshape(1,-1)[0]

######Test
# mse_test_tem, R2_test_tem = test(model, data_batch_test, device=device)
data_batch_test_tem = DataLoader(data_test, batch_size=1)
y_hat_test = predict_y_hat(model, data_batch_test_tem).reshape(1,-1)[0]

#Load dataset

##Split data into train and test
Energy_alloys_train = np.array(sort_list(obj.Bonding_energy, len_data_train, index_sort))
Features_surf_alloys_train = sort_list(obj.Features_surf, len_data_train, index_sort)
radius_surf_alloys_train = np.array([item['atomic_radius'] for item in Features_surf_alloys_train])

diff_energy_train = y_hat_train - Energy_alloys_train

ads_alloys = sort_list(obj.Adsorbate, len_data_train, index_sort)
unique_ads = np.array(['C', 'CH', 'CH2', 'CH3', 'O', 'OH', 'H2O', 'N', 'NH', 'S', 'SH', 'H'])

index_ads = []
for ads in unique_ads:
    index_ads.append(np.array(np.where(ads==np.array(ads_alloys))[0],dtype=int))

energy_index_ads = []
radius_index_ads = []
for i in index_ads:
    energy_index_ads.append(Energy_alloys_train[i])
    radius_index_ads.append(radius_surf_alloys_train[i])


fig, ax = plt.subplots()

vp = ax.violinplot(energy_index_ads, showmeans=False, showmedians=True, widths=0.7)

for pc in vp['bodies']:
    pc.set_facecolor('#D43F3A')  # custom color
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
    pc.set_linewidth(0.3)

vp['cmedians'].set_color('black')
vp['cmedians'].set_linewidth(0.5)

vp['cmaxes'].set_color('black')
vp['cmaxes'].set_linewidth(0.5)
vp['cmins'].set_color('black')
vp['cmins'].set_linewidth(0.5)
vp['cbars'].set_color('black')
vp['cbars'].set_linewidth(0.5)


for i, group in enumerate(energy_index_ads):
    x = np.random.normal(i+1, 0.05, size=len(group))  # jitter x-values
    sc = ax.scatter(x, group, c=radius_index_ads[i], alpha=0.6, s=1, norm=plt.Normalize(), cmap="viridis")

clb = plt.colorbar(sc)
clb.set_label('Atomic radius [$\AA$]')
clb.solids.set(alpha=1)
ax.set_title('Violin plot for type of adsorbate')
ax.set_xlabel('Type of adsorbate')
ax.set_ylabel('Adsorption energy [eV]')
ax.set_xticks(np.arange(1, len(unique_ads)+1))
ax.set_xticklabels(unique_ads)

# Show plot
plt.tight_layout()
plt.savefig('violin_alloys_ads_energy.png', dpi=600)
plt.show()

energy_index_ads = []
radius_index_ads = []
for i in index_ads:
    energy_index_ads.append(diff_energy_train[i])
    radius_index_ads.append(radius_surf_alloys_train[i])


fig, ax = plt.subplots()

vp = ax.violinplot(energy_index_ads, showmeans=False, showmedians=True, widths=0.7)

for pc in vp['bodies']:
    pc.set_facecolor('#D43F3A')  # custom color
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
    pc.set_linewidth(0.3)

vp['cmedians'].set_color('black')
vp['cmedians'].set_linewidth(0.5)

vp['cmaxes'].set_color('black')
vp['cmaxes'].set_linewidth(0.5)
vp['cmins'].set_color('black')
vp['cmins'].set_linewidth(0.5)
vp['cbars'].set_color('black')
vp['cbars'].set_linewidth(0.5)


for i, group in enumerate(energy_index_ads):
    x = np.random.normal(i+1, 0.05, size=len(group))  # jitter x-values
    sc = ax.scatter(x, group, c=radius_index_ads[i], alpha=0.6, s=1, norm=plt.Normalize(), cmap="viridis")

clb = plt.colorbar(sc)
clb.set_label('Atomic radius [$\AA$]')
clb.solids.set(alpha=1)
ax.set_title('Violin plot for type of adsorbate')
ax.set_xlabel('Type of adsorbate')
ax.set_ylabel('MAE [eV]')
ax.set_xticks(np.arange(1, len(unique_ads)+1))
ax.set_xticklabels(unique_ads)

# Show plot
plt.tight_layout()
plt.savefig('violin_alloys_mae_ads_energy.png', dpi=600)
plt.show()