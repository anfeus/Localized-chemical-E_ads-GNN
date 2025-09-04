#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 20:54:08 2025

@author: usuga
"""

import pickle
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

import warnings
warnings.filterwarnings("ignore")

cmap = get_cmap('viridis')
norm = Normalize(vmin=0.1, vmax=0.55)

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

y = list(obj['Bonding_energy'])

y_Er = [y[i] for i in index_sort]

y_train = np.array(y_Er[:len_data_train])
y_val = np.array(y_Er[len_data_train:(len_data_train + len_data_val)])
y_test = np.array(y_Er[(len_data_train + len_data_val):])

structures = list(obj['Structure'])
structures_Er = [structures[i] for i in index_sort]

structures_train = structures_Er[:len_data_train]
structures_val = structures_Er[len_data_train:(len_data_train + len_data_val)]
structures_test = structures_Er[(len_data_train + len_data_val):]


filename = "train_nodes_mask_alloys.pkl"

with open(filename, "rb") as f:
    train_nodes_mask = pickle.load(f)

val_min = np.min([np.min(i) for i in train_nodes_mask])
val_max = np.max([np.max(i) for i in train_nodes_mask])

dpi_qua = 600


node_import = []

for i in train_nodes_mask:
    node_import.append(np.mean(i, axis=0))

node_import = np.mean(node_import, axis=0)


index_dec = np.argsort(node_import)

Features = ['Atomic_number', 'Work_function', 's', 'p', 'd', 'f', 'Atomic_radius', 'Electronegativity', 'Step']

node_import = [node_import[i] for i in index_dec]
Features_sort = [Features[i] for i in index_dec]

x_range = np.arange(len(node_import))

fig, ax = plt.subplots()

colors = cmap(norm(node_import[1:]))

ax.barh(Features_sort[1:], node_import[1:], color=colors)
# ax.set_yticks(y_pos, labels=people)
# ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Impact a.u.')
ax.set_title('Average feature impact for all atoms in the GNN model')
ax.xaxis.set_ticks(np.arange(0, 0.55, 0.1))
plt.xlim(0, 0.55)
plt.tight_layout()
plt.savefig('feature_importance_alloys.png', dpi=dpi_qua, bbox_inches='tight')
plt.show()
plt.clf()

####Surface and adsorbates
atoms_ads = ['H', 'C', 'O', 'N', 'S']

list_surf_train = []
list_ads_train = []
for struc in structures_train:
    list_surf_train_tem = []
    list_ads_train_tem = []
    for ind, elem in enumerate(struc.get_chemical_symbols()):
        if elem not in atoms_ads:
            list_surf_train_tem.append(ind)
        else:
            list_ads_train_tem.append(ind)
    list_surf_train.append(list_surf_train_tem)
    list_ads_train.append(list_ads_train_tem)

node_import_surf = []
node_import_ads = []
for ind, elem in enumerate(train_nodes_mask):
    node_import_surf.append(np.mean(elem[list_surf_train[ind]], axis=0))
    node_import_ads.append(np.mean(elem[list_ads_train[ind]], axis=0))

node_import_surf = np.mean(node_import_surf, axis=0)
node_import_ads = np.mean(node_import_ads, axis=0)

index_dec = np.argsort(node_import_surf)

Features = ['Atomic_number', 'Work_function', 's', 'p', 'd', 'f', 'Atomic_radius', 'Electronegativity', 'Step']

node_import_surf = [node_import_surf[i] for i in index_dec]
Features_sort = [Features[i] for i in index_dec]

x_range = np.arange(len(node_import_surf))

fig, ax = plt.subplots()

colors = cmap(norm(node_import_surf[1:]))

ax.barh(Features_sort[1:], node_import_surf[1:], color=colors)
# ax.set_yticks(y_pos, labels=people)
# ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Impact a.u.')
ax.set_title('Average feature impact for the surface atoms in the GNN model')
ax.xaxis.set_ticks(np.arange(0, 0.55, 0.1))
plt.xlim(0, 0.55)
plt.tight_layout()
plt.savefig('feature_importance_alloys_surf.png', dpi=dpi_qua, bbox_inches='tight')
plt.show()
plt.clf()



index_dec = np.argsort(node_import_ads)

Features = ['Atomic_number', 'HOMO_LUMO', 's', 'p', 'd', 'f', 'Atomic_radius', 'Electronegativity', 'Step']

node_import_ads = [node_import_ads[i] for i in index_dec]
Features_sort = [Features[i] for i in index_dec]

x_range = np.arange(len(node_import_ads))

fig, ax = plt.subplots()

colors = cmap(norm(node_import_ads[2:]))

ax.barh(Features_sort[2:], node_import_ads[2:], color=colors)
# ax.set_yticks(y_pos, labels=people)
# ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Impact a.u.')
ax.set_title('Average feature impact for the adsorbate atoms in the GNN model')
ax.xaxis.set_ticks(np.arange(0, 0.55, 0.1))
plt.xlim(0, 0.55)
plt.tight_layout()
plt.savefig('feature_importance_alloys_ads.png', dpi=dpi_qua, bbox_inches='tight')
plt.show()
plt.clf()