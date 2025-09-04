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
import torch

from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from scipy.spatial import distance

import warnings
warnings.filterwarnings("ignore")

cmap = get_cmap('viridis')
norm = Normalize(vmin=0.1, vmax=0.55)

pca = PCA(n_components=2)

def spread_points(points_2d, min_dist=0.05, max_iter=100):
    
    points_2d = points_2d.copy()

    for _ in range(max_iter):
        dists = distance.squareform(distance.pdist(points_2d))
        np.fill_diagonal(dists, np.inf)  # Ignore self-distance
        
        close_points = np.where(dists < min_dist)
        
        if len(close_points[0]) == 0:
            break  # Done! all points are separated enough
        
        for i, j in zip(*close_points):
            if i < j:  # Only move one of the two points
                direction = points_2d[i] - points_2d[j]
                if np.linalg.norm(direction) == 0:
                    direction = np.random.rand(2) - 0.5  # random small move
                direction /= np.linalg.norm(direction)
                points_2d[i] += direction * (min_dist - dists[i, j]) / 2

    return points_2d

def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

device = torch.device('cuda:0')


filename = "dict_keys_complete_NiGa.pkl"

with open(filename, "rb") as f:
    dict_data = pickle.load(f)

data_pd = pd.DataFrame.from_dict(dict_data)
data_graph = dict_data['Graphs']

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

y = dict_data['Er']

y_Er = [y[i] for i in index_sort]

y_train = np.array(y_Er[:len_data_train])
y_val = np.array(y_Er[len_data_train:(len_data_train + len_data_val)])
y_test = np.array(y_Er[(len_data_train + len_data_val):])

structures = dict_data['Structures']

structures_Er = [structures[i] for i in index_sort]

structures_train = structures_Er[:len_data_train]
structures_val = structures_Er[len_data_train:(len_data_train + len_data_val)]
structures_test = structures_Er[(len_data_train + len_data_val):]

color = dict_data['Color']
color_Er = [color[i] for i in index_sort]
color_train = color_Er[:len_data_train]
color_val = color_Er[len_data_train:(len_data_train + len_data_val)]
color_test = color_Er[(len_data_train + len_data_val):]

filename = "train_nodes_mask_NiGa.pkl"

with open(filename, "rb") as f:
    train_nodes_mask = pickle.load(f)
    
filename = "train_edge_mask_NiGa.pkl"
with open(filename, "rb") as f:
    train_edges_mask = pickle.load(f)
    
val_min = np.min([np.min(i) for i in train_nodes_mask])
# val_min = -0.1
val_max = np.max([np.max(i) for i in train_nodes_mask])
# val_max = 0.55

alphas = []
for i in train_nodes_mask:
    alphas.append(np.mean(i, axis=1))

dpi_qua = 600

reaction = dict_data['Reaction']
unique_reaction = list(dict.fromkeys(reaction))
reaction_Er = [reaction[i] for i in index_sort]
reaction_train = reaction_Er[:len_data_train]
reaction_val = reaction_Er[len_data_train:(len_data_train + len_data_val)]
reaction_test = reaction_Er[(len_data_train + len_data_val):]

num_reaction = [0, 1, 3, 5]
save_fig_plt = True
index_text = ['min', 'max']
Features = ['Atomic_number', 's', 'p', 'd', 'Atomic_radius', 'Electronegativity', 'Work_function', 'Step']


for ind in num_reaction:
    # angle_plot = [[10, 20], [-100, -130], [-100, -130], [-120, -130], [-100, -130],
    #               [-120, -170], [-110, -180]]
    
    index_reac = [i for i, val in enumerate(reaction_train) if val==unique_reaction[ind]]
    
    y_train_index = y_train[index_reac]
    y_train_min = np.min(y_train_index)
    y_train_max = np.max(y_train_index)
    index = [int([i for i, val in enumerate(reaction_train) if (val==unique_reaction[ind]) and (y_train[i]==y_train_min)][0]),
              int([i for i, val in enumerate(reaction_train) if (val==unique_reaction[ind]) and (y_train[i]==y_train_max)][0])]
    
    intermediate = ['HCOO', 'CO', 'CO', 'CH3O', 'CH3O', 'C', 'C']
    for ind_sca, scale in enumerate(index_text):
        fig, ax = plt.subplots()
        im = ax.imshow(train_nodes_mask[index[ind_sca]].T, cmap='viridis', norm=Normalize(vmin=val_min, vmax=val_max))
        ax.set_yticks(np.arange(len(Features)), labels=Features)
        ax.set_xticks(np.arange(len(train_nodes_mask[index[ind_sca]])))
        # cbar = ax.figure.colorbar(im, ax=ax) 
        plt.title("Heatmap atoms and impact of features")
        plt.xlabel('Index of atom')
        plt.savefig('Heatmap_' + intermediate[ind] + '_' + scale + '.png', dpi=dpi_qua, bbox_inches='tight', transparent=True)
        plt.show() 
        plt.clf()
               
        edge = data_train[index[ind_sca]].edge_index.cpu().numpy().T
        
        # print(edge[0])
                
        pos = structures_train[index[ind_sca]].get_positions()
        pos_mean = np.mean(pos, axis=0)
        pos = pos - pos_mean        
        
        
        # pos_2d = spread_points(pos, min_dist=2)
        pos_2d = pca.fit_transform(pos)
        pos_2d = spread_points(pos_2d, min_dist=1.3)
        # pos_2d = tse.fit_transform(pos)
        
        min_x = int(min([min(pos_2d[:, 0]), min(pos_2d[:, 1])]))-2
        max_x = int(max([max(pos_2d[:, 0]), max(pos_2d[:, 1])]))+2
        
        edge_2d = []
        
        for u, v in edge:
            # print(u, v)
            edge_2d.append(np.array([pos_2d[u], pos_2d[v]]).T)
            # = np.array([(pos_2d[u], pos_2d[v]) for u, v in edge])
        
        # Create the 3D figure
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axes.axis('square')
        ax.axes.set_xlim(min_x, max_x)
        ax.axes.set_ylim(min_x, max_x)
        
        # Plot the edges
        for vizedge in edge_2d:
            ax.plot(vizedge[0], vizedge[1], color="black", lw=0.8)
            
        # Plot the nodes - alpha is scaled by "depth" automatically
        for val_node, node in enumerate(pos_2d):
            ax.scatter(node[0], node[1], s=700, c=color_train[index[ind_sca]][val_node], linewidths=2, marker='o', edgecolors='black', zorder=2)
        
        for val_node, node in enumerate(pos_2d):
            ax.text(*(node-np.array([0.15, 0.1])).T, str(val_node))
                    # ax.text(*(node).T, str(val_node))
        
        
        fig.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.savefig('Nodes_label_' + intermediate[ind] + '_' + scale + '_pca.png', dpi=dpi_qua, bbox_inches='tight', transparent=True)
        plt.show()
        plt.clf()
        
        
        # Create the 3D figure
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axes.axis('square')
        ax.axes.set_xlim(min_x, max_x)
        ax.axes.set_ylim(min_x, max_x)
        
        # Plot the edges
        for vizedge in edge_2d:
            ax.plot(vizedge[0], vizedge[1], color="black", lw=0.8)
            
        # Plot the nodes - alpha is scaled by "depth" automatically
        for val_node, node in enumerate(pos_2d):
            ax.scatter(node[0], node[1], s=700, c=color_train[index[ind_sca]][val_node], linewidths=2, marker='o', edgecolors='black', zorder=2)
        
        # for val_node, node in enumerate(pos_2d):
        #     ax.text(*(node-np.array([0.15, 0.1])).T, str(val_node))
        #             # ax.text(*(node).T, str(val_node))
        
        
        fig.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.savefig('Nodes_' + intermediate[ind] + '_' + scale + '_pca.png', dpi=dpi_qua, bbox_inches='tight', transparent=True)
        plt.show()
        plt.clf()       

node_import = []

for i in train_nodes_mask:
    node_import.append(np.mean(i, axis=0))

node_import = np.mean(node_import, axis=0)


index_dec = np.argsort(node_import)

Features = ['Atomic_number', 's', 'p', 'd', 'Atomic_radius', 'Electronegativity', 'Work_function', 'Step']

node_import = [node_import[i] for i in index_dec]
Features_sort = [Features[i] for i in index_dec]

x_range = np.arange(len(node_import))

fig, ax = plt.subplots()

colors = cmap(norm(node_import))

ax.barh(Features_sort, node_import, color=colors)
# ax.set_yticks(y_pos, labels=people)
# ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Impact a.u.')
ax.set_title('Average feature impact for all atoms in the GNN model')
ax.xaxis.set_ticks(np.arange(0, 0.7, 0.1))
plt.xlim(0, 0.7)
plt.tight_layout()
plt.savefig('feature_importance_NiGa.png', dpi=dpi_qua, bbox_inches='tight')
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
    # print(ind, list_surf_train[ind])
    node_import_surf.append(np.mean(elem[list_surf_train[ind]], axis=0))    
    node_import_ads.append(np.mean(elem[list_ads_train[ind]], axis=0))

node_import_surf = np.mean(node_import_surf, axis=0)
node_import_ads = np.mean(node_import_ads, axis=0)

index_dec = np.argsort(node_import_surf)

Features = ['Atomic_number', 's', 'p', 'd', 'Atomic_radius', 'Electronegativity', 'Work_function', 'Step']

node_import_surf = [node_import_surf[i] for i in index_dec]
Features_sort = [Features[i] for i in index_dec]

x_range = np.arange(len(node_import_surf))

fig, ax = plt.subplots()

colors = cmap(norm(node_import_surf))

ax.barh(Features_sort, node_import_surf, color=colors)
# ax.set_yticks(y_pos, labels=people)
# ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Impact a.u.')
ax.set_title('Average feature impact for the surface atoms in the GNN model')
ax.xaxis.set_ticks(np.arange(0, 0.7, 0.1))
plt.xlim(0, 0.7)
plt.tight_layout()
plt.savefig('feature_importance_NiGa_surf.png', dpi=dpi_qua, bbox_inches='tight')
plt.show()
plt.clf()



index_dec = np.argsort(node_import_ads)

Features = ['Atomic_number', 's', 'p', 'd', 'Atomic_radius', 'Electronegativity', 'HOMO_LUMO', 'Step']

node_import_ads = [node_import_ads[i] for i in index_dec]
Features_sort = [Features[i] for i in index_dec]

x_range = np.arange(len(node_import_ads))

fig, ax = plt.subplots()

colors = cmap(norm(node_import_ads[1:]))

ax.barh(Features_sort[1:], node_import_ads[1:], color=colors)
# ax.set_yticks(y_pos, labels=people)
# ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Impact a.u.')
ax.set_title('Average feature impact for the adsorbate atoms in the GNN model')
ax.xaxis.set_ticks(np.arange(0, 0.7, 0.1))
plt.xlim(0, 0.7)
plt.tight_layout()
plt.savefig('feature_importance_NiGa_ads.png', dpi=dpi_qua, bbox_inches='tight')
plt.show()
plt.clf()