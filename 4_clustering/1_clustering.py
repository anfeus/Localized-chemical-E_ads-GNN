#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 23:39:13 2025

@author: usuga
"""
import pickle
from umap import UMAP
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

def UMAP_plot(list_X, y, color_umad, size, alpha, dpi_qua, mol, prop):
    
    X_2d = UMAP(n_components=2, n_neighbors=10, min_dist=0, random_state=123).fit_transform(list_X)

    min_x = int(min([min(X_2d[:, 0]), min(X_2d[:, 1])]))-2
    max_x = int(max([max(X_2d[:, 0]), max(X_2d[:, 1])]))+2
    
    p_points = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, norm=plt.Normalize(), s=size, alpha=alpha, cmap=color_umad)
    p_points.axes.axis('square')
    p_points.axes.set_xlim(min_x, max_x)
    p_points.axes.set_ylim(min_x, max_x)
    # ax.xaxis.set_ticks(np.arange(start, end, 0.712123))
    clb = plt.colorbar(p_points)
    clb.set_label(prop + ' [eV]')
    clb.solids.set(alpha=1)
    plt.xlabel('Arbitrary x-axis (a.u.)')
    plt.ylabel('Arbitrary y-axis (a.u.)')    
    plt.title(label='Clustering of the 2D reduced features\nwith the ' + prop, fontsize=13)
    plt.tight_layout()
    plt.savefig('umap_2d_' + str(mol) + '_DFT_total.png', dpi=dpi_qua, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    return X_2d

def UMAP_plot_only(X_2d, y, color_umad, size, alpha, dpi_qua, mol, prop, prop_aux, prop_name):

    min_x = int(min([min(X_2d[:, 0]), min(X_2d[:, 1])]))-2
    max_x = int(max([max(X_2d[:, 0]), max(X_2d[:, 1])]))+2
    
    p_points = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, norm=plt.Normalize(), s=size, alpha=alpha, cmap=color_umad)
    p_points.axes.axis('square')
    p_points.axes.set_xlim(min_x, max_x)
    p_points.axes.set_ylim(min_x, max_x)
    # ax.xaxis.set_ticks(np.arange(start, end, 0.712123))
    clb = plt.colorbar(p_points)
    clb.set_label(prop_aux)
    clb.solids.set(alpha=1)
    plt.xlabel('Arbitrary x-axis (a.u.)')
    plt.ylabel('Arbitrary y-axis (a.u.)')
    plt.title(label='Clustering of the 2D reduced features\nwith the ' + prop, fontsize=13)
    plt.tight_layout()
    plt.savefig('umap_2d_' + str(mol) + '_DFT_total_' + prop_name + '.png', dpi=dpi_qua, bbox_inches='tight', pad_inches=0)
    plt.show()

def sort_list(list_to_sort, index_max, index_list):
    tem_list = list(list_to_sort)
    sort_list = [tem_list[i] for i in index_list]
    sort_list = sort_list[:index_max]
    
    return sort_list

with open("Input_linear_GNN_alloys.pkl",'rb') as f:
    Input_linear_alloys = pickle.load(f)
    
with open("Input_linear_GNN_NiGa.pkl",'rb') as f:
    Input_linear_NiGa = pickle.load(f)

###Load dataset alloys    

df_alloys = pd.read_pickle(r"dict_keys_complete_alloys.pkl")
index_sort_alloys = shuffle(np.arange(len(df_alloys), dtype=int),
                            random_state=42).tolist()
Input_linear_alloys = [item[0] for item in Input_linear_alloys]

len_data_alloys = len(df_alloys)
len_data_train_alloys = int(len_data_alloys*.70)

Energy_alloys = sort_list(df_alloys.Bonding_energy, len_data_train_alloys, index_sort_alloys)
Work_function_alloys = sort_list(df_alloys.HOMO, len_data_train_alloys, index_sort_alloys)
Step_alloys = sort_list(df_alloys.Step, len_data_train_alloys, index_sort_alloys)
Features_ads_alloys = sort_list(df_alloys.Features_ads, len_data_train_alloys, index_sort_alloys)
Features_surf_alloys = sort_list(df_alloys.Features_surf, len_data_train_alloys, index_sort_alloys)


Electronegativity_ads_alloys = [item['electronegativity'] for item in Features_ads_alloys]
s_ads_alloys = [item['s'] for item in Features_ads_alloys]
at_num_ads_alloys = [item['atomic_number'] for item in Features_ads_alloys]
p_ads_alloys = [item['p'] for item in Features_ads_alloys]

radius_surf_alloys = [item['atomic_radius'] for item in Features_surf_alloys]
d_surf_alloys = [item['d'] for item in Features_surf_alloys]

###Load dataset NiGa  

with open("dict_keys_complete_NiGa.pkl", "rb") as f:
    dict_data = pickle.load(f)
df_NiGa = pd.DataFrame.from_dict(dict_data)
index_sort_NiGa = shuffle(np.arange(len(df_NiGa), dtype=int),
                            random_state=42).tolist()

len_data_NiGa = len(df_NiGa)
len_data_train_NiGa = int(len_data_NiGa*.70)

Energy_NiGa = sort_list(df_NiGa.Er, len_data_train_NiGa, index_sort_NiGa)

list_storage_Er = np.where(np.array(Energy_NiGa) <= 4)[0]

Energy_NiGa = np.array(Energy_NiGa)[list_storage_Er]
Input_linear_NiGa = np.array([item[0] for item in Input_linear_NiGa])[list_storage_Er]


Work_function_NiGa = np.array(sort_list(df_NiGa.F_work_function_metal, len_data_train_NiGa, index_sort_NiGa))[list_storage_Er]
Step_NiGa = np.array(sort_list(df_NiGa.F_step, len_data_train_NiGa, index_sort_NiGa))[list_storage_Er]
Features_ads_NiGa = np.array(sort_list(df_NiGa.Features_ads, len_data_train_NiGa, index_sort_NiGa))[list_storage_Er]
Features_surf_NiGa = np.array(sort_list(df_NiGa.Features_surf, len_data_train_NiGa, index_sort_NiGa))[list_storage_Er]
Num_O_NiGa = np.array(sort_list(df_NiGa.Num_O, len_data_train_NiGa, index_sort_NiGa))[list_storage_Er]

Electronegativity_ads_NiGa = [item[5] for item in Features_ads_NiGa]
s_ads_NiGa = [item[1] for item in Features_ads_NiGa]
at_num_ads_NiGa = [item[0] for item in Features_ads_NiGa]
p_ads_NiGa = [item[2] for item in Features_ads_NiGa]
Homo_ads_NiGa = [item[6] for item in Features_ads_NiGa]

radius_surf_NiGa = [item[4] for item in Features_surf_NiGa]
d_surf_NiGa = [item[3] for item in Features_surf_NiGa]

reaction_NiGa = sort_list(df_NiGa.Reaction, len_data_train_NiGa, index_sort_NiGa)
unique_reaction = list(dict.fromkeys(reaction_NiGa))

reaction_NiGa = np.array(reaction_NiGa)[list_storage_Er]

index_reac = []
for val in reaction_NiGa:
    index_reac.append(np.where(val==np.array(unique_reaction))[0][0])




alpha = 0.5
size = 6
dpi_qua = 600
color_umad = "viridis"
color_umad_1 = "turbo"


##Plot alloys

X_2d_alloys = UMAP_plot(Input_linear_alloys, Energy_alloys, color_umad, size, alpha, dpi_qua, 'alloys', 'Adsorption energy')

UMAP_plot_only(X_2d_alloys, Work_function_alloys, color_umad_1, size, alpha, dpi_qua, 'alloys', 'surface\'s work function', 'Work function [eV]', 'work_function')
UMAP_plot_only(X_2d_alloys, Step_alloys, color_umad_1, size, alpha, dpi_qua, 'alloys', 'surface\'s facet step', 'Facet step', 'facet_step')
UMAP_plot_only(X_2d_alloys, Electronegativity_ads_alloys, color_umad_1, size, alpha, dpi_qua, 'alloys', 'adsorbate\'s eletronegativity', 'Eletronegativity', 'eletronegativity_ads')
UMAP_plot_only(X_2d_alloys, s_ads_alloys, color_umad_1, size, alpha, dpi_qua, 'alloys', 'adsorbate\'s s-occupancy', 's-occupancy', 's-occupancy_ads')
UMAP_plot_only(X_2d_alloys, at_num_ads_alloys, color_umad_1, size, alpha, dpi_qua, 'alloys', 'adsorbate\'s atomic number', 'Atomic number', 'atomic_number_ads')
UMAP_plot_only(X_2d_alloys, p_ads_alloys, color_umad_1, size, alpha, dpi_qua, 'alloys', 'adsorbate\'s p-occupancy', 'p-occupancy', 'p-occupancy_ads')
UMAP_plot_only(X_2d_alloys, radius_surf_alloys, color_umad_1, size, alpha, dpi_qua, 'alloys', 'surface\'s atomic radius', 'Atomic radius [$\AA$]', 'atomic radius_surf')
UMAP_plot_only(X_2d_alloys, d_surf_alloys, color_umad_1, size, alpha, dpi_qua, 'alloys', 'surface\'s d-occupancy', 'd-occupancy', 'd-occupancy_surf')


##Plot NiGa

X_2d_NiGa = UMAP_plot(Input_linear_NiGa, Energy_NiGa, color_umad, size, alpha, dpi_qua, 'NiGa', 'Adsorption energy')

UMAP_plot_only(X_2d_NiGa, Work_function_NiGa, color_umad_1, size, alpha, dpi_qua, 'NiGa', 'surface\'s work function', 'Work function [eV]', 'work_function')
UMAP_plot_only(X_2d_NiGa, Step_NiGa, color_umad_1, size, alpha, dpi_qua, 'NiGa', 'surface\'s facet step', 'Facet step', 'facet_step')
UMAP_plot_only(X_2d_NiGa, Electronegativity_ads_NiGa, color_umad_1, size, alpha, dpi_qua, 'NiGa', 'adsorbate\'s eletronegativity', 'Eletronegativity', 'eletronegativity_ads')
UMAP_plot_only(X_2d_NiGa, s_ads_NiGa, color_umad_1, size, alpha, dpi_qua, 'NiGa', 'adsorbate\'s s-occupancy', 's-occupancy', 's-occupancy_ads')
UMAP_plot_only(X_2d_NiGa, at_num_ads_NiGa, color_umad_1, size, alpha, dpi_qua, 'NiGa', 'adsorbate\'s atomic number', 'Atomic number', 'atomic_number_ads')
UMAP_plot_only(X_2d_NiGa, p_ads_NiGa, color_umad_1, size, alpha, dpi_qua, 'NiGa', 'adsorbate\'s p-occupancy', 'p-occupancy', 'p-occupancy_ads')
UMAP_plot_only(X_2d_NiGa, radius_surf_NiGa, color_umad_1, size, alpha, dpi_qua, 'NiGa', 'surface\'s atomic radius', 'Atomic radius [$\AA$]', 'atomic radius_surf')
UMAP_plot_only(X_2d_NiGa, d_surf_NiGa, color_umad_1, size, alpha, dpi_qua, 'NiGa', 'surface\'s d-occupancy', 'd-occupancy', 'd-occupancy_surf')
UMAP_plot_only(X_2d_NiGa, index_reac, color_umad_1, size, alpha, dpi_qua, 'NiGa', 'reaction', 'Reaction', 'reaction')
UMAP_plot_only(X_2d_NiGa, Homo_ads_NiGa, color_umad_1, size, alpha, dpi_qua, 'NiGa', 'adsorbate\'s HOMO-LUMO', 'HOMO_LUMO [eV]', 'homo_lumo')
