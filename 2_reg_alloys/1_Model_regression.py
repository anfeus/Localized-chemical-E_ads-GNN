#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 20:11:43 2024

@author: ubuntu
"""
from torch_geometric.explain import Explainer, GNNExplainer
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

from sklearn.utils import shuffle

import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_max_pool
from torch.nn import Linear
from torch.nn.parameter import Parameter
import torch.optim as optim
import os
import time
from torcheval.metrics import MeanSquaredError, R2Score
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx

import warnings
warnings.filterwarnings("ignore")

pwd = os.getcwd()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# dpi_qua = 200



####Model GNN
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

####Function to train GNN
def train(model, data, epochs, path, optimizer, device):
    loss_ave = []
    R2_ave = []
    mse_ave = []

    for epoch in range(epochs):
        model.train()

        loss_ave_epoch = []
        mse_ave_epoch = []
        R2_ave_epoch = []

        for batch in data:

            batch.to(device)
            optimizer.zero_grad()
            y_hat = model(batch.x, batch.edge_index, batch.batch)
            loss = torch.nn.MSELoss()(y_hat, batch.y)
            loss.backward()
            optimizer.step()
        
        
        # all_preds = []
        # all_targets = []

        model.eval()
        with torch.no_grad():
            for item, batch in enumerate(data):
                batch.to(device)
                y_hat_batch = model(batch.x, batch.edge_index, batch.batch)
                y_batch = torch.reshape(batch.y, (-1,))
                if item == 0:
                    all_preds = y_hat_batch.cpu().numpy()
                    all_targets = y_batch.cpu().numpy()
                else:
                    all_preds = np.concatenate((all_preds, y_hat_batch.cpu().numpy()))
                    all_targets = np.concatenate((all_targets, y_batch.cpu().numpy()))
                
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        mse = mean_squared_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        print('Training of epoch: ', epoch, "mse %.5f" % mse, "R2 %.5f" % r2)

    torch.save(model.state_dict(), path)

    return mse, r2

####Function to test GNN
def test(model, data, device):
    
    model.eval()

    with torch.no_grad():
        for val, batch in enumerate(data):
            batch.to(device)
            y_hat_batch = model(batch.x, batch.edge_index, batch.batch)
            y_batch = torch.reshape(batch.y, (-1,))
            if val == 0:
                all_preds = y_hat_batch.cpu().numpy()
                all_targets = y_batch.cpu().numpy()
            else:
                all_preds = np.concatenate((all_preds, y_hat_batch.cpu().numpy()))
                all_targets = np.concatenate((all_targets, y_batch.cpu().numpy()))

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        mse = mean_squared_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        print("Metrics : mse %.5f" % mse, "R2 %.5f" % r2)

    return mse, r2


####Function to load weigths of model
def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=False))
    model.eval()

####Function to get predicted ML values
def predict_y_hat(model, data):
    model.eval()

    with torch.no_grad():
        for val, batch in enumerate(data):
            batch.to(device)
            y_hat_batch = model(batch.x, batch.edge_index, batch.batch)
            y_batch = torch.reshape(batch.y, (-1,))
            if val == 0:
                all_preds = y_hat_batch.cpu().numpy()
                all_targets = y_batch.cpu().numpy()
            else:
                all_preds = np.concatenate((all_preds, y_hat_batch.cpu().numpy()))
                all_targets = np.concatenate((all_targets, y_batch.cpu().numpy()))

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

    return all_preds


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

# ###Definition of hyperparameters tuning
# num_features = data_graph[0].num_features
# batch_size_tem = [16, 32, 64]
# hidden_dim = [32, 64, 128, 256, 512]      ### Number of units in hidden layer 1
# dropout = [0.1, 0.3, 0.5]
# learning_rate = [0.0005, 0.001, 0.005]

###Definition of hyperparameters final
num_features = data_graph[0].num_features
batch_size_tem = [64]
hidden_dim = [512]      ### Number of units in hidden layer 1
dropout = [0.1]
learning_rate = [0.0005]

hidden_array = []
dropout_array = []
rate_array = []
batch_array = []

R2_train_array = []
mse_train_array = []
R2_valid_array = []
mse_valid_array = []
R2_test_array = []
mse_test_array = []
time_array = []

for batch in batch_size_tem:
    for hidden in hidden_dim:
        for drop in dropout:
            for rate in learning_rate:
                
                ###Loading data as batch
                data_batch_train = DataLoader(data_train, batch_size=batch)
                data_batch_val = DataLoader(data_val, batch_size=batch)
                
                batch_array.append(batch)
                hidden_array.append(hidden)
                dropout_array.append(drop)
                rate_array.append(rate)
                start_time = time.time()
                model = GCN(num_features, hidden, drop)
                # model = AEncoder(num_features=num_features, hidden_dim=hidden, encod_dim=encod,
                #                   kernel_size=kernel_size, stride=stride, padding=padding)
                optimizer = optim.Adam(model.parameters(), lr=rate)
                model = model.to(device)
                model.cuda()
                
                # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                # print(f"Trainable parameters: {trainable_params}")
                    
                print('Training of model \n')
                print('Parameter: Hidden_dim', hidden, 'Dropout', drop, 'Learning rate', rate, 'Batch size', batch)
                mse_tem, R2_tem = train(model, data_batch_train, epochs=1000, path=path_parameters, optimizer=optimizer, device=device)
    
                total_time = float(time.time() - start_time)
    
                print("--- %s seconds ---" % total_time)
                print("--- %s minutes ---" % (total_time/60))
    
                time_array.append(total_time/60)
    
                R2_train_array.append(R2_tem)
                mse_train_array.append(mse_tem)
    
                print('\n')
                print('Results of validation set \n')
                mse_tem, R2_tem = test(model, data_batch_val, device=device)
                R2_test_array.append(R2_tem)
                mse_test_array.append(mse_tem)


##To save results for tuning hyperparameters

dict_tem = {'Batch_size':np.array(batch_array),
            'Hidden_dim':np.array(hidden_array),
            'Dropout':np.array(dropout_array),
            'Learning_rate':np.array(rate_array),
            'R2_score_train':np.array(R2_train_array),
            'MSE_train':np.array(mse_train_array),
            'R2_score_test':np.array(R2_test_array),
            'MSE_test':np.array(mse_test_array),
            'Total_time':np.array(time_array)}

pickle.dump(dict_tem, open("dict_keys_complete_values_hyperparameters_v1.pkl", "wb"))
# # data_pd = pd.DataFrame.from_dict(dict_tem)

# filename = "dict_keys_complete_values_hyperparameters_v1.pkl"

# with open(filename, "rb") as f:
#     dict_tem = pickle.load(f)
# dict_tem.pop('Loss', None)
# data_pd = pd.DataFrame.from_dict(dict_tem)

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


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
######Train
# mse_train_tem, R2_train_tem = test(model, data_batch_train, device=device)
data_batch_train_tem = DataLoader(data_train, batch_size=1)
y_hat_train = predict_y_hat(model, data_batch_train_tem).reshape(1,-1)[0]

mae_train = mean_absolute_error(y_train, y_hat_train)
mse_train = mean_squared_error(y_train, y_hat_train)
r2_train = r2_score(y_train, y_hat_train)

# error = np.mean(np.power((y_hat_train - y_train), 2))
# print(error)

######Validation
# mse_test_tem, R2_test_tem = test(model, data_batch_test, device=device)
data_batch_val_tem = DataLoader(data_val, batch_size=1)
y_hat_val = predict_y_hat(model, data_batch_val_tem).reshape(1,-1)[0]

mae_val = mean_absolute_error(y_val, y_hat_val)
mse_val = mean_squared_error(y_val, y_hat_val)
r2_val = r2_score(y_val, y_hat_val)

######Test
# mse_test_tem, R2_test_tem = test(model, data_batch_test, device=device)
data_batch_test_tem = DataLoader(data_test, batch_size=1)
y_hat_test = predict_y_hat(model, data_batch_test_tem).reshape(1,-1)[0]

mae_test = mean_absolute_error(y_test, y_hat_test)
mse_test = mean_squared_error(y_test, y_hat_test)
r2_test = r2_score(y_test, y_hat_test)

print('')

print("Metrics train: mae %.5f" % mae_train, "mse %.5f" % mse_train, "R2 %.5f" % r2_train)
print("Metrics validation: mae %.5f" % mae_val, "mse %.5f" % mse_val, "R2 %.5f" % r2_val)
print("Metrics test: mae %.5f" % mae_test, "mse %.5f" % mse_test, "R2 %.5f" % r2_test)


# Metrics train: mae 0.17766 mse 0.06905 R2 0.98249
# Metrics validation: mae 0.28770 mse 0.22740 R2 0.94141
# Metrics test: mae 0.27482 mse 0.19037 R2 0.95118

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
plt.title(label='ML vs DFT adsorption energy\nfor the GNN model', fontsize=13)
plt.tight_layout()
plt.savefig('Predict_vs_Real_alloys.png', dpi=dpi_qua)
plt.show()

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=1000),
    explanation_type='phenomenon',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='graph',
        return_type='raw',
    ),
)
seed = 8

data_batch_train = DataLoader(data_train, batch_size=1)

import pickle

def save_pickle(filename, var):
    with open(filename, "wb") as pkl_wb_obj:
        pickle.dump(var, pkl_wb_obj)
        
node_import = []
cont = 0

list_node_mask = []
list_edge_mask = []

for val, batch in enumerate(data_batch_train):
    
    print('Step number ' + str(cont))

    batch.to(device)

    explanation = explainer(x=batch.x, edge_index=batch.edge_index, batch=batch.batch, target=batch.y)
    # print(f'Generated explanations in {explanation.available_explanations}')

    node_mask = explanation.get('node_mask')
    # print(node_mask)
    node_mask = node_mask.cpu().numpy()
    list_node_mask.append(node_mask)
    
    node_sum = np.sum(node_mask,axis=0)
    node_import.append(node_sum)
    
    edge_mask = explanation.get('edge_mask')
    edge_mask = edge_mask.cpu().numpy()
    
    list_edge_mask.append(edge_mask)    
    
    cont += 1


    
node_import = np.array(node_import)
node_import_sum = np.sum(node_import,axis=0)

save_pickle("train_nodes_mask_alloys.pkl", list_node_mask)
save_pickle("train_edge_mask_alloys.pkl", list_edge_mask)
save_pickle("train_node_import_alloys.pkl", node_import)

data_batch_train = DataLoader(data_train, batch_size=1)
data_batch_test = DataLoader(data_test, batch_size=1)


feats = {} #an empty dictionary
def hook_func(m , inp ,op):
    feats['feat'] = inp

handles = []

for step, batch in enumerate(data_batch_train):
    batch.to(device)
    # print(batch)        
    model.fc.register_forward_hook(hook_func)
    out = model(batch.x, batch.edge_index, batch.batch)
    tem = feats['feat'][0].detach()
    handles.append(tem.cpu().numpy())
    # hook_handle.remove()

import pickle

def save_pickle(filename, var):
    with open(filename, "wb") as pkl_wb_obj:
        pickle.dump(var, pkl_wb_obj)

save_pickle("Input_linear_GNN_alloys.pkl", handles)

sorted_df = obj.iloc[index_sort].reset_index(drop=True)

sorted_df.to_pickle("sorted_data_clustering_alloys.pkl")
