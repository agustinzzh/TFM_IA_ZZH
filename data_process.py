#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:58:36 2022

@author: agustinzhang
"""
import pandas as pd
import numpy as np
import math
import torch
from torch_geometric.data import Data
import torch_geometric.data as data

df_node_total = pd.read_csv('df_node_total.csv')

def get_distance( point1, point2):
    return math.sqrt((point2[0]- point1[0])**2 + (point2[1]- point1[1])**2)

list_graph = []
for i in df_node_total.graph.unique():
    df_g = df_node_total[df_node_total['graph'] == i]
    for j in df_g.Candidate.unique():
        df_c = df_g[df_g['Candidate'] == j]
        node0 = [df_c['X'][df_c['Node'] == 'Node0'].values.tolist()[0] , df_c['Y'][df_c['Node'] == 'Node0'].values.tolist()[0]]
        node1 = [df_c['X'][df_c['Node'] == 'Node1'].values.tolist()[0] , df_c['Y'][df_c['Node'] == 'Node1'].values.tolist()[0]]
        node2 = [df_c['X'][df_c['Node'] == 'Node2'].values.tolist()[0] , df_c['Y'][df_c['Node'] == 'Node2'].values.tolist()[0]]
        node3 = [df_c['X'][df_c['Node'] == 'Node3'].values.tolist()[0] , df_c['Y'][df_c['Node'] == 'Node3'].values.tolist()[0]]
        node4 = [df_c['X'][df_c['Node'] == 'Node4'].values.tolist()[0] , df_c['Y'][df_c['Node'] == 'Node4'].values.tolist()[0]]
        
        
        #sum of distance to other 3 nodes
        feature_1 = get_distance(node1, node2)+ get_distance(node1, node3)+ get_distance(node1, node4)
        feature_2 = get_distance(node2, node1)+ get_distance(node2, node3)+ get_distance(node2, node4)
        feature_3 = get_distance(node3, node1)+ get_distance(node3, node2)+ get_distance(node3, node4)
        feature_4 = get_distance(node4, node1)+ get_distance(node4, node2)+ get_distance(node4, node3)


        #distance to (0,0)
        p0 = [0,0]
        feature_1_0 = get_distance(node1, p0)
        feature_2_0 = get_distance(node2, p0)
        feature_3_0 = get_distance(node3, p0)
        feature_4_0 = get_distance(node4, p0)
        
        
        matrix_f = np.zeros((4,2))
        matrix_f[0,0] = feature_1
        matrix_f[0,1] = feature_1_0
        matrix_f[1,0] = feature_2
        matrix_f[1,1] = feature_2_0
        matrix_f[2,0] = feature_3
        matrix_f[2,1] = feature_3_0
        matrix_f[3,0] = feature_4
        matrix_f[3,1] = feature_4_0
        
        label = np.zeros((1))
        label[0] = df_c.PT.values.tolist()[0]
        
        #edges
        edge_index = []
        edge_index.append([0,1])
        edge_index.append([0,2])
        edge_index.append([0,3])
        edge_index.append([1,0])
        edge_index.append([1,2])
        edge_index.append([1,3])
        edge_index.append([2,0])
        edge_index.append([2,1])
        edge_index.append([3,0])
        edge_index.append([3,1])
        
        edge_data = torch.tensor(edge_index, dtype = torch.long)
        
        #edge feature
        edge_feature = np.zeros((10,1))
        edge_feature[0,0] = get_distance(node1, node2)
        edge_feature[1,0] = get_distance(node1, node3)
        edge_feature[2,0] = get_distance(node1, node4)
        edge_feature[3,0] = get_distance(node2, node1)
        edge_feature[4,0] = get_distance(node2, node3)
        edge_feature[5,0] = get_distance(node2, node4)
        edge_feature[6,0] = get_distance(node3, node1)
        edge_feature[7,0] = get_distance(node3, node2)
        edge_feature[8,0] = get_distance(node4, node1)
        edge_feature[9,0] = get_distance(node4, node2)
        
        edge_attr = torch.tensor(edge_feature, dtype = torch.long)
        # data、
        x = torch.tensor(matrix_f, dtype = torch.float)
        y = torch.tensor(label, dtype = torch.int)
    
        graph = Data(x = matrix_f, edge_index = edge_data.t() , edge_attr = edge_attr ,y = y, num_nodes = 4)
        list_graph.append(graph)
        
        
        
from torch.utils.data import Dataset,  random_split
from torch_geometric.loader import DataLoader

        
class graph_data_set(Dataset):
    
    def __init__(self):
        
        self.list_graph = list_graph
        
    def __len__(self):
        return len(self.list_graph) 
    
    def __getitem__(self, idx):
        return self.list_graph[idx]
    
    def split_idx(self):
        
        train_size = int(0.8 * len(self.list_graph))
        test_size = len(self.list_graph) - train_size
        train_dataset, test_dataset = random_split(self.list_graph, [train_size, test_size])
        
        return train_dataset, test_dataset
    
g_dataset = graph_data_set()
train_dataset, test_dataset = g_dataset.split_idx()
train_loader = DataLoader(train_dataset, batch_size= 32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle=False, num_workers = 0)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GCN_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(features, hidden)   
        self.conv2 = GCNConv(hidden, classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, trainning = self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim = 1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_Net(2, 16, 1).to(device)
data = train_loader.dataset
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.])
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        