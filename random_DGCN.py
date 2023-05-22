#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:22:34 2022

@author: agustinzhang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import pandas as pd
import numpy as np
import math
from stellargraph import StellarGraph, StellarDiGraph
import random

df_node_total = pd.read_csv('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/df_node_total.csv')
df_node_test = pd.read_csv('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/df_node_test.csv')

def get_distance( point1, point2):
    return math.sqrt((point2[0]- point1[0])**2 + (point2[1]- point1[1])**2)

# df_node_processed = df_node_total
# for i in df_node_total.graph.unique():
#     df_g = df_node_total[df_node_total['graph'] == i]
#     df_no_PT = df_g[df_g.PT == 7]
    
#     candidate_no_PT = list(set(df_no_PT.Candidate.values.tolist()))
#     df_PT = df_g[df_g.PT != 7]
#     num_PT = len(df_PT.PT.unique())
#     remove = random.sample(candidate_no_PT, len(candidate_no_PT)-(num_PT//5))
    
#     index_remove = []
#     for candidate in remove:
#         index = df_g[df_g['Candidate'] == candidate].index.values.tolist()
#         index_remove.extend(index)

#     df_node_processed = df_node_processed.drop(index_remove)
    

def process_node(df):
    pt_count = df.PT.value_counts()
    min_count = pt_count.min()
    remove_num = (pt_count-min_count)/5
    for PT in df.PT.unique():
        df_PT = df[df.PT == PT]
        candidate_PT = list(set(df_PT.Candidate.values.tolist()))
        remove = random.sample(candidate_PT, int(remove_num[PT]))
        df = df[~df.Candidate.isin(remove)]
        
    return df
        
        
    


df_node_processed = process_node(df_node_total)
df_node_processed_test = process_node(df_node_test)

def generate_dataset(df_node_processed):
    
    graphs = []
    graph_label = []
    for i in df_node_processed.graph.unique():
        df_g = df_node_processed[df_node_processed['graph'] == i]
        for j in df_g.Candidate.unique():
            df_c = df_g[df_g['Candidate'] == j]
            node0 = [df_c['X'][df_c['Point'] == 'e'].values.tolist()[0] , df_c['Y'][df_c['Point'] == 'e'].values.tolist()[0]]
            node1 = [df_c['X'][df_c['Point'] == 'a'].values.tolist()[0] , df_c['Y'][df_c['Point'] == 'a'].values.tolist()[0]]
            node2 = [df_c['X'][df_c['Point'] == 'b'].values.tolist()[0] , df_c['Y'][df_c['Point'] == 'b'].values.tolist()[0]]
            node3 = [df_c['X'][df_c['Point'] == 'c'].values.tolist()[0] , df_c['Y'][df_c['Point'] == 'c'].values.tolist()[0]]
            node4 = [df_c['X'][df_c['Point'] == 'd'].values.tolist()[0] , df_c['Y'][df_c['Point'] == 'd'].values.tolist()[0]]
        
        
            #sum of distance to other 3 nodes
            feature_1 = get_distance(node1, node2)+ get_distance(node1, node3)+ get_distance(node1, node4)
            feature_2 = get_distance(node2, node1)+ get_distance(node2, node3)+ get_distance(node2, node4)
            feature_3 = get_distance(node3, node1)+ get_distance(node3, node2)+ get_distance(node3, node4)
            feature_4 = get_distance(node4, node1)+ get_distance(node4, node2)+ get_distance(node4, node3)
        
            feature_1_x = node1[0]
            feature_1_y = node1[1]
            feature_2_x = node2[0]
            feature_2_y = node2[1]
            feature_3_x = node3[0]
            feature_3_y = node3[1]
            feature_4_x = node4[0]
            feature_4_y = node4[1]
        
            feature_1_0 = get_distance(node1, node0)
            feature_2_0 = get_distance(node2, node0)
            feature_3_0 = get_distance(node3, node0)
            feature_4_0 = get_distance(node4, node0)
        
            square_node_data = pd.DataFrame(
                {"x": [1, 2, 3, 4], "y":[feature_1, feature_2, feature_3, feature_4],  "c": [feature_1_0, feature_2_0, feature_3_0, feature_4_0]}, index = ["d", "b", "c", "a"])
            #"a": [feature_1_x, feature_2_x, feature_3_x, feature_4_x],
            #"b": [feature_1_y, feature_2_y, feature_3_y, feature_4_y],
        
            square_edges = pd.DataFrame(
                {"source": ['d', 'd', 'd', 'b', 'b', 'a'], "target":['b','c','a','c','a', 'c']})
        
            square_node_features = StellarDiGraph(square_node_data, square_edges)
            square_named_node_features = StellarDiGraph(
                {"corner": square_node_data}, {"line": square_edges})
        
            edge_feature1 = get_distance(node1, node2)
            edge_feature2 = get_distance(node1, node3)
            edge_feature3 = get_distance(node1, node4)
            edge_feature4 = get_distance(node2, node3)
            edge_feature5 = get_distance(node2, node4)
            edge_feature6 = get_distance(node3, node4)
        
            square_edge_data = pd.DataFrame(
                {
                    "source": ['d', 'd', 'd', 'b', 'b','a'],
                    "target": ['b','c','a','c','a','c'],
                    "Weights": [edge_feature1, edge_feature2, edge_feature3, edge_feature4, edge_feature5, edge_feature6]})
        
            # square_edges_types = square_edge_data.assign(
            #     orientation= ['side', 'side', 'side', 'common_side', 'side'])
        
            squared_named_features = StellarDiGraph(
                {"corner": square_node_data}, edges = square_edge_data )
        
        
        
            graphs.append(squared_named_features)
            graph_label.append(df_c.PT.values.tolist()[0])
        graph_labels = pd.DataFrame()
        graph_labels['label'] = graph_label
        graph_labels = graph_labels.label
        
    return graph_labels, graphs

graph_labels, graphs = generate_dataset(df_node_processed)
graph_labels_test, graphs_test = generate_dataset(df_node_processed_test)
    

# graphs = []
# graph_label = []
# for i in df_node_processed.graph.unique():
#     df_g = df_node_processed[df_node_processed['graph'] == i]
#     for j in df_g.Candidate.unique():
#         df_c = df_g[df_g['Candidate'] == j]
#         node0 = [df_c['X'][df_c['Point'] == 'e'].values.tolist()[0] , df_c['Y'][df_c['Point'] == 'e'].values.tolist()[0]]
#         node1 = [df_c['X'][df_c['Point'] == 'a'].values.tolist()[0] , df_c['Y'][df_c['Point'] == 'a'].values.tolist()[0]]
#         node2 = [df_c['X'][df_c['Point'] == 'b'].values.tolist()[0] , df_c['Y'][df_c['Point'] == 'b'].values.tolist()[0]]
#         node3 = [df_c['X'][df_c['Point'] == 'c'].values.tolist()[0] , df_c['Y'][df_c['Point'] == 'c'].values.tolist()[0]]
#         node4 = [df_c['X'][df_c['Point'] == 'd'].values.tolist()[0] , df_c['Y'][df_c['Point'] == 'd'].values.tolist()[0]]
        
        
#         #sum of distance to other 3 nodes
#         feature_1 = get_distance(node1, node2)+ get_distance(node1, node3)+ get_distance(node1, node4)
#         feature_2 = get_distance(node2, node1)+ get_distance(node2, node3)+ get_distance(node2, node4)
#         feature_3 = get_distance(node3, node1)+ get_distance(node3, node2)+ get_distance(node3, node4)
#         feature_4 = get_distance(node4, node1)+ get_distance(node4, node2)+ get_distance(node4, node3)
        
#         feature_1_x = node1[0]
#         feature_1_y = node1[1]
#         feature_2_x = node2[0]
#         feature_2_y = node2[1]
#         feature_3_x = node3[0]
#         feature_3_y = node3[1]
#         feature_4_x = node4[0]
#         feature_4_y = node4[1]
        
#         feature_1_0 = get_distance(node1, node0)
#         feature_2_0 = get_distance(node2, node0)
#         feature_3_0 = get_distance(node3, node0)
#         feature_4_0 = get_distance(node4, node0)
        
#         square_node_data = pd.DataFrame(
#             {"x": [1, 2, 3, 4], "y":[feature_1, feature_2, feature_3, feature_4],  "c": [feature_1_0, feature_2_0, feature_3_0, feature_4_0]}, index = ["d", "b", "c", "a"])
# #"a": [feature_1_x, feature_2_x, feature_3_x, feature_4_x],
# #"b": [feature_1_y, feature_2_y, feature_3_y, feature_4_y],
        
#         square_edges = pd.DataFrame(
#             {"source": ['d', 'd', 'd', 'b', 'b', 'a'], "target":['b','c','a','c','a', 'c']})
        
#         square_node_features = StellarDiGraph(square_node_data, square_edges)
#         square_named_node_features = StellarDiGraph(
#             {"corner": square_node_data}, {"line": square_edges})
        
#         edge_feature1 = get_distance(node1, node2)
#         edge_feature2 = get_distance(node1, node3)
#         edge_feature3 = get_distance(node1, node4)
#         edge_feature4 = get_distance(node2, node3)
#         edge_feature5 = get_distance(node2, node4)
#         edge_feature6 = get_distance(node3, node4)
        
#         square_edge_data = pd.DataFrame(
#             {
#                 "source": ['d', 'd', 'd', 'b', 'b','a'],
#                 "target": ['b','c','a','c','a','c'],
#                 "Weights": [edge_feature1, edge_feature2, edge_feature3, edge_feature4, edge_feature5, edge_feature6]})
        
#         # square_edges_types = square_edge_data.assign(
#         #     orientation= ['side', 'side', 'side', 'common_side', 'side'])
        
#         squared_named_features = StellarDiGraph(
#             {"corner": square_node_data}, edges = square_edge_data )
        
        
        
#         graphs.append(squared_named_features)
#         graph_label.append(df_c.PT.values.tolist()[0])
#     graph_labels = pd.DataFrame()
#     graph_labels['label'] = graph_label
#     graph_labels = graph_labels.label
    
    
import pandas as pd

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN


from sklearn import model_selection

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import categorical_crossentropy


graph_labels.value_counts()
graph_labels_test.value_counts()
graph_labels = pd.get_dummies(graph_labels, drop_first=False)
graph_labels_test = pd.get_dummies(graph_labels_test, drop_first=False)

generator = PaddedGraphGenerator(graphs=graphs)
  

k = 35  # the number of rows for the output tensor
layer_sizes = [32, 32, 32, 7]

dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()
    

x_out = Conv1D(filters=8, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=4, kernel_size=3, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=7, activation="softmax")(x_out)
    
    

model = Model(inputs=x_inp, outputs=predictions)

model.compile(
    optimizer=Adam(lr=0.0001), loss=categorical_crossentropy, metrics=["acc"],
) 
    

train_graphs, test_graphs = model_selection.train_test_split(
    graph_labels, train_size=0.8, test_size=None, stratify=graph_labels,
)
    


train_set = PaddedGraphGenerator(graphs=graphs)
test_set = PaddedGraphGenerator(graphs=graphs_test)

random_orden_train = graph_labels.sample(frac = 1)
random_orden_test = graph_labels_test.sample(frac = 1)

train_gen = train_set.flow(
    list(random_orden_train.index ),
    targets=random_orden_train.values,
    batch_size=64,
    symmetric_normalization=False,              
)

test_gen = test_set.flow(
    list(random_orden_test.index ),
    targets=random_orden_test.values,
    batch_size=1,
    symmetric_normalization=False,
)
    
epochs = 600
    
history = model.fit(
    train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
)


sg.utils.plot_history(history)
    

    
    
    
    