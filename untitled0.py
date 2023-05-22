#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:56:54 2023

@author: agustinzhang
"""

import networkx as nx
import matplotlib.pyplot as plt
import random

# 创建节点列表
nodes = list(range(1, 9))

# 创建边列表
edges = [(1, 2), (2, 3), (3, 4), (4, 8), (5, 6), (6, 7), (7, 8), (8, 1), (1, 7)]

# 创建有向图
directed_graph = nx.DiGraph()
directed_graph.add_nodes_from(nodes)
directed_graph.add_edges_from(edges)

# 创建无向图
undirected_graph = nx.Graph()
undirected_graph.add_nodes_from(nodes)
undirected_graph.add_edges_from(edges)

# 绘制有向图
plt.subplot(1, 2, 1)
nx.draw_networkx(directed_graph, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
plt.title("Directed Graph")

# 绘制无向图
plt.subplot(1, 2, 2)
nx.draw_networkx(undirected_graph, with_labels=True, node_color='lightgreen', edge_color='gray')
plt.title("Undirected Graph")

# 显示图形
plt.tight_layout()
plt.show()


import networkx as nx
import matplotlib.pyplot as plt

# 创建节点列表
nodes = [1, 2, 3, 4]

# 创建边列表
edges = [(1, 2), (1, 3), (1, 4)]

# 创建无向图
undirected_graph = nx.Graph()
undirected_graph.add_nodes_from(nodes)
undirected_graph.add_edges_from(edges)

# 创建标签字典，用于标注节点和边
node_labels = {1: "Node 1", 2: "Node 2", 3: "Node 3", 4: "Node 4"}
edge_labels = {(1, 2): "Edge (1, 2)", (1, 3): "Edge (1, 3)", (1, 4): "Edge (1, 4)"}

# 绘制无向图
plt.subplot(1, 1, 1)
pos = nx.spring_layout(undirected_graph)
nx.draw_networkx(undirected_graph, pos, with_labels=False, node_color='lightgreen', edge_color='gray')
nx.draw_networkx_labels(undirected_graph, pos, labels=node_labels, font_color='black', verticalalignment='bottom')
nx.draw_networkx_edge_labels(undirected_graph, pos, edge_labels=edge_labels, font_color='red')
plt.title("Undirected Graph with Node Labels")

# 显示图形
plt.tight_layout()
plt.show()

