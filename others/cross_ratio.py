#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:24:25 2022

@author: agustinzhang
"""


#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math

#df1 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion/IMG_2968.txt', sep = '\t', skiprows = 2)
df1 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Dato_prueba/IMG_2922.txt', sep = '\t', skiprows = 2)
df2 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Dato_prueba/180MUA_Formatted_F.txt', sep = '\t', header = None)
df2 = df2.dropna(axis = 0, how = 'any')

catlist = []
for n in range(0, 10):
    for i in range(0,8):
        catlist.append(n)
        
df2['5'] = catlist
df2.columns = ['D_AB', 'X', 'Y', 'Z', 'r', 'PT']

df_point = df2[~ df2['D_AB'].str.contains('B')]
df_point = df_point[['D_AB', 'X', 'Y', 'r', 'PT']]



df_trans = df1[['H', 'V', 'R', 'C', 'PT']]

n_marker = 6
pointlist = []
linelist = []
dislist = []


df_trans = df_trans.sort_values(by = 'PT', ascending = True)   ### sorted by date
df_trans = df_trans.reset_index(drop = True)                          ### reset index

pointname = [ 'a', 'b', 'c', 'd']* n_marker
df_trans['Point'] = pointname





class Node:

    def __init__(self, x, y):
        self.x = x
        self.y = y





def get_distance(point1, point2):
    return math.sqrt((point2[0]- point1[0])**2 + (point2[1]- point1[1])**2)

def get_area(side1, side2, side3):
    s = (side1+side2+side3)/2
    area = (s*(s-side1)*(s-side2)*(s-side3))**0.5
    return area

def get_affine_invariant(triangle1, triangle2):
    invariant = triangle1/triangle2
    return invariant

    

# for i in range(n_marker):
#     pointlist = df_trans [['H', 'V']][df_trans.PT == i].values.tolist()
#     for j in range(len(pointlist)):
#         for k in range(len(pointlist)):
#             linedis = get_distance(pointlist[j], pointlist[k])
#             if linedis != 0: 
#                 PT = i
#                 dat = [linedis, PT]
#                 linelist.append(dat)
#                 df_line = pd.DataFrame()    
#                 df_line[['Dis','PT']] = linelist
#                 df_line = df_line.drop_duplicates()
#                 line_unique = df_line.values.tolist()

p1list = []
p2list = []
linename = []
box = []
for i in range(n_marker):#for each PT 按照标签
    pointlist = df_trans [['H', 'V', 'Point']][df_trans.PT == i].values.tolist()#list to save the points under the same PT 获取各个标签下的坐标
    for j in range(len(pointlist)):
        b = [j]
        box.append(b) #Create a box to avoid calculate repeated line 每次算完一次后舍弃点，放入box内
        for k in range(len(pointlist)):
            if [k] not in box:
                linedis = get_distance(pointlist[j], pointlist[k])
                point1 = pointlist[j][2]
                point2 = pointlist[k][2]
                line12 = point1+point2
                PT = i
                dat = [linedis, PT]
                p1list.append(point1)
                p2list.append(point2)
                linename.append(line12)

                linelist.append(dat)
                
    box = []


df_line = pd.DataFrame()
df_line[['Dis', 'PT']] = linelist
df_line['P1']= p1list
df_line['P2']= p2list
df_line['Line'] = linename

invariant = []

for i in range(n_marker):
    sidelist = df_line[['Dis', 'P1', 'P2', 'Line']][df_line.PT == i].values.tolist()
    df_side = df_line[['Dis', 'P1', 'P2', 'Line']][df_line.PT == i]
    for j in range(len(sidelist)):
        pointbox = ['a', 'b', 'c', 'd']
        side1 = sidelist[j][3]
        side_dis_1 = sidelist[j][0]
        side_dis_2 = side_dis_1   #the same side of triangle 三角形同边
        sidepoint = [sidelist[j][1], sidelist[j][2]]
        complementaryset = list(set(pointbox)- set(sidepoint))
        extraside1 = complementaryset[0] +complementaryset[1]
        extraside2 = complementaryset[1]+ complementaryset[0]
        df_side_left = df_side[df_side.Line != extraside1]
        df_side_left = df_side_left[ df_side_left.Line != extraside2]
        triangle1 = df_side_left[df_side_left.Line.str.contains(complementaryset[0])].values.tolist()
        triangle2 = df_side_left[df_side_left.Line.str.contains(complementaryset[1])].values.tolist()
        
        side_dis_1_2 = triangle1[0][0]
        side_dis_1_3 = triangle1[1][0]
        side_dis_2_2 = triangle2[0][0]
        side_dis_2_3 = triangle2[1][0]
        
        
        area_triangle1 = get_area(side_dis_1 , side_dis_1_2, side_dis_1_3)
        area_triangle2 = get_area(side_dis_2, side_dis_2_2, side_dis_2_3)
        
        cross_ratio = get_affine_invariant(area_triangle1, area_triangle2)
        cross_ratio2 = get_affine_invariant(area_triangle2, area_triangle1)

        PT = i
        dat = [cross_ratio, PT]
        dat2 = [cross_ratio2, PT]
        
        invariant.append(dat)
        invariant.append(dat2)
        
df_invariant = pd.DataFrame()   
df_invariant[['Cross_ratio', 'PT']] = invariant 









#%%

# =============================================================================
# #读取测试数据并获得所有Feature Point与最近n个点中的m个点所能组成4个点的组合的所有情况
# =============================================================================

from sklearn.neighbors import NearestNeighbors
from itertools import combinations, permutations
df_test = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion/IMG_2926.txt', sep = '\t', skiprows = 2)

df_test_trans = df_test[['H', 'V', 'R', 'C', 'PT']]


#从n个点中选取m个点排列组合
f = 4
m = 5           
n = 6

num_point = len(df_test_trans)

def get_combinations(point_list, number_of_selection):
    l = []
    for i in range(len(point_list)):
        data = list(combinations(point_list[i], number_of_selection))
        l.append(data)
    
    return l

def get_df_combination(combination_list, number_of_selection):
    df = pd.DataFrame()
    for i in range(0, np.array(combination_list).shape[0]):
        df_c = pd.DataFrame()
        df_c['Combination'] = combination_list[i][:]
        
        df= df.append(df_c)
        
    #删除不包含feature point的组合
    df = df.drop(axis = 0, index = number_of_selection)
    n_repeat = len(df)/ num_point
    p = list(np.repeat(range(0,num_point), n_repeat))
    
    df['Point'] = p
    df = df.drop_duplicates()
    df = df.reset_index(drop = True)
    
    return df





feature_point = df_test_trans[['H', 'V']].values.tolist()

    
#获取每个feature point最近的点共n个，距离存在distances中，index存在indices中
nbrs = NearestNeighbors(n_neighbors = n, metric = get_distance).fit(feature_point)
#这n个点之间的距离及其index
distances, indices = nbrs.kneighbors(feature_point)        
        


near_point = []
# for i in range(1, n+1):
    


#     point = feature_point[indices[0,i]]
#     near_point.append(point)

# 提取这n个点

index_point = []
for i in range(len(indices)):
    
    point = list(indices[i,0:n])
    index_point.append(point)


        

# 在n个最近点中取m个点，并排列所有组合
c_index = get_combinations(index_point, m)

#转换为dataframe
df_c_index = get_df_combination(c_index, m)



list_c_index = df_c_index.Combination.values.tolist()

# 从m个点里取f个点排列组合
c_f_index = get_combinations(list_c_index, f)

# 转换为dataframe
df_f_index = get_df_combination(c_f_index, f)

#%%

# =============================================================================
# 获取所有Invariant
# =============================================================================


#获取组合内的四个点的坐标
def get_point_coordinate(data_frame_point_combination, data_frame_feature_points, n):
    index1 = data_frame_point_combination.Combination[n][0]
    index2 = data_frame_point_combination.Combination[n][1]
    index3 = data_frame_point_combination.Combination[n][2]
    index4 = data_frame_point_combination.Combination[n][3]
    point1 = [data_frame_feature_points.H[index1], data_frame_feature_points.V[index1]]
    point2 = [data_frame_feature_points.H[index2], data_frame_feature_points.V[index2]]
    point3 = [data_frame_feature_points.H[index3], data_frame_feature_points.V[index3]]
    point4 = [data_frame_feature_points.H[index4], data_frame_feature_points.V[index4]]
    
    return point1, point2, point3, point4

point1, point2, point3, point4 = get_point_coordinate(df_f_index, df_test_trans, 60)


def get_invariant(A,B,C,D):
    dis_AB = get_distance(A, B)
    dis_AD = get_distance(A, D)
    dis_BD = get_distance(B, D)
    dis_AC = get_distance(A, C)
    dis_BC = get_distance(B, C)
    s_triangle1 = get_area(dis_AB, dis_AD, dis_BD)
    s_triangle2 = get_area(dis_AB, dis_AC, dis_BC)
    invariant = get_affine_invariant(s_triangle1, s_triangle2)
    
    
    return invariant







invariant_test = get_invariant(point1,point2,point3,point4)





df_invariant = df_invariant.round(2)


df_invariant['e1'] = df_invariant.Cross_ratio**10
df_invariant['e2'] = df_invariant.Cross_ratio**10
df_invariant['e3'] = df_invariant.Cross_ratio**10
df_invariant['e4'] = df_invariant.Cross_ratio**10
df_invariant['e5'] = df_invariant.Cross_ratio**10
df_invariant['e6'] = df_invariant.Cross_ratio**10

df_invariant.e1 = df_invariant.e1.round(2)
df_invariant.e2 = df_invariant.e2.round(2)
df_invariant.e3 = df_invariant.e3.round(2)
df_invariant.e4 = df_invariant.e4.round(2)
df_invariant.e5 = df_invariant.e5.round(2)
df_invariant.e6 = df_invariant.e6.round(2)

l = df_invariant.e1.values.tolist()
l2 = df_invariant1.Discrete.values.tolist()

any(l) in l2










