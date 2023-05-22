#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 09:02:58 2022

@author: agustinzhang
"""


import pandas as pd
import numpy as np
import cv2
import math
from itertools import combinations
import os
import random
from sklearn.neighbors import NearestNeighbors
from itertools import combinations

# =============================================================================
# 随机选取一个txt
# =============================================================================
file_path = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion'
file_list = []
for i in os.listdir(file_path):
    file_list.append(os.path.join(file_path, i))

def get_random_file():
    
    index_file = random.randint(0,len(file_list)-1)
    df_random = pd.read_table(file_list[index_file], sep = '\t', skiprows = 2)
    
    return df_random

# =============================================================================
# 将df去除na行，并去除PT负数项，选取重要的列(仅为测试用，实际情况仅需保证坐标无na)
# =============================================================================
def trans_df( df ): 
    df = df.dropna(subset=["PT"])
    df = df[df.PT >= 0]
    df_trans = df[['H', 'V', 'R', 'PT']]
    df_trans['H'] = df_trans.H+2000.5
    df_trans['V'] = df_trans.V+1500.5
    df_trans = df_trans.reset_index(drop = True)

        
    # n_marker = len(df.PT.unique())
        
    # df_trans = df_trans.sort_values(by = 'PT', ascending = True)
    # df_trans = df_trans.reset_index(drop = True)
        
    # pointname = ['a','b','c','d']* n_marker
        
    # df_trans['Point'] = pointname
    return df_trans

# =============================================================================
# 实际用版本trans——df
# =============================================================================
def trans_df_real( df ): 
    df = df.dropna(subset=['V','H'])
    df_trans = df[['H', 'V', 'R', 'PT']]
    df_trans['H'] = df_trans.H+2000.5
    df_trans['V'] = df_trans.V+1500.5
    df_trans = df_trans.reset_index(drop = True)

    return df_trans
# =============================================================================
# #读取测试数据并获得所有Feature Point与最近n个点所能组成f个点的组合的所有情况
# =============================================================================

f = 4
# m = 5
n = 5

def get_distance(point1, point2):
    return math.sqrt((point2[0]- point1[0])**2 + (point2[1]- point1[1])**2)


def get_closest_n_point(df_trans, n):
    feature_point = df_trans[['H', 'V']].values.tolist()
    
    #获取每个feature point最近的点共n个，距离存在distances中，index存在indices中
    nbrs = NearestNeighbors(n_neighbors = n, metric = get_distance).fit(feature_point)
#这n个点之间的距离及其index
    distances, indices = nbrs.kneighbors(feature_point)  
    return distances, indices

def get_combination(indices, f):
    feature_point = list(indices[:,0])
    rest_point = []
    for i in range(len(indices)):
        point = list(indices[i, 1:])
        rest_point.append(point)
    
    l_combination = []
    for j in range(len(rest_point)):
        dat = list(combinations(rest_point[j], f-1))
        l_combination.append(dat)
    


    dim_list = np.array(l_combination).shape[1]
    list_combination = []
    for k in range(len(feature_point)):
        for l in range(dim_list):
            ll = list(l_combination[k][l])
            ll.append(feature_point[k])
            ll.sort()
            list_combination.append(ll)
            
    #消除重复组合
    unique_list = []
    for element in list_combination:
        if element not in unique_list:
            unique_list.append(element)
        
        

    return unique_list
            
    




# =============================================================================
# 
# =============================================================================
def get_line(df):
    p1list = []
    p2list = []
    linename = []
    linelist = []
    box = []
    
    pointlist = df[['Corrdinate', 'Point']].values.tolist()
    for i in range(len(pointlist)):
        b = [i]
        box.append(b)
        for j in range(len(pointlist)):
            if [j] not in box:
                linedis = get_distance(pointlist[i][0], pointlist[j][0])
                point1 = pointlist[i][1]
                point2 = pointlist[j][1]
                line12 = point1+point2
                p1list.append(point1)
                p2list.append(point2)
                linename.append(line12)
                linelist.append(linedis)
        box = []
        
    df_line = pd.DataFrame()
    df_line['Dis'] = linelist
    df_line['P1'] = p1list
    df_line['P2'] = p2list
    df_line['Line'] = linename
    df_line = df_line.drop_duplicates(subset = 'Dis')
    return df_line

def get_area(side1, side2, side3):
    s = (side1+side2+side3)/2
    area = (s*(s-side1)*(s-side2)*(s-side3))**0.5
    return area

def get_affine_invariant(list_combination_unique, decimal):
    affine_invariant = []
    total_point = []
    total_commonside = []
    total_index = []
    for i in range(len(list_combination_unique)):
        tri1 = []
        tri2 = []
        common_side = []
        point = []
        invariant = []
        #获取四个点所在index
        a_index = list_combination_unique[i][0]
        b_index = list_combination_unique[i][1]
        c_index = list_combination_unique[i][2]
        d_index = list_combination_unique[i][3]
        l_index = [a_index, b_index, c_index, d_index]
        #获取四个点的坐标
        a = [df_trans.H[a_index], df_trans.V[a_index]]
        b = [df_trans.H[b_index], df_trans.V[b_index]]
        c = [df_trans.H[c_index], df_trans.V[c_index]]
        d = [df_trans.H[d_index], df_trans.V[d_index]]
        l = [a,b,c,d]
        l_p = ['a', 'b','c','d']
        df_point = pd.DataFrame()
        df_point['Corrdinate'] = l
        df_point['Point'] = l_p
        df_point['index'] = l_index
        
        df_line = get_line(df_point)
        df_line = df_line.reset_index(drop = True)
        sidelist = df_line.values.tolist()
        for j in range(len(sidelist)):
            pointbox = ['a','b','c','d']
            sidecommon = sidelist[j][3]
            side_dis_common = sidelist[j][0]
            common_point = [sidelist[j][1], sidelist[j][2]]
            complementaryset = list(set(pointbox)- set(common_point))
            extraside1 = complementaryset[0]+ complementaryset[1]
            extraside2 = complementaryset[1]+ complementaryset[0]
            df_side_left = df_line[df_line.Line != extraside1]
            df_side_left = df_side_left[df_side_left.Line != extraside2]
            triangle1 = df_side_left[df_side_left.Line.str.contains(complementaryset[0])].values.tolist()
            triangle2 = df_side_left[df_side_left.Line.str.contains(complementaryset[1])].values.tolist()
            
            side_dis_1_2 = triangle1[0][0]
            side_dis_1_3 = triangle1[1][0]
            side_dis_2_2 = triangle2[0][0]
            side_dis_2_3 = triangle2[1][0]           
            
            area_triangle1 = get_area(side_dis_common , side_dis_1_2, side_dis_1_3)
            area_triangle2 = get_area(side_dis_common, side_dis_2_2, side_dis_2_3)
            
            cross_ratio = area_triangle1/ area_triangle2
            cross_ratio2 = area_triangle2/ area_triangle1


            tri1.append(triangle1)
            tri1.append(triangle2)
        
            tri2.append(triangle2)
            tri2.append(triangle1)
            
            dat = cross_ratio
            dat2 = cross_ratio2
            
            dat3 = [triangle1[0][1], triangle1[0][2], 
                   triangle1[1][1], triangle1[1][2],
                   triangle2[0][1], triangle2[0][2],
                   triangle2[1][1], triangle2[1][2]]
            
            dat4 = [triangle2[0][1], triangle2[0][2], 
                   triangle2[1][1], triangle2[1][2],
                   triangle1[0][1], triangle1[0][2],
                   triangle1[1][1], triangle1[1][2]]
            
            point.append(dat3)
            point.append(dat4)
            common_side.append(sidecommon)
            common_side.append(sidecommon)
            invariant.append(dat)
            invariant.append(dat2)
            
        affine_invariant.append(invariant)
        total_commonside.append(common_side)
        total_point.append(point)
        total_index.append(l_index)
    df_invariant = pd.DataFrame()
    df_invariant[['Value1', 'Value2', 'Value3', 'Value4',
                  'Value5', 'Value6', 'Value7', 'Value8',
                  'Value9', 'Value10', 'Value11', 'Value12']] =affine_invariant
    
    df_invariant['Sum'] = df_invariant.apply(lambda x: x.sum(), axis = 1)
    df_invariant['Min'] = df_invariant.apply(lambda x: x.min(), axis = 1)
    df_invariant['Max'] = df_invariant.iloc[:,:12].apply(lambda x: x.max(), axis = 1)
    df_invariant.Sum = df_invariant.Sum.round(decimal)
    df_invariant.Max = df_invariant.Max.round(decimal)
    df_invariant.Min = df_invariant.Min.round(decimal)
    df_invariant['Combination'] = total_index
    
    # df_invariant[['Com1', 'Com2', 'Com3',
    #               'Com4', 'Com5', 'Com6',
    #               'Com6', 'Com8', 'Com9',
    #               'Com10', 'Com11', 'Com12']] = total_point
    # df_invariant['Common_side'] = total_commonside
    
    return df_invariant
        




# =============================================================================
# 自闭ing
# =============================================================================

df_random = get_random_file()
df_trans = trans_df(df_random)
df_trans_real = trans_df_real(df_random)


#这最近n个点之间的距离及其index
distances, indices = get_closest_n_point(df_trans, n)
    

list_combination_unique = get_combination(indices, f)

df_invariant = get_affine_invariant(list_combination_unique, 2)


































