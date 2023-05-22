#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:10:32 2022

@author: agustinzhang
"""
import matplotlib.pyplot as plt

from data_real import get_real_data
import pandas as pd
import numpy as np
import math
from itertools import combinations
import os
import random
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from ref import get_reference
import cv2





d = 2
n = 5    
path_ref = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Dato_prueba/180MUA_Formatted_F.txt' 
file_path = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion'
ref = get_reference(path_ref, d)
df_feature_ref, df_points_ref = ref.get_invariant_ref()
df_trans_ref = ref.trans_df()
df = get_real_data(file_path,df_feature_ref, d, n)
list_df_trans = df.trans_df_data()
list_chosen_point, list_features = df.main_process()


i = 0
list_point = list_chosen_point[i]
features = list_features[i]
df_trans = list_df_trans[i]
names = locals()
tri1 = []
tri2 = []
common_side = []
point = []
invariant = []
    ##构成每个PT的点（第一个就是PT0，最后一个是PT5）
    # Points that make up each PT (the first PT is PT0, and the last is PT5)
for j in range(6):

        
    index_feature = list_point[j]
    points = features.Combination[index_feature].values.tolist()
    point_a = points[0][0]
    point_b = points[0][1]
    point_c = points[0][2]
    point_d = points[0][3]
    l_index = [point_a, point_b, point_c, point_d]
    a = [df_trans.H[point_a], df_trans.V[point_a]]
    b = [df_trans.H[point_b], df_trans.V[point_b]]
    c = [df_trans.H[point_c], df_trans.V[point_c]]
    d = [df_trans.H[point_d], df_trans.V[point_d]]
    l = [a , b, c, d]
    
    ax = df_trans.H[point_a]
    bx = df_trans.H[point_b]
    cx = df_trans.H[point_c]
    dx = df_trans.H[point_d]
    lx = [ax,bx,cx,dx]
    
    ay = df_trans.V[point_a]
    by = df_trans.V[point_b]
    cy = df_trans.V[point_c]
    dy = df_trans.V[point_d]
    ly = [ay, by, cy, dy]
    l_p = ['a', 'b', 'c', 'd']
    names['df_point_%s'%j] = pd.DataFrame()           
    names['df_point_%s'%j]['X'] = lx
    names['df_point_%s'%j]['Y'] = ly
    names['df_point_%s'%j]['Point'] = l_p
    names['df_point_%s'%j]['index'] = l_index
    names['df_point_%s'%j]['Real_point'] = np.nan
    names['df_point_%s'%j]['PT'] = [j]*4
    names['df_point_%s'%j]['Corrdinate'] = l
    df_line = df.get_df_line_data(names['df_point_%s'%j])
    df_line = df_line.reset_index(drop = True)
    
    sidelist = df_line.values.tolist()
    for k in range(len(sidelist)):
        pointbox = ['a', 'b', 'c', 'd']
        sidecommon = sidelist[k][3]
        side_dis_common = sidelist[k][0]
        common_point = [sidelist[k][1], sidelist[k][2]]
        complementaryset = list(set(pointbox)- set(common_point))
        extraside1 = complementaryset[0]+ complementaryset[1]
        extraside2 = complementaryset[1]+ complementaryset[0]
        sidepoint = [sidelist[k][1], sidelist[k][2]]
        df_side_left = df_line[df_line.Line != extraside1]
        df_side_left = df_side_left[df_side_left.Line != extraside2]
        triangle1 = df_side_left[df_side_left.Line.str.contains(complementaryset[0])].values.tolist()
        triangle2 = df_side_left[df_side_left.Line.str.contains(complementaryset[1])].values.tolist()
        
            
        side_dis_1_2 = triangle1[0][0]
        side_dis_1_3 = triangle1[1][0]
        side_dis_2_2 = triangle2[0][0]
        side_dis_2_3 = triangle2[1][0]     
        
        area_triangle1 = df.get_area(side_dis_common , side_dis_1_2, side_dis_1_3)
        area_triangle2 = df.get_area(side_dis_common, side_dis_2_2, side_dis_2_3)
            
        cross_ratio = area_triangle1/ area_triangle2
        cross_ratio2 = area_triangle2/ area_triangle1
        
        PT = j
        dat = [cross_ratio, PT]
        dat2 = [cross_ratio2, PT]
        
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
        
    df_PT = pd.DataFrame()       
    df_PT[['Invariant','PT']] = invariant
    df_PT[['Point_1', 'Point_2','Point_3', 'Point_4', 'Point_5', 'Point_6', 'Point_7', 'Point_8']] = point
         
    df_PT['Common_side'] = common_side
    df_PT = df_PT.groupby('PT').apply(lambda x : x.sort_values('Invariant', ascending = True))   
    df_PT = df_PT.reset_index(drop = True)
    
    
    
for i in range(6):
    #从ref中选取最小值的一行，得出大三角形的顶点和小三角形的顶点 (tri1为较小三角形)
    # Select the min value of the reference, get the vertexs of the two triangle(triangle1 is the samller triangle)
    df = df_points_ref[df_points_ref.PT == i]
    #index = 0 because in df_PT its already ordened by Invariant value
    tri_1_ref = df.iloc[0:1, 2:6 ].values.tolist()[0]

    tri_2_ref = df.iloc[0:1, 6:10].values.tolist()[0]

    tri_1_count_ref = pd.DataFrame()
    tri_1_count_ref['Count'] = pd.value_counts(tri_1_ref)
    
    tri_2_count_ref = pd.DataFrame()
    tri_2_count_ref['Count'] = pd.value_counts(tri_2_ref)
    
    index_tri_1_ref = tri_1_count_ref[tri_1_count_ref.Count == 2].index.tolist()[0]
    index_tri_2_ref = tri_2_count_ref[tri_2_count_ref.Count == 2].index.tolist()[0]
    
    #the same with the test data
    df2 = df_PT[df_PT.PT == i]
    
    tri_1 = df2.iloc[0:1, 2:6].values.tolist()[0]
    tri_2 = df2.iloc[0:1, 6:10].values.tolist()[0]

    tri_1_count = pd.DataFrame()
    tri_1_count['Count'] = pd.value_counts(tri_1)
    
    tri_2_count = pd.DataFrame()
    tri_2_count['Count'] = pd.value_counts(tri_2)
    
    index_tri_1 = tri_1_count[tri_1_count.Count == 2].index.tolist()[0]
    index_tri_2 = tri_2_count[tri_2_count.Count == 2].index.tolist()[0]
    
    new_common_side1_ref = index_tri_1_ref+index_tri_2_ref
    new_common_side2_ref = index_tri_2_ref+ index_tri_1_ref
    
    new_common_side1 = index_tri_1+index_tri_2
    new_common_side2 = index_tri_2+ index_tri_1
# =============================================================================
# new ref 
# =============================================================================
    #由于不知道公共边的字母顺序，因此建立两个条件如公共边等于ad或da,并选取ratio小于1的 (ratio <1, triangle1 > triangle2)
    #Since we dont know the orden of the letters of common side, we make two conditions:
    # 1. ratio < 1, this mean area of triangle1 is bigger than the area of triangle2
    # 2. if the common side is ad or da
    df_ref_new = df_points_ref[((df_points_ref.Common_side == new_common_side1_ref)|(df_points_ref.Common_side == new_common_side2_ref)) & (df_points_ref.PT == i)]
    df_ref_new = df_ref_new[df_ref_new.Cross_ratio < 1]
    
    #triangle 1 (smaller)
    tri_3_ref = df_ref_new.iloc[0:1, 2:6 ].values.tolist()[0]
    #triangle 2 (larger)
    tri_4_ref = df_ref_new.iloc[0:1, 6:10].values.tolist()[0]
    
    
    tri_3_count_ref = pd.DataFrame()
    tri_3_count_ref['Count'] = pd.value_counts(tri_3_ref)
    
    tri_4_count_ref = pd.DataFrame()
    tri_4_count_ref['Count'] = pd.value_counts(tri_4_ref)
            
    index_tri_3_ref = tri_3_count_ref[tri_3_count_ref.Count == 2].index.tolist()[0]
    index_tri_4_ref = tri_4_count_ref[tri_4_count_ref.Count == 2].index.tolist()[0]
    
    
# =============================================================================
# new data, match the letters with reference
# =============================================================================
    df2_new = df_PT[((df_PT.Common_side == new_common_side1)|(df_PT.Common_side == new_common_side2)) & (df_PT.PT == i)]
    df2_new = df2_new[df2_new.Invariant < 1]
    
     #triangle 1 (smaller)
    tri_3 = df2_new.iloc[0:1, 2:6 ].values.tolist()[0]
    #triangle 2 (larger)
    tri_4 = df2_new.iloc[0:1, 6:10].values.tolist()[0]
    
    
    tri_3_count = pd.DataFrame()
    tri_3_count['Count'] = pd.value_counts(tri_3)
    
    tri_4_count = pd.DataFrame()
    tri_4_count['Count'] = pd.value_counts(tri_4)
            
    index_tri_3 = tri_3_count[tri_3_count.Count == 2].index.tolist()[0]
    index_tri_4 = tri_4_count[tri_4_count.Count == 2].index.tolist()[0]
    
    
    names['df_point_%s'%i]['Real_point'][names['df_point_%s'%i].Point == index_tri_1 ] = index_tri_1_ref
    names['df_point_%s'%i]['Real_point'][names['df_point_%s'%i].Point == index_tri_2 ] = index_tri_2_ref
    names['df_point_%s'%i]['Real_point'][names['df_point_%s'%i].Point == index_tri_3 ] = index_tri_3_ref
    names['df_point_%s'%i]['Real_point'][names['df_point_%s'%i].Point == index_tri_4 ] = index_tri_4_ref
        
        
        
        
        
        
        
        
        






# =============================================================================
# graphic
# =============================================================================
        
        


    

newImg = np.zeros((3001,4001,3), np.uint8)
newImg.fill(0)
cv2.imshow('newimg', newImg)





for i in range(6):
    listpoint = names['df_point_%s'%i][['X', 'Y']].values.tolist()
    for (x, y) in listpoint:
        r = 24
        cv2.circle(newImg, (int(x),int(y)), int(r), (i*40,255-i*40,i*45), 3)
        cv2.putText(newImg, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (i*40,255-i*40,i*45), 3)
        for j in range(4):
            cv2.putText(newImg, str(names['df_point_%s'%i]['Real_point'][j]), 
                        (int(names['df_point_%s'%i]['X'][j]), int(names['df_point_%s'%i]['Y'][j]+500)), 
                        cv2.FONT_HERSHEY_COMPLEX, 2, (i*40,255-i*40,i*45), 3)

# cv2.line(newImg, (0,1501), (4001,1501), (255,255,255), 2)
# cv2.line(newImg, (2001,0), (2001,3001), (255,255,255), 2)
cv2.imshow('newimg', newImg)












df_trans_ref_graph = df_trans_ref
df_trans_ref_graph['X'] = df_trans_ref_graph.X - 2000 
df_trans_ref_graph['Y'] = df_trans_ref_graph.Y - 1500





for i in range(6):
    plt.subplot(2, 3, i+1)
    listpoint = df_trans_ref_graph[['X', 'Y', 'r','Point']][df_trans_ref_graph.PT == i].values.tolist()
    for (x, y, r, p) in listpoint:
        plt.plot(x, y, 'o', ms = 5)
        plt.annotate(p, xy= (x,y), xytext = (x+0.1,y+0.1))
        
    plt.grid()
    x_major_locator = plt.MultipleLocator(1)
    y_major_locator = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    





















        
        
    
