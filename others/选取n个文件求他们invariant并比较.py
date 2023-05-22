#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 08:53:35 2022

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
import matplotlib.pyplot as plt
import seaborn as sns

def get_n_file(n):
    file_path = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion'
    file_list = []
    for i in os.listdir(file_path):
        file_list.append(os.path.join(file_path, i))
    
    index_file = random.sample(range(len(file_list)), n)   
    
    names = locals()
    list_df = []
    for j in range(n):
        names['df_%s'%j] = pd.read_table(file_list[index_file[j]], sep = '\t', skiprows = 2)
        list_df.append(names['df_%s'%j])
   
    return list_df


# =============================================================================
# 将df去除na行，并去除PT负数项，选取重要的列(仅为测试用，实际情况仅需保证坐标无na)
# =============================================================================
def trans_df( list_df ): 
    list_df_trans = []
    names = locals()
    for i in range(len(list_df)):
        df = list_df[i]
        
    
        df = df.dropna(subset=["PT"])
        df = df[df.PT >= 0]
        n_marker = len(df.PT.unique())
        df_trans = df[['H', 'V', 'R', 'PT']]
        df_trans['H'] = df_trans.H+2000.5
        df_trans['V'] = df_trans.V+1500.5
        df_trans = df_trans.sort_values(by = 'PT', ascending = True)
        df_trans = df_trans.reset_index(drop = True)
        
        PT_unique = np.sort(df.PT.unique())##按照每个PT点的个数标abcd
        num = df.PT.value_counts()
        l_p = ['a', 'b', 'c', 'd']
        pointname = []
        for j in range(len(PT_unique)):
            point = l_p[:num[PT_unique[j]]]
            pointname.extend(point)
        
        
        df_trans['Point'] = pointname
        
        if len(df_trans) != 0:
            if len(df_trans) % 4 ==0:
                names['df_trans_%s'%i] = df_trans
        
        
                list_df_trans.append(names['df_trans_%s'%i])

        
    # n_marker = len(df.PT.unique())
        
    # df_trans = df_trans.sort_values(by = 'PT', ascending = True)
    # df_trans = df_trans.reset_index(drop = True)
        
    # pointname = ['a','b','c','d']* n_marker
        
    # df_trans['Point'] = pointname
    return list_df_trans

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
# 获取点间距离， 面积
# =============================================================================

def get_distance(point1, point2):
    return math.sqrt((point2[0]- point1[0])**2 + (point2[1]- point1[1])**2)

def get_area(side1, side2, side3):
    s = (side1+side2+side3)/2
    area = (s*(s-side1)*(s-side2)*(s-side3))**0.5
    return area
# =============================================================================
# 获得每个边
# =============================================================================

def get_df_line(list_df_trans):
    list_df_lines = []
    names = locals()
    for k in range(len(list_df_trans)):
        df_trans = list_df_trans[k]
        n_marker = np.sort(df_trans.PT.unique())
        p1list = []
        p2list = []
        linename = []
        linelist = []
        box = []

        for i in n_marker:
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
                        PT_1 = i
                        dat = [linedis, PT_1]
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
        names['df_line_%s'%k] = df_line
        
        list_df_lines.append(df_line)
    
    return list_df_lines

def get_affine_invariant(triangle1, triangle2):
    invariant = triangle1/triangle2
    return invariant

# =============================================================================
# 获取每个df的invariant
# =============================================================================
def get_invariant(list_df_lines, d, z):
    names = locals()
    list_df_invariant = []
    for m in range(len(list_df_lines)):
        invariant = []
        tri1 = []
        tri2 = []
        common_side = []
        point = []
        df_line = list_df_lines[m]
        df_trans = list_df_trans[m]
        n_marker = np.sort(df_trans.PT.unique())
        for i in n_marker:
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
        
                cross_ratio = round(get_affine_invariant(area_triangle1, area_triangle2), d) **z
                cross_ratio2 = round(get_affine_invariant(area_triangle2, area_triangle1), d) **z
        
                tri1.append(triangle1)
                tri1.append(triangle2)
        
                tri2.append(triangle2)
                tri2.append(triangle1)
                

                PT_2 = i
                dat = [cross_ratio, PT_2] 
                   
                dat2 = [cross_ratio2, PT_2]
                   
    
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
                common_side.append(sidepoint)
                common_side.append(sidepoint)
                invariant.append(dat)
                invariant.append(dat2)
        
            df_invariant = pd.DataFrame()   
            df_invariant[['Cross_ratio', 'PT']] = invariant 
            df_invariant = df_invariant.round(d)
            df_invariant[['Point_1', 'Point_2','Point_3', 'Point_4', 'Point_5', 'Point_6', 'Point_7', 'Point_8']] = point
         
            df_invariant['Common_side'] = common_side

            df_invariant = df_invariant.groupby('PT').apply(lambda x : x.sort_values('Cross_ratio', ascending = True))
            df_invariant = df_invariant.reset_index(drop = True)
            
        names['df_invariant_%s'%m] = df_invariant
        list_df_invariant.append(names['df_invariant_%s'%m])
        
    
    
    return list_df_invariant

    
# =============================================================================
# 获取最大值，最小值，求和
# =============================================================================
def get_max(list_invariant):
    df_max = pd.DataFrame()
    for i in range(len(list_invariant)):
        df_invariant = list_invariant[i]
        maxv = df_invariant.groupby('PT').max('Cross_ratio')
        if i == 0:
            df_max = maxv
            
        else:
            df_max = df_max.merge(maxv, how = 'outer', left_index= True, right_index = True)
        
    return df_max

        
def get_min(list_invariant):
    df_min = pd.DataFrame()
    for i in range(len(list_invariant)):
        df_invariant = list_invariant[i]
        minv = df_invariant.groupby('PT').min('Cross_ratio')
        if i == 0:
            df_min = minv
            
        else:
            df_min = df_min.merge(minv, how = 'outer', left_index= True, right_index = True)
        
    return df_min

def get_sum(list_invariant):
    df_sum = pd.DataFrame()
    for i in range(len(list_invariant)):
        df_invariant = list_invariant[i]
        sumv = df_invariant.groupby('PT').sum('Cross_ratio')
        if i == 0:
            df_sum = sumv
            
        else:
            df_sum = df_sum.merge(sumv, how = 'outer', left_index= True, right_index = True)
        
    return df_sum






# =============================================================================
# 自闭ing
# =============================================================================

list_df = get_n_file(46)
list_df_trans = trans_df(list_df)

z = 1

list_df_lines = get_df_line(list_df_trans)
list_invariant = get_invariant(list_df_lines, 2, z)
df_max = get_max(list_invariant)
df_min = get_min(list_invariant)
df_sum = get_sum(list_invariant)

# =============================================================================
# 自闭画图
# =============================================================================



df_max = df_max.T
df_max = df_max.reset_index(drop = True)



df_min = df_min.T
df_min = df_min.reset_index(drop = True)


df_sum = df_sum.T
df_sum = df_sum.reset_index(drop = True)






df_Max = pd.DataFrame(columns = ['PT0', 'PT1', 'PT2', 'PT3', 'PT4', 'PT5']) 
df_Max[['PT0', 'PT1', 'PT2', 'PT3', 'PT4', 'PT5']] = df_max

df_Min = pd.DataFrame(columns = ['PT0', 'PT1', 'PT2', 'PT3', 'PT4', 'PT5']) 
df_Min[['PT0', 'PT1', 'PT2', 'PT3', 'PT4', 'PT5']] = df_min

df_Sum = pd.DataFrame(columns = ['PT0', 'PT1', 'PT2', 'PT3', 'PT4', 'PT5']) 
df_Sum[['PT0', 'PT1', 'PT2', 'PT3', 'PT4', 'PT5']] = df_sum

# #sns.swarmplot( data = df_Max)

# fig1, ax1 = plt.subplots(2, 3, constrained_layout = True)

# sns.countplot(x = 'PT0',  data = df_Max, ax = ax1[0][0])
# sns.countplot(x = 'PT1',  data = df_Max, ax = ax1[0][1])
# sns.countplot(x = 'PT2',  data = df_Max, ax = ax1[0][2])
# sns.countplot(x = 'PT3',  data = df_Max, ax = ax1[1][0])
# sns.countplot(x = 'PT4',  data = df_Max, ax = ax1[1][1])
# sns.countplot(x = 'PT5',  data = df_Max, ax = ax1[1][2])






# fig2, ax2 = plt.subplots(2, 3, constrained_layout = True)

# sns.countplot(x = 'PT0',  data = df_Min, ax = ax2[0][0])
# sns.countplot(x = 'PT1',  data = df_Min, ax = ax2[0][1])
# sns.countplot(x = 'PT2',  data = df_Min, ax = ax2[0][2])
# sns.countplot(x = 'PT3',  data = df_Min, ax = ax2[1][0])
# sns.countplot(x = 'PT4',  data = df_Min, ax = ax2[1][1])
# sns.countplot(x = 'PT5',  data = df_Min, ax = ax2[1][2])


# fig3, ax3 = plt.subplots(2, 3, constrained_layout = True)

# sns.countplot(x = 'PT0',  data = df_Sum, ax = ax3[0][0])
# sns.countplot(x = 'PT1',  data = df_Sum, ax = ax3[0][1])
# sns.countplot(x = 'PT2',  data = df_Sum, ax = ax3[0][2])
# sns.countplot(x = 'PT3',  data = df_Sum, ax = ax3[1][0])
# sns.countplot(x = 'PT4',  data = df_Sum, ax = ax3[1][1])
# sns.countplot(x = 'PT5',  data = df_Sum, ax = ax3[1][2])


# =============================================================================
# 中位数，平均数
# =============================================================================
names = locals()



def get_mean(df):
    names = locals()
    media = []
    for i in range(6):
        col = 'PT'+str(i)
        value = df[col].mean()
        PT = i
        dat = [value, PT]
        media.append(dat)
    
    df_mean = pd.DataFrame()
    df_mean[['Value', 'PT']] = media
        
    return df_mean
        

def get_median(df):
    names = locals()
    mediana = []
    for i in range(6):
        col = 'PT'+str(i)
        value = df[col].median()
        PT = i
        dat = [value, PT]
        mediana.append(dat)
    df_median = pd.DataFrame()
    df_median[['Value', 'PT']] = mediana    
    
    return df_median
        





df_mean_max = get_mean(df_Max)
df_mean_min = get_mean(df_Min)
df_mean_sum = get_mean(df_Sum)


df_median_max = get_median(df_Max)
df_median_min = get_median(df_Min)
df_median_sum = get_median(df_Sum)

df_median_max = df_median_max.rename(columns = {'Value': "Max"})
df_median_min = df_median_min.rename(columns = {'Value': "Min"})
df_median_sum = df_median_sum.rename(columns = {'Value': "Sum"})

df_feature = pd.concat([df_median_max, df_median_min, df_median_sum], axis = 1)
df_feature_ref = df_feature.T.drop_duplicates().T



















