#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:25:05 2022

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


#%%
# =============================================================================
# =============================================================================
# =============================================================================
# # # 
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# 求reference
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # # # =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
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
        
                cross_ratio = round(get_affine_invariant(area_triangle1, area_triangle2), d) 
                cross_ratio2 = round(get_affine_invariant(area_triangle2, area_triangle1), d) 
        
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

#sns.swarmplot( data = df_Max)

fig1, ax1 = plt.subplots(2, 3, constrained_layout = True)

sns.countplot(x = 'PT0',  data = df_Max, ax = ax1[0][0])
sns.countplot(x = 'PT1',  data = df_Max, ax = ax1[0][1])
sns.countplot(x = 'PT2',  data = df_Max, ax = ax1[0][2])
sns.countplot(x = 'PT3',  data = df_Max, ax = ax1[1][0])
sns.countplot(x = 'PT4',  data = df_Max, ax = ax1[1][1])
sns.countplot(x = 'PT5',  data = df_Max, ax = ax1[1][2])






fig2, ax2 = plt.subplots(2, 3, constrained_layout = True)

sns.countplot(x = 'PT0',  data = df_Min, ax = ax2[0][0])
sns.countplot(x = 'PT1',  data = df_Min, ax = ax2[0][1])
sns.countplot(x = 'PT2',  data = df_Min, ax = ax2[0][2])
sns.countplot(x = 'PT3',  data = df_Min, ax = ax2[1][0])
sns.countplot(x = 'PT4',  data = df_Min, ax = ax2[1][1])
sns.countplot(x = 'PT5',  data = df_Min, ax = ax2[1][2])


fig3, ax3 = plt.subplots(2, 3, constrained_layout = True)

sns.countplot(x = 'PT0',  data = df_Sum, ax = ax3[0][0])
sns.countplot(x = 'PT1',  data = df_Sum, ax = ax3[0][1])
sns.countplot(x = 'PT2',  data = df_Sum, ax = ax3[0][2])
sns.countplot(x = 'PT3',  data = df_Sum, ax = ax3[1][0])
sns.countplot(x = 'PT4',  data = df_Sum, ax = ax3[1][1])
sns.countplot(x = 'PT5',  data = df_Sum, ax = ax3[1][2])


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




#%%

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



def trans_df( df ): 
    df = df.dropna(subset=["PT"])
    df = df[df.PT >= 0]
    df_trans = df[['H', 'V', 'R', 'PT']]
    df_trans['H'] = df_trans.H+2000.5
    df_trans['V'] = df_trans.V+1500.5
    if len(df_trans) != 0:
        if len(df_trans) % 4 ==0:            
            df_trans = df_trans.reset_index(drop = True)
    
    

        
    # n_marker = len(df.PT.unique())
        
    # df_trans = df_trans.sort_values(by = 'PT', ascending = True)
    # df_trans = df_trans.reset_index(drop = True)
        
    # pointname = ['a','b','c','d']* n_marker
        
    # df_trans['Point'] = pointname
            return df_trans





# =============================================================================
# #读取测试数据并获得所有Feature Point与最近n个点所能组成f个点的组合的所有情况
# =============================================================================

f = 4
# m = 5
n = 6

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

def get_affine_invariant(list_combination_unique, df_trans, decimal):
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
# 获取每个PT的index
# =============================================================================
def get_PT_index (df):
    index_PT = []
    PT = []
    for i in range(6):
        if i in df['PT'].values:
            index = df[df['PT'] == i].index.tolist()
            index_PT.append(index)
            PT.append(i)
        else:
            index = []
            index_PT.append(index)
            PT.append(i)
        
    df_PT_index = pd.DataFrame()
    df_PT_index['Index'] = index_PT
    df_PT_index['PT'] = PT
    return df_PT_index



# =============================================================================
# 自闭运行参数 p为选择df数量，decimal为保留小数
# =============================================================================

names = locals()
p = 46
valid_p = list(range(p))
list_data_test = get_n_file(p)

decimal = 3


for i in range(p):
    

    names['df_trans%s'%i] = trans_df(list_data_test[i])
    if names['df_trans%s'%i] is None:
        valid_p.remove(i)
        continue
    names['distances%s'%i], names['indices%s'%i] = get_closest_n_point(names['df_trans%s'%i], n)

    names['list_combination_unique%s'%i] = get_combination(names['indices%s'%i], f)
    names['df_invariant%s'%i]= get_affine_invariant(names['list_combination_unique%s'%i], names['df_trans%s'%i], decimal)


    names['df_features_%s'%i] = names['df_invariant%s'%i][['Sum', 'Min', 'Max', 'Combination']]
    names['df_index_PT_%s'%i] = get_PT_index(names['df_trans%s'%i])
    
    
# =============================================================================
# 尝试配对  
# =============================================================================
def get_index(lis, value):
    return[i for i in range(len(lis)) if abs(lis[i]-value) == min(list(abs((lis - value))))]

names = locals()

for i in valid_p:
    names['Maxlist%s'%i] = names['df_features_%s'%i]['Max'].values.tolist()
    names['Minlist%s'%i] = names['df_features_%s'%i]['Min'].values.tolist()
    names['Sumlist%s'%i] = names['df_features_%s'%i]['Sum'].values.tolist()
    
    names['list_index_max_%s'%i] = []
    names['list_index_min_%s'%i] = []
    names['list_index_sum_%s'%i] = []
    for m in range(6):
        dat = get_index(names['Maxlist%s'%i], df_feature_ref.Max[m])
        dat2 = get_index(names['Minlist%s'%i], df_feature_ref.Min[m])
        dat3 = get_index(names['Sumlist%s'%i], df_feature_ref.Sum[m])
        names['list_index_max_%s'%i].append(dat)
        names['list_index_min_%s'%i].append(dat2)
        names['list_index_sum_%s'%i].append(dat3)       
        
    list_chosen_index = []
    for j in range(6):        #算出每行(每个PT)的index的所有误差和，从而选出误差和最小的index，将此index当作为最合适的marker
        all_date = names['list_index_max_%s'%i][j]+names['list_index_min_%s'%i][j]+names['list_index_sum_%s'%i][j] #不去除重复情况下的index
        date = list(set(names['list_index_max_%s'%i][j]+names['list_index_min_%s'%i][j]+names['list_index_sum_%s'%i][j])) #合并第j行的三个max min sum list，在其中每一个index求error和
        error_data = []
        count_data = []
        for k in range(len(date)): #求第j行的第k个元素的error和，并保存他的index
            m = j
            n = date[k]
            count = all_date.count(date[k])   #数第j行第k个元素出现次数
            count_index = [count, n]
            count_data.append(count_index)
            
            error = abs(names['df_features_%s'%i]['Sum'][n]- df_feature_ref['Sum'][m])+abs(names['df_features_%s'%i]['Max'][n]- df_feature_ref['Max'][m])+abs(names['df_features_%s'%i]['Min'][n]- df_feature_ref['Min'][m])
            error_index = n
            dat_error = [error, n]
            error_data.append(dat_error)
        
        df_index_count = pd.DataFrame( )
        df_index_count[['Count', 'Index']] = count_data
            
        df_error_data = pd.DataFrame()
        df_error_data[['Error','Index']] = error_data
        
        max_count = df_index_count['Count'].max()
        list_index_count = df_index_count['Count'].values.tolist()
        unique = list_index_count.count(max_count)
        

        if unique == 1:
            index_chosen = df_index_count['Index'][df_index_count['Count'] == max_count].values.tolist()
            list_chosen_index.append(index_chosen)
            
        else:
            max_count_index = df_index_count[df_index_count['Count'] == max_count].index.tolist()
            value_list = []
            for q in max_count_index:
                index_error = df_index_count['Index'][q]

                value = float(df_error_data['Error'][df_error_data['Index'] == index_error])
                dat_value = [value, index_error]
                value_list.append(dat_value)
            df_error_max_count = pd.DataFrame() 
            df_error_max_count[['Error', 'Index']] = value_list
            df_error_max_count['Index'] = df_error_max_count['Index'].astype(int)
            error_min = df_error_max_count.Error.min()
            index_chosen = df_error_max_count['Index'][df_error_max_count['Error'] == error_min].values.tolist()
            list_chosen_index.append(index_chosen)
        
        

    names['list_chosen_index_%s'%i] = list_chosen_index
        
    
    
def test(list_chosen, df_features, df_index_PT):
    fenmu = 0
    fenzi = 0
    for i in range(6):
        if df_index_PT['Index'][i] != []:
            index_calculated = list_chosen[i]
            fenmu = fenmu+1
            if df_features['Combination'][index_calculated[0]] == df_index_PT['Index'][i]:
                fenzi = fenzi+1
    
    Accuracy = fenzi/ fenmu 

    return Accuracy
        
        
# =============================================================================
# 校验
# =============================================================================
        
# m = 3 #PT
# n = 8 #index    所有index都是从df_trans表中查询
# abs(df_features_8['Sum'][n]- df_feature_ref['Sum'][m])+abs(df_features_8['Max'][n]- df_feature_ref['Max'][m])+abs(df_features_8['Min'][n]- df_feature_ref['Min'][m])



names = locals()
total = 0

for i in valid_p:
    names['Accuracy%s'%i] = test(names['list_chosen_index_%s'%i], names['df_features_%s'%i], names['df_index_PT_%s'%i])
    total = total+names['Accuracy%s'%i]

    print( names['Accuracy%s'%i])
    

total_accuracy = total/len(valid_p)
print('total accuracy is ' + str(total_accuracy))













        

