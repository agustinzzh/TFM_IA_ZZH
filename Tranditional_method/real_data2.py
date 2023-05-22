#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 23:12:16 2023

@author: agustinzhang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:54:38 2022

@author: agustinzhang
"""

import pandas as pd
import numpy as np
import math
from itertools import combinations
import os
import random
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from ref2 import get_reference
class get_real_data:
    
    #d: the number of decimals reserved
    #n: the number of the nearest points selected around the feature point
    #path_data_file: the direction of the folder where contains the txt.
    def __init__(self, path_data_file, df_feature_ref, d, n):
        self.path_data_file = path_data_file
        self.d = d
        self.n = n
        self.df_feature_ref = df_feature_ref
    
    # import all the files in the folder as dataframe, and name them as df_n
    def get_file(self):
        file_path = self.path_data_file
        file_list = []
        
        for i in os.listdir(file_path):
            file_list.append(os.path.join(file_path, i))
        
        names = locals()
        list_df = []
        
        for j in range(len(file_list)):
            names['df_%s'%j] = pd.read_table(file_list[j], sep = '\t', skiprows = 2)
            list_df.append(names['df_%s'%j]) 
            
        return list_df
    
    # in this case only consider the points with PT marked (not NA or not -1)
    def trans_df_data(self):
        list_df = self.get_file()
        names = locals()
        
        list_df_trans = []
        for i in range(len(list_df)):
            df = list_df[i]
            # get ride of NA and -1
            df = df.dropna(subset = ['PT'])
            df = df[df.PT >= 0]
            # number of PT
            n_marker = len(df.PT.unique())
            df_trans = df[['H', 'V', 'R', 'PT']]
            # put the (0,0) in the middle of the image
            df_trans['H'] = df_trans.H+2000.5
            df_trans['V'] = df_trans.V+1500.5
            # rearrange the orden of the dataframe by PT, and reset the index
            df_trans = df_trans.sort_values(by = 'PT', ascending = True)
            df_trans = df_trans.reset_index(drop = True)
             
            # # Mark the points of each PT by a b c d, depends on how many points there are
            # # For example, there are several dataframes only have 3 points or less in under the PT value,
            # # in this case only mark the points as a b or a b c
            # # but in this script the situation of lacking of points in PT (the occultation situation) is not under the consideration
            # # in the last step, the PT which contains less than 4 points will be removed.
            # PT_unique = np.sort(df.PT.unique())
            # num = df.PT.value_counts()
            # l_p = ['a', 'b', 'c', 'd']
            # pointname = []
            # for j in range(len(PT_unique)):
            #     point = l_p[:num[PT_unique[j]]]
            #     pointname.extend(point)
             
            # # Create a new column 'Point' to put names on each point    
            # df_trans['Point'] = pointname
            
            # get rid of the PT which doesn't contain 4 points
            # also to check if the df is empty, sometimes there is certain DF doesn't have any PT marked
            if len(df_trans) != 0:
                if len(df_trans)% 4 ==0:
                    names['df_trans_%s'%i] = df_trans
                    
                    list_df_trans.append(names['df_trans_%s'%i])
                
        
        return list_df_trans
    
    
    
    def get_distance(self, point1, point2):
        return math.sqrt((point2[0]- point1[0])**2 + (point2[1]- point1[1])**2)


    def check_points_on_opposite_sides(self, points):
        list_comb=[]
    # points is a list of four tuples, each tuple is a pair of (x, y) coordinates
        for comb in combinations(points, 2):
        # calculate the line equation: y = kx + b
            k = (comb[1][1] - comb[0][1]) / (comb[1][0] - comb[0][0])
            b = comb[0][1] - k * comb[0][0]
            # check the other two points
            rest_points = [p for p in points if p not in comb]
            y1 = k * rest_points[0][0] + b
            y2 = k * rest_points[1][0] + b
            if (y1 - rest_points[0][1]) * (y2 - rest_points[1][1]) < 0:
                list_comb.append(comb)

        return list_comb

    def get_closest_n_point(self, df_trans):
        n = self.n
        feature_point = df_trans[['H', 'V']].values.tolist()
    
        #获取每个feature point最近的点共n个，距离存在distances中，index存在indices中
        nbrs = NearestNeighbors(n_neighbors = n, metric = self.get_distance).fit(feature_point)
        #这n个点之间的距离及其index
        distances, indices = nbrs.kneighbors(feature_point)  
        return distances, indices
    
    def get_combination(self, indices):
        f = 4
        feature_point = list(indices[:, 0])
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
                
        unique_list = []
        for element in list_combination:
            if element not in unique_list:
                unique_list.append(element)
                
        
        return unique_list



    def get_area(self, side1, side2, side3):
        s = (side1+side2+side3)/2
        area = (s*(s-side1)*(s-side2)*(s-side3))**0.5
        return area
    
    def get_affine_invariant(self, triangle1, triangle2):
        invariant = triangle1/triangle2
        return invariant
    
    def get_df_line_data(self, df_point):
        
        p1list = []
        p2list = []
        linename = []
        linelist = []
        box = []
    
        pointlist = df_point[['Corrdinate', 'Point']].values.tolist()
        for i in range(len(pointlist)):
            b = [i]
            box.append(b)
            for j in range(len(pointlist)):
                if [j] not in box:
                    linedis = self.get_distance(pointlist[i][0], pointlist[j][0])
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
    
    def get_affine_invariant(self, list_combination_unique, df_trans, decimal):
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
            
            df_line = self.get_df_line_data(df_point)
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
                
                area_triangle1 = self.get_area(side_dis_common , side_dis_1_2, side_dis_1_3)
                area_triangle2 = self.get_area(side_dis_common, side_dis_2_2, side_dis_2_3)
            
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
        
        
        #only choose the invariant that passed the rules
        result = []
        for pt in range(6):
            df_pt = df_trans[df_trans['PT'] == pt]
            points = [(row['X'], row['Y']) for _, row in df_pt.iterrows()]
            point_names = [row['Point'] for _, row in df_pt.iterrows()]
            line_points = df.check_points_on_opposite_sides(points)
            for lp in line_points:
                line_point_names = [point_names[points.index(p)] for p in lp]
                result.append((pt, line_point_names))
            
        list_chosen_side_name = []
        for chosen in result:
            chosen_points= chosen[1]
            chosen_side_name = chosen_points[0]+chosen_points[1]
            chosen_data = [chosen_side_name, chosen[0]]
            list_chosen_side_name.append(chosen_data)

        df_chosen_side_name = pd.DataFrame()
        df_chosen_side_name[[ 'Common_side', 'PT',]] = list_chosen_side_name
        df_chosen_side_name['PT'] = df_chosen_side_name['PT'].astype(int)

        df_selected_invariant = pd.merge(df_invariant, df_chosen_side_name, on=['PT', 'Common_side'], how='right')

        
        df_selected_invariant['Sum'] = df_selected_invariant.apply(lambda x: x.sum(), axis = 1)
        df_selected_invariant['Min'] = df_selected_invariant.apply(lambda x: x.min(), axis = 1)
        df_selected_invariant['Max'] = df_selected_invariant.iloc[:,:12].apply(lambda x: x.max(), axis = 1)
        df_selected_invariant.Sum = df_selected_invariant.Sum.round(decimal)
        df_selected_invariant.Max = df_selected_invariant.Max.round(decimal)
        df_selected_invariant.Min = df_selected_invariant.Min.round(decimal)
        df_selected_invariant['Combination'] = total_index
        df_selected_invariant['Common_side'] = total_commonside
        
        df_features = df_selected_invariant[['Sum', 'Max', 'Min', 'Combination','Common_side']]
        
    
    # df_invariant[['Com1', 'Com2', 'Com3',
    #               'Com4', 'Com5', 'Com6',
    #               'Com6', 'Com8', 'Com9',
    #               'Com10', 'Com11', 'Com12']] = total_point
    
    
    
        return df_features
        
    
    def get_index(self, lis, value):
        return[i for i in range(len(lis)) if abs(lis[i]-value) == min(list(abs((lis - value))))]

    def main_process(self):
        names = locals()
        list_df_trans = self.trans_df_data()
        df_feature_ref = self.df_feature_ref
        i = 0
        list_features = []
        list_chosen_point = []
        for df_trans in list_df_trans:
            distances, indices = self.get_closest_n_point(df_trans)
            unique_list = self.get_combination(indices)
            df_features = self.get_affine_invariant(unique_list, df_trans, d)
            
            Maxlist = df_features['Max'].values.tolist()
            Minlist = df_features['Min'].values.tolist()
            Sumlist = df_features['Sum'].values.tolist()
            
            list_index_max = []
            list_index_min = []
            list_index_sum = []

            names['df_features_%s'%i] = df_features
            list_features.append(names['df_features_%s'%i])
            
            
            for m in range(6):
                dat = self.get_index(Maxlist, df_feature_ref.Max[m])
                dat2 = self.get_index(Minlist, df_feature_ref.Min[m])
                dat3 = self.get_index(Sumlist, df_feature_ref.Sum[m])
                list_index_max.append(dat)
                list_index_min.append(dat2)
                list_index_sum.append(dat3)
            
            list_chosen_index = []
            for j in range(6):#算出每行(每个PT)的index的所有误差和，从而选出误差和最小的index，将此index当作为最合适的marker
                            #Calculate the sum of all errors of the index of each row (each PT), so as to select the index with the smallest error sum, and use this index as the most suitable marker
                all_date = list_index_max[j]+ list_index_min[j]+ list_index_sum[j]
                date = list(set(list_index_max[j]+ list_index_min[j]+ list_index_sum[j]))
                error_data = []
                count_data = []
                
                for k in range(len(date)): #求第j行的第k个元素的error和，并保存他的index Find the error sum of the kth element of the jth row and save its index
                    m = j
                    n = date[k]
                    count = all_date.count(date[k])   #数第j行第k个元素出现次数 Count the occurrences of the kth element in the jth row
                    count_index = [count, n]
                    count_data.append(count_index)
            
                    error = abs(names['df_features_%s'%i]['Sum'][n]- df_feature_ref['Sum'][m])+abs(names['df_features_%s'%i]['Max'][n]- df_feature_ref['Max'][m])+abs(names['df_features_%s'%i]['Min'][n]- df_feature_ref['Min'][m])
                    error_index = n
                    dat_error = [error, n]
                    error_data.append(dat_error)

                df_index_count = pd.DataFrame( )
                df_index_count[['Count', 'Index']] = count_data #count the occurrences of each Index
            
                df_error_data = pd.DataFrame()
                df_error_data[['Error','Index']] = error_data # the error of each index
        
                max_count = df_index_count['Count'].max()  # the index that has appeared most times
                list_index_count = df_index_count['Count'].values.tolist()
                unique = list_index_count.count(max_count) # count the occurences of the index that has been counted the most time

                if unique == 1:  # if the largest count of an index is 1, it means all the three index are unique, in this case we chosse the one with minum error value
                    index_chosen = df_index_count['Index'][df_index_count['Count'] == max_count].values.tolist()
                    list_chosen_index.append(index_chosen)
            
                else: # if the largest count is not 1, it mneans there is an index has occured at least twice or 3 times
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
            
            names['chosen_point%s'%i] = list_chosen_index
            list_chosen_point.append(names['chosen_point%s'%i])
            i += 1
        
        return list_chosen_point, list_features
    
    def get_PT_index (self): #get the index of each PT in df_trans
        list_df_trans = self.trans_df_data()
        list_PT_index = []
        names = locals()
        i = 0
        for df_trans in list_df_trans:
            index_PT = []
            PT = []
            for i in range(6):
                if i in df_trans['PT'].values:
                    index = df_trans[df_trans['PT'] == i].index.tolist()
                    index_PT.append(index)
                    PT.append(i)
                    
                else:
                    index = []
                    index_PT.append(index)
                    PT.append(i)
                    
            df_PT_index = pd.DataFrame()
            df_PT_index['Index'] = index_PT
            df_PT_index['PT'] = PT
            names['df_PT_index_%s'%i] = df_PT_index
            list_PT_index.append(names['df_PT_index_%s'%i])
            
            i = i+1
        return list_PT_index
    
    def evaluation(self, list_features, list_chosen_point):
        fenmu = 0
        fenzi = 0
        list_PT_index = self.get_PT_index()
        for i in range(len(list_PT_index)):
            df_features = list_features[i]
            df_chosen_point = list_chosen_point[i]
            df_PT_index = list_PT_index[i]
            for j in range(6):
                if df_PT_index ['Index'][j] != []:
                    index_calculated = df_chosen_point[j]
                    fenmu = fenmu+1
                    if df_features['Combination'][index_calculated[0]] == df_PT_index['Index'][j]:
                        fenzi = fenzi+1
                        
        Accuracy = fenzi/fenmu
        
        return print('Total Accuracy is:' + str(Accuracy))
        
        
    
    

    
    
    
    
    
    
d = 2
n = 5    
path_ref = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Dato_prueba/180MUA_Formatted_F.txt' 
    
ref = get_reference(path_ref, d)
df_feature_ref, df_points_ref = ref.get_invariant_ref()



file_path = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion'
df = get_real_data(file_path,df_feature_ref, d, n)
df_trans = df.trans_df_data()
distances, indices = df.get_closest_n_point(df_trans[0])

unique_list = df.get_combination(indices)
invariant = df.get_affine_invariant(unique_list, df_trans[0], d)
list_trans_df = df.trans_df_data()

list_PT_index = df.get_PT_index()
list_chosen_point, list_features = df.main_process()
df.evaluation(list_features, list_chosen_point)
