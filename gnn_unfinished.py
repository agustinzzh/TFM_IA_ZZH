#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 15:54:51 2022

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
from ref import get_reference
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
        augmentated_path = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_augmented'
        augmentated_list = []
        file_list = []
        
        for i in os.listdir(file_path):
            file_list.append(os.path.join(file_path, i))

        for k in os.listdir(augmentated_path):
            augmentated_list.append(os.path.join(augmentated_path, k))
        
        names = locals()
        list_df = []
        list_df2 = []
        
        for j in range(len(file_list)):
            names['df_%s'%j] = pd.read_table(file_list[j], sep='\t', skiprows=2)
            list_df.append(names['df_%s'%j])

        for n in range(len(augmentated_list)):
            names['df_augmentated_%s'%n] = pd.read_table(augmentated_list[n], sep='\t')
            list_df2.append(names['df_augmentated_%s'%n])
            
        return list_df, list_df2
    
    # in this case only consider the points with PT marked (not NA or not -1)
    def trans_df_data(self):
        list_df, list_df2 = self.get_file()
        list_df = list_df+list_df2

        names = locals()
        
        list_df_trans = []
        for i in range(len(list_df)):
            df = list_df[i]
            # get ride of NA and -1
            df = df.dropna(subset = ['PT'])
            df = df[df.PT >= 0]
            df = df[df.PT <= 5]
            
            for j in df.PT.unique():
                if len(df[df.PT == j]) != 4:
                    df = df.drop(df[df.PT == j].index)
            # number of PT
            n_marker = len(df.PT.unique())
            df_trans = df[['H', 'V', 'PT']]
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
        
        df_invariant['Sum'] = df_invariant.apply(lambda x: x.sum(), axis = 1)
        df_invariant['Min'] = df_invariant.apply(lambda x: x.min(), axis = 1)
        df_invariant['Max'] = df_invariant.iloc[:,:12].apply(lambda x: x.max(), axis = 1)
        df_invariant.Sum = df_invariant.Sum.round(decimal)
        df_invariant.Max = df_invariant.Max.round(decimal)
        df_invariant.Min = df_invariant.Min.round(decimal)
        df_invariant['Combination'] = total_index
        df_invariant['Common_side'] = total_commonside
        
        df_features = df_invariant[['Sum', 'Max', 'Min', 'Combination','Common_side']]
        
    
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
            print(i)
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
                all_date = list_index_max[j]+ list_index_min[j]+ list_index_sum[j]
                date = list(set(list_index_max[j]+ list_index_min[j]+ list_index_sum[j]))
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
            
            names['chosen_point%s'%i] = list_chosen_index
            list_chosen_point.append(names['chosen_point%s'%i])
            i += 1
        
        return list_chosen_point, list_features
    

    def get_PT_index (self):
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
    
    
    
d = 2
n = 5    
path_ref = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Dato_prueba/180MUA_Formatted_F.txt' 
file_path = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion'
ref = get_reference(path_ref, d)
df_feature_ref, df_points_ref = ref.get_invariant_ref()
df_trans_ref = ref.trans_df()
realdata = get_real_data(file_path,df_feature_ref, d, n)
list_df_trans = realdata.trans_df_data()
list_chosen_point, list_features = realdata.main_process()
list_PT_index = realdata.get_PT_index()




list_df_Candidate = []
list_point_total = []
candidate_number = 0
for i in range(len(list_features)):
    print(i)
    features = list_features[i]
    df_trans = list_df_trans[i]
    PT_index = list_PT_index[i]
    PT_points = PT_index['Index'].values.tolist()
    tri1 = []
    tri2 = []
    common_side = []
    point = []
    invariant = []
    names = locals()
    
    names['list_point_%s'%i] = []
    for j in range(len(features)):
        
        
        # get the index of the points of each combination (index of the point in df_trans)
        points = features.Combination[j]
        point_a = points[0]
        point_b = points[1]
        point_c = points[2]
        point_d = points[3]
        l_index = [point_a, point_b, point_c, point_d]
        l_index_graph = [str(i) + '_' + str(points[0]), str(i) + '_' + str(points[1]), str(i) + '_' + str(points[2]), str(i) + '_' + str(points[3])]
        
        a = [df_trans.H[point_a], df_trans.V[point_a]]
        b = [df_trans.H[point_b], df_trans.V[point_b]]
        c = [df_trans.H[point_c], df_trans.V[point_c]]
        d = [df_trans.H[point_d], df_trans.V[point_d]]
        l = [a,b,c,d]
        
        
        # get the x cordinate of each point
        ax = df_trans.H[point_a]
        bx = df_trans.H[point_b]
        cx = df_trans.H[point_c]
        dx = df_trans.H[point_d]
        lx = [ax,bx,cx,dx]
        
        # get the y cordinate of each point
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
        names['df_point_%s'%j]['index_g'] = l_index_graph
        names['df_point_%s'%j]['Node'] = np.nan
        names['df_point_%s'%j]['Candidate'] = [candidate_number]*4
        candidate_number = candidate_number+1
        names['df_point_%s'%j]['Corrdinate'] = l
        names['df_point_%s'%j]['graph'] = [i]*4
        if points in PT_points:
            PT = PT_points.index(points) #the index of PT_points is the PT number
            names['df_point_%s'%j]['PT'] = PT
        else:
            names['df_point_%s'%j]['PT'] = 7
    
        names['list_point_%s'%i].append(names['df_point_%s'%j])
        
        
        df_line = realdata.get_df_line_data(names['df_point_%s'%j])
        df_line = df_line.reset_index(drop = True)
        
        sidelist = df_line.values.tolist()
        
        # get the two point that belong to the uncommon side of each triangle. 
        #for example, the triangle1 is abc and triangle2 is bcd, the common side is bc
        # here in the final datafrmae save bc to the column common side, point 1 point 2 are for points of line bd, and point3 pint4 for line cd
        # the same, point5 and 6 for bd, point 7 and 8 for cd
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
        
            area_triangle1 = realdata.get_area(side_dis_common , side_dis_1_2, side_dis_1_3)
            area_triangle2 = realdata.get_area(side_dis_common, side_dis_2_2, side_dis_2_3)
            
            cross_ratio = area_triangle1/ area_triangle2
            cross_ratio2 = area_triangle2/ area_triangle1
        
            candidate = j
            dat = [cross_ratio, candidate]
            dat2 = [cross_ratio2, candidate]
        
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
        
    names['df_Candidate_%s'%i] = pd.DataFrame()       
    names['df_Candidate_%s'%i][['Invariant', 'Candidate']] = invariant
    names['df_Candidate_%s'%i][['Point_1', 'Point_2','Point_3', 'Point_4', 'Point_5', 'Point_6', 'Point_7', 'Point_8']] = point
         
    names['df_Candidate_%s'%i]['Common_side'] = common_side
    names['df_Candidate_%s'%i] = names['df_Candidate_%s'%i].groupby('Candidate').apply(lambda x : x.sort_values('Invariant', ascending = True))   
    names['df_Candidate_%s'%i] = names['df_Candidate_%s'%i].reset_index(drop = True)
    
    list_df_Candidate.append(names['df_Candidate_%s'%i])
    list_point_total.append(names['list_point_%s'%i])
    
    
    
for j in range(len(list_df_Candidate)):
     
     num_candidate = len(list_df_Candidate)
     print(str(j)+ '/'+ str(num_candidate))
     df_Candidate = list_df_Candidate[j]
     list_point = list_point_total[j]
     candidate_unique = np.sort(df_Candidate.Candidate.unique())
     for i in range(len(candidate_unique)):
         
         df_point = list_point[i]
         
         df = df_Candidate[df_Candidate.Candidate == i]
         
         # datafrmae that contains the points that compose the two uncommon side, thus the vertex of the triangle will be counted twice
         # in this case we only need to find the point in the data frame that has appeared twice, this mean it is the vertex of this triangle
         tri_1 = df.iloc[0:1, 2:6 ].values.tolist()[0]#Select the min value of the reference, get the vertexs of the two triangle(triangle1 is the samller triangle)
         tri_2 = df.iloc[0:1, 6:10].values.tolist()[0]#index = 0 because in df_Candidate its already ordened by Invariant value
         
         tri_1_count = pd.DataFrame()   
         tri_2_count = pd.DataFrame()
         # this return a dataframe the index is the name of the point, and with column count which stands for the time of this point counted
         tri_1_count['Count'] = pd.value_counts(tri_1)
         tri_2_count['Count'] = pd.value_counts(tri_2)
         
         vertex_1 = tri_1_count[tri_1_count.Count == 2].index.tolist()[0]
         vertex_2 = tri_2_count[tri_2_count.Count == 2].index.tolist()[0]
         
         new_common_side1 = vertex_1+ vertex_2
         new_common_side2 = vertex_2+ vertex_1

# =============================================================================
# new common side and find other two vertex
# =============================================================================
         df2 = df_Candidate[((df_Candidate.Common_side == new_common_side1)|(df_Candidate.Common_side == new_common_side2)) & (df_Candidate.Candidate == i)]
         df2 = df2[df2.Invariant < 1]
         
         tri_3 = df2.iloc[0:1, 2:6].values.tolist()[0]
         tri_4 = df2.iloc[0:1, 6:10].values.tolist()[0]
         
         tri_3_count = pd.DataFrame()
         tri_3_count['Count'] = pd.value_counts(tri_3)
    
         tri_4_count = pd.DataFrame()
         tri_4_count['Count'] = pd.value_counts(tri_4)
            
         vertex_3 = tri_3_count[tri_3_count.Count == 2].index.tolist()[0]
         vertex_4 = tri_4_count[tri_4_count.Count == 2].index.tolist()[0]
         
         df_point['Node'][names['df_point_%s'%i].Point == vertex_1 ] = 'Node1'
         df_point['Node'][names['df_point_%s'%i].Point == vertex_2 ] = 'Node2'
         df_point['Node'][names['df_point_%s'%i].Point == vertex_3 ] = 'Node3'
         df_point['Node'][names['df_point_%s'%i].Point == vertex_4 ] = 'Node4'
         

    
         
    
    

df_node_total = pd.DataFrame(columns = ['X', 'Y', 'Point','index','index_g', 'Node', 'Candidate', 'PT', 'graph'])  
    

    
    
for j in range(len(list_point_total)):
    print(str(j)+ '/'+ str(len(list_point_total)))
    list_node = list_point_total[j]
    for i in range(len(list_node)):
        df_node = list_node[i]
        df_node = df_node.drop(columns = ['Corrdinate'])
        X = (df_node.X.sum())/4
        Y = (df_node.Y.sum())/4
        Point = 'e'
        index_0 = -1
        index_0_g = str(j) + '_' + '-1'
        Node = 'Node0'
        Candidate = df_node.Candidate[0]
        graph = df_node.graph[0]
        PT = df_node.PT[0]
        dat = {"X":X, "Y":Y, "Point":Point, "index":index_0, 'index_g':index_0_g , "Node":Node, "Candidate":Candidate, "PT":PT, 'graph':graph}
        df_node = df_node.append(dat, ignore_index = True)
        df_node_total = df_node_total.append(df_node, ignore_index = True)
    
    
df_4nodes = df_node_total.drop(df_node_total[df_node_total.Node == 'Node0'].index).reset_index(drop = True)

    
    




#edge list
# edge = []
# for graph in range(len(list_point_total)):
#     df1 = df_4nodes[df_4nodes.graph == graph]
#     for candidate in df1.Candidate.unique():
#         df2 = df1[df1.Candidate == candidate]
#         vertex1 = df2.index_g[df2.Node == 'Node1'].values.tolist()[0]
#         vertex2 = df2.index_g[df2.Node == 'Node2'].values.tolist()[0]
#         vertex3 = df2.index_g[df2.Node == 'Node3'].values.tolist()[0]
#         vertex4 = df2.index_g[df2.Node == 'Node4'].values.tolist()[0]
#         edge1_3 = [vertex1, vertex3]
#         edge1_4 = [vertex1, vertex4]
#         edge2_3 = [vertex2, vertex3]
#         edge2_4 = [vertex2, vertex4]
#         edge.append(edge1_3)
#         edge.append(edge1_4)
#         edge.append(edge2_3)
#         edge.append(edge2_4)
    
#     df_edge = pd.DataFrame()
#     df_edge[['Origin', 'Object']] = edge




# import networkx as nx

# G = nx.Graph()

# G = nx.from_pandas_edgelist(df_edge, 'Origin', 'Object')

# from matplotlib.pyplot import figure
# figure(figsize = (10, 8))
# nx.draw_shell(G, with_labels=True)


df_node_total.to_csv(os.path.join('/Users/agustinzhang/Downloads/master_AI/TFM/Dato', 'df_node_total.csv'), index=False)
# =============================================================================
# Feature matrix
# =============================================================================
# import pandas as pd
# import numpy as np
# import math


# df_node_total = pd.read_csv('df_node_total.csv')
# import torch
# from torch_geometric.data import Data
# import torch_geometric.data as data
# import os.path as osp

# from torch_geometric.data import Dataset, download_url

# list_graph = []
# for i in df_node_total.graph.unique():
#     df_g = df_node_total[df_node_total['graph'] == i]
#     for j in df_g.Candidate.unique():
#         df_c = df_g[df_g['Candidate'] == j]
#         node0 = [df_c['X'][df_c['Node'] == 'Node0'].values.tolist()[0] , df_c['Y'][df_c['Node'] == 'Node0'].values.tolist()[0]]
#         node1 = [df_c['X'][df_c['Node'] == 'Node1'].values.tolist()[0] , df_c['Y'][df_c['Node'] == 'Node1'].values.tolist()[0]]
#         node2 = [df_c['X'][df_c['Node'] == 'Node2'].values.tolist()[0] , df_c['Y'][df_c['Node'] == 'Node2'].values.tolist()[0]]
#         node3 = [df_c['X'][df_c['Node'] == 'Node3'].values.tolist()[0] , df_c['Y'][df_c['Node'] == 'Node3'].values.tolist()[0]]
#         node4 = [df_c['X'][df_c['Node'] == 'Node4'].values.tolist()[0] , df_c['Y'][df_c['Node'] == 'Node4'].values.tolist()[0]]
        
        
#         #sum of distance to other 3 nodes
#         feature_1 = realdata.get_distance(node1, node2)+ realdata.get_distance(node1, node3)+ realdata.get_distance(node1, node4)
#         feature_2 = realdata.get_distance(node2, node1)+ realdata.get_distance(node2, node3)+ realdata.get_distance(node2, node4)
#         feature_3 = realdata.get_distance(node3, node1)+ realdata.get_distance(node3, node2)+ realdata.get_distance(node3, node4)
#         feature_4 = realdata.get_distance(node4, node1)+ realdata.get_distance(node4, node2)+ realdata.get_distance(node4, node3)


#         #distance to (0,0)
#         p0 = [0,0]
#         feature_1_0 = realdata.get_distance(node1, p0)
#         feature_2_0 = realdata.get_distance(node2, p0)
#         feature_3_0 = realdata.get_distance(node3, p0)
#         feature_4_0 = realdata.get_distance(node4, p0)
        
        
#         matrix_f = np.zeros((4,2))
#         matrix_f[0,0] = feature_1
#         matrix_f[0,1] = feature_1_0
#         matrix_f[1,0] = feature_2
#         matrix_f[1,1] = feature_2_0
#         matrix_f[2,0] = feature_3
#         matrix_f[2,1] = feature_3_0
#         matrix_f[3,0] = feature_4
#         matrix_f[3,1] = feature_4_0
        
#         label = np.zeros((1))
#         label[0] = df_c.PT.values.tolist()[0]
        
#         #edges
#         edge_index = []
#         edge_index.append([0,1])
#         edge_index.append([0,2])
#         edge_index.append([0,3])
#         edge_index.append([1,0])
#         edge_index.append([1,2])
#         edge_index.append([1,3])
#         edge_index.append([2,0])
#         edge_index.append([2,1])
#         edge_index.append([3,0])
#         edge_index.append([3,1])
        
#         edge_data = torch.tensor(edge_index, dtype = torch.long)
        
#         #edge feature
#         edge_feature = np.zeros((10,1))
#         edge_feature[0,0] = realdata.get_distance(node1, node2)
#         edge_feature[1,0] = realdata.get_distance(node1, node3)
#         edge_feature[2,0] = realdata.get_distance(node1, node4)
#         edge_feature[3,0] = realdata.get_distance(node2, node1)
#         edge_feature[4,0] = realdata.get_distance(node2, node3)
#         edge_feature[5,0] = realdata.get_distance(node2, node4)
#         edge_feature[6,0] = realdata.get_distance(node3, node1)
#         edge_feature[7,0] = realdata.get_distance(node3, node2)
#         edge_feature[8,0] = realdata.get_distance(node4, node1)
#         edge_feature[9,0] = realdata.get_distance(node4, node2)
        
#         edge_attr = torch.tensor(edge_feature, dtype = torch.long)
#         # data、
#         x = torch.tensor(matrix_f, dtype = torch.float)
#         y = torch.tensor(label, dtype = torch.int)
    
#         graph = Data(x = matrix_f, edge_index = edge_data.t() , edge_attr = edge_attr ,y = y, num_nodes = 4)
#         list_graph.append(graph)

        
# from torch_geometric.utils.convert import to_networkx


# graph_vis = list_graph[235]
# edge_vis = graph_vis.edge_index.t().numpy()
# edge_weight = graph_vis.edge_attr.numpy()
# edges = [(edge_vis[0][0], edge_vis[0][1], edge_weight[0][0]),
#          (edge_vis[1][0], edge_vis[1][1], edge_weight[1][0]),
#          (edge_vis[2][0], edge_vis[2][1], edge_weight[2][0]),
#          (edge_vis[4][0], edge_vis[4][1], edge_weight[4][0]),
#          (edge_vis[5][0], edge_vis[5][1], edge_weight[5][0])]


# G1 = nx.Graph()
# G1.add_weighted_edges_from(edges)

# nx.draw(G1, font_weight= 'bold')



# vis = to_networkx(graph_vis)
# figure(1, figsize = (15,13))
# nx.draw(vis)



# from torch.utils.data import Dataset,  random_split
# from torch_geometric.loader import DataLoader

# loader = DataLoader(list_graph, batch_size = 32)




        
# class graph_data_set(Dataset):
    
#     def __init__(self):
        
#         self.list_graph = list_graph
        
#     def __len__(self):
#         return len(self.list_graph) 
    
#     def __getitem__(self, idx):
#         return self.list_graph[idx]
    
#     def split_idx(self):
        
#         train_size = int(0.8 * len(self.list_graph))
#         test_size = len(self.list_graph) - train_size
#         train_dataset, test_dataset = random_split(self.list_graph, [train_size, test_size])
        
#         return train_dataset, test_dataset
            
            



# g_dataset = graph_data_set()
# train_dataset, test_dataset = g_dataset.split_idx()
# train_loader = DataLoader(train_dataset, batch_size= 32, shuffle=True, num_workers=0)
# test_loader = DataLoader(test_dataset, batch_size = 32, shuffle=False, num_workers = 0)

# # =============================================================================
# # model
# # =============================================================================
# from ogb.graphproppred.mol_encoder import AtomEncoder
# from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F
# from tqdm.notebook import tqdm
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
#                  dropout, return_embeds=False):
#         # TODO: Implement this function that initializes self.convs,
#         # self.bns, and self.softmax.

#         super(GCN, self).__init__()

#         # A list of GCNConv layers
#         self.convs = None

#         # A list of 1D batch normalization layers
#         self.bns = None

#         # The log softmax layer
#         self.softmax = None

#         ############# Your code here ############
#         ## Note:
#         ## 1. You should use torch.nn.ModuleList for self.convs and self.bns
#         ## 2. self.convs has num_layers GCNConv layers
#         ## 3. self.bns has num_layers - 1 BatchNorm1d layers
#         ## 4. You should use torch.nn.LogSoftmax for self.softmax
#         ## 5. The parameters you can set for GCNConv include 'in_channels' and
#         ## 'out_channels'. More information please refer to the documentation:
#         ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
#         ## 6. The only parameter you need to set for BatchNorm1d is 'num_features'
#         ## More information please refer to the documentation:
#         ## https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
#         ## (~10 lines of code)
#         self.convs = torch.nn.ModuleList([GCNConv(input_dim,hidden_dim)] + [GCNConv(hidden_dim,hidden_dim) for i in range(0,num_layers-2)] + [GCNConv(hidden_dim,output_dim)])
#         self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for i in range(0,num_layers-1)])
#         self.softmax = torch.nn.LogSoftmax(dim=-1)
#         #########################################

#         # Probability of an element to be zeroed
#         self.dropout = dropout

#         # Skip classification layer and return node embeddings
#         self.return_embeds = return_embeds

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()

#     def forward(self, x, adj_t):
#         # TODO: Implement this function that takes the feature tensor x,
#         # edge_index tensor adj_t and returns the output tensor as
#         # shown in the figure.

#         out = None
#         ############# Your code here ############
#         ## Note:
#         ## 1. Construct the network as showing in the figure
#         ## 2. torch.nn.functional.relu and torch.nn.functional.dropout are useful
#         ## More information please refer to the documentation:
#         ## https://pytorch.org/docs/stable/nn.functional.html
#         ## 3. Don't forget to set F.dropout training to self.training
#         ## 4. If return_embeds is True, then skip the last softmax layer
#         ## (~10 lines of code)
#         for i in range(0,len(self.bns)):
#           x = self.convs[i](x,adj_t)
#           x = self.bns[i](x)
#           x = F.relu(x)
#           x = F.dropout(x,p=self.dropout,training=self.training)
#         x = self.convs[-1](x,adj_t)
#         if self.return_embeds:
#           out = x
#         else:
#           out = self.softmax(x)


#         #########################################

#         return out

# class GCN_Graph(torch.nn.Module):
#     def __init__(self, hidden_dim, output_dim, num_layers, dropout):
#         super(GCN_Graph, self).__init__()
        
#         self.node_encoder = AtomEncoder(hidden_dim)
        
#         self.gnn_node = GCN(hidden_dim, hidden_dim, 
#                             hidden_dim, num_layers, dropout, return_embeds = True)
        
#         self.pool = None
        
#         self.pool = global_max_pool
        
#         self.linear = torch.nn.Linear(hidden_dim, output_dim)

#     def reset_parameters(self):
#       self.gnn_node.reset_parameters()
#       self.linear.reset_parameters()

#     def forward(self, batched_data):
#         # TODO: Implement this function that takes the input tensor batched_data,
#         # returns a batched output tensor for each graph.
#         x, adj_t, batch = batched_data.x, batched_data.edge_index, batched_data.batch
#         embed = self.node_encoder(x)

#         out = None

#         ############# Your code here ############
#         ## Note:
#         ## 1. Construct node embeddings using existing GCN model
#         ## 2. Use global pooling layer to construct features for the whole graph
#         ## More information please refer to the documentation:
#         ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
#         ## 3. Use a linear layer to predict the graph property
#         ## (~3 lines of code)
#         out = self.gnn_node(embed,adj_t)
#         out = self.pool(out,batch)
#         out = self.linear(out)
#         #########################################
#         return out
    
# def train(model, device, data_loader, optimizer, loss_fn):
#     # TODO: Implement this function that trains the model by
#     # using the given optimizer and loss_fn.
#         model.train()
#         loss = 0

#         for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
#             batch = batch.to(device)

#             if  batch.batch[-1] == 0:
#                 pass
#             else:
#         ## ignore nan targets (unlabeled) when computing training loss.
#                 is_labeled = batch.y == batch.y

#         ############# Your code here ############
#         ## Note:
#         ## 1. Zero grad the optimizer
#         ## 2. Feed the data into the model
#         ## 3. Use `is_labeled` mask to filter output and labels
#         ## 4. You might change the type of label
#         ## 5. Feed the output and label to loss_fn
#         ## (~5 lines of code)
#                 optimizer.zero_grad()
#                 out = model(batch)
#                 out = out[is_labeled]
#                 labels = batch.y[is_labeled]
#                 labels = labels.float()
#                 loss = loss_fn(out,labels)

#         #########################################

#                 loss.backward()
#                 optimizer.step()

#         return loss.item()
    
    
    
# # The evaluation function
# def eval(model, device, loader, evaluator):
#     model.eval()
#     y_true = []
#     y_pred = []

#     for step, batch in enumerate(tqdm(loader, desc="Iteration")):
#         batch = batch.to(device)

#         if batch.x.shape[0] == 1:
#             pass
#         else:
#             with torch.no_grad():
#                 pred = model(batch)

#             y_true.append(batch.y.view(pred.shape).detach().cpu())
#             y_pred.append(pred.detach().cpu())
#     y_true = torch.cat(y_true, dim = 0).numpy()
#     y_pred = torch.cat(y_pred, dim = 0).numpy()


#     input_dict = {"y_true": y_true, "y_pred": y_pred}

#     return evaluator.eval(input_dict)





# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Please do not change the args
# args = {
#     'device': device,
#     'num_layers': 5,
#     'hidden_dim': 256,
#     'dropout': 0.5,
#     'lr': 0.001,
#     'epochs': 30,
# }
# args

# model = GCN_Graph(args['hidden_dim'],
#             1, args['num_layers'],
#             args['dropout']).to(device)

# import torch.nn as nn
# evaluator = nn.Softmax


# model.reset_parameters()

# optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
# loss_fn = torch.nn.BCEWithLogitsLoss()

# best_model = None
# best_valid_acc = 0

# for epoch in range(1, 1 + args["epochs"]):
#   print('Training...')
#   loss = train(model, device, train_loader, optimizer, loss_fn)

#   print('Evaluating...')
#   train_result = eval(model, device, train_loader, evaluator)
#   test_result = eval(model, device, test_loader, evaluator)

#   train_acc,  test_acc = train_result[dataset.eval_metric],  test_result[dataset.eval_metric]

#   print(f'Epoch: {epoch:02d}, '
#         f'Loss: {loss:.4f}, '
#         f'Train: {100 * train_acc:.2f}%, '
#         f'Test: {100 * test_acc:.2f}%')





































    
    