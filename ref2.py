#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:47:35 2022

@author: agustinzhang
"""



import pandas as pd
import numpy as np
#import cv2
import math
from itertools import combinations


decimal = 3


class get_reference:
    
    def __init__(self, path_ref, d):
        self.path_ref = path_ref
        self.d = d
    
    def trans_df(self):
    #Preprocessing the dataframe of reference, and convert it into an appropriate formate.
    #reference预处理，去掉NA，并转换为适合的格式
        df_ref = pd.read_table(self.path_ref, sep = '\t', header = None)
        df_ref = df_ref.dropna(axis = 0, how = 'any')
        # generate PT column, "0,0,0,0,0,0,0,0","1,1,1,1,1,1,1,1,1"....
        catlist = []
        for n in range(0, 10):
            for i in range(0, 8):
                catlist.append(n)
        df_ref['5'] = catlist
        df_ref.columns = ['D_AB', 'X', 'Y', 'Z', 'r', 'PT']
        #get rid of D_B 
        df_ref = df_ref[~ df_ref['D_AB'].str.contains('B')]
        df_ref = df_ref[df_ref['PT'] <= 5]
        
        df_trans = df_ref[['X', 'Y', 'r','PT' ]]
        df_trans['X'] = df_trans.X.values + 2000.5
        df_trans['Y'] = df_trans.Y.values + 1500.5
        df_trans = df_trans.sort_values(by = 'PT', ascending = True)
        df_trans = df_trans.reset_index(drop = True)
        
        #给df标上abcd
        #mark points in each PT with a, b, c, d
        PT_unique = np.sort(df_ref.PT.unique())
        num = df_ref.PT.value_counts()
        l_p = ['a', 'b', 'c', 'd']
        pointname = []
        for j in range(len(PT_unique)):
            point = l_p[:num[PT_unique[j]]]
            pointname.extend(point)
            
        df_trans['Point'] = pointname
        return df_trans
    
    def get_distance(self, point1, point2):
        return math.sqrt((point2[0]- point1[0])**2 + (point2[1]- point1[1])**2)
    
    def get_area(self,side1, side2, side3):
        s = (side1+side2+side3)/2
        area = (s*(s-side1)*(s-side2)*(s-side3))**0.5
        return area
    
    def get_affine_invariant(self,triangle1, triangle2):
        invariant = triangle1/triangle2
        return invariant

# get the distance of line and the identity ("a,b,c,d") of its two endpoints.
    def get_line_dis_and_id(self ):
        df_trans = self.trans_df()
        
        linelist = []
        p1list = []
        p2list = []
        linename = []
        box = []
        # get the number of markers
        n_marker = np.sort(df_trans.PT.unique())
        #in each marker, calcualte the distance of each two points that belong to this marker,
        # and save the name of these two points that we used for calculating the distance.
        for i in n_marker:
            # get all the points that belong to this PT
            pointlist = df_trans[['X', 'Y', 'Point']][df_trans.PT == i].values.tolist()
            for j in range(len(pointlist)):
                
                #in this two loop, first to create a box that contains the index of the point that we are going to use,
                #then, in the second loop, if the index is not in the box then we calculate the distance betwwen the point of the first index and the second index.
                # this is for not calculating repeatly the same points.
                b = [j]
                box.append(b)
                for k in range(len(pointlist)):
                    if [k] not in box:
                        linedis = self.get_distance(pointlist[j], pointlist[k])
                        point1 = pointlist[j][2]
                        point2 = pointlist[k][2]
                        line12 = point1+point2
                        PT_1 = i
                        dat = [linedis, PT_1]
                        #save the point1 name
                        p1list.append(point1)
                        #save the point2 name
                        p2list.append(point2)
                        #save the line name
                        linename.append(line12)
                        #save the distance and the PT
                        linelist.append(dat)
            box = []
        df_line = pd.DataFrame()   
        df_line[['Dis', 'PT']] = linelist
        df_line['P1'] = p1list
        df_line['P2'] = p2list
        df_line['Line'] = linename
        ## in the end, return a data frame that contains the lenth of all the lines composed by each posible combination of two points that belong to each PT,
        ## the identity of the line, and the identity of the two endpoints that compose this line
        ## and the PT that each line belongs to.
        
        return df_line
    
    def get_invariant_ref (self):
        df_line = self.get_line_dis_and_id()
        df_trans = self.trans_df()        
        
        invariant = []
        tri1 = []
        tri2 = []
        common_side = []
        point = []
        n_marker = np.sort(df_trans.PT.unique())
        
        for i in n_marker:
            # extract all the line information to sidelist looped by PT
            sidelist = df_line[['Dis', 'P1', 'P2', 'Line']][df_line.PT == i].values.tolist()
            df_side = df_line[['Dis', 'P1', 'P2', 'Line']][df_line.PT == i]
            # now lets regard each line as the common line of the two triangles and calculate the two possbile ratio
            for j in range(len(sidelist)):
                #create a pointbox that contains all the points, since we know the two points that compose the line,
                #we can get rid of these two points from the pointbox, thus we can get the rest two points of two triangles.
                pointbox = ['a', 'b', 'c', 'd']
                # side1 is the identity of the common side
                side1 = sidelist[j][3]
                # side_dis_1 and side_dis_2 are the lenth of the common side of the two triagnles
                side_dis_1 = sidelist[j][0]
                side_dis_2 = side_dis_1   #the same side of triangle 三角形同边
                
                # get the name of the two points that compose this line
                sidepoint = [sidelist[j][1], sidelist[j][2]]
                # get the name of the two rest points
                complementaryset = list(set(pointbox)- set(sidepoint))
                # the line that composed by these two rest points is not in neither of these two triangle,
                # so we get the extra line name, and then get rid of them from the side data frame
                extraside1 = complementaryset[0] +complementaryset[1]
                extraside2 = complementaryset[1]+ complementaryset[0]
                df_side_left = df_side[df_side.Line != extraside1]
                df_side_left = df_side_left[ df_side_left.Line != extraside2]
                # the two unique sides of triangle 1 are the lines in the df_side_left which contains one of the rest points
                triangle1 = df_side_left[df_side_left.Line.str.contains(complementaryset[0])].values.tolist()
                # the two unique sides of triangle2 are the lines that contains another point of the two rest points
                # for example: if the common side is ab, then c and d are the rest two points
                # so the unique sides of triangle1 must contain point c but won't contain point d, so now we only have to find the line that contains letter c
                # then we can do the same thing with the triangle2
                triangle2 = df_side_left[df_side_left.Line.str.contains(complementaryset[1])].values.tolist()
        
                side_dis_1_2 = triangle1[0][0]
                side_dis_1_3 = triangle1[1][0]
                side_dis_2_2 = triangle2[0][0]
                side_dis_2_3 = triangle2[1][0]
        
# =============================================================================
#                 This part is trying to save the two possible ratio of these traingles
#                   for example, ratio1 = triangle1/triangle2, ratio2 = triangle2/tirangle1,
#                    then we save the ratio and the corrsesponding sides of the triangle by the orden:
#                    sides of triangle used as molecules, then the triangle used as denominator
# =============================================================================
                area_triangle1 = self.get_area(side_dis_1 , side_dis_1_2, side_dis_1_3)
                area_triangle2 = self.get_area(side_dis_2, side_dis_2_2, side_dis_2_3)
        
                cross_ratio = round(self.get_affine_invariant(area_triangle1, area_triangle2), self.d)
                cross_ratio2 = round(self.get_affine_invariant(area_triangle2, area_triangle1), self.d) 
        
                tri1.append(triangle1)
                tri1.append(triangle2)
        
                tri2.append(triangle2)
                tri2.append(triangle1)
                

                PT_2 = i
                dat = [cross_ratio, PT_2] 
                   
                dat2 = [cross_ratio2, PT_2]
                   
                # the two unique sides of triangle1 and triangle2
                dat3 = [triangle1[0][1], triangle1[0][2], 
                   triangle1[1][1], triangle1[1][2],
                   triangle2[0][1], triangle2[0][2],
                   triangle2[1][1], triangle2[1][2]]
                # the two unique sides of triangle2 and tirangle1
                dat4 = [triangle2[0][1], triangle2[0][2], 
                   triangle2[1][1], triangle2[1][2],
                   triangle1[0][1], triangle1[0][2],
                   triangle1[1][1], triangle1[1][2]]
            
                point.append(dat3)
                point.append(dat4)
                common_side.append(side1)
                common_side.append(side1)
                invariant.append(dat)
                invariant.append(dat2)
        
            df_invariant = pd.DataFrame()   
            df_invariant[['Cross_ratio', 'PT']] = invariant 
            df_invariant = df_invariant.round(self.d)
            df_invariant[['Point_1', 'Point_2','Point_3', 'Point_4', 'Point_5', 'Point_6', 'Point_7', 'Point_8']] = point
         
            df_invariant['Common_side'] = common_side

            df_invariant = df_invariant.groupby('PT').apply(lambda x : x.sort_values('Cross_ratio', ascending = True))
            df_invariant = df_invariant.reset_index(drop = True)
            # generate the feature dataframe by extracting the max min and sum value of each PT
            
            
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

            
            
            df_max = self.get_max(df_selected_invariant)
            df_min = self.get_min(df_selected_invariant)
            df_sum = self.get_sum(df_selected_invariant)
            
            df_feature_ref = pd.concat([df_max, df_min, df_sum], axis = 1)
            df_feature_ref = df_feature_ref.T.drop_duplicates().T
      
            
            
        return df_feature_ref, df_selected_invariant
    
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

    
    
    def get_max(self,df_invariant):
        maxv = df_invariant.groupby('PT').max('Cross_ratio')
        maxv = maxv.rename(columns = {'Cross_ratio': 'Max'})
        return maxv
        
    
    def get_min(self,df_invariant):
        minv = df_invariant.groupby('PT').min('Cross_ratio')
        minv = minv.rename(columns = {'Cross_ratio': 'Min'})
        return minv

    def get_sum(self,df_invariant):
        sumv = df_invariant.groupby('PT').sum('Cross_ratio')
        sumv = sumv.rename(columns = {'Cross_ratio': 'Sum'})
        return sumv
        


path_ref = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Dato_prueba/180MUA_Formatted_F.txt'
df = get_reference(path_ref, decimal)
df_feature_ref, df_points_ref = df.get_invariant_ref()
df_line = df.get_line_dis_and_id()

# df_trans = df.trans_df()
# result = []
# for pt in range(6):
#     df_pt = df_trans[df_trans['PT'] == pt]
#     points = [(row['X'], row['Y']) for _, row in df_pt.iterrows()]
#     point_names = [row['Point'] for _, row in df_pt.iterrows()]
#     line_points = df.check_points_on_opposite_sides(points)
#     for lp in line_points:
#         line_point_names = [point_names[points.index(p)] for p in lp]
#         result.append((pt, line_point_names))

# # result is a list of tuples, each tuple contains a PT value and the names of two points
# print(result)

# list_chosen_side_name = []
# for chosen in result:
#     chosen_points= chosen[1]
#     chosen_side_name = chosen_points[0]+chosen_points[1]
#     chosen_data = [chosen_side_name, chosen[0]]
#     list_chosen_side_name.append(chosen_data)

# df_chosen_side_name = pd.DataFrame()
# df_chosen_side_name[[ 'Common_side', 'PT',]] = list_chosen_side_name
# df_chosen_side_name['PT'] = df_chosen_side_name['PT'].astype(int)

# df_selected_invariant = pd.merge(df_points_ref, df_chosen_side_name, on=['PT', 'Common_side'], how='right')
