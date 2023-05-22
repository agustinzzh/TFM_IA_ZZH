#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:09:54 2022

@author: agustinzhang
"""

import pandas as pd
import numpy as np
import cv2
import math


def trans_df( df1, PT):
    df_trans = df1[['H', 'V', 'R', 'C', 'PT']]
    df_trans['H'] = df_trans.H+2000.5
    df_trans['V'] = df_trans.V+1500.5

    df_trans = df_trans[df_trans.PT == PT]
        
    n_marker = 1

        
    df_trans = df_trans.sort_values(by = 'PT', ascending = True)
    df_trans = df_trans.reset_index(drop = True)
        
    pointname = ['a','b','c','d']* n_marker
        
    df_trans['Point'] = pointname
    return df_trans


def draw_background(df, PT):
    
    df = trans_df(df, PT)
    newImg = np.zeros((3001, 4001, 3), np.uint8)
    newImg.fill(0)
    
    listpoint = df[['H', 'V', 'R']].values.tolist()
    for (x, y, r) in listpoint:
        cv2.circle(newImg, (int(x),int(y)), int(r), (40, 255, 45), 3)
        cv2.putText(newImg, str(PT), (int(x),int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (40,255-40,45), 3)
        
    return newImg

def get_distance(point1, point2):
    return math.sqrt((point2[0]- point1[0])**2 + (point2[1]- point1[1])**2)

def get_area(side1, side2, side3):
    s = (side1+side2+side3)/2
    area = (s*(s-side1)*(s-side2)*(s-side3))**0.5
    return area

def get_affine_invariant(triangle1, triangle2):
    invariant = triangle1/triangle2
    return invariant

    
def get_df_line(df, PT):
    p1list = []
    p2list = []
    linename = []
    linelist = []
    box = []
    df_trans = trans_df(df, PT)
    for i in range(PT, PT+1):
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
    return df_line


def get_invariante_and_IMG(df, PT,d):
    invariant = []
    tri1 = []
    tri2 = []
    common_side = []
    point = []
    df_line = get_df_line(df, PT)
    df_trans = trans_df(df, PT)
    for i in range(PT, PT+1):
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

        df_invariant = df_invariant.sort_values(by = 'Cross_ratio', ascending = True)
        df_invariant = df_invariant.reset_index(drop = True)

        
        green = (0, 255, 0)
        red = (0, 0, 255)
        
        image_list =[]
        for n in range(len(df_invariant)):
            IMG = draw_background(df, PT)
            cv2.line(IMG, (int(df_trans.H[df_trans.Point == df_invariant.Point_1[n]]),int(df_trans.V[df_trans.Point == df_invariant.Point_1[n]])), 
             (int(df_trans.H[df_trans.Point == df_invariant.Point_2[n]]),int(df_trans.V[df_trans.Point == df_invariant.Point_2[n]])), green,3 )
            cv2.line(IMG, (int(df_trans.H[df_trans.Point == df_invariant.Point_3[n]]),int(df_trans.V[df_trans.Point == df_invariant.Point_3[n]])), 
             (int(df_trans.H[df_trans.Point == df_invariant.Point_4[n]]),int(df_trans.V[df_trans.Point == df_invariant.Point_4[n]])), green,3 )
    
    
            cv2.line(IMG, (int(df_trans.H[df_trans.Point == df_invariant.Point_5[n]]),int(df_trans.V[df_trans.Point == df_invariant.Point_5[n]])), 
             (int(df_trans.H[df_trans.Point == df_invariant.Point_6[n]]),int(df_trans.V[df_trans.Point == df_invariant.Point_6[n]])), red,3 )
            cv2.line(IMG, (int(df_trans.H[df_trans.Point == df_invariant.Point_7[n]]),int(df_trans.V[df_trans.Point == df_invariant.Point_7[n]])), 
             (int(df_trans.H[df_trans.Point == df_invariant.Point_8[n]]),int(df_trans.V[df_trans.Point == df_invariant.Point_8[n]])), red,3 )    
    
    
    
    
    
    
            cv2.line(IMG, (int(df_trans.H[df_trans.Point == df_invariant.Common_side[n][0]]), int(df_trans.V[df_trans.Point == df_invariant.Common_side[n][0]])), 
             (int(df_trans.H[df_trans.Point == df_invariant.Common_side[n][1]]), int(df_trans.V[df_trans.Point == df_invariant.Common_side[n][1]])), (0,255,255),3)
            cv2.putText(IMG, str(df_invariant.Cross_ratio[n]), (int(df_trans.H[df_trans.Point == tri1[n][0][1]])-200,int(df_trans.V[df_trans.Point == tri1[n][0][1]])-200),
            cv2.FONT_HERSHEY_COMPLEX, 2.0, (100,200,200), 3)
            image_list.append(list(IMG))

        return df_invariant, image_list



PT = 5

n=5
n1 = n
n2 = n
n3 = n
n4 = n
n5 = n
n6 = n
df1 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Dato_prueba/IMG_2922.txt', sep = '\t', skiprows = 2)
df2 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion/IMG_2968.txt', sep = '\t', skiprows = 2)
df3 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion/IMG_2974.txt', sep = '\t', skiprows = 2)
df4 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion/IMG_3011.txt', sep = '\t', skiprows = 2)
df5 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion/IMG_3030.txt', sep = '\t', skiprows = 2)
df6 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion/IMG_2981.txt', sep = '\t', skiprows = 2)
#df_line = get_df_line(df, 3)
#cv2.imshow('1', IMG)

d = 2
# df1_invariant, img1_list = get_invariante_and_IMG(df1, PT, d)
# df2_invariant, img2_list = get_invariante_and_IMG(df2, PT, d)
# df3_invariant, img3_list = get_invariante_and_IMG(df3, PT, d)
# df4_invariant, img4_list = get_invariante_and_IMG(df4, PT, d)
# df5_invariant, img5_list = get_invariante_and_IMG(df5, PT, d)
# df6_invariant, img6_list = get_invariante_and_IMG(df6, PT, d)

# img1 = np.array(img1_list)
# img2 = np.array(img2_list)
# img3 = np.array(img3_list)
# img4 = np.array(img4_list)
# img5 = np.array(img5_list)
# img6 = np.array(img6_list)


# cv2.imshow('1', img1[n1])
# cv2.imshow('2', img2[n2])
# cv2.imshow('3', img3[n3])
# cv2.imshow('4', img4[n4])
# cv2.imshow('5', img5[n5])
# cv2.imshow('6', img6[n6])

def get_sum(df):
    l = []
    for i in range(6):
        df_invariant, img_list = get_invariante_and_IMG(df, i, d)
        v = df_invariant.Cross_ratio.sum()
        #v = (v* (10**(i+1)))//6
        pt = i
        dat = [v, pt]
        l.append(dat)
    df_sum = pd.DataFrame()     
    df_sum[['Sum', 'PT']]= l
    
    return df_sum

df_sum1 = get_sum(df1)
df_sum2 = get_sum(df2)
df_sum3 = get_sum(df3)
df_sum4 = get_sum(df4)
df_sum5 = get_sum(df5)
df_sum6 = get_sum(df6)

df_sum = pd.merge(df_sum1, df_sum2, on = 'PT')

df_sum = pd.merge(df_sum, df_sum3, on = 'PT')

df_sum = pd.merge(df_sum, df_sum4, on = 'PT')

df_sum = pd.merge(df_sum, df_sum5, on = 'PT')

df_sum = pd.merge(df_sum, df_sum6, on = 'PT')


def get_max(df):
    l = []
    for i in range(6):
        df_invariant, img_list = get_invariante_and_IMG(df, i, d)
        v = df_invariant.Cross_ratio.max()
        #v = (v* (10**(i+1)))//6
        pt = i
        dat = [v, pt]
        l.append(dat)
    df_max = pd.DataFrame()     
    df_max[['Max', 'PT']]= l
    
    return df_max

def get_min(df):
    l = []
    for i in range(6):
        df_invariant, img_list = get_invariante_and_IMG(df, i, d)
        v = df_invariant.Cross_ratio.min()
        #v = (v* (10**(i+1)))//6
        pt = i
        dat = [v, pt]
        l.append(dat)
    df_max = pd.DataFrame()     
    df_max[['Min', 'PT']]= l
    
    return df_max

df_max1 = get_max(df1)
df_max2 = get_max(df2)
df_max3 = get_max(df3)
df_max4 = get_max(df4)
df_max5 = get_max(df5)
df_max6 = get_max(df6)

df_max = pd.merge(df_max1, df_max2, on = 'PT')
df_max = pd.merge(df_max, df_max3, on = 'PT')
df_max = pd.merge(df_max, df_max4, on = 'PT')
df_max = pd.merge(df_max, df_max5, on = 'PT')
df_max = pd.merge(df_max, df_max6, on = 'PT')

# v1 = df1_invariant.Cross_ratio.sum()
# v2 = df2_invariant.Cross_ratio.sum()
# v3 = df3_invariant.Cross_ratio.sum()
# v4 = df4_invariant.Cross_ratio.sum()
# v5 = df5_invariant.Cross_ratio.sum()
# v6 = df6_invariant.Cross_ratio.sum()

df_min1 = get_min(df1)
df_min2 = get_min(df2)
df_min3 = get_min(df3)
df_min4 = get_min(df4)
df_min5 = get_min(df5)
df_min6 = get_min(df6)

df_min = pd.merge(df_min1, df_min2, on = 'PT')
df_min = pd.merge(df_min, df_min3, on = 'PT')
df_min = pd.merge(df_min, df_min4, on = 'PT')
df_min = pd.merge(df_min, df_min5, on = 'PT')
df_min = pd.merge(df_min, df_min6, on = 'PT')




names = locals()
l = []

for i in range(1,5):
    names['df1%s'%i] = names['df_max%s'%i] 
    l.append(names['df1%s'%i])
    
d = l[0]

































