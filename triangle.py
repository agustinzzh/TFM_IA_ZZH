#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:38:38 2022

@author: agustinzhang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
PT = 4

df1 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Dato_prueba/IMG_2922.txt', sep = '\t', skiprows = 2)


newImg = np.zeros((3001,4001,3), np.uint8)
newImg.fill(0)



df_trans = df1[['H', 'V', 'R', 'C', 'PT']]
df_trans['H'] = df_trans.H+2000.5
df_trans['V'] = df_trans.V+1500.5

df_trans = df_trans[df_trans.PT == PT]



listpoint = df_trans[['H', 'V', 'R']].values.tolist()
for (x, y, r) in listpoint:
    cv2.circle(newImg, (int(x),int(y)), int(r), (40,255,45), 3)
    cv2.putText(newImg, str(PT), (int(x),int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (40,255-40,45), 3)

# cv2.line(newImg, (0,1501), (4001,1501), (255,255,255), 2)
# cv2.line(newImg, (2001,0), (2001,3001), (255,255,255), 2)



img1 = newImg
img2 = newImg
img3 = newImg
img4 = newImg
img5 = newImg
img6 = newImg








n_marker = 1
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
for i in range(PT,PT+1):#for each PT 按照标签
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
tri1 = []
tri2 = []
common_side = []
for i in range(PT,PT+1):
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
        

        PT = i
        dat = [cross_ratio, PT]
        dat2 = [cross_ratio2, PT]
        
        common_side.append(sidepoint)
        common_side.append(sidepoint)
        invariant.append(dat)
        invariant.append(dat2)
        
df_invariant = pd.DataFrame()   
df_invariant[['Cross_ratio', 'PT']] = invariant 


def draw_triangle (imagine, sanjiao1, sanjiao2, n):
    green = (0, 255, 0)
    red = (0, 0, 255)
    cv2.line(imagine, (int(df_trans.H[df_trans.Point == sanjiao1[n][0][1]]),int(df_trans.V[df_trans.Point == sanjiao1[n][0][1]])), 
             (int(df_trans.H[df_trans.Point == sanjiao1[n][0][2]]),int(df_trans.V[df_trans.Point == sanjiao1[n][0][2]])), green,3 )
    cv2.line(imagine, (int(df_trans.H[df_trans.Point == sanjiao1[n][1][1]]),int(df_trans.V[df_trans.Point == sanjiao1[n][1][1]])), 
             (int(df_trans.H[df_trans.Point == sanjiao1[n][1][2]]),int(df_trans.V[df_trans.Point == sanjiao1[n][1][2]])), green, 3)

    cv2.line(imagine, (int(df_trans.H[df_trans.Point == sanjiao2[n][0][1]]),int(df_trans.V[df_trans.Point == sanjiao2[n][0][1]])), 
             (int(df_trans.H[df_trans.Point == sanjiao2[n][0][2]]),int(df_trans.V[df_trans.Point == sanjiao2[n][0][2]])), red,  3)
    cv2.line(imagine, (int(df_trans.H[df_trans.Point == sanjiao2[n][1][1]]),int(df_trans.V[df_trans.Point == sanjiao2[n][1][1]])), 
             (int(df_trans.H[df_trans.Point == sanjiao2[n][1][2]]),int(df_trans.V[df_trans.Point == sanjiao2[n][1][2]])), red,  3)
    
    cv2.line(imagine, (int(df_trans.H[df_trans.Point == common_side[n][0]]), int(df_trans.V[df_trans.Point == common_side[n][0]])), 
             (int(df_trans.H[df_trans.Point == common_side[n][1]]), int(df_trans.V[df_trans.Point == common_side[n][1]])), (0,255,255),3)
draw_triangle(img1, tri1, tri2, 1)

cv2.imshow('img1', img2)


# cv2.destroyAllWindows()

























