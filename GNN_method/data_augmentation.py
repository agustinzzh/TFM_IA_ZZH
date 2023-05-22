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
        file_list = []
        
        for i in os.listdir(file_path):
            file_list.append(os.path.join(file_path, i))
        
        names = locals()
        list_df = []
        
        for j in range(len(file_list)):
            names['df_%s' % j] = pd.read_table(file_list[j], sep='\t', skiprows=2)
            list_df.append(names['df_%s' % j])
            
        return list_df
    
    # in this case only consider the points with PT marked (not NA or not -1)
    def trans_df_data(self):
        list_df = self.get_file()
        names = locals()
        
        list_df_trans = []
        for i in range(len(list_df)):
            df = list_df[i]
            # get ride of NA and -1
            df = df.dropna(subset=['PT'])
            df = df[df.PT >= 0]
            df = df[df.PT <= 5]
            #make sure PT has only 4 values
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
            df_trans = df_trans.sort_values(by='PT', ascending=True)
            df_trans = df_trans.reset_index(drop=True)

            if len(df_trans) != 0:
                if len(df_trans)% 4 ==0:
                    names['df_trans_%s'%i] = df_trans
                    
                    list_df_trans.append(names['df_trans_%s'%i])

        return list_df_trans

    def get_distance(self, point1, point2):

        return math.sqrt((point2[0]- point1[0])**2 + (point2[1]- point1[1])**2)

#绕点（cx，cy）旋转
    def rotate_data(self, list_x, list_y, x0, y0, distance):
        flag = False

        n = 0
        while flag == False:
            print(n)
            
            cx = x0 + random.uniform(-distance, distance)
            cy = y0 + random.uniform(-distance, distance)
            angle = random.uniform(0,360)*np.pi/ 180
            list_new_x = []
            list_new_y = []
            for i in range(len(list_x)):
                x = list_x[i]
                y = list_y[i]
            
                x_new = (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle) + cx
                y_new = (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle) + cy
            
                list_new_x.append(x_new)
                list_new_y.append(y_new)
            
            outrange_x = [j for j in range(len(list_new_x)) if list_new_x[j] >= 4000-150 or list_new_x[j] <= 0]
            outrange_y = [k for k in range(len(list_new_y)) if list_new_y[k] >= 3000-150 or list_new_y[k] <= 0]
            print(list_new_x)

            if outrange_x == [] and outrange_y ==[]:
                flag = True
            n = n+1

        return list_new_x, list_new_y
    
    def data_augmentation(self, list_df, num_augmentation, distance):

        list_df_augmentated = []
        for j in range(num_augmentation):
            print(j)
            for i in range(len(list_df)):
                flag2 = False
                while flag2 == False:
                    df = list_df[i]
                    x_list_total = []
                    y_list_total = []
                    PT_list = []
                    x0y0_list = []


                    for PT in df.PT.unique():
                        df_PT = df[df.PT == PT]
                        x_list = df_PT.H.values.tolist()
                        y_list = df_PT.V.values.tolist()

                        x0 = sum(x_list)/4
                        y0 = sum(x_list)/4

                        list_new_x, list_new_y = self.rotate_data(x_list, y_list, x0, y0, distance)

                        x0_rotated = sum(list_new_x)/4
                        y0_rotated = sum(list_new_y)/4
                        x0y0 = [x0_rotated, y0_rotated]
                        x0y0_list.append(x0y0)

                        x_list_total = x_list_total + list_new_x
                        y_list_total = y_list_total + list_new_y
                        PT_list = PT_list + [PT] * 4
                        
                    nbrs = NearestNeighbors(n_neighbors=2, metric=self.get_distance).fit(x0y0_list)
                    # 这n个点之间的距离及其index
                    distances, indices = nbrs.kneighbors(x0y0_list)
                    min_dis = distances[:, 1].min()
# to avoid markers overlap
                    if min_dis >= 360:

                        df_augmented = pd.DataFrame()
                        df_augmented['H'] = x_list_total
                        df_augmented['V'] = y_list_total
                        df_augmented['PT'] = PT_list
                        list_df_augmentated.append(df_augmented)
                        flag2 = True
        return list_df_augmentated
                
                
# =============================================================================
# 
# =============================================================================
    
d = 2
n = 5    
path_ref = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Dato_prueba/180MUA_Formatted_F.txt' 
file_path = '/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion'
ref = get_reference(path_ref, d)
df_feature_ref, df_points_ref = ref.get_invariant_ref()
df_trans_ref = ref.trans_df()
realdata = get_real_data(file_path,df_feature_ref, d, n)
list_df_trans = realdata.trans_df_data()
list_df_augmentated = realdata.data_augmentation(list_df_trans, 30, 100 )

for i in range(len(list_df_augmentated)):
    df = list_df_augmentated[i]
    df.to_csv(os.path.join('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_augmented', 'df'+'_'+str(i)), sep='\t', index=False)




import cv2
from matplotlib import pyplot as plt

n = int(random.uniform(0, len(list_df_augmentated)))
df_trans = list_df_augmentated[n]

x = df_trans['H'][0]
y = df_trans['V'][0]
r = 22

newImg = np.zeros((3001, 4001, 3), np.uint8)
newImg.fill(0)
cv2.circle(newImg, (int(x), int(y)), int(r), (255, 255, 255))


for i in range(6):
    listpoint = df_trans[['H', 'V' ]][df_trans.PT == i].values.tolist()
    for (x, y) in listpoint:
        cv2.circle(newImg, (int(x), int(y)), int(r), (i*40, 255-i*40, i*45), 3)
        cv2.putText(newImg, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (i*40, 255-i*40, i*45), 3)

cv2.line(newImg, (0, 1501), (4001, 1501), (255, 255, 255), 2)
cv2.line(newImg, (2001, 0), (2001, 3001), (255, 255, 255), 2)



Img_origin = np.zeros((3001,4001,3), np.uint8)
Img_origin.fill(0)

df_trans_origin = list_df_trans[n - (n//len(list_df_trans)* len(list_df_trans))]

for i in range(6):
    listpoint = df_trans_origin[['H', 'V' ]][df_trans_origin.PT == i].values.tolist()
    for (x, y) in listpoint:
        cv2.circle(Img_origin, (int(x),int(y)), int(r), (i*40,255-i*40,i*45), 3)
        cv2.putText(Img_origin, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (i*40,255-i*40,i*45), 3)

cv2.line(Img_origin, (0,1501), (4001,1501), (255,255,255), 2)
cv2.line(Img_origin, (2001,0), (2001,3001), (255,255,255), 2)

while(True):
    cv2.imshow('origin', Img_origin)
    cv2.imshow('new', newImg)
    if cv2.waitKey(5) & 0xff == ord('q'):
        break
#cv2.destroyAllWindows()
#plt.subplot(1,1,1)
#plt.imshow(newImg)
#plt.title('New_img')


#plt.show()

















