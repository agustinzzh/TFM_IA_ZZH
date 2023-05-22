
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 08:33:16 2022

@author: agustinzhang
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
#df1 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion/IMG_2968.txt', sep = '\t', skiprows = 2)
#df1 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Datos_asociacion/IMG_2926.txt', sep = '\t', skiprows = 2)
df1 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Dato_prueba/IMG_2922.txt', sep = '\t', skiprows = 2)
df2 = pd.read_table('/Users/agustinzhang/Downloads/master_AI/TFM/Dato/Dato_prueba/180MUA_Formatted_F.txt', sep = '\t', header = None)
df2 = df2.dropna(axis = 0, how = 'any')

catlist = []
for n in range(0, 10):
    for i in range(0,8):
        catlist.append(n)
        
df2['5'] = catlist
df2.columns = ['D_AB', 'X', 'Y', 'Z', 'r', 'PT']

df_point = df2[~ df2['D_AB'].str.contains('B')]
df_point = df_point[['D_AB', 'X', 'Y', 'r', 'PT']]


listpoint = []
for i in range(10):
    plt.subplot(2, 5, i+1)
    listpoint = df_point[['X', 'Y', 'r']][df_point.PT == i].values.tolist()
    for (x, y, r) in listpoint:
        plt.plot(x, y, 'o', ms = 5)
        index = listpoint.index([x, y, r])
        plt.annotate(index, xy= (x,y), xytext = (x+0.1,y+0.1))
        
    plt.grid()
    x_major_locator = plt.MultipleLocator(1)
    y_major_locator = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    
    
    



df_trans = df1[['H', 'V', 'R', 'C', 'PT']]
df_trans['H'] = df_trans.H+2000.5
df_trans['V'] = df_trans.V+1500.5

x = df_trans['H'][0]
y = df_trans['V'][0]
r = df_trans['R'][0]


newImg = np.zeros((3001,4001,3), np.uint8)
newImg.fill(0)



for i in range(6):
    listpoint = df_trans[['H', 'V', 'R']][df_trans.PT == i].values.tolist()
    for (x, y, r) in listpoint:
        cv2.circle(newImg, (int(x),int(y)), int(r), (i*40,255-i*40,i*45), 3)
        cv2.putText(newImg, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (i*40,255-i*40,i*45), 3)

cv2.line(newImg, (0,1501), (4001,1501), (255,255,255), 2)
cv2.line(newImg, (2001,0), (2001,3001), (255,255,255), 2)
cv2.imshow('newimg', newImg)
















