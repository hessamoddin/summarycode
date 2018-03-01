# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 08:14:24 2018

@author: hessam
"""
import matplotlib.pyplot as plt
import os
import numpy as np
from os.path import basename
from glob import glob


def DRM_Plotter(csv_file):
    
    base_csv_file=basename(os.path.splitext(csv_file)[0])
    if('MAPLE' in base_csv_file):
        line_color='c'
    elif('Cordova' in base_csv_file):
        line_color='b'
    else:
        line_color='r'

    
    data = np.genfromtxt(csv_file, delimiter=',', names=['x', 'y'])
    num_points=data.shape[0]
    X=[]
    y=[]
    for i in xrange(num_points):
        current_row=data[i]
        X.append(current_row[0])
        y.append(current_row[1])
        
        
        
        
        
        
    fig, ax = plt.subplots()
    plt.xlabel('UTC', fontsize=16, fontweight='bold')
    plt.ylabel('SNR', fontsize=16, fontweight='bold')



    plt.xlabel('UTC', fontsize=16, fontweight='bold')
    plt.ylabel('SNR', fontsize=16, fontweight='bold')

    plt.xticks(np.arange(0, 25, 4))

    #plt.title(base_csv_file)
    plt.ylim( (-1, 35) ) 
    plt.xlim( (-1, 25) ) 
    ax.grid(True)
    ax.margins(1) # 5% padding in all directions
    ax.tick_params(axis='x', which='major', pad=15)



    lines=ax.plot(X,y)
    plt.setp(lines, linewidth=3, color=line_color)
    plt.tight_layout()

    plt.show()
    file_path=os.path.splitext(csv_file)[0] 
    fig.savefig(base_csv_file+'.jpeg',dpi=1200)
    return plt

DRM_folder='/home/hessam/Downloads/DRM/'
subfolders_list=glob(DRM_folder+'*/')


for current_subfolder in subfolders_list:
    print(current_subfolder)
    csv_files=glob(current_subfolder+'*.csv')
    for csv_file in csv_files:
        plt=DRM_Plotter(csv_file)



        

