#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:23:08 2020

@author: Xintao Ding
x_train: component image in size of img_rows*img_cols
per_label: used to direct the split of the original intensity list
output:
x_train_ii: one intensity exchanged component image
"""

import numpy as np
#import matplotlib.pyplot as plt
#import PIL
#from PIL import Image

def pert(x_train, per_label, nb_classes=10, img_rows=32, img_cols=32):
#split gray levels into 8,16,32,64 segments, seg0<-->seg1, seg2<-->seg3 depend on per_label
    xi_sort = np.unique(x_train)
    len_xi=len(xi_sort)
    n_sge = 2**(3+per_label)
    div_4 = len_xi//n_sge
#    mod_4 = len_xi%n_sge
    for i in range(img_rows):
        for j in range(img_cols):
            intij=x_train[i,j]
            for k in range(n_sge):
                if intij>=xi_sort[k*div_4] and intij<=xi_sort[(k+1)*div_4-1]:
                    for m in range(div_4):
                        if intij==xi_sort[k*div_4+m]:
                            break
                    if k%2==0:
                        x_train[i,j]=xi_sort[(k+1)*div_4+m]
                    else:
                        x_train[i,j]=xi_sort[(k-1)*div_4+m]
                    break

    return x_train
