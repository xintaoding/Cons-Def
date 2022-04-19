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

  x_train_ii = np.zeros((img_rows, img_cols), dtype = np.float32)

  if per_label==0:#split gray levels into 8 segments, seg0<-->seg1, seg2<-->seg3
          xi_unique = np.unique(x_train)
          len_xi=len(xi_unique)
          div_4 = len_xi//8
          mod_4 = len_xi%8
          for j in range(div_4):
              xi0 = np.equal(x_train, xi_unique[j])#0
              xi1 = np.equal(x_train, xi_unique[j+div_4])
              xi2 = np.equal(x_train, xi_unique[j+2*div_4])
              xi3 = np.equal(x_train, xi_unique[j+3*div_4])
              xi4 = np.equal(x_train, xi_unique[j+4*div_4])
              xi5 = np.equal(x_train, xi_unique[j+5*div_4])
              xi6 = np.equal(x_train, xi_unique[j+6*div_4])
              xi7 = np.equal(x_train, xi_unique[j+7*div_4])
              x_train_ii = x_train_ii + xi_unique[j+div_4]*xi0
              x_train_ii = x_train_ii + xi_unique[j]*xi1
              x_train_ii = x_train_ii + xi_unique[j+3*div_4]*xi2
              x_train_ii = x_train_ii + xi_unique[j+2*div_4]*xi3
              x_train_ii = x_train_ii + xi_unique[j+5*div_4]*xi4
              x_train_ii = x_train_ii + xi_unique[j+4*div_4]*xi5
              x_train_ii = x_train_ii + xi_unique[j+7*div_4]*xi6
              x_train_ii = x_train_ii + xi_unique[j+6*div_4]*xi7
          for j in range(mod_4):
              xi = np.equal(x_train, xi_unique[j+8*div_4])
              x_train_ii = x_train_ii + xi_unique[j+8*div_4]*xi

  if per_label==1:#split gray levels into 16 segments, seg0<-->seg1, seg2<-->seg3
          xi_unique = np.unique(x_train)
          len_xi=len(xi_unique)
          n_sge = 16
          div_4 = len_xi//n_sge
          mod_4 = len_xi%n_sge
          xi0 = np.zeros(((n_sge,) + np.shape(x_train)),dtype=np.bool)
          for j in range(div_4):
              for m in range(n_sge):
                  xi0[m,:,:] = np.equal(x_train, xi_unique[m*div_4+j])#0
              m = 0
              while m < n_sge:
                  x_train_ii = x_train_ii + xi_unique[(m+1)*div_4+j]*xi0[m,:,:]
                  x_train_ii = x_train_ii + xi_unique[m*div_4+j]*xi0[m+1,:,:]
                  m = m+2
          for j in range(mod_4):
              xi = np.equal(x_train, xi_unique[j+n_sge*div_4])
              x_train_ii = x_train_ii + xi_unique[j+n_sge*div_4]*xi

  if per_label==2:#split gray levels into 32 segments
          xi_unique = np.unique(x_train)
          len_xi=len(xi_unique)
          n_sge = 32#n_sge is the number of blocks
          div_4 = len_xi//n_sge#div_4 is the block length for intensity exchange
          mod_4 = len_xi%n_sge
          xi0 = np.zeros(((n_sge,) + np.shape(x_train)),dtype=np.bool)
          for j in range(div_4):
              for m in range(n_sge):
                  xi0[m,:,:] = np.equal(x_train, xi_unique[m*div_4+j])#0
              m = 0
              while m < n_sge:
                  x_train_ii = x_train_ii + xi_unique[(m+1)*div_4+j]*xi0[m,:,:]
                  x_train_ii = x_train_ii + xi_unique[m*div_4+j]*xi0[m+1,:,:]
                  m = m+2
          for j in range(mod_4):
              xi = np.equal(x_train, xi_unique[j+n_sge*div_4])
              x_train_ii = x_train_ii + xi_unique[j+n_sge*div_4]*xi


  if per_label==3:#split gray levels into 64 segments
          xi_unique = np.unique(x_train)
          len_xi=len(xi_unique)
          n_sge = 64
          div_4 = len_xi//n_sge
          mod_4 = len_xi%n_sge
          xi0 = np.zeros(((n_sge,) + np.shape(x_train)),dtype=np.bool)
          for j in range(div_4):
              for m in range(n_sge):
                  xi0[m,:,:] = np.equal(x_train, xi_unique[m*div_4+j])#0
              m = 0
              while m < n_sge:
                  x_train_ii = x_train_ii + xi_unique[(m+1)*div_4+j]*xi0[m,:,:]
                  x_train_ii = x_train_ii + xi_unique[m*div_4+j]*xi0[m+1,:,:]
                  m = m+2
          for j in range(mod_4):
              xi = np.equal(x_train, xi_unique[j+n_sge*div_4])
              x_train_ii = x_train_ii + xi_unique[j+n_sge*div_4]*xi
      
  return x_train_ii
