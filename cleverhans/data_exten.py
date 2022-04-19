"""
This file is used to augment training examples which are in gray levels.
Input:
x_train: 4D (N*Height*Width*channel) tensorflow training set
y_train: 2D label (N*nb_classes) tensor, nb_classes is the number of classes of the MNIST dataset
img_rows: image height
img_cols: image width
Output: Cons-Def training set including: original and augmented images

Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""#coding=utf-8

import numpy as np
#from cleverhans.pert import pert
import matplotlib.pyplot as plt
import PIL
from PIL import Image

def data_exten(x_train, y_train, train_end, nb_classes=10, img_rows=32, img_cols=32, nchannels=3):
  expansion =6#5 augmentations + 1 original set
  x_tra = np.zeros((expansion*train_end, img_rows, img_cols, nchannels), dtype = np.float32)
  y_tra = np.zeros((expansion*train_end, nb_classes), dtype = np.float32)
  
  for i in range(train_end):#split gray levels into 4 segments, seg0<-->seg1, seg2<-->seg3
      x_train_ii = np.zeros((img_rows, img_cols, nchannels), dtype = np.float32)
      for k in range(nchannels):
          x_train_i=x_train[i,:,:,k]
          xi_unique = np.unique(x_train_i)
          len_xi=len(xi_unique)
          div_4 = len_xi//4
          mod_4 = len_xi%4
          for j in range(div_4):
              xi0 = np.equal(x_train_i, xi_unique[j])
              xi1 = np.equal(x_train_i, xi_unique[j+div_4])
              xi2 = np.equal(x_train_i, xi_unique[j+2*div_4])
              xi3 = np.equal(x_train_i, xi_unique[j+3*div_4])
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+div_4]*xi0
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j]*xi1
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+3*div_4]*xi2
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+2*div_4]*xi3
          for j in range(mod_4):
              xi = np.equal(x_train_i, xi_unique[j+4*div_4])
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+4*div_4]*xi
      x_tra[i,:,:,:] = x_train_ii
#      plt.imshow(x_train_ii)
#      plt.show
      y_tra[i,:] = y_train[i,:]


  for i in range(train_end):#split gray levels into 8 segments, seg0<-->seg1, seg2<-->seg3,  ..., seg6<-->seg7
      x_train_ii = np.zeros((img_rows, img_cols, nchannels), dtype = np.float32)
      for k in range(nchannels):
          x_train_i=x_train[i,:,:,k]
          xi_unique = np.unique(x_train_i)
          len_xi=len(xi_unique)
          div_4 = len_xi//8
          mod_4 = len_xi%8
          for j in range(div_4):
              xi0 = np.equal(x_train_i, xi_unique[j])#0
              xi1 = np.equal(x_train_i, xi_unique[j+div_4])
              xi2 = np.equal(x_train_i, xi_unique[j+2*div_4])
              xi3 = np.equal(x_train_i, xi_unique[j+3*div_4])
              xi4 = np.equal(x_train_i, xi_unique[j+4*div_4])
              xi5 = np.equal(x_train_i, xi_unique[j+5*div_4])
              xi6 = np.equal(x_train_i, xi_unique[j+6*div_4])
              xi7 = np.equal(x_train_i, xi_unique[j+7*div_4])
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+div_4]*xi0
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j]*xi1
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+3*div_4]*xi2
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+2*div_4]*xi3
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+5*div_4]*xi4
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+4*div_4]*xi5
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+7*div_4]*xi6
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+6*div_4]*xi7
          for j in range(mod_4):
              xi = np.equal(x_train_i, xi_unique[j+8*div_4])
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+8*div_4]*xi
      x_tra[train_end+i,:,:,:] = x_train_ii
      y_tra[train_end+i,:] = y_train[i,:]

  for i in range(train_end):#split gray levels into 16 segments, seg0<-->seg1, seg2<-->seg3, ..., seg14<-->seg15
      x_train_ii = np.zeros((img_rows, img_cols, nchannels), dtype = np.float32)
      for k in range(nchannels):
          x_train_i=x_train[i,:,:,k]
          xi_unique = np.unique(x_train_i)
          len_xi=len(xi_unique)
          div_4 = len_xi//16
          mod_4 = len_xi%16
          for j in range(div_4):
              xi0 = np.equal(x_train_i, xi_unique[j])#0
              xi1 = np.equal(x_train_i, xi_unique[j+div_4])
              xi2 = np.equal(x_train_i, xi_unique[j+2*div_4])
              xi3 = np.equal(x_train_i, xi_unique[j+3*div_4])
              xi4 = np.equal(x_train_i, xi_unique[j+4*div_4])
              xi5 = np.equal(x_train_i, xi_unique[j+5*div_4])
              xi6 = np.equal(x_train_i, xi_unique[j+6*div_4])
              xi7 = np.equal(x_train_i, xi_unique[j+7*div_4])
              xi8 = np.equal(x_train_i, xi_unique[j]+8*div_4)
              xi9 = np.equal(x_train_i, xi_unique[j+9*div_4])
              xi10 = np.equal(x_train_i, xi_unique[j+10*div_4])
              xi11 = np.equal(x_train_i, xi_unique[j+11*div_4])
              xi12 = np.equal(x_train_i, xi_unique[j+12*div_4])
              xi13 = np.equal(x_train_i, xi_unique[j+13*div_4])
              xi14 = np.equal(x_train_i, xi_unique[j+14*div_4])
              xi15 = np.equal(x_train_i, xi_unique[j+15*div_4])
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+div_4]*xi0
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j]*xi1
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+3*div_4]*xi2
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+2*div_4]*xi3
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+5*div_4]*xi4
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+4*div_4]*xi5
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+7*div_4]*xi6
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+6*div_4]*xi7
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+9*div_4]*xi8
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+8*div_4]*xi9
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+11*div_4]*xi10
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+10*div_4]*xi11
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+13*div_4]*xi12
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+12*div_4]*xi13
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+15*div_4]*xi14
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+14*div_4]*xi15
          for j in range(mod_4):
              xi = np.equal(x_train_i, xi_unique[j+16*div_4])
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+16*div_4]*xi
      x_tra[2*train_end+i,:,:,:] = x_train_ii
      y_tra[2*train_end+i,:] = y_train[i,:]

  for i in range(train_end):#split gray levels into 16 segments, seg0<-->seg1, seg2<-->seg3, ..., seg30<-->seg31
      x_train_ii = np.zeros((img_rows, img_cols, nchannels), dtype = np.float32)
      for k in range(nchannels):
          x_train_i=x_train[i,:,:,k]
          xi_unique = np.unique(x_train_i)
          len_xi=len(xi_unique)
          n_sge = 32
          div_4 = len_xi//n_sge
          mod_4 = len_xi%n_sge
          xi0 = np.zeros(((n_sge,) + np.shape(x_train_i)),dtype=np.bool)
          for j in range(div_4):
              for m in range(n_sge):
                  xi0[m,:,:] = np.equal(x_train_i, xi_unique[m*div_4+j])#0
              m = 0
              while m < n_sge:
                  x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[(m+1)*div_4+j]*xi0[m,:,:]
                  x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[m*div_4+j]*xi0[m+1,:,:]
                  m = m+2
          for j in range(mod_4):
              xi = np.equal(x_train_i, xi_unique[j+n_sge*div_4])
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+n_sge*div_4]*xi
      x_tra[3*train_end+i,:,:,:] = x_train_ii
#      plt.imshow(x_train_ii[:,:,0])
#      plt.show
      y_tra[3*train_end+i,:] = y_train[i,:]
  
  for i in range(train_end):
      x_train_ii = np.zeros((img_rows, img_cols, nchannels), dtype = np.float32)
      for k in range(nchannels):
          x_train_i=x_train[i,:,:,k]
          xi_unique = np.unique(x_train_i)
          len_xi=len(xi_unique)
          div_4 = len_xi//2
          mod_4 = len_xi%2
          for j in range(div_4):
              xi0 = np.equal(x_train_i, xi_unique[2*j])#0
              xi1 = np.equal(x_train_i, xi_unique[2*j+1])
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j+1]*xi0
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[j]*xi1
          if mod_4:
              xi = np.equal(x_train_i, xi_unique[div_4*2])
              x_train_ii[:,:,k] = x_train_ii[:,:,k] + xi_unique[div_4*2]*xi
      x_tra[4*train_end+i,:,:,:] = x_train_ii
      y_tra[4*train_end+i,:] = y_train[i,:]

  x_tra[5*train_end:6*train_end,:,:,:] = x_train
  y_tra[5*train_end:6*train_end,:] = y_train
  
  x_train = x_tra
  y_train = y_tra
  return x_train, y_train
