"""
This file is used to augment training examples which are in RGB 3 channels.
The function in this file is almost the same as that in the file of data_exten_multipro.py except the output


Input:
x_train: 4D (N*Height*Width*nchannels) tensorflow training set
y_train: 2D label (N*nb_classes) tensor, nb_classes is the number of classes of the dataset
img_rows: image height
img_cols: image width
Output: return_res for parallel multiple-core processing.
Input: example_0, example_1,....
Output: augmatation of example_0[0], augmatation of example_0[1], ..., augmatation of example_0[per_range-1], augmatation of example_1[0],
augmatation of example_1[1], ..., augmatation of example_1[per_range-1], ...example_0, example_1,....
Note: The output append the original set at the end of it.

Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""#coding=utf-8
import numpy as np
from cleverhans.pert8_16_32_64 import pert

def data_exten(x_train, y_train, train_end, base_range=3, nb_classes=10, img_rows=32, img_cols=32, nchannels=3):

  per_range = base_range**nchannels
  x_tra = np.zeros(((1+per_range)*train_end, img_rows, img_cols, nchannels), dtype = np.float32)
  y_tra = np.zeros(((1+per_range)*train_end, nb_classes), dtype = np.float32)
  xi_pert = np.zeros((per_range, img_rows, img_cols, nchannels),dtype=np.float32)
  yi_pert = np.zeros((per_range, nb_classes),dtype=np.float32)

  xi_pertr = np.zeros((base_range, img_rows, img_cols),dtype=np.float32)
  xi_pertg = np.zeros((base_range, img_rows, img_cols),dtype=np.float32)
  xi_pertb = np.zeros((base_range, img_rows, img_cols),dtype=np.float32)
  for i in range(train_end):
#      if np.mod(i,50)==0:
#          print("processing the i-th image:{}".format(i))
      xi = x_train[i,:,:,:]
      for r in range(base_range):
          pertkchannel = xi[:,:,0]
#          pertkchannel = pertkchannel.reshape(pertkchannel.shape+(1,))
          xi_pertr[r,:,:] = pert(np.ascontiguousarray(pertkchannel, dtype=np.double), r, img_rows=img_rows, img_cols=img_cols)
          
      for g in range(base_range):
          pertkchannel = xi[:,:,1]
#        pertkchannel = pertkchannel.reshape(pertkchannel.shape+(1,))
          xi_pertg[g,:,:] = pert(np.ascontiguousarray(pertkchannel, dtype=np.double), g, img_rows=img_rows, img_cols=img_cols)
      for b in range(base_range):
          pertkchannel = xi[:,:,2]
#          pertkchannel = pertkchannel.reshape(pertkchannel.shape+(1,))
          xi_pertb[b,:,:] = pert(np.ascontiguousarray(pertkchannel, dtype=np.double), b, img_rows=img_rows, img_cols=img_cols)
      for r in range(base_range):
          for g in range(base_range):
              for b in range(base_range):
                  bat_ind = r*base_range*base_range+g*base_range+b
                  xi_pert[bat_ind,:,:,0] = xi_pertr[r,:,:]
                  xi_pert[bat_ind,:,:,1] = xi_pertg[g,:,:]
                  xi_pert[bat_ind,:,:,2] = xi_pertb[b,:,:]
      x_tra[i*per_range:(i+1)*per_range,:,:,:] = xi_pert
      for k in range(per_range):
          yi_pert[k,:] = y_train[i,:]
      y_tra[i*per_range:(i+1)*per_range,:] = yi_pert
 
  x_tra[per_range*train_end:(1+per_range)*train_end,:,:,:] = x_train
  y_tra[per_range*train_end:(1+per_range)*train_end,:] = y_train
  x_train = x_tra
  y_train = y_tra

  return x_train, y_train
