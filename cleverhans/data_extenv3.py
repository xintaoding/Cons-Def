import numpy as np
#from cleverhans.pert8_16_32 import pert
#from cleverhans.pert8_16_32_64 import pert
from cleverhans.pert8_16_32 import pert
import multiprocessing
import time

def data_exten(x_train, y_train, train_end, base_range=3, nb_classes=10, img_rows=32, img_cols=32, nchannels=3):

  per_range = base_range**nchannels
  x_tra = np.zeros(((1+per_range)*train_end, img_rows, img_cols, nchannels), dtype = np.float32)
  y_tra = np.zeros(((1+per_range)*train_end, nb_classes), dtype = np.float32)

  time1=time.time()
  cores=multiprocessing.cpu_count()
  pool=multiprocessing.Pool(processes=cores)
  a=np.linspace(0,train_end-1,train_end)
  a=a.astype(np.uint8)

  def deres_data(i):
      xi_pert = np.zeros((per_range, img_rows, img_cols, nchannels),dtype=np.float32)
      yi_pert = np.zeros((per_range, nb_classes),dtype=np.float32)
      if np.mod(i,2)==0:
          print("processing the i-th image:{}".format(i))
      xi = x_train[i,:,:,:]
      for r in range(base_range):
          pertkchannel = xi[:,:,0]
#          pertkchannel = pertkchannel.reshape(pertkchannel.shape+(1,))
          xi_pertr = pert(np.ascontiguousarray(pertkchannel, dtype=np.double), r, img_rows=img_rows, img_cols=img_cols)
          for g in range(base_range):
                    pertkchannel = xi[:,:,1]
#                    pertkchannel = pertkchannel.reshape(pertkchannel.shape+(1,))
                    xi_pertg = pert(np.ascontiguousarray(pertkchannel, dtype=np.double), g, img_rows=img_rows, img_cols=img_cols)
                    for b in range(base_range):
                        pertkchannel = xi[:,:,2]
#                        pertkchannel = pertkchannel.reshape(pertkchannel.shape+(1,))
                        xi_pertb = pert(np.ascontiguousarray(pertkchannel, dtype=np.double), b, img_rows=img_rows, img_cols=img_cols)
                        bat_ind = r*base_range*base_range+g*base_range+b
                        xi_pert[bat_ind,:,:,0] = xi_pertr[:,:]
                        xi_pert[bat_ind,:,:,1] = xi_pertg[:,:]
                        xi_pert[bat_ind,:,:,2] = xi_pertb[:,:]
      for k in range(per_range):
          yi_pert[k,:] = y_train[i,:]
      return(xi_pert,yi_pert)
  a=[(0,)]
  res=pool.map(deres_data,a)
  time2=time.time()
  print("time cost is: {}".format(time2-time1))
  for i in range(train_end):
      x_tra[i*per_range:(i+1)*per_range,:,:,:]=res[i][0]
      y_tra[i*per_range:(i+1)*per_range,:]=res[i][1]
  
  x_tra[per_range*train_end:(1+per_range)*train_end,:,:,:] = x_train
  y_tra[per_range*train_end:(1+per_range)*train_end,:] = y_train
  x_train = x_tra
  y_train = y_tra

  return x_train, y_train
