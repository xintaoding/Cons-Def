"""
This tutorial shows how to implement Cons-Def against JSMA white-box attacks.
Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.compat import flags
from cleverhans.dataset import CIFAR10

from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_tf import model_eval, model_argmax

#from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional#Uncomments this, if use the all conv network
from cleverhans.model_zoo.four_conv2FC import Model4Convolutional2FC#using the network of 4 conv + 2 fully connected layers

#Added by Ding
from cleverhans.augmentation import random_horizontal_flip, random_shift
#from tensorflow.python import pywrap_tensorflow


from cleverhans.data_exten_mulpro import data_exten
#from cleverhans.return_detection_label import rt_detec_lab
from sklearn.metrics import roc_curve, roc_auc_score#Added by Ding
import multiprocessing


FLAGS = flags.FLAGS
#VIZ_ENABLED = True
VIZ_ENABLED = False

NB_EPOCHS = 6
BATCH_SIZE = 128
SOURCE_SAMPLES = 1000
NB_FILTERS = 64

def cifar10_tutorial(train_start=0, train_end=50000, test_start=0,
                     test_end=1000, viz_enabled=VIZ_ENABLED, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                     source_samples=SOURCE_SAMPLES,
                     nb_filters=NB_FILTERS, num_threads=None):
  """
  CIFAR10 cleverhans tutorial
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param clean_train: perform normal training on clean examples only
                      before performing adversarial training.
  :param backprop_through_attack: If True, backprop through adversarial
                                  example construction process during
                                  adversarial training.
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))
  print("Created TensorFlow session.")

  # Get CIFAR10 data
  data = CIFAR10(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
  dataset_size = data.x_train.shape[0]
  dataset_train = data.to_tensorflow()[0]
  dataset_train = dataset_train.map(
      lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
  dataset_train = dataset_train.batch(batch_size)
  dataset_train = dataset_train.prefetch(16)
  x_train, y_train = data.get_set('train')
  x_test, y_test = data.get_set('test')

  # Use Image Parameters
  print("y_train_shape:{},{}".format(x_train.shape,type(y_train)))#########################################
  img_rows, img_cols, nchannels = x_test.shape[1:4]
  nb_classes = y_train.shape[1]
  
  print("nb_classes:{}".format(nb_classes))#########################################

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))
  
  nb_filters = 64

  # Define TF model graph
#  model = ModelAllConvolutional('model1', nb_classes, nb_filters,
#                                input_shape=[32, 32, 3])#Load the model graph of all conv
  net_height=32
  net_width=32
#Load the model graph of 4 conv and 2 fc layers, Revised by Ding==========
  model = Model4Convolutional2FC('model1', nb_classes, nb_filters, input_shape=[net_height, net_width, nchannels])
  preds = model.get_logits(x)
  probs = model.get_probs(x)
  
  print("Defined TensorFlow model graph.")

  #########################################Added by Ding  
  print("aaaaaaaaaaaaaaaaa=+++++++++++++++++++++++++++++")
  saver = tf.train.Saver() 
  print("aaaaaaaaaaaaaaaaa=--------------------------------")
  saver.restore(sess,'models/cifar10/cifar10_dtcnn_train_epoch89')
  print("aaaaaaaaaaaaaaaaa=************************************")
  eval_params = {'batch_size': batch_size}
  accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)

  assert x_test.shape[0] == test_end - test_start, x_test.shape

  ###########################################################################
  # Craft adversarial examples using the Jacobian-based saliency map approach
  ###########################################################################
  print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes - 1) + ' adversarial examples')

  # Keep track of success (adversarial example classified in target)
  results = np.zeros((1, source_samples), dtype='i')
  auc = np.zeros((1, source_samples), dtype='i')
  dsae = np.zeros((1, source_samples), dtype='f')

  # Rate of perturbed features for each test set example and target class
  perturbations = np.zeros((1, source_samples), dtype='f')

  # Initialize our array for grid visualization
  grid_shape = (nb_classes, nb_classes, img_rows, img_cols, nchannels)
  grid_viz_data = np.zeros(grid_shape, dtype='f')

  # Instantiate a SaliencyMapMethod attack object
  jsma = SaliencyMapMethod(model, sess=sess)
  jsma_params = {'theta': 1., 'gamma': 0.1,
                 'clip_min': 0., 'clip_max': 1.,
                 'y_target': None}

  figure = None
  
  #####################################################################
  #Randomly select source_samples samples in x_test
  x_test=x_test[:source_samples,:,:,:]
  y_test=y_test[:source_samples]
  adv=np.zeros((source_samples, img_rows, img_cols, 3),dtype='float32')
  print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx_test:{}",x_test.shape)
  #####################################################################
  
  # Loop over the samples we want to perturb into adversarial examples
  for sample_ind in xrange(0, source_samples):
#    print('--------------------------------------')
    if sample_ind%10==0:
        print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
    sample = x_test[sample_ind:(sample_ind + 1)]

    # We want to find an adversarial example for each possible target class
    # (i.e. all classes that differ from the label given in the dataset)
    current_class = int(np.argmax(y_test[sample_ind]))
    target_classes = other_classes(nb_classes, current_class)
    target = target_classes[np.random.randint(0,len(target_classes),1)[0]]
    
    # For the grid visualization, keep original images along the diagonal
    grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
        sample, (img_rows, img_cols, nchannels))

    # Loop over all target classes
#    print('Generating adv. example for target class %i' % target)

    # This call runs the Jacobian-based saliency map approach
    one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
    one_hot_target[0, target] = 1
    jsma_params['y_target'] = one_hot_target
    adv_x = jsma.generate_np(sample, **jsma_params)
    
    adv[sample_ind,:,:,:]=adv_x
###################################################
    adv_x1=adv_x[0,:,:]
    sample1=sample[0,:,:]
    dsaeij=(np.sum((adv_x1 - sample1)**2)**.5)/adv_x1.shape[0]/adv_x1.shape[1]
#    print("==============Avg. distortion of successful adversarial examples wether it can or cannot fool the model:{}".format(dsaeij))
###################################################


  # Check if success was achieved
    netpred = model_argmax(sess, x, preds, adv_x)
    res = int(netpred == target)
    acc = int(netpred == current_class)

    # Compute number of modified features
    adv_x_reshape = adv_x.reshape(-1)
    test_in_reshape = x_test[sample_ind].reshape(-1)
    nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
    percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]
#    print("nb_changed:{}".format(nb_changed))

      # Display the original and adversarial images side-by-side
    if viz_enabled:
        figure = pair_visual(
            np.reshape(sample, (img_rows, img_cols, nchannels)),
            np.reshape(adv_x, (img_rows, img_cols, nchannels)), figure)

      # Add our adversarial example to our grid data
    grid_viz_data[target, current_class, :, :, :] = np.reshape(
          adv_x, (img_rows, img_cols, nchannels))

    # Update the arrays for later analysis
    results[0, sample_ind] = res
    auc[0, sample_ind] = acc
    dsae[0, sample_ind] = dsaeij
    adv[sample_ind,:,:,:] = adv_x1
#Matrix, For an i-th targeted attack, perturbations[i,j] is the ratio of attacked pixels to the size of the j-th sample.
    perturbations[0, sample_ind] = percent_perturb
  print('--------------------------------------')
  np.save("cifar10_DistillationNet_augmodel_jsma_1000adv",adv)#save advs produced on clean model:
#  adv=np.load("cifar10_DistillationNet_augmodel_jsma_1000adv.npy")

  print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
  # Compute the number of adversarial examples that were successfully found
  nb_targets_tried = source_samples
  succ_rate = float(np.sum(results)) / nb_targets_tried
  acc = float(np.sum(auc)) / nb_targets_tried
  print("Test accuracy on adversarial examples: {}".format(acc))
  print('Avg. rate of successful adv. examples / ASR {}'.format(succ_rate))
  print('For untargeted attack, the number of misclassified examples: sum(results==0):{}'.format(np.sum(results==0)))
  
  wheresae=np.where(dsae>0)#The successful adv. example is different from clean example
  avg_dsae=np.mean(dsae[wheresae])
  print('Avg. L_2 norm of perturbations (Avg. Distortion on successful adv. examples): {0:.4f}'.format(avg_dsae))

  
#  print("77777777777777777777777777:{}".format(results))
  avg_dsae_foolmod=0
  k=0
  for i in xrange(0, source_samples):
          if results[0,i]>0:
              avg_dsae_foolmod=avg_dsae_foolmod+dsae[0,i]
              k=k+1
  if k>0:
      avg_dsae_foolmod=avg_dsae_foolmod/k
  print('The number of successful attack:{}, Avg. Distortion on successful adv. examples that fool model/ D-SAE {}'.format(k, avg_dsae_foolmod))

  # Compute the average distortion introduced by the algorithm
  percent_perturbed = np.mean(perturbations[np.where(perturbations!=0)])
  print('perturbation ratio of the number of pixels that their intensity have been changed to the number of image pixels {}'.format(np.mean(perturbations)))
  print('Avg. perturbation ratio on the dataset {0:.4f}'.format(percent_perturbed))

  # Compute the average distortion introduced for successful samples only
  percent_perturb_succ = np.mean(perturbations[np.where(perturbations!=0)] * (results[np.where(perturbations!=0)] == 1))
  print('Avg. rate of perturbed features for successful '
        'adversarial examples {0:.4f}'.format(percent_perturb_succ))
  
  x_test = np.round(x_test*256)/256.0
  x_test40 = np.zeros((len(x_test),8+img_rows,8+img_rows,3),dtype=np.float32)
  x_testcrop = np.zeros((len(x_test),img_rows,img_cols,3),dtype=np.float32)
  x_test40[:,4:36,4:36,:] = x_test
  x_test40[:, 4:36, :4, :]=x_test[:, :, [3,2,1,0], :]#reflect
  x_test40[:, 4:36, 36:, :]=x_test[:, :, [31,30,29,28], :]
  x_test40[:, :4, :, :]=x_test40[:, [7,6,5,4], :, :]
  x_test40[:, 36:, :, :]=x_test40[:, [35,34,33,32], :, :]
  adv = np.round(adv*256)/256.0
  adv40 = np.zeros((len(adv),8+img_rows,8+img_rows,3),dtype=np.float32)
  advcrop = np.zeros((len(adv),img_rows,img_cols,3),dtype=np.float32)
  adv40[:, 4:36, 4:36, :] = adv
  adv40[:, 4:36, :4, :]=adv[:, :, [3,2,1,0], :]#reflect
  adv40[:, 4:36, 36:, :]=adv[:, :, [31,30,29,28], :]
  adv40[:, :4, :, :]=adv40[:, [7,6,5,4], :, :]
  adv40[:, 36:, :, :]=adv40[:, [35,34,33,32], :, :]
  for i in range(len(adv)):
      tf_image = adv40[i,:,:,:]
      test_image = x_test40[i,:,:,:]
      lu1 = np.random.randint(0,8)
      lu2 = np.random.randint(0,8)
      advcrop[i,:,:,:] = tf_image[lu1:lu1+img_rows,lu2:lu2+img_cols,:]
      x_testcrop[i,:,:,:] = test_image[lu1:lu1+img_rows,lu2:lu2+img_cols,:]
  adv = advcrop
  x_test = x_testcrop

#  np.save("cifar10_DistillationNet_augmodel_fgsmlinf_1000adv",adv[:1000,:,:,:])#save advs produced on augmented model:
    
  batch_size = 500  #
  base_range=4
  n_pert = base_range**nchannels
  ext_bat = n_pert+1
    
  logits_ext = np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  logits_adv_ext = np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  test_prob_pertpart=np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  adv_prob_pertpart=np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  y_test_pertpart = np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  y_adv_pertpart = np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  x_adv_pertpart = np.zeros((batch_size*n_pert*2,net_height,net_width,nchannels),dtype=np.float32)
  x_test_pertpart = np.zeros((batch_size*n_pert*2,net_height,net_width,nchannels),dtype=np.float32)
  val_max_steps = int(len(adv) / batch_size/2)
    
  adv_prob_legit = np.zeros((test_end,nb_classes),dtype=np.float32)
  test_prob_legit = np.zeros((test_end,nb_classes),dtype=np.float32)
    
  manager=multiprocessing.Manager()
  
  for i in range(val_max_steps):
      rt_res_adv1=manager.dict()
      rt_res_adv2=manager.dict()
      rt_res_test1=manager.dict()
      rt_res_test2=manager.dict()
      p1 = multiprocessing.Process(target=data_exten,args=(adv[i*2*batch_size:(2*i+1)*batch_size,:,:,:],
                                            y_test[2*i*batch_size:(2*i+1)*batch_size,:], 
                                            batch_size, base_range,nb_classes,net_height, net_width,nchannels,
                                            rt_res_adv1))
      p2 = multiprocessing.Process(target=data_exten,args=(adv[(2*i+1)*batch_size:2*(i+1)*batch_size,:,:,:],
                                            y_test[(2*i+1)*batch_size:2*(i+1)*batch_size,:], 
                                            batch_size, base_range,nb_classes,net_height, net_width,nchannels,
                                            rt_res_adv2))
      p3 = multiprocessing.Process(target=data_exten,args=(x_test[2*i*batch_size:(2*i+1)*batch_size,:,:,:],
                                            y_test[2*i*batch_size:(2*i+1)*batch_size,:], 
                                            batch_size, base_range,nb_classes,net_height, net_width,nchannels,
                                            rt_res_test1))
      p4 = multiprocessing.Process(target=data_exten,args=(x_test[(2*i+1)*batch_size:2*(i+1)*batch_size,:,:,:],
                                            y_test[(2*i+1)*batch_size:2*(i+1)*batch_size,:], 
                                            batch_size, base_range,nb_classes,net_height, net_width,nchannels,
                                            rt_res_test2))
      p1.start()
      p2.start()
      p3.start()
      p4.start()
      p1.join()
      x_adv_extended1, y_adv_extended1 = rt_res_adv1.values()
      p2.join()
      x_adv_extended2, y_adv_extended2 = rt_res_adv2.values()
      p3.join()
      x_test_extended1, y_test_extended1 = rt_res_test1.values()
      p4.join()
      x_test_extended2, y_test_extended2 = rt_res_test2.values()
      
      
      x_adv_pertpart[:batch_size*n_pert,:,:,:] = x_adv_extended1[:batch_size*n_pert,:,:,:]
      x_adv_pertpart[batch_size*n_pert:2*batch_size*n_pert,:,:,:] = x_adv_extended2[:batch_size*n_pert,:,:,:]
      y_adv_pertpart[2*i*batch_size*n_pert:(2*i+1)*batch_size*n_pert,:] = y_adv_extended1[:batch_size*n_pert,:]
      y_adv_pertpart[(2*i+1)*batch_size*n_pert:2*(i+1)*batch_size*n_pert,:] = y_adv_extended2[:batch_size*n_pert,:]
                     
      x_test_pertpart[:batch_size*n_pert,:,:,:] = x_test_extended1[:batch_size*n_pert,:,:,:]
      x_test_pertpart[batch_size*n_pert:2*batch_size*n_pert,:,:,:] = x_test_extended2[:batch_size*n_pert,:,:,:]
      y_test_pertpart[2*i*batch_size*n_pert:(2*i+1)*batch_size*n_pert,:] = y_test_extended1[:batch_size*n_pert,:]
      y_test_pertpart[(2*i+1)*batch_size*n_pert:2*(i+1)*batch_size*n_pert,:] = y_test_extended2[:batch_size*n_pert,:]

#for test accuracy on legitimate examples extended by x_test
      feed_dict = {x: adv[2*i*batch_size:2*(i+1)*batch_size,:,:,:]}
      adv_prob_legit[2*i*batch_size:2*(i+1)*batch_size,:] = sess.run(probs,feed_dict = feed_dict)
      feed_dict = {x: x_test[2*i*batch_size:2*(i+1)*batch_size,:,:,:]}
      test_prob_legit[2*i*batch_size:2*(i+1)*batch_size,:] = sess.run(probs,feed_dict = feed_dict)

      l_bat=len(x_adv_pertpart)
      jsteps = int(l_bat/batch_size)
      for j in range(jsteps):
#        if j%10 == 0:
#            print("j:{}".format(j))
        val_x_bat = x_test_pertpart[j*batch_size:(j+1)*batch_size]
        val_adv_bat = x_adv_pertpart[j*batch_size:(j+1)*batch_size]

        feed_dict = {x: val_x_bat}
        logits_bat = sess.run(preds, feed_dict=feed_dict)
        
        feed_dict = {x: val_adv_bat}#range to [-0.5, 0.5]
        logits_adv_bat = sess.run(preds,feed_dict=feed_dict)
#        loss_adv = sess.run(loss_x,feed_dict=feed_dict)

        y_test_prob = sess.run(probs,feed_dict = {x: val_x_bat})
        y_adv_prob = sess.run(probs,feed_dict = {x: val_adv_bat})

        #for tensorflow1.2, keep_prob cannot be defined as placeholder, it must be a scalar and it needn't be fed to session

        logits_ext[2*i*batch_size*n_pert+j*batch_size:2*i*batch_size*n_pert+(j+1)*batch_size,:] = logits_bat
        logits_adv_ext[2*i*batch_size*n_pert+j*batch_size:2*i*batch_size*n_pert+(j+1)*batch_size,:]  = logits_adv_bat
        
        test_prob_pertpart[2*i*batch_size*n_pert+j*batch_size:2*i*batch_size*n_pert+(j+1)*batch_size,:] = y_test_prob
        adv_prob_pertpart[2*i*batch_size*n_pert+j*batch_size:2*i*batch_size*n_pert+(j+1)*batch_size,:]  = y_adv_prob

  #########################################
  auc_score_test = roc_auc_score(y_test, test_prob_legit)
  auc_score_adv = roc_auc_score(y_test, adv_prob_legit)
  print("auc_score_test:{},auc_score_adv:{}".format(auc_score_test, auc_score_adv))
  auc_score_test_ext = roc_auc_score(y_test_pertpart, test_prob_pertpart)
  auc_score_adv_ext = roc_auc_score(y_test_pertpart, adv_prob_pertpart)
  print("auc on extended examples, auc_score_test_ext:{},auc_score_adv_ext:{}".format(auc_score_test_ext, auc_score_adv_ext))
    
  y_test_ext = np.argmax(y_test_pertpart,axis=1)
  cur_preds = np.argmax(logits_ext,axis=1)
  cur_preds_adv = np.argmax(logits_adv_ext,axis=1)
  y_test_ext =  np.reshape(y_test_ext,(len(y_test_pertpart)//n_pert,n_pert))
  logits_ext = np.reshape(cur_preds,(len(cur_preds)//n_pert,n_pert))
  logits_adv_ext = np.reshape(cur_preds_adv,(len(cur_preds_adv)//n_pert,n_pert))
  acc_ext = np.sum(np.equal(logits_ext,y_test_ext))/y_test_ext.shape[0]/y_test_ext.shape[1]
  acc_adv_ext = np.sum(np.equal(logits_adv_ext,y_test_ext))/y_test_ext.shape[0]/y_test_ext.shape[1]
  print('Test accuracy on legitimate examples extened by x_test: %0.4f' % (acc_ext))
  print('Test accuracy on extended examples of adversarials: %0.4f' % (acc_adv_ext))

    
  test_result_stat=np.zeros((ext_bat,),dtype=np.float32)
  adv_result_stat=np.zeros((ext_bat,),dtype=np.float32)
    
  eva_thresh = np.linspace(32,64,9).astype('int32')#from 32 to 64 with a length 9
  len_thresh = len(eva_thresh)
  distrib_incons_preds = np.zeros((len_thresh,n_pert),dtype=np.int32)
  distrib_incons_preds_adv = np.zeros((len_thresh,n_pert),dtype=np.int32)

  auc_div_mat = np.zeros((len(cur_preds),n_pert+1),dtype=np.int32)
  auc_div_mat_adv = np.zeros((len(cur_preds),n_pert+1),dtype=np.int32)

  for i in range(len(y_test_ext)):
      temp = np.sum(np.equal(logits_ext[i,:],y_test_ext[i,:]))
      auc_div_mat[i,temp] = 1
      test_result_stat[temp] = test_result_stat[temp]+1
      a = np.unique(logits_ext[i,:])
      for j in range(len_thresh):
        if temp<eva_thresh[j]:
          kk = []
          for k in range(len(a)):
              kk.extend([np.sum(logits_ext[i,:]==a[k])])
          ind = np.max(np.array(kk))
          distrib_incons_preds[j,ind-1] = distrib_incons_preds[j,ind-1]+1

  for i in range(len(y_test_ext)):
      temp = np.sum(np.equal(logits_adv_ext[i,:],y_test_ext[i,:]))
      auc_div_mat_adv[i,temp] = 1
      adv_result_stat[temp] = adv_result_stat[temp]+1#there is a inconsensus detection results of the 27 perturbations
      a = np.unique(logits_adv_ext[i,:])
      for j in range(len_thresh):
        if temp<eva_thresh[j]:
          kk = []
          for k in range(len(a)):
              kk.extend([np.sum(logits_adv_ext[i,:]==a[k])])
          ind = np.max(np.array(kk))
          distrib_incons_preds_adv[j,ind-1] = distrib_incons_preds_adv[j,ind-1]+1
    
#For a benign, thare are n_pert extension images.
#And there are n_pert classifications of the extension of a benign. They may be different or same
#The maximum occurrence of the classification labels is called consistent rank.
#e.g., n_pert=5, and the classification labels of a benign are (0, 2, 2, 1, 2), then the consistent rank of the benign is 3 that is the occurrence of the label 2.
#Furthermore, correct consistent rank is the number of the extensions of a benign that are correctly classified
#test_result_stat[i]=k
#i: correct consistent rank, i=0, 1, 2, ..., n_pert-1
#k is the count of the correct consistent rank i on test images
  print("test_result_stat:{},{}".format(np.sum(test_result_stat),test_result_stat))
  print("adv_result_stat:{},{}".format(np.sum(adv_result_stat),adv_result_stat))
  for i in range(len_thresh):
#distrib_incons_preds3 is the count of consistent rank on the test images with correct consistent rank less than 3
      print("test_result cannot be classified stat (Threshold {}):{},{}".format(eva_thresh[i],np.sum(distrib_incons_preds[i,:]),distrib_incons_preds[i,:]))
#distrib_incons_preds4 is the count of consistent rank on the test images with correct consistent rank less than 4
#    print("test_result cannot be classified stat (Threshold 4):{},{}".format(np.sum(distrib_incons_preds4),distrib_incons_preds4))
#classfication: a benign with N(consistent rank)>=3 is labeled consistent rank
#The number of correctly classified benign is N(correct consistent rank)>=3
      print("The number of benigns that are correctly classified (Threshold {}):{}".format(eva_thresh[i],np.sum(test_result_stat[eva_thresh[i]-len(adv_result_stat):])))
#The number of incorrectly classified benign is the cardinality of the set {example | N(consistent rank)>=3,  true-label(example)~=consistent rank}
      print("The number of benigns that are misclassified (Threshold {}):{}".format(eva_thresh[i],np.sum(distrib_incons_preds[i,eva_thresh[i]-len(adv_result_stat):])))
      print("The number of benigns that are incorrectly detected as adv (Threshold {}):{}".format(eva_thresh[i],np.sum(distrib_incons_preds[i,:eva_thresh[i]-1])))
  
      print("adv_result cannot be classifed stat (Threshold {}):{},{}".format(eva_thresh[i],np.sum(distrib_incons_preds_adv[i,:]),distrib_incons_preds_adv[i,:]))
      print("The number of adv that are correctly classified (Threshold {}):{}".format(eva_thresh[i],np.sum(adv_result_stat[eva_thresh[i]-len(adv_result_stat):])))
      print("The number of adv that are misclassified (Threshold {}):{}".format(eva_thresh[i],np.sum(distrib_incons_preds_adv[i,eva_thresh[i]-len(adv_result_stat):])))
      print("The number of adv that are correctly detected as adv (Threshold {}):{}".format(eva_thresh[i],np.sum(distrib_incons_preds_adv[i,:eva_thresh[i]-1])))    
    
    ####calculate auc
      benign_ind_clc = np.argwhere(np.sum(auc_div_mat[:,eva_thresh[i]:],axis=1)==1)[:,0]
      adv_ind_clc = np.argwhere(np.sum(auc_div_mat_adv[:,eva_thresh[i]:],axis=1)==1)[:,0]
      benign_inds = benign_ind_clc*n_pert
      adv_inds = adv_ind_clc*n_pert
      for j in range(1,n_pert):
        benign_inds = np.concatenate((benign_inds, benign_ind_clc*n_pert + j), axis=0)
        adv_inds = np.concatenate((adv_inds, adv_ind_clc*n_pert + j), axis=0)
      ground_labels = y_test_pertpart[tuple(benign_inds),:]
      del_ind=[]
      for j in range(nb_classes):
        if np.sum(ground_labels[:,j])==0:
            del_ind.append(j)
      del_ind = np.array(del_ind)
      ground_labels = np.delete(ground_labels,del_ind,axis=1)
      preded_probs = test_prob_pertpart[tuple(benign_inds),:]
      preded_probs = np.delete(preded_probs,del_ind,axis=1) 
      auc_score_clc_bn = roc_auc_score(ground_labels, preded_probs)
                              
      ground_labels_adv = y_adv_pertpart[tuple(adv_inds),:]
      del_ind=[]
      for j in range(nb_classes):
        if np.sum(ground_labels_adv[:,j])==0:
            del_ind.append(j)
      del_ind = np.array(del_ind)
      ground_labels_adv = np.delete(ground_labels_adv,del_ind,axis=1)
      preded_probs_adv = adv_prob_pertpart[tuple(adv_inds),:]
      preded_probs_adv = np.delete(preded_probs_adv,del_ind,axis=1) 
      auc_score_clc_adv = roc_auc_score(ground_labels_adv, preded_probs_adv)
#    auc_score_clc_adv = roc_auc_score(y_adv_pertpart[tuple(adv_inds),:], y_pertpart_prob_adv[tuple(adv_inds),:])
      print("(Threshold {}:) auc_score_clc_bn:{},auc_score_clc_adv:{}".format(eva_thresh[i],auc_score_clc_bn, auc_score_clc_adv))
  
  # Close TF session
  sess.close()

  
def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  
  cifar10_tutorial(viz_enabled=FLAGS.viz_enabled,
                   nb_epochs=FLAGS.nb_epochs,
                   batch_size=FLAGS.batch_size,
                   source_samples=FLAGS.source_samples)

  
  
if __name__ == '__main__':
  flags.DEFINE_boolean('viz_enabled', VIZ_ENABLED,
                       'Visualize adversarial ex.')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_integer('source_samples', SOURCE_SAMPLES,
                       'Nb of test inputs to attack')

  tf.app.run()
