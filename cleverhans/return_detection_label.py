#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:22:20 2020

@author: lab
"""
import logging
import numpy as np
from cleverhans.utils import _ArgsWrapper, create_logger
import tensorflow as tf
import math

_logger = create_logger("cleverhans.utils.tf")
_logger.setLevel(logging.INFO)
def rt_detec_lab(sess, x, y, predictions, X_test, Y_test, n_pert=64, args=None, feed=None, nchannels=3):
  global _model_eval_cache
  args = _ArgsWrapper(args or {})

  assert args.batch_size, "Batch size was not given in args dict"
  if X_test is None or Y_test is None:
    raise ValueError("X_test argument and Y_test argument "
                     "must be supplied.")

  # Define accuracy symbolically
  key = (y, predictions)
#  print("key:{},y:{},predictions:{}".format(key,y,predictions))
#  assert 1==2
  pred_labels = tf.argmax(predictions, axis=-1)
  if key in _model_eval_cache:
    correct_preds = _model_eval_cache[key]
#    print("correct_preds:{}===================================".format(correct_preds))
  else:
#    pred_labels = tf.argmax(predictions, axis=-1)
    correct_preds = tf.equal(tf.argmax(y, axis=-1), pred_labels)
    
    _model_eval_cache[key] = correct_preds

#  print("correct_preds:{}".format(correct_preds))
#  assert 1==2

  # Init result var
  accuracy = 0.0
  ASR = 0.0
  suc_att_exam=[]
  detec_exam = []

  with sess.as_default():
    # Compute number of batches
    nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
    assert nb_batches * args.batch_size >= len(X_test)
#    plt.imshow(X_test[0,:,:,0])
#    plt.show()
#    print("cleverhans/utils_tf.py==============X_test:{},Y_test:{},predictions:{}".format(X_test.shape,Y_test.shape,predictions))

    X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
                     dtype=X_test.dtype)
    Y_cur = np.zeros((args.batch_size,) + Y_test.shape[1:],
                     dtype=Y_test.dtype)
#    print("cleverhans/utils_tf.py==============X_test:{},X_cur:{},nb_batches:{}".format(X_test.shape,X_cur.shape,nb_batches))
#    assert 1==2
    for batch in range(nb_batches):
      if batch % 5000 == 0 and batch > 0:
        _logger.debug("Batch " + str(batch))

      # Must not use the `batch_indices` function here, because it
      # repeats some examples.
      # It's acceptable to repeat during training, but not eval.
      start = batch * args.batch_size
      end = min(len(X_test), start + args.batch_size)
      
      # The last batch may be smaller than all others. This should not
      # affect the accuarcy disproportionately.
      cur_batch_size = end - start
      X_cur[:cur_batch_size] = X_test[start:end]
      Y_cur[:cur_batch_size] = Y_test[start:end]
      feed_dict = {x: X_cur, y: Y_cur}
      if feed is not None:
        feed_dict.update(feed)
      cur_preds = pred_labels.eval(feed_dict=feed_dict)
      cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)
#      ppp=predictions.eval(feed_dict=feed_dict)
      
      #for untargeted attack, suc_att_exam[i] is true means a successful classified examples
      #for targeted attack, suc_att_exam[i] is true means a successful attack, it counts succeful attacked examples
      cur_preds =cur_preds[:cur_batch_size]     
      detec_exam=np.append(detec_exam,cur_preds)
      cur_corr_preds =cur_corr_preds[:cur_batch_size]     
      suc_att_exam=np.append(suc_att_exam,cur_corr_preds)
#      print("ccccccccccccccccccccccccur_corr_preds:{},cur_corr_preds{}".format(cur_corr_preds,X_test[start:end].shape))

      
#      print("-----------------------cur_corr_preds:{},Y_cur:{},predictions:{},".format(cur_corr_preds,Y_cur,suc_att_exam))
#      assert 2==1

#      print("suc_att_exam:{},{},cur_batch_size:{}".format(suc_att_exam,cur_corr_preds,cur_batch_size))
      accuracy += cur_corr_preds[:cur_batch_size].sum()
      ASR += cur_batch_size - cur_corr_preds[:cur_batch_size].sum()

    assert end >= len(X_test)

    # Divide by number of examples to get final value
    accuracy /= len(X_test)
#    print("cur_corr_preds:{},{},cur_batch_size:{},accuracy:{},ASR:{}".format(cur_corr_preds.shape,cur_corr_preds,cur_batch_size,accuracy,ASR))
#    base_range=4
#    n_pert = base_range**nchannels    
    n_samples = len(suc_att_exam)//n_pert
    rem = len(suc_att_exam)%n_pert
    assert rem == 0
    
    pert_labels = np.zeros((n_samples, n_pert), dtype= np.float32)
    suc_att_labels = np.zeros((n_samples, n_pert), dtype= np.float32)
    for i in range(n_samples):
        suc_att_labels[i,:] = suc_att_exam[i*n_pert:(i+1)*n_pert]
        pert_labels[i,:] = detec_exam[i*n_pert:(i+1)*n_pert]

  return accuracy, suc_att_labels, pert_labels
_model_eval_cache = {}   
    
