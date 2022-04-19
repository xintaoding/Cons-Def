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
from cleverhans.dataset import MNIST
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_tf import model_eval, model_argmax
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN

#Added by Ding
#from tensorflow.python import pywrap_tensorflow
from cleverhans.data_exten import data_exten
from sklearn.metrics import roc_curve, roc_auc_score#Added by Ding


FLAGS = flags.FLAGS
VIZ_ENABLED = False#True

BATCH_SIZE = 128
LEARNING_RATE = 0.001
SOURCE_SAMPLES = 1000


def mnist_tutorial_jsma(train_start=0, train_end=60000, test_start=0,
                        test_end=10000, viz_enabled=VIZ_ENABLED,
                        batch_size=BATCH_SIZE,
                        source_samples=SOURCE_SAMPLES,
                        learning_rate=LEARNING_RATE):
  """
  MNIST tutorial for the Jacobian-based saliency map approach (JSMA)
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param viz_enabled: (boolean) activate plots of adversarial examples
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param nb_classes: number of output classes
  :param source_samples: number of test inputs to attack
  :param learning_rate: learning rate for training
  :return: an AccuracyReport object
  """

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  sess = tf.Session()
  print("Created TensorFlow session.")

  # Get MNIST data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Use Image Parameters
  print("y_train_shape:{}".format(y_train.shape))#########################################
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))
  
  nb_filters = 64

  # Define TF model graph
  model = ModelBasicCNN('model1', nb_classes, nb_filters)
  preds = model.get_logits(x)
  print("Defined TensorFlow model graph.")

#    loss = CrossEntropy(model, smoothing=label_smoothing)
  #########################################Added by Ding  
  print("aaaaaaaaaaaaaaaaa=+++++++++++++++++++++++++++++")
  saver = tf.train.Saver() 
  print("aaaaaaaaaaaaaaaaa=--------------------------------")
  saver.restore(sess,'./cifar10_extensions/mnist_models/mnist_train_2_4_8_16_32Aug50iters')
  print("aaaaaaaaaaaaaaaaa=************************************")
  eval_params = {'batch_size': batch_size}
#  rng = np.random.RandomState([2017, 8, 30])
  accuracy, suc_att_exam = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)

  assert x_test.shape[0] == test_end - test_start, x_test.shape
  print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
  report.clean_train_clean_eval = accuracy

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
  test_ind=np.random.randint(0,x_test.shape[0],source_samples)
  x_test=x_test[test_ind,:,:,:]
  y_test=y_test[test_ind]
  accuracy, suc_att_exam = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)

  print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx_test:{}",x_test.shape)
  #####################################################################

  adv = np.zeros((np.shape(x_test)), dtype='f')
  # Loop over the samples we want to perturb into adversarial examples
  for sample_ind in xrange(0, source_samples):
    print('--------------------------------------')
    print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
    sample = x_test[sample_ind:(sample_ind + 1)]

    # We want to find an adversarial example for each possible target class
    # (i.e. all classes that differ from the label given in the dataset)
    current_class = int(np.argmax(y_test[sample_ind]))
    target_classes = other_classes(nb_classes, current_class)
    target = target_classes[np.random.randint(0,len(target_classes),1)[0]]
    
    # For the grid visualization, keep original images along the diagonal
    grid_viz_data[current_class, current_class, :, :, :] = np.reshape(sample, (img_rows, img_cols, nchannels))

    # Loop over all target classes
    print('Generating adv. example for target class %i' % target)

      # This call runs the Jacobian-based saliency map approach
    one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
    one_hot_target[0, target] = 1
    jsma_params['y_target'] = one_hot_target
    adv_x = jsma.generate_np(sample, **jsma_params)
      
###################################################
    adv_x1=adv_x[0,:,:]
    sample1=sample[0,:,:]
    dsaeij=(np.sum((adv_x1 - sample1)**2)**.5)/adv_x1.shape[0]/adv_x1.shape[1]
    print("==============Avg. distortion of successful adversarial examples wether it can or cannot fool the model:{}".format(dsaeij))
###################################################

   # Check if success was achieved
    netpred = model_argmax(sess, x, preds, adv_x)
    res = int(netpred == target)
    acc = int(netpred == current_class)

    # Compute number of modified features
    adv_x_reshape = adv_x.reshape(-1)
    test_in_reshape = x_test[sample_ind].reshape(-1)
    nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
    percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]#ratio of attacked pixels to image size
    print("nb_changed:{}".format(nb_changed))

   # Display the original and adversarial images side-by-side
    if viz_enabled:
      figure = pair_visual(
          np.reshape(sample, (img_rows, img_cols, nchannels)),
          np.reshape(adv_x, (img_rows, img_cols, nchannels)), figure)

   # Add our adversarial example to our grid data
    grid_viz_data[target, current_class, :, :, :] = np.reshape(
        adv_x, (img_rows, img_cols, nchannels))

   # Update the arrays for later analysis,res is true if the targeted attack is successful
    results[0, sample_ind] = res
    auc[0, sample_ind] = acc
    dsae[0, sample_ind] = dsaeij
    adv[sample_ind,:,:,:] = adv_x1

#Matrix, For an i-th targeted attack, perturbations[i,j] is the ratio of attacked pixels to the size of the j-th sample.
    perturbations[0, sample_ind] = percent_perturb

  print('--------------------------------------')
  
  print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
  # Compute the number of adversarial examples that were successfully found
  nb_targets_tried =  source_samples
  succ_rate = float(np.sum(results)) / nb_targets_tried
  acc = float(np.sum(auc)) / nb_targets_tried
  print("Test accuracy on adversarial examples: {}".format(acc))
  print('Avg. rate of successful adv. examples / ASR {}'.format(succ_rate))
  print('For untargeted attack, the number of misclassified examples: sum(results==0):{}'.format(np.sum(results==0)))
  report.clean_train_adv_eval = 1. - succ_rate

  wheresae=np.where(dsae>0)#The successful adv. example is different from clean example
  avg_dsae=np.mean(dsae[wheresae])
  print('Avg. L_2 norm of perturbations (Avg. Distortion on successful adv. examples): {0:.4f}'.format(avg_dsae))

  
#  print("77777777777777777777777777:{}".format(results))
  avg_dsae_foolmod=0
  k=0
  for i in xrange(0, source_samples):
#      print("i={}".format(i))
          if results[0,i]>0:
#              print("results[j,i]:{},meanperturb[j,i]:{},meandist:{},k:{}".format(results[j,i],meanperturb[j,i],meandist,k))
              avg_dsae_foolmod=avg_dsae_foolmod+dsae[0,i]
              k=k+1
  if k>0:
      avg_dsae_foolmod=avg_dsae_foolmod/k
  print('Avg. Distortion on successful adv. examples that fool model/ D-SAE {0:.4f}'.format(avg_dsae_foolmod))

  # Compute the average distortion introduced by the algorithm, percent_perturbed is the perturbed images
  percent_perturbed = np.mean(perturbations[np.where(perturbations!=0)])
  print('perturbation ratio of the number of pixels that their intensity have been changed to the number of image pixels {}'.format(np.mean(perturbations)))
  print('Avg. perturbation ratio on the dataset {0:.4f}'.format(percent_perturbed))

  # Compute the average distortion introduced for successful samples only
  percent_perturb_succ = np.mean(perturbations[np.where(perturbations!=0)] * (results[np.where(perturbations!=0)] == 1))
  print('Avg. rate of perturbed features for successful '
        'adversarial examples {0:.4f}'.format(percent_perturb_succ))

#  adv = np.round(adv*256)/256.0
  adv34 = np.zeros((len(adv),6+img_rows,6+img_rows,1),dtype=np.float32)
  advcrop = np.zeros((len(adv),img_rows,img_cols,1),dtype=np.float32)
  adv34[:,3:31,3:31,:] = adv
  for i in range(len(adv)):
      tf_image = adv34[i,:,:,:]
      lu1 = np.random.randint(0,6)
      lu2 = np.random.randint(0,6)
      advcrop[i,:,:,:] = tf_image[lu1:lu1+img_rows,lu2:lu2+img_cols,:]
  adv = advcrop

  n_pert = 5
  n_exam = len(adv)
  x_adv_extended, y_adv_extended = data_exten(adv, y_test, n_exam, nb_classes, img_rows, img_cols, 1)#
  x_adv_pertpart = x_adv_extended[:n_pert*n_exam,:,:,:]
  y_adv_pertpart = y_adv_extended[:n_pert*n_exam,:]
  n_exam = len(y_test)
  x_test_extended, y_test_extended = data_exten(x_test, y_test, n_exam, nb_classes, img_rows, img_cols, 1)
  x_test_pertpart = x_test_extended[:n_pert*n_exam,:,:,:]
  y_test_pertpart = y_test_extended[:n_pert*n_exam,:]

  acc_ext,suc_att_ext = model_eval(sess, x, y, preds, x_test_pertpart, y_test_pertpart, args=eval_params)
  print('Test accuracy on legitimate examples extened by x_test: %0.4f' % (acc_ext))
#  print("JSMA attack on images extened by adv of x_test")

#  acc_adv_ext,suc_att_adv_ext = model_eval(sess, x, y, preds, adv_ext, y_adv_pertpart, args=eval_params)
#  print('Test accuracy on adversarial examples: %0.4f' % (acc_adv_ext))
  acc_adv_ext,suc_att_adv_ext = model_eval(sess, x, y, preds, x_adv_pertpart, y_adv_pertpart, args=eval_params)
  print('Test accuracy on extended examples of adversarials: %0.4f' % (acc_adv_ext))

  eva_thresh = np.linspace(3,5,3).astype('int32')#5 aug test
  feed_dict = {x: x_test_pertpart}
  y_pertpart_pred = sess.run(preds,feed_dict=feed_dict)
  probs = model.get_probs(x)
  y_pertpart_prob = sess.run(probs,feed_dict=feed_dict)
  auc_score_test = roc_auc_score(y_test_pertpart, y_pertpart_prob)
  
  cur_preds = np.argmax(y_pertpart_pred, axis=1)
  cur_preds = np.reshape(cur_preds,(len(y_test_pertpart)//n_pert,n_pert),order='F')

  feed_dict = {x: x_adv_pertpart}
  y_adv_pertpart_pred = sess.run(preds,feed_dict=feed_dict)
  y_pertpart_prob_adv = sess.run(probs,feed_dict=feed_dict)
  auc_score_adv = roc_auc_score(y_adv_pertpart, y_pertpart_prob_adv)
  print("auc_score_test:{},auc_score_adv:{}".format(auc_score_test, auc_score_adv))
  
  cur_preds_adv = np.argmax(y_adv_pertpart_pred, axis=1)
  cur_preds_adv = np.reshape(cur_preds_adv,(len(y_adv_pertpart)//n_pert,n_pert),order='F')
    
  test_result_stat=np.zeros((n_pert+1,),dtype=np.float32)
  adv_result_stat=np.zeros((n_pert+1,),dtype=np.float32)
    
  len_thresh = len(eva_thresh)
  distrib_incons_preds = np.zeros((len_thresh,n_pert),dtype=np.int32)
  distrib_incons_preds_adv = np.zeros((len_thresh,n_pert),dtype=np.int32)
  
  y_test5=np.argmax(y_test_pertpart,axis=1)
  y_test5=np.reshape(y_test5,(len(y_test_pertpart)//n_pert,n_pert),order='F')
  
  auc_div_mat = np.zeros((len(cur_preds),n_pert+1),dtype=np.int32)
  auc_div_mat_adv = np.zeros((len(cur_preds),n_pert+1),dtype=np.int32)
  
  for i in range(len(cur_preds)):
      temp = np.sum(np.equal(cur_preds[i,:],y_test5[i,:]))
      auc_div_mat[i,temp] = 1
      test_result_stat[temp] = test_result_stat[temp]+1
      a = np.unique(cur_preds[i,:])
      for j in range(len_thresh):
        if temp<eva_thresh[j]:
          kk = []
          for k in range(len(a)):
              kk.extend([np.sum(cur_preds[i,:]==a[k])])
          ind = np.max(np.array(kk))
          distrib_incons_preds[j,ind-1] = distrib_incons_preds[j,ind-1]+1

  y_test5=np.argmax(y_adv_pertpart,axis=1)
  y_test5=np.reshape(y_test5,(len(y_adv_pertpart)//n_pert,n_pert),order='F')
  for i in range(len(cur_preds_adv)):
      temp = np.sum(np.equal(cur_preds_adv[i,:],y_test5[i,:]))
      auc_div_mat_adv[i,temp] = 1
      adv_result_stat[temp] = adv_result_stat[temp]+1#there is a inconsensus detection results of the 27 perturbations
      a = np.unique(cur_preds_adv[i,:])
      for j in range(len_thresh):
        if temp<eva_thresh[j]:
          kk = []
          for k in range(len(a)):
              kk.extend([np.sum(cur_preds_adv[i,:]==a[k])])
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
    benign_inds = benign_ind_clc
    adv_inds = adv_ind_clc
    for j in range(1,n_pert):
        benign_inds = np.concatenate((benign_inds, benign_ind_clc + source_samples*j), axis=0)
        adv_inds = np.concatenate((adv_inds, adv_ind_clc + source_samples*j), axis=0)
    ground_labels = y_test_pertpart[tuple(benign_inds),:]
    del_ind=[]
    for j in range(nb_classes):
        if np.sum(ground_labels[:,j])==0:
            del_ind.append(j)
    del_ind = np.array(del_ind)
    ground_labels = np.delete(ground_labels,del_ind,axis=1)
    preded_probs = y_pertpart_prob[tuple(benign_inds),:]
    preded_probs = np.delete(preded_probs,del_ind,axis=1) 
    auc_score_clc_bn = roc_auc_score(ground_labels, preded_probs)
                              
    ground_labels_adv = y_adv_pertpart[tuple(adv_inds),:]
    del_ind=[]
    for j in range(nb_classes):
        if np.sum(ground_labels_adv[:,j])==0:
            del_ind.append(j)
    del_ind = np.array(del_ind)
    ground_labels_adv = np.delete(ground_labels_adv,del_ind,axis=1)
    preded_probs_adv = y_pertpart_prob_adv[tuple(adv_inds),:]
    preded_probs_adv = np.delete(preded_probs_adv,del_ind,axis=1) 
    auc_score_clc_adv = roc_auc_score(ground_labels_adv, preded_probs_adv)
#    auc_score_clc_adv = roc_auc_score(y_adv_pertpart[tuple(adv_inds),:], y_pertpart_prob_adv[tuple(adv_inds),:])
    print("(Threshold {}:) auc_score_clc_bn:{},auc_score_clc_adv:{}".format(eva_thresh[i],auc_score_clc_bn, auc_score_clc_adv))
    print('sssss')
    
# Close TF session
  sess.close()

  # Finally, block & display a grid of all the adversarial examples
  if viz_enabled:
    import matplotlib.pyplot as plt
    plt.close(figure)
    _ = grid_visual(grid_viz_data)

  return report



def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial_jsma(viz_enabled=FLAGS.viz_enabled,
                      batch_size=FLAGS.batch_size,
                      source_samples=FLAGS.source_samples,
                      learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
  flags.DEFINE_boolean('viz_enabled', VIZ_ENABLED,
                       'Visualize adversarial ex.')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_integer('source_samples', SOURCE_SAMPLES,
                       'Nb of test inputs to attack')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')

  tf.app.run()
