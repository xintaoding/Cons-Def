"""
This tutorial shows how to generate adversarial examples using FGSM
Revised from Cleverhans
Xintao Ding
"""
# pylint: disable=missing-docstring
import numpy as np
import tensorflow as tf

import slim.nets.inception_v3 as inception_v3
import tensorflow.contrib.slim as slim
from create_tf_record import get_example_nums,read_records,get_batch_images

from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta, random_lp_vector
from cleverhans.compat import reduce_max, reduce_sum, softmax_cross_entropy_with_logits

from sklearn.metrics import roc_curve, roc_auc_score#Added by Ding
#from tensorflow.python import pywrap_tensorflow


batch_size = 25  #
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64

labels_nums = 10  # the number of labels
resize_height = 299  #ImageNet size
resize_width = 299
depths = 3

is_training = tf.placeholder(tf.bool, name='is_training')



input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')
y = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums])

keep_prob = 0.5#tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')


#test data
val_record_file='data/record/val299.tfrecords'
val_nums=get_example_nums(val_record_file)
print('val nums:%d'%(val_nums))
#    val_images, val_labels = read_records([val_record_file], resize_height, resize_width, type='normalization')
val_images, val_labels = read_records([val_record_file], resize_height, resize_width, type='centralization')
val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False,num_threads=1)

# Define the model:
with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=keep_prob, is_training=is_training)
#    out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, is_training=is_training)
probs = tf.nn.softmax(out)
tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)
loss = tf.losses.get_total_loss(add_regularization_losses=True)

accuracy = tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1))

def fgm(x,
        logits,
        y = None,
        eps=0.3,
        ord=np.inf,
        loss_fn=softmax_cross_entropy_with_logits,
        clip_min=None,
        clip_max=None,
        clip_grad=False,
        targeted=False,
        sanity_checks=True):

  asserts = []

  print("OOOOOOOOOOOOOOOOOOOOOOOO:{}".format(ord))

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    asserts.append(utils_tf.assert_greater_equal(x, tf.cast(clip_min, x.dtype)))

  if clip_max is not None:
    asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

  # Make sure the caller has not passed probs by accident

#  if y is None:
    # Using model predictions as ground truth to avoid label leaking
#    preds_max = reduce_max(logits, 1, keepdims=True)
#    y = tf.to_float(tf.equal(logits, preds_max))
#    y = tf.stop_gradient(y)
#  else:
#  ty = tf.Variable(np.zeros((batch_size, labels_nums)), dtype=tf.float32)
  
#  ty.assign(y)
  y = y / reduce_sum(y, 1, keepdims=True)
  
  print("yyyyyyyyyyyyyyyyyyyyyy={}".format(y))

  loss = loss_fn(labels=y, logits=logits)
  # Compute loss
  if targeted:
    loss = -loss

  # Define gradient of loss wrt input
  grad, = tf.gradients(loss, x)

  if clip_grad:
    grad = utils_tf.zero_out_clipped_grads(grad, x, clip_min, clip_max)

  optimal_perturbation = optimize_linear(grad, eps, ord)

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + optimal_perturbation
  
  print("adv_x=============================:{}".format((adv_x)))
#  assert 1==2

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    # We don't currently support one-sided clipping
    assert clip_min is not None and clip_max is not None
    adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)
    print("optimal_perturbation:{},adv_x:{}, clip_min:{}, clip_max:{}".format(optimal_perturbation,adv_x, clip_min, clip_max))

  if sanity_checks:
    with tf.control_dependencies(asserts):
      adv_x = tf.identity(adv_x)

  return adv_x, grad, loss, y, optimal_perturbation


def optimize_linear(grad, eps, ord=np.inf):

  # In Python 2, the `list` call in the following line is redundant / harmless.
  # In Python 3, the `list` call is needed to convert the iterator returned by `range` into a list.
  red_ind = list(range(1, len(grad.get_shape())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    # Take sign of gradient
    optimal_perturbation = tf.sign(grad)
    # The following line should not change the numerical results.
    # It applies only because `optimal_perturbation` is the output of
    # a `sign` op, which has zero derivative anyway.
    # It should not be applied for the other norms, where the
    # perturbation has a non-zero derivative.
    optimal_perturbation = tf.stop_gradient(optimal_perturbation)
  elif ord == 1:
    abs_grad = tf.abs(grad)
    sign = tf.sign(grad)
    max_abs_grad = tf.reduce_max(abs_grad, red_ind, keepdims=True)
    tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
    num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)
    optimal_perturbation = sign * tied_for_max / num_ties
  elif ord == 2:
    square = tf.maximum(avoid_zero_div,
                        tf.reduce_sum(tf.square(grad),
                                   reduction_indices=red_ind,
                                   keepdims=True))
    optimal_perturbation = grad / tf.sqrt(square)
  else:
    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                              "currently implemented.")

  # Scale perturbation to be the solution for the norm=eps rather than
  # norm=1 problem
  scaled_perturbation = utils_tf.mul(eps, optimal_perturbation)
  return scaled_perturbation
    


def generate_pgdhead(x, logits,eps=0.1,norm_ord=np.inf,clip_min=-0.5,clip_max=0.5,rand_init=1):
    
    y=None
    rand_init_eps=eps

    # Save attack-specific parameters

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(x, tf.cast(clip_min, x.dtype)))

    if clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max,  x.dtype)))

    # Initialize loop variables
    if rand_init:
      eta = random_lp_vector(tf.shape(x), norm_ord, tf.cast(rand_init_eps, x.dtype), dtype=x.dtype)
    else:
      eta = tf.zeros(tf.shape(x))

    # Clip eta
    eta = clip_eta(eta, norm_ord, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
      adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

    if y is not None:
      y = y
    else:
#    model_preds = model.get_probs(x)
      model_preds = tf.nn.softmax(logits=logits)
      preds_max = tf.reduce_max(model_preds, 1, keepdims=True)
      y = tf.to_float(tf.equal(model_preds, preds_max))
      y = tf.stop_gradient(y)
      del model_preds
    
    return adv_x, y

    
saver = tf.train.Saver()

val_max_steps = int(val_nums / batch_size)

eps=0.03#0.3,
norm_ord=np.inf
clip_min=-0.5
clip_max=0.5
rand_init=1

eps_iter=0.0075#0.01#0.05,

nb_iter=10

  
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


  #########################################Added by Ding  
    print("aaaaaaaaaaaaaaaaa=+++++++++++++++++++++++++++++")
    saver.restore(sess,'models/caffe_ilsvrc12/best_models_71200_0.7500.ckpt')
    print("aaaaaaaaaaaaaaaaa=************************************")

   # Initialize the Projected Gradient Descent Method (PGDM) attack object and
    # graph
#    pgd = ProjectedGradientDescent(model, sess=sess)
    adv_x_head, attack_label = generate_pgdhead(input_images, out,
                                                eps=eps,norm_ord=norm_ord,clip_min=clip_min,
                                                clip_max=clip_max,rand_init=rand_init)
    adv_x, grad_x, loss_x, _, _ = fgm(input_images,logits=out,y=y, eps = eps_iter,
                                      clip_min=clip_min,clip_max=clip_max)

    x_test = np.zeros((val_nums,resize_height,resize_width,depths),dtype=np.float32)
    y_test = np.zeros((val_nums,labels_nums),dtype=np.float32)
    logits = np.zeros((val_nums,labels_nums),dtype=np.float32)
    logits_adv = np.zeros((val_nums,labels_nums),dtype=np.float32)
    adv = np.zeros((val_nums,resize_height,resize_width,depths),dtype=np.float32)
    for i in range(val_max_steps):
      if i%10 == 0:
          print("i:{}".format(i))
      val_x_bat, val_y_bat = sess.run([val_images_batch, val_labels_batch])
        
      feed_dict = {input_images: val_x_bat,  is_training: False}
      logits_bat = sess.run(out, feed_dict=feed_dict)
      feed_dict = {input_images: val_x_bat,  is_training: False}
      adv_bat, label_host_bat = sess.run([adv_x_head, attack_label], feed_dict=feed_dict)
      
      for k in range(nb_iter):
        feed_dict = {input_images: adv_bat, y:label_host_bat, is_training: False}
        advtemp_bat = sess.run(adv_x, feed_dict=feed_dict)
        eta = advtemp_bat - val_x_bat
        eta = np.clip(eta, -eps, eps)
        adv_bat = val_x_bat + eta

      # Redo the clipping.
      # FGM already did it, but subtracting and re-adding eta can add some
      # small numerical error.
        if clip_min is not None or clip_max is not None:
          adv_bat = np.clip(adv_bat, clip_min, clip_max)

        
      feed_dict = {input_images: adv_bat,  is_training: False}
      logits_adv_bat = sess.run(out, feed_dict=feed_dict)
        
        #for tensorflow1.2, keep_prob cannot be defined as placeholder, it must be a scalar and it needn't be fed to session
      x_test[i*batch_size:(i+1)*batch_size,:,:,:] = val_x_bat
      y_test[i*batch_size:(i+1)*batch_size,:] = val_y_bat
      adv[i*batch_size:(i+1)*batch_size,:,:,:] = adv_bat#Ranged in [0, 1]
      logits[i*batch_size:(i+1)*batch_size,:] = logits_bat
      logits_adv[i*batch_size:(i+1)*batch_size,:] = logits_adv_bat
  #########################################
    coord.request_stop()
    coord.join(threads)

    percent_perturbed = np.mean(np.sum((adv - x_test)**2, axis=(1, 2, 3))**.5)

 #for untargeted attack, suc_att_exam[i] is true means a successful classified examples
 #for targeted attack, suc_att_exam[i] is true means a successful attack, it counts succeful attacked examples

    dsae=0
    kk=0
    adv_suc_att_exam = np.equal(np.argmax(logits_adv,axis=1),np.argmax(y_test,axis=1))
    suc_att_exam = np.equal(np.argmax(logits,axis=1),np.argmax(y_test,axis=1))
    for i in range(len(adv_suc_att_exam)):
      if adv_suc_att_exam[i]==0 and suc_att_exam[i]>0:#adversarial is misclassified but its corresponding binign example is correctly detected
          dsae+=np.sum((adv[i,:,:,:] - x_test[i,:,:,:])**2)**.5
          kk += 1
    dsae=dsae/kk
    print("For untargeted attack, the number of misclassified examples (successful attack), sum(adv_suc_att_exam==0):{}, dsae:{}".format(sum(adv_suc_att_exam==0),dsae))

    print('Avg. L_2 norm of perturbations {}'.format(percent_perturbed))
    print('The number of successful attack:{}, Avg. L_2 norm of perturbations on successful attack / dsae:{}'.format(kk,dsae))
    
    val_max_steps = int(len(adv) / batch_size / 2)

    adv_prob_legit = np.zeros((val_nums,labels_nums),dtype=np.float32)
    test_prob_legit = np.zeros((val_nums,labels_nums),dtype=np.float32)
  
    for i in range(val_max_steps):
#for test accuracy on legitimate examples extended by x_test
      feed_dict = {input_images: adv[2*i*batch_size:2*(i+1)*batch_size,:,:,:],  is_training: False}
      adv_prob_legit[2*i*batch_size:2*(i+1)*batch_size,:] = sess.run(probs,feed_dict = feed_dict)
      feed_dict = {input_images: x_test[2*i*batch_size:2*(i+1)*batch_size,:,:,:],  is_training: False}
      test_prob_legit[2*i*batch_size:2*(i+1)*batch_size,:] = sess.run(probs,feed_dict = feed_dict)
  #########################################
    auc_score_test = roc_auc_score(y_test, test_prob_legit)
    auc_score_adv = roc_auc_score(y_test, adv_prob_legit)
    print("auc_score_test:{},auc_score_adv:{}".format(auc_score_test, auc_score_adv))

    logits = np.argmax(logits,axis=1)
    logits_adv = np.argmax(logits_adv,axis=1)
    y_test_argmax = np.argmax(y_test,axis=1)
    acc = np.sum(np.equal(logits,y_test_argmax))/len(y_test_argmax)
    acc_adv = np.sum(np.equal(logits_adv,y_test_argmax))/len(y_test_argmax)
    print('Test accuracy on legitimate test examples: {}' .format (acc))
    print('Test accuracy on adversarial test examples: {}' .format (acc_adv))
    sess.close()




