"""
This tutorial shows how to implement FGSM white-box attacks on clean model.
Xintao Ding
"""

import numpy as np
import tensorflow as tf

import slim.nets.vgg160 as vgg
from cifar10_create_tf_record import get_example_nums, read_records, get_batch_images
import tensorflow.contrib.slim as slim

from cleverhans.compat import reduce_max, reduce_sum, softmax_cross_entropy_with_logits
from cleverhans import utils_tf

from cleverhans.data_exten_mulpro import data_exten
from sklearn.metrics import roc_curve, roc_auc_score

labels_nums = 10  #  the number of labels
batch_size = 20  
resize_height = 32  # Cifar10 size
resize_width = 32   
net_height= 160#the size must be suitable with layer in vgg (net, 4096, [5, 5], padding=fc_conv_padding, scope='fc6'), 224*224 corresponds to [7, 7], 160*160 correpsonds to [5, 5],
net_width = 160#vgg16 size
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

input_images = tf.placeholder(dtype=tf.float32, shape=[None, net_height, net_width, depths], name='input')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')
y = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums])
is_training = tf.placeholder(tf.bool, name='is_training')

val_record_file='cifar10_extensions/cifar10_test.tfrecords_seg'
val_nums=get_example_nums(val_record_file)
print('val nums:%d'%(val_nums))

val_images, val_labels = read_records([val_record_file], resize_height, resize_width, depths,
                                              type='normalization') 
val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=False, num_threads=1)
val_images_batch = tf.image.resize_images(val_images_batch,size=(net_height, net_width))
val_images_batch = tf.rint(val_images_batch*256.)*(1. / 256)

with slim.arg_scope(vgg.vgg_arg_scope()):
    out, end_points = vgg.vgg_16(inputs=input_images, num_classes=labels_nums, is_training=is_training)
probs = tf.nn.softmax(out)

tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)
loss = tf.losses.get_total_loss(add_regularization_losses=True)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))

#########################################Added by Ding  
saver = tf.train.Saver()
eps = 0.03
clip_min = 0.
clip_max = 1.
ord=np.inf

def fgm(x,
        logits,
        y=None,
        eps=0.3,
        ord=np.inf,
        loss_fn=softmax_cross_entropy_with_logits,
        clip_min=None,
        clip_max=None,
        clip_grad=False,
        targeted=False,
        sanity_checks=True):
  """
  TensorFlow implementation of the Fast Gradient Method.
  :param x: the input placeholder
  :param logits: output of model.get_logits
  :param y: (optional) A placeholder for the true labels. If targeted
            is true, then provide the target label. Otherwise, only provide
            this parameter if you'd like to use true labels when crafting
            adversarial samples. Otherwise, model predictions are used as
            labels to avoid the "label leaking" effect (explained in this
            paper: https://arxiv.org/abs/1611.01236). Default is None.
            Labels should be one-hot-encoded.
  :param eps: the epsilon (input variation parameter)
  :param ord: (optional) Order of the norm (mimics NumPy).
              Possible values: np.inf, 1 or 2.
  :param loss_fn: Loss function that takes (labels, logits) as arguments and returns loss
  :param clip_min: Minimum float value for adversarial example components
  :param clip_max: Maximum float value for adversarial example components
  :param clip_grad: (optional bool) Ignore gradient components
                    at positions where the input is already at the boundary
                    of the domain, and the update step will get clipped out.
  :param targeted: Is the attack targeted or untargeted? Untargeted, the
                   default, will try to make the label incorrect. Targeted
                   will instead try to move in the direction of being more
                   like y.
  :return: a tensor for the adversarial example
  """

  asserts = []

  print("OOOOOOOOOOOOOOOOOOOOOOOO:{}".format(ord))

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    asserts.append(utils_tf.assert_greater_equal(
        x, tf.cast(clip_min, x.dtype)))

  if clip_max is not None:
    asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

  # Make sure the caller has not passed probs by accident
  assert logits.op.type != 'Softmax'

  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    preds_max = reduce_max(logits, 1, keepdims=True)
    y = tf.to_float(tf.equal(logits, preds_max))
    y = tf.stop_gradient(y)
  y = y / reduce_sum(y, 1, keepdims=True)
  
  print("yyyyyyyyyyyyyyyyyyyyyy={}".format(y))

  # Compute loss
  loss = loss_fn(labels=y, logits=logits)
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
  """
  Solves for the optimal input to a linear function under a norm constraint.

  Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)

  :param grad: tf tensor containing a batch of gradients
  :param eps: float scalar specifying size of constraint region
  :param ord: int specifying order of norm
  :returns:
    tf tensor containing optimal perturbation
  """

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
                        reduce_sum(tf.square(grad),
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

adv_x, grad_x, loss_x, _, _  = fgm(input_images, logits=out, eps=eps, clip_min=clip_min, clip_max=clip_max)

val_max_steps = int(val_nums/batch_size)
n_lev=255

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  
  saver.restore(sess,'models/cifar10/vgg16_clean_models_416000_0.8974.ckpt')
  print("aaaaaaaaaaaaaaaaa=+++++++++++++++++++++++++++++")
#########################################

  x_test = np.zeros((val_nums,net_height,net_width,depths),dtype=np.float32)
  y_test = np.zeros((val_nums,labels_nums),dtype=np.float32)
  logits = np.zeros((val_nums,labels_nums),dtype=np.float32)
  logits_adv = np.zeros((val_nums,labels_nums),dtype=np.float32)
  adv = np.zeros((val_nums,net_height,net_width,depths),dtype=np.float32)
  
  for i in range(val_max_steps):
      if i%10==0:
          print("i:{}".format(i))
      val_x_bat, val_y_bat = sess.run([val_images_batch, val_labels_batch])

      feed_dict = {input_images: val_x_bat, is_training:False}
      logits_bat = sess.run(out, feed_dict=feed_dict)
  
      feed_dict = {input_images: val_x_bat, is_training:False}
      adv_bat = sess.run(adv_x, feed_dict=feed_dict)

      feed_dict = {input_images: adv_bat, is_training:False}
      logits_adv_bat = sess.run(out, feed_dict=feed_dict)

      x_test[i*batch_size:(i+1)*batch_size,:,:,:] = val_x_bat
      y_test[i*batch_size:(i+1)*batch_size,:] = val_y_bat
      adv[i*batch_size:(i+1)*batch_size,:,:,:] = adv_bat
      logits[i*batch_size:(i+1)*batch_size,:] = logits_bat
      logits_adv[i*batch_size:(i+1)*batch_size,:] = logits_adv_bat
#########################################
  coord.request_stop()
  coord.join(threads)
  
#  np.save("cifar10_vgg16_augmodel_fgsminf_10000adv", adv)
  
 #for untargeted attack, suc_att_exam[i] is true means a successful classified examples
 #for targeted attack, suc_att_exam[i] is true means a successful attack, it counts succeful attacked examples
  percent_perturbed = np.mean(np.sum((adv - x_test)**2, axis=(1, 2, 3))**.5)
  
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

  print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
  print('The number of successful attack:{}, Avg. L_2 norm of perturbations on successful attack / dsae:{}'.format(kk,dsae))
  
  logits = np.argmax(logits,axis=1)
  logits_adv = np.argmax(logits_adv,axis=1)
  y_test_argmax = np.argmax(y_test,axis=1)
  acc = np.sum(np.equal(logits,y_test_argmax))/len(y_test_argmax)
  acc_adv = np.sum(np.equal(logits_adv,y_test_argmax))/len(y_test_argmax)
  print('Test accuracy on legitimate test examples: %0.4f' % (acc))
  print('Test accuracy on adversarial test examples: %0.4f' % (acc_adv))

  sess.close()  


