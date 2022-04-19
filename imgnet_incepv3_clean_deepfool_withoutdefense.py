"""
This tutorial shows how to generate adversarial examples using DeepFool
Revised from Cleverhans
Xintao Ding
"""
# pylint: disable=missing-docstring
import numpy as np
import tensorflow as tf

import slim.nets.inception_v3 as inception_v3
import tensorflow.contrib.slim as slim
from create_tf_record import get_example_nums,read_records,get_batch_images

from sklearn.metrics import roc_curve, roc_auc_score#Added by Ding
import copy

batch_size = 25  #

labels_nums = 10  # the number of labels
resize_height = 299  # ImageNet size
resize_width = 299
depths = 3

input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')
#for tensorflow1.2, keep_prob cannot be defined as placeholder, it must be a scalar and it needn't be fed to session
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

tf_dtype = tf.as_dtype('float32')
np_dtype = np.dtype('float32')

def deepfool_batch(sess, x, predictions, logits, grads, sample, nb_candidate, overshoot, max_iter, clip_min, clip_max, nb_classes, feed=None):
  """
  TensorFlow implementation of DeepFool.
  Paper link: see https://arxiv.org/pdf/1511.04599.pdf
  :param sess: TF session
  :param x: The input placeholder
  :param predictions: The model's sorted symbolic output of logits, only the
                     top nb_candidate classes are contained
  :param logits: The model's unnormalized output tensor (the input to
                 the softmax layer)
  :param grads: Symbolic gradients of the top nb_candidate classes, procuded
               from gradient_graph
  :param sample: Numpy array with sample input
  :param nb_candidate: The number of classes to test against, i.e.,
                       deepfool only consider nb_candidate classes when
                       attacking(thus accelerate speed). The nb_candidate
                       classes are chosen according to the prediction
                       confidence during implementation.
  :param overshoot: A termination criterion to prevent vanishing updates
  :param max_iter: Maximum number of iteration for DeepFool
  :param clip_min: Minimum value for components of the example returned
  :param clip_max: Maximum value for components of the example returned
  :return: Adversarial examples
  """
  adv_x = copy.copy(sample)
  # Initialize the loop variables
  iteration = 0
  if feed is None:
       feed_dict = {x: adv_x, is_training:False}
  else:
      feed_dict.update(feed)
  print("feed:{}".format(feed))
  probabilities = sess.run(logits, feed_dict)
  current = np.argmax(probabilities, axis=1)
#  assert 1==2
#  current = model_argmax(sess, x, logits, adv_x, is_training, feed=feed)
  if current.shape == ():
    current = np.array([current])
  w = np.squeeze(np.zeros(sample.shape[1:]))  # same shape as original image
  r_tot = np.zeros(sample.shape)
  original = current  # use original label as the reference

  print("Starting DeepFool attack up to {} iterations".format(max_iter))
  # Repeat this main loop until we have achieved misclassification
  while (np.any(current == original) and iteration < max_iter):

    if iteration % 5 == 0 and iteration > 0:
      print("Attack result at iteration {} is {}".format(iteration, current))
#    gradients = sess.run(grads, feed_dict={x: adv_x})
#    predictions_val = sess.run(predictions, feed_dict={x: adv_x})
    gradients = sess.run(grads, feed_dict={x: adv_x, is_training:False})
    predictions_val = sess.run(predictions, feed_dict={x: adv_x, is_training:False})
    for idx in range(sample.shape[0]):
      pert = np.inf
      if current[idx] != original[idx]:
        continue
      for k in range(1, nb_candidate):
        w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
        f_k = predictions_val[idx, k] - predictions_val[idx, 0]
        # adding value 0.00001 to prevent f_k = 0
        pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten())
        if pert_k < pert:
          pert = pert_k
          w = w_k
      r_i = pert * w / np.linalg.norm(w)
      r_tot[idx, ...] = r_tot[idx, ...] + r_i

    adv_x = np.clip(r_tot + sample, clip_min, clip_max)
    
#    if feed is None:
#       feed_dict = {x: adv_x, is_training:False}
#    else:
#      feed_dict.update(feed)
    feed_dict = {x: adv_x, is_training:False}
    probabilities = sess.run(logits, feed_dict)
    current = np.argmax(probabilities, axis=1)
#    current = model_argmax(sess, x, logits, adv_x, feed=feed)
    if current.shape == ():
      current = np.array([current])
    # Update loop variables
    iteration = iteration + 1

  # need more revision, including info like how many succeed
  print("Attack result at iteration {} is {}".format(iteration, current))
  print("{} out of {} become adversarial examples at iteration {}".format(sum(current != original),
               sample.shape[0],
               iteration))
  # need to clip this image into the given range
  adv_x = np.clip((1 + overshoot) * r_tot + sample, clip_min, clip_max)
  return np.asarray(adv_x, dtype=np_dtype)
#  return adv_x

def generate_deepfool(sess, x, logits, nb_candidate=10,overshoot=0.02,max_iter=50,clip_min=-0.5,clip_max=0.5):
    nb_classes = logits.get_shape().as_list()[-1]
    assert nb_candidate <= nb_classes, \
        'nb_candidate should not be greater than nb_classes'
    from cleverhans.utils_tf import jacobian_graph
    preds = tf.reshape(
        tf.nn.top_k(logits, k=nb_candidate)[0],
        [-1, nb_candidate])
    # grads will be the shape [batch_size, nb_candidate, image_size]
    grads = tf.stack(jacobian_graph(preds, x, nb_candidate), axis=1)
#    return logits, preds, grads
    # Define graph
    def deepfool_wrap(x_val):
      """deepfool function for py_func"""
      return deepfool_batch(sess, x, preds, logits, grads, x_val,
                            nb_candidate, overshoot,
                            max_iter, clip_min, clip_max, nb_classes)

    wrap = tf.py_func(deepfool_wrap, [x], tf_dtype)
    wrap.set_shape(x.get_shape())
    return wrap

    
saver = tf.train.Saver()

val_max_steps = int(val_nums / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


  #########################################Added by Ding  
    print("aaaaaaaaaaaaaaaaa=+++++++++++++++++++++++++++++")
    saver.restore(sess,'models/caffe_ilsvrc12/best_models_71200_0.7500.ckpt')
    print("aaaaaaaaaaaaaaaaa=************************************")

    adv_x = generate_deepfool(sess, input_images, out, nb_candidate=10,overshoot=0.02,
                              max_iter=50,clip_min=-0.5,clip_max=0.5)

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
      adv_bat = sess.run(adv_x, feed_dict=feed_dict)
      
      feed_dict = {input_images: adv_bat,  is_training: False}
      logits_adv_bat = sess.run(out, feed_dict=feed_dict)
        
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




