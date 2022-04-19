"""
This tutorial shows how to implement Cons-Def against FGSM white-box attacks.
Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
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

from cleverhans.data_exten_mulpro import data_exten#Added by Ding
from sklearn.metrics import roc_curve, roc_auc_score#Added by Ding
import multiprocessing
import matplotlib.pyplot as plt
#from imgnet_incepv3_test_dataext_fgsm_defense import fgm
#from tensorflow.python import pywrap_tensorflow


batch_size = 20  #
labels_nums = 10  #  the number of labels
resize_height = 299  # imagenet size
resize_width = 299
net_height = resize_height
net_width = resize_width
depths = 3


input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')
y = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums])
#for tensorflow1.2, keep_prob cannot be defined as placeholder, it must be a scalar and it needn't be fed to session
keep_prob = 0.5#tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')


#test data
val_record_file='data/caffe_ilsvrc12_record/val299.tfrecords'
val_nums=get_example_nums(val_record_file)
print('val nums:%d'%(val_nums))
#    val_images, val_labels = read_records([val_record_file], resize_height, resize_width, type='normalization')
val_images, val_labels = read_records([val_record_file], resize_height, resize_width, type='centralization')
val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False,num_threads=1)
#val_images_batch = tf.image.resize_images(val_images_batch,size=(resize_height, resize_width))
#val_images_batch = tf.rint(val_images_batch*255.)*(1. / 255)

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
clip_min=-0.50
clip_max=0.50
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
    saver.restore(sess,'models/caffe_ilsvrc12/inceptv3_best_models_682600_0.7458.ckpt')
    print("aaaaaaaaaaaaaaaaa=************************************")

   # Initialize the Projected Gradient Descent Method (PGDM) attack object and
    # graph
    #adv=np.load("imagenet_inceptv3_augmodel_pgd_10000adv.npy")#load advs
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
        
      x_test[i*batch_size:(i+1)*batch_size,:,:,:] = val_x_bat
      y_test[i*batch_size:(i+1)*batch_size,:] = val_y_bat
      adv[i*batch_size:(i+1)*batch_size,:,:,:] = adv_bat#Ranged in [0, 1]
      logits[i*batch_size:(i+1)*batch_size,:] = logits_bat
      logits_adv[i*batch_size:(i+1)*batch_size,:] = logits_adv_bat
  #########################################
    coord.request_stop()
    coord.join(threads)

    
#    np.save("imagenet_inceptv3_augmodel_pgd_10000adv",adv)#save advs
    
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
#          dsae+=np.sum((trans_adv[i,:,:,:] - x_test[i,:,:,:])**2)**.5
          kk += 1
    dsae=dsae/kk
    print("For untargeted attack, the number of misclassified examples (successful attack), sum(adv_suc_att_exam==0):{}, dsae:{}".format(sum(adv_suc_att_exam==0),dsae))

    print('Avg. L_2 norm of perturbations {}'.format(percent_perturbed))
    print('The number of successful attack:{}, Avg. L_2 norm of perturbations on successful attack / dsae:{}'.format(kk,dsae))
    
    pad_size = 30
    left_cols=np.arange(pad_size)
    left_cols=left_cols[::-1]
    right_cols=np.arange(net_height-pad_size,net_height)
    right_cols=right_cols[::-1]
    top_rows=np.arange(pad_size,pad_size*2)
    top_rows=top_rows[::-1]
    foot_rows=np.arange(net_height,net_height+pad_size)
    foot_rows=foot_rows[::-1]
    temp_test = np.zeros((net_height+2*pad_size,net_width+2*pad_size,3))
    temp_adv = np.zeros((net_height+2*pad_size,net_width+2*pad_size,3))
    for i in range(len(adv)):
      tf_image = adv[i,:,:,:]
      tf_image = np.round(tf_image*256)/256.0
      temp_adv[pad_size:net_height+pad_size,pad_size:net_width+pad_size,:] = tf_image
      temp_adv[pad_size:net_height+pad_size, :pad_size, :]=tf_image[:, left_cols, :]#reflect left
      temp_adv[pad_size:net_height+pad_size, net_height+pad_size:, :]=tf_image[:, right_cols, :]#reflect right
      temp_adv[:pad_size, :, :]=temp_adv[top_rows, :, :]
      temp_adv[net_height+pad_size:, :, :]=temp_adv[foot_rows, :, :]
      tf_image = temp_adv
      test_image = x_test[i,:,:,:]
      temp_test[pad_size:net_height+pad_size,pad_size:net_width+pad_size,:] = test_image
      temp_test[pad_size:net_height+pad_size, :pad_size, :]=test_image[:, left_cols, :]#reflect left
      temp_test[pad_size:net_height+pad_size, net_height+pad_size:, :]=test_image[:, right_cols, :]#reflect right
      temp_test[:pad_size, :, :]=temp_test[top_rows, :, :]
      temp_test[net_height+pad_size:, :, :]=temp_test[foot_rows, :, :]
      test_image = temp_test
      lu1 = np.random.randint(0,pad_size*2)
      lu2 = np.random.randint(0,pad_size*2)
      adv[i,:,:,:] = tf_image[lu1:lu1+net_height,lu2:lu2+net_width,:]
      x_test[i,:,:,:] = test_image[lu1:lu1+net_height,lu2:lu2+net_width,:]

    batch_size = 5  
    base_range=4
    n_pert = base_range**depths
    ext_bat = n_pert+1
    
    logits_ext = np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    logits_adv_ext = np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    test_prob_pertpart=np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    adv_prob_pertpart=np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    y_test_pertpart = np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    y_adv_pertpart = np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    x_adv_pertpart = np.zeros((batch_size*n_pert*2,resize_height,resize_width,depths),dtype=np.float32)
    x_test_pertpart = np.zeros((batch_size*n_pert*2,resize_height,resize_width,depths),dtype=np.float32)
    val_max_steps = int(len(adv) / batch_size/2)
    
    adv_prob_legit = np.zeros((val_nums,labels_nums),dtype=np.float32)
    test_prob_legit = np.zeros((val_nums,labels_nums),dtype=np.float32)
    
#    xxx=(x_adv_pertpart[60,:,:,:]+0.5)*255
#    xxx=xxx.astype(np.uint8)
#    plt.imshow(xxx)
#    xxx=(x_test_pertpart[60,:,:,:]+0.5)*255
#    xxx=xxx.astype(np.uint8)
#    plt.imshow(xxx)
    
    manager=multiprocessing.Manager()
  
    for i in range(val_max_steps):
      if i%10 == 0:
          print("i:{}".format(i))  
      rt_res_adv1=manager.dict()
      rt_res_adv2=manager.dict()
      rt_res_test1=manager.dict()
      rt_res_test2=manager.dict()
      p1 = multiprocessing.Process(target=data_exten,args=(adv[i*2*batch_size:(2*i+1)*batch_size,:,:,:],
                                            y_test[2*i*batch_size:(2*i+1)*batch_size,:], 
                                            batch_size, base_range,labels_nums,resize_height, resize_width,3,
                                            rt_res_adv1))
      p2 = multiprocessing.Process(target=data_exten,args=(adv[(2*i+1)*batch_size:2*(i+1)*batch_size,:,:,:],
                                            y_test[(2*i+1)*batch_size:2*(i+1)*batch_size,:], 
                                            batch_size, base_range,labels_nums,resize_height, resize_width,3,
                                            rt_res_adv2))
      p3 = multiprocessing.Process(target=data_exten,args=(x_test[2*i*batch_size:(2*i+1)*batch_size,:,:,:],
                                            y_test[2*i*batch_size:(2*i+1)*batch_size,:], 
                                            batch_size, base_range,labels_nums,resize_height, resize_width,3,
                                            rt_res_test1))
      p4 = multiprocessing.Process(target=data_exten,args=(x_test[(2*i+1)*batch_size:2*(i+1)*batch_size,:,:,:],
                                            y_test[(2*i+1)*batch_size:2*(i+1)*batch_size,:], 
                                            batch_size, base_range,labels_nums,resize_height, resize_width,3,
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
      feed_dict = {input_images: adv[2*i*batch_size:2*(i+1)*batch_size,:,:,:],  is_training: False}
      adv_prob_legit[2*i*batch_size:2*(i+1)*batch_size,:] = sess.run(probs,feed_dict = feed_dict)
      feed_dict = {input_images: x_test[2*i*batch_size:2*(i+1)*batch_size,:,:,:],  is_training: False}
      test_prob_legit[2*i*batch_size:2*(i+1)*batch_size,:] = sess.run(probs,feed_dict = feed_dict)

      l_bat=len(x_adv_pertpart)
      jsteps = int(l_bat/batch_size)
      for j in range(jsteps):
#        if j%10 == 0:
#            print("j:{}".format(j))
        val_x_bat = x_test_pertpart[j*batch_size:(j+1)*batch_size]
        val_adv_bat = x_adv_pertpart[j*batch_size:(j+1)*batch_size]

        feed_dict = {input_images: val_x_bat,  is_training: False}
        logits_bat = sess.run(out, feed_dict=feed_dict)
        
        feed_dict = {input_images: val_adv_bat, is_training: False}#range to [-0.5, 0.5]
        logits_adv_bat = sess.run(out,feed_dict=feed_dict)

        y_test_prob = sess.run(probs,feed_dict = {input_images: val_x_bat, is_training: False})
        y_adv_prob = sess.run(probs,feed_dict = {input_images: val_adv_bat, is_training: False})

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
    
    logits = np.argmax(logits,axis=1)
    logits_adv = np.argmax(logits_adv,axis=1)
    y_test_argmax = np.argmax(y_test,axis=1)
    acc = np.sum(np.equal(logits,y_test_argmax))/len(y_test_argmax)
    acc_adv = np.sum(np.equal(logits_adv,y_test_argmax))/len(y_test_argmax)
    print('Test accuracy on legitimate test examples: %0.4f' % (acc))
    print('Test accuracy on adversarial test examples: %0.4f' % (acc_adv))

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
      adv_result_stat[temp] = adv_result_stat[temp]+1#there is a inconsensus detection results of the 65 perturbations
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
      for j in range(labels_nums):
        if np.sum(ground_labels[:,j])==0:
            del_ind.append(j)
      del_ind = np.array(del_ind)
      ground_labels = np.delete(ground_labels,del_ind,axis=1)
      preded_probs = test_prob_pertpart[tuple(benign_inds),:]
      preded_probs = np.delete(preded_probs,del_ind,axis=1) 
      auc_score_clc_bn = roc_auc_score(ground_labels, preded_probs)
                              
      ground_labels_adv = y_adv_pertpart[tuple(adv_inds),:]
      del_ind=[]
      for j in range(labels_nums):
        if np.sum(ground_labels_adv[:,j])==0:
            del_ind.append(j)
      del_ind = np.array(del_ind)
      ground_labels_adv = np.delete(ground_labels_adv,del_ind,axis=1)
      preded_probs_adv = adv_prob_pertpart[tuple(adv_inds),:]
      preded_probs_adv = np.delete(preded_probs_adv,del_ind,axis=1) 
      auc_score_clc_adv = roc_auc_score(ground_labels_adv, preded_probs_adv)
#    auc_score_clc_adv = roc_auc_score(y_adv_pertpart[tuple(adv_inds),:], y_pertpart_prob_adv[tuple(adv_inds),:])
      print("(Threshold {}:) auc_score_clc_bn:{},auc_score_clc_adv:{}".format(eva_thresh[i],auc_score_clc_bn, auc_score_clc_adv))
    sess.close()
  #########################################

