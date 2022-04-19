"""
This tutorial shows how to generate transferable attacks from adversarial examples of the source model
The target model is ResNet50.
The target model requires the input with the size of 128*128, therefore, the source examples are resized for implementation.

Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""
import numpy as np
import tensorflow as tf

import slim.nets.resnet_v1 as resnet_v1
import tensorflow.contrib.slim as slim
from cifar10_create_tf_record import get_example_nums,read_records,get_batch_images

from cleverhans.compat import reduce_max, reduce_sum, softmax_cross_entropy_with_logits

from cleverhans.data_exten_mulpro import data_exten#Added by Ding
from sklearn.metrics import roc_curve, roc_auc_score#Added by Ding
import multiprocessing
import cv2

#import copy
#from tensorflow.python import pywrap_tensorflow

batch_size = 20  #

labels_nums = 10  #  the number of labels
resize_height = 32  # Cifar10 size
resize_width = 32  
net_height = 128#resnet size
net_width = 128
depths = 3



input_images = tf.placeholder(dtype=tf.float32, shape=[None, net_height, net_width, depths], name='input')

input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')
is_training = tf.placeholder(tf.bool, name='is_training')

#test data
val_record_file='./cifar10_extensions/cifar10_test.tfrecords_seg'
val_nums=get_example_nums(val_record_file)
print('val nums:%d'%(val_nums))
#    val_images, val_labels = read_records([val_record_file], resize_height, resize_width, type='normalization')
val_images, val_labels = read_records([val_record_file], resize_height, resize_width, type='normalization')
val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False,num_threads=1)
val_images_batch = tf.image.resize_images(val_images_batch,size=(net_height, net_width))
val_images_batch = tf.rint(val_images_batch*256.)*(1. / 256)

# Define the model:
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    out, end_points = resnet_v1.resnet_v1_50(inputs=input_images, num_classes=labels_nums, is_training=is_training)
#    out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, is_training=is_training)
probs = tf.nn.softmax(out)
tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)
loss = tf.losses.get_total_loss(add_regularization_losses=True)

#accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))
accuracy = tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1))

saver = tf.train.Saver()

val_max_steps = int(val_nums / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


  #########################################Added by Ding  
    print("aaaaaaaaaaaaaaaaa=+++++++++++++++++++++++++++++")
    saver.restore(sess,'./models/cifar10/resnet50_models_2362500_0.9993.ckpt')#target model
    print("aaaaaaaaaaaaaaaaa=************************************")
    adv = np.load("cifar10_vgg16_augmodel_deepfool_10000adv.npy")#source examples
    adv_cnn = np.zeros((len(adv),net_height,net_width,depths))
    for i in range(len(adv)):
      advi = adv[i,:,:,:]
      adv_cnn[i,:,:,:] = cv2.resize(advi,(net_height,net_width))
#      plt.imshow(np.uint8(adv_cnn[i,:,:,:]*255)) 
    adv = adv_cnn    
    
    x_test = np.zeros((val_nums,net_height,net_width,depths),dtype=np.float32)
    y_test = np.zeros((val_nums,labels_nums),dtype=np.float32)
    logits = np.zeros((val_nums,labels_nums),dtype=np.float32)
    logits_adv = np.zeros((val_nums,labels_nums),dtype=np.float32)
    for i in range(val_max_steps):
        if i%10 == 0:
            print("i:{}".format(i))
        val_x_bat, val_y_bat = sess.run([val_images_batch, val_labels_batch])
        
        feed_dict = {input_images: val_x_bat,  is_training: False}
        logits_bat = sess.run(out, feed_dict=feed_dict)
        
        adv_bat = adv[i*batch_size:(i+1)*batch_size,:,:,:]
        
        feed_dict = {input_images: adv_bat, is_training: False}
        logits_adv_bat = sess.run(out,feed_dict=feed_dict)
#        loss_adv = sess.run(loss_x,feed_dict=feed_dict)

        x_test[i*batch_size:(i+1)*batch_size,:,:,:] = val_x_bat
        y_test[i*batch_size:(i+1)*batch_size,:] = val_y_bat
        logits[i*batch_size:(i+1)*batch_size,:] = logits_bat
        logits_adv[i*batch_size:(i+1)*batch_size,:] = logits_adv_bat
  #########################################
    coord.request_stop()
    coord.join(threads)

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
    
    pad_size = 13
    left_cols=np.arange(pad_size)
    left_cols=left_cols[::-1]
    right_cols=np.arange(net_height-pad_size,net_height)
    right_cols=right_cols[::-1]
    top_rows=np.arange(pad_size,pad_size*2)
    top_rows=top_rows[::-1]
    foot_rows=np.arange(net_height,net_height+pad_size)
    foot_rows=foot_rows[::-1]
#    x_test=np.pad(x_test,((0,0),(pad_size,pad_size),(pad_size,pad_size),(0,0)),'reflect')#very slow, and memery consume is too large
#    x_testcrop = np.zeros((len(x_test),net_height,net_width,3),dtype=np.float32)
#    adv = np.round(adv*256)/256.0#very slow, and memery consume is too large
#    adv = np.pad(adv,((0,0),(pad_size,pad_size),(pad_size,pad_size),(0,0)),'reflect')
    temp_test = np.zeros((net_height+2*pad_size,net_width+2*pad_size,3))
    temp_adv = np.zeros((net_height+2*pad_size,net_width+2*pad_size,3))
#    advcrop = np.zeros((len(adv),net_height,net_width,3),dtype=np.float32)
    for i in range(len(adv)):
      tf_image = adv[i,:,:,:]
      tf_image = np.round(tf_image*256)/256.0
      temp_adv[pad_size:net_height+pad_size,pad_size:net_width+pad_size,:] = tf_image
      temp_adv[pad_size:net_height+pad_size, :pad_size, :]=tf_image[:, left_cols, :]#reflect left
      temp_adv[pad_size:net_height+pad_size, net_height+pad_size:, :]=tf_image[:, right_cols, :]#reflect right
      temp_adv[:pad_size, :, :]=temp_adv[top_rows, :, :]
      temp_adv[net_height+pad_size:, :, :]=temp_adv[foot_rows, :, :]
      tf_image = temp_adv
#      tf_image = np.pad(tf_image,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'reflect')
      test_image = x_test[i,:,:,:]
      temp_test[pad_size:net_height+pad_size,pad_size:net_width+pad_size,:] = test_image
      temp_test[pad_size:net_height+pad_size, :pad_size, :]=test_image[:, left_cols, :]#reflect left
      temp_test[pad_size:net_height+pad_size, net_height+pad_size:, :]=test_image[:, right_cols, :]#reflect right
      temp_test[:pad_size, :, :]=temp_test[top_rows, :, :]
      temp_test[net_height+pad_size:, :, :]=temp_test[foot_rows, :, :]
      test_image = temp_test
#      test_image = np.pad(test_image,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'reflect')
      lu1 = np.random.randint(0,pad_size*2)
      lu2 = np.random.randint(0,pad_size*2)
      adv[i,:,:,:] = tf_image[lu1:lu1+net_height,lu2:lu2+net_width,:]
      x_test[i,:,:,:] = test_image[lu1:lu1+net_height,lu2:lu2+net_width,:]
#    adv = advcrop
#    x_test = x_testcrop
    
    batch_size = 10  #
    base_range=4
    n_pert = base_range**depths
    ext_bat = n_pert+1
    
    logits_ext = np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    logits_adv_ext = np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    test_prob_pertpart=np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    adv_prob_pertpart=np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    y_test_pertpart = np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    y_adv_pertpart = np.zeros((val_nums*n_pert,labels_nums),dtype=np.float32)
    x_adv_pertpart = np.zeros((batch_size*n_pert*2,net_height,net_width,depths),dtype=np.float32)
    x_test_pertpart = np.zeros((batch_size*n_pert*2,net_height,net_width,depths),dtype=np.float32)
    val_max_steps = int(len(adv) / batch_size/2)
    
#    logits_adv = np.zeros((val_nums,labels_nums),dtype=np.float32)
    adv_prob_legit = np.zeros((val_nums,labels_nums),dtype=np.float32)
#    logits = logits_adv
    test_prob_legit = np.zeros((val_nums,labels_nums),dtype=np.float32)
    
    manager=multiprocessing.Manager()
  
    for i in range(val_max_steps):
#      x_adv_extended, y_adv_extended = data_exten(adv[i*batch_size:(i+1)*batch_size,:,:,:], 
#                                                    y_test[i*batch_size:(i+1)*batch_size,:], 
#                  batch_size, base_range=base_range,img_rows=net_height, img_cols=net_width)#
#      y_test_copy=copy.deepcopy(y_test[i*batch_size:(i+1)*batch_size,:])
#      x_test_copy=copy.deepcopy(x_test[i*batch_size:(i+1)*batch_size,:,:,:])
#      print("i:{}".format(i))
      rt_res_adv1=manager.dict()
      rt_res_adv2=manager.dict()
      rt_res_test1=manager.dict()
      rt_res_test2=manager.dict()
      p1 = multiprocessing.Process(target=data_exten,args=(adv[i*2*batch_size:(2*i+1)*batch_size,:,:,:],
                                            y_test[2*i*batch_size:(2*i+1)*batch_size,:], 
                                            batch_size, base_range,labels_nums,net_height, net_width,3,
                                            rt_res_adv1))
      p2 = multiprocessing.Process(target=data_exten,args=(adv[(2*i+1)*batch_size:2*(i+1)*batch_size,:,:,:],
                                            y_test[(2*i+1)*batch_size:2*(i+1)*batch_size,:], 
                                            batch_size, base_range,labels_nums,net_height, net_width,3,
                                            rt_res_adv2))
#      x_test_extended, y_test_extended = data_exten(x_test[i*batch_size:(i+1)*batch_size,:,:,:], 
#                                                      y_test[i*batch_size:(i+1)*batch_size,:], 
#                  batch_size, base_range=base_range,img_rows=net_height, img_cols=net_width)
      p3 = multiprocessing.Process(target=data_exten,args=(x_test[2*i*batch_size:(2*i+1)*batch_size,:,:,:],
                                            y_test[2*i*batch_size:(2*i+1)*batch_size,:], 
                                            batch_size, base_range,labels_nums,net_height, net_width,3,
                                            rt_res_test1))
      p4 = multiprocessing.Process(target=data_exten,args=(x_test[(2*i+1)*batch_size:2*(i+1)*batch_size,:,:,:],
                                            y_test[(2*i+1)*batch_size:2*(i+1)*batch_size,:], 
                                            batch_size, base_range,labels_nums,net_height, net_width,3,
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
#      logits_adv[2*i*batch_size:2*(i+1)*batch_size,:] = sess.run(out, feed_dict=feed_dict)
      adv_prob_legit[2*i*batch_size:2*(i+1)*batch_size,:] = sess.run(probs,feed_dict = feed_dict)
      feed_dict = {input_images: x_test[2*i*batch_size:2*(i+1)*batch_size,:,:,:],  is_training: False}
#      logits[2*i*batch_size:2*(i+1)*batch_size,:] = sess.run(out, feed_dict=feed_dict)
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
#        loss_adv = sess.run(loss_x,feed_dict=feed_dict)

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

#    logits = np.argmax(logits,axis=1)
#    logits_adv = np.argmax(logits_adv,axis=1)
#    y_test = np.argmax(y_test,axis=1)
#    acc = np.sum(np.equal(logits,y_test))/len(y_test)
#    acc_adv = np.sum(np.equal(logits_adv,y_test))/len(y_test)
#    print('Test accuracy on legitimate test examples: %0.4f' % (acc))
#    print('Test accuracy on adversarial test examples: %0.4f' % (acc_adv))
    
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