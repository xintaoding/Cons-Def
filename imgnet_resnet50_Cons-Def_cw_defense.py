"""
This tutorial shows how to implement Cons-Def against C&W white-box attacks.
Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""
import numpy as np
import tensorflow as tf

#import slim.nets.inception_v3 as inception_v3
#import tensorflow.contrib.slim as slim
from create_tf_record import get_example_nums,read_records,get_batch_images

from cleverhans import utils_tf

from cleverhans.data_exten_mulpro import data_exten#Added by Ding
from sklearn.metrics import roc_curve, roc_auc_score#Added by Ding
#from tensorflow.python import pywrap_tensorflow
import multiprocessing

batch_size = 20  #

labels_nums = 10  #  the number of labels
resize_height = 299
resize_width = 299
net_height = 224 #ResNet50 size
net_width = 224
depths = 3

input_images = tf.placeholder(dtype=tf.float32, shape=[None, net_height, net_width, depths])
#input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums])
#is_training = tf.placeholder(tf.bool, name='is_training')
is_training = tf.placeholder(tf.bool)

#test data
val_record_file='data/caffe_ilsvrc12_record/val299.tfrecords'
val_nums=get_example_nums(val_record_file)
print('val nums:%d'%(val_nums))
val_images, val_labels = read_records([val_record_file], resize_height, resize_width, type='centralization')
#val_images, val_labels = read_records([val_record_file], resize_height, resize_width, type='normalization')
val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False,num_threads=1)
#val_images_batch = tf.image.resize_images(val_images_batch,size=(net_height, net_width))
#val_images_batch = tf.rint(val_images_batch*256.)*(1./256)
val_images_batch=val_images_batch[:,37:261,37:261,:]


TARGETED = False
if TARGETED:
    cw_params_batch_size = batch_size * labels_nums
else:
    cw_params_batch_size = batch_size

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

def ZERO():
  return np.asarray(0., dtype=np_dtype)

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(
    #  FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
      'models/caffe_ilsvrc12/resnet50_best_models_1393000_0.7667.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    #for line in repr(graph_def).split("\n"):
    #  if "tensor_content" not in line:
    #    print(line)
    _ = tf.import_graph_def(graph_def, name='')
            

class InceptionModelPrediction:
  def __init__(self, sess, use_logits = True):
    self.sess = sess
    self.use_logits = use_logits
    if self.use_logits:
#      output_name = 'InceptionV3/Predictions/Reshape:0'
#      print("dddddddddddddddddddddddddd")
#      assert 1==2
      output_name = "resnet_v1_50/SpatialSqueeze:0"
    else:
      output_name = 'resnet_v1_50/SpatialSqueeze:0'
    self.img = tf.placeholder(tf.float32, (None, 224,224,3))
    self.softmax_tensor = tf.import_graph_def(
            sess.graph.as_graph_def(),
            input_map={'input:0': self.img, 'is_training:0': False},#scaled down model, restored from meta
#            input_map={'input:0': self.img},#frozen pb model without training indication because frozen model cannot be retrained
            return_elements=[output_name])
  def predict(self, dat):
    dat = np.squeeze(dat)
    # scaled = (0.5 + dat) * 255
    scaled = dat.reshape((1,) + dat.shape)
    predictions = self.sess.run(self.softmax_tensor,
                         {self.img: scaled, 'is_training:0': False})#scaled down model, restored from meta
#                         {self.img: scaled})#frozen pb model without training indication because frozen model cannot be retrained
    predictions = np.squeeze(predictions)
    return predictions


CREATED_GRAPH = False
class InceptionModel:
  global labels_nums
  num_labels = labels_nums
  num_channels = 3
  def __init__(self, sess, use_logits = True):
    global CREATED_GRAPH
    self.sess = sess
    self.use_logits = use_logits
    if not CREATED_GRAPH:
      create_graph()
#      print(sess.run('InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean:0'))
#      assert 1==2
#      tf.summary.FileWriter('./summary',sess.graph)
      CREATED_GRAPH = True
    self.model = InceptionModelPrediction(sess, use_logits)

  def predict(self, img):
    if self.use_logits:
#      output_name = 'InceptionV3/Predictions/Reshape:0'
      output_name = "resnet_v1_50/SpatialSqueeze:0"
    else:
      output_name = 'resnet_v1_50/SpatialSqueeze:0'
    # scaled = (0.5+tf.reshape(img,((299,299,3))))*255
    # scaled = (0.5+img)*255
    print("Resnet50.predict:{},img.shape:{}".format(output_name,img.shape))
    if img.shape.as_list()[0]:
      # check if a shape has been specified explicitly
      softmax_tensor = tf.import_graph_def(
        self.sess.graph.as_graph_def(),
        input_map={'input:0': img, 'is_training:0': False},#scaled down model, restored from meta
        return_elements=[output_name])
    else:
      # placeholder shape
      softmax_tensor = tf.import_graph_def(
        self.sess.graph.as_graph_def(),
        input_map={'input:0': img, 'is_training:0': False},#scaled down model, restored from meta
        return_elements=[output_name])
    print("softmax_tensor[0] shape:{}".format(softmax_tensor[0]))
    return softmax_tensor[0]

class CWL2(object):
  def __init__(self, sess, model, batch_size,net_height, net_width, depths):
#    yname = adv_ys
    self.sess = sess
    self.confidence = 0
    self.learning_rate = 5e-3
    self.BINARY_SEARCH_STEPS = BINARY_SEARCH_STEPS = 1 #'binary_search_steps': 9,#'binary_search_steps': 1,#9 is the CW author proposed, 1 is the cleverhans setting

    self.MAX_ITERATIONS = 1000
    self.abort_early = True
    self.clip_min = clip_min = -0.5
    self.clip_max = clip_max = 0.5

    self.initial_const = 10#'initial_const': 10#10 is the default set of cleverhans,0.001 is the CW author proposed

    self.CONFIDENCE = 0
    self.batch_size = batch_size
    self.repeat = BINARY_SEARCH_STEPS >= 10
    shape = (batch_size, net_height, net_width, depths)
    # the variable we're going to optimize over
    modifier = tf.Variable(np.zeros(shape, dtype=np.float32))
    # these are variables to be more efficient in sending data to tf
    self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
    self.tlab = tf.Variable(np.zeros((batch_size, labels_nums)), dtype=tf.float32)
    self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

    # and here's what we use to assign them
    self.assign_timg = tf.placeholder(tf.float32, shape)
    self.assign_tlab = tf.placeholder(tf.float32, (batch_size,labels_nums))
    self.assign_const = tf.placeholder(tf.float32, [batch_size])

    # the resulting instance, tanh'd to keep bounded from clip_min
    # to clip_max
    self.newimg = (tf.tanh(modifier + self.timg) + 1) / 2
    self.newimg = self.newimg * (clip_max - clip_min) + clip_min

  # prediction BEFORE-SOFTMAX of the model
    self.output = model.predict(self.newimg)

#    output = end_points['Predictions']

  # distance to the input data
    self.other = (tf.tanh(self.timg) + 1) / \
        2 * (clip_max - clip_min) + clip_min
    self.l2dist = tf.reduce_sum(tf.square(self.newimg - self.other), list(range(1, len(shape))))

  # compute the probability of the label class versus the maximum other
    real = tf.reduce_sum((self.tlab) * self.output, 1)
    other = tf.reduce_max((1 - self.tlab) * self.output - self.tlab * 10000, 1)
    self.real = real
    self.other = other

    if TARGETED:
    # if targeted, optimize for making the other class most likely
      ini_loss1 = tf.maximum(ZERO(), other - real + self.CONFIDENCE)
    else:
    # if untargeted, optimize for making this class least likely.
      ini_loss1 = tf.maximum(ZERO(), real - other + self.CONFIDENCE)

  # sum up the losses
    self.loss2 = tf.reduce_sum(self.l2dist)
    self.loss1 = tf.reduce_sum(self.const * ini_loss1)
    self.loss = self.loss1 + self.loss2

  # Setup the adam optimizer and keep track of variables we're creating
    start_vars = set(x.name for x in tf.global_variables())
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.train = optimizer.minimize(self.loss, var_list=[modifier])
#    add_modifier = modifier.assign_add(np.ones(shape))
    
    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]

  # these are the variables to initialize when we run
    self.setup = []
    self.setup.append(self.timg.assign(self.assign_timg))
    self.setup.append(self.tlab.assign(self.assign_tlab))
    self.setup.append(self.const.assign(self.assign_const))
#    setup.append(output.assign(assign_output))

    self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

  
  def attack(self, imgs, labs):

    def compare(x, y):
      if not isinstance(x, (float, int, np.int64)):
        global TARGETED
        x = np.copy(x)
        if TARGETED:
          x[y] -= self.CONFIDENCE
        else:
          x[y] += self.CONFIDENCE
        x = np.argmax(x)
      if TARGETED:
        return x == y
      else:
        return x != y


    oimgs = np.clip(imgs, self.clip_min, self.clip_max)

# re-scale instances to be within range [0, 1]
    imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
    imgs = np.clip(imgs, 0, 1)
  # now convert to [-1, 1]
    imgs = (imgs * 2) - 1
  # convert to tanh-space
    imgs = np.arctanh(imgs * .999999)
    
    batch_size = self.batch_size

  # set the lower and upper bounds accordingly
    lower_bound = np.zeros(batch_size)
    CONST = np.ones(batch_size) * self.initial_const
    upper_bound = np.ones(batch_size) * 1e10

  # placeholders for the best l2, score, and instance attack found so far
    o_bestl2 = [1e10] * batch_size
    o_bestscore = [-1] * batch_size
    o_bestattack = np.copy(oimgs)
  
    for outer_step in range(self.BINARY_SEARCH_STEPS):
      # completely reset adam's internal state.
      self.sess.run(self.init)
      batch = imgs[:batch_size]
      batchlab = labs[:batch_size]

      bestl2 = [1e10] * batch_size
      bestscore = [-1] * batch_size
      print("  Binary search step {} of {}".format(outer_step, self.BINARY_SEARCH_STEPS))

      # The last iteration (if we run many steps) repeat the search once.
      if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
        CONST = upper_bound
        

      # set the variables so that we don't have to send them over again
      self.sess.run(
          self.setup, {
              self.assign_timg: batch,
              self.assign_tlab: batchlab,
              self.assign_const: CONST#,
#              assign_output: output_host
          })

      prev = 1e20
      for iteration in range(self.MAX_ITERATIONS):
        # perform the attack
        scores, llllbbbb = self.sess.run([self.output, self.tlab])
#        print("scores:{}".format(np.sum(scores)))
        _, l, l1, other,real, l2, l2s, scores, nimg, timage = self.sess.run([
            self.train, self.loss, self.loss1, self.other,self.real, self.loss2, self.l2dist, self.output,
            self.newimg, self.timg])
#        add_modifier = modifier.assign_add(np.ones(shape))
#        print("nimg:{}".format(np.sum(nimg)))
#        if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
#          _logger.debug(("    Iteration {} of {}: loss={:.3g},loss1={:.3g},other={},real={}, loss2={:.3g} " +
#                         "l2={:.3g} f={:.3g}").format(
#                             iteration, self.MAX_ITERATIONS, l, l1,other, real, l2,
#                             np.mean(l2s), np.mean(scores)))
          
#          real = np.sum((batchlab) * scores, 1)
#          other = np.sum((1 - batchlab) * scores - batchlab * 10000, 1)
#          loss1 = np.maximum(ZERO(), real - other + self.CONFIDENCE)
#          loss11= np.sum(CONST * loss1)

#          print("loss1:{},loss11:{}".format(loss1,loss11))


        # check if we should abort search if we're getting nowhere.
        self.ABORT_EARLY=False
        if self.ABORT_EARLY and \
           iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
          print("===================l:{},prev:{}".format(l,prev))
          if l > prev * .9999:
            msg = "    Failed to make progress; stop early"
            print(msg)
            break
          prev = l

        # adjust the best result found so far
        for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
#          print('e:{}'.format(e))
          lab = np.argmax(batchlab[e])
          if l2 < bestl2[e] and compare(sc, lab):
            bestl2[e] = l2
            bestscore[e] = np.argmax(sc)
          if l2 < o_bestl2[e] and compare(sc, lab):
            o_bestl2[e] = l2
            o_bestscore[e] = np.argmax(sc)
            o_bestattack[e] = ii

      # adjust the constant as needed
      for e in range(batch_size):
        if compare(bestscore[e], np.argmax(batchlab[e])) and \
           bestscore[e] != -1:
          # success, divide const by two
          upper_bound[e] = min(upper_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
        else:
          # failure, either multiply by 10 if no solution found yet
          #          or do binary search with the known upper bound
          lower_bound[e] = max(lower_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
          else:
            CONST[e] *= 10#revised by Ding
      print("  Successfully generated adversarial examples  on {} of {} instances.".format(sum(upper_bound < 1e9), batch_size))
      o_bestl2 = np.array(o_bestl2)
      mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
      print("   Mean successful distortion: {}".format(mean))

    # return the best solution found
    o_bestl2 = np.array(o_bestl2)
    return o_bestattack


    
val_max_steps = int(val_nums / batch_size)
base_range=4
n_pert = base_range**depths   

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  model = InceptionModel(sess)#import a pb graphy using a class initialization
  preds = model.predict(input_images)
  probs = tf.nn.softmax(preds)
  cw=CWL2(sess, model, batch_size,net_height, net_width, depths)#initial cw attach

  x_test = np.zeros((val_nums,net_height,net_width,depths),dtype=np.float32)
  y_test = np.zeros((val_nums,labels_nums),dtype=np.float32)
  logits = np.zeros((val_nums,labels_nums),dtype=np.float32)
  logits_adv = np.zeros((val_nums,labels_nums),dtype=np.float32)
  adv = np.zeros((val_nums,net_height,net_width,depths),dtype=np.float32)
  for i in range(val_max_steps):
        print("i:{}".format(i))
        val_x_bat, val_y_bat = sess.run([val_images_batch, val_labels_batch])

        adv_bat = cw.attack(val_x_bat, val_y_bat)
        logits_bat = sess.run([preds],feed_dict = {input_images: val_x_bat, is_training: False})
        logits_adv_bat = sess.run([preds],feed_dict = {input_images: adv_bat, is_training: False})
        logits_adv_bat = np.array(logits_adv_bat[0])
        logits_bat = np.array(logits_bat[0])
        
        val_acc = np.equal(np.argmax(logits_bat, axis=1), np.argmax(val_y_bat, axis=1))
        val_adv_acc = np.equal(np.argmax(logits_adv_bat, axis=1), np.argmax(val_y_bat, axis=1))
        
        x_test[i*batch_size:(i+1)*batch_size,:,:,:] = val_x_bat
        y_test[i*batch_size:(i+1)*batch_size,:] = val_y_bat
        adv[i*batch_size:(i+1)*batch_size,:,:,:] = adv_bat#Ranged in [0, 1]
        logits[i*batch_size:(i+1)*batch_size,:] = logits_bat
        logits_adv[i*batch_size:(i+1)*batch_size,:] = logits_adv_bat
  #########################################
  coord.request_stop()
  coord.join(threads)
  
  np.save("cifar10_ResNet50_augmodel_cwl2_10000adv",adv)#save advs
  
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
  
  pad_size=22
  x_test=np.pad(x_test,((0,0),(pad_size,pad_size),(pad_size,pad_size),(0,0)),'reflect')
  x_testcrop = np.zeros((len(x_test),net_height,net_width,3),dtype=np.float32)
  adv = np.round(adv*256)/256.0
  adv = np.pad(adv,((0,0),(pad_size,pad_size),(pad_size,pad_size),(0,0)),'reflect')
  advcrop = np.zeros((len(adv),net_height,net_width,3),dtype=np.float32)
  for i in range(len(adv)):
      tf_image = adv[i,:,:,:]
      test_image = x_test[i,:,:,:]
      lu1 = np.random.randint(0,pad_size*2)
      lu2 = np.random.randint(0,pad_size*2)
      advcrop[i,:,:,:] = tf_image[lu1:lu1+net_height,lu2:lu2+net_width,:]
      x_testcrop[i,:,:,:] = test_image[lu1:lu1+net_height,lu2:lu2+net_width,:]
  adv = advcrop
  x_test = x_testcrop
  
  batch_size = 10
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
  
  adv_prob_legit = np.zeros((val_nums,labels_nums),dtype=np.float32)
  test_prob_legit = np.zeros((val_nums,labels_nums),dtype=np.float32)
    
  manager=multiprocessing.Manager()
  
  for i in range(val_max_steps):
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
        logits_bat = sess.run([preds], feed_dict=feed_dict)
        logits_bat = np.array(logits_bat[0])
        
        feed_dict = {input_images: val_adv_bat, is_training: False}#range to [-0.5, 0.5]
        logits_adv_bat = sess.run([preds],feed_dict=feed_dict)
        logits_adv_bat = np.array(logits_adv_bat[0])

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
  y_test = np.argmax(y_test,axis=1)
  acc = np.sum(np.equal(logits,y_test))/len(y_test)
  acc_adv = np.sum(np.equal(logits_adv,y_test))/len(y_test)
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

  
