#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:21:33 2020
Freeze meta graph to pb file
@author: Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""

import os
import tensorflow as tf

from tensorflow.python import pywrap_tensorflow

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file 
# Get the current directory
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
def freeze_graph(input_checkpoint,output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: pb file
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #check the status of ckpt file
 
    # the output of the network of ckpt
#    output_node_names = "resnet_v1_50/SpatialSqueeze"#InceptionV3/Logits/SpatialSqueeze,InceptionV3/AuxLogits/SpatialSqueeze"
#    output_node_names = "vgg_16/fc8/squeezed"#InceptionV3/Logits/SpatialSqueeze,InceptionV3/AuxLogits/SpatialSqueeze"
    output_node_names = "InceptionV3/Logits/SpatialSqueeze,InceptionV3/AuxLogits/SpatialSqueeze"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 
    input_graph_def = graph.as_graph_def()  # 
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #
#        print(sess.run('InceptionV3/Conv2d_1a_3x3/weights:0'))
#        print(sess.run('InceptionV3/Conv2d_1a_3x3/BatchNorm/moving_variance:0'))
        output_graph_def = graph_util.convert_variables_to_constants(  # 
            sess=sess,
            input_graph_def=input_graph_def,# µÈÓÚ:sess.graph_def
            output_node_names=output_node_names.split(","))# if there are more than one node, the multiple output nodes should be split with ","
 
        with tf.gfile.GFile(output_graph, "wb") as f: 
            f.write(output_graph_def.SerializeToString()) 
        print("%d ops in the final graph." % len(output_graph_def.node))
 
        # for op in graph.get_operations():
        #     print(op.name, op.values())


dir_path = os.path.dirname(os.path.realpath(__file__))
print
["Current directory : ", dir_path]
save_dir = dir_path + '/models'
     
graph = tf.get_default_graph()
     
# Create a session for running Ops on the Graph.
#ckpt_f_name='./cifar10_extensions/resnet50_64x_aug_train2400k_iters/trained_unnormed2_256levels/best_models_2362500_0.9993.ckpt'
#ckpt_f_name='./cifar10_extensions/resnet50_clean_1000k_iters/best_models_987000_0.9069.ckpt'
ckpt_f_name='models/caffe_ilsvrc12/inceptv3_best_models_1104800_0.8080.ckpt'
reader=pywrap_tensorflow.NewCheckpointReader(ckpt_f_name)
var_to_shape_map=reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print('tensor_name:{}'.format(key))

#print_tensors_in_checkpoint_file(ckpt_f_name,tensor_name='vgg_16/fc8/squeezed',all_tensors=False)    

print_tensors_in_checkpoint_file(ckpt_f_name,tensor_name='InceptionV3/Conv2d_1a_3x3/BatchNorm/moving_mean',all_tensors=False)    

input_checkpoint=ckpt_f_name
#out_pb_path=save_dir + '/imagenet_inceptv3_best_models_682600_0.7458.pb'
#out_pb_path=save_dir + '/best_models_987000_0.9069.pb'
out_pb_path=save_dir + '/inceptv3_best_models_1104800_0.8080.pb'
#out_pb_path=save_dir + '/imagenet_vgg16_clean1315000_0.7333.pb'
freeze_graph(input_checkpoint,out_pb_path)

print("Saving Done .. ")
