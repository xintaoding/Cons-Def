"""Extremely simple model where all parameters are from convolutions.
"""

import math
import tensorflow as tf

from cleverhans import initializers
from cleverhans.serial import NoRefModel


class Model4Convolutional2FC(NoRefModel):
  """
  A simple model that uses only convolution and downsampling---no batch norm or other techniques that can complicate
  adversarial training.
  """
  def __init__(self, scope, nb_classes, nb_filters, input_shape, **kwargs):
    del kwargs
    NoRefModel.__init__(self, scope, nb_classes, locals())
    self.nb_filters = nb_filters
    self.input_shape = input_shape

    # Do a dummy run of fprop to create the variables from the start
#    self.fprop(tf.placeholder(tf.float32, [32] + input_shape))
    self.fprop(tf.placeholder(tf.float32, [128] + input_shape))#Revised by Ding========================
    # Put a reference to the params in self so that the params get pickled
    self.params = self.get_params()

  def fprop(self, x, **kwargs):
    del kwargs
    conv_args = dict(
        activation=tf.nn.relu,
        kernel_initializer=initializers.HeReLuNormalInitializer,
        kernel_size=3,
        padding='same')
    y = x

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
        y = tf.layers.conv2d(y, self.nb_filters, **conv_args)
        y = tf.layers.conv2d(y, self.nb_filters, **conv_args)
        y = tf.layers.max_pooling2d(y, 2, 2)
        
        y = tf.layers.conv2d(y, 2*self.nb_filters, **conv_args)
        y = tf.layers.conv2d(y, 2*self.nb_filters, **conv_args)
        y = tf.layers.max_pooling2d(y, 2, 2)
        
        
        y = tf.layers.dense(
          tf.layers.flatten(y), 256, activation=tf.nn.relu,
          kernel_initializer=initializers.HeReLuNormalInitializer)
        y = tf.layers.dense(
          y, 256, activation=tf.nn.relu,
          kernel_initializer=initializers.HeReLuNormalInitializer)
        
        logits = tf.layers.dense(
          y, self.nb_classes,
          kernel_initializer=initializers.HeReLuNormalInitializer)
        
        return {self.O_LOGITS: logits,
              self.O_PROBS: tf.nn.softmax(logits=logits)}
