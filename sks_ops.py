# Copyright 2017 The Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
  SmartKids 2 Operations
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sks_active_kid as active_kid

from sks_debugger import debugger
from sks_logger import logger

import sys
if sys.version_info[0] == 3:
  xrange = range

SEED = 66478  # Set to None for random seed.

__all__ = [
    'reset_all_default_names',
    'conv_op',
    'relu_op',
    'pool_op',
    'fc_op',
    'bn_op',
    'cbr_op',
    'concat_op',
]


def weight_variable(name, shape, initializer=None, dtype=tf.float32, wd=None):
  """weight_variable generates a weight variable of a given shape. """
  if initializer is None:
    initializer = tf.truncated_normal_initializer(
        mean=0.0, stddev=0.1, seed=SEED, dtype=dtype)
  var = tf.get_variable(name, shape, dtype=dtype, initializer=initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection(active_kid.loss_name, weight_decay)
  return var


def bias_variable(name,
                  shape,
                  value=0,
                  dtype=tf.float32,
                  trainable=True,
                  collections=None):
  """bias_variable generates a bias variable of a given shape."""
  if isinstance(shape, list):
    size = 1
    for i in xrange(len(shape)):
      size = size * shape[i]
  elif isinstance(shape, int):
    size = shape
  else:
    raise ValueError('Shape do not support')
  logger.debug('bias size %d' % size)
  if isinstance(value, list):
    assert len(value) == size
  else:
    NumberTypes = (int, float)
    assert isinstance(value, NumberTypes), "should be a number"
    value = [value] * size

  # initializer = tf.zeros_initializer(dtype=dtype)
  initializer = tf.constant_initializer(value=value, dtype=dtype)
  return tf.get_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      trainable=trainable,
      collections=collections)


class DefaultNameFactory(object):
  def __init__(self, name_prefix):
    self.__counter__ = 0
    self.__name_prefix__ = name_prefix

  def clear_count(self):
    self.__counter__ = 0

  def __call__(self, func):
    if self.__name_prefix__ is None:
      self.__name_prefix__ = func.__name__
    name = "_%s_%d_" % (self.__name_prefix__, self.__counter__)
    self.__counter__ += 1
    return name


def default_name(name_prefix=None):
  """
  Decorator to set "name" arguments default to "{name_prefix}_{invoke_count}".

  ..  code:: python

    @default_name("some_name")
    def func(name=None):
      print name    # name will never be None.
                    # If name is not set, name will be "some_name_%d"

  :param name_prefix: name prefix.
  :type name_prefix: basestring
  :return: a decorator to set default name
  :rtype: callable
  """
  import functools
  import inspect
  assert name_prefix is None or isinstance(name_prefix, str)
  name_factory = DefaultNameFactory(name_prefix)

  def __impl__(func):
    @functools.wraps(func)
    def __wrapper__(*args, **kwargs):
      def check_args():
        if len(args) != 0:
          argspec = inspect.getargspec(func)
        #print(argspec)
        num_positional = len(argspec.args)
        if argspec.defaults:
          num_positional -= len(argspec.defaults)
        if not argspec.varargs and len(args) > num_positional:
          logger.warning("Should use keyword arguments for non-positional args")

      reset_key = 'reset_default_name'
      if reset_key in kwargs and kwargs[reset_key] is True:
        name_factory.clear_count()
        return
      else:
        key = 'name'
        check_args()
        if key not in kwargs or kwargs[key] is None:
          kwargs[key] = name_factory(func)
        return func(*args, **kwargs)

    return __wrapper__

  return __impl__


def reset_all_default_names():
  '''reset the all op's default name from count 0
  '''
  conv_op(reset_default_name=True)
  relu_op(reset_default_name=True)
  pool_op(reset_default_name=True)
  fc_op(reset_default_name=True)
  bn_op(reset_default_name=True)
  cbr_op(reset_default_name=True)
  concat_op(reset_default_name=True)


#######################
# Operations          #
#######################
@default_name('conv')
def conv_op(x,
            ic,
            oc,
            kh,
            kw,
            sh,
            sw,
            padding,
            data_format,
            dtype=tf.float32,
            with_bias=True,
            name=None):
  """conv_layer returns a 2d convolution layer.

  param:
    input_channels, num_kernels and kernel_size are required
    size of kernel and stride are list as: [height, width]
    data_format: "nhwc" or "nchw"
    padding refer to tf.nn.conv2d padding.
  """
  assert isinstance(name, str)
  assert padding in ['SAME', 'VALID']
  data_format = data_format.upper()
  assert data_format in ['NHWC', 'NCHW']

  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].
  if data_format == 'NCHW':
    stride = [1, 1, sh, sw]
  else:
    stride = [1, sh, sw, 1]

  logger.debug(name)
  with tf.variable_scope(name):
    wgt = weight_variable('weight', [kh, kw, ic, oc], dtype=dtype)
    debugger.add_var_summaries(wgt)
    y = tf.nn.conv2d(
        x, wgt, stride, padding, data_format=data_format, name=name)
    debugger.add_histogram_summaries('post_' + name + '/pre_bias', y)
    if with_bias:
      bias = bias_variable('bias', [oc], value=0, dtype=dtype)
      debugger.add_var_summaries(bias)
      y = tf.nn.bias_add(y, bias, data_format, 'add_bias')
      debugger.add_histogram_summaries('post_bias' + name, y)
    logger.debug(y)
    return y


@default_name('relu')
def relu_op(x, capping=None, name=None):
  """ReLU layer, clipped if set capping
  y = min(max(0, x), capping)
  """
  assert isinstance(name, str)
  # Bias and rectified linear non-linearity.
  y = tf.nn.relu(x, name)
  if capping is not None:
    y = tf.minimum(y, capping)
  debugger.add_histogram_summaries('post_' + name, y)
  logger.debug(y)
  return y


@default_name('pool')
def pool_op(
    x,
    kh,
    kw,
    sh,
    sw,
    pooling_type,  # 'AVG' or 'MAX'
    data_format,
    padding='VALID',  #'SAME'
    dilation_rate=None,
    name=None):
  assert isinstance(name, str)
  data_format = data_format.upper()
  assert data_format in ['NCHW', 'NHWC']
  logger.debug(x.get_shape())
  logger.debug(name)
  strides = [sh, sw]  # no matter nchw or nhwc
  window_shape = [kh, kw]
  y = tf.nn.pool(
      x,
      window_shape=window_shape,
      pooling_type=pooling_type,
      padding=padding,
      dilation_rate=dilation_rate,
      strides=strides,
      name=name,
      data_format=data_format)
  debugger.add_histogram_summaries('post_' + name, y)
  logger.debug(y)
  return y


@default_name('fc')
def fc_op(x,
          dim_in,
          dim_out,
          with_bias=True,
          dtype=tf.float32,
          bias_init=0.1,
          name=None):
  '''fc_layer returns a full connected layer
     Wx + b
     param:
      if dim_in is None, will set as dim_out
  '''
  assert isinstance(name, str)
  logger.debug(name)
  with tf.variable_scope(name):
    x = tf.reshape(x, [-1, dim_in])
    wgt = weight_variable('weight', [dim_in, dim_out], dtype=dtype, wd=0.004)
    debugger.add_var_summaries(wgt)
    y = tf.matmul(x, wgt)
    debugger.add_histogram_summaries('post_' + name + '/pre_bias', y)
    if with_bias:
      bias = bias_variable('bias', [dim_out], bias_init, dtype)
      debugger.add_var_summaries(wgt)
      y = tf.add(y, bias, 'add_bias')
      debugger.add_histogram_summaries('post_' + name, y)
    logger.debug(y)
    return y


@default_name('bn')
def bn_op(x,
          is_training,
          data_format,
          dtype=tf.float32,
          use_global_stat=None,
          eps=1e-5,
          name=None):
  """batch normalization layer
  return bn_op
  """
  assert isinstance(is_training, bool)
  assert isinstance(name, str)
  data_format = data_format.upper()
  assert data_format in ['NCHW', 'NHWC']

  if use_global_stat is None:
    use_global_stat = not is_training
  input_shape = x.get_shape().as_list()
  if data_format == 'NCHW':
    shape = input_shape[1]
  else:
    shape = input_shape[-1]
  with tf.variable_scope(name) as scope:
    #with tf.device('/cpu'):  # why? must be cpu device?
    shift = bias_variable('shift', shape, 0, dtype)
    scale = bias_variable('scale', shape, 1, dtype)
    # logger.info(scope.name)
    if is_training:
      y, mean, var = tf.nn.fused_batch_norm(
          x,
          scale,
          shift,
          mean=None,
          variance=None,
          epsilon=eps,
          data_format=data_format,
          is_training=is_training,
          name=name)
      # logger.info(mean.name)
      # logger.info(var.name)
      # mean should always before var
      active_kid.addto_bn_moving_average_list(mean)
      active_kid.addto_bn_moving_average_list(var)
    else:
      if use_global_stat:
        moving_mean = active_kid.get_moving_average(scope.name, 'mean')
        moving_var = active_kid.get_moving_average(scope.name, 'var')
        y, _, _ = tf.nn.fused_batch_norm(
            x,
            scale,
            shift,
            mean=moving_mean,
            variance=moving_var,
            epsilon=eps,
            data_format=data_format,
            is_training=is_training,
            name=name)
      else:
        y, _, _ = tf.nn.fused_batch_norm(
            x,
            scale,
            shift,
            mean=None,
            variance=None,
            epsilon=eps,
            data_format=data_format,
            is_training=True,
            name=name)

    y.set_shape(input_shape)
    logger.debug(y)
    return y


@default_name('cbr')
def cbr_op(x,
           ic,
           oc,
           kh,
           kw,
           sh,
           sw,
           padding,
           is_training,
           data_format,
           dtype=tf.float32,
           with_bias=True,
           eps=1e-5,
           name=None):
  '''
    conv + bn + relu
  '''
  with tf.variable_scope(name):
    y = conv_op(
        x,
        ic,
        oc,
        kh,
        kw,
        sh,
        sw,
        padding,
        data_format,
        dtype=dtype,
        with_bias=with_bias,
        name='conv')
    y = bn_op(y, is_training, data_format, dtype=dtype, eps=eps, name='bn')
    y = relu_op(y, name='relu')
    return y


@default_name('concat')
def concat_op(values, axis, name=None):
  '''
  https://www.tensorflow.org/api_docs/python/tf/concat
  values: list of tensor
  axis: means which axis in shape add.
    example:
      a.shape(1, 2, 3), b.shape(1, 2, 3)
      c = concat([a, b], 0) ==> c.shape(2, 2, 3)
      c = concat([a, b], 1) ==> c.shape(1, 4, 3)
      c = concat([a, b], 2) ==> c.shape(1, 2, 6)
  '''
  y = tf.concat(values, axis, name=name)
  logger.debug(y)
  return y
