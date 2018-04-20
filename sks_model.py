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
  SmartKids Model
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sks_active_kid as active_kid

from sks_logger import logger
from sks_debugger import debugger
from sks_ops import *

import sys
if sys.version_info[0] == 3:
  xrange = range


def init_variables(sess, saver=None, checkpoint_dir=None):
  """ Initialize variables of the graph
  will init from checkpoint if provided
  return the last iter number
  """
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  assert sess is not None, 'session should not be none, use tf.Session()'

  # logger.debug(ckpt)
  if ckpt is None or saver is None or ckpt.model_checkpoint_path is None:
    sess.run(tf.global_variables_initializer())
    return 0

  if ckpt.model_checkpoint_path:
    # Restores from checkpoint
    checkpoint_path = ckpt.model_checkpoint_path
    saver.restore(sess, checkpoint_path)
    # Assuming model_checkpoint_path looks something like:
    #   /my-path/model-1000,
    # extract last_iter from it.
    last_iter = checkpoint_path.split('/')[-1].split('-')[-1]
    logger.info("Init from checkpoint " + checkpoint_path)
    return int(last_iter)


def accuracy_rate(predictions, labels, num_classes):
  """Return the accuracy rate based on dense predictions and sparse labels.
    predictions: [bs, num_classes]
    labels     : [bs] (range:[1, num_classes])
    num_classes: int
    return accuracy (0~100%), list: err of every kids, list: sample count of every kids
  """
  import numpy as np
  if isinstance(predictions, list):
    predictions = np.array(predictions)
  if isinstance(labels, list):
    labels = np.array(labels)
  assert predictions.shape[0] == labels.shape[0]
  top1_index = np.argmax(predictions, 1)
  top1_index = top1_index + 1  # label start from index 1
  #logger.debug('predict: %s' % top1_index)
  #logger.debug('label  : %s' % labels)
  acc = 100.0 * np.sum(top1_index == labels) / predictions.shape[0]

  missed_kids = labels[top1_index != labels]
  kids_counts = []
  kids_err = []
  for kid in range(1, num_classes + 1):
    cnt = labels[labels == kid]
    failed_cnt = missed_kids[missed_kids == kid]
    kids_err.append('%.2f%%' % (100 * failed_cnt.size / cnt.size))
    kids_counts.append(cnt.size)
  logger.info('err cnt size: %s' % kids_counts)
  logger.info('err rate: %s' % kids_err)
  return acc, kids_err, kids_counts


def eval_in_batches(sess, batch_size, val_data, val_op, feed_name, num_classes,
                    use_fp16):
  """Get all predictions for a dataset by running it in small batches.
      val_data shape: [total_size, feat_dims]
  """
  import numpy as np
  total_size = val_data.shape[0]
  if total_size < batch_size:
    logger.warning(
        "data image sizes %d should larger than batch size %d. Force change batchsize to %d"
        % (total_size, batch_size, total_size))
    batch_size = total_size
  predictions = np.ndarray(
      shape=(total_size, num_classes),
      dtype=np.float32 if not use_fp16 else np.float16)
  for begin in xrange(0, total_size, batch_size):
    end = begin + batch_size
    if end <= total_size:
      predictions[begin:end, :] = sess.run(
          val_op,
          feed_dict={
              feed_name: val_data[begin:end, ...].reshape([batch_size, -1])
          })
    else:
      batch_predictions = sess.run(
          val_op,
          feed_dict={
              feed_name: val_data[-batch_size:, ...].reshape([batch_size, -1])
          })
      predictions[begin:, :] = batch_predictions[begin - total_size:, :]
  return predictions


def infer_one_kid_with_all_dataset_in_batches(sess, infer_op, dataset, subset,
                                              feed_name, batch_size, use_fp16):
  """Get all logits for a whole dataset by running it in small batches.
      dataset: a Dataset Class
      return logits, labels (both type: np.ndarray, size: total_size)
  """
  import numpy as np
  assert dataset.name() in ['imagenet', 'mnist']
  assert subset in ['val', 'test']
  total_size = dataset.val_samples(
  ) if subset == 'val' else dataset.test_samples()
  if total_size < batch_size:
    logger.warning(
        "batch size %d is larger than the dataset samples %d. Force change batchsize to %d"
        % (batch_size, total_size, total_size))
    batch_size = total_size
  if dataset.name() == 'imagenet':
    logits = np.ndarray(
        shape=(total_size, 1), dtype=np.float32 if not use_fp16 else np.float16)
    for begin in range(0, total_size, batch_size):
      # logger.info("infer load batch %d" % begin)
      batch_data = dataset.next_infer_batch(subset, batch_size).reshape(
          [batch_size, -1])
      batch_logit = sess.run(infer_op, feed_dict={feed_name: batch_data})
      end = begin + batch_size
      end = end if end < total_size else total_size
      logits[begin:end, :] = batch_logit[0:(
          end - begin), :]  # the end is not included
    return logits.reshape([total_size])
  else:
    # TODO: use the same way as imagenet, then remove the this if and else
    data = dataset.val_data() if subset == 'val' else dataset.test_data()
    return infer_one_kid_with_all_mnist_in_batches(
        sess, infer_op, data, batch_size, feed_name, use_fp16)


def infer_one_kid_with_all_mnist_in_batches(sess, infer_op, dataset, batch_size,
                                            feed_name, use_fp16):
  """Get all logits for a whole mnist dataset by running it in small batches.
      dataset: numpy.ndarray with shape: [total_size, feat_dims]
      return logits: [total_size]
  """
  import numpy as np
  total_size = dataset.shape[0]
  if total_size < batch_size:
    logger.warning(
        "data image sizes %d should larger than batch size %d. Force change batchsize to %d"
        % (total_size, batch_size, total_size))
    batch_size = total_size
  logits = np.ndarray(
      shape=(total_size, 1), dtype=np.float32 if not use_fp16 else np.float16)
  for begin in xrange(0, total_size, batch_size):
    end = begin + batch_size
    if end <= total_size:
      logits[begin:end, :] = sess.run(
          infer_op,
          feed_dict={
              feed_name: dataset[begin:end, ...].reshape([batch_size, -1])
          })
    else:
      batch_predictions = sess.run(
          infer_op,
          feed_dict={
              feed_name: dataset[-batch_size:, ...].reshape([batch_size, -1])
          })
      logits[begin:, :] = batch_predictions[begin - total_size:, :]
  return logits.reshape([total_size])


"""
def infer_one_kid_with_all_imagenet_in_batches(sess,
    infer_op, data_name, label_name, total_size, batch_size):
  '''Get all logits for a whole imagenet dataset by running it in small batches.
      dataset: numpy.ndarray with shape: [total_size, feat_dims]
      return logits, labels (both type: np.ndarray, size: total_size)
  '''
  import numpy as np
  logger.debug('total size %s, batch size %s' % (total_size, batch_size))
  label = []
  logit = []
  current_size = 0
  while 1:
    # label
    batch_label = sess.run(label_name)
    assert len(batch_label) == batch_size

    # data
    batch_data = sess.run(data_name)
    batch_logit = sess.run(infer_op, feed_dict={data_name: batch_data})
    assert len(batch_logit) == batch_size

    if current_size + batch_size < total_size:
      label[current_size : ] = batch_label
      logit[current_size : ] = batch_logit
      current_size += batch_size
    elif current_size + batch_size == total_size:
      label[current_size : ] = batch_label
      logit[current_size : ] = batch_logit
      break
    else:
      offset = current_size + batch_size - total_size
      label[current_size : ] = batch_label[: offset]
      logit[current_size : ] = batch_logit[: offset]
      break
    #logger.info('current total size %s' % current_size)

  assert len(label) == total_size and len(logit) == total_size
  return np.array(logit).reshape([total_size]), np.array(label).reshape([total_size])
"""


def convert_label_to_kid_label(labels, kid):
  '''
  labels: numpy.ndarray, shape[size] (range:[1, num_classes])
  kid: int
  num_classes: int
  return np.array with int type
  '''
  import numpy as np
  if isinstance(labels, list):
    labels = np.array(labels)
  assert isinstance(labels, np.ndarray)
  kid_label = [int(kid == lbl) for lbl in labels]
  kid_label = np.array(kid_label)
  #print(kid_label)
  assert kid_label.shape[0] == labels.shape[0]
  return kid_label


def cpu_loss_one_kid(logits, labels):
  '''
  inputs are numpy.ndarray type
    with same shape[size]

  https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
  x = logits, z = labels
  z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
  losses = max(x, 0) - x * z + log(1 + exp(-abs(x)))
  return avg(losses)
  '''
  import numpy as np
  assert isinstance(logits, np.ndarray) and isinstance(labels, np.ndarray)
  x = logits.astype(float)
  z = labels.astype(float)
  losses = x.clip(0.0) - np.multiply(x, z) + np.log(1 + np.exp(-np.abs(x)))
  return np.mean(losses)


def cpu_sigmoid(x):
  '''
  y = e^x / (e^x + 1) = 1 / (1 + e^-x)
  '''
  import numpy as np
  x = x.astype(float)
  return 1.0 / (np.add(1.0, np.exp(-x)))


def roc():
  '''
  get ROC of preds, and return the best threshold
  '''
  # TODO: implement me
  return 0


def func_best_threshold_cost(f1, tpr, tnr, pprec, nprec):
  # TODO: implement me
  # f1
  # or f1 + sqrt(tpr^2 + tnr^2) + sqrt(pprec^2 + nprec^2)
  return 0


def find_best_threshold(preds, labels, func_get_cost):
  # TODO: implement me
  # cal f1, tpr, tnr, pprec, nprec from preds and labels and a moving threshold
  # then get cost from these vars and func_get_cost
  # find a threshold to make the cost ma
  return 0.5


def cpu_acc_one_kid(predictions, labels, threshold=0.5):
  '''
  inputs are numpy.ndarray type with same shape[size].
  prediction type float, range: [0, 1]
  label type int, range: 0 or 1

  Positive Precision(also called Precision only):
    (measure the predicted capability, of how many right data in all pridected data.
    the true positive / all predicted as positve)
    = true_positive / (true_positive + false_positive)
  Negative precision:
    = true_negative / (true_negative + false_negative)
  True positive rate (Recall)
    (measure about the samples, of how many right pridected data in all should be right.
    the true positive / all should be positive)
    = true_positive / (true_positive + false_negative)
  False positive rate:
    = false_positive / (false_positive + true_negative)
  True negtive rate:
    = 1 - False positive rate
  F1 value: 2/F1 = 1/Positive_Precision + 1/Recall ==> F1 = 2TP / (2TP + FP + FN)
    F1 value is the best for overall judgement, the larger the better.
  ACC (all right choice / total size) = (true_positive + true_negative) / totalsize

  return F1 value, ACC, Positive Precision , Negative precision,
    True positive rate (Recall), True negtive rate
    all values should be larger the better.
  '''
  # TODO: optimaze me
  # sort then cal
  import numpy as np
  if isinstance(labels, list):
    labels = np.array(labels)
  assert isinstance(labels, np.ndarray)
  assert predictions.ndim == 1 and labels.ndim == 1, 'predictions.ndim %s, labels.ndim %s' % (
      predictions.ndim, labels.ndim)
  assert predictions.size == labels.size
  sz = labels.size
  lbl = labels.astype(int)
  pred = np.array([1 if p >= threshold else 0 for p in predictions]).astype(int)
  #print(lbl)
  #print(pred)
  for i in xrange(sz):
    assert lbl[i] == 0 or lbl[i] == 1
    assert pred[i] == 0 or pred[i] == 1
  should_be_true = pred[lbl == 1]
  should_be_false = pred[lbl != 1]
  should_be_true_cnt = should_be_true.size
  should_be_false_cnt = should_be_false.size
  true_positive = should_be_true[should_be_true == 1].size
  false_negative = should_be_true_cnt - true_positive
  true_negative = should_be_false[should_be_false == 0].size
  false_positive = should_be_false_cnt - true_negative
  logger.debug("tp %d, fp %d, tn %d, fn %d" % (true_positive, false_positive,
                                               true_negative, false_negative))
  positive_prec = float(true_positive) / float(
      true_positive + false_positive) if (
          true_positive + false_positive) > 0 else 0
  negative_prec = float(true_negative) / float(
      true_negative + false_negative) if (
          true_negative + false_negative) > 0 else 0
  # == recall
  tpr = float(true_positive) / float(true_positive + false_negative) if (
      true_positive + false_negative) > 0 else 0
  fpr = float(false_positive) / float(false_positive + true_negative) if (
      false_positive + true_negative) > 0 else 0
  tnr = 1.0 - fpr
  F1 = float(2 * true_positive) / float(
      2 * true_positive + false_positive + false_negative)
  acc = float(true_positive + true_negative) / float(sz)
  logger.debug(
      "F1 %f, ACC %f, positive-prec %f, negative-prec %f, tpr(recall) %f, tnr %f"
      % (F1, acc, positive_prec, negative_prec, tpr, tnr))
  return F1, acc, positive_prec, negative_prec, tpr, tnr


def test_acc():
  import numpy as np
  sz = 15
  lbl = np.arange(sz)
  for i in xrange(sz):
    lbl[i] = np.random.randint(0, 2)
  prd = np.random.rand(15)
  print(lbl)
  print(prd)
  cpu_acc_one_kid(prd, lbl, 0.5)


def cpu_softmax(x):
  '''
  return y = exp(x-max(x)) / sum(exp(x-max(x)))
  '''
  import numpy as np
  assert x.ndim == 1
  m = x.max()
  y = (x - m).clip(-64)
  y = np.exp(y)
  return y / np.sum(y)


def infer_one_kid(input_data,
                  input_channels,
                  img_height,
                  img_width,
                  is_training,
                  data_format='nchw',
                  use_gpu=True,
                  dtype=tf.float32):
  '''inference
  kid_index
  return logits
  '''
  assert data_format in ['nchw', 'nhwc'], "only support nchw or nhwc yet"
  assert active_kid.index >= 1
  logger.info("Inference Active Kid is %s" % active_kid.name)

  #assert data_format == 'nchw', "only support NCHW yet"
  if data_format == 'nchw':
    shape = [-1, input_channels, img_height, img_width]
  else:
    shape = [-1, img_height, img_width, input_channels]
  feat = tf.reshape(input_data, shape)

  concat_axis = 1 if data_format == 'nchw' else 3

  #def infer_mnist():

  #def infer_imagenet():

  with tf.variable_scope(active_kid.name):
    reset_all_default_names()
    logger.info(feat.get_shape().as_list())
    y = cbr_op(
        feat,
        input_channels,
        3,
        3,
        3,
        1,
        1,
        'VALID',
        is_training,
        data_format,
        dtype=dtype)
    logger.info(y.get_shape().as_list())

    pyramid_depth = 4
    # brach 1
    concat_list = []
    b1 = conv_op(y, 3, 1, 3, 3, 1, 1, 'SAME', data_format, dtype=dtype)
    logger.info(b1.get_shape().as_list())
    for repeat in range(pyramid_depth):
      b1 = pool_op(b1, 3, 3, 2, 2, 'MAX', data_format, padding='VALID')
      logger.info(b1.get_shape().as_list())
      # replace 3x3 ==> 1x3 && 3x1
      b1 = conv_op(b1, 1, 1, 3, 3, 1, 1, 'SAME', data_format, dtype=dtype)
      logger.info(b1.get_shape().as_list())
    concat_list.append(b1)

    pyramid = []
    pool = y
    for idx in xrange(pyramid_depth):
      pool = pool_op(pool, 3, 3, 2, 2, 'MAX', data_format, padding='VALID')
      logger.info(pool.get_shape().as_list())
      pyramid.append(pool)

    p_idx = 1
    for p in pyramid:
      p = conv_op(p, 3, 1, 3, 3, 1, 1, 'SAME', data_format, dtype=dtype)
      logger.info(p.get_shape().as_list())
      for repeat in range(pyramid_depth - p_idx):
        p = pool_op(p, 3, 3, 2, 2, 'MAX', data_format, padding='VALID')
        # replace 3x3 ==> 1x3 && 3x1
        p = conv_op(p, 1, 1, 3, 3, 1, 1, 'SAME', data_format, dtype=dtype)
        logger.info(p.get_shape().as_list())
      concat_list.append(p)
      p_idx += 1

    concat = concat_op(concat_list, concat_axis)

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    shape = concat.get_shape().as_list()  # tf.shape(pool)
    dim_in = shape[1] * shape[2] * shape[3]
    logger.info("last fc input shape: %s" % shape)
    logger.info("last fc input dims: %d" % dim_in)

    y = fc_op(concat, dim_in=dim_in, dim_out=1, dtype=dtype, bias_init=0.1)

    return y if is_training else tf.sigmoid(
        y, name=active_kid.name + '_sigmoid')


def loss_one_kid(logits, labels):
  '''get loss from logits and labels
  input params
    logits    : tf.Tensor
    labels    : tf.placeholder shape[bs]
    kid       : kid index number, started from 1

  return the tf losses operation
  '''
  assert active_kid.index >= 1
  logger.info("Loss Active Kid is %s" % active_kid.name)
  kid = active_kid.index
  # assert labels.get_shape().as_list()[0] == num_kids

  bs = labels.get_shape().as_list()[0]
  lbl = tf.constant([kid] * bs, tf.int64, name=active_kid.name + '_label')
  lbl = tf.cast(tf.equal(labels, lbl), logits.dtype)
  l = tf.expand_dims(lbl, 1, name=active_kid.name + '_expand_dim_label')
  sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      logits=logits, labels=l, name=active_kid.name + "_sigmoid_entropy")
  cross_entropy_mean = tf.reduce_mean(
      sigmoid_cross_entropy, name=active_kid.name + '_loss')
  debugger.add_scalar_summaries(
      name=active_kid.name + '_loss', x=cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  tf.add_to_collection(active_kid.loss_name, cross_entropy_mean)
  return tf.add_n(
      tf.get_collection(active_kid.loss_name),
      name=active_kid.name + 'total_loss')
