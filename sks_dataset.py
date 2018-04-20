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
  A dataset for SmartKids
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sks_logger import logger
import random
import os


class Dataset(object):
  """A simple class for handling data sets."""

  def __init__(self, name):
    self.name_ = name

  def name(self):
    """Return the name of this dataset"""
    return self.name_

  def num_classes(self):
    """Returns the number of classes in the data set."""
    raise NotImplementedError(
        'subclasses of Dataset must provide a num_classes() method')

  def num_channels(self):
    """Returns the number of images channels in the data set."""
    raise NotImplementedError(
        'subclasses of Dataset must provide a num_channels() method')

  def img_height(self):
    """Returns the height of image in the data set."""
    raise NotImplementedError(
        'subclasses of Dataset must provide a img_height() method')

  def img_width(self):
    """Returns the width of image in the data set."""
    raise NotImplementedError(
        'subclasses of Dataset must provide a img_width() method')

  def train_samples(self):
    """Returns the number of examples in the training data subset."""
    raise NotImplementedError(
        'subclasses of Dataset must provide a train_samples() method')

  def val_samples(self):
    """Returns the number of examples in the validation data subset."""
    raise NotImplementedError(
        'subclasses of Dataset must provide a val_samples() method')

  def test_samples(self):
    """Returns the number of examples in the test data subset."""
    raise NotImplementedError(
        'subclasses of Dataset must provide a test_samples() method')

  def val_data(self):
    '''Return validation data '''
    raise NotImplementedError(
        'subclasses of Dataset must provide a val_data() method')

  def test_data(self):
    '''Return test data '''
    raise NotImplementedError(
        'subclasses of Dataset must provide a test_data() method')

  def infer_labels(self, subset):
    '''return the labels of validation or test dataset'''
    raise NotImplementedError(
        'subclasses of Dataset must provide a infer_labels(subset) method')


def get_images_data(image_list, height, width, data_format='nhwc'):
  '''
  get all images as (height, width) with format(nchw or nhwc)
  return numpy ndarray with shape(num_images, 3*height*width)
  '''
  import cv2
  import numpy as np
  assert isinstance(image_list, list)
  assert height * width != 0
  data_format = data_format.upper()
  PIXEL_DEPTH = 255
  num_images = len(image_list)
  out = np.full([num_images, height, width, 3], 0, dtype=float)
  for i in range(num_images):
    # the result is numpy
    img = cv2.imread(
        image_list[i]
        .strip())  # cv will always read image as RGB, even the src is gray
    # Check that image is RGB and 3 channels
    assert isinstance(
        img, np.ndarray), "load %d image failed: %s" % (i, image_list[i])
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    # uint8(0 ~ 255) to float(-0.5 ~ 0.5)
    img = img.astype(float)
    img = img / PIXEL_DEPTH  # 0.0~1.0 (img - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    # resize image
    out[i] = cv2.resize(img,
                        (height, width))  #, interpolation = cv2.INTER_CUBIC)
  # transpose
  if data_format == 'NCHW':
    np.transpose(out, (0, 3, 1, 2))
    out = out.reshape(num_images, 3, height, width)

  out = out.reshape([num_images, -1])
  return out


def get_one_image(imagepath, height, width, transpose_channel):
  '''
  get one image as (height, width) with format chw or hwc
  default is hwc format, transpose_channel means chw
  return img as np.array(astype(float)), every pixel (-0.5~0.5)
  '''
  import cv2
  import numpy as np
  # the result is numpy
  img = cv2.imread(
      imagepath)  # cv will always read image as RGB, even the src is gray
  # Check that image is RGB and 3 channels
  assert len(img.shape) == 3 and img.shape[2] == 3

  # uint8(0 ~ 255) to float(-0.5 ~ 0.5)
  img = img.astype(float)
  PIXEL_DEPTH = 255
  img = (img - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH

  # resize image
  img = cv2.resize(img, (height, width))  #, interpolation = cv2.INTER_CUBIC)

  # transpose
  return np.transpose(img, (2, 0, 1)) if transpose_channel else img


def get_images_data_parallel(image_list, height, width, data_format='nhwc'):
  '''
  get all images as (height, width) with format(nchw or nhwc)
  return numpy ndarray with shape(num_images, 3*height*width) every pixel (-0.5~0.5)
  '''
  import multiprocessing
  import numpy as np
  assert isinstance(image_list, list)
  assert height * width != 0
  data_format = data_format.upper()
  assert data_format in ['NCHW', 'NHWC']

  multiprocessing.freeze_support(
  )  # must add this on Windows platform ,in case of RuntimeError
  cpus = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(cpus if cpus <= 4 else 4)  # max parallel number
  results = []
  for imagepath in image_list:
    result = pool.apply_async(
        get_one_image, args=(imagepath, height, width, data_format == 'NCHW'))
    results.append(result)
  pool.close()
  pool.join()

  img_dims = height * width * 3
  out = np.full([len(results), img_dims], 0, dtype=float)
  i = 0
  for r in results:
    out[i] = r.get().reshape(img_dims)
    i += 1
  return out


class ImagenetDataset(Dataset):
  """ImageNet dataset, includes train, val and test dataset."""

  def __init__(self, data_dir, data_format):
    assert data_format in ['nhwc', 'nchw']
    assert isinstance(data_dir, str)
    self.data_dir = data_dir + '/imagenet'
    self.data_format = data_format
    super(ImagenetDataset, self).__init__('imagenet')
    train_txt = os.path.join(self.data_dir, 'train.txt')
    val_txt = os.path.join(self.data_dir, 'val.txt')
    test_txt = os.path.join(self.data_dir, 'test.txt')
    assert os.path.isfile(train_txt), 'should have train list txt'
    assert os.path.isfile(val_txt), 'should have val list txt'
    assert os.path.isfile(test_txt), 'should have test list txt'

    def get_list_from_txt(txtfile):
      '''get data list from txt
        input format: label_id, filepath
        return label_list, filepath_lis
      '''
      labels = []
      files = []
      with open(txtfile, 'r') as f:
        for line in f:
          label, filepath = line.strip('\n').split(',')
          labels.append(int(label))
          files.append(filepath)
      return labels, files

    self.train_data_list = dict()
    # train data dict:
    # {1: ['', '', ...],
    #  3: ['', '', ...], ... }
    # val and test dict
    # {'labels' : [1, 3, 54, ...]
    #  'files'  : ['', '', '', ...]}
    self.val_data_list = dict()
    self.test_data_list = dict()
    # get val and test lists
    self.val_data_list['labels'], self.val_data_list[
        'files'] = get_list_from_txt(val_txt)
    assert len(self.val_data_list['labels']) == len(self.val_data_list['files'])
    self.val_num_samples_ = len(self.val_data_list['labels'])
    self.test_data_list['labels'], self.test_data_list[
        'files'] = get_list_from_txt(test_txt)
    assert len(self.test_data_list['labels']) == len(
        self.test_data_list['files'])
    self.test_num_samples_ = len(self.test_data_list['labels'])

    # get train lists
    train_labels, train_files = get_list_from_txt(train_txt)
    assert len(train_labels) == len(train_files)
    self.train_num_samples_ = len(train_files)
    for i in range(self.train_num_samples_):
      label = train_labels[i]
      filepath = train_files[i]
      if self.train_data_list.get(label) is None:
        self.train_data_list[label] = []
      self.train_data_list[label].append(train_files[i])

    assert len(self.train_data_list) == 1000
    self.active_kid = 0
    self.offset_pos = 0
    self.offset_neg = 0

    # infer
    self.infer_val_offset_pos = 0
    self.infer_test_offset_pos = 0

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 1000

  def num_channels(self):
    """Returns the number of images channels in the data set."""
    return 3

  def img_height(self):
    """Returns the height of image in the data set."""
    # the first gen use 224, then use smaller later with small model,
    # last use larger and we can pool it and still have small weights
    return 128

  def img_width(self):
    """Returns the width of image in the data set."""
    return 128

  def train_samples(self):
    """Returns the number of examples in the data set."""
    # Bounding box data consists of 615299 bounding boxes for 544546 images.
    return self.train_num_samples_

  def next_batch(self, batch_size, kid=0):
    '''get next batch with batch size when training
    return batch_data(shape:[bs, dim]), batch_labels(shape:[bs])
    '''
    assert kid >= 1 and kid <= self.num_classes()
    #################### check if kid index changed
    if self.active_kid != kid:
      self.active_kid = kid
      self.offset_pos = 0
      self.offset_neg = 0
      self.kid_neg_list = []
      self.kid_neg_label = []
      for lbl in self.train_data_list:
        if lbl != kid:
          self.kid_neg_list += self.train_data_list[lbl]
          self.kid_neg_label += [lbl] * len(self.train_data_list[lbl])
      # shuffle them
      idx = list(range(len(self.kid_neg_list)))
      random.shuffle(idx)
      self.kid_neg_list = [self.kid_neg_list[i] for i in idx]
      self.kid_neg_label = [self.kid_neg_label[i] for i in idx]

    #####################  randome positive numbers, but always keep have postitve samples
    # TODO: the ratio should be changing with the learning rate
    positive_ratio = random.uniform(0.2, 0.8)  # 0.2~0.8 ?
    positive_num = int(positive_ratio * batch_size)
    kid_num_samples = len(self.train_data_list[kid])
    if kid_num_samples < positive_num:
      positive_num = kid_num_samples
      batch_size = int(positive_num / positive_ratio)
    #####################  positive samples
    positive_list = []
    new_offset = self.offset_pos + positive_num
    if new_offset < kid_num_samples:
      positive_list += self.train_data_list[kid][self.offset_pos:new_offset]
      self.offset_pos = new_offset
    else:
      #print("new_offset:%d, kid_num_samples:%d" %(new_offset, kid_num_samples))
      positive_list += self.train_data_list[kid][self.offset_pos:]
      # shuffle
      random.shuffle(self.train_data_list[kid])
      self.offset_pos = new_offset - kid_num_samples
      positive_list += self.train_data_list[kid][0:self.offset_pos]
    assert len(positive_list
               ) == positive_num, 'len(positive_list)=%d, positive_num=%d' % (
                   len(positive_list), positive_num)

    #####################  negative samples
    neg_list = []
    neg_lbls = []
    new_offset = self.offset_neg + batch_size - positive_num
    if new_offset < len(self.kid_neg_list):
      neg_list += self.kid_neg_list[self.offset_neg:new_offset]
      neg_lbls += self.kid_neg_label[self.offset_neg:new_offset]
      self.offset_neg = new_offset
    else:
      neg_list += self.kid_neg_list[self.offset_neg:]
      neg_lbls += self.kid_neg_label[self.offset_neg:]
      # shuffle
      idx = list(range(len(self.kid_neg_list)))
      random.shuffle(idx)
      self.kid_neg_list = [self.kid_neg_list[i] for i in idx]
      self.kid_neg_label = [self.kid_neg_label[i] for i in idx]
      self.offset_neg = new_offset - len(self.kid_neg_list)
      neg_list += self.kid_neg_list[0:self.offset_neg]
      neg_lbls += self.kid_neg_label[0:self.offset_neg]
    ##################### all samples
    next_list = positive_list
    next_list += neg_list
    next_lbls = [kid] * positive_num
    next_lbls += neg_lbls
    assert len(next_list) == batch_size and len(next_lbls) == batch_size
    shufl = list(range(batch_size))
    random.shuffle(shufl)
    next_list = [next_list[i] for i in shufl]
    next_lbls = [next_lbls[i] for i in shufl]
    #logger.info("label index: %s" % next_lbls)
    #logger.info("positive numbers: %s" % positive_num)
    next_data = get_images_data(next_list, self.img_height(), self.img_width(),
                                self.data_format)
    return next_data, next_lbls

  def val_samples(self):
    """Returns the number of examples in the validation data subset."""
    return self.val_num_samples_

  def test_samples(self):
    """Returns the number of examples in the test data subset."""
    return self.test_num_samples_

  def next_infer_batch(self, subset, batch_size):
    '''Get next batch when inference, only support val and test dataset
    '''
    assert subset in ['val', 'test']
    imagelist = self.val_data_list[
        'files'] if subset == 'val' else self.test_data_list['files']
    total_size = self.val_samples() if subset == 'val' else self.test_samples()
    begin = self.infer_val_offset_pos if subset == 'val' else self.infer_test_offset_pos

    end = begin + batch_size
    end = end if end < total_size else total_size
    batchlist = imagelist[begin:end]  # the end index is not included

    # when equal do not need +=
    if begin + batch_size > total_size:
      end = begin + batch_size - total_size
      batchlist += imagelist[0:end]

    if begin + batch_size >= total_size:
      if subset == 'val':
        self.infer_val_offset_pos = 0
      else:
        self.infer_test_offset_pos = 0
    assert len(batchlist) == batch_size
    return get_images_data(batchlist, self.img_height(), self.img_width(),
                           self.data_format)

  def infer_labels(self, subset):
    '''return the labels of val or test dataset'''
    assert subset in ['val', 'test']
    return self.val_data_list[
        'labels'] if subset == 'val' else self.test_data_list['labels']

  # TODO: remove belows: data_files and reader:
  def data_files(self, subset):
    """Returns a python list of all (sharded) data subset files.

    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    import os
    if subset == 'train':
      tf_record_pattern = os.path.join(self.data_dir, '%s' % subset,
                                       '%s-*' % subset)
    elif subset == 'val':
      tf_record_pattern = os.path.join(self.data_dir, 'validation',
                                       'validation-*')
    elif subset == 'test':
      logger.error("not implemented")
    else:
      logger.error("unknow subset %s" % subset)
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
      logger.error('No files found for dataset %s/%s at %s' %
                   (self.name(), subset, self.data_dir))
    return data_files

  def reader(self):
    """Return a reader for a single entry from the data set.

    See io_ops.py for details of Reader class.

    Returns:
      Reader object that reads the data set.
    """
    return tf.TFRecordReader()


class MnistDataset(Dataset):
  """Mnist dataset, includes train, val and test.
     All labels start from 1, skipping 0.
     So lable 1 means number 0
     lable 10 means number 9"""

  def __init__(self, data_dir, data_format, use_dummy=False, seed=None):
    '''init, seed is for dummy'''
    if use_dummy:
      raise NotImplementedError('use_dummy not implemented')

    assert data_format in ['nhwc', 'nchw']
    assert isinstance(data_dir, str)
    self.data_dir = data_dir + '/mnist'
    self.data_format = data_format
    super(MnistDataset, self).__init__('mnist')

    self.prepare(self.data_dir)
    self.offset = 0

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 10

  def num_channels(self):
    """Returns the number of images channels in the data set."""
    return 1

  def img_height(self):
    """Returns the height of image in the data set."""
    return 28

  def img_width(self):
    """Returns the width of image in the data set."""
    return 28

  def train_samples(self):
    """Returns the number of examples in the training data subset."""
    return int(5.5e4)

  def val_samples(self):
    """Returns the number of examples in the validation data subset."""
    return int(5e3)

  def test_samples(self):
    """Returns the number of examples in the test data subset."""
    return int(1e4)

  def prepare(self, data_dir):
    '''prepare the dataset from data_dir
    prepare self train_data, train_labels, val_data, val_labels, test_data, test_labels
    '''
    # Get the data.
    train_data_filename = self.maybe_download(data_dir,
                                              'train-images-idx3-ubyte.gz')
    train_labels_filename = self.maybe_download(data_dir,
                                                'train-labels-idx1-ubyte.gz')
    test_data_filename = self.maybe_download(data_dir,
                                             't10k-images-idx3-ubyte.gz')
    test_labels_filename = self.maybe_download(data_dir,
                                               't10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    total_images = self.train_samples() + self.val_samples()
    train_data = self.extract_data(train_data_filename, total_images)
    train_label = self.extract_labels(train_labels_filename, total_images)
    self.test_data_ = self.extract_data(test_data_filename, self.test_samples())
    self.test_label_ = self.extract_labels(test_labels_filename,
                                           self.test_samples())

    # Generate a validation set.
    # the first 5000 is used to as validation
    self.val_data_ = train_data[:self.val_samples(), ...]
    self.val_label_ = train_label[:self.val_samples()]
    self.train_data_ = train_data[self.val_samples():, ...]
    self.train_label_ = train_label[self.val_samples():]
    assert self.train_samples() == self.train_data_.shape[0]
    assert self.train_samples() == self.train_label_.shape[0]
    assert self.val_samples() == self.val_data_.shape[0]
    assert self.val_samples() == self.val_label_.shape[0]

  def maybe_download(self, data_dir, filename):
    """Download the data from Yann's website, unless it's already here."""
    import os
    from six.moves import urllib
    data_url = 'http://yann.lecun.com/exdb/mnist/'
    if not tf.gfile.Exists(data_dir):
      tf.gfile.MakeDirs(data_dir)
    filepath = os.path.join(data_dir, filename)
    if not tf.gfile.Exists(filepath):
      filepath, _ = urllib.request.urlretrieve(data_url + filename, filepath)
      with tf.gfile.GFile(filepath) as f:
        size = f.size()
      logger.info('Successfully downloaded %s %d' % (filename, size) + 'bytes.')
    return filepath

  def extract_data(self, filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    import gzip
    logger.debug('Extracting %s' % filename)
    PIXEL_DEPTH = 255
    with gzip.open(filename) as bytestream:
      bytestream.read(16)  # skip header
      buf = bytestream.read(num_images * self.num_channels() *
                            self.img_height() * self.img_width())
      data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
      data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
      data = data.reshape(num_images, self.img_height(), self.img_width(),
                          self.num_channels())
      if self.data_format == 'nchw':
        np.transpose(data, (0, 3, 1, 2))
        data = data.reshape(num_images, self.num_channels(), self.img_height(),
                            self.img_width())
      return data

  def extract_labels(self, filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    import gzip
    logger.info('Extracting %s' % filename)
    with gzip.open(filename) as bytestream:
      bytestream.read(8)  # skip head
      buf = bytestream.read(1 * num_images)
      labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
      # all labels start from 1 now, skipping 0!
      labels = np.add(labels, 1)
      # print (labels)
    return labels

  def next_batch(self, batch_size, kid=0):
    '''get next batch with batch size
    return batch_data(shape:[bs, dim]), batch_labels(shape:[bs])
    '''
    if self.offset + batch_size >= self.train_samples():
      # no more datas
      # logger.info("one epoch done, restart dataset")
      # TODO shuffle dataset
      self.offset = np.random.randint(0, self.train_samples() - batch_size - 1)

    end = self.offset + batch_size
    batch_data = self.train_data_[self.offset:end, ...]
    batch_labels = self.train_label_[self.offset:end]
    self.offset += batch_size

    batch_data = batch_data.reshape([batch_size, -1])
    return batch_data, batch_labels

  def val_data(self):
    '''Return validation data '''
    return self.val_data_

  def test_data(self):
    '''Return test data '''
    return self.test_data_

  def infer_labels(self, subset):
    '''return the labels of val or test dataset'''
    assert subset in ['val', 'test']
    return self.val_label_ if subset == 'val' else self.test_label_

  # TODO: remove below use infer_labels instead.

  def test_label(self):
    '''Return test label '''
    return self.test_label_
