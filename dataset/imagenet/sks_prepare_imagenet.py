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
  SmartKids Prepare ImageNet dataset
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

# This file is the output of process_bounding_box.py
# Assumes each line of the file looks like:
#
#   n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940
#
# where each line corresponds to one bounding box annotation associated
# with an image. Each line can be parsed as:
#
#   <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>
#
# Note that there might exist mulitple bounding box annotations associated
# with an image file.
tf.app.flags.DEFINE_string('bounding_box_file',
                           './imagenet_2012_bounding_boxes.csv',
                           'Bounding box file')

tf.app.flags.DEFINE_bool('use_bounding_box', False,
                         'Whether to use bounding box data')

tf.app.flags.DEFINE_string('output_directory', './', 'Output data directory')

tf.app.flags.DEFINE_string('val_label_file',
                           'imagenet_2012_validation_synset_labels.txt',
                           'Validation Labels file')

FLAGS = tf.app.flags.FLAGS


def get_raw_val_data(data_dir, label_file):
  """
  Return:
    val_dataset:
      ['synset'] = ['n0000123', 'n0004310', ...]
      ['filepath']  = ['xxx', 'xxx', ...]
  """
  synsets = [l.strip() for l in tf.gfile.FastGFile(label_file, 'r').readlines()]

  filepath = []
  for i in range(len(synsets)):
    basename = 'ILSVRC2012_val_000%.5d.JPEG' % (i + 1)
    f = os.path.join(data_dir, basename)
    if not os.path.exists(f):
      print('Failed to find: %s' % f)
      sys.exit(-1)
    filepath.append(f)

  val_dataset = dict()
  val_dataset['synset'] = synsets
  val_dataset['filepath'] = filepath
  print('Total have %d raw val samples' % len(filepath))
  return val_dataset


def get_val_label(raw_train, raw_val):
  """
  return the val label index list from the unique synset in raw_val,
  the unique synset can correspond to a unique label from raw_train dataset

  return a int list
  [1, 40, 20, ...] labels started from 1, len: lenght of val data samples
  """
  labels = []
  # cal how many val samples of each kid
  cnts = [0] * len(raw_train)
  for synset in raw_val['synset']:
    l = raw_train[synset]['label']
    labels.append(l)
    cnts[l - 1] += 1
  # for i in range(len(cnts)):
  #  print("Kid %d has %d sample in val dataset" % (i + 1, cnts[i]))
  return labels


def get_raw_train_data(data_dir):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.

      Assumes that the ImageNet data set resides in JPEG files located in
      the following directory structure.

        data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
        data_dir/n01440764/ILSVRC2012_val_00000543.JPEG

      where 'n01440764' is the unique synset label associated with these images.
      and it's a dir.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        n01440764
        n01443537
        n01484850
      where each line corresponds to a label expressed as a synset. We map
      each synset contained in the file to an integer (based on the alphabetical
      ordering) starting with the integer 1 corresponding to the synset
      contained in the first line.

      The reason we start the integer labels at 1 is to reserve label 0 as an
      unused background class.

  Returns:
    train_dataset =
      ['n0000001'] =
          ['label'] = 1
          ['filepath']  = ['xxx', 'xxx', ... 'xxx']
      ['n0000002'] =
          ['label'] = 2
          ['filepath']  = ['xxx', 'xxx', ... 'xxx']
  """

  dirs = [
      f for f in os.listdir(data_dir)
      if os.path.isdir(os.path.join(data_dir, f))
  ]
  train_dataset = dict()
  # Leave label index 0 empty as a background class.
  # Label index start from 1
  label_index = 1

  total_samples = 0
  # synset such as 'n01440764' is the unquie label
  for synset in dirs:
    train_dataset[synset] = dict()
    # Construct the list of JPEG files and labels.
    jpeg_file_path = '%s/%s/*.JPEG' % (data_dir, synset)
    train_dataset[synset]['label'] = label_index
    train_dataset[synset]['filepath'] = tf.gfile.Glob(jpeg_file_path)
    num_samples = len(train_dataset[synset]['filepath'])
    total_samples += num_samples
    print('Kid %d have %d raw samples.' % (train_dataset[synset]['label'],
                                           num_samples))
    label_index += 1
  print('Total have %d classes and %d raw samples' % (len(train_dataset),
                                                      total_samples))
  return train_dataset


def build_bounding_box_dict(bounding_box_file):
  """Build a lookup from image file to bounding boxes.

  Args:
    bounding_box_file: string, path to file with bounding boxes annotations.

      Assumes each line of the file looks like:

        n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940

      where each line corresponds to one bounding box annotation associated
      with an image. Each line can be parsed as:

        <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>

      Note that there might exist mulitple bounding box annotations associated
      with an image file. This file is the output of process_bounding_boxes.py.

  Returns:
    Dictionary mapping image file names to a list of bounding boxes. This list
    contains 0+ bounding boxes.
    dict =
      ['n00007846_64193.JPEG'] = [[xmin, ymin, xmax, ymax], [...], ...]
      [xxx] = [...]
  """
  lines = tf.gfile.FastGFile(bounding_box_file, 'r').readlines()
  images_to_bboxes = {}
  num_bbox = 0
  num_image = 0
  for l in lines:
    if l:
      parts = l.split(',')
      assert len(parts) == 5, ('Failed to parse: %s' % l)
      filename = parts[0]
      xmin = float(parts[1])
      ymin = float(parts[2])
      xmax = float(parts[3])
      ymax = float(parts[4])
      box = [xmin, ymin, xmax, ymax]

      if filename not in images_to_bboxes:
        images_to_bboxes[filename] = []
        num_image += 1
      images_to_bboxes[filename].append(box)
      num_bbox += 1

  print('Successfully read %d bounding boxes '
        'across %d images.' % (num_bbox, num_image))
  return images_to_bboxes


def boxing_raw_data(raw_data, box_dict, output_dir=FLAGS.output_directory):
  '''
  apply box on raw_train data.
  and save the boxed image to output_dir/boxed_images/xxx.jpeg
  input:
    raw_data =
      ['n0000001'] =
          ['label'] = 1
          ['filepath']  = ['xxx', 'xxx', ... 'xxx']
      ['n0000002'] =
          ['label'] = 2
          ['filepath']  = ['xxx', 'xxx', ... 'xxx']
    box_dict =
      ['n00007846_64193.JPEG'] = [[xmin, ymin, xmax, ymax], [...], ...]
      [xxx] = [...]
  output:
    just like raw_data
  '''
  import os
  import cv2
  output_dir = os.path.join(output_dir, 'boxed_images')
  print("saving boxed images to %s" % output_dir)
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

  removed_cnt = 0
  add_cnt = 0
  for synset in raw_data:
    file_list = raw_data[synset]['filepath']
    for f in file_list:
      f = f.strip()
      filename = f.split('/')[-1]
      if filename in box_dict:
        # remove the src un-boxed image from the raw_data lists
        raw_data[synset]['filepath'].remove(f)
        removed_cnt += 1
        # grab and save the boxing image to output_dir
        idx = 0
        for box in box_dict[filename]:
          assert isinstance(box, list) and len(box) == 4
          src_img = cv2.imread(f)
          height = src_img.shape[0]
          width = src_img.shape[1]
          xmin = int(box[0] * width)
          xmax = int(box[2] * width)
          ymin = int(box[1] * height)
          ymax = int(box[3] * height)
          boxed_img = src_img[ymin:ymax, xmin:xmax]
          img_name = filename.split('.')[0] + (
              "_%d." % idx) + filename.split('.')[1]
          boxed_imgpath = os.path.join(output_dir, img_name)
          cv2.imwrite(boxed_imgpath, boxed_img.astype(np.uint8))
          #print(boxed_imgpath)
          raw_data[synset]['filepath'].append(boxed_imgpath)
          idx += 1
          add_cnt += 1
  print("removed %d images and added %d images" % (removed_cnt, add_cnt))
  return raw_data


def random_some_val_from_train_data(raw_train, num_samples_each_kid=20):
  '''get some data randomly from train dataset as val data
  return
    train_dataset =
      ['n0000001'] =
          ['label'] = 1
          ['filepath']  = ['xxx', 'xxx', ... 'xxx']
      ['n0000002'] =
          ['label'] = 2
          ['filepath']  = ['xxx', 'xxx', ... 'xxx']
    val_dataset:
      ['label'] = [1, 4, ...]
      ['synset'] = ['n0000123', 'n0004310', ...]
      ['filepath']  = ['xxx', 'xxx', ...]
  '''
  import random
  val_dataset = dict()
  val_dataset['label'] = []
  val_dataset['synset'] = []
  val_dataset['filepath'] = []

  for synset in raw_train:
    kid = raw_train[synset]
    assert isinstance(kid.get('label'), int)
    assert isinstance(kid.get('filepath'), list)
    randomindex = random.sample(
        range(len(kid['filepath'])), num_samples_each_kid)

    val_dataset['label'] += [kid['label']] * num_samples_each_kid
    val_dataset['synset'] += [synset] * num_samples_each_kid
    val_files = [kid['filepath'][i] for i in randomindex]
    val_dataset['filepath'] += val_files
    # remove the list from raw train
    for f in val_files:
      raw_train[synset]['filepath'].remove(f)

  return raw_train, val_dataset


def save_train_txt(dataset, output_dir=FLAGS.output_directory):
  '''save dataset to a txt list like:
  label_id, filepath

  input:
    train_dataset =
      ['n0000001'] =
          ['label'] = 1
          ['filepath']  = ['xxx', 'xxx', ... 'xxx']
      ['n0000002'] =
          ['label'] = 2
          ['filepath']  = ['xxx', 'xxx', ... 'xxx']
  output train.txt: (label index, filepath)
      1,data_dir/*/*.JPEG
      32,data_dir/*/*.JPEG
      ...
      1000,data_dir/*/*.JPEG
  '''
  txt = os.path.join(output_dir, 'train.txt')
  cnt = 0  # count total samples
  with open(txt, 'wb') as out:
    for key in dataset:
      assert dataset[key].get('label',
                              0) > 0, 'should have label and start from 1'
      assert isinstance(dataset[key].get('filepath'), list), 'should have list'
      lbl = dataset[key]['label']
      for f in dataset[key]['filepath']:
        assert isinstance(f, str)
        line = '%d,%s\n' % (lbl, os.path.abspath(f))
        out.write(line)
        cnt += 1
  print('Saved %d samples list to %s' % (cnt, txt))


def save_val_txt(dataset, name='val', output_dir=FLAGS.output_directory):
  '''save dataset to a txt list like:
  label_id, filepath

  input:
    val_dataset:
      ['label'] = [1, 4, ...]
      ['synset'] = ['n0000123', 'n0004310', ...]
      ['filepath']  = ['xxx', 'xxx', ...]
  output like val.txt (label index, filepath)
    1,data_dir/*/*.JPEG
    32,data_dir/*/*.JPEG
    ...
    1000,data_dir/*/*.JPEG
  '''
  assert isinstance(dataset.get('label'), list), 'should have label'
  assert isinstance(dataset.get('filepath'), list), 'should have filepath'
  assert len(dataset['label']) == len(dataset['filepath'])
  txt = os.path.join(output_dir, '%s.txt' % name)
  cnt = 0  # count total samples
  with open(txt, 'wb') as out:
    for i in range(len(dataset['label'])):
      lbl = dataset['label'][i]
      f = dataset['filepath'][i]
      assert isinstance(f, str)
      assert lbl > 0, 'label should larger than 0'
      line = '%d,%s\n' % (lbl, os.path.abspath(f))
      out.write(line)
      cnt += 1
  print('Saved %d samples list to %s' % (cnt, txt))


def main(unused_argv):
  '''
  prepare input:
    imagenet/
      /raw_data
        /train  # point to the ILSVRC2012_img_train
        /val    # point to the ILSVRC2012_img_val
  output:
    imagenet/
      boxed_data/
        boxed.jpeg
        ...
      train.txt
        label,filename
        1,    data_dir/*/*.JPEG
        1,    data_dir/*/*.JPEG
        ...
        1000, data_dir/*/*.JPEG
      val.txt
        ...
      test.txt
        ...
  '''
  print('Saving results to %s' % FLAGS.output_directory)

  # get raw train and val dataset
  # train_dataset =
  #   ['n0000001'] =
  #      ['label'] = 1
  #      ['filepath']  = ['xxx', 'xxx', ... 'xxx']
  #   ['n0000002'] =
  #      ['label'] = 2
  #      ['filepath']  = ['xxx', 'xxx', ... 'xxx']
  # val_dataset:
  #   ['synset'] = ['n0000123', 'n0004310', ...]
  #   ['label'] = [3, 31, ...]
  #   ['filepath']  = ['xxx', 'xxx', ...]
  raw_train = get_raw_train_data('raw_data/train')
  raw_val = get_raw_val_data('raw_data/val', FLAGS.val_label_file)
  val_labels = get_val_label(raw_train, raw_val)
  raw_val['label'] = val_labels
  # print('raw train dataset: %s' % raw_train)
  # print('raw val dataset: %s' % raw_val)

  # raw train data to boxed_train_data
  if FLAGS.use_bounding_box:
    image_to_bboxes = build_bounding_box_dict(FLAGS.bounding_box_file)
    boxed_train_data = boxing_raw_data(raw_train, image_to_bboxes)
  else:
    boxed_train_data = raw_train

  # get some data from train dataset as val data
  train_dataset, val_dataset = random_some_val_from_train_data(
      boxed_train_data, num_samples_each_kid=20)

  # save as txt
  save_train_txt(train_dataset)
  save_val_txt(val_dataset, 'val')
  # use the raw val dataset as test dataset
  save_val_txt(raw_val, 'test')


if __name__ == '__main__':
  tf.app.run()
