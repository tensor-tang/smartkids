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
  SmartKids
"""

# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse


def parse_args():
  '''parse arguments
  return [args, unparsed args]
  '''
  parser = argparse.ArgumentParser()
  #######################
  # Kids needs training #
  #######################
  parser.add_argument(
      '--kids',
      required=False,
      nargs='+',
      type=int,
      help=
      'List the kids id you want to train. The number start from 1 to num_classes, skipping 0. eg. --kids 1 3 5'
  )

  #######################
  # Dataset Flags       #
  #######################
  parser.add_argument(
      '--data_dir',
      required=False,
      type=str,
      default='./dataset',
      help='Directory to load the dataset')
  parser.add_argument(
      '--dataset',
      required=True,
      type=str,
      default='mnist',
      help='Which dataset to choose, only support mnist and imagenet yet')
  parser.add_argument(
      '--batch_size',
      required=False,
      type=int,
      default=64,
      help='batch size for training.')
  parser.add_argument(
      '--val_batch_size',
      required=False,
      type=int,
      default=50,
      help='batch size for validation or test dataset in inference.')
  parser.add_argument(
      '--use_fp16',
      required=False,
      type=bool,
      default=False,
      help='Whether to use float32 data type.')
  parser.add_argument(
      '--use_dummy',
      required=False,
      type=bool,
      default=True,
      help='If true, uses dummy(fake) data.')
  parser.add_argument(
      '--use_gpu',
      required=False,
      type=bool,
      default=True,
      help='Use GPU for training and inference')
  parser.add_argument(
      '--data_format',
      required=False,
      type=str,
      default='nchw',
      choices=["nchw", "nhwc"],
      help='data format only support \"nchw\" or \"nhwc\"')
  parser.add_argument(
      '--inference_only',
      required=False,
      type=bool,
      default=False,
      help='Only run inference')
  #######################
  # Learning Rate Flags #
  #######################
  parser.add_argument(
      '--learning_rate',
      required=False,
      type=float,
      default=1e-3,
      help='Initial learning rate')
  parser.add_argument(
      '--learning_rate_decay_rate',
      required=False,
      type=float,
      default=0.95,
      help='Learning rate decay rate in exponential_decay')
  #######################
  # Stop conditions     #
  #######################
  ## stop when (iter > max_iter) || (epoch > num_epoch)
  parser.add_argument(
      '-m',
      '--max_iter',
      required=False,
      type=int,
      default=int(1e+5),
      help='Number of iterations to run trainer.')
  parser.add_argument(
      '--num_epochs',
      required=False,
      type=float,
      default=10.0,
      help='Number of epoches needed to be trained.')
  parser.add_argument(
      '--end_learning_rate',
      required=False,
      type=float,
      default=1e-4,
      help=
      'The minimal end learning rate used by a polynomial decay learning rate.')
  #######################
  # Logging Flags       #
  #######################
  parser.add_argument(
      '--log_dir',
      required=False,
      type=str,
      default='./logs',
      help='Directory to save the logs')
  parser.add_argument(
      '--log_every_n_iters',
      required=False,
      type=int,
      default=10,
      help='The frequency with which logs are print.')
  parser.add_argument(
      '--val_every_n_epoch',
      required=False,
      type=float,
      default=0.2,
      help='validation every n epoch.')
  parser.add_argument(
      '--test_every_n_epoch',
      required=False,
      type=float,
      default=1.0,
      help='test all test_dataset every n epoch.')
  parser.add_argument(
      '--checkpoint_dir',
      required=False,
      type=str,
      default='./models',
      help='Directory to load and save checkpoints')
  parser.add_argument(
      '--ckpt_every_n_epoch',
      required=False,
      type=float,
      default=0.5,
      help='Save checkpoint every n epoch.')
  #######################
  # Debug Flags         #
  #######################
  ## --debug or --no_debug
  debug_parser = parser.add_mutually_exclusive_group(required=False)
  debug_parser.add_argument(
      '--debug',
      dest='debug',
      action='store_true',
      help='run in debug mode and logging debug')
  debug_parser.add_argument(
      '--no_debug',
      dest='debug',
      action='store_false',
      help='run in no debug mode and do not logging debug')
  parser.set_defaults(debug=False)
  #parser.add_argument(
  #'--debug',
  #action='store_true',
  #help='run in debug mode and logging debug')
  parser.add_argument(
      '--profil_every_n_epoch',
      type=float,
      default=2.0,
      help='Save profiling data(and timeline) every n epoch')
  parser.add_argument(
      '--summary_every_n_epoch',
      required=False,
      type=float,
      default=0.5,
      help='Save summary every n epoch.')
  args, unparsed = parser.parse_known_args()
  ## change all relative path to abs and uppercase format
  args.log_dir = os.path.abspath(args.log_dir)
  args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
  args.data_format = args.data_format.lower()
  assert args.data_format in ['nchw',
                              'nhwc'], "do not support %s" % args.data_format
  args.dataset = args.dataset.lower()
  assert args.dataset in ['mnist',
                          'imagenet'], "do not support %s" % args.dataset

  # TODO: handle unparsed
  return args, unparsed
