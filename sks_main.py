# Copyright 2017 Jian Tang. All Rights Reserved.
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
  SmartKids main
"""

# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import sks_train as sks

from sks_args import parse_args
from sks_dataset import MnistDataset, ImagenetDataset
from sks_logger import logger
from sks_debugger import debugger


def main(_):
  logger.info(ARGS)
  debugger.set_debug_mode(ARGS.debug)

  if ARGS.dataset == 'mnist':
    dataset = MnistDataset(ARGS.data_dir, ARGS.data_format)
  else:
    dataset = ImagenetDataset(ARGS.data_dir, ARGS.data_format)

  # check kids is valid
  # kid number start from 1 to num_classes, skipping 0
  if ARGS.kids is None:
    ARGS.kids = range(1, dataset.num_classes() + 1)
  for i in ARGS.kids:
    assert i >= 1 and i <= dataset.num_classes(), 'invalid kid id: %d' % i
  logger.info("Train Kids:")
  logger.info(ARGS.kids)

  # tf.gfile.DeleteRecursively(ARGS.log_dir)
  if not tf.gfile.Exists(ARGS.log_dir):
    tf.gfile.MakeDirs(ARGS.log_dir)
  if not tf.gfile.Exists(ARGS.checkpoint_dir):
    tf.gfile.MakeDirs(ARGS.checkpoint_dir)

  sks.train(dataset, ARGS)


if __name__ == '__main__':
  ARGS, unparsed = parse_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
