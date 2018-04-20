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
  SmartKids Current Active Kid Info
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

index = 0
name = ''
loss_name = ''

# BatchNorm ExponentialMovingAverage
bn_ema = None
bn_moving_average_list = []
bn_avg_results = []


def set_active_kid(kid_index):
  '''set active kid, and init base info'''
  global index
  index = kid_index
  global name
  name = 'kid_%d' % index
  global loss_name
  loss_name = 'losses_%d' % index
  global bn_moving_average_list
  bn_moving_average_list = []
  global bn_ema
  bn_ema = None


def addto_bn_moving_average_list(var):
  '''add this var to moving average list'''
  global bn_moving_average_list
  bn_moving_average_list.append(var)


def get_bn_moving_result_list(ema):
  '''get the batch norm moving average reslut var list'''
  global bn_ema
  bn_ema = ema
  global bn_avg_results
  global bn_moving_average_list
  bn_avg_results = [ema.average(var) for var in bn_moving_average_list]
  return bn_avg_results


def get_moving_average(scope_name, var_type):
  '''get the moving_average var from scope name and var type('mean' or 'var')'''
  import re
  global bn_avg_results
  assert isinstance(bn_avg_results, list)
  assert var_type == 'mean' or var_type == 'var'
  mid = ''
  sp = scope_name.split('/')
  for i in range(1, len(sp)):
    mid += ('/' + sp[i])
  #print("scope name %s, mid: %s" % (scope_name, mid))
  end = 'ExponentialMovingAverage%s$' % ':0' if var_type == 'mean' else '_1:0'
  substr = '^sks(.*)' + mid + '(.*)' + end
  #print("str:%s" % substr)
  pattern = re.compile(substr)
  for var in bn_avg_results:
    # print(var.name)
    matched = re.match(pattern, var.name)
    if matched:
      # print("%s matched with %s" , (var.name, scope_name))
      return var
  assert False, "should find it in the list"
