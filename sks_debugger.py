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
  SmartKids Debugger
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# timeline and tfprof for profiling
from tensorflow.python.client import timeline
from tensorflow.contrib import tfprof

import time
import logging
from sks_logger import logger


class SKSDebugger(object):
  def set_debug_mode(self, with_debug):
    '''
    set use debug mode or not
    '''
    self.__with_debug__ = with_debug
    if with_debug:
      logger.setLevel(logging.DEBUG)
    else:
      logger.setLevel(logging.INFO)

  def set_skip_var(self, skip_var):
    self.__skip_var__ = True if not self.__with_debug__ else skip_var

  def set_skip_hist(self, skip_hist):
    self.__skip_hist__ = True if not self.__with_debug__ else skip_hist

  def __init__(self, with_debug, skip_var=False, skip_hist=False):
    self.set_debug_mode(with_debug)
    # skip adding var summaries or adding_histogram_summaries
    self.set_skip_var(skip_var)
    self.set_skip_hist(skip_hist)
    self.__var_summary_name__ = 'summaries'
    self.__hist_summary_name__ = 'summaries'
    self.summary_writer = None

  def add_var_summaries(self, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if self.__skip_var__:
      return
    with tf.name_scope(self.__var_summary_name__):  # add var name
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def add_histogram_summaries(self, name, x):
    '''add histogram summaries
    '''
    if self.__skip_hist__:
      return
    with tf.name_scope(self.__hist_summary_name__):
      tf.summary.histogram(name, x)

  def add_scalar_summaries(self, name, x):
    if not self.__with_debug__:
      return
    tf.summary.scalar(name, x)

  def save_timeline(self, filename, run_metadata):
    '''save profiling timeline data with tf.RunMetadata()
    run_metadata: generate in sess.run()
    '''
    if not self.__with_debug__:
      return
    if run_metadata is None:
      logger.warning("run_metadata is none, skip save timeline")
      return
    assert isinstance(filename, str), 'filename should be an string!'
    logger.debug('save timeline to: ' + filename)
    with open(filename, 'w') as f:
      tl = timeline.Timeline(run_metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      f.write(ctf)

  def save_tfprof(self, prefix_path, graph, run_metadata=None):
    '''save TFprof all files
    includes:
      params.log        - trainable params
      flops.log         - float_ops
      timing_memory.log - timeing memory
      device.log        - params on device
    if graph is none will get default graph from tf
    '''
    if not self.__with_debug__:
      return
    if tf.__version__[0] == '0':
      # 0.12 do not support below
      return
    # Print trainable variable parameter statistics to stdout
    if graph is None:
      logger.warning("input graph is none, use tf.get_default_graph() instead")
      graph = tf.get_default_graph()
    assert isinstance(prefix_path, str), 'filename should be an string!'

    logger.debug('save TFprof to: ' + prefix_path)
    analyzer = tfprof.model_analyzer

    prof_options = analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
    prof_options['output'] = 'file:outfile=' + prefix_path + "/params.log"
    analyzer.print_model_analysis(
        graph, run_meta=run_metadata, tfprof_options=prof_options)

    prof_options = analyzer.FLOAT_OPS_OPTIONS
    prof_options['output'] = 'file:outfile=' + prefix_path + "/flops.log"
    analyzer.print_model_analysis(
        graph, run_meta=run_metadata, tfprof_options=prof_options)

    prof_options = analyzer.PRINT_ALL_TIMING_MEMORY
    prof_options[
        'output'] = 'file:outfile=' + prefix_path + "/timing_memory.log"
    analyzer.print_model_analysis(
        graph, run_meta=run_metadata, tfprof_options=prof_options)

    prof_options = analyzer.PRINT_PARAMS_ON_DEVICE
    prof_options['output'] = 'file:outfile=' + prefix_path + "/device.log"
    analyzer.print_model_analysis(
        graph, run_meta=run_metadata, tfprof_options=prof_options)

  def init_merge_all(self, log_dir, graph):
    '''merge summary and init FileWriter
    Merge all the summaries and write them out to log_dir
    return run_options, run_metadata
    '''
    if not self.__with_debug__:
      return None, None
    self.summary_op = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(log_dir, graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    return run_options, run_metadata

  def save_summaries(self, sess, feed_dict, run_options, run_metadata, index):
    '''save the summaries
    '''
    if not self.__with_debug__:
      return
    s = time.time()
    summary = sess.run(
        self.summary_op,
        feed_dict=feed_dict,
        options=run_options,
        run_metadata=run_metadata)
    self.summary_writer.add_run_metadata(run_metadata, 'iter%03d' % index)
    self.summary_writer.add_summary(summary, index)
    d = time.time() - s
    logger.debug('save summary %.3f sec' % d)

  def __del__(self):
    if not self.__with_debug__:
      return
    if self.summary_writer is not None:
      self.summary_writer.close()


debugger = SKSDebugger(False)
