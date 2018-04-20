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
  SmartKids train
"""

# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import tensorflow as tf
import sks_model as sks
import numpy as np
import sks_active_kid as active_kid

from sks_args import parse_args
from sks_logger import logger
from sks_debugger import debugger

ARGS = None
num_classes = None

BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997


def cpu_setting(set=False):
  if not set:
    return 0
  os.environ["KMP_BLOCKTIME"] = "1"
  os.environ["KMP_SETTINGS"] = "1"
  os.environ["OMP_NUM_THREADS"] = "32"
  os.environ["MKL_NUM_THREADS"] = "32"
  os.environ["OMP_DYNAMIC"] = "false"
  os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,0,0"


def data_type():
  """Return the data type of the activations, weights, and placeholder variables."""
  if ARGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def save_reports(reports, prefix):
  '''save reports to txt
  '''
  import pickle
  # save dict
  with open(prefix + '//reports.txt', "wb") as f:
    pickle.dump(reports, f)


def check_kid_report(report):
  '''check this report is one kid report
  report: (dict)
    ['train']: (dict)
        ['epoch'] : [...]
        ['loss']  : [...]
        ['lr']    : [...]
        ['f1']    : [...]
        ['acc']   : [...]
        ['pprec'] : [...]  # positive precsion
        ['nprec'] : [...]  # negative precsion
        ['tpr']   : [...]  # true positive rate == recall
        ['tnr']   : [...]  # true negative rate
    ['val']: (dict)
        ['epoch'] : [...]
        ['loss']  : [...]
        ['f1']    : [...]
        ['acc']   : [...]
        ['pprec'] : [...]
        ['nprec'] : [...]
        ['tpr']   : [...]
        ['tnr']   : [...]
    ['test']: (dict)
        ['epoch'] : [...]
        ['loss']  : [...]
        ['f1']    : [...]
        ['acc']   : [...]
        ['pprec'] : [...]
        ['nprec'] : [...]
        ['tpr']   : [...]
        ['tnr']   : [...]
  '''
  assert isinstance(report, dict)
  assert isinstance(report.get('train', 0), dict)
  assert isinstance(report.get('val', 0), dict)
  assert isinstance(report.get('test', 0), dict)
  assert isinstance(report['train'].get('epoch', 0), list)
  assert isinstance(report['train'].get('loss', 0), list)
  assert isinstance(report['train'].get('lr', 0), list)
  assert isinstance(report['train'].get('f1', 0), list)
  assert isinstance(report['train'].get('acc', 0), list)
  assert isinstance(report['train'].get('pprec', 0), list)
  assert isinstance(report['train'].get('nprec', 0), list)
  assert isinstance(report['train'].get('tpr', 0), list)
  assert isinstance(report['train'].get('tnr', 0), list)
  assert isinstance(report['val'].get('epoch', 0), list)
  assert isinstance(report['val'].get('loss', 0), list)
  assert isinstance(report['val'].get('f1', 0), list)
  assert isinstance(report['val'].get('acc', 0), list)
  assert isinstance(report['val'].get('pprec', 0), list)
  assert isinstance(report['val'].get('nprec', 0), list)
  assert isinstance(report['val'].get('tpr', 0), list)
  assert isinstance(report['val'].get('tnr', 0), list)
  assert isinstance(report['test'].get('epoch', 0), list)
  assert isinstance(report['test'].get('loss', 0), list)
  assert isinstance(report['test'].get('f1', 0), list)
  assert isinstance(report['test'].get('acc', 0), list)
  assert isinstance(report['test'].get('pprec', 0), list)
  assert isinstance(report['test'].get('nprec', 0), list)
  assert isinstance(report['test'].get('tpr', 0), list)
  assert isinstance(report['test'].get('tnr', 0), list)


def load_reports(prefix):
  '''load reports if exist, otherwise create one
  return dict
  '''
  import pickle
  import os
  filename = prefix + '//reports.txt'
  if os.path.isfile(filename):
    logger.info("last report %s" % filename)
    with open(filename, "rb") as f:
      reports = pickle.load(f)
      logger.info("last overall report: %s" % reports['overall'])
  else:
    reports = dict()
    for kid in range(1, num_classes + 1):  # [1, num_classes]
      reports[kid] = dict()
      reports[kid]['train'] = dict()
      reports[kid]['val'] = dict()
      reports[kid]['test'] = dict()
      reports[kid]['train']['epoch'] = []
      reports[kid]['train']['loss'] = []
      reports[kid]['train']['lr'] = []
      reports[kid]['train']['f1'] = []
      reports[kid]['train']['acc'] = []
      reports[kid]['train']['pprec'] = []
      reports[kid]['train']['nprec'] = []
      reports[kid]['train']['tpr'] = []
      reports[kid]['train']['tnr'] = []
      reports[kid]['val']['epoch'] = []
      reports[kid]['val']['loss'] = []
      reports[kid]['val']['f1'] = []
      reports[kid]['val']['acc'] = []
      reports[kid]['val']['pprec'] = []
      reports[kid]['val']['nprec'] = []
      reports[kid]['val']['tpr'] = []
      reports[kid]['val']['tnr'] = []
      reports[kid]['test']['epoch'] = []
      reports[kid]['test']['loss'] = []
      reports[kid]['test']['f1'] = []
      reports[kid]['test']['acc'] = []
      reports[kid]['test']['pprec'] = []
      reports[kid]['test']['nprec'] = []
      reports[kid]['test']['tpr'] = []
      reports[kid]['test']['tnr'] = []
    reports['overall'] = dict()

  for kid in range(1, num_classes + 1):
    check_kid_report(reports.get(kid, 0))
  return reports


def plot_reports(reports, prefix, in_one_pic=False):
  import matplotlib.pyplot as plt
  assert len(reports) >= num_classes
  # TODO: plot acc, tpr and tnr
  if in_one_pic:
    fig_name = 'Overall_kids'
    fig, axes = plt.subplots(
        num=fig_name, figsize=(8, 12), dpi=200, nrows=3, ncols=1)
    fig.suptitle(fig_name, fontsize=18, weight='bold')
    fig_train = axes[0]
    fig_val = axes[1]
    fig_test = axes[2]
  for kid in range(1, num_classes + 1):
    train = reports[kid]['train']
    val = reports[kid]['val']
    test = reports[kid]['test']

    legend_label = 'kid_%d' % kid
    if not in_one_pic:
      # kid figure
      fig_name = 'kid_%d' % kid
      fig, axes = plt.subplots(
          num=fig_name, figsize=(8, 12), dpi=200, nrows=3, ncols=1)
      fig.suptitle(fig_name, fontsize=18, weight='bold')
      fig_train = axes[0]
      fig_val = axes[1]
      fig_test = axes[2]

    # train figure
    assert len(train['epoch']) == len(train['loss'])
    fig_train.plot(train['epoch'], train['loss'], label=legend_label)  #, 'b.-')
    # fig_train.axis([0, 6, 0, 20])
    fig_train.set_title("Train loss and lr")
    fig_train.set_xlabel('epoch')
    fig_train.set_ylabel('train loss')
    # share same x-axis
    ax_lr = fig_train.twinx()
    ax_lr.plot(train['epoch'], train['lr'], 'r.-')
    #ax_lr.set_xlim([0, np.e])
    ax_lr.set_xlabel('epoch')
    ax_lr.set_ylabel('learning rate')
    ax_lr.grid(True)

    # val figure
    assert len(val['epoch']) == len(val['acc'])
    fig_val.plot(val['epoch'], val['acc'], '.-', label=legend_label)
    fig_val.set_title("Val acc")
    fig_val.set_xlabel('epoch')
    fig_val.set_ylabel('val acc')
    fig_val.grid(True)

    # test figure
    assert len(test['epoch']) == len(test['acc'])
    fig_test.plot(test['epoch'], test['acc'], '.-', label=legend_label)
    fig_test.set_title("Test acc")
    fig_test.set_xlabel('epoch')
    fig_test.set_ylabel('test acc')
    fig_test.grid(True)

    # layout
    fig.tight_layout(rect=(0, 0, 1, 0.96))  #(left, bottom, right, top)

    # save figure
    if in_one_pic:
      # Now add the legend with some customizations.
      fig_val.legend(loc='right', shadow=True)
      # only save the last one
      if kid == num_classes:
        fig.savefig(prefix + '/' + fig_name + '.png')
    else:
      fig.savefig(prefix + '/' + fig_name + '.png')


def train_one_kid(kid, dataset, report):
  ''' train one kid
  param:
    kid (input): kid index
    dataset (input): dataset
    report (output).
  return the last pred_values of test data: the logist of test ops includes sigoimd!
  '''
  active_kid.set_active_kid(kid)
  logger.info('Training ' + active_kid.name)
  assert kid <= dataset.num_classes() and kid >= 1, 'invalid kid id: %d' % kid
  input_channels = dataset.num_channels()
  img_height = dataset.img_height()
  img_width = dataset.img_width()
  dims = input_channels * img_height * img_width
  with tf.name_scope('inputs'):
    train_data = tf.placeholder(
        dtype=data_type(), shape=[ARGS.batch_size, dims])
    train_label = tf.placeholder(tf.int64, shape=[ARGS.batch_size])
    val_data = tf.placeholder(
        dtype=data_type(), shape=[ARGS.val_batch_size, dims])
    test_data = tf.placeholder(
        dtype=data_type(), shape=[ARGS.val_batch_size, dims])

  with tf.variable_scope('sks'):
    logits = sks.infer_one_kid(train_data, input_channels, img_height,
                               img_width, True, ARGS.data_format, ARGS.use_gpu,
                               data_type())
    loss = sks.loss_one_kid(logits, train_label)

  with tf.name_scope('trainer'):
    ################# MomentumTrainer #################
    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch_num = tf.Variable(0, dtype=data_type())  #, trainable=False)
    #global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    # TODO: self-design lr policy
    learning_rate = tf.train.exponential_decay(
        ARGS.learning_rate,
        global_step=batch_num * ARGS.batch_size,
        decay_steps=dataset.train_samples() / 25,  # tested 25 is better than 50
        decay_rate=ARGS.learning_rate_decay_rate,
        staircase=True
    )  #True means (global_step / decay_steps) is an integer division
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    debugger.add_scalar_summaries('learning_rate_kid_%d' % kid, learning_rate)
    opt_op = optimizer.minimize(loss, global_step=batch_num)
    # batchnorm moving average
    bn_averages = tf.train.ExponentialMovingAverage(
        BATCHNORM_MOVING_AVERAGE_DECAY, batch_num)
    bn_averages_op = bn_averages.apply(active_kid.bn_moving_average_list)
    trainer = tf.group(opt_op, bn_averages_op)
    avg_result_list = active_kid.get_bn_moving_result_list(bn_averages)
    for var in avg_result_list:
      # logger.info(var.name)
      trainer = tf.group(trainer, var)

    ################# MomentumTrainer #################
    # optimizer = tf.train.AdamOptimizer(ARGS.learning_rate)
    # trainers.append(optimizer.minimize(losses[kid]))
    # Predictions for the test and validation

  with tf.variable_scope('sks', reuse=True):
    val_infer_ops = sks.infer_one_kid(val_data, input_channels, img_height,
                                      img_width, False, ARGS.data_format,
                                      ARGS.use_gpu, data_type())
    test_infer_ops = sks.infer_one_kid(test_data, input_channels, img_height,
                                       img_width, False, ARGS.data_format,
                                       ARGS.use_gpu, data_type())

  # Create a saver. TODO: how about only inference?
  saver = tf.train.Saver(tf.global_variables())
  check_kid_report(report)
  train_report = report['train']
  val_report = report['val']
  test_report = report['test']
  with tf.Session() as sess:
    # sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = False, log_device_placement = False, inter_op_parallelism_threads = 8, intra_op_parallelism_threads = 32))
    ckp_path = ARGS.checkpoint_dir + '/kid_%d' % kid
    last_iter = sks.init_variables(sess, saver, ckp_path)
    # TODO: do not need merge all, only need merge this kid's ops
    run_options, run_metadata = debugger.init_merge_all(ARGS.log_dir,
                                                        sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sum_time = 0
    min_time = float('inf')
    total_start = time.time()
    log_val = 1
    log_test = 1
    log_ckpt = 1
    log_prof = 0
    log_summary = 0
    i = 0
    while (i < ARGS.max_iter):
      epoch = i * ARGS.batch_size / dataset.train_samples()
      if (epoch > ARGS.num_epochs):
        break
      global_iter = last_iter + i

      start_time = time.time()
      batch_data, batch_label = dataset.next_batch(ARGS.batch_size, kid)
      feed_dict = {train_data: batch_data, train_label: batch_label}
      sess.run(
          trainer,
          feed_dict=feed_dict,
          options=run_options,
          run_metadata=run_metadata)
      duration = time.time() - start_time
      sum_time += duration
      min_time = duration if min_time > duration else min_time

      # print and save log
      if i % ARGS.log_every_n_iters == 0:
        _logits, _loss, _lr = sess.run(
            [logits, loss, learning_rate],
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata)
        train_report['epoch'].append(epoch)
        train_report['loss'].append(_loss)
        train_report['lr'].append(_lr)

        # cal the training acc
        _labels = sks.convert_label_to_kid_label(batch_label, kid)
        #print("src label: %s" % batch_label)
        #print("dst kid label: %s" % _labels)
        #print("src logits: %s" % _logits.reshape(len(_labels)))
        _preds = _logits.reshape(len(_labels))
        #print("predicts: %s" % _preds)
        _f1, _acc, _pprec, _nprec, _tpr, _tnr = sks.cpu_acc_one_kid(
            _preds, _labels)
        train_report['f1'].append(_f1)
        train_report['acc'].append(_acc)
        train_report['pprec'].append(_pprec)
        train_report['nprec'].append(_nprec)
        train_report['tpr'].append(_tpr)
        train_report['tnr'].append(_tnr)

        # print logs
        format_str = 'kid: %d, iter %d (epoch %.2f), lr %.6f, loss : %.4f, f1 %.2f%%, acc: %.2f%%, pprec: %.2f%%, nprec: %.2f%%, tpr: %.2f%%, tnr: %.2f%% (%.1f ms/iter)'
        log_content = (kid, i, epoch, _lr, _loss, _f1 * 100, _acc * 100,
                       _pprec * 100, _nprec * 100, _tpr * 100, _tnr * 100,
                       duration * 1000)
        logger.info(format_str % log_content)

      def log_test_results(infer_ops, subset, feed_name, report):
        '''
        obtain loss, acc, tpr, tnr for infer ops,
        log and save the report
        return pred_values
        '''
        assert subset in ['val', 'test']
        logits_values = sks.infer_one_kid_with_all_dataset_in_batches(
            sess, infer_ops, dataset, subset, feed_name, ARGS.val_batch_size,
            ARGS.use_fp16)
        raw_labels = dataset.infer_labels(subset)
        labels_values = sks.convert_label_to_kid_label(raw_labels, kid)
        loss_value = sks.cpu_loss_one_kid(logits_values, labels_values)
        pred_values = logits_values
        f1, acc, pprec, nprec, tpr, tnr = sks.cpu_acc_one_kid(
            pred_values, labels_values)
        report['loss'].append(loss_value)
        report['f1'].append(f1)
        report['acc'].append(acc)
        report['pprec'].append(pprec)
        report['nprec'].append(nprec)
        report['tpr'].append(tpr)
        report['tnr'].append(tnr)
        log_content = (kid, loss_value, f1 * 100, acc * 100.0, pprec * 100,
                       nprec * 100, tpr * 100.0, tnr * 100.0)
        logger.info(
            'kid: %d, loss: %.4f, f1 %.2f%%, acc: %.2f%%, pprec: %.2f%%, nprec: %.2f%%, tpr: %.2f%%, tnr: %.2f%%'
            % log_content)
        return pred_values

      if epoch / ARGS.val_every_n_epoch > log_val or (i + 1) == ARGS.max_iter:
        logger.info('kid: %d, epoch %.2f, validation info:' % (kid, epoch))
        log_test_results(val_infer_ops, 'val', val_data, val_report)
        val_report['epoch'].append(epoch)
        log_val += 1

      if epoch / ARGS.test_every_n_epoch > log_test or (i + 1) == ARGS.max_iter:
        logger.info('kid: %d, epoch %.2f, test info: -------- ' % (kid, epoch))
        log_test_results(test_infer_ops, 'test', test_data, test_report)
        test_report['epoch'].append(epoch)
        log_test += 1

      if epoch / ARGS.ckpt_every_n_epoch > log_ckpt or (i + 1) == ARGS.max_iter:
        checkpoint_path = ckp_path + '/kid_%d_model' % kid
        saver.save(sess, checkpoint_path, global_step=global_iter)
        logger.info('save checkpoint to %s-%d' % (checkpoint_path, global_iter))
        log_ckpt += 1

      if epoch / ARGS.profil_every_n_epoch > log_prof:
        filename = ARGS.log_dir + ('/timeline_iter%d' % global_iter) + '.json'
        debugger.save_timeline(filename, run_metadata)
        debugger.save_tfprof(ARGS.log_dir, sess.graph, run_metadata)
        log_prof += 1

      if epoch % ARGS.summary_every_n_epoch > log_summary:
        debugger.save_summaries(sess, feed_dict, run_options, run_metadata, i)
        log_summary += 1

      sys.stdout.flush()
      i += 1
    test_pred = log_test_results(test_infer_ops, 'test', test_data, test_report)
    coord.request_stop()
    coord.join(threads)
    content = (time.time() - total_start, sum_time * 1000.0 / i,
               min_time * 1000.0)
    spend_time = 'total time: %.1f sec, train avg time: %.3f ms, train min_time: %.3f ms' % content
    logger.info(spend_time)
    report['spend_time'] = spend_time

  return test_pred


def predict_overall_from_kids(kids_predicts,
                              val_rates,
                              dataset,
                              use_belief=False,
                              threshold=0.5):
  '''predict overall from all avaiable kids on test data.
  if some kids do not have data, use 0.5.
  can choose whether to use the belief policy.
  return the prediction(shape: (test_num, num_classes)) overall
  '''
  import numpy as np
  res = []
  num_samples = dataset.test_samples()
  num_classes = dataset.num_classes()
  for kid in range(1, num_classes + 1):
    logits = kids_predicts.get(kid, 0)
    if isinstance(logits, int):
      # have not predict this kid
      logits = [0.5] * num_samples
    assert len(logits) == num_samples
    rate = val_rates.get(kid, 0)
    logger.info("kid %d belief rate:" % kid)
    logger.info(rate)
    if isinstance(rate, int):
      # have not pre-validate rate
      rate = dict()
      rate['acc'] = 0.0
      rate['pprec'] = 0.0
      rate['nprec'] = 0.0
    assert rate.get('acc', -1) != -1 and \
        rate.get('pprec', -1) != -1 and \
        rate.get('nprec', -1) != -1
    if use_belief:
      logger.warning("need optimaze")
      # TODO: optimaze it
      for i in xrange(num_samples):
        if logits[i] > threshold:
          logits[i] = threshold + (logits[i] - threshold) * rate['pprec']
        else:
          logits[i] = threshold - (threshold - logits[i]) * rate['nprec']
    res.append(logits)
  out = np.array(res)
  out = out.transpose()
  assert out.shape == (num_samples, num_classes)
  for i in xrange(num_samples):
    out[i] = sks.cpu_softmax(out[i])
  return out


def train(dataset, args):
  '''sks main train loop
  '''
  global ARGS
  ARGS = args
  cpu_setting()
  global num_classes
  num_classes = dataset.num_classes()

  total_start = time.time()
  reports = load_reports(ARGS.log_dir)
  kids_predictions_on_test_data = dict()
  val_rates = dict()

  for kid in ARGS.kids:
    kids_predictions_on_test_data[kid] = train_one_kid(kid, dataset,
                                                       reports[kid])
    val_rates[kid] = dict()
    # for justice, only use validation's f1, acc, pprec, nprec, tpr and tnr
    # for belief policy.
    # DO NOT use test acc, because we will finnaly check on test data.
    # it is cheat if we use test acc.
    f1 = reports[kid]['val']['f1']
    val_rates[kid]['f1'] = f1[len(f1) - 1]
    acc = reports[kid]['val']['acc']
    val_rates[kid]['acc'] = acc[len(acc) - 1]
    pprec = reports[kid]['val']['pprec']
    val_rates[kid]['pprec'] = pprec[len(pprec) - 1]
    nprec = reports[kid]['val']['nprec']
    val_rates[kid]['nprec'] = nprec[len(nprec) - 1]
    tpr = reports[kid]['val']['tpr']
    val_rates[kid]['tpr'] = tpr[len(tpr) - 1]
    tnr = reports[kid]['val']['tnr']
    val_rates[kid]['tnr'] = tnr[len(tnr) - 1]

  test_predict = predict_overall_from_kids(kids_predictions_on_test_data,
                                           val_rates, dataset)
  test_acc, kids_err, kids_counts = sks.accuracy_rate(
      test_predict, dataset.infer_labels('test'), dataset.num_classes())

  test_predict_with_policy = predict_overall_from_kids(
      kids_predictions_on_test_data, val_rates, dataset, True)
  test_acc_with_policy, kids_err_with_policy, kids_counts_with_policy = sks.accuracy_rate(
      test_predict_with_policy, dataset.infer_labels('test'),
      dataset.num_classes())
  logger.info('test accruracy        = %.2f%%' % test_acc)
  logger.info('test accruracy policy = %.2f%%' % test_acc_with_policy)

  logger.info('test accruracy = %.2f%%' % test_acc)
  reports['overall']['test_accruracy'] = test_acc
  reports['overall']['kids_err'] = kids_err
  reports['overall']['test_samples_of_every_kid'] = kids_counts
  spend_time = 'total time: %.1f sec' % (time.time() - total_start)
  reports['overall']['spend_time'] = spend_time
  logger.info(spend_time)
  logger.info("saving reports...")
  logger.debug(reports)
  save_reports(reports, ARGS.log_dir)
  plot_reports(reports, ARGS.log_dir, False)
  plot_reports(reports, ARGS.log_dir, True)
