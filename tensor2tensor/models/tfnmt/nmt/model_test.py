# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Tests for model.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import sys
import numpy as np
import tensorflow as tf

from . import attention_model
from . import gnmt_model
from . import model
from .utils import common_test_utils
from .utils import nmt_utils


float32 = np.float32
int32 = np.int32
array = np.array


class ModelTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.actual_vars_values = {}
    cls.expected_vars_values = {
        'AttentionMechanismBahdanau/att_layer_weight/shape': (10, 5),
        'AttentionMechanismBahdanau/att_layer_weight/sum':
            -0.64981574,
        'AttentionMechanismBahdanau/last_dec_weight/shape': (10, 20),
        'AttentionMechanismBahdanau/last_dec_weight/sum':
            0.058069646,
        'AttentionMechanismBahdanau/last_enc_weight/shape': (10, 20),
        'AttentionMechanismBahdanau/last_enc_weight/sum':
            0.058028102,
        'AttentionMechanismLuong/att_layer_weight/shape': (10, 5),
        'AttentionMechanismLuong/att_layer_weight/sum':
            -0.64981574,
        'AttentionMechanismLuong/last_dec_weight/shape': (10, 20),
        'AttentionMechanismLuong/last_dec_weight/sum':
            0.058069646,
        'AttentionMechanismLuong/last_enc_weight/shape': (10, 20),
        'AttentionMechanismLuong/last_enc_weight/sum':
            0.058028102,
        'AttentionMechanismNormedBahdanau/att_layer_weight/shape': (10, 5),
        'AttentionMechanismNormedBahdanau/att_layer_weight/sum':
            -0.64981973,
        'AttentionMechanismNormedBahdanau/last_dec_weight/shape': (10, 20),
        'AttentionMechanismNormedBahdanau/last_dec_weight/sum':
            0.058067322,
        'AttentionMechanismNormedBahdanau/last_enc_weight/shape': (10, 20),
        'AttentionMechanismNormedBahdanau/last_enc_weight/sum':
            0.058022559,
        'AttentionMechanismScaledLuong/att_layer_weight/shape': (10, 5),
        'AttentionMechanismScaledLuong/att_layer_weight/sum':
            -0.64981574,
        'AttentionMechanismScaledLuong/last_dec_weight/shape': (10, 20),
        'AttentionMechanismScaledLuong/last_dec_weight/sum':
            0.058069646,
        'AttentionMechanismScaledLuong/last_enc_weight/shape': (10, 20),
        'AttentionMechanismScaledLuong/last_enc_weight/sum':
            0.058028102,
        'GNMTModel_gnmt/last_dec_weight/shape': (15, 20),
        'GNMTModel_gnmt/last_dec_weight/sum':
            -0.48634407,
        'GNMTModel_gnmt/last_enc_weight/shape': (10, 20),
        'GNMTModel_gnmt/last_enc_weight/sum':
            0.058025002,
        'GNMTModel_gnmt/mem_layer_weight/shape': (5, 5),
        'GNMTModel_gnmt/mem_layer_weight/sum':
            -0.44815454,
        'GNMTModel_gnmt_v2/last_dec_weight/shape': (15, 20),
        'GNMTModel_gnmt_v2/last_dec_weight/sum':
            -0.48634392,
        'GNMTModel_gnmt_v2/last_enc_weight/shape': (10, 20),
        'GNMTModel_gnmt_v2/last_enc_weight/sum':
            0.058024824,
        'GNMTModel_gnmt_v2/mem_layer_weight/shape': (5, 5),
        'GNMTModel_gnmt_v2/mem_layer_weight/sum':
            -0.44815454,
        'NoAttentionNoResidualUniEncoder/last_dec_weight/shape': (10, 20),
        'NoAttentionNoResidualUniEncoder/last_dec_weight/sum':
            0.057424068,
        'NoAttentionNoResidualUniEncoder/last_enc_weight/shape': (10, 20),
        'NoAttentionNoResidualUniEncoder/last_enc_weight/sum':
            0.058453858,
        'NoAttentionResidualBiEncoder/last_dec_weight/shape': (10, 20),
        'NoAttentionResidualBiEncoder/last_dec_weight/sum':
            0.058025062,
        'NoAttentionResidualBiEncoder/last_enc_weight/shape': (10, 20),
        'NoAttentionResidualBiEncoder/last_enc_weight/sum':
            0.058053195,
        'UniEncoderBottomAttentionArchitecture/last_dec_weight/shape': (10, 20),
        'UniEncoderBottomAttentionArchitecture/last_dec_weight/sum':
            0.058024943,
        'UniEncoderBottomAttentionArchitecture/last_enc_weight/shape': (10, 20),
        'UniEncoderBottomAttentionArchitecture/last_enc_weight/sum':
            0.058025122,
        'UniEncoderBottomAttentionArchitecture/mem_layer_weight/shape': (5, 5),
        'UniEncoderBottomAttentionArchitecture/mem_layer_weight/sum':
            -0.44815454,
        'UniEncoderStandardAttentionArchitecture/last_dec_weight/shape': (10,
                                                                          20),
        'UniEncoderStandardAttentionArchitecture/last_dec_weight/sum':
            0.058025002,
        'UniEncoderStandardAttentionArchitecture/last_enc_weight/shape': (10,
                                                                          20),
        'UniEncoderStandardAttentionArchitecture/last_enc_weight/sum':
            0.058024883,
        'UniEncoderStandardAttentionArchitecture/mem_layer_weight/shape': (5,
                                                                           5),
        'UniEncoderStandardAttentionArchitecture/mem_layer_weight/sum':
            -0.44815454,
    }

    cls.actual_train_values = {}
    cls.expected_train_values = {
        'AttentionMechanismBahdanau/loss': 8.8519039,
        'AttentionMechanismLuong/loss': 8.8519039,
        'AttentionMechanismNormedBahdanau/loss': 8.851902,
        'AttentionMechanismScaledLuong/loss': 8.8519039,
        'GNMTModel_gnmt/loss': 8.8519087,
        'GNMTModel_gnmt_v2/loss': 8.8519087,
        'NoAttentionNoResidualUniEncoder/loss': 8.8516064,
        'NoAttentionResidualBiEncoder/loss': 8.851984,
        'UniEncoderStandardAttentionArchitecture/loss': 8.8519087,
        'InitializerGlorotNormal/loss': 8.9779415,
        'InitializerGlorotUniform/loss': 8.7643699,
    }

    cls.actual_eval_values = {}
    cls.expected_eval_values = {
        'AttentionMechanismBahdanau/loss': 8.8517132,
        'AttentionMechanismBahdanau/predict_count': 11.0,
        'AttentionMechanismLuong/loss': 8.8517132,
        'AttentionMechanismLuong/predict_count': 11.0,
        'AttentionMechanismNormedBahdanau/loss': 8.8517132,
        'AttentionMechanismNormedBahdanau/predict_count': 11.0,
        'AttentionMechanismScaledLuong/loss': 8.8517132,
        'AttentionMechanismScaledLuong/predict_count': 11.0,
        'GNMTModel_gnmt/loss': 8.8443403,
        'GNMTModel_gnmt/predict_count': 11.0,
        'GNMTModel_gnmt_v2/loss': 8.8443756,
        'GNMTModel_gnmt_v2/predict_count': 11.0,
        'NoAttentionNoResidualUniEncoder/loss': 8.8440113,
        'NoAttentionNoResidualUniEncoder/predict_count': 11.0,
        'NoAttentionResidualBiEncoder/loss': 8.8291245,
        'NoAttentionResidualBiEncoder/predict_count': 11.0,
        'UniEncoderBottomAttentionArchitecture/loss': 8.844492,
        'UniEncoderBottomAttentionArchitecture/predict_count': 11.0,
        'UniEncoderStandardAttentionArchitecture/loss': 8.8517151,
        'UniEncoderStandardAttentionArchitecture/predict_count': 11.0
    }

    cls.actual_infer_values = {}
    cls.expected_infer_values = {
        'AttentionMechanismBahdanau/logits_sum': -0.026374687,
        'AttentionMechanismLuong/logits_sum': -0.026374735,
        'AttentionMechanismNormedBahdanau/logits_sum': -0.026376063,
        'AttentionMechanismScaledLuong/logits_sum': -0.026374735,
        'GNMTModel_gnmt/logits_sum': -1.10848486,
        'GNMTModel_gnmt_v2/logits_sum': -1.10950875,
        'NoAttentionNoResidualUniEncoder/logits_sum': -1.0808625,
        'NoAttentionResidualBiEncoder/logits_sum': -2.8147559,
        'UniEncoderBottomAttentionArchitecture/logits_sum': -0.97026241,
        'UniEncoderStandardAttentionArchitecture/logits_sum': -0.02665353
    }

    cls.actual_beam_sentences = {}
    cls.expected_beam_sentences = {
        'BeamSearchAttentionModel: batch 0 of beam 0': '',
        'BeamSearchAttentionModel: batch 0 of beam 1': 'sos a sos a',
        'BeamSearchAttentionModel: batch 1 of beam 0': '',
        'BeamSearchAttentionModel: batch 1 of beam 1': 'b',
        'BeamSearchBasicModel: batch 0 of beam 0': 'b b b b',
        'BeamSearchBasicModel: batch 0 of beam 1': 'b b b sos',
        'BeamSearchBasicModel: batch 0 of beam 2': 'b b b c',
        'BeamSearchBasicModel: batch 1 of beam 0': 'b b b b',
        'BeamSearchBasicModel: batch 1 of beam 1': 'a b b b',
        'BeamSearchBasicModel: batch 1 of beam 2': 'b b b sos',
        'BeamSearchGNMTModel: batch 0 of beam 0': '',
        'BeamSearchGNMTModel: batch 1 of beam 0': '',
    }

  @classmethod
  def tearDownClass(cls):
    print('ModelTest - actual_vars_values: ')
    pprint.pprint(cls.actual_vars_values)
    sys.stdout.flush()

    print('ModelTest - actual_train_values: ')
    pprint.pprint(cls.actual_train_values)
    sys.stdout.flush()

    print('ModelTest - actual_eval_values: ')
    pprint.pprint(cls.actual_eval_values)
    sys.stdout.flush()

    print('ModelTest - actual_infer_values: ')
    pprint.pprint(cls.actual_infer_values)
    sys.stdout.flush()

    print('ModelTest - actual_beam_sentences: ')
    pprint.pprint(cls.actual_beam_sentences)
    sys.stdout.flush()

  def assertAllClose(self, *args, **kwargs):
    kwargs['atol'] = 5e-2
    kwargs['rtol'] = 5e-2
    return super(ModelTest, self).assertAllClose(*args, **kwargs)

  def _assertModelVariableNames(self, expected_var_names, model_var_names,
                                name):

    print('{} variable names are: '.format(name), model_var_names)

    self.assertEqual(len(expected_var_names), len(model_var_names))
    self.assertEqual(sorted(expected_var_names), sorted(model_var_names))

  def _assertModelVariable(self, variable, sess, name):
    var_shape = tuple(variable.get_shape().as_list())
    var_res = sess.run(variable)
    var_weight_sum = np.sum(var_res)

    print('{} weight sum is: '.format(name), var_weight_sum)
    expected_sum = self.expected_vars_values[name + '/sum']
    expected_shape = self.expected_vars_values[name + '/shape']
    self.actual_vars_values[name + '/sum'] = var_weight_sum
    self.actual_vars_values[name + '/shape'] = var_shape

    self.assertEqual(expected_shape, var_shape)
    self.assertAllClose(expected_sum, var_weight_sum)

  def _assertTrainStepsLoss(self, m, sess, name, num_steps=1):
    for _ in range(num_steps):
      _, loss, _, _, _, _, _, _, _ = m.train(sess)

    print('{} {}-th step loss is: '.format(name, num_steps), loss)
    expected_loss = self.expected_train_values[name + '/loss']
    self.actual_train_values[name + '/loss'] = loss

    self.assertAllClose(expected_loss, loss)

  def _assertEvalLossAndPredictCount(self, m, sess, name):
    loss, predict_count, _ = m.eval(sess)

    print('{} eval loss is: '.format(name), loss)
    print('{} predict count is: '.format(name), predict_count)
    expected_loss = self.expected_eval_values[name + '/loss']
    expected_predict_count = self.expected_eval_values[name + '/predict_count']
    self.actual_eval_values[name + '/loss'] = loss
    self.actual_eval_values[name + '/predict_count'] = predict_count

    self.assertAllClose(expected_loss, loss)
    self.assertAllClose(expected_predict_count, predict_count)

  def _assertInferLogits(self, m, sess, name):
    results = m.infer(sess)
    logits_sum = np.sum(results[0])

    print('{} infer logits sum is: '.format(name), logits_sum)
    expected_logits_sum = self.expected_infer_values[name + '/logits_sum']
    self.actual_infer_values[name + '/logits_sum'] = logits_sum

    self.assertAllClose(expected_logits_sum, logits_sum)

  def _assertBeamSearchOutputs(self, m, sess, assert_top_k_sentence, name):
    nmt_outputs, _ = m.decode(sess)

    for i in range(assert_top_k_sentence):
      output_words = nmt_outputs[i]
      for j in range(output_words.shape[0]):
        sentence = nmt_utils.get_translation(
            output_words, j, tgt_eos='eos', subword_option='')
        sentence_key = ('%s: batch %d of beam %d' % (name, j, i))
        self.actual_beam_sentences[sentence_key] = sentence
        expected_sentence = self.expected_beam_sentences[sentence_key]
        self.assertEqual(expected_sentence, sentence)

  def _createTestTrainModel(self, m_creator, hparams, sess):
    train_mode = tf.contrib.learn.ModeKeys.TRAIN
    train_iterator, src_vocab_table, tgt_vocab_table = (
        common_test_utils.create_test_iterator(hparams, train_mode))
    train_m = m_creator(
        hparams,
        train_mode,
        train_iterator,
        src_vocab_table,
        tgt_vocab_table,
        scope='dynamic_seq2seq')
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(train_iterator.initializer)
    return train_m

  def _createTestEvalModel(self, m_creator, hparams, sess):
    eval_mode = tf.contrib.learn.ModeKeys.EVAL
    eval_iterator, src_vocab_table, tgt_vocab_table = (
        common_test_utils.create_test_iterator(hparams, eval_mode))
    eval_m = m_creator(
        hparams,
        eval_mode,
        eval_iterator,
        src_vocab_table,
        tgt_vocab_table,
        scope='dynamic_seq2seq')
    sess.run(tf.tables_initializer())
    sess.run(eval_iterator.initializer)
    return eval_m

  def _createTestInferModel(
      self, m_creator, hparams, sess, init_global_vars=False):
    infer_mode = tf.contrib.learn.ModeKeys.INFER
    (infer_iterator, src_vocab_table,
     tgt_vocab_table, reverse_tgt_vocab_table) = (
         common_test_utils.create_test_iterator(hparams, infer_mode))
    infer_m = m_creator(
        hparams,
        infer_mode,
        infer_iterator,
        src_vocab_table,
        tgt_vocab_table,
        reverse_tgt_vocab_table,
        scope='dynamic_seq2seq')
    if init_global_vars:
      sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(infer_iterator.initializer)
    return infer_m

  def _get_session_config(self):
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    return config

  ## Testing 3 encoders:
  # uni: no attention, no residual, 1 layers
  # bi: no attention, with residual, 4 layers
  def testNoAttentionNoResidualUniEncoder(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='uni',
        num_layers=1,
        attention='',
        attention_architecture='',
        use_residual=False,)

    workers, _ = tf.test.create_local_cluster(1, 0)
    worker = workers[0]

    # pylint: disable=line-too-long
    expected_var_names = [
        'dynamic_seq2seq/encoder/embedding_encoder:0',
        'dynamic_seq2seq/decoder/embedding_decoder:0',
        'dynamic_seq2seq/encoder/rnn/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/output_projection/kernel:0'
    ]
    # pylint: enable=line-too-long

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        train_m = self._createTestTrainModel(model.Model, hparams, sess)

        m_vars = tf.trainable_variables()
        self._assertModelVariableNames(expected_var_names,
                                       [v.name for v in m_vars],
                                       'NoAttentionNoResidualUniEncoder')

        with tf.variable_scope('dynamic_seq2seq', reuse=True):
          last_enc_weight = tf.get_variable(
              'encoder/rnn/basic_lstm_cell/kernel')
          last_dec_weight = tf.get_variable('decoder/basic_lstm_cell/kernel')
        self._assertTrainStepsLoss(train_m, sess,
                                   'NoAttentionNoResidualUniEncoder')
        self._assertModelVariable(
            last_enc_weight, sess,
            'NoAttentionNoResidualUniEncoder/last_enc_weight')
        self._assertModelVariable(
            last_dec_weight, sess,
            'NoAttentionNoResidualUniEncoder/last_dec_weight')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        eval_m = self._createTestEvalModel(model.Model, hparams, sess)
        self._assertEvalLossAndPredictCount(eval_m, sess,
                                            'NoAttentionNoResidualUniEncoder')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        infer_m = self._createTestInferModel(model.Model, hparams, sess)
        self._assertInferLogits(infer_m, sess,
                                'NoAttentionNoResidualUniEncoder')

  def testNoAttentionResidualBiEncoder(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='bi',
        num_layers=4,
        attention='',
        attention_architecture='',
        use_residual=True,)

    workers, _ = tf.test.create_local_cluster(1, 0)
    worker = workers[0]

    # pylint: disable=line-too-long
    expected_var_names = [
        'dynamic_seq2seq/encoder/embedding_encoder:0',
        'dynamic_seq2seq/decoder/embedding_decoder:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/fw/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/fw/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/fw/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/fw/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/bw/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/bw/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/bw/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/bw/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_2/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_2/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_3/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_3/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/output_projection/kernel:0'
    ]
    # pylint: enable=line-too-long

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        train_m = self._createTestTrainModel(model.Model, hparams, sess)

        m_vars = tf.trainable_variables()
        self._assertModelVariableNames(expected_var_names,
                                       [v.name for v in m_vars],
                                       'NoAttentionResidualBiEncoder')
        with tf.variable_scope('dynamic_seq2seq', reuse=True):
          last_enc_weight = tf.get_variable(
              'encoder/bidirectional_rnn/bw/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'
          )
          last_dec_weight = tf.get_variable(
              'decoder/multi_rnn_cell/cell_3/basic_lstm_cell/kernel')
        self._assertTrainStepsLoss(train_m, sess,
                                   'NoAttentionResidualBiEncoder')
        self._assertModelVariable(
            last_enc_weight, sess,
            'NoAttentionResidualBiEncoder/last_enc_weight')
        self._assertModelVariable(
            last_dec_weight, sess,
            'NoAttentionResidualBiEncoder/last_dec_weight')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        eval_m = self._createTestEvalModel(model.Model, hparams, sess)
        self._assertEvalLossAndPredictCount(eval_m, sess,
                                            'NoAttentionResidualBiEncoder')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        infer_m = self._createTestInferModel(model.Model, hparams, sess)
        self._assertInferLogits(infer_m, sess, 'NoAttentionResidualBiEncoder')

  ## Test attention mechanisms: luong, scaled_luong, bahdanau, normed_bahdanau
  def testAttentionMechanismLuong(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='uni',
        attention='luong',
        attention_architecture='standard',
        num_layers=2,
        use_residual=False,)

    workers, _ = tf.test.create_local_cluster(1, 0)
    worker = workers[0]

    # pylint: disable=line-too-long
    expected_var_names = [
        'dynamic_seq2seq/encoder/embedding_encoder:0',
        'dynamic_seq2seq/decoder/embedding_decoder:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/memory_layer/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/attention_layer/kernel:0',
        'dynamic_seq2seq/decoder/output_projection/kernel:0'
    ]
    # pylint: enable=line-too-long
    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        train_m = self._createTestTrainModel(attention_model.AttentionModel,
                                             hparams, sess)

        m_vars = tf.trainable_variables()
        self._assertModelVariableNames(
            expected_var_names, [v.name
                                 for v in m_vars], 'AttentionMechanismLuong')

        with tf.variable_scope('dynamic_seq2seq', reuse=True):
          # pylint: disable=line-too-long
          last_enc_weight = tf.get_variable(
              'encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel')
          last_dec_weight = tf.get_variable(
              'decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel')
          att_layer_weight = tf.get_variable(
              'decoder/attention/attention_layer/kernel')
          # pylint: enable=line-too-long
        self._assertTrainStepsLoss(train_m, sess, 'AttentionMechanismLuong')
        self._assertModelVariable(last_enc_weight, sess,
                                  'AttentionMechanismLuong/last_enc_weight')
        self._assertModelVariable(last_dec_weight, sess,
                                  'AttentionMechanismLuong/last_dec_weight')
        self._assertModelVariable(att_layer_weight, sess,
                                  'AttentionMechanismLuong/att_layer_weight')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        eval_m = self._createTestEvalModel(attention_model.AttentionModel,
                                           hparams, sess)
        self._assertEvalLossAndPredictCount(eval_m, sess,
                                            'AttentionMechanismLuong')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        infer_m = self._createTestInferModel(attention_model.AttentionModel,
                                             hparams, sess)
        self._assertInferLogits(infer_m, sess, 'AttentionMechanismLuong')

  def testAttentionMechanismScaledLuong(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='uni',
        attention='scaled_luong',
        attention_architecture='standard',
        num_layers=2,
        use_residual=False,)

    workers, _ = tf.test.create_local_cluster(1, 0)
    worker = workers[0]

    # pylint: disable=line-too-long
    expected_var_names = [
        'dynamic_seq2seq/encoder/embedding_encoder:0',
        'dynamic_seq2seq/decoder/embedding_decoder:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/memory_layer/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/luong_attention/attention_g:0',
        'dynamic_seq2seq/decoder/attention/attention_layer/kernel:0',
        'dynamic_seq2seq/decoder/output_projection/kernel:0'
    ]
    # pylint: enable=line-too-long
    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        train_m = self._createTestTrainModel(attention_model.AttentionModel,
                                             hparams, sess)

        m_vars = tf.trainable_variables()
        self._assertModelVariableNames(expected_var_names,
                                       [v.name for v in m_vars],
                                       'AttentionMechanismScaledLuong')

        with tf.variable_scope('dynamic_seq2seq', reuse=True):
          # pylint: disable=line-too-long
          last_enc_weight = tf.get_variable(
              'encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel')
          last_dec_weight = tf.get_variable(
              'decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel')
          att_layer_weight = tf.get_variable(
              'decoder/attention/attention_layer/kernel')
          # pylint: enable=line-too-long

        self._assertTrainStepsLoss(train_m, sess,
                                   'AttentionMechanismScaledLuong')
        self._assertModelVariable(
            last_enc_weight, sess,
            'AttentionMechanismScaledLuong/last_enc_weight')
        self._assertModelVariable(
            last_dec_weight, sess,
            'AttentionMechanismScaledLuong/last_dec_weight')
        self._assertModelVariable(
            att_layer_weight, sess,
            'AttentionMechanismScaledLuong/att_layer_weight')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        eval_m = self._createTestEvalModel(attention_model.AttentionModel,
                                           hparams, sess)
        self._assertEvalLossAndPredictCount(eval_m, sess,
                                            'AttentionMechanismScaledLuong')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        infer_m = self._createTestInferModel(attention_model.AttentionModel,
                                             hparams, sess)
        self._assertInferLogits(infer_m, sess, 'AttentionMechanismScaledLuong')

  def testAttentionMechanismBahdanau(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='uni',
        attention='bahdanau',
        attention_architecture='standard',
        num_layers=2,
        use_residual=False,)

    workers, _ = tf.test.create_local_cluster(1, 0)
    worker = workers[0]

    # pylint: disable=line-too-long
    expected_var_names = [
        'dynamic_seq2seq/encoder/embedding_encoder:0',
        'dynamic_seq2seq/decoder/embedding_decoder:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/memory_layer/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/bahdanau_attention/query_layer/kernel:0',
        'dynamic_seq2seq/decoder/attention/bahdanau_attention/attention_v:0',
        'dynamic_seq2seq/decoder/attention/attention_layer/kernel:0',
        'dynamic_seq2seq/decoder/output_projection/kernel:0'
    ]
    # pylint: enable=line-too-long
    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        train_m = self._createTestTrainModel(attention_model.AttentionModel,
                                             hparams, sess)

        m_vars = tf.trainable_variables()
        self._assertModelVariableNames(
            expected_var_names, [v.name
                                 for v in m_vars], 'AttentionMechanismBahdanau')

        with tf.variable_scope('dynamic_seq2seq', reuse=True):
          # pylint: disable=line-too-long
          last_enc_weight = tf.get_variable(
              'encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel')
          last_dec_weight = tf.get_variable(
              'decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel')
          att_layer_weight = tf.get_variable(
              'decoder/attention/attention_layer/kernel')
          # pylint: enable=line-too-long
        self._assertTrainStepsLoss(train_m, sess, 'AttentionMechanismBahdanau')
        self._assertModelVariable(last_enc_weight, sess,
                                  'AttentionMechanismBahdanau/last_enc_weight')
        self._assertModelVariable(last_dec_weight, sess,
                                  'AttentionMechanismBahdanau/last_dec_weight')
        self._assertModelVariable(att_layer_weight, sess,
                                  'AttentionMechanismBahdanau/att_layer_weight')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        eval_m = self._createTestEvalModel(attention_model.AttentionModel,
                                           hparams, sess)
        self._assertEvalLossAndPredictCount(eval_m, sess,
                                            'AttentionMechanismBahdanau')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        infer_m = self._createTestInferModel(attention_model.AttentionModel,
                                             hparams, sess)
        self._assertInferLogits(infer_m, sess, 'AttentionMechanismBahdanau')

  def testAttentionMechanismNormedBahdanau(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='uni',
        attention='normed_bahdanau',
        attention_architecture='standard',
        num_layers=2,
        use_residual=False,)

    workers, _ = tf.test.create_local_cluster(1, 0)
    worker = workers[0]

    # pylint: disable=line-too-long
    expected_var_names = [
        'dynamic_seq2seq/encoder/embedding_encoder:0',
        'dynamic_seq2seq/decoder/embedding_decoder:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/memory_layer/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/bahdanau_attention/query_layer/kernel:0',
        'dynamic_seq2seq/decoder/attention/bahdanau_attention/attention_v:0',
        'dynamic_seq2seq/decoder/attention/bahdanau_attention/attention_g:0',
        'dynamic_seq2seq/decoder/attention/bahdanau_attention/attention_b:0',
        'dynamic_seq2seq/decoder/attention/attention_layer/kernel:0',
        'dynamic_seq2seq/decoder/output_projection/kernel:0'
    ]
    # pylint: enable=line-too-long

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        train_m = self._createTestTrainModel(attention_model.AttentionModel,
                                             hparams, sess)

        m_vars = tf.trainable_variables()
        self._assertModelVariableNames(expected_var_names,
                                       [v.name for v in m_vars],
                                       'AttentionMechanismNormedBahdanau')

        with tf.variable_scope('dynamic_seq2seq', reuse=True):
          # pylint: disable=line-too-long
          last_enc_weight = tf.get_variable(
              'encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel')
          last_dec_weight = tf.get_variable(
              'decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel')
          att_layer_weight = tf.get_variable(
              'decoder/attention/attention_layer/kernel')
          # pylint: enable=line-too-long
        self._assertTrainStepsLoss(train_m, sess,
                                   'AttentionMechanismNormedBahdanau')
        self._assertModelVariable(
            last_enc_weight, sess,
            'AttentionMechanismNormedBahdanau/last_enc_weight')
        self._assertModelVariable(
            last_dec_weight, sess,
            'AttentionMechanismNormedBahdanau/last_dec_weight')
        self._assertModelVariable(
            att_layer_weight, sess,
            'AttentionMechanismNormedBahdanau/att_layer_weight')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        eval_m = self._createTestEvalModel(attention_model.AttentionModel,
                                           hparams, sess)
        self._assertEvalLossAndPredictCount(eval_m, sess,
                                            'AttentionMechanismNormedBahdanau')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        infer_m = self._createTestInferModel(attention_model.AttentionModel,
                                             hparams, sess)
        self._assertInferLogits(infer_m, sess,
                                'AttentionMechanismNormedBahdanau')

  ## Test encoder vs. attention (all use residual):
  # uni encoder, standard attention
  def testUniEncoderStandardAttentionArchitecture(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='uni',
        num_layers=4,
        attention='scaled_luong',
        attention_architecture='standard',)

    workers, _ = tf.test.create_local_cluster(1, 0)
    worker = workers[0]

    # pylint: disable=line-too-long
    expected_var_names = [
        'dynamic_seq2seq/encoder/embedding_encoder:0',
        'dynamic_seq2seq/decoder/embedding_decoder:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_2/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_3/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_3/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/memory_layer/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_2/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_2/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_3/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_3/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/attention/luong_attention/attention_g:0',
        'dynamic_seq2seq/decoder/attention/attention_layer/kernel:0',
        'dynamic_seq2seq/decoder/output_projection/kernel:0'
    ]
    # pylint: enable=line-too-long

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        train_m = self._createTestTrainModel(attention_model.AttentionModel,
                                             hparams, sess)

        m_vars = tf.trainable_variables()
        self._assertModelVariableNames(expected_var_names, [
            v.name for v in m_vars
        ], 'UniEncoderStandardAttentionArchitecture')
        with tf.variable_scope('dynamic_seq2seq', reuse=True):
          last_enc_weight = tf.get_variable(
              'encoder/rnn/multi_rnn_cell/cell_3/basic_lstm_cell/kernel')
          last_dec_weight = tf.get_variable(
              'decoder/attention/multi_rnn_cell/cell_3/basic_lstm_cell/kernel')
          mem_layer_weight = tf.get_variable('decoder/memory_layer/kernel')
        self._assertTrainStepsLoss(train_m, sess,
                                   'UniEncoderStandardAttentionArchitecture')
        self._assertModelVariable(
            last_enc_weight, sess,
            'UniEncoderStandardAttentionArchitecture/last_enc_weight')
        self._assertModelVariable(
            last_dec_weight, sess,
            'UniEncoderStandardAttentionArchitecture/last_dec_weight')
        self._assertModelVariable(
            mem_layer_weight, sess,
            'UniEncoderStandardAttentionArchitecture/mem_layer_weight')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        eval_m = self._createTestEvalModel(attention_model.AttentionModel,
                                           hparams, sess)
        self._assertEvalLossAndPredictCount(
            eval_m, sess, 'UniEncoderStandardAttentionArchitecture')

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        infer_m = self._createTestInferModel(attention_model.AttentionModel,
                                             hparams, sess)
        self._assertInferLogits(infer_m, sess,
                                'UniEncoderStandardAttentionArchitecture')

  # Test gnmt model.
  def _testGNMTModel(self, architecture):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='gnmt',
        num_layers=4,
        attention='scaled_luong',
        attention_architecture=architecture)

    workers, _ = tf.test.create_local_cluster(1, 0)
    worker = workers[0]

    # pylint: disable=line-too-long
    expected_var_names = [
        'dynamic_seq2seq/encoder/embedding_encoder:0',
        'dynamic_seq2seq/decoder/embedding_decoder:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/fw/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/fw/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/bw/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/bidirectional_rnn/bw/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_2/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/memory_layer/kernel:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_0_attention/attention/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_0_attention/attention/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_0_attention/attention/luong_attention/attention_g:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_2/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_2/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_3/basic_lstm_cell/kernel:0',
        'dynamic_seq2seq/decoder/multi_rnn_cell/cell_3/basic_lstm_cell/bias:0',
        'dynamic_seq2seq/decoder/output_projection/kernel:0'
    ]
    # pylint: enable=line-too-long

    test_prefix = 'GNMTModel_%s' % architecture
    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        train_m = self._createTestTrainModel(gnmt_model.GNMTModel, hparams,
                                             sess)

        m_vars = tf.trainable_variables()
        self._assertModelVariableNames(expected_var_names,
                                       [v.name for v in m_vars], test_prefix)
        with tf.variable_scope('dynamic_seq2seq', reuse=True):
          last_enc_weight = tf.get_variable(
              'encoder/rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel')
          last_dec_weight = tf.get_variable(
              'decoder/multi_rnn_cell/cell_3/basic_lstm_cell/kernel')
          mem_layer_weight = tf.get_variable('decoder/memory_layer/kernel')
        self._assertTrainStepsLoss(train_m, sess, test_prefix)

        self._assertModelVariable(last_enc_weight, sess,
                                  '%s/last_enc_weight' % test_prefix)
        self._assertModelVariable(last_dec_weight, sess,
                                  '%s/last_dec_weight' % test_prefix)
        self._assertModelVariable(mem_layer_weight, sess,
                                  '%s/mem_layer_weight' % test_prefix)

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        eval_m = self._createTestEvalModel(gnmt_model.GNMTModel, hparams, sess)
        self._assertEvalLossAndPredictCount(eval_m, sess, test_prefix)

    with tf.Graph().as_default():
      with tf.Session(worker.target, config=self._get_session_config()) as sess:
        infer_m = self._createTestInferModel(gnmt_model.GNMTModel, hparams,
                                             sess)
        self._assertInferLogits(infer_m, sess, test_prefix)

  def testGNMTModel(self):
    self._testGNMTModel('gnmt')

  def testGNMTModelV2(self):
    self._testGNMTModel('gnmt_v2')

  # Test beam search.
  def testBeamSearchBasicModel(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='uni',
        num_layers=1,
        attention='',
        attention_architecture='',
        use_residual=False,)
    hparams.beam_width = 3
    hparams.tgt_max_len_infer = 4
    assert_top_k_sentence = 3

    with self.test_session() as sess:
      infer_m = self._createTestInferModel(
          model.Model, hparams, sess, True)
      self._assertBeamSearchOutputs(
          infer_m, sess, assert_top_k_sentence, 'BeamSearchBasicModel')

  def testBeamSearchAttentionModel(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='uni',
        attention='scaled_luong',
        attention_architecture='standard',
        num_layers=2,
        use_residual=False,)
    hparams.beam_width = 3
    hparams.tgt_max_len_infer = 4
    assert_top_k_sentence = 2

    with self.test_session() as sess:
      infer_m = self._createTestInferModel(
          attention_model.AttentionModel, hparams, sess, True)
      self._assertBeamSearchOutputs(
          infer_m, sess, assert_top_k_sentence, 'BeamSearchAttentionModel')

  def testBeamSearchGNMTModel(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='gnmt',
        num_layers=4,
        attention='scaled_luong',
        attention_architecture='gnmt')
    hparams.beam_width = 3
    hparams.tgt_max_len_infer = 4
    assert_top_k_sentence = 1

    with self.test_session() as sess:
      infer_m = self._createTestInferModel(
          gnmt_model.GNMTModel, hparams, sess, True)
      self._assertBeamSearchOutputs(
          infer_m, sess, assert_top_k_sentence, 'BeamSearchGNMTModel')

  def testInitializerGlorotNormal(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='uni',
        num_layers=1,
        attention='',
        attention_architecture='',
        use_residual=False,
        init_op='glorot_normal')

    with self.test_session() as sess:
      train_m = self._createTestTrainModel(model.Model, hparams, sess)
      self._assertTrainStepsLoss(train_m, sess,
                                 'InitializerGlorotNormal')

  def testInitializerGlorotUniform(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type='uni',
        num_layers=1,
        attention='',
        attention_architecture='',
        use_residual=False,
        init_op='glorot_uniform')

    with self.test_session() as sess:
      train_m = self._createTestTrainModel(model.Model, hparams, sess)
      self._assertTrainStepsLoss(train_m, sess,
                                 'InitializerGlorotUniform')

if __name__ == '__main__':
  tf.test.main()
