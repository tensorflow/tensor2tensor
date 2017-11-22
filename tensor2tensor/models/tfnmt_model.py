# coding=utf-8
"""TF-NMT models for T2T.

This class contains an adaptor model implementation for using models from the
TensorFlow NMT tutorial

https://github.com/tensorflow/nmt
"""


from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_hparams
from tensor2tensor.models.tfnmt.nmt import alternating_model
from tensor2tensor.models.tfnmt.nmt import gnmt_model
from tensor2tensor.models.tfnmt.nmt import model as nmt_model
from tensor2tensor.models.tfnmt.nmt.utils import iterator_utils
from tensor2tensor.data_generators import text_encoder

import tensorflow as tf


@registry.register_model
class TFNmt(t2t_model.T2TModel):
  """Adaptor class for TF-NMT models."""

  def model_fn_body(self, features):
    hparams = self._hparams
    inputs, inputs_length = get_feature_with_length(features, "inputs")
    targets, targets_length = get_feature_with_length(features, "targets")
    if hparams.mode == tf.contrib.learn.ModeKeys.INFER:
      targets_length = targets_length + 1
    # inputs_length of 0 breaks things
    inputs_length = tf.maximum(inputs_length, 1)
    tfnmt_model = get_tfnmt_model(
        hparams, inputs, inputs_length, targets, targets_length)
    decoder_output = tfnmt_model.logits
    return tf.expand_dims(decoder_output, axis=2)


def get_feature_with_length(features, name):
  """Reads out embeddings and sequence lengths for a symbol modality 
  from the features.

  Args:
    features (dict): Dictionary with features.
    name (string): Feature to extract (will read features[name] and
                   features[name_raw])

  Returns:
    Pair of (embed, length) tensors, where `embed` is a (batch_size,
    max_len, embed_size) float32 tensor with embeddings, and `length`
    is a (batch_size,) int32 tensor with sequence lengths.
  """
  # features[name] shape: (batch_size, max_len, 1, embed_size)
  embed = common_layers.flatten4d3d(features[name])
  # embed shape: (batch_size, max_len, embed_size)
  raw = tf.squeeze(features["%s_raw" % name], axis=[2, 3])
  not_padding = tf.not_equal(raw, text_encoder.PAD_ID)
  not_padding_with_guardian = tf.pad(not_padding, [[0, 0], [0, 1]])
  indices = tf.where(tf.logical_not(not_padding_with_guardian))
  length = tf.segment_min(indices[:, 1], indices[:, 0])
  return embed, tf.cast(length, tf.int32)


def get_tfnmt_model(hparams, inputs, inputs_length, targets, targets_length):
  """Adapted from nmt.train.train()."""
  if not hparams.attention:
    model_class = nmt_model.Model
  elif hparams.attention_architecture == "standard":
    model_class = alternating_model.AlternatingEncoderModel
  elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
    model_class = gnmt_model.GNMTModel
  else:
    raise ValueError("Unknown model architecture")
  tfnmt_model = model_class(
      convert_to_tfnmt_hparams(hparams),
      iterator=get_fake_iterator(
          inputs, inputs_length, targets, targets_length),
      mode=tf.contrib.learn.ModeKeys.EVAL,  # We use eval graph for training
      source_vocab_table=FakeVocabTable(),
      target_vocab_table=FakeVocabTable())
  return tfnmt_model


class FakeVocabTable(object):
  """A null-object vocab table implementation."""
  def lookup(self, unused_arg):
    return 99999999


def get_fake_iterator(inputs, inputs_length, targets, targets_length):
  return iterator_utils.BatchedInput(
      initializer=None,
      source=inputs,
      target_input=common_layers.shift_right_3d(targets),
      target_output=None, # Loss is computed in T2T
      source_sequence_length=inputs_length,
      target_sequence_length=targets_length)



# The following hparams are taken from the nmt.standard_hparams directory in
# the TF NMT tutorial.

def tfnmt_base():
  """TF-NMT base configuration.

  Note: the SGD learning rate schedule is not replicated exactly. TF-NMT uses
  `learning_rate` until `start_decay_steps`, and then multiplies the learning
  rate with `decay_factor` every `decay_steps` steps.

  T2T uses the inverse decay rate until `learning_rate_warmup_steps` and then
  applies the `noam` decay scheme.

  Following fields are not covered by this as T2T defines them somewhere else.
    "num_train_steps": 12000,
    "steps_per_external_eval": null,
    "steps_per_stats": 100,
  """
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 4096  # Roughly equivalent to TF-NMT's batch_size=128
  hparams.dropout = 0.2
  hparams.learning_rate = 1.0
  hparams.clip_grad_norm = 5.0  # Called max_gradient_norm in TF-NMT
  hparams.optimizer = "SGD"  # sgd in TF-NMT
  hparams.learning_rate_decay_scheme = "noam" # See docstring
  hparams.max_input_seq_length = 50  # Called max_src_len* in TF-NMT
  hparams.max_target_seq_length = 50  # Called max_trg_len* in TF-NMT
  hparams.initializer = "uniform"
  hparams.initializer_gain = 1.0
  hparams.add_hparam("attention", "normed_bahdanau")
  hparams.add_hparam("attention_architecture", "standard")
  hparams.add_hparam("encoder_type", "bi")
  hparams.add_hparam("forget_bias", 1.0)
  hparams.add_hparam("unit_type", "lstm")
  hparams.add_hparam("residual", False)
  hparams.add_hparam("pass_hidden_state", True)
  hparams.add_hparam("output_attention", False)
  hparams.add_hparam("init_op", "uniform")
  return hparams


def tfnmt_default():
  """Inspired by the stacked architecture in the WMT17 UEdin submission.
  
  Differs from the evaluation system as follows:
    - No backtranslation
    - LSTM instead of GRU
  """
  hparams = tfnmt_base()
  hparams.attention = "normed_bahdanau"
  hparams.encoder_type = "bi"
  hparams.num_hidden_layers = 4
  hparams.hidden_size = 1024
  hparams.residual = True
  hparams.unit_type = "layer_norm_lstm"
  hparams.learning_rate_warmup_steps = 4000
  hparams.learning_rate = 0.1
  hparams.optimizer = "Adam"
  hparams.optimizer_adam_epsilon = 1e-7
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  return hparams


def convert_to_tfnmt_hparams(hparams):
  """Add hyper parameters required by TF-NMT but which are not directly
  accessible via T2T. This method extends the T2T hparams for using 
  them in the TF-NMT subpackage.
  """
  try:
    hparams.add_hparam("num_layers", hparams.num_hidden_layers)
  except ValueError:
    # A value error occurs when hparams.num_layers already exists, for example
    #  when using multiple GPUs. In this case we assume that hparams is
    # already converted.
    return hparams
  hparams.add_hparam("src_vocab_size", None)  # Not used
  hparams.add_hparam("tgt_vocab_size", None)  # Not used 
  hparams.add_hparam("num_gpus", 1)  # Not used
  hparams.add_hparam("time_major", False)  # True in TF-NMT 
  hparams.add_hparam("init_weight", 0.1 * hparams.initializer_gain)
  hparams.add_hparam("random_seed", None) 
  hparams.add_hparam("num_units", hparams.hidden_size) 
  hparams.add_hparam("sos", "foobar")  # Not used 
  hparams.add_hparam("eos", "foobar")  # Not used 
  hparams.add_hparam("tgt_max_len_infer", None)  # Not used 
  hparams.add_hparam("beam_width", 1)  # Not used 
  # See nmt.nmt.extend_hparams()
  # Sanity checks
  if hparams.encoder_type == "bi" and hparams.num_layers % 2 != 0:
    raise ValueError("For bi, num_layers %d should be even" %
                     hparams.num_layers)
  if (hparams.attention_architecture in ["gnmt"] and
      hparams.num_layers < 2):
    raise ValueError("For gnmt attention architecture, "
                     "num_layers %d should be >= 2" % hparams.num_layers)
  # Set num_residual_layers
  if hparams.residual and hparams.num_layers > 1:
    if hparams.encoder_type == "gnmt":
      # The first unidirectional layer (after the bi-directional layer) in
      # the GNMT encoder can't have residual connection since the input is
      # the concatenation of fw_cell and bw_cell's outputs.
      num_residual_layers = hparams.num_layers - 2
    else:
      # Last layer cannot have residual connections since the decoder
      # expects num_unit dimensional input
      num_residual_layers = hparams.num_layers - 1
  else:
    num_residual_layers = 0
  hparams.add_hparam("num_residual_layers", num_residual_layers)
  return hparams


@registry.register_hparams
def tfnmt_iwslt15():
  """TF-NMT iwslt15 configuration.
  
  Following fields differ in the original setup:
    "decay_factor": 0.5,
    "decay_steps": 1000,
  """
  hparams = tfnmt_base()
  hparams.attention = "scaled_luong"
  hparams.num_hidden_layers = 2  # Called num_layers in TF-NMT
  hparams.hidden_size = 512  # Called num_units in TF-NMT
  hparams.learning_rate_warmup_steps = 8000  # start_decay_step in TF-NMT
  return hparams


@registry.register_hparams
def tfnmt_wmt16():
  """TF-NMT wmt16 configuration.
  
  Following fields differ in the original setup:
    "decay_factor": 0.5,
    "decay_steps": 17000,
  """
  hparams = tfnmt_base()
  hparams.attention = "normed_bahdanau"
  hparams.num_hidden_layers = 4  # Called num_layers in TF-NMT
  hparams.hidden_size = 1024  # Called num_units in TF-NMT
  hparams.learning_rate_warmup_steps = 170000  # start_decay_step in TF-NMT
  return hparams


@registry.register_hparams
def tfnmt_wmt16_gnmt_4_layer():
  """TF-NMT wmt16_gnmt_4_layer configuration.
  
  Following fields differ in the original setup:
    "decay_factor": 0.5,
    "decay_steps": 17000,
  """
  hparams = tfnmt_base()
  hparams.attention = "normed_bahdanau"
  hparams.attention_architecture = "gnmt_v2"
  hparams.encoder_type = "gnmt"
  hparams.num_hidden_layers = 4  # Called num_layers in TF-NMT
  hparams.hidden_size = 1024  # Called num_units in TF-NMT
  hparams.residual = True
  hparams.learning_rate_warmup_steps = 170000  # start_decay_step in TF-NMT
  return hparams


@registry.register_hparams
def tfnmt_wmt16_gnmt_8_layer():
  """GNMT wmt16_gnmt_8_layer configuration.
  
  Following fields differ in the original setup:
    "decay_factor": 0.5,
    "decay_steps": 17000,
  """
  hparams = tfnmt_base()
  hparams.attention = "normed_bahdanau"
  hparams.attention_architecture = "gnmt_v2"
  hparams.encoder_type = "gnmt"
  hparams.num_hidden_layers = 8  # Called num_layers in TF-NMT
  hparams.hidden_size = 1024  # Called num_units in TF-NMT
  hparams.residual = True
  hparams.learning_rate_warmup_steps = 170000  # start_decay_step in TF-NMT
  return hparams


@registry.register_hparams
def tfnmt_wmt16_gnmt_8_layer_adam():
  """GNMT wmt16_gnmt_8_layer configuration with Adam."""
  hparams = tfnmt_wmt16_gnmt_8_layer()
  hparams.attention = "normed_bahdanau"
  hparams.attention_architecture = "gnmt_v2"
  hparams.encoder_type = "gnmt"
  hparams.num_hidden_layers = 8  # Called num_layers in TF-NMT
  hparams.hidden_size = 1024  # Called num_units in TF-NMT
  hparams.residual = True
  hparams.learning_rate_warmup_steps = 4000
  hparams.learning_rate = 0.1
  hparams.optimizer = "Adam"
  hparams.optimizer_adam_epsilon = 1e-7
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  return hparams


@registry.register_hparams
def tfnmt_wmt16_gnmt_8_layer_layer_norm():
  """GNMT wmt16_gnmt_8_layer configuration with layer normalization."""
  hparams = tfnmt_wmt16_gnmt_8_layer()
  hparams.unit_type = "layer_norm_lstm"
  return hparams


@registry.register_hparams
def tfnmt_wmt16_gnmt_8_layer_adam_layer_norm():
  """GNMT wmt16_gnmt_8_layer configuration with Adam and layer norm."""
  hparams = tfnmt_wmt16_gnmt_8_layer_adam()
  hparams.unit_type = "layer_norm_lstm"
  return hparams


@registry.register_hparams
def tfnmt_12gb_gpu():
  """Inspired by WMT17 UEdin submission, but with a different training
  setup.
  """
  hparams = tfnmt_default()
  hparams.learning_rate_warmup_steps = 6000
  hparams.label_smoothing = 0.1
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def tfnmt_12gb_gpu_alternating():
  """tfnmt_12gb_gpu with alternating encoder."""
  hparams = tfnmt_12gb_gpu()
  hparams.residual = True
  hparams.encoder_type = "alternating"
  hparams.batch_size = 4096
  return hparams

