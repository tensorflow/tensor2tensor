# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
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

"""Problems for Seq2Edits (see models/research/transformer_seq2edits.py)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


@modalities.is_pointwise
def pointer_top(body_output, targets, model_hparams, vocab_size):
  """Like identity_top() with is_pointwise annotation."""
  del targets, model_hparams, vocab_size  # unused arg
  return body_output


def pointer_bottom(x, model_hparams, vocab_size):
  """Like identity_bottom() without converting to float."""
  del model_hparams, vocab_size  # unused arg
  return x


@registry.register_problem
class Seq2editsGec(text_problems.Text2TextProblem):
  """Seq2Edits for grammatical error correction."""

  def dataset_filename(self):
    return "edit_ops_gec"

  @property
  def vocab_file(self):
    return "vocab.subwords"

  @property
  def vocab_filename(self):
    return "vocab.subwords"

  @property
  def error_tag_vocab_file(self):
    return "vocab.error_tags"

  def feature_encoders(self, data_dir):
    subword_encoder = text_encoder.SubwordTextEncoder(
        os.path.join(data_dir, self.vocab_file))
    error_tag_encoder = text_encoder.TokenTextEncoder(
        os.path.join(data_dir, self.error_tag_vocab_file))
    return {
        "inputs": subword_encoder,
        "targets": subword_encoder,
        "targets_error_tag": error_tag_encoder
    }

  def hparams(self, defaults, model_hparams):
    super(Seq2editsGec, self).hparams(defaults, model_hparams)

    for pointer_feat in ["targets_start_token", "targets_end_token"]:
      defaults.modality[pointer_feat] = modalities.ModalityType.IDENTITY
      defaults.vocab_size[pointer_feat] = None
      model_hparams.bottom[pointer_feat] = pointer_bottom
      model_hparams.top[pointer_feat] = pointer_top
    # Whether to use tags.
    if "use_error_tags" not in model_hparams:
      model_hparams.add_hparam("use_error_tags", True)
    # If true, span and tag prediction is in the middle of the decoder layer
    # stack. Otherwise, they are at the end of the decoder layer stack.
    if "middle_prediction" not in model_hparams:
      model_hparams.add_hparam("middle_prediction", True)
    # If middle_prediction=True, divide num_decoder_layers by this to get the
    # number of layers before and after the middle prediction.
    if "middle_prediction_layer_factor" not in model_hparams:
      model_hparams.add_hparam("middle_prediction_layer_factor", 2)
    # Whether to predict the targets_start_token feature. If this is false, use
    # the previous end token as implicit start token.
    if "use_start_token" not in model_hparams:
      model_hparams.add_hparam("use_start_token", False)
    # Whether to feed back targets_end_token to the next time step. If false,
    # only feed back targets_start_token.
    if "feedback_end_token" not in model_hparams:
      model_hparams.add_hparam("feedback_end_token", False)
    # Number of feedforward layers between prediction layers in the cascade.
    if "ffn_in_prediction_cascade" not in model_hparams:
      model_hparams.add_hparam("ffn_in_prediction_cascade", 1)
    # Embedding size for error tags.
    if "error_tag_embed_size" not in model_hparams:
      model_hparams.add_hparam("error_tag_embed_size", 6)
    if model_hparams.use_error_tags:
      defaults.modality["targets_error_tag"] = modalities.ModalityType.SYMBOL
      error_tag_vocab_size = self._encoders["targets_error_tag"].vocab_size
      defaults.vocab_size["targets_error_tag"] = error_tag_vocab_size
      model_hparams.top["targets_error_tag"] = pointer_top

  def example_reading_spec(self):
    data_fields, _ = super(Seq2editsGec, self).example_reading_spec()
    data_fields["targets_start_token"] = tf.VarLenFeature(tf.int64)
    data_fields["targets_end_token"] = tf.VarLenFeature(tf.int64)
    data_fields["targets_error_tag"] = tf.VarLenFeature(tf.int64)
    return data_fields, None


@registry.register_problem
class Seq2editsGecPacked256(Seq2editsGec):
  """Packed version for TPU."""

  def dataset_filename(self):
    return "edit_ops_gec_packed256"

  @property
  def packed_length(self):
    return 256

  @property
  def max_segment_length(self):
    return 256


@registry.register_problem
class Seq2editsGecNoTags(Seq2editsGec):
  """Seq2Edits for grammatical error correction without tags."""

  def dataset_filename(self):
    return "edit_ops_gec"

  def hparams(self, defaults, model_hparams):
    super(Seq2editsGecNoTags, self).hparams(defaults, model_hparams)
    model_hparams.use_error_tags = False


@registry.register_problem
class Seq2editsGecNoTagsPacked256(Seq2editsGecPacked256):
  """Packed version for TPU."""

  def dataset_filename(self):
    return "edit_ops_gec_packed256"

  def hparams(self, defaults, model_hparams):
    super(Seq2editsGecNoTagsPacked256, self).hparams(defaults, model_hparams)
    model_hparams.use_error_tags = False


@registry.register_problem
class Seq2editsGecDeep(Seq2editsGec):
  """Seq2Edits for grammatical error correction with deeper decoder."""

  def hparams(self, defaults, model_hparams):
    super(Seq2editsGecDeep, self).hparams(defaults, model_hparams)
    model_hparams.middle_prediction_layer_factor = 1.5


@registry.register_problem
class Seq2editsGecDeepPacked256(Seq2editsGecPacked256):
  """Packed version for TPU."""

  def hparams(self, defaults, model_hparams):
    super(Seq2editsGecDeepPacked256, self).hparams(defaults, model_hparams)
    model_hparams.middle_prediction_layer_factor = 1.5


@registry.register_problem
class Seq2editsGecDeepNoTags(Seq2editsGec):
  """Deep Seq2Edits model for grammatical error correction without tags."""

  def hparams(self, defaults, model_hparams):
    super(Seq2editsGecDeepNoTags, self).hparams(defaults, model_hparams)
    model_hparams.middle_prediction_layer_factor = 1.5
    model_hparams.use_error_tags = False


@registry.register_problem
class Seq2editsGecDeepNoTagsPacked256(Seq2editsGecPacked256):
  """Packed version for TPU."""

  def hparams(self, defaults, model_hparams):
    super(Seq2editsGecDeepNoTagsPacked256, self).hparams(
        defaults, model_hparams)
    model_hparams.middle_prediction_layer_factor = 1.5
    model_hparams.use_error_tags = False


@registry.register_problem
class Seq2editsTextnorm(Seq2editsGec):
  """Seq2Edits for text normalization."""

  def dataset_filename(self):
    return "edit_ops_textnorm"

  @property
  def source_vocab_file(self):
    return "vocab.source"

  @property
  def target_vocab_file(self):
    return "vocab.target"

  @property
  def error_tag_vocab_file(self):
    return "vocab.error_tags"

  def feature_encoders(self, data_dir):
    source_encoder = text_encoder.TokenTextEncoder(
        os.path.join(data_dir, self.source_vocab_file))
    target_encoder = text_encoder.TokenTextEncoder(
        os.path.join(data_dir, self.target_vocab_file))
    error_tag_encoder = text_encoder.TokenTextEncoder(
        os.path.join(data_dir, self.error_tag_vocab_file))
    return {
        "inputs": source_encoder,
        "targets": target_encoder,
        "targets_error_tag": error_tag_encoder
    }


@registry.register_problem
class Seq2editsTextnormPacked256(Seq2editsTextnorm):
  """Packed version for TPU."""

  def dataset_filename(self):
    return "edit_ops_textnorm_packed256"

  @property
  def packed_length(self):
    return 256

  @property
  def max_segment_length(self):
    return 256


@registry.register_problem
class Seq2editsTextnormNoTags(Seq2editsTextnorm):
  """Seq2Edits for text normalization without tags."""

  def hparams(self, defaults, model_hparams):
    super(Seq2editsTextnormNoTags, self).hparams(defaults, model_hparams)
    model_hparams.use_error_tags = False


@registry.register_problem
class Seq2editsTextnormNoTagsPacked256(Seq2editsTextnormPacked256):
  """Packed version for TPU."""

  def hparams(self, defaults, model_hparams):
    super(Seq2editsTextnormNoTagsPacked256, self).hparams(
        defaults, model_hparams)
    model_hparams.use_error_tags = False
