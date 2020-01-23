# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Neural Assistant."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf


@registry.register_model
class NeuralAssistant(transformer.Transformer):
  """Attention net.  See file docstring."""

  def __init__(self, *args, **kwargs):
    super(NeuralAssistant, self).__init__(*args, **kwargs)
    self.attention_weights = dict()  # For visualizing attention heads.

    # Loss scheduling.
    hparams = self._hparams
    self.triple_num = hparams.train_triple_num

  def model_fn(self, features):
    with tf.variable_scope(tf.get_variable_scope(), use_resource=True) as vs:
      self._add_variable_scope("model_fn", vs)
      transformed_features = self.bottom(features)

      if self.hparams.activation_dtype == "bfloat16":
        for k, v in sorted(six.iteritems(transformed_features)):
          if v.dtype == tf.float32:
            transformed_features[k] = tf.cast(v, tf.bfloat16)

      with tf.variable_scope("body") as body_vs:
        self._add_variable_scope("body", body_vs)
        body_out = self.body(transformed_features)
      output, losses = self._normalize_body_output(body_out)

      if "training" in losses:
        tf.logging.info(
            "Skipping T2TModel top and loss because training loss returned from body"
        )
        logits = output
      else:
        tf.logging.warn("The loss will be computed in model_fn now.")
        logits = self.top(output, features)
        losses["training"] = 0.0
        cur_kb_loss = losses["kb_loss"]
        cur_knowledge_training_loss = losses["transe_loss"]
        cur_kb_loss_weight = self._hparams.kb_loss_weight
        kb_train_weight = self._hparams.kb_train_weight
        cur_lm_loss_weight = 1.0 - cur_kb_loss_weight
        # Finalize loss
        if (self._hparams.mode != tf.estimator.ModeKeys.PREDICT and
            self._hparams.mode != "attack"):
          lm_loss_num, lm_loss_denom = self.loss(logits, features)
          total_loss = (kb_train_weight) * cur_knowledge_training_loss + (
              1 - kb_train_weight) * (
                  cur_kb_loss * cur_kb_loss_weight +
                  (lm_loss_num / lm_loss_denom) * cur_lm_loss_weight)
          tf.summary.scalar("kb_loss", cur_kb_loss)
          tf.summary.scalar("transe_loss", cur_knowledge_training_loss)
          tf.summary.scalar("lm_loss", (lm_loss_num / lm_loss_denom))
          tf.summary.scalar("cur_kb_loss_weight",
                            tf.reshape(cur_kb_loss_weight, []))
          tf.logging.info("Loss computed " + str(total_loss))
          losses = {"training": total_loss}

      return logits, losses

  def encode_knowledge_bottom(self, features):
    tf.logging.info("Encoding knowledge " + str(self.triple_num))
    # Make sure this is embeddings for triples
    # <tf.float32>[batch_size, triple_num*max_triple_length, 1, emb_dim]
    fact_embedding = features["encoded_triples"]
    # [batch_size, triple_num*max_triple_length, emb_dim]
    fact_embedding = tf.squeeze(fact_embedding, 2)

    kb_shape = common_layers.shape_list(fact_embedding)
    batch_size = kb_shape[0]
    embed_dim = kb_shape[2]
    # <tf.float32>[batch_size*triple_num, max_triple_length, emb_dim]
    re_fact_embedding = tf.reshape(
        fact_embedding, [batch_size * self.triple_num, -1, embed_dim],
        name="reshape_fact_embedding")

    # <tf.int64>[batch_size, triple_num]
    input_fact_lengths = features["triple_lens"]
    # Stack the fact lengths.
    # <tf.int64>[batch_size*max_triple_num]
    re_fact_lengths = tf.reshape(
        input_fact_lengths, [batch_size * self.triple_num, 1],
        name="reshape_fact_lengths")

    return re_fact_embedding, re_fact_lengths

  def compute_knowledge_selection_and_loss(self, features, encoder_output,
                                           fact_embedding, fact_lengths, margin,
                                           num_negative_samples):
    """Compute knowledge selection and loss.

    Args:
      features: features.
      encoder_output: <tf.float32>[batch_size, input_length, hidden_dim]
      fact_embedding: <tf.float32>[batch_size*triple_num, max_triple_length,
        emb_dim]
      fact_lengths: # <tf.int32>[batch_size*triple_num]
      margin: integer value for max margin in TransE loss,
      num_negative_samples: shuffle and sample multiple negative examples for
      the TransE loss

    Returns:
      knowledge_weights:
      knowledge_loss:
    """
    hparams = self._hparams
    encoder_output_shape = common_layers.shape_list(encoder_output)
    encoder_hidden_dim = encoder_output_shape[-1]
    inputs = features["inputs"]
    # <tf.float32>[batch_size, input_length, emb_dim]
    inputs = tf.squeeze(inputs, 2)
    # <tf.float32>[batch_size, input_length]
    context_padding = common_attention.embedding_to_padding(inputs)
    # <tf.float32>[batch_size]
    context_lens = tf.to_float(
        common_attention.padding_to_length(context_padding))
    # <tf.float32>[batch_size, 1]
    context_lens = tf.expand_dims(context_lens, -1)
    # Compute context vector summary.
    # <tf.float32>[batch_size, hidden_dim]
    context_vector_summary = compute_summary_embedding(encoder_output,
                                                       context_lens, hparams)
    knowledge_encoder_output = compute_average_embedding(
        fact_embedding, fact_lengths)
    # <tf.float32>[batch_size, triple_num, emb_dim]
    knowledge_encoder_output = tf.reshape(
        knowledge_encoder_output, [-1, self.triple_num, encoder_hidden_dim])
    original_knowledge_encoder_output = knowledge_encoder_output
    if hparams.similarity_fuction == "dot_product":
      triple_logits = tf.squeeze(
          tf.matmul(knowledge_encoder_output,
                    tf.expand_dims(context_vector_summary, 2)), -1)
    elif hparams.similarity_fuction == "bilinear":
      # Tile the context vector summary.
      # <tf.float32>[batch_size, triple_num*hidden_dim]
      tiled_context_vector = tf.tile(context_vector_summary,
                                     [1, self.triple_num])
      # <tf.float32>[batch_size, triple_num, hidden_dim]
      context_vector = tf.reshape(tiled_context_vector,
                                  [-1, self.triple_num, encoder_hidden_dim])
      # compute outer product
      context_vector = tf.expand_dims(context_vector, -1)
      knowledge_encoder_output = tf.expand_dims(knowledge_encoder_output, 2)
      # <tf.float32>[batch_size, triple_num, hidden_dim, hidden_dim]
      outer_product = tf.matmul(context_vector, knowledge_encoder_output)
      outer_product = tf.reshape(
          outer_product,
          [-1, self.triple_num, encoder_hidden_dim * encoder_hidden_dim])
      triple_logits = tf.squeeze(
          tf.layers.dense(outer_product, 1, name="knolwedge_final_mlp"), -1)

    avg_triple_loss = 0.0
    triple_labels = features["triple_labels"]

    subject_mask = tf.reshape(features["subject_mask"],
                              [-1, self.triple_num, hparams.max_triple_length])
    subject_mask = tf.reshape(subject_mask, [-1, hparams.max_triple_length])

    predicate_mask = tf.reshape(
        features["predicate_mask"],
        [-1, self.triple_num, hparams.max_triple_length])
    predicate_mask = tf.reshape(predicate_mask, [-1, hparams.max_triple_length])

    object_mask = tf.reshape(features["object_mask"],
                             [-1, self.triple_num, hparams.max_triple_length])
    object_mask = tf.reshape(object_mask, [-1, hparams.max_triple_length])

    # mask : [bs, max_seq_len, triple_num]
    # the below operation will result in [bs*triple_num,emb_dim]
    subject_length = tf.cast(
        tf.expand_dims(tf.reduce_sum(subject_mask, -1), 1),
        tf.float32)  # [bs*tn]
    object_length = tf.cast(
        tf.expand_dims(tf.reduce_sum(object_mask, -1), 1), tf.float32)
    predicate_length = tf.cast(
        tf.expand_dims(tf.reduce_sum(predicate_mask, -1), 1), tf.float32)

    # expand dimension 2 to be able to broadcast
    subject_mask = tf.cast(tf.expand_dims(subject_mask, 2), tf.float32)
    predicate_mask = tf.cast(tf.expand_dims(predicate_mask, 2), tf.float32)
    object_mask = tf.cast(tf.expand_dims(object_mask, 2), tf.float32)

    subject_vect = tf.reduce_sum(tf.multiply(
        fact_embedding, subject_mask), 1) / (
            subject_length +
            tf.broadcast_to(tf.constant([1e-5]), tf.shape(subject_length)))
    object_vect = tf.reduce_sum(tf.multiply(fact_embedding, object_mask), 1) / (
        object_length +
        tf.broadcast_to(tf.constant([1e-5]), tf.shape(object_length)))
    predicate_vect = tf.reduce_sum(
        tf.multiply(fact_embedding, predicate_mask), 1) / (
            predicate_length +
            tf.broadcast_to(tf.constant([1e-5]), tf.shape(predicate_length)))

    # Shuffled rows to generate adversarial samples
    shuffled_subject_vect = []
    shuffled_object_vect = []

    for _ in range(num_negative_samples):
      shuffled_subject_vect += [
          tf.gather(subject_vect,
                    tf.random.shuffle(tf.range(tf.shape(subject_vect)[0])))
      ]  # [bs*tn,d]
      shuffled_object_vect += [
          tf.gather(object_vect,
                    tf.random.shuffle(tf.range(tf.shape(object_vect)[0])))
      ]  # [bs*tn,d]

    # KB pretraining loss

    positive_loss = tf.reduce_mean(
        tf.squared_difference(subject_vect + predicate_vect, object_vect))
    negative_loss = 0
    for n_adv in range(num_negative_samples):
      negative_loss += tf.reduce_mean(
          tf.squared_difference(shuffled_subject_vect[n_adv] + predicate_vect,
                                object_vect))
      negative_loss += tf.reduce_mean(
          tf.squared_difference(subject_vect + predicate_vect,
                                shuffled_object_vect[n_adv]))

    # TransE Loss

    negative_loss = negative_loss / (2 * num_negative_samples)

    transe_loss = tf.clip_by_value(
        margin + positive_loss - negative_loss,
        clip_value_min=0,
        clip_value_max=100)
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      triple_losses = tf.nn.weighted_cross_entropy_with_logits(
          labels=triple_labels,
          logits=triple_logits,
          pos_weight=hparams.pos_weight)
      avg_triple_loss = tf.reduce_mean(triple_losses)
      tf.summary.scalar("triple_loss", avg_triple_loss)

    return triple_logits, avg_triple_loss, original_knowledge_encoder_output, transe_loss

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "targets": Target decoder outputs. [batch_size, decoder_length,
            hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    tf.logging.info("Using PgScratch BODY function.")
    hparams = self._hparams

    losses = {}
    inputs = features["inputs"]
    target_space = features["target_space_id"]
    # encoder_output: <tf.float32>[batch_size, input_length, hidden_dim]
    # encoder_decoder_attention_bias: <tf.float32>[batch_size, input_length]
    encoder_output, encoder_decoder_attention_bias = self.encode(
        inputs, target_space, hparams, features=features, losses=losses)

    with tf.variable_scope("knowledge"):
      with tf.name_scope("knowledge_encoding"):
        # Encode knowledge.
        # <tf.float32>[batch_size, triple_num, emb_dim]
        fact_embedding, fact_lengths = self.encode_knowledge_bottom(features)
        tf.logging.info("Encoded knowledge")

      with tf.name_scope("knowledge_selection_and_loss"):
        # Compute knowledge selection and loss.
        triple_logits, avg_triple_selection_loss, knowledge_encoder_output, transe_loss = self.compute_knowledge_selection_and_loss(
            features, encoder_output, fact_embedding, fact_lengths,
            hparams.margin, hparams.num_negative_samples)
        losses["kb_loss"] = avg_triple_selection_loss
        losses["transe_loss"] = transe_loss

    if hparams.attend_kb:
      tf.logging.info("ATTEND_KB is ACTIVE")
      with tf.name_scope("knowledge_attention"):

        knowledge_padding = tf.zeros_like(triple_logits, dtype=tf.float32)
        knowledge_attention_bias = common_attention.attention_bias_ignore_padding(
            knowledge_padding)
        encoder_output = tf.concat([knowledge_encoder_output, encoder_output],
                                   1)
        encoder_decoder_attention_bias = tf.concat(
            [knowledge_attention_bias, encoder_decoder_attention_bias], -1)

    else:
      tf.logging.info("ATTEND_KB is INACTIVE")

    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)

    (decoder_input,
     decoder_self_attention_bias) = transformer.transformer_prepare_decoder(
         targets, hparams, features=features)

    decode_kwargs = {}
    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=transformer.features_to_nonpadding(features, "targets"),
        losses=losses,
        **decode_kwargs)

    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      return ret, losses
    else:
      return ret

  def _normalize_body_output(self, body_out):
    if len(body_out) == 2:
      output, losses = body_out
      if not isinstance(losses, dict):
        losses = {"extra": tf.reduce_mean(losses)}
    else:
      output = body_out
      losses = {"extra": 0.0}

    return output, losses

  def _beam_decode(self,
                   features,
                   decode_length,
                   beam_size,
                   top_beams,
                   alpha,
                   use_tpu=False):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: A bool, whether to do beam decode on TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    """
    return super(transformer.Transformer,
                 self)._beam_decode_slow(features, decode_length, beam_size,
                                         top_beams, alpha, use_tpu)

  def _greedy_infer(self, features, decode_length, use_tpu=False):
    """Fast version of greedy decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      use_tpu: A bool. Whether to build the inference graph for TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    return super(transformer.Transformer,
                 self)._greedy_infer(features, decode_length)


def compute_last_embedding(input_embeddings, input_lengths, hparams):
  """Computes average of last K embedding.

  Args:
    input_embeddings: <tf.float32>[bs, max_seq_len, emb_dim]
    input_lengths: <tf.int64>[bs, 1]
    hparams: model hparams

  Returns:
    last_k_embedding: <tf.float32>[bs, emb_dim]
  """
  max_seq_len = tf.shape(input_embeddings)[1]
  # <tf.float32>[bs, 1, max_seq_len]
  mask = tf.sequence_mask(input_lengths, max_seq_len, dtype=tf.float32)
  del_mask = tf.sequence_mask(
      input_lengths - hparams.last_k, max_seq_len, dtype=tf.float32)
  final_mask = mask - del_mask
  # <tf.float32>[bs, 1, emb_dim]
  sum_embedding = tf.matmul(final_mask, input_embeddings)
  # <tf.float32>[bs, 1, emb_dim]
  last_k_embedding = sum_embedding / tf.to_float(
      tf.expand_dims(
          tf.ones([tf.shape(input_embeddings)[0], 1]) * hparams.last_k, 2))
  # <tf.float32>[bs, dim]
  return tf.squeeze(last_k_embedding, 1)


def compute_max_pool_embedding(input_embeddings, input_lengths):
  """Computes max pool embedding.

  Args:
    input_embeddings: <tf.float32>[bs, max_seq_len, emb_dim]
    input_lengths: <tf.int64>[bs, 1]

  Returns:
    max_pool_embedding: <tf.float32>[bs, emb_dim]
  """
  max_seq_len = tf.shape(input_embeddings)[1]
  # <tf.float32>[bs, max_seq_len]
  mask = 1.0 - tf.sequence_mask(input_lengths, max_seq_len, dtype=tf.float32)
  mask = tf.squeeze(mask * (-1e-6), 1)
  mask = tf.expand_dims(mask, 2)
  # <tf.float32>[bs, emb_dim]
  max_pool_embedding = tf.reduce_max(input_embeddings + mask, 1)
  # <tf.float32>[bs, dim]
  return max_pool_embedding


def compute_average_embedding(input_embeddings, input_lengths):
  """Computes bag-of-words embedding.

  Args:
    input_embeddings: <tf.float32>[bs, max_seq_len, emb_dim]
    input_lengths: <tf.int64>[bs, 1]

  Returns:
    bow_embedding: <tf.float32>[bs, emb_dim]
  """
  max_seq_len = tf.shape(input_embeddings)[1]
  # <tf.float32>[bs, 1, max_seq_len]
  mask = tf.sequence_mask(input_lengths, max_seq_len, dtype=tf.float32)
  # <tf.float32>[bs, 1, emb_dim]
  sum_embedding = tf.matmul(mask, input_embeddings)
  # <tf.float32>[bs, 1, emb_dim]
  avg_embedding = sum_embedding / tf.to_float(tf.expand_dims(input_lengths, 2))
  # <tf.float32>[bs, dim]
  return tf.squeeze(avg_embedding, 1)


def compute_summary_embedding(input_embeddings, input_lengths, hparams):
  """Convert list of embedding to single embedding.

  Args:
    input_embeddings: <tf.float32>[bs, max_seq_len, emb_dim]
    input_lengths: <tf.int64>[bs, 1]
    hparams: model hparams

  Returns:
    embedding: <tf.float32>[bs, emb_dim]
  """
  if hparams.pool_technique == "average":
    return compute_average_embedding(input_embeddings, input_lengths)
  elif hparams.pool_technique == "max_pool":
    return compute_max_pool_embedding(input_embeddings, input_lengths)
  elif hparams.pool_technique == "last":
    return compute_last_embedding(input_embeddings, input_lengths, hparams)


@registry.register_hparams
def neural_assistant_base():
  """HParams for a base neural_assistant model."""
  hparams = transformer.transformer_tpu()
  hparams.add_hparam("pos_weight", 1.0)  # weight for positive triples
  hparams.add_hparam("similarity_fuction",
                     "bilinear")  # dot_product or bilinear
  hparams.add_hparam("pool_technique", "average")  # avg or max pool or last
  hparams.add_hparam("last_k", 1)  # number of last indices for averaging
  hparams.add_hparam("max_triple_length", 30)  # max length of every triple
  hparams.add_hparam("train_triple_num",
                     5000)  # max number of triples during training
  hparams.add_hparam("attend_kb", True)  # if False, it's a transformer model
  hparams.add_hparam("kb_loss_weight", 0.0)  # weight for distant supervision
  hparams.add_hparam("test_triple_num",
                     28483)  # max triples of KB
  hparams.add_hparam("margin", 0.0)  # KB training max-margin loss
  hparams.add_hparam(
      "num_negative_samples",
      1)  # Sampling number of different adversarial training examples
  hparams.add_hparam("kb_train_weight", 0.0)
  # KB_training loss weight which combines Language model and KB selection loss
  return hparams


@registry.register_hparams
def neural_assistant_tiny():
  """HParams for tiny neural_assistant model."""
  hparams = transformer.transformer_tiny_tpu()
  hparams.add_hparam("pos_weight", 1.0)  # weight for positive triples
  hparams.add_hparam("similarity_fuction",
                     "bilinear")  # dot_product or bilinear
  hparams.add_hparam("pool_technique", "average")  # avg or max pool or last
  hparams.add_hparam("last_k", 1)  # number of last indices for averaging
  hparams.add_hparam("max_triple_length", 30)  # max length of every triple
  hparams.add_hparam("train_triple_num",
                     5000)  # max number of triples during training
  hparams.add_hparam("attend_kb", True)  # if False, it's a transformer model
  hparams.add_hparam("kb_loss_weight", 0.0)  # weight for distant supervision
  hparams.add_hparam("test_triple_num",
                     28483)  # max triples of KB
  hparams.add_hparam("margin", 1.0)  # KB training max-margin loss
  hparams.add_hparam(
      "num_negative_samples",
      1)  # Sampling number of different adversarial training examples
  hparams.add_hparam("kb_train_weight", 0.0)
  # KB_training loss weight which combines Language model and KB selection loss
  return hparams


@registry.register_hparams
def neural_assistant_tiny_ds():
  """HParams for tiny neural_assistant model with distant supervision loss."""
  hparams = neural_assistant_tiny()
  hparams.kb_loss_weight = 0.2
  return hparams
