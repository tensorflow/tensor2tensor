"""Utility functions for building models."""
from __future__ import print_function

import time

import tensorflow as tf

from .utils import misc_utils as utils


__all__ = [
    "get_device_str", "create_emb_for_encoder_and_decoder", "create_rnn_cell",
    "count_embeddings", "gradient_clip", "apply_grad_multiplier",
    "create_or_load_model", "compute_perplexity"
]


def get_device_str(device_id, num_gpus):
  """Return a device string for multi-GPU setup."""
  if num_gpus == 0:
    return "/cpu:0"
  device_str_output = "/gpu:%d" % (device_id % num_gpus)
  return device_str_output


def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       src_vocab_table,
                                       tgt_vocab_table,
                                       dtype=tf.float32,
                                       scope=None):
  """Create embedding matrix for both encoder and decoder.

  Args:
    share_vocab: A boolean. Whether to share embedding matrix for both
      encoder and decoder.
    src_vocab_size: An integer. The source vocab size.
    tgt_vocab_size: An integer. The target vocab size.
    src_embed_size: An integer. The embedding dimension for the encoder's
      embedding.
    tgt_embed_size: An integer. The embedding dimension for the decoder's
      embedding.
    dtype: dtype of the embedding matrix. Default to float32.
    scope: VariableScope for the created subgraph. Default to "embedding".

  Returns:
    embedding_encoder: Encoder's embedding matrix.
    embedding_decoder: Decoder's embedding matrix.
    embed_vars: A list of all embedding variables.

  Raises:
    ValueError: if use share_vocab but source and target have different vocab
      size.
  """
  embed_vars = []
  with tf.variable_scope(
      scope or "embeddings", dtype=dtype) as scope:
    # Share embedding
    if share_vocab:
      if src_vocab_size != tgt_vocab_size:
        raise ValueError("Share embedding but different src/tgt vocab sizes"
                         " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
      utils.print_out("# Use the same source embeddings for target")
      embedding = tf.get_variable(
          "embedding_share", [src_vocab_size, src_embed_size], dtype)
      embed_vars.append(embedding)
      embedding_encoder = embedding
      embedding_decoder = embedding
    else:
      with tf.variable_scope("encoder"):
        embedding_encoder = tf.get_variable(
            "embedding_encoder", [src_vocab_size, src_embed_size], dtype)
        embed_vars.append(embedding_encoder)

      with tf.variable_scope("decoder"):
        embedding_decoder = tf.get_variable(
            "embedding_decoder", [tgt_vocab_size, tgt_embed_size], dtype)
        embed_vars.append(embedding_decoder)
  return (embedding_encoder, embedding_decoder, embed_vars)


def _single_cell(hparams, mode, residual_connection=False, device_str=None):
  """Create an instance of a single RNN cell."""
  # dropout (= 1 - keep_prob) is set to 0 during eval and infer
  dropout = hparams.dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

  unit_type = hparams.unit_type
  num_units = hparams.num_units
  forget_bias = hparams.forget_bias

  # Cell Type
  if unit_type == "lstm":
    utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
    single_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units,
        forget_bias=forget_bias)
  elif unit_type == "gru":
    utils.print_out("  GRU", new_line=False)
    single_cell = tf.contrib.rnn.GRUCell(num_units)
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)

  # Dropout (= 1 - keep_prob)
  if dropout > 0.0:
    single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))
    utils.print_out("  %s, dropout=%g " %(type(single_cell).__name__, dropout),
                    new_line=False)

  # Residual
  if residual_connection:
    single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)
    utils.print_out("  %s" % type(single_cell).__name__, new_line=False)

  # Device Wrapper
  if device_str:
    single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
    utils.print_out("  %s, device=%s" %
                    (type(single_cell).__name__, device_str), new_line=False)

  return single_cell


def _cell_list(hparams, num_layers, num_residual_layers, mode, base_gpu=0):
  """Create a list of RNN cells.

  Args:
    hparams: arguments to create an RNN cell.
    num_layers: number of cells.
    num_residual_layers: Number of residual layers from top to bottom. For
      example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
      cells in the returned list will be wrapped with `ResidualWrapper`.
    mode: either tf.contrib.learn.TRAIN/EVAL/INFER
    base_gpu: The gpu device id to use for the first RNN cell in the
      returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
      as its device id.

  Returns:
    A list of RNN cells.
  """
  num_gpus = hparams.num_gpus

  # Multi-GPU
  cell_list = []
  for i in range(num_layers):
    utils.print_out("  cell %d" % i, new_line=False)
    single_cell = _single_cell(
        hparams, mode,
        residual_connection=(i >= num_layers - num_residual_layers),
        device_str=get_device_str(i + base_gpu, num_gpus),
    )
    utils.print_out("")
    cell_list.append(single_cell)

  return cell_list


def create_rnn_cell(hparams, num_layers, num_residual_layers, mode, base_gpu=0):
  """Create multi-layer RNN cell."""

  cell_list = _cell_list(hparams, num_layers, num_residual_layers, mode,
                         base_gpu=base_gpu)

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    return tf.contrib.rnn.MultiRNNCell(cell_list)


def count_embeddings(embs, grads):
  """Returns the number of embedding lookups."""
  assert len(embs) > 1
  assert len(embs) == len(grads)
  num_ids = []
  for var, grad in zip(embs, grads):
    assert grad is not None, ("No grad found for ", var.name)
    with tf.device(grad.device):
      assert isinstance(grad, tf.IndexedSlices)
      num_ids.append(tf.shape(grad.indices)[0])
  return tf.cast(tf.add_n(num_ids), embs[0].dtype)


def gradient_clip(gradients, params, emb_vars, hparams):
  """Clipping gradients of a model."""
  # Prepare clipping by embedding and other gradient separately.
  # Note: model_helper.count_embeddings can't be called after tf.clip_by_value
  if emb_vars and hparams.max_emb_gradient_norm is not None:
    emb_grads = gradients[:len(emb_vars)]
    # To caculate avg_emb_norm depending on how many words in a mini-batch.
    emb_count = tf.sqrt(count_embeddings(emb_vars, emb_grads))

  # Clip by values.
  if hparams.gradient_clip_value is not None:
    pattern = hparams.gradient_clip_pattern
    clip_value = hparams.gradient_clip_value
    clipped_gradients = []
    for (param, grad) in zip(params, gradients):
      if not pattern or pattern in param.name:  # clip everything or pattern
        utils.print_out("  clip %s to value %g" % (param.name, clip_value))
        clipped_gradients.append(
            tf.clip_by_value(grad, -clip_value, clip_value))
      else:
        clipped_gradients.append(grad)
    gradients = clipped_gradients

  # Clip by norm.
  if emb_vars and hparams.max_emb_gradient_norm is not None:
    emb_grads = gradients[:len(emb_vars)]
    other_grads = gradients[len(emb_vars):]

    avg_emb_grad_norm = tf.global_norm(emb_grads) / emb_count
    emb_grads_scale = tf.minimum(
        1.0, (hparams.max_emb_gradient_norm / avg_emb_grad_norm))

    other_grad_norm = tf.global_norm(other_grads)
    other_grads_scale = tf.minimum(
        1.0, (hparams.max_gradient_norm / other_grad_norm))

    grads_scale = tf.minimum(emb_grads_scale, other_grads_scale)

    clipped_gradients = apply_grad_multiplier(
        params, gradients, grads_scale)

    gradient_norm_summary = [
        tf.summary.scalar("emb_grads_scale", emb_grads_scale),
        tf.summary.scalar("other_grads_scale", other_grads_scale),
        tf.summary.scalar("grads_scale", grads_scale),
        tf.summary.scalar("avg_emb_grad_norm", avg_emb_grad_norm),
        tf.summary.scalar("other_grad_norm", other_grad_norm)
    ]
  else:
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, hparams.max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
  gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

  return clipped_gradients, gradient_norm_summary


def apply_grad_multiplier(vs, gs, grad_scale):
  """Scales all gradients by grad_scale."""
  final_grad_in_order = []
  assert len(vs) == len(gs)
  for var, grad in zip(vs, gs):
    assert grad is not None, ("No grad found for ", var.name)
    with tf.device(var.device):
      if isinstance(grad, tf.IndexedSlices):
        final_grad_in_order.append(tf.IndexedSlices(
            grad_scale * tf.check_numerics(grad.values, "%s is not finite." %
                                           var.name), grad.indices,
            grad.dense_shape))
      else:
        final_grad_in_order.append(
            grad_scale *
            tf.check_numerics(grad, "%s is not finite." % var.name))
  return final_grad_in_order


def create_or_load_model(model, model_dir, session, hparams, name):
  """Create translation model and initialize or load parameters in session."""
  start_time = time.time()
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    model.saver.restore(session, latest_ckpt)
    utils.print_out(
        "  loaded %s model parameters from %s, time %.2fs" %
        (name, latest_ckpt, time.time() - start_time))
  else:
    utils.print_out("  created %s model with fresh parameters, time %.2fs." %
                    (name, time.time() - start_time))
    session.run(tf.global_variables_initializer())
    model.saver.save(session, hparams.out_dir, global_step=0)

  session.run(tf.initialize_all_tables())

  global_step = model.global_step.eval(session=session)
  return model, global_step


def compute_perplexity(model, sess, name):
  """Compute perplexity of the output of the model.

  Args:
    model: model for compute perplexity.
    sess: tensorflow session to use.
    name: name of the batch.

  Returns:
    The perplexity of the eval outputs.
  """
  total_loss = 0
  total_predict_count = 0
  start_time = time.time()

  while True:
    try:
      loss, predict_count, batch_size = model.eval(sess)
      total_loss += loss * batch_size
      total_predict_count += predict_count
    except tf.errors.OutOfRangeError:
      break

  perplexity = utils.safe_exp(total_loss / total_predict_count)
  utils.print_time("  eval %s: perplexity %.2f" % (name, perplexity),
                   start_time)
  return perplexity
