"""Utility functions for building models."""
from __future__ import print_function

import time

import tensorflow as tf

import utils.misc_utils as utils


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
                                       src_embed_file,
                                       tgt_embed_file,
                                       src_embed_trainable,
                                       tgt_embed_trainable,
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
    trainable_embed_vars: A list of all trainable embedding variables.

  Raises:
    ValueError: if use share_vocab but source and target have different vocab
      size.
  """
  def maybe_get_file_initializer(embed_file, vocab_size, embed_size,
                                 vocab_table):
    """Returns a variable initializer and an initializer for *that*.

    Args:
      embed_file: may be None.
      vocab_size: Python integer.
      embed_size: Python integer.
      vocab_table: A Lookup table that maps incoming words to ids.

    Returns:
      A tuple (data_initializer, variable_initializer), where:

      - data_initializer is an Operation that initializes the underlying data.
      - variable_initizlier returns a Tensor with the initial variable value.
        Note: data_initializer must be executed before this tensor is accessed.
    """
    if embed_file is None:
      return tf.no_op(), tf.zeros_initializer()
    else:
      def from_file_initializer(shape, dtype, partition_info):
        """Variable initializer that loads floating point values from a file."""
        del partition_info  # Unused
        assert shape == [vocab_size, embed_size]
        # Read the whole file into a string.
        data = tf.read_file(embed_file)
        # Split the data by spaces/newlines to get a bunch of string tensors.
        lines = tf.string_split([data], "\n").values
        # Convert number strings to floating point types.
        values = tf.decode_csv(
            lines, [[tf.string]] + [[dtype]] * embed_size,
            field_delim=" ", use_quote_delim=False)
        words = values[0]
        word_ids = vocab_table.lookup(words)  # Maps embedding lines -> ids
        neg_order_word_ids, vocabulary_to_embedding_line_order = tf.nn.top_k(
            -word_ids, k=vocab_size)  # Negate to get increasing order vocab id.
        order_word_ids = -neg_order_word_ids
        missing_vocab_ids = tf.setdiff1d(
            tf.range(0, vocab_size), order_word_ids)
        with tf.control_dependencies([
            tf.assert_equal(
                tf.size(missing_vocab_ids), 0,
                message=("Pretrained embedding does not contain all required "
                         "words from the vocabulary.  Missing vocab ids: "),
                data=[missing_vocab_ids],
                summarize=100)]):
          vocabulary_to_embedding_line_order = tf.identity(
              vocabulary_to_embedding_line_order)

        # Need to map ids -> embedding vectors
        embedding_values = tf.transpose(tf.stack(values[1:]))
        embedding_values.shape.assert_is_compatible_with(
            tf.TensorShape([vocab_size, embed_size]))
        reordered_values = tf.gather(
            embedding_values, vocabulary_to_embedding_line_order)
        # For each id in the vocabulary, identify the associated row
        # in the embedding file.
        return reordered_values
      return tf.no_op(), from_file_initializer

  def get_pretrained_embeddings(vocab_size, embed_size, embed_trainable,
                                embed_file, vocab_table):
    """Get pretrained embeddings."""
    #  Start with a standard initializer; then assign
    # into rows containing vocab in the pretrained embeddings.
    data_initializer, variable_initializer = maybe_get_file_initializer(
        embed_file, vocab_size, embed_size, vocab_table)
    with tf.control_dependencies([data_initializer]):
      pretrained_embeddings = tf.get_variable(
          "embedding",
          [vocab_size, embed_size],
          dtype,
          variable_initializer,
          trainable=embed_trainable)
    return pretrained_embeddings, data_initializer

  trainable_embed_vars = []
  with tf.variable_scope(
      scope or "embeddings", dtype=dtype) as scope:
    # Share embedding
    if share_vocab:
      # Pretrained embeddings
      assert not src_embed_file, "Loading pre-trained embeddings not supported"
      if src_embed_file:
        (embedding, embedding_init) = get_pretrained_embeddings(
            src_vocab_size, src_embed_size, src_embed_file,
            src_embed_trainable, src_vocab_table)
        embedding_encoder_init = embedding_init
        embedding_decoder_init = embedding_init
        if src_embed_trainable:
          trainable_embed_vars.append(embedding)
      else:
        if src_vocab_size != tgt_vocab_size:
          raise ValueError("Share embedding but different src/tgt vocab sizes"
                           " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
        utils.print_out("# Use the same source embeddings for target")
        embedding = tf.get_variable(
            "embedding_share", [src_vocab_size, src_embed_size], dtype)
        embedding_encoder_init = None
        embedding_decoder_init = None
        trainable_embed_vars.append(embedding)
      embedding_encoder = embedding
      embedding_decoder = embedding
    else:
      with tf.variable_scope("encoder"):
        assert not src_embed_file, "Loading pre-trained embeddings not supported"
        if src_embed_file:
          (embedding_encoder,
           embedding_encoder_init) = get_pretrained_embeddings(
               src_vocab_size, src_embed_size, src_embed_trainable,
               src_embed_file, src_vocab_table)
          if src_embed_trainable:
            trainable_embed_vars.append(embedding_encoder)
        else:
          embedding_encoder = tf.get_variable(
              "embedding_encoder", [src_vocab_size, src_embed_size], dtype)
          embedding_encoder_init = None
          trainable_embed_vars.append(embedding_encoder)

      with tf.variable_scope("decoder"):
        assert not tgt_embed_file, "Loading pre-trained embeddings not supported"
        if tgt_embed_file:
          (embedding_decoder, embedding_decoder_init) = get_pretrained_embeddings(
              tgt_vocab_size, tgt_embed_size, tgt_embed_trainable,
              tgt_embed_file, tgt_vocab_table)
        else:
          embedding_decoder = tf.get_variable(
              "embedding_decoder", [tgt_vocab_size, tgt_embed_size], dtype)
          embedding_decoder_init = None
        if tgt_embed_trainable:
          trainable_embed_vars.append(embedding_decoder)

  return (embedding_encoder, embedding_decoder, trainable_embed_vars,
          embedding_encoder_init, embedding_decoder_init)


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


def create_or_load_model(model, model_dir, session, hparams):
  """Create translation model and initialize or load parameters in session."""
  utils.print_out("# Creating model, model_dir %s" % model_dir)
  utils.print_out("  num layers=%d, num units=%d, unit_type=%s, attention=%s,"
                  " num_gpus=%d, attention_type=%s" %
                  (hparams.num_layers, hparams.num_units, hparams.unit_type,
                   hparams.attention, hparams.num_gpus, hparams.attention_type))
  start_time = time.time()

  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    utils.print_out(
        "  nmt model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    utils.print_out("  created model with fresh parameters, time %.2fs." %
                    (time.time() - start_time))
    session.run(tf.global_variables_initializer())

  session.run(tf.initialize_all_tables())

  return model


def compute_perplexity(model, sess, batches, name):
  """Subclass must implement this method.

  Compute perplexity of the output of given batches.

  Args:
    model: model for compute perplexity.
    sess: tensorflow session to use.
    batches: data to eval for compute perplexity.
    name: name of the batch.

  Returns:
    The perplexity of the eval outputs.
  """
  total_loss = 0
  total_predict_count = 0
  start_time = time.time()
  for batch in batches:
    loss, predict_count = model.step(sess, batch,
                                     tf.contrib.learn.ModeKeys.EVAL)
    total_loss += (loss * batch["size"])
    total_predict_count += predict_count

  perplexity = utils.safe_exp(total_loss / total_predict_count)
  utils.print_time("  eval %s: perplexity %.2f" % (name, perplexity),
                   start_time)
  return perplexity
