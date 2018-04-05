# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Implemetation of beam seach with penalties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_layers

import tensorflow as tf

from tensorflow.python.util import nest

# Assuming EOS_ID is 1
EOS_ID = 1
# Default value for INF
INF = 1. * 1e7


def _merge_beam_dim(tensor):
  """Reshapes first two dimensions in to single dimension.

  Args:
    tensor: Tensor to reshape of shape [A, B, ...]

  Returns:
    Reshaped tensor of shape [A*B, ...]
  """
  shape = common_layers.shape_list(tensor)
  shape[0] *= shape[1]  # batch -> batch * beam_size
  shape.pop(1)  # Remove beam dim
  return tf.reshape(tensor, shape)


def _unmerge_beam_dim(tensor, batch_size, beam_size):
  """Reshapes first dimension back to [batch_size, beam_size].

  Args:
    tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
    batch_size: Tensor, original batch size.
    beam_size: int, original beam size.

  Returns:
    Reshaped tensor of shape [batch_size, beam_size, ...]
  """
  shape = common_layers.shape_list(tensor)
  new_shape = [batch_size] + [beam_size] + shape[1:]
  return tf.reshape(tensor, new_shape)


def _expand_to_beam_size(tensor, beam_size):
  """Tiles a given tensor by beam_size.

  Args:
    tensor: tensor to tile [batch_size, ...]
    beam_size: How much to tile the tensor by.

  Returns:
    Tiled tensor [batch_size, beam_size, ...]
  """
  tensor = tf.expand_dims(tensor, axis=1)
  tile_dims = [1] * tensor.shape.ndims
  tile_dims[1] = beam_size

  return tf.tile(tensor, tile_dims)


def get_state_shape_invariants(tensor):
  """Returns the shape of the tensor but sets middle dims to None."""
  shape = tensor.shape.as_list()
  for i in range(1, len(shape) - 1):
    shape[i] = None
  return tf.TensorShape(shape)


def log_prob_from_logits(logits):
  return logits - tf.reduce_logsumexp(logits, axis=2, keep_dims=True)


def compute_batch_indices(batch_size, beam_size):
  """Computes the i'th coodinate that contains the batch index for gathers.

  Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
  batch the beam item is in. This will create the i of the i,j coordinate
  needed for the gather.

  Args:
    batch_size: Batch size
    beam_size: Size of the beam.
  Returns:
    batch_pos: [batch_size, beam_size] tensor of ids
  """
  batch_pos = tf.range(batch_size * beam_size) // beam_size
  batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
  return batch_pos


def compute_topk_scores_and_seq(sequences, scores, scores_to_gather, flags,
                                beam_size, batch_size, prefix="default",
                                states_to_gather=None):
  """Given sequences and scores, will gather the top k=beam size sequences.

  This function is used to grow alive, and finished. It takes sequences,
  scores, and flags, and returns the top k from sequences, scores_to_gather,
  and flags based on the values in scores.

  This method permits easy introspection using tfdbg.  It adds three named ops
  that are prefixed by `prefix`:
    - _topk_seq: the tensor for topk_seq returned by this method.
    - _topk_flags: the tensor for topk_finished_flags returned by this method.
    - _topk_scores: the tensor for tokp_gathered_scores returned by this method.

  Args:
    sequences: Tensor of sequences that we need to gather from.
      [batch_size, beam_size, seq_length]
    scores: Tensor of scores for each sequence in sequences.
      [batch_size, beam_size]. We will use these to compute the topk.
    scores_to_gather: Tensor of scores for each sequence in sequences.
      [batch_size, beam_size]. We will return the gathered scores from here.
      Scores to gather is different from scores because for grow_alive, we will
      need to return log_probs, while for grow_finished, we will need to return
      the length penalized scors.
    flags: Tensor of bools for sequences that say whether a sequence has reached
      EOS or not
    beam_size: int
    batch_size: int
    prefix: string that will prefix unique names for the ops run.
    states_to_gather: dict (possibly nested) of decoding states.
  Returns:
    Tuple of
    (topk_seq [batch_size, beam_size, decode_length],
     topk_gathered_scores [batch_size, beam_size],
     topk_finished_flags[batch_size, beam_size])
  """
  _, topk_indexes = tf.nn.top_k(scores, k=beam_size)
  # The next three steps are to create coordinates for tf.gather_nd to pull
  # out the topk sequences from sequences based on scores.
  # batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
  # batch the beam item is in. This will create the i of the i,j coordinate
  # needed for the gather
  batch_pos = compute_batch_indices(batch_size, beam_size)

  # top coordinates will give us the actual coordinates to do the gather.
  # stacking will create a tensor of dimension batch * beam * 2, where the
  # last dimension contains the i,j gathering coordinates.
  top_coordinates = tf.stack([batch_pos, topk_indexes], axis=2)

  # Gather up the highest scoring sequences.  For each operation added, give it
  # a concrete name to simplify observing these operations with tfdbg.  Clients
  # can capture these tensors by watching these node names.
  def gather(tensor, name):
    return tf.gather_nd(tensor, top_coordinates, name=(prefix + name))
  topk_seq = gather(sequences, "_topk_seq")
  topk_flags = gather(flags, "_topk_flags")
  topk_gathered_scores = gather(scores_to_gather, "_topk_scores")
  if states_to_gather:
    topk_gathered_states = nest.map_structure(
        lambda state: gather(state, "_topk_states"), states_to_gather)
  else:
    topk_gathered_states = states_to_gather
  return topk_seq, topk_gathered_scores, topk_flags, topk_gathered_states


def beam_search(symbols_to_logits_fn,
                initial_ids,
                beam_size,
                decode_length,
                vocab_size,
                alpha,
                states=None,
                eos_id=EOS_ID,
                stop_early=True):
  """Beam search with length penalties.

  Requires a function that can take the currently decoded sybmols and return
  the logits for the next symbol. The implementation is inspired by
  https://arxiv.org/abs/1609.08144.

  When running, the beam search steps can be visualized by using tfdbg to watch
  the operations generating the output ids for each beam step.  These operations
  have the pattern:
    (alive|finished)_topk_(seq,scores)

  Operations marked `alive` represent the new beam sequences that will be
  processed in the next step.  Operations marked `finished` represent the
  completed beam sequences, which may be padded with 0s if no beams finished.

  Operations marked `seq` store the full beam sequence for the time step.
  Operations marked `scores` store the sequence's final log scores.

  The beam search steps will be processed sequentially in order, so when
  capturing observed from these operations, tensors, clients can make
  assumptions about which step is being recorded.

  WARNING: Assumes 2nd dimension of tensors in `states` and not invariant, this
  means that the shape of the 2nd dimension of these tensors will not be
  available (i.e. set to None) inside symbols_to_logits_fn.

  Args:
    symbols_to_logits_fn: Interface to the model, to provide logits.
        Shoud take [batch_size, decoded_ids] and return [batch_size, vocab_size]
    initial_ids: Ids to start off the decoding, this will be the first thing
        handed to symbols_to_logits_fn (after expanding to beam size)
        [batch_size]
    beam_size: Size of the beam.
    decode_length: Number of steps to decode for.
    vocab_size: Size of the vocab, must equal the size of the logits returned by
        symbols_to_logits_fn
    alpha: alpha for length penalty.
    states: dict (possibly nested) of decoding states.
    eos_id: ID for end of sentence.
    stop_early: a boolean - stop once best sequence is provably determined.
  Returns:
    Tuple of
    (decoded beams [batch_size, beam_size, decode_length]
     decoding probablities [batch_size, beam_size])
  """
  batch_size = common_layers.shape_list(initial_ids)[0]

  # Assume initial_ids are prob 1.0
  initial_log_probs = tf.constant([[0.] + [-float("inf")] * (beam_size - 1)])
  # Expand to beam_size (batch_size, beam_size)
  alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1])

  # Expand each batch and state to beam_size
  alive_seq = _expand_to_beam_size(initial_ids, beam_size)
  alive_seq = tf.expand_dims(alive_seq, axis=2)  # (batch_size, beam_size, 1)
  if states:
    states = nest.map_structure(
        lambda state: _expand_to_beam_size(state, beam_size), states)
  else:
    states = {}

  # Finished will keep track of all the sequences that have finished so far
  # Finished log probs will be negative infinity in the beginning
  # finished_flags will keep track of booleans
  finished_seq = tf.zeros(common_layers.shape_list(alive_seq), tf.int32)
  # Setting the scores of the initial to negative infinity.
  finished_scores = tf.ones([batch_size, beam_size]) * -INF
  finished_flags = tf.zeros([batch_size, beam_size], tf.bool)

  def grow_finished(finished_seq, finished_scores, finished_flags, curr_seq,
                    curr_scores, curr_finished):
    """Given sequences and scores, will gather the top k=beam size sequences.

    Args:
      finished_seq: Current finished sequences.
        [batch_size, beam_size, current_decoded_length]
      finished_scores: scores for each of these sequences.
        [batch_size, beam_size]
      finished_flags: finished bools for each of these sequences.
        [batch_size, beam_size]
      curr_seq: current topk sequence that has been grown by one position.
        [batch_size, beam_size, current_decoded_length]
      curr_scores: scores for each of these sequences. [batch_size, beam_size]
      curr_finished: Finished flags for each of these sequences.
        [batch_size, beam_size]
    Returns:
      Tuple of
        (Topk sequences based on scores,
         log probs of these sequences,
         Finished flags of these sequences)
    """
    # First append a column of 0'ids to finished to make the same length with
    # finished scores
    finished_seq = tf.concat(
        [finished_seq,
         tf.zeros([batch_size, beam_size, 1], tf.int32)], axis=2)

    # Set the scores of the unfinished seq in curr_seq to large negative
    # values
    curr_scores += (1. - tf.to_float(curr_finished)) * -INF
    # concatenating the sequences and scores along beam axis
    curr_finished_seq = tf.concat([finished_seq, curr_seq], axis=1)
    curr_finished_scores = tf.concat([finished_scores, curr_scores], axis=1)
    curr_finished_flags = tf.concat([finished_flags, curr_finished], axis=1)
    return compute_topk_scores_and_seq(
        curr_finished_seq, curr_finished_scores, curr_finished_scores,
        curr_finished_flags, beam_size, batch_size, "grow_finished")

  def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished, states):
    """Given sequences and scores, will gather the top k=beam size sequences.

    Args:
      curr_seq: current topk sequence that has been grown by one position.
        [batch_size, beam_size, i+1]
      curr_scores: scores for each of these sequences. [batch_size, beam_size]
      curr_log_probs: log probs for each of these sequences.
        [batch_size, beam_size]
      curr_finished: Finished flags for each of these sequences.
        [batch_size, beam_size]
      states: dict (possibly nested) of decoding states.
    Returns:
      Tuple of
        (Topk sequences based on scores,
         log probs of these sequences,
         Finished flags of these sequences)
    """
    # Set the scores of the finished seq in curr_seq to large negative
    # values
    curr_scores += tf.to_float(curr_finished) * -INF
    return compute_topk_scores_and_seq(curr_seq, curr_scores, curr_log_probs,
                                       curr_finished, beam_size, batch_size,
                                       "grow_alive", states)

  def grow_topk(i, alive_seq, alive_log_probs, states):
    r"""Inner beam seach loop.

    This function takes the current alive sequences, and grows them to topk
    sequences where k = 2*beam. We use 2*beam because, we could have beam_size
    number of sequences that might hit <EOS> and there will be no alive
    sequences to continue. With 2*beam_size, this will not happen. This relies
    on the assumption the vocab size is > beam size. If this is true, we'll
    have at least beam_size non <EOS> extensions if we extract the next top
    2*beam words.
    Length penalty is given by = (5+len(decode)/6) ^ -\alpha. Pls refer to
    https://arxiv.org/abs/1609.08144.

    Args:
      i: loop index
      alive_seq: Topk sequences decoded so far [batch_size, beam_size, i+1]
      alive_log_probs: probabilities of these sequences. [batch_size, beam_size]
      states: dict (possibly nested) of decoding states.
    Returns:
      Tuple of
        (Topk sequences extended by the next word,
         The log probs of these sequences,
         The scores with length penalty of these sequences,
         Flags indicating which of these sequences have finished decoding,
         dict of transformed decoding states)
    """
    # Get the logits for all the possible next symbols
    flat_ids = tf.reshape(alive_seq, [batch_size * beam_size, -1])

    # (batch_size * beam_size, decoded_length)
    if states:
      flat_states = nest.map_structure(_merge_beam_dim, states)
      flat_logits, flat_states = symbols_to_logits_fn(flat_ids, i, flat_states)
      states = nest.map_structure(
          lambda t: _unmerge_beam_dim(t, batch_size, beam_size), flat_states)
    else:
      flat_logits = symbols_to_logits_fn(flat_ids)
    logits = tf.reshape(flat_logits, [batch_size, beam_size, -1])

    # Convert logits to normalized log probs
    candidate_log_probs = log_prob_from_logits(logits)

    # Multiply the probabilites by the current probabilites of the beam.
    # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
    log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

    length_penalty = tf.pow(((5. + tf.to_float(i + 1)) / 6.), alpha)

    curr_scores = log_probs / length_penalty
    # Flatten out (beam_size, vocab_size) probs in to a list of possibilites
    flat_curr_scores = tf.reshape(curr_scores, [-1, beam_size * vocab_size])

    topk_scores, topk_ids = tf.nn.top_k(flat_curr_scores, k=beam_size * 2)

    # Recovering the log probs because we will need to send them back
    topk_log_probs = topk_scores * length_penalty

    # Work out what beam the top probs are in.
    topk_beam_index = topk_ids // vocab_size
    topk_ids %= vocab_size  # Unflatten the ids

    # The next three steps are to create coordinates for tf.gather_nd to pull
    # out the correct seqences from id's that we need to grow.
    # We will also use the coordinates to gather the booleans of the beam items
    # that survived.
    batch_pos = compute_batch_indices(batch_size, beam_size * 2)

    # top beams will give us the actual coordinates to do the gather.
    # stacking will create a tensor of dimension batch * beam * 2, where the
    # last dimension contains the i,j gathering coordinates.
    topk_coordinates = tf.stack([batch_pos, topk_beam_index], axis=2)

    # Gather up the most probable 2*beams both for the ids and finished_in_alive
    # bools
    topk_seq = tf.gather_nd(alive_seq, topk_coordinates)
    if states:
      states = nest.map_structure(
          lambda state: tf.gather_nd(state, topk_coordinates), states)

    # Append the most probable alive
    topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2)

    topk_finished = tf.equal(topk_ids, eos_id)

    return topk_seq, topk_log_probs, topk_scores, topk_finished, states

  def inner_loop(i, alive_seq, alive_log_probs, finished_seq, finished_scores,
                 finished_flags, states):
    """Inner beam seach loop.

    There are three groups of tensors, alive, finished, and topk.
    The alive group contains information about the current alive sequences
    The topk group contains information about alive + topk current decoded words
    the finished group contains information about finished sentences, that is,
    the ones that have decoded to <EOS>. These are what we return.
    The general beam search algorithm is as follows:
    While we haven't terminated (pls look at termination condition)
      1. Grow the current alive to get beam*2 topk sequences
      2. Among the topk, keep the top beam_size ones that haven't reached EOS
      into alive
      3. Among the topk, keep the top beam_size ones have reached EOS into
      finished
    Repeat
    To make things simple with using fixed size tensors, we will end
    up inserting unfinished sequences into finished in the beginning. To stop
    that we add -ve INF to the score of the unfinished sequence so that when a
    true finished sequence does appear, it will have a higher score than all the
    unfinished ones.

    Args:
      i: loop index
      alive_seq: Topk sequences decoded so far [batch_size, beam_size, i+1]
      alive_log_probs: probabilities of the beams. [batch_size, beam_size]
      finished_seq: Current finished sequences.
        [batch_size, beam_size, i+1]
      finished_scores: scores for each of these sequences.
        [batch_size, beam_size]
      finished_flags: finished bools for each of these sequences.
        [batch_size, beam_size]
      states: dict (possibly nested) of decoding states.

    Returns:
      Tuple of
        (Incremented loop index
         New alive sequences,
         Log probs of the alive sequences,
         New finished sequences,
         Scores of the new finished sequences,
         Flags inidicating which sequence in finished as reached EOS,
         dict of final decoding states)
    """

    # Each inner loop, we carry out three steps:
    # 1. Get the current topk items.
    # 2. Extract the ones that have finished and haven't finished
    # 3. Recompute the contents of finished based on scores.
    topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(
        i, alive_seq, alive_log_probs, states)
    alive_seq, alive_log_probs, _, states = grow_alive(
        topk_seq, topk_scores, topk_log_probs, topk_finished, states)
    finished_seq, finished_scores, finished_flags, _ = grow_finished(
        finished_seq, finished_scores, finished_flags, topk_seq, topk_scores,
        topk_finished)

    return (i + 1, alive_seq, alive_log_probs, finished_seq, finished_scores,
            finished_flags, states)

  def _is_finished(i, unused_alive_seq, alive_log_probs, unused_finished_seq,
                   finished_scores, finished_in_finished, unused_states):
    """Checking termination condition.

    We terminate when we decoded up to decode_length or the lowest scoring item
    in finished has a greater score that the higest prob item in alive divided
    by the max length penalty

    Args:
      i: loop index
      alive_log_probs: probabilities of the beams. [batch_size, beam_size]
      finished_scores: scores for each of these sequences.
        [batch_size, beam_size]
      finished_in_finished: finished bools for each of these sequences.
        [batch_size, beam_size]

    Returns:
      Bool.
    """
    if not stop_early:
      return tf.less(i, decode_length)
    max_length_penalty = tf.pow(((5. + tf.to_float(decode_length)) / 6.), alpha)
    # The best possible score of the most likley alive sequence
    lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty

    # Now to compute the lowest score of a finished sequence in finished
    # If the sequence isn't finished, we multiply it's score by 0. since
    # scores are all -ve, taking the min will give us the score of the lowest
    # finished item.
    lowest_score_of_fininshed_in_finished = tf.reduce_min(
        finished_scores * tf.to_float(finished_in_finished), axis=1)
    # If none of the sequences have finished, then the min will be 0 and
    # we have to replace it by -ve INF if it is. The score of any seq in alive
    # will be much higher than -ve INF and the termination condition will not
    # be met.
    lowest_score_of_fininshed_in_finished += (
        (1. - tf.to_float(tf.reduce_any(finished_in_finished, 1))) * -INF)

    bound_is_met = tf.reduce_all(
        tf.greater(lowest_score_of_fininshed_in_finished,
                   lower_bound_alive_scores))

    return tf.logical_and(
        tf.less(i, decode_length), tf.logical_not(bound_is_met))

  (_, alive_seq, alive_log_probs, finished_seq, finished_scores,
   finished_flags, _) = tf.while_loop(
       _is_finished,
       inner_loop, [
           tf.constant(0), alive_seq, alive_log_probs, finished_seq,
           finished_scores, finished_flags, states
       ],
       shape_invariants=[
           tf.TensorShape([]),
           tf.TensorShape([None, None, None]),
           alive_log_probs.get_shape(),
           tf.TensorShape([None, None, None]),
           finished_scores.get_shape(),
           finished_flags.get_shape(),
           nest.map_structure(get_state_shape_invariants, states),
       ],
       parallel_iterations=1,
       back_prop=False)

  alive_seq.set_shape((None, beam_size, None))
  finished_seq.set_shape((None, beam_size, None))

  # Accounting for corner case: It's possible that no sequence in alive for a
  # particular batch item ever reached EOS. In that case, we should just copy
  # the contents of alive for that batch item. tf.reduce_any(finished_flags, 1)
  # if 0, means that no sequence for that batch index had reached EOS. We need
  # to do the same for the scores as well.
  finished_seq = tf.where(
      tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)
  finished_scores = tf.where(
      tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)
  return finished_seq, finished_scores
