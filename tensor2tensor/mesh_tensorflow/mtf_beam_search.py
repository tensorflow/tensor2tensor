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
"""Implementation of beam search with penalties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
import tensorflow as tf

# Assuming EOS_ID is 1
EOS_ID = 1
# Default value for INF
INF = 1. * 1e7


def compute_topk_scores_and_seq(sequences, scores, scores_to_gather, flags,
                                beam_dim, prefix="default",
                                states=None):
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
      the length penalized scores.
    flags: Tensor of bools for sequences that say whether a sequence has reached
      EOS or not
    beam_dim: mtf.Dimension
    prefix: an optional string
    states: an optional list of mtf.Tensor
  Returns:
    Tuple of
    (topk_seq [batch_size, beam_size, decode_length],
     topk_gathered_scores [batch_size, beam_size],
     topk_finished_flags[batch_size, beam_size],
     topk_gathered_states)
  """
  unused_batch_dim, old_beam_dim, unused_length_dim = sequences.shape.dims
  topk_indices, _ = mtf.top_k(scores, old_beam_dim, beam_dim)

  # Gather up the highest scoring sequences.
  # For each operation added, give it
  # a concrete name to simplify observing these operations with tfdbg.
  # Clients can capture these tensors by watching these node names.
  def gather(tensor, name):
    with tf.name_scope(prefix + name):
      output_shape = mtf.Shape(
          [beam_dim if d == old_beam_dim else d for d in tensor.shape.dims])
      return mtf.gather(
          tensor, topk_indices, old_beam_dim, output_shape=output_shape)
  topk_seq = gather(sequences, "_seq")
  topk_flags = gather(flags, "_flags")
  topk_gathered_scores = gather(scores_to_gather, "_scores")
  if states is None:
    topk_gathered_states = None
  else:
    topk_gathered_states = [gather(state, "_topk_states") for state in states]
  return topk_seq, topk_gathered_scores, topk_flags, topk_gathered_states


def beam_search(logits_fn,
                initial_ids,
                alpha,
                states=None,
                eos_id=EOS_ID,
                stop_early=True,
                decode_length=None,
                use_tpu=True):
  """Beam search with length penalties.

  Requires a function that can take the currently decoded symbols and return
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
  available (i.e. set to None) inside logits_fn.

  Args:
    logits_fn: Interface to the model, to provide logits.
        Shoud take:
          step_num - mtf Scalar
          ids - mtf Tensor with shape [batch, beam, length]
        Should return:
          logits - [batch, beam, vocab_size]
    initial_ids: a mtf.Tensor with shape [batch_dim, beam_dim, length_dim])
    alpha: alpha for length penalty.
    states: list of mtf.Tensor
    eos_id: ID for end of sentence.
    stop_early: a boolean - stop once best sequence is provably determined.
    decode_length: a mtf Scalar of dtype tf.int32 - maximum length of decodes
    use_tpu: a boolean
  Returns:
    Tuple of
    (decoded beams [batch, beam, length]
     decoding probabilities [batch, beam_size])
  """
  batch_dim, beam_dim, length_dim = initial_ids.shape.dims
  mesh = initial_ids.mesh

  batch_by_beam = mtf.Shape([batch_dim, beam_dim])
  initial_log_probs = mtf.broadcast(
      mtf.one_hot(
          mtf.constant(mesh, 0, dtype=tf.int32),
          beam_dim,
          on_value=0.0,
          off_value=-INF),
      batch_by_beam)

  length_scalar = mtf.constant(mesh, length_dim.size, dtype=tf.int32)
  if decode_length is None:
    decode_length = length_scalar
  else:
    decode_length = mtf.minimum(decode_length, length_scalar)

  alive_log_probs = initial_log_probs
  alive_seq = initial_ids

  # Finished will keep track of all the sequences that have finished so far
  # Finished log probs will be negative infinity in the beginning
  # finished_flags will keep track of booleans
  finished_seq = initial_ids
  finished_scores = mtf.constant(mesh, -INF, batch_by_beam)

  # Setting the scores of the initial to negative infinity.
  finished_flags = mtf.constant(mesh, False, batch_by_beam, tf.bool)

  def grow_finished(finished_seq, finished_scores, finished_flags, curr_seq,
                    curr_scores, curr_finished):
    """Given sequences and scores, will gather the top k=beam size sequences.

    Args:
      finished_seq: Current finished sequences.
        [batch, beam, length]
      finished_scores: scores for each of these sequences.
        [batch, beam]
      finished_flags: finished bools for each of these sequences.
        [batch, beam]
      curr_seq: current topk sequence that has been grown by one position.
        [batch, beam, length]
      curr_scores: scores for each of these sequences. [batch, beam]
      curr_finished: Finished flags for each of these sequences.
        [batch, beam]
    Returns:
      Tuple of
        (Topk sequences based on scores,
         log probs of these sequences,
         Finished flags of these sequences,
         None (no states))
    """

    # Set the scores of the unfinished seq in curr_seq to large negative
    # values
    curr_scores += (1. - mtf.to_float(curr_finished)) * -INF
    unused_batch_dim, beam_dim, unused_length_dim = finished_seq.shape.dims
    # concatenating the sequences and scores along beam axis
    def _my_concat(a, b):
      a = mtf.rename_dimension(a, "beam", "triple_beam")
      b = mtf.rename_dimension(b, "double_beam", "triple_beam")
      return mtf.concat([a, b], "triple_beam")

    curr_finished_seq = _my_concat(finished_seq, curr_seq)
    curr_finished_scores = _my_concat(finished_scores, curr_scores)
    curr_finished_flags = _my_concat(finished_flags, curr_finished)
    return compute_topk_scores_and_seq(
        curr_finished_seq, curr_finished_scores, curr_finished_scores,
        curr_finished_flags, beam_dim, "grow_finished", states=None)

  def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished, states):
    """Given sequences and scores, will gather the top k=beam size sequences.

    Args:
      curr_seq: current topk sequence that has been grown by one position.
        [batch, beam, length]
      curr_scores: scores for each of these sequences. [batch_size, beam_size]
      curr_log_probs: log probs for each of these sequences.
        [batch, beam]
      curr_finished: Finished flags for each of these sequences.
        [batch, beam]
      states: list of mtf.Tensor
    Returns:
      Tuple of
        (Topk sequences based on scores,
         log probs of these sequences,
         Finished flags of these sequences)
    """
    # Set the scores of the finished seq in curr_seq to large negative
    # values
    curr_scores += mtf.to_float(curr_finished) * -INF
    return compute_topk_scores_and_seq(curr_seq, curr_scores, curr_log_probs,
                                       curr_finished, beam_dim,
                                       "grow_alive", states)

  def grow_topk(i, alive_seq, alive_log_probs, states=None):
    r"""Inner beam search loop.

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
      alive_seq: Topk sequences decoded so far [batch, beam, length]
      alive_log_probs: probabilities of these sequences. [batch, beam]
      states: optional list of mtf.Tensor
    Returns:
      Tuple of
        (Topk sequences extended by the next word,
         The log probs of these sequences,
         The scores with length penalty of these sequences,
         Flags indicating which of these sequences have finished decoding,
         list of transformed decoding states)
    """
    logits, new_states = logits_fn(i, alive_seq, states)
    batch_dim, beam_dim, vocab_dim = logits.shape.dims

    # Convert logits to normalized log probs
    candidate_log_probs = mtf.log_softmax(logits, vocab_dim)

    # Multiply the probabilities by the current probabilities of the beam.
    # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
    log_probs = candidate_log_probs + alive_log_probs

    length_penalty = mtf.pow(((5. + mtf.to_float(i + 1)) / 6.), alpha)

    curr_scores = log_probs / length_penalty

    # scores have shape [batch, beam, vocab]
    beam_and_vocab_dim = mtf.Dimension(
        "beam_and_vocab", beam_dim.size * vocab_dim.size)
    flat_shape = mtf.Shape([batch_dim, beam_and_vocab_dim])
    double_beam = mtf.Dimension("double_beam", beam_dim.size * 2)
    # Flatten out (beam_size, vocab_size) probs in to a list of possibilities
    flat_curr_scores = mtf.reshape(curr_scores, flat_shape)

    top_ids, top_scores = mtf.top_k(
        flat_curr_scores, reduced_dim=beam_and_vocab_dim, new_dim=double_beam)

    # Recovering the log probs because we will need to send them back
    top_log_probs = top_scores * length_penalty

    # Work out what beam the top probs are in.
    top_beam_index = top_ids // vocab_dim.size
    top_ids %= vocab_dim.size  # Unflatten the ids

    def my_gather(tensor):
      return mtf.gather(
          tensor, top_beam_index, beam_dim,
          output_shape=mtf.Shape(
              [double_beam if d == beam_dim else d for d in tensor.shape.dims]))

    # Gather up the most probable 2*beams both for the ids and finished_in_alive
    # bools
    top_seq = my_gather(alive_seq)

    if states:
      states = [my_gather(state) for state in new_states]

    # Append the most probable alive
    top_seq += top_ids * mtf.one_hot(i, length_dim, dtype=tf.int32)
    top_finished = mtf.equal(top_ids, eos_id)

    return top_seq, top_log_probs, top_scores, top_finished, states

  def inner_loop(i, alive_seq, alive_log_probs, finished_seq, finished_scores,
                 finished_flags, *states):
    """Inner beam search loop.

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
      *states: mtf Tensors

    Returns:
      Tuple of
        (Incremented loop index
         New alive sequences,
         Log probs of the alive sequences,
         New finished sequences,
         Scores of the new finished sequences,
         Flags indicating which sequence in finished as reached EOS,
         dict of final decoding states)
    """

    # Each inner loop, we carry out three steps:
    # 1. Get the current topk items.
    # 2. Extract the ones that have finished and haven't finished
    # 3. Recompute the contents of finished based on scores.
    (top2k_seq, top2k_log_probs, top2k_scores, top2k_finished,
     top2k_states) = grow_topk(i, alive_seq, alive_log_probs, states)
    alive_seq, alive_log_probs, _, states = grow_alive(
        top2k_seq, top2k_scores, top2k_log_probs, top2k_finished, top2k_states)
    finished_seq, finished_scores, finished_flags, _ = grow_finished(
        finished_seq, finished_scores, finished_flags, top2k_seq, top2k_scores,
        top2k_finished)
    return (i + 1, alive_seq, alive_log_probs, finished_seq, finished_scores,
            finished_flags) + tuple(states)

  def _is_finished(i, unused_alive_seq, alive_log_probs, unused_finished_seq,
                   finished_scores, finished_in_finished, *unused_states):
    """Checking termination condition.

    We terminate when we decoded up to decode_length or the lowest scoring item
    in finished has a greater score that the highest prob item in alive divided
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
    # TODO(noam): support a different decode length...
    # decode_length = mtf.constant(mesh, length_dim.size, dtype=tf.int32)

    # del alive_log_probs, finished_scores, finished_in_finished
    # return mtf.less(i, length_dim.size)
    if not stop_early:
      return mtf.less(i, decode_length)
    max_length_penalty = mtf.pow(
        ((5. + mtf.to_float(decode_length)) / 6.), alpha)
    # The best possible score of the most likely alive sequence.
    lower_bound_alive_scores = mtf.gather(
        alive_log_probs, mtf.constant(mesh, 0, dtype=tf.int32),
        beam_dim) / max_length_penalty

    # Now to compute the lowest score of a finished sequence in finished
    # If the sequence isn't finished, we multiply it's score by 0. since
    # scores are all -ve, taking the min will give us the score of the lowest
    # finished item.
    lowest_score_of_finished_in_finished = mtf.reduce_min(
        finished_scores * mtf.to_float(finished_in_finished),
        reduced_dim=beam_dim)

    # If none of the sequences have finished, then the min will be 0 and
    # we have to replace it by -ve INF if it is. The score of any seq in alive
    # will be much higher than -ve INF and the termination condition will not
    # be met.
    lowest_score_of_finished_in_finished += (
        (1. - mtf.to_float(mtf.reduce_any(
            finished_in_finished, reduced_dim=beam_dim))) * -INF)

    bound_is_met = mtf.reduce_all(
        mtf.greater(lowest_score_of_finished_in_finished,
                    lower_bound_alive_scores))
    return mtf.logical_and(
        mtf.less(i, decode_length), mtf.logical_not(bound_is_met))

  initial_step_num = mtf.constant(mesh, 0, dtype=tf.int32)
  while_loop_inputs = [
      initial_step_num, alive_seq, alive_log_probs, finished_seq,
      finished_scores, finished_flags] + states

  (_, alive_seq, alive_log_probs, finished_seq, finished_scores,
   finished_flags) = mtf.while_loop(
       _is_finished, inner_loop, while_loop_inputs,
       num_loop_vars=None if use_tpu else 6)[:6]

  # Accounting for corner case: It's possible that no sequence in alive for a
  # particular batch item ever reached EOS. In that case, we should just copy
  # the contents of alive for that batch item. tf.reduce_any(finished_flags, 1)
  # if 0, means that no sequence for that batch index had reached EOS. We need
  # to do the same for the scores as well.
  finished_seq = mtf.where(
      mtf.reduce_any(finished_flags, reduced_dim=beam_dim),
      finished_seq, alive_seq)
  finished_scores = mtf.where(
      mtf.reduce_any(finished_flags, reduced_dim=beam_dim),
      finished_scores, alive_log_probs)
  return finished_seq, finished_scores


def greedy_decode(logits_fn,
                  initial_ids,
                  temperature=0.0,
                  initial_states=None,
                  eos_id=EOS_ID,
                  forced_ids=None,
                  use_tpu=True):
  """Greedy decoding.

  Args:
    logits_fn: Interface to the model, to provide logits.
        Shoud take:
          step_num - mtf Scalar
          ids - mtf Tensor with shape [..., length]
          states - list of mtf.Tensor
        Should return:
          logits - [batch, vocab_size]
          new_states - list of mtf.Tensor
    initial_ids: mtf.Tensor with shape [..., length], containing zeros.
    temperature: a float between 0.0 (argmax) and 1.0 (random)
    initial_states: list of mtf.Tensor
    eos_id: ID for end of sentence.
    forced_ids: optional mtf.Tensor with shape [..., length]
    use_tpu: a boolean
  Returns:
    Tensor with shape [..., length]
  """
  length_dim = initial_ids.shape.dims[-1]
  mesh = initial_ids.mesh
  num_steps = mtf.constant(mesh, length_dim.size, dtype=tf.int32)
  def cond_fn(step_num, prev_ids, *unused_states):
    """Should we run another loop iteration."""
    overflow = mtf.equal(step_num, num_steps)
    has_eos = mtf.reduce_any(
        mtf.equal(prev_ids, eos_id), reduced_dim=length_dim)
    all_has_eos = mtf.reduce_all(has_eos)
    return mtf.logical_not(mtf.logical_or(overflow, all_has_eos))
  def body_fn(step_num, ids, *states):
    """Body function for greedy decoding.

    Args:
      step_num: a mtf.Tensor
      ids: a mtf.Tensor
      *states: additional mtf.Tensors
    Returns:
      new_step_num, new_ids, *new_states
    """
    logits, new_states = logits_fn(step_num, ids, states)
    vocab_dim = logits.shape.dims[-1]
    new_ids = mtf.sample_with_temperature(
        logits, vocab_dim, temperature)
    if forced_ids is not None:
      # force the new ids to equal the partial targets where specified
      # (positions where partial_targets contain nonzero values)
      forced = mtf.gather(forced_ids, step_num, length_dim)
      new_ids = forced + new_ids * mtf.to_int32(mtf.equal(forced, 0))
    ids += new_ids * mtf.one_hot(step_num, length_dim, dtype=tf.int32)
    new_step_num = step_num + 1
    return [new_step_num, ids] + new_states
  initial_step_num = mtf.constant(mesh, 0, dtype=tf.int32)
  while_loop_inputs = [initial_step_num, initial_ids] + initial_states
  final_step_num, mtf_samples = mtf.while_loop(
      cond_fn, body_fn, while_loop_inputs,
      num_loop_vars=None if use_tpu else 2)[:2]
  mtf_samples = mtf.Print(mtf_samples, [final_step_num], "output_length")
  return mtf_samples
