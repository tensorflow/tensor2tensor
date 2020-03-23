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

"""Decoding utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import operator
import os
import re
import string
import time

import numpy as np
import six

from six.moves import input  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import problem as problem_lib
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import contrib
from tensor2tensor.utils import hparam
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

# Number of samples to draw for an image input (in such cases as captioning)
IMAGE_DECODE_LENGTH = 100


def decode_hparams(overrides=""):
  """Hyperparameters for decoding."""
  hp = hparam.HParams(
      save_images=False,
      log_results=True,
      extra_length=100,
      min_length_ratio=0.0,
      batch_size=0,
      beam_size=4,
      alpha=0.6,
      eos_penalty=0.0,
      block_size=0,
      guess_and_check_top_k=0,
      guess_and_check_epsilon=-1,
      insertion_parallel=False,
      return_beams=False,
      write_beam_scores=False,
      max_input_size=-1,
      identity_output=False,
      num_samples=-1,  # Number of examples to decode.
      delimiter="\n",
      decode_to_file="",  # str. Prefix for filename to write decodings to.
      decode_reference="",  # str. Filename to read references from.
      decode_in_memory=False,
      # How much decode should wait for the next checkpoint
      decode_timeout_mins=240,
      summaries_log_dir="decode",  # Directory to write hook summaries.
      shards=1,    # How many shards of data to decode (treating 1 as None).
      shard_id=0,  # Which shard are we decoding if more than 1 above.
      shards_start_offset=0,  # Number of the first shard to decode.
      shard_google_format=False,  # If True use Google shard naming format.
      num_decodes=1,  # Number of times to go over the dataset.
      force_decode_length=False,
      display_decoded_images=False,
      # Multi-problem decoding task id.
      multiproblem_task_id=-1,
      # Used for video decoding.
      frames_per_second=10,
      skip_eos_postprocess=False,
      # Creates a blue/red border covering border_percent of the frame.
      border_percent=2,
      # Maximum number of videos displayed.
      # number of videos displayed = max_display_outputs * max_display_decodes
      max_display_outputs=10,
      max_display_decodes=5,
      # Used in computation of VGG feature based video metrics.
      # Set this to be the path to a trained VGG ckpt to output
      # useful metrics.
      vgg_ckpt_path="",
      # Used for MLPerf compliance logging.
      mlperf_decode_step=0.0,
      mlperf_threshold=25.0,
      mlperf_success=False,
      # A comma-delimited list of additional infer() outputs to be exported.
      export_extra_infer_outputs="")
  hp.parse(overrides)
  return hp


def log_decode_results(inputs,
                       outputs,
                       problem_name,
                       prediction_idx,
                       inputs_vocab,
                       targets_vocab,
                       targets=None,
                       save_images=False,
                       output_dir=None,
                       identity_output=False,
                       log_results=True,
                       skip_eos_postprocess=False):
  """Log inference results."""

  # TODO(lukaszkaiser) refactor this into feature_encoder
  is_video = "video" in problem_name or "gym" in problem_name
  if is_video:
    def fix_and_save_video(vid, prefix):
      save_path_template = os.path.join(
          output_dir,
          "%s_%s_%05d_{:05d}.png" % (problem_name, prefix, prediction_idx))
      # this is only required for predictions
      if vid.shape[-1] == 1:
        vid = np.squeeze(vid, axis=-1)
      save_video(vid, save_path_template)
    tf.logging.info("Saving video: {}".format(prediction_idx))
    fix_and_save_video(inputs, "inputs")
    fix_and_save_video(outputs, "outputs")
    fix_and_save_video(targets, "targets")

  is_image = "image" in problem_name
  is_text2class = isinstance(registry.problem(problem_name),
                             text_problems.Text2ClassProblem)
  skip_eos_postprocess = is_image or is_text2class or skip_eos_postprocess

  decoded_inputs = None
  if is_image and save_images:
    save_path = os.path.join(
        output_dir, "%s_prediction_%d.jpg" % (problem_name, prediction_idx))
    show_and_save_image(inputs / 255., save_path)
  elif inputs is not None and inputs_vocab:
    if identity_output:
      decoded_inputs = " ".join(map(str, inputs.flatten()))
    else:
      decoded_inputs = inputs_vocab.decode(_save_until_eos(
          inputs, skip_eos_postprocess))

    if log_results and not is_video:
      tf.logging.info("Inference results INPUT: %s" % decoded_inputs)

  decoded_targets = None
  decoded_outputs = None
  if identity_output:
    decoded_outputs = " ".join(map(str, outputs.flatten()))
    if targets is not None:
      decoded_targets = " ".join(map(str, targets.flatten()))
  else:
    decoded_outputs = targets_vocab.decode(_save_until_eos(
        outputs, skip_eos_postprocess))
    if targets is not None and log_results:
      decoded_targets = targets_vocab.decode(_save_until_eos(
          targets, skip_eos_postprocess))
  if log_results and not is_video:
    tf.logging.info("Inference results OUTPUT: %s" % decoded_outputs)
  if targets is not None and log_results and not is_video:
    tf.logging.info("Inference results TARGET: %s" % decoded_targets)
  return decoded_inputs, decoded_outputs, decoded_targets


def decode_from_dataset(estimator,
                        problem_name,
                        hparams,
                        decode_hp,
                        decode_to_file=None,
                        dataset_split=None,
                        checkpoint_path=None):
  """Perform decoding from dataset."""
  tf.logging.info("Performing local inference from dataset for %s.",
                  str(problem_name))
  # We assume that worker_id corresponds to shard number.
  shard = decode_hp.shard_id if decode_hp.shards > 1 else None

  # Setup output directory for any artifacts that may be written out.
  output_dir = os.path.join(estimator.model_dir, "decode")
  tf.gfile.MakeDirs(output_dir)

  # If decode_hp.batch_size is specified, use a fixed batch size
  if decode_hp.batch_size:
    hparams.batch_size = decode_hp.batch_size
    hparams.use_fixed_batch_size = True

  dataset_kwargs = {
      "shard": shard,
      "dataset_split": dataset_split,
      "max_records": decode_hp.num_samples
  }

  # Build the inference input function
  problem = hparams.problem
  infer_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.PREDICT, hparams, dataset_kwargs=dataset_kwargs)

  predictions, output_dirs = [], []
  for decode_id in range(decode_hp.num_decodes):
    tf.logging.info("Decoding {}".format(decode_id))

    # Create decode directory if not in-memory decoding.
    if not decode_hp.decode_in_memory:
      output_dir = os.path.join(estimator.model_dir, "decode_%05d" % decode_id)
      tf.gfile.MakeDirs(output_dir)
      output_dirs.append(output_dir)

    result = decode_once(estimator,
                         problem_name,
                         hparams,
                         infer_input_fn,
                         decode_hp,
                         decode_to_file,
                         output_dir,
                         log_results=decode_hp.log_results,
                         checkpoint_path=checkpoint_path)

    if decode_hp.decode_in_memory:
      output_dirs = [output_dir]
      predictions.append(result)

  if decode_hp.decode_to_file:
    decode_hp.decode_to_file = _decode_filename(
        decode_hp.decode_to_file, problem_name, decode_hp)

  run_postdecode_hooks(DecodeHookArgs(
      estimator=estimator,
      problem=problem,
      output_dirs=output_dirs,
      hparams=hparams,
      decode_hparams=decode_hp,
      predictions=predictions
  ), dataset_split)
  return predictions


def decode_once(estimator,
                problem_name,
                hparams,
                infer_input_fn,
                decode_hp,
                decode_to_file,
                output_dir,
                log_results=True,
                checkpoint_path=None):
  """Decodes once.

  Args:
    estimator: tf.estimator.Estimator instance. Used to generate encoded
      predictions.
    problem_name: str. Name of problem.
    hparams: HParams instance. HParams for model training.
    infer_input_fn: zero-arg function. Input function for estimator.
    decode_hp: HParams instance. See decode_hparams() above.
    decode_to_file: str. Prefix for filenames. Used to generated filenames to
      which decoded predictions are written.
    output_dir: str. Output directory. Only used for writing images.
    log_results: bool. If False, return encoded predictions without any
      further processing.
    checkpoint_path: str. Path to load model checkpoint from. If unspecified,
      Estimator's default is used.

  Returns:
    If decode_hp.decode_in_memory is True:
      List of dicts, one per example. Values are either numpy arrays or decoded
      strings.
    If decode_hp.decode_in_memory is False:
      An empty list.
  """

  # Get the predictions as an iterable
  predictions = estimator.predict(infer_input_fn,
                                  checkpoint_path=checkpoint_path)

  if not log_results:
    return list(predictions)

  # Prepare output file writers if decode_to_file passed
  decode_to_file = decode_to_file or decode_hp.decode_to_file
  if decode_to_file:
    output_filepath = _decode_filename(decode_to_file, problem_name, decode_hp)
    parts = output_filepath.split(".")
    parts[-1] = "targets"
    target_filepath = ".".join(parts)
    parts[-1] = "inputs"
    input_filepath = ".".join(parts)

    output_file = tf.gfile.Open(output_filepath, "w")
    target_file = tf.gfile.Open(target_filepath, "w")
    input_file = tf.gfile.Open(input_filepath, "w")

  problem_hparams = hparams.problem_hparams
  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  has_input = "inputs" in problem_hparams.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = problem_hparams.vocabulary[inputs_vocab_key]
  targets_vocab = problem_hparams.vocabulary["targets"]

  num_eval_samples = 0

  # all_outputs[i][j] = (input: str, output: str, target: str). Input,
  # decoded output, and target strings for example i, beam rank j.
  all_outputs = []
  for num_predictions, prediction in enumerate(predictions):
    num_eval_samples += 1
    num_predictions += 1
    inputs = prediction.get("inputs")
    targets = prediction.get("targets")
    outputs = prediction.get("outputs")

    # Log predictions
    decoded_outputs = []  # [(str, str, str)]. See all_outputs above.
    if decode_hp.decode_in_memory:
      all_outputs.append(decoded_outputs)
    decoded_scores = []

    if decode_hp.return_beams:
      output_beams = np.split(outputs, decode_hp.beam_size, axis=0)
      scores = None
      if "scores" in prediction:
        scores = np.split(prediction["scores"], decode_hp.beam_size, axis=0)
      for i, beam in enumerate(output_beams):
        tf.logging.info("BEAM %d:" % i)
        score = scores and scores[i]
        decoded = log_decode_results(
            inputs,
            beam,
            problem_name,
            num_predictions,
            inputs_vocab,
            targets_vocab,
            save_images=decode_hp.save_images,
            output_dir=output_dir,
            identity_output=decode_hp.identity_output,
            targets=targets,
            log_results=log_results)
        decoded_outputs.append(decoded)
        if decode_hp.write_beam_scores:
          decoded_scores.append(score)
    else:
      decoded = log_decode_results(
          inputs,
          outputs,
          problem_name,
          num_predictions,
          inputs_vocab,
          targets_vocab,
          save_images=decode_hp.save_images,
          output_dir=output_dir,
          identity_output=decode_hp.identity_output,
          targets=targets,
          log_results=log_results,
          skip_eos_postprocess=decode_hp.skip_eos_postprocess)
      decoded_outputs.append(decoded)

    # Write out predictions if decode_to_file passed
    if decode_to_file:
      for i, (d_input, d_output, d_target) in enumerate(decoded_outputs):
        # Skip if all padding
        if d_input and re.match("^({})+$".format(text_encoder.PAD), d_input):
          continue
        beam_score_str = ""
        if decode_hp.write_beam_scores:
          beam_score_str = "\t%.2f" % decoded_scores[i]
        output_file.write(str(d_output) + beam_score_str + decode_hp.delimiter)
        target_file.write(str(d_target) + decode_hp.delimiter)
        input_file.write(str(d_input) + decode_hp.delimiter)

    if (decode_hp.num_samples >= 0 and
        num_predictions >= decode_hp.num_samples):
      break

  mlperf_log.transformer_print(key=mlperf_log.EVAL_SIZE,
                               value=num_eval_samples,
                               hparams=hparams)

  if decode_to_file:
    output_file.close()
    target_file.close()
    input_file.close()

  return all_outputs


def decode_from_file(estimator,
                     filename,
                     hparams,
                     decode_hp,
                     decode_to_file=None,
                     checkpoint_path=None):
  """Compute predictions on entries in filename and write them out."""
  if not decode_hp.batch_size:
    decode_hp.batch_size = 32
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)

  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  p_hp = hparams.problem_hparams
  has_input = "inputs" in p_hp.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  targets_vocab = p_hp.vocabulary["targets"]
  problem_name = FLAGS.problem
  filename = _add_shard_to_filename(filename, decode_hp)
  tf.logging.info("Performing decoding from file (%s)." % filename)
  if has_input:
    sorted_inputs, sorted_keys = _get_sorted_inputs(
        filename, decode_hp.delimiter)
  else:
    sorted_inputs = _get_language_modeling_inputs(
        filename, decode_hp.delimiter, repeat=decode_hp.num_decodes)
    sorted_keys = range(len(sorted_inputs))
  num_sentences = len(sorted_inputs)
  num_decode_batches = (num_sentences - 1) // decode_hp.batch_size + 1

  if estimator.config.use_tpu:
    length = getattr(hparams, "length", 0) or hparams.max_length
    batch_ids = []
    for line in sorted_inputs:
      if has_input:
        ids = inputs_vocab.encode(line.strip()) + [1]
      else:
        ids = targets_vocab.encode(line)
      if len(ids) < length:
        ids.extend([0] * (length - len(ids)))
      else:
        ids = ids[:length]
      batch_ids.append(ids)
    np_ids = np.array(batch_ids, dtype=np.int32)
    def input_fn(params):
      batch_size = params["batch_size"]
      dataset = tf.data.Dataset.from_tensor_slices({"inputs": np_ids})
      dataset = dataset.map(
          lambda ex: {"inputs": tf.reshape(ex["inputs"], (length, 1, 1))})
      dataset = dataset.batch(batch_size)
      return dataset
  else:
    def input_fn():
      input_gen = _decode_batch_input_fn(
          num_decode_batches, sorted_inputs,
          inputs_vocab, decode_hp.batch_size,
          decode_hp.max_input_size,
          task_id=decode_hp.multiproblem_task_id, has_input=has_input)
      gen_fn = make_input_fn_from_generator(input_gen)
      example = gen_fn()
      return _decode_input_tensor_to_features_dict(example, hparams, decode_hp)
  decodes = []
  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)

  start_time = time.time()
  total_time_per_step = 0
  total_cnt = 0

  def timer(gen):
    while True:
      try:
        start_time = time.time()
        item = next(gen)
        elapsed_time = time.time() - start_time
        yield elapsed_time, item
      except StopIteration:
        break

  for elapsed_time, result in timer(result_iter):
    if decode_hp.return_beams:
      beam_decodes = []
      beam_scores = []
      output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
      scores = None
      if "scores" in result:
        if np.isscalar(result["scores"]):
          result["scores"] = result["scores"].reshape(1)
        scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
      for k, beam in enumerate(output_beams):
        tf.logging.info("BEAM %d:" % k)
        score = scores and scores[k]
        _, decoded_outputs, _ = log_decode_results(
            result["inputs"],
            beam,
            problem_name,
            None,
            inputs_vocab,
            targets_vocab,
            log_results=decode_hp.log_results,
            skip_eos_postprocess=decode_hp.skip_eos_postprocess)
        beam_decodes.append(decoded_outputs)
        if decode_hp.write_beam_scores:
          beam_scores.append(score)
      if decode_hp.write_beam_scores:
        decodes.append("\t".join([
            "\t".join([d, "%.2f" % s])
            for d, s in zip(beam_decodes, beam_scores)
        ]))
      else:
        decodes.append("\t".join(beam_decodes))
    else:
      _, decoded_outputs, _ = log_decode_results(
          result["inputs"],
          result["outputs"],
          problem_name,
          None,
          inputs_vocab,
          targets_vocab,
          log_results=decode_hp.log_results,
          skip_eos_postprocess=decode_hp.skip_eos_postprocess)
      decodes.append(decoded_outputs)
    total_time_per_step += elapsed_time
    total_cnt += result["outputs"].shape[-1]
  duration = time.time() - start_time
  tf.logging.info("Elapsed Time: %5.5f" % duration)
  tf.logging.info("Averaged Single Token Generation Time: %5.7f "
                  "(time %5.7f count %d)" %
                  (total_time_per_step / total_cnt,
                   total_time_per_step, total_cnt))
  if decode_hp.batch_size == 1:
    tf.logging.info("Inference time %.4f seconds "
                    "(Latency = %.4f ms/setences)" %
                    (duration, 1000.0*duration/num_sentences))
  else:
    tf.logging.info("Inference time %.4f seconds "
                    "(Throughput = %.4f sentences/second)" %
                    (duration, num_sentences/duration))

  # If decode_to_file was provided use it as the output filename without change
  # (except for adding shard_id if using more shards for decoding).
  # Otherwise, use the input filename plus model, hp, problem, beam, alpha.
  decode_filename = decode_to_file if decode_to_file else filename
  if not decode_to_file:
    decode_filename = _decode_filename(decode_filename, problem_name, decode_hp)
  else:
    decode_filename = _add_shard_to_filename(decode_filename, decode_hp)
  tf.logging.info("Writing decodes into %s" % decode_filename)
  outfile = tf.gfile.Open(decode_filename, "w")
  for index in range(len(sorted_inputs)):
    outfile.write("%s%s" % (decodes[sorted_keys[index]], decode_hp.delimiter))
  outfile.flush()
  outfile.close()

  output_dir = os.path.join(estimator.model_dir, "decode")
  tf.gfile.MakeDirs(output_dir)

  run_postdecode_hooks(DecodeHookArgs(
      estimator=estimator,
      problem=hparams.problem,
      output_dirs=[output_dir],
      hparams=hparams,
      decode_hparams=decode_hp,
      predictions=list(result_iter)
  ), None)


def _add_shard_to_filename(filename, decode_hp):
  if decode_hp.shards > 1:
    shard_id = decode_hp.shard_id + decode_hp.shards_start_offset
    if decode_hp.shard_google_format:
      filename = filename + "-{0:05d}-of-{1:05d}".format(shard_id,
                                                         decode_hp.shards)
    else:
      filename = filename + ("%.3d" % shard_id)
  return filename


def _decode_filename(base_filename, problem_name, decode_hp):
  """Generates decode filename.

  Args:
    base_filename: A string, base of the decode filename.
    problem_name: A string, name of the problem.
    decode_hp: HParams for decoding.

  Returns:
    A string, produced decode filename.
  """
  if decode_hp.shards > 1:
    base_filename = _add_shard_to_filename(base_filename, decode_hp)
  if ("beam{beam}.alpha{alpha}.decodes".format(
      beam=str(decode_hp.beam_size), alpha=str(decode_hp.alpha))
      in base_filename):
    return base_filename
  else:
    return (
        "{base}.{model}.{hp}.{problem}.beam{beam}.alpha{alpha}.decodes".format(
            base=base_filename,
            model=FLAGS.model,
            hp=FLAGS.hparams_set,
            problem=problem_name,
            beam=str(decode_hp.beam_size),
            alpha=str(decode_hp.alpha)))


def make_input_fn_from_generator(gen):
  """Use py_func to yield elements from the given generator."""
  first_ex = six.next(gen)
  flattened = contrib.framework().nest.flatten(first_ex)
  types = [t.dtype for t in flattened]
  shapes = [[None] * len(t.shape) for t in flattened]
  first_ex_list = [first_ex]

  def py_func():
    if first_ex_list:
      example = first_ex_list.pop()
    else:
      example = six.next(gen)
    return contrib.framework().nest.flatten(example)

  def input_fn():
    flat_example = tf.py_func(py_func, [], types)
    _ = [t.set_shape(shape) for t, shape in zip(flat_example, shapes)]
    example = contrib.framework().nest.pack_sequence_as(first_ex, flat_example)
    return example

  return input_fn


def decode_interactively(estimator, hparams, decode_hp, checkpoint_path=None):
  """Interactive decoding."""

  is_image = "image" in hparams.problem.name
  is_text2class = isinstance(hparams.problem,
                             text_problems.Text2ClassProblem)
  skip_eos_postprocess = (
      is_image or is_text2class or decode_hp.skip_eos_postprocess)

  def input_fn():
    gen_fn = make_input_fn_from_generator(
        _interactive_input_fn(hparams, decode_hp))
    example = gen_fn()
    example = _interactive_input_tensor_to_features_dict(example, hparams)
    return example

  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)
  for result in result_iter:
    targets_vocab = hparams.problem_hparams.vocabulary["targets"]

    if decode_hp.return_beams:
      beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
      scores = None
      if "scores" in result:
        if np.isscalar(result["scores"]):
          result["scores"] = result["scores"].reshape(1)
        scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
      for k, beam in enumerate(beams):
        tf.logging.info("BEAM %d:" % k)
        beam_string = targets_vocab.decode(_save_until_eos(
            beam, skip_eos_postprocess))
        if scores is not None:
          tf.logging.info("\"%s\"\tScore:%f" % (beam_string, scores[k]))
        else:
          tf.logging.info("\"%s\"" % beam_string)
    else:
      if decode_hp.identity_output:
        tf.logging.info(" ".join(map(str, result["outputs"].flatten())))
      else:
        tf.logging.info(
            targets_vocab.decode(_save_until_eos(
                result["outputs"], skip_eos_postprocess)))


def _decode_batch_input_fn(num_decode_batches, sorted_inputs, vocabulary,
                           batch_size, max_input_size,
                           task_id=-1, has_input=True):
  """Generator to produce batches of inputs."""
  tf.logging.info(" batch %d" % num_decode_batches)
  for b in range(num_decode_batches):
    tf.logging.info("Decoding batch %d" % b)
    batch_length = 0
    batch_inputs = []
    for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
      input_ids = vocabulary.encode(inputs)
      if max_input_size > 0:
        # Subtract 1 for the EOS_ID.
        input_ids = input_ids[:max_input_size - 1]
      if has_input or task_id > -1:  # Do not append EOS for pure LM tasks.
        final_id = text_encoder.EOS_ID if task_id < 0 else task_id
        input_ids.append(final_id)
      batch_inputs.append(input_ids)
      if len(input_ids) > batch_length:
        batch_length = len(input_ids)
    final_batch_inputs = []
    for input_ids in batch_inputs:
      assert len(input_ids) <= batch_length
      x = input_ids + [0] * (batch_length - len(input_ids))
      final_batch_inputs.append(x)

    yield {
        "inputs": np.array(final_batch_inputs).astype(np.int32),
    }


def _interactive_input_fn(hparams, decode_hp):
  """Generator that reads from the terminal and yields "interactive inputs".

  Due to temporary limitations in tf.learn, if we don't want to reload the
  whole graph, then we are stuck encoding all of the input as one fixed-size
  numpy array.

  We yield int32 arrays with shape [const_array_size].  The format is:
  [num_samples, decode_length, len(input ids), <input ids>, <padding>]

  Args:
    hparams: model hparams
    decode_hp: decode hparams
  Yields:
    numpy arrays

  Raises:
    Exception: when `input_type` is invalid.
  """
  num_samples = decode_hp.num_samples if decode_hp.num_samples > 0 else 1
  decode_length = decode_hp.extra_length
  input_type = "text"
  p_hparams = hparams.problem_hparams
  has_input = "inputs" in p_hparams.modality
  vocabulary = p_hparams.vocabulary["inputs" if has_input else "targets"]
  # This should be longer than the longest input.
  const_array_size = 10000
  # Import readline if available for command line editing and recall.
  try:
    import readline  # pylint: disable=g-import-not-at-top,unused-variable
  except ImportError:
    pass
  while True:
    prompt = ("INTERACTIVE MODE  num_samples=%d  decode_length=%d  \n"
              "  it=<input_type>     ('text' or 'image' or 'label', default: "
              "text)\n"
              "  ns=<num_samples>    (changes number of samples, default: 1)\n"
              "  dl=<decode_length>  (changes decode length, default: 100)\n"
              "  <%s>                (decode)\n"
              "  q                   (quit)\n"
              ">" % (num_samples, decode_length,
                     "source_string" if has_input else "target_prefix"))
    input_string = input(prompt)
    if input_string == "q":
      return
    elif input_string[:3] == "ns=":
      num_samples = int(input_string[3:])
    elif input_string[:3] == "dl=":
      decode_length = int(input_string[3:])
    elif input_string[:3] == "it=":
      input_type = input_string[3:]
    else:
      if input_type == "text":
        input_ids = vocabulary.encode(input_string)
        if has_input:
          input_ids.append(text_encoder.EOS_ID)
        x = [num_samples, decode_length, len(input_ids)] + input_ids
        assert len(x) < const_array_size
        x += [0] * (const_array_size - len(x))
        features = {
            "inputs": np.array(x).astype(np.int32),
        }
      elif input_type == "image":
        input_path = input_string
        img = vocabulary.encode(input_path)
        features = {
            "inputs": img.astype(np.int32),
        }
      elif input_type == "label":
        input_ids = [int(input_string)]
        x = [num_samples, decode_length, len(input_ids)] + input_ids
        features = {
            "inputs": np.array(x).astype(np.int32),
        }
      else:
        raise Exception("Unsupported input type.")
      for k, v in six.iteritems(
          problem_lib.problem_hparams_to_features(p_hparams)):
        features[k] = np.array(v).astype(np.int32)
      yield features


def save_video(video, save_path_template):
  """Save frames of the videos into files."""
  try:
    from PIL import Image  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    tf.logging.warning(
        "Showing and saving an image requires PIL library to be "
        "installed: %s", e)
    raise NotImplementedError("Image display and save not implemented.")

  for i, frame in enumerate(video):
    save_path = save_path_template.format(i)
    with tf.gfile.Open(save_path, "wb") as sp:
      Image.fromarray(np.uint8(frame)).save(sp)


def show_and_save_image(img, save_path):
  """Shows an image using matplotlib and saves it."""
  try:
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    tf.logging.warning(
        "Showing and saving an image requires matplotlib to be "
        "installed: %s", e)
    raise NotImplementedError("Image display and save not implemented.")
  plt.imshow(img)
  with tf.gfile.Open(save_path, "wb") as sp:
    plt.savefig(sp)


def _get_language_modeling_inputs(filename,
                                  delimiter="\n",
                                  repeat=1,
                                  append_space_to_final_punctionation=True):
  """Read a file of partial texts to continue.

  The purpose of append_space_to_final_punctionation is that SubwordTokenizer
  groups punctuation and the ensuing space in the same token.  Adding a space
  causes the token to be completed.

  Args:
    filename: a string
    delimiter: a string
    repeat: an integer - we repeat the entire file that many times.
    append_space_to_final_punctionation: a boolean

  Returns:
    a list of strings
  """
  with tf.gfile.Open(filename) as f:
    text = f.read()
  inputs = text.split(delimiter)
  if not inputs[-1]:
    inputs.pop()
  inputs *= repeat
  if append_space_to_final_punctionation:
    inputs = [
        s + " " if s and s[-1] in string.punctuation else s for s in inputs]
  return inputs


def _get_sorted_inputs(filename, delimiter="\n"):
  """Returning inputs sorted according to decreasing length.

  This causes inputs of similar lengths to be processed in the same batch,
  facilitating early stopping for short sequences.

  Longer sequences are sorted first so that if you're going to get OOMs,
  you'll see it in the first batch.

  Args:
    filename: path to file with inputs, 1 per line.
    delimiter: str, delimits records in the file.

  Returns:
    a sorted list of inputs

  """
  tf.logging.info("Getting sorted inputs")
  with tf.gfile.Open(filename) as f:
    text = f.read()
    records = text.split(delimiter)
    inputs = [record.strip() for record in records]
    # Strip the last empty line.
    if not inputs[-1]:
      inputs.pop()
  input_lens = [(i, -len(line.split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
  # We'll need the keys to rearrange the inputs back into their original order
  sorted_keys = {}
  sorted_inputs = []
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys


def _save_until_eos(ids, skip=False):
  """Strips everything after the first <EOS> token, which is normally 1."""
  ids = ids.flatten()
  if skip:
    return ids
  try:
    index = list(ids).index(text_encoder.EOS_ID)
    return ids[0:index]
  except ValueError:
    # No EOS_ID: return the array as-is.
    return ids


def _interactive_input_tensor_to_features_dict(feature_map, hparams):
  """Convert the interactive input format (see above) to a dictionary.

  Args:
    feature_map: dict with inputs.
    hparams: model hyperparameters

  Returns:
    a features dictionary, as expected by the decoder.
  """
  inputs = tf.convert_to_tensor(feature_map["inputs"])
  input_is_image = False if len(inputs.get_shape()) < 3 else True

  x = inputs
  if input_is_image:
    x = tf.image.resize_images(x, [299, 299])
    x = tf.reshape(x, [1, 299, 299, -1])
    x = tf.to_int32(x)
  else:
    # Remove the batch dimension.
    num_samples = x[0]
    length = x[2]
    x = tf.slice(x, [3], tf.to_int32([length]))
    x = tf.reshape(x, [1, -1, 1, 1])
    # Transform into a batch of size num_samples to get that many random
    # decodes.
    x = tf.tile(x, tf.to_int32([num_samples, 1, 1, 1]))

  p_hparams = hparams.problem_hparams
  input_space_id = tf.constant(p_hparams.input_space_id)
  target_space_id = tf.constant(p_hparams.target_space_id)

  features = {}
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = (
      IMAGE_DECODE_LENGTH if input_is_image else inputs[1])
  features["inputs"] = x
  # Save inputs to "partial_targets" when prepending inputs to targets. Also
  # keep "inputs" as some models crash if they don't exist.
  if getattr(hparams, "prepend_mode", "none") != "none":
    shape = tf.shape(x)
    partial_targets = tf.reshape(x, [shape[0], shape[1]])
    partial_targets = tf.pad(partial_targets, [[0, 0], [0, 1]])
    features["partial_targets"] = partial_targets
  return features


def _decode_input_tensor_to_features_dict(feature_map, hparams, decode_hp):
  """Convert the interactive input format (see above) to a dictionary.

  Args:
    feature_map: dict with inputs.
    hparams: model hyperparameters
    decode_hp: decode hyperparameters

  Returns:
    a features dictionary, as expected by the decoder.
  """
  inputs = tf.convert_to_tensor(feature_map["inputs"])
  input_is_image = False

  x = inputs
  p_hparams = hparams.problem_hparams
  # Add a third empty dimension
  x = tf.expand_dims(x, axis=[2])
  x = tf.to_int32(x)
  input_space_id = tf.constant(p_hparams.input_space_id)
  target_space_id = tf.constant(p_hparams.target_space_id)

  features = {}
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = (
      IMAGE_DECODE_LENGTH if input_is_image else
      tf.constant(decode_hp.extra_length))
  features["inputs"] = x
  # Save inputs to "partial_targets" when prepending inputs to targets. Also
  # keep "inputs" as some models crash if they don't exist.
  if getattr(hparams, "prepend_mode", "none") != "none":
    shape = tf.shape(x)
    partial_targets = tf.reshape(x, [shape[0], shape[1]])
    partial_targets = tf.pad(partial_targets, [[0, 0], [0, 1]])
    features["partial_targets"] = partial_targets
  return features


def get_step_from_ckpt_path(path):
  return int(os.path.basename(path).split("-")[-1])


def latest_checkpoint_step(ckpt_dir):
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if not ckpt:
    return None
  path = ckpt.model_checkpoint_path
  return get_step_from_ckpt_path(path)


class DecodeHookArgs(collections.namedtuple(
    "DecodeHookArgs",
    ["estimator", "problem", "output_dirs", "hparams",
     "decode_hparams", "predictions"])):
  pass


def run_postdecode_hooks(decode_hook_args, dataset_split):
  """Run hooks after decodes have run."""
  hooks = decode_hook_args.problem.decode_hooks
  if not hooks:
    return
  global_step = latest_checkpoint_step(decode_hook_args.estimator.model_dir)
  if global_step is None:
    tf.logging.info(
        "Skipping decode hooks because no checkpoint yet available.")
    return
  tf.logging.info("Running decode hooks.")
  parent_dir = os.path.join(decode_hook_args.output_dirs[0], os.pardir)
  child_dir = decode_hook_args.decode_hparams.summaries_log_dir
  if dataset_split is not None:
    child_dir += "_{}".format(dataset_split)
  final_dir = os.path.join(parent_dir, child_dir)
  summary_writer = tf.summary.FileWriter(final_dir)

  for hook in hooks:
    # Isolate each hook in case it creates TF ops
    with tf.Graph().as_default():
      summaries = hook(decode_hook_args)
    if summaries:
      summary = tf.Summary(value=list(summaries))
      summary_writer.add_summary(summary, global_step)
  summary_writer.close()
  tf.logging.info("Decode hooks done.")
