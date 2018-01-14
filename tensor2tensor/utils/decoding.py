# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

import operator
import os

# Dependency imports

import numpy as np
import six

from six.moves import input  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import text_encoder
import tensorflow as tf

FLAGS = tf.flags.FLAGS

# Number of samples to draw for an image input (in such cases as captioning)
IMAGE_DECODE_LENGTH = 100


def decode_hparams(overrides=""):
  """Hyperparameters for decoding."""
  hp = tf.contrib.training.HParams(
      save_images=False,
      problem_idx=0,
      extra_length=50,
      batch_size=0,
      beam_size=4,
      alpha=0.6,
      return_beams=False,
      write_beam_scores=False,
      max_input_size=-1,
      identity_output=False,
      num_samples=-1,
      delimiter="\n")
  hp = hp.parse(overrides)
  return hp


def log_decode_results(inputs,
                       outputs,
                       problem_name,
                       prediction_idx,
                       inputs_vocab,
                       targets_vocab,
                       targets=None,
                       save_images=False,
                       model_dir=None,
                       identity_output=False):
  """Log inference results."""
  is_image = "image" in problem_name
  if is_image and save_images:
    save_path = os.path.join(model_dir, "%s_prediction_%d.jpg" %
                             (problem_name, prediction_idx))
    show_and_save_image(inputs / 255., save_path)
  elif inputs_vocab:
    if identity_output:
      decoded_inputs = " ".join(map(str, inputs.flatten()))
    else:
      decoded_inputs = inputs_vocab.decode(_save_until_eos(inputs, is_image))

    tf.logging.info("Inference results INPUT: %s" % decoded_inputs)

  decoded_targets = None
  if identity_output:
    decoded_outputs = "".join(map(str, outputs.flatten()))
    if targets is not None:
      decoded_targets = "".join(map(str, targets.flatten()))
  else:
    decoded_outputs = targets_vocab.decode(_save_until_eos(outputs, is_image))
    if targets is not None:
      decoded_targets = targets_vocab.decode(_save_until_eos(targets, is_image))

  tf.logging.info("Inference results OUTPUT: %s" % decoded_outputs)
  if targets is not None:
    tf.logging.info("Inference results TARGET: %s" % decoded_targets)
  return decoded_outputs, decoded_targets


def decode_from_dataset(estimator,
                        problem_names,
                        hparams,
                        decode_hp,
                        decode_to_file=None,
                        dataset_split=None):
  """Perform decoding from dataset."""
  tf.logging.info("Performing local inference from dataset for %s.",
                  str(problem_names))
  # We assume that worker_id corresponds to shard number.
  shard = decode_hp.shard_id if decode_hp.shards > 1 else None

  # If decode_hp.batch_size is specified, use a fixed batch size
  if decode_hp.batch_size:
    hparams.batch_size = decode_hp.batch_size
    hparams.use_fixed_batch_size = True

  dataset_kwargs = {
      "shard": shard,
      "dataset_split": dataset_split,
  }

  for problem_idx, problem_name in enumerate(problem_names):
    # Build the inference input function
    problem = hparams.problem_instances[problem_idx]
    infer_input_fn = problem.make_estimator_input_fn(
        tf.estimator.ModeKeys.PREDICT, hparams, dataset_kwargs=dataset_kwargs)

    # Get the predictions as an iterable
    predictions = estimator.predict(infer_input_fn)

    # Prepare output file writers if decode_to_file passed
    if decode_to_file:
      if decode_hp.shards > 1:
        decode_filename = decode_to_file + ("%.2d" % decode_hp.shard_id)
      else:
        decode_filename = decode_to_file
      output_filepath = _decode_filename(decode_filename, problem_name,
                                         decode_hp)
      parts = output_filepath.split(".")
      parts[-1] = "targets"
      target_filepath = ".".join(parts)

      output_file = tf.gfile.Open(output_filepath, "w")
      target_file = tf.gfile.Open(target_filepath, "w")

    problem_hparams = hparams.problems[problem_idx]
    # Inputs vocabulary is set to targets if there are no inputs in the problem,
    # e.g., for language models where the inputs are just a prefix of targets.
    has_input = "inputs" in problem_hparams.vocabulary
    inputs_vocab_key = "inputs" if has_input else "targets"
    inputs_vocab = problem_hparams.vocabulary[inputs_vocab_key]
    targets_vocab = problem_hparams.vocabulary["targets"]
    for num_predictions, prediction in enumerate(predictions):
      num_predictions += 1
      inputs = prediction["inputs"]
      targets = prediction["targets"]
      outputs = prediction["outputs"]

      # Log predictions
      decoded_outputs = []
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
              model_dir=estimator.model_dir,
              identity_output=decode_hp.identity_output,
              targets=targets)
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
            model_dir=estimator.model_dir,
            identity_output=decode_hp.identity_output,
            targets=targets)
        decoded_outputs.append(decoded)

      # Write out predictions if decode_to_file passed
      if decode_to_file:
        for i, (decoded_output, decoded_target) in enumerate(decoded_outputs):
          beam_score_str = ""
          if decode_hp.write_beam_scores:
            beam_score_str = "\t%.2f" % decoded_scores[i]
          output_file.write(
              str(decoded_output) + beam_score_str + decode_hp.delimiter)
          target_file.write(str(decoded_target) + decode_hp.delimiter)

      if (decode_hp.num_samples >= 0 and
          num_predictions >= decode_hp.num_samples):
        break

    if decode_to_file:
      output_file.close()
      target_file.close()

    tf.logging.info("Completed inference on %d samples." % num_predictions)  # pylint: disable=undefined-loop-variable


def decode_from_file(estimator,
                     filename,
                     hparams,
                     decode_hp,
                     decode_to_file=None):
  """Compute predictions on entries in filename and write them out."""
  if not decode_hp.batch_size:
    decode_hp.batch_size = 32
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)

  problem_id = decode_hp.problem_idx
  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  has_input = "inputs" in hparams.problems[problem_id].vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = hparams.problems[problem_id].vocabulary[inputs_vocab_key]
  targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
  problem_name = FLAGS.problems.split("-")[problem_id]
  tf.logging.info("Performing decoding from a file.")
  sorted_inputs, sorted_keys = _get_sorted_inputs(filename, decode_hp.shards,
                                                  decode_hp.delimiter)
  num_decode_batches = (len(sorted_inputs) - 1) // decode_hp.batch_size + 1

  def input_fn():
    input_gen = _decode_batch_input_fn(
        problem_id, num_decode_batches, sorted_inputs, inputs_vocab,
        decode_hp.batch_size, decode_hp.max_input_size)
    gen_fn = make_input_fn_from_generator(input_gen)
    example = gen_fn()
    return _decode_input_tensor_to_features_dict(example, hparams)

  decodes = []
  result_iter = estimator.predict(input_fn)
  for result in result_iter:
    if decode_hp.return_beams:
      beam_decodes = []
      beam_scores = []
      output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
      scores = None
      if "scores" in result:
        scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
      for k, beam in enumerate(output_beams):
        tf.logging.info("BEAM %d:" % k)
        score = scores and scores[k]
        decoded_outputs, _ = log_decode_results(result["inputs"], beam,
                                                problem_name, None,
                                                inputs_vocab, targets_vocab)
        beam_decodes.append(decoded_outputs)
        if decode_hp.write_beam_scores:
          beam_scores.append(score)
      if decode_hp.write_beam_scores:
        decodes.append("\t".join(
            ["\t".join([d, "%.2f" % s]) for d, s
             in zip(beam_decodes, beam_scores)]))
      else:
        decodes.append("\t".join(beam_decodes))
    else:
      decoded_outputs, _ = log_decode_results(result["inputs"],
                                              result["outputs"], problem_name,
                                              None, inputs_vocab, targets_vocab)
      decodes.append(decoded_outputs)

  # Reversing the decoded inputs and outputs because they were reversed in
  # _decode_batch_input_fn
  sorted_inputs.reverse()
  decodes.reverse()
  # If decode_to_file was provided use it as the output filename without change
  # (except for adding shard_id if using more shards for decoding).
  # Otherwise, use the input filename plus model, hp, problem, beam, alpha.
  decode_filename = decode_to_file if decode_to_file else filename
  if decode_hp.shards > 1:
    decode_filename += "%.2d" % decode_hp.shard_id
  if not decode_to_file:
    decode_filename = _decode_filename(decode_filename, problem_name, decode_hp)
  tf.logging.info("Writing decodes into %s" % decode_filename)
  outfile = tf.gfile.Open(decode_filename, "w")
  for index in range(len(sorted_inputs)):
    outfile.write("%s%s" % (decodes[sorted_keys[index]], decode_hp.delimiter))


def _decode_filename(base_filename, problem_name, decode_hp):
  return "{base}.{model}.{hp}.{problem}.beam{beam}.alpha{alpha}.decodes".format(
      base=base_filename,
      model=FLAGS.model,
      hp=FLAGS.hparams_set,
      problem=problem_name,
      beam=str(decode_hp.beam_size),
      alpha=str(decode_hp.alpha))


def make_input_fn_from_generator(gen):
  """Use py_func to yield elements from the given generator."""
  first_ex = six.next(gen)
  flattened = tf.contrib.framework.nest.flatten(first_ex)
  types = [t.dtype for t in flattened]
  shapes = [[None] * len(t.shape) for t in flattened]
  first_ex_list = [first_ex]

  def py_func():
    if first_ex_list:
      example = first_ex_list.pop()
    else:
      example = six.next(gen)
    return tf.contrib.framework.nest.flatten(example)

  def input_fn():
    flat_example = tf.py_func(py_func, [], types)
    _ = [t.set_shape(shape) for t, shape in zip(flat_example, shapes)]
    example = tf.contrib.framework.nest.pack_sequence_as(first_ex, flat_example)
    return example

  return input_fn


def decode_interactively(estimator, hparams, decode_hp):
  """Interactive decoding."""

  def input_fn():
    gen_fn = make_input_fn_from_generator(_interactive_input_fn(hparams))
    example = gen_fn()
    example = _interactive_input_tensor_to_features_dict(example, hparams)
    return example

  result_iter = estimator.predict(input_fn)
  for result in result_iter:
    problem_idx = result["problem_choice"]
    is_image = False  # TODO(lukaszkaiser): find out from problem id / class.
    targets_vocab = hparams.problems[problem_idx].vocabulary["targets"]

    if decode_hp.return_beams:
      beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
      scores = None
      if "scores" in result:
        scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
      for k, beam in enumerate(beams):
        tf.logging.info("BEAM %d:" % k)
        beam_string = targets_vocab.decode(_save_until_eos(beam, is_image))
        if scores is not None:
          tf.logging.info("\"%s\"\tScore:%f" % (beam_string, scores[k]))
        else:
          tf.logging.info("\"%s\"" % beam_string)
    else:
      if decode_hp.identity_output:
        tf.logging.info(" ".join(map(str, result["outputs"].flatten())))
      else:
        tf.logging.info(
            targets_vocab.decode(_save_until_eos(result["outputs"], is_image)))


def _decode_batch_input_fn(problem_id, num_decode_batches, sorted_inputs,
                           vocabulary, batch_size, max_input_size):
  tf.logging.info(" batch %d" % num_decode_batches)
  # First reverse all the input sentences so that if you're going to get OOMs,
  # you'll see it in the first batch
  sorted_inputs.reverse()
  for b in range(num_decode_batches):
    tf.logging.info("Decoding batch %d" % b)
    batch_length = 0
    batch_inputs = []
    for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
      input_ids = vocabulary.encode(inputs)
      if max_input_size > 0:
        # Subtract 1 for the EOS_ID.
        input_ids = input_ids[:max_input_size - 1]
      input_ids.append(text_encoder.EOS_ID)
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
        "problem_choice": np.array(problem_id).astype(np.int32),
    }


def _interactive_input_fn(hparams):
  """Generator that reads from the terminal and yields "interactive inputs".

  Due to temporary limitations in tf.learn, if we don't want to reload the
  whole graph, then we are stuck encoding all of the input as one fixed-size
  numpy array.

  We yield int32 arrays with shape [const_array_size].  The format is:
  [num_samples, decode_length, len(input ids), <input ids>, <padding>]

  Args:
    hparams: model hparams
  Yields:
    numpy arrays

  Raises:
    Exception: when `input_type` is invalid.
  """
  num_samples = 1
  decode_length = 100
  input_type = "text"
  problem_id = 0
  p_hparams = hparams.problems[problem_id]
  has_input = "inputs" in p_hparams.input_modality
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
              "  pr=<problem_num>    (set the problem number, default: 0)\n"
              "  in=<input_problem>  (set the input problem number)\n"
              "  ou=<output_problem> (set the output problem number)\n"
              "  ns=<num_samples>    (changes number of samples, default: 1)\n"
              "  dl=<decode_length>  (changes decode length, default: 100)\n"
              "  <%s>                (decode)\n"
              "  q                   (quit)\n"
              ">" % (num_samples, decode_length, "source_string"
                     if has_input else "target_prefix"))
    input_string = input(prompt)
    if input_string == "q":
      return
    elif input_string[:3] == "pr=":
      problem_id = int(input_string[3:])
      p_hparams = hparams.problems[problem_id]
      has_input = "inputs" in p_hparams.input_modality
      vocabulary = p_hparams.vocabulary["inputs" if has_input else "targets"]
    elif input_string[:3] == "in=":
      problem = int(input_string[3:])
      p_hparams.input_modality = hparams.problems[problem].input_modality
      p_hparams.input_space_id = hparams.problems[problem].input_space_id
    elif input_string[:3] == "ou=":
      problem = int(input_string[3:])
      p_hparams.target_modality = hparams.problems[problem].target_modality
      p_hparams.target_space_id = hparams.problems[problem].target_space_id
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
        yield {
            "inputs": np.array(x).astype(np.int32),
        }
      elif input_type == "image":
        input_path = input_string
        img = vocabulary.encode(input_path)
        yield {
            "inputs": img.astype(np.int32),
        }
      elif input_type == "label":
        input_ids = [int(input_string)]
        x = [num_samples, decode_length, len(input_ids)] + input_ids
        yield {
            "inputs": np.array(x).astype(np.int32),
        }
      else:
        raise Exception("Unsupported input type.")


def show_and_save_image(img, save_path):
  try:
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    tf.logging.warning("Showing and saving an image requires matplotlib to be "
                       "installed: %s", e)
    raise NotImplementedError("Image display and save not implemented.")
  plt.imshow(img)
  plt.savefig(save_path)


def _get_sorted_inputs(filename, num_shards=1, delimiter="\n"):
  """Returning inputs sorted according to length.

  Args:
    filename: path to file with inputs, 1 per line.
    num_shards: number of input shards. If > 1, will read from file filename.XX,
      where XX is FLAGS.worker_id.
    delimiter: str, delimits records in the file.

  Returns:
    a sorted list of inputs

  """
  tf.logging.info("Getting sorted inputs")
  # read file and sort inputs according them according to input length.
  if num_shards > 1:
    decode_filename = filename + ("%.2d" % FLAGS.worker_id)
  else:
    decode_filename = filename

  with tf.gfile.Open(decode_filename) as f:
    text = f.read()
    records = text.split(delimiter)
    inputs = [record.strip() for record in records]
    # Strip the last empty line.
    if not inputs[-1]:
      inputs.pop()
  input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
  # We'll need the keys to rearrange the inputs back into their original order
  sorted_keys = {}
  sorted_inputs = []
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys


def _save_until_eos(hyp, is_image):
  """Strips everything after the first <EOS> token, which is normally 1."""
  hyp = hyp.flatten()
  if is_image:
    return hyp
  try:
    index = list(hyp).index(text_encoder.EOS_ID)
    return hyp[0:index]
  except ValueError:
    # No EOS_ID: return the array as-is.
    return hyp


def _interactive_input_tensor_to_features_dict(feature_map, hparams):
  """Convert the interactive input format (see above) to a dictionary.

  Args:
    feature_map: a dictionary with keys `problem_choice` and `input` containing
      Tensors.
    hparams: model hyperparameters

  Returns:
    a features dictionary, as expected by the decoder.
  """
  inputs = tf.convert_to_tensor(feature_map["inputs"])
  input_is_image = False if len(inputs.get_shape()) < 3 else True

  def input_fn(problem_choice, x=inputs):  # pylint: disable=missing-docstring
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

    p_hparams = hparams.problems[problem_choice]
    return (tf.constant(p_hparams.input_space_id), tf.constant(
        p_hparams.target_space_id), x)

  input_space_id, target_space_id, x = cond_on_index(
      input_fn, feature_map["problem_choice"], len(hparams.problems) - 1)

  features = {}
  features["problem_choice"] = tf.convert_to_tensor(
      feature_map["problem_choice"])
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = (
      IMAGE_DECODE_LENGTH if input_is_image else inputs[1])
  features["inputs"] = x
  return features


def _decode_input_tensor_to_features_dict(feature_map, hparams):
  """Convert the interactive input format (see above) to a dictionary.

  Args:
    feature_map: a dictionary with keys `problem_choice` and `input` containing
      Tensors.
    hparams: model hyperparameters

  Returns:
    a features dictionary, as expected by the decoder.
  """
  inputs = tf.convert_to_tensor(feature_map["inputs"])
  input_is_image = False

  def input_fn(problem_choice, x=inputs):  # pylint: disable=missing-docstring
    p_hparams = hparams.problems[problem_choice]
    # Add a third empty dimension
    x = tf.expand_dims(x, axis=[2])
    x = tf.to_int32(x)
    return (tf.constant(p_hparams.input_space_id), tf.constant(
        p_hparams.target_space_id), x)

  input_space_id, target_space_id, x = cond_on_index(
      input_fn, feature_map["problem_choice"], len(hparams.problems) - 1)

  features = {}
  features["problem_choice"] = feature_map["problem_choice"]
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = (
      IMAGE_DECODE_LENGTH if input_is_image else tf.shape(x)[1] + 50)
  features["inputs"] = x
  return features


def cond_on_index(fn, index_tensor, max_idx, cur_idx=0):
  """Call fn(index_tensor) using tf.cond in [cur_id, max_idx]."""
  if cur_idx == max_idx:
    return fn(cur_idx)

  return tf.cond(
      tf.equal(index_tensor, cur_idx),
      lambda: fn(cur_idx),
      lambda: cond_on_index(fn, index_tensor, max_idx, cur_idx + 1)
  )
