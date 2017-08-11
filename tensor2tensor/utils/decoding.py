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
from tensor2tensor.utils import data_reader
from tensor2tensor.utils import devices
from tensor2tensor.utils import input_fn_builder
import tensorflow as tf

FLAGS = tf.flags.FLAGS


def decode_from_dataset(estimator):
  hparams = estimator.hparams
  for i, problem in enumerate(FLAGS.problems.split("-")):
    inputs_vocab = hparams.problems[i].vocabulary.get("inputs", None)
    targets_vocab = hparams.problems[i].vocabulary["targets"]
    tf.logging.info("Performing local inference.")
    infer_problems_data = data_reader.get_data_filepatterns(
        FLAGS.problems, hparams.data_dir, tf.contrib.learn.ModeKeys.INFER)

    infer_input_fn = input_fn_builder.build_input_fn(
        mode=tf.contrib.learn.ModeKeys.INFER,
        hparams=hparams,
        data_file_patterns=infer_problems_data,
        num_datashards=devices.data_parallelism().n,
        fixed_problem=i)

    def log_fn(inputs,
               targets,
               outputs,
               problem,
               j,
               inputs_vocab=inputs_vocab,
               targets_vocab=targets_vocab):
      """Log inference results."""
      if "image" in problem and FLAGS.decode_save_images:
        save_path = os.path.join(estimator.model_dir,
                                 "%s_prediction_%d.jpg" % (problem, j))
        show_and_save_image(inputs / 255., save_path)
      elif inputs_vocab:
        decoded_inputs = inputs_vocab.decode(
            _save_until_eos(inputs.flatten()))
        tf.logging.info("Inference results INPUT: %s" % decoded_inputs)

      if FLAGS.identity_output:
        decoded_outputs = " ".join(map(str, outputs.flatten()))
        decoded_targets = " ".join(map(str, targets.flatten()))
      else:
        decoded_outputs = targets_vocab.decode(
            _save_until_eos(outputs.flatten()))
        decoded_targets = targets_vocab.decode(
            _save_until_eos(targets.flatten()))

      tf.logging.info("Inference results OUTPUT: %s" % decoded_outputs)
      tf.logging.info("Inference results TARGET: %s" % decoded_targets)
      if FLAGS.decode_to_file:
        output_filepath = FLAGS.decode_to_file + ".outputs." + problem
        output_file = tf.gfile.Open(output_filepath, "a")
        output_file.write(decoded_outputs + "\n")
        target_filepath = FLAGS.decode_to_file + ".targets." + problem
        target_file = tf.gfile.Open(target_filepath, "a")
        target_file.write(decoded_targets + "\n")
    result_iter = estimator.predict(input_fn=infer_input_fn, as_iterable=True)
    count = 0
    for result in result_iter:
      # predictions from the test input. We use it to log inputs and decodes.
      inputs = result["inputs"]
      targets = result["targets"]
      outputs = result["outputs"]
      if FLAGS.decode_return_beams:
        output_beams = np.split(outputs, FLAGS.decode_beam_size, axis=0)
        for k, beam in enumerate(output_beams):
          tf.logging.info("BEAM %d:" % k)
          log_fn(inputs, targets, beam, problem, count)
      else:
        log_fn(inputs, targets, outputs, problem, count)

      count += 1
      if FLAGS.decode_num_samples != -1 and count >= FLAGS.decode_num_samples:
        break
    tf.logging.info("Completed inference on %d samples." % count)


def decode_from_file(estimator, filename):
  """Compute predictions on entries in filename and write them out."""
  hparams = estimator.hparams
  problem_id = FLAGS.decode_problem_id
  inputs_vocab = hparams.problems[problem_id].vocabulary["inputs"]
  targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
  tf.logging.info("Performing decoding from a file.")
  sorted_inputs, sorted_keys = _get_sorted_inputs(filename)
  num_decode_batches = (len(sorted_inputs) - 1) // FLAGS.decode_batch_size + 1
  input_fn = _decode_batch_input_fn(problem_id, num_decode_batches,
                                    sorted_inputs, inputs_vocab)

  decodes = []
  for _ in range(num_decode_batches):
    result_iter = estimator.predict(
        input_fn=input_fn.next if six.PY2 else input_fn.__next__,
        as_iterable=True)
    for result in result_iter:

      def log_fn(inputs, outputs):
        decoded_inputs = inputs_vocab.decode(_save_until_eos(inputs.flatten()))
        tf.logging.info("Inference results INPUT: %s" % decoded_inputs)

        decoded_outputs = targets_vocab.decode(
            _save_until_eos(outputs.flatten()))
        tf.logging.info("Inference results OUTPUT: %s" % decoded_outputs)
        return decoded_outputs

      if FLAGS.decode_return_beams:
        beam_decodes = []
        output_beams = np.split(
            result["outputs"], FLAGS.decode_beam_size, axis=0)
        for k, beam in enumerate(output_beams):
          tf.logging.info("BEAM %d:" % k)
          beam_decodes.append(log_fn(result["inputs"], beam))
        decodes.append("\t".join(beam_decodes))

      else:
        decodes.append(log_fn(result["inputs"], result["outputs"]))

  # Reversing the decoded inputs and outputs because they were reversed in
  # _decode_batch_input_fn
  sorted_inputs.reverse()
  decodes.reverse()
  # Dumping inputs and outputs to file filename.decodes in
  # format result\tinput in the same order as original inputs
  if FLAGS.decode_to_file:
    output_filename = FLAGS.decode_to_file
  else:
    output_filename = filename
  if FLAGS.decode_shards > 1:
    base_filename = output_filename + ("%.2d" % FLAGS.worker_id)
  else:
    base_filename = output_filename
  decode_filename = (base_filename + "." + FLAGS.model + "." + FLAGS.hparams_set
                     + ".beam" + str(FLAGS.decode_beam_size) + ".alpha" +
                     str(FLAGS.decode_alpha) + ".decodes")
  tf.logging.info("Writing decodes into %s" % decode_filename)
  outfile = tf.gfile.Open(decode_filename, "w")
  for index in range(len(sorted_inputs)):
    outfile.write("%s\n" % (decodes[sorted_keys[index]]))


def decode_interactively(estimator):
  hparams = estimator.hparams

  infer_input_fn = _interactive_input_fn(hparams)
  for problem_idx, example in infer_input_fn:
    targets_vocab = hparams.problems[problem_idx].vocabulary["targets"]
    result_iter = estimator.predict(input_fn=lambda e=example: e)
    for result in result_iter:
      if FLAGS.decode_return_beams:
        beams = np.split(result["outputs"], FLAGS.decode_beam_size, axis=0)
        scores = None
        if "scores" in result:
          scores = np.split(result["scores"], FLAGS.decode_beam_size, axis=0)
        for k, beam in enumerate(beams):
          tf.logging.info("BEAM %d:" % k)
          beam_string = targets_vocab.decode(_save_until_eos(beam.flatten()))
          if scores is not None:
            tf.logging.info("%s\tScore:%f" % (beam_string, scores[k]))
          else:
            tf.logging.info(beam_string)
      else:
        if FLAGS.identity_output:
          tf.logging.info(" ".join(map(str, result["outputs"].flatten())))
        else:
          tf.logging.info(
              targets_vocab.decode(
                  _save_until_eos(result["outputs"].flatten())))


def _decode_batch_input_fn(problem_id, num_decode_batches, sorted_inputs,
                           vocabulary):
  tf.logging.info(" batch %d" % num_decode_batches)
  # First reverse all the input sentences so that if you're going to get OOMs,
  # you'll see it in the first batch
  sorted_inputs.reverse()
  for b in range(num_decode_batches):
    tf.logging.info("Decoding batch %d" % b)
    batch_length = 0
    batch_inputs = []
    for inputs in sorted_inputs[b * FLAGS.decode_batch_size:(
        b + 1) * FLAGS.decode_batch_size]:
      input_ids = vocabulary.encode(inputs)
      if FLAGS.decode_max_input_size > 0:
        # Subtract 1 for the EOS_ID.
        input_ids = input_ids[:FLAGS.decode_max_input_size - 1]
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
        "inputs": np.array(final_batch_inputs),
        "problem_choice": np.array(problem_id)
    }


def _interactive_input_fn(hparams):
  """Generator that reads from the terminal and yields "interactive inputs".

  Due to temporary limitations in tf.learn, if we don't want to reload the
  whole graph, then we are stuck encoding all of the input as one fixed-size
  numpy array.

  We yield int64 arrays with shape [const_array_size].  The format is:
  [num_samples, decode_length, len(input ids), <input ids>, <padding>]

  Args:
    hparams: model hparams
  Yields:
    numpy arrays

  Raises:
    Exception: when `input_type` is invalid.
  """
  num_samples = 3
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
              "  it=<input_type>     ('text' or 'image' or 'label')\n"
              "  pr=<problem_num>    (set the problem number)\n"
              "  in=<input_problem>  (set the input problem number)\n"
              "  ou=<output_problem> (set the output problem number)\n"
              "  ns=<num_samples>    (changes number of samples)\n"
              "  dl=<decode_length>  (changes decode legnth)\n"
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
        yield problem_id, {
            "inputs": np.array(x),
            "problem_choice": np.array(problem_id)
        }
      elif input_type == "image":
        input_path = input_string
        img = read_image(input_path)
        yield problem_id, {
            "inputs": img,
            "problem_choice": np.array(problem_id)
        }
      elif input_type == "label":
        input_ids = [int(input_string)]
        x = [num_samples, decode_length, len(input_ids)] + input_ids
        yield problem_id, {
            "inputs": np.array(x),
            "problem_choice": np.array(problem_id)
        }
      else:
        raise Exception("Unsupported input type.")


def read_image(path):
  try:
    import matplotlib.image as im  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    tf.logging.warning(
        "Reading an image requires matplotlib to be installed: %s", e)
    raise NotImplementedError("Image reading not implemented.")
  return im.imread(path)


def show_and_save_image(img, save_path):
  try:
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    tf.logging.warning("Showing and saving an image requires matplotlib to be "
                       "installed: %s", e)
    raise NotImplementedError("Image display and save not implemented.")
  plt.imshow(img)
  plt.savefig(save_path)


def _get_sorted_inputs(filename):
  """Returning inputs sorted according to length.

  Args:
    filename: path to file with inputs, 1 per line.

  Returns:
    a sorted list of inputs

  """
  tf.logging.info("Getting sorted inputs")
  # read file and sort inputs according them according to input length.
  if FLAGS.decode_shards > 1:
    decode_filename = filename + ("%.2d" % FLAGS.worker_id)
  else:
    decode_filename = filename
  inputs = [line.strip() for line in tf.gfile.Open(decode_filename)]
  input_lens = [(i, len(line.strip().split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
  # We'll need the keys to rearrange the inputs back into their original order
  sorted_keys = {}
  sorted_inputs = []
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys


def _save_until_eos(hyp):
  """Strips everything after the first <EOS> token, which is normally 1."""
  try:
    index = list(hyp).index(text_encoder.EOS_ID)
    return hyp[0:index]
  except ValueError:
    # No EOS_ID: return the array as-is.
    return hyp
