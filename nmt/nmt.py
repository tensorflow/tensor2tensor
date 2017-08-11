# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""TensorFlow NMT model implementation."""
from __future__ import print_function

import argparse
import os
import random
import sys

# import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from . import inference
from . import train
from .utils import evaluation_utils
from .utils import misc_utils as utils
from .utils import vocab_utils

utils.check_tensorflow_version()

FLAGS = None


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")

  # network
  parser.add_argument("--num_units", type=int, default=32, help="Network size.")
  parser.add_argument("--num_layers", type=int, default=2,
                      help="Network depth.")
  parser.add_argument("--encoder_type", type=str, default="uni", help="""\
      uni | bi | gnmt. For bi, we build num_layers/2 bi-directional layers.For
      gnmt, we build 1 bi-directional layer, and (num_layers - 1) uni-
      directional layers.\
      """)
  parser.add_argument("--residual", type="bool", nargs="?", const=True,
                      default=False,
                      help="Whether to add residual connections.")
  parser.add_argument("--time_major", type="bool", nargs="?", const=True,
                      default=True,
                      help="Whether to use time-major mode for dynamic RNN.")
  parser.add_argument("--num_embeddings_partitions", type=int, default=0,
                      help="Number of partitions for embedding vars.")

  # attention mechanisms
  parser.add_argument("--attention", type=str, default="", help="""\
      luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
      attention\
      """)
  parser.add_argument(
      "--attention_architecture",
      type=str,
      default="standard",
      help="""\
      standard | gnmt | gnmt_v2.
      standard: use top layer to compute attention.
      gnmt: GNMT style of computing attention, use previous bottom layer to
          compute attention.
      gnmt_v2: similar to gnmt, but use current bottom layer to compute
          attention.\
      """)
  parser.add_argument(
      "--pass_hidden_state", type="bool", nargs="?", const=True,
      default=True,
      help="""\
      Whether to pass encoder's hidden state to decoder when using an attention
      based model.\
      """)

  # optimizer
  parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
  parser.add_argument("--learning_rate", type=float, default=1.0,
                      help="Learning rate. Adam: 0.001 | 0.0001")
  parser.add_argument("--start_decay_step", type=int, default=0,
                      help="When we start to decay")
  parser.add_argument("--decay_steps", type=int, default=10000,
                      help="How frequent we decay")
  parser.add_argument("--decay_factor", type=float, default=0.98,
                      help="How much we decay.")
  parser.add_argument(
      "--num_train_steps", type=int, default=12000, help="Num steps to train.")
  parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?",
                      const=True,
                      default=True,
                      help=("Whether try colocating gradients with "
                            "corresponding op"))

  # initializer
  parser.add_argument("--init_op", type=str, default="uniform",
                      help="uniform | glorot_normal | glorot_uniform")
  parser.add_argument("--init_weight", type=float, default=0.1,
                      help=("for uniform init_op, initialize weights "
                           "between [-this, this]."))

  # data
  parser.add_argument("--src", type=str, default=None,
                      help="Source suffix, e.g., en.")
  parser.add_argument("--tgt", type=str, default=None,
                      help="Target suffix, e.g., de.")
  parser.add_argument("--train_prefix", type=str, default=None,
                      help="Train prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--dev_prefix", type=str, default=None,
                      help="Dev prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--test_prefix", type=str, default=None,
                      help="Test prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--out_dir", type=str, default=None,
                      help="Store log/model files.")

  # Vocab
  parser.add_argument("--vocab_prefix", type=str, default=None, help="""\
      Vocab prefix, expect files with src/tgt suffixes.If None, extract from
      train files.\
      """)
  parser.add_argument("--sos", type=str, default="<s>",
                      help="Start-of-sentence symbol.")
  parser.add_argument("--eos", type=str, default="</s>",
                      help="End-of-sentence symbol.")
  parser.add_argument("--share_vocab", type="bool", nargs="?", const=True,
                      default=False,
                      help="""\
      Whether to use the source vocab and embeddings for both source and
      target.\
      """)

  # Sequence lengths
  parser.add_argument("--src_max_len", type=int, default=50,
                      help="Max length of src sequences during training.")
  parser.add_argument("--tgt_max_len", type=int, default=50,
                      help="Max length of tgt sequences during training.")
  parser.add_argument("--src_max_len_infer", type=int, default=None,
                      help="Max length of src sequences during inference.")
  parser.add_argument("--tgt_max_len_infer", type=int, default=None,
                      help="""\
      Max length of tgt sequences during inference.  Also use to restrict the
      maximum decoding length.\
      """)

  # Default settings works well (rarely need to change)
  parser.add_argument("--unit_type", type=str, default="lstm",
                      help="lstm | gru | layer_norm_lstm")
  parser.add_argument("--forget_bias", type=float, default=1.0,
                      help="Forget bias for BasicLSTMCell.")
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout rate (not keep_prob)")
  parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                      help="Clip gradients to this norm.")
  parser.add_argument("--source_reverse", type="bool", nargs="?", const=True,
                      default=False, help="Reverse source sequence.")
  parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

  parser.add_argument("--steps_per_stats", type=int, default=100,
                      help=("How many training steps to do per stats logging."
                            "Save checkpoint every 10x steps_per_stats"))
  parser.add_argument("--max_train", type=int, default=0,
                      help="Limit on the size of training data (0: no limit).")
  parser.add_argument("--num_buckets", type=int, default=5,
                      help="Put data into similar-length buckets.")

  # BPE
  parser.add_argument("--bpe_delimiter", type=str, default=None,
                      help="Set to @@ to activate BPE")

  # Misc
  parser.add_argument("--num_gpus", type=int, default=1,
                      help="Number of gpus in each worker.")
  parser.add_argument("--log_device_placement", type="bool", nargs="?",
                      const=True, default=False, help="Debug GPU allocation.")
  parser.add_argument("--metrics", type=str, default="bleu",
                      help=("Comma-separated list of evaluations "
                            "metrics (bleu,rouge,accuracy)"))
  parser.add_argument("--steps_per_external_eval", type=int, default=None,
                      help="""\
      How many training steps to do per external evaluation.  Automatically set
      based on data if None.\
      """)
  parser.add_argument("--scope", type=str, default=None,
                      help="scope to put variables under")
  parser.add_argument("--hparams_path", type=str, default=None,
                      help=("Path to standard hparams json file that overrides"
                            "hparams values from FLAGS."))
  parser.add_argument("--random_seed", type=int, default=None,
                      help="Random seed (>0, set a specific seed).")

  # Inference
  parser.add_argument("--ckpt", type=str, default="",
                      help="Checkpoint file to load a model for inference.")
  parser.add_argument("--inference_input_file", type=str, default=None,
                      help="Set to the text to decode.")
  parser.add_argument("--inference_list", type=str, default=None,
                      help=("A comma-separated list of sentence indices "
                            "(0-based) to decode."))
  parser.add_argument("--infer_batch_size", type=int, default=32,
                      help="Batch size for inference mode.")
  parser.add_argument("--inference_output_file", type=str, default=None,
                      help="Output file to store decoding results.")
  parser.add_argument("--inference_ref_file", type=str, default=None,
                      help=("""\
      Reference file to compute evaluation scores (if provided).\
      """))
  parser.add_argument("--beam_width", type=int, default=0,
                      help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))
  parser.add_argument("--length_penalty_weight", type=float, default=0.0,
                      help="Length penalty for beam search.")

  # Job info
  parser.add_argument("--jobid", type=int, default=0,
                      help="Task id of the worker.")
  parser.add_argument("--num_workers", type=int, default=1,
                      help="Number of workers (inference only).")


def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Data
      src=flags.src,
      tgt=flags.tgt,
      train_prefix=flags.train_prefix,
      dev_prefix=flags.dev_prefix,
      test_prefix=flags.test_prefix,
      vocab_prefix=flags.vocab_prefix,
      out_dir=flags.out_dir,

      # Networks
      num_units=flags.num_units,
      num_layers=flags.num_layers,
      dropout=flags.dropout,
      unit_type=flags.unit_type,
      encoder_type=flags.encoder_type,
      residual=flags.residual,
      time_major=flags.time_major,
      num_embeddings_partitions=flags.num_embeddings_partitions,

      # Attention mechanisms
      attention=flags.attention,
      attention_architecture=flags.attention_architecture,
      pass_hidden_state=flags.pass_hidden_state,

      # Train
      optimizer=flags.optimizer,
      num_train_steps=flags.num_train_steps,
      batch_size=flags.batch_size,
      init_op=flags.init_op,
      init_weight=flags.init_weight,
      max_gradient_norm=flags.max_gradient_norm,
      learning_rate=flags.learning_rate,
      start_decay_step=flags.start_decay_step,
      decay_factor=flags.decay_factor,
      decay_steps=flags.decay_steps,
      colocate_gradients_with_ops=flags.colocate_gradients_with_ops,

      # Data constraints
      num_buckets=flags.num_buckets,
      max_train=flags.max_train,
      src_max_len=flags.src_max_len,
      tgt_max_len=flags.tgt_max_len,
      source_reverse=flags.source_reverse,

      # Inference
      src_max_len_infer=flags.src_max_len_infer,
      tgt_max_len_infer=flags.tgt_max_len_infer,
      infer_batch_size=flags.infer_batch_size,
      beam_width=flags.beam_width,
      length_penalty_weight=flags.length_penalty_weight,

      # Vocab
      sos=flags.sos if flags.sos else vocab_utils.SOS,
      eos=flags.eos if flags.eos else vocab_utils.EOS,
      bpe_delimiter=flags.bpe_delimiter,

      # Misc
      forget_bias=flags.forget_bias,
      num_gpus=flags.num_gpus,
      epoch_step=0,  # record where we were within an epoch.
      steps_per_stats=flags.steps_per_stats,
      steps_per_external_eval=flags.steps_per_external_eval,
      share_vocab=flags.share_vocab,
      metrics=flags.metrics.split(","),
      log_device_placement=flags.log_device_placement,
      random_seed=flags.random_seed,
  )


def extend_hparams(hparams):
  """Extend training hparams."""
  # Sanity checks
  if hparams.encoder_type == "bi" and hparams.num_layers % 2 != 0:
    raise ValueError("For bi, num_layers %d should be even" %
                     hparams.num_layers)
  if (hparams.attention_architecture in ["gnmt"] and
      hparams.num_layers < 2):
    raise ValueError("For gnmt attention architecture, "
                     "num_layers %d should be >= 2" % hparams.num_layers)

  # Flags
  utils.print_out("# hparams:")
  utils.print_out("  src=%s" % hparams.src)
  utils.print_out("  tgt=%s" % hparams.tgt)
  utils.print_out("  train_prefix=%s" % hparams.train_prefix)
  utils.print_out("  dev_prefix=%s" % hparams.dev_prefix)
  utils.print_out("  test_prefix=%s" % hparams.test_prefix)
  utils.print_out("  out_dir=%s" % hparams.out_dir)

  # Set num_residual_layers
  if hparams.residual and hparams.num_layers > 1:
    if hparams.encoder_type == "gnmt":
      # The first unidirectional layer (after the bi-directional layer) in
      # the GNMT encoder can't have residual connection due to the input is
      # the concatenation of fw_cell and bw_cell's outputs.
      num_residual_layers = hparams.num_layers - 2
    else:
      num_residual_layers = hparams.num_layers - 1
  else:
    num_residual_layers = 0
  hparams.add_hparam("num_residual_layers", num_residual_layers)

  ## Vocab
  # Get vocab file names first
  if hparams.vocab_prefix:
    src_vocab_file = hparams.vocab_prefix + "." + hparams.src
    tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
  else:
    raise ValueError("hparams.vocab_prefix must be provided.")

  # Source vocab
  src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
      src_vocab_file,
      hparams.out_dir,
      sos=hparams.sos,
      eos=hparams.eos,
      unk=vocab_utils.UNK)

  # Target vocab
  if hparams.share_vocab:
    utils.print_out("  using source vocab for target")
    tgt_vocab_file = src_vocab_file
    tgt_vocab_size = src_vocab_size
  else:
    tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
        tgt_vocab_file,
        hparams.out_dir,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)
  hparams.add_hparam("src_vocab_size", src_vocab_size)
  hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
  hparams.add_hparam("src_vocab_file", src_vocab_file)
  hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)

  # Check out_dir
  if not tf.gfile.Exists(hparams.out_dir):
    utils.print_out("# Creating output directory %s ..." % hparams.out_dir)
    tf.gfile.MakeDirs(hparams.out_dir)

  # Evaluation
  for metric in hparams.metrics:
    hparams.add_hparam("best_" + metric, 0)  # larger is better
    best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
    hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
    tf.gfile.MakeDirs(best_metric_dir)

  return hparams


def ensure_compatible_hparams(hparams, default_hparams, hparams_path):
  """Make sure the loaded hparams is compatible with new changes."""
  default_hparams = utils.maybe_parse_standard_hparams(
      default_hparams, hparams_path)

  # For compatible reason, if there are new fields in default_hparams,
  #   we add them to the current hparams
  default_config = default_hparams.values()
  config = hparams.values()
  for key in default_config:
    if key not in config:
      hparams.add_hparam(key, default_config[key])

  # Make sure that the loaded model has latest values for the below keys
  updated_keys = [
      "out_dir", "num_gpus", "test_prefix", "beam_width",
      "length_penalty_weight", "num_train_steps"
  ]
  for key in updated_keys:
    if key in default_config and getattr(hparams, key) != default_config[key]:
      utils.print_out("# Updating hparams.%s: %s -> %s" %
                      (key, str(getattr(hparams, key)),
                       str(default_config[key])))
      setattr(hparams, key, default_config[key])
  return hparams


def create_or_load_hparams(out_dir, default_hparams, hparams_path):
  """Create hparams or load hparams from out_dir."""
  hparams = utils.load_hparams(out_dir)
  if not hparams:
    hparams = default_hparams
    hparams = utils.maybe_parse_standard_hparams(
        hparams, hparams_path)
    hparams = extend_hparams(hparams)
  else:
    hparams = ensure_compatible_hparams(hparams, default_hparams, hparams_path)

  # Save HParams
  utils.save_hparams(out_dir, hparams)

  for metric in hparams.metrics:
    utils.save_hparams(getattr(hparams, "best_" + metric + "_dir"), hparams)

  # Print HParams
  utils.print_hparams(hparams)
  return hparams


def run_main(flags, default_hparams, train_fn, inference_fn, target_session=""):
  """Run main."""
  # Job
  jobid = flags.jobid
  num_workers = flags.num_workers
  utils.print_out("# Job id %d" % jobid)

  # Random
  random_seed = flags.random_seed
  if random_seed is not None and random_seed > 0:
    utils.print_out("# Set random seed to %d" % random_seed)
    random.seed(random_seed + jobid)
    np.random.seed(random_seed + jobid)

  ## Train / Decode
  out_dir = flags.out_dir
  if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)

  # Load hparams.
  hparams = create_or_load_hparams(out_dir, default_hparams, flags.hparams_path)

  if flags.inference_input_file:
    # Inference indices
    hparams.inference_indices = None
    if flags.inference_list:
      (hparams.inference_indices) = (
          [int(token)  for token in flags.inference_list.split(",")])

    # Inference
    trans_file = flags.inference_output_file
    ckpt = flags.ckpt
    if not ckpt:
      ckpt = tf.train.latest_checkpoint(out_dir)
    inference_fn(ckpt, flags.inference_input_file,
                 trans_file, hparams, num_workers, jobid)

    # Evaluation
    ref_file = flags.inference_ref_file
    if ref_file and tf.gfile.Exists(trans_file):
      for metric in hparams.metrics:
        score = evaluation_utils.evaluate(
            ref_file,
            trans_file,
            metric,
            hparams.bpe_delimiter)
        utils.print_out("  %s: %.1f" % (metric, score))
  else:
    # Train
    train_fn(hparams, target_session=target_session)


def main(unused_argv):
  default_hparams = create_hparams(FLAGS)
  train_fn = train.train
  inference_fn = inference.inference
  run_main(FLAGS, default_hparams, train_fn, inference_fn)


if __name__ == "__main__":
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
