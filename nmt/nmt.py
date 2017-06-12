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
from .utils import misc_utils as utils
from .utils import vocab_utils


FLAGS = None


def create_hparams():
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Data
      src=FLAGS.src,
      tgt=FLAGS.tgt,
      train_prefix=FLAGS.train_prefix,
      dev_prefix=FLAGS.dev_prefix,
      test_prefix=FLAGS.test_prefix,
      vocab_prefix=FLAGS.vocab_prefix,
      out_dir=FLAGS.out_dir,

      # Networks
      num_units=FLAGS.num_units,
      num_layers=FLAGS.num_layers,
      dropout=FLAGS.dropout,
      unit_type=FLAGS.unit_type,
      encoder_type=FLAGS.encoder_type,
      residual=FLAGS.residual,
      time_major=FLAGS.time_major,

      # Attention mechanisms
      attention=FLAGS.attention,
      attention_type=FLAGS.attention_type,
      attention_architecture=FLAGS.attention_architecture,
      alignment_history=FLAGS.alignment_history,

      # Train
      optimizer=FLAGS.optimizer,
      num_train_steps=FLAGS.num_train_steps,
      batch_size=FLAGS.batch_size,
      init_weight=FLAGS.init_weight,
      gradient_clip_value=FLAGS.gradient_clip_value,
      gradient_clip_pattern=FLAGS.gradient_clip_pattern,
      max_gradient_norm=FLAGS.max_gradient_norm,
      max_emb_gradient_norm=FLAGS.max_emb_gradient_norm,
      learning_rate=FLAGS.learning_rate,
      start_decay_step=FLAGS.start_decay_step,
      decay_factor=FLAGS.decay_factor,
      decay_steps=FLAGS.decay_steps,
      colocate_gradients_with_ops=FLAGS.colocate_gradients_with_ops,

      # Data constraints
      num_buckets=FLAGS.num_buckets,
      max_train=FLAGS.max_train,
      src_max_len=FLAGS.src_max_len,
      tgt_max_len=FLAGS.tgt_max_len,
      src_max_len_infer=FLAGS.src_max_len_infer,
      tgt_max_len_infer=FLAGS.tgt_max_len_infer,
      source_reverse=FLAGS.source_reverse,

      # Vocab
      sos=FLAGS.sos if FLAGS.sos else vocab_utils.SOS,
      eos=FLAGS.eos if FLAGS.eos else vocab_utils.EOS,
      bpe_delimiter=FLAGS.bpe_delimiter,
      src_max_vocab_size=FLAGS.src_max_vocab_size,
      tgt_max_vocab_size=FLAGS.tgt_max_vocab_size,

      # Seq2label
      cl_num_classes=FLAGS.cl_num_classes,
      cl_num_layers=FLAGS.cl_num_layers,
      cl_hidden_size=FLAGS.cl_hidden_size,
      cl_dropout=FLAGS.cl_dropout,

      # Misc
      forget_bias=FLAGS.forget_bias,
      num_gpus=FLAGS.num_gpus,
      steps_per_stats=FLAGS.steps_per_stats,
      steps_per_external_eval=FLAGS.steps_per_external_eval,
      share_vocab=FLAGS.share_vocab,
      metrics=FLAGS.metrics.split(","),
      log_device_placement=FLAGS.log_device_placement,
      random_seed=FLAGS.random_seed,

      # Experimental
      ignore_list_file=FLAGS.ignore_list_file,
      src_embed_file=FLAGS.src_embed_file,
      tgt_embed_file=FLAGS.tgt_embed_file,
      src_embed_trainable=FLAGS.src_embed_trainable,
      tgt_embed_trainable=FLAGS.tgt_embed_trainable,
      task=FLAGS.task,
  )


def extend_hparams(hparams):
  """Extend training hparams."""
  # Sanity checks
  if hparams.encoder_type == "bi" and hparams.num_layers % 2 != 0:
    raise ValueError("For bi, num_layers %d should be even" %
                     hparams.num_layers)
  if (hparams.attention_architecture in ["gnmt", "bottom", "gnmt_new"] and
      hparams.num_layers < 2):
    raise ValueError("For gnmt, bottom, and gnmt_new attention, "
                     "num_layers %d should be >= 2" % hparams.num_layers)
  if hparams.task == "seq2label" and hparams.source_reverse:
    raise ValueError("For seq2label tasks, "
                     "we don't expect source_reverse to be True")

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

  # Set output_attention & use_attention_layer
  output_attention = True
  use_attention_layer = True
  if hparams.attention_architecture in ["gnmt", "bottom", "gnmt_new"]:
    output_attention = False
    use_attention_layer = False
  hparams.add_hparam("output_attention", output_attention)
  hparams.add_hparam("use_attention_layer", use_attention_layer)

  # Ignore map
  ignore_map = None
  if hparams.ignore_list_file:
    ignore_map = utils.load_list_map(hparams.ignore_list_file)
  hparams.add_hparam("ignore_map", ignore_map)

  ## Vocab
  # Get vocab file names first
  if hparams.vocab_prefix:
    src_vocab_file = hparams.vocab_prefix + "." + hparams.src
    tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
  else:  # Create from train files
    if hparams.src_max_vocab_size:
      src_vocab_str = ".vocab.%d." % hparams.src_max_vocab_size
    else:
      src_vocab_str = ".vocab."
    if hparams.tgt_max_vocab_size:
      tgt_vocab_str = ".vocab.%d." % hparams.tgt_max_vocab_size
    else:
      tgt_vocab_str = ".vocab."
    src_vocab_file = hparams.train_prefix + src_vocab_str + hparams.src
    tgt_vocab_file = hparams.train_prefix + tgt_vocab_str + hparams.tgt

  # Source vocab
  src_vocab_size = vocab_utils.check_and_extract_vocab(
      src_vocab_file,
      hparams.train_prefix + "." + hparams.src,
      sos=hparams.sos,
      eos=hparams.eos,
      unk=vocab_utils.UNK,
      max_vocab_size=hparams.src_max_vocab_size)

  # Target vocab
  if hparams.share_vocab:
    utils.print_out("  using source vocab for target")
    tgt_vocab_file = src_vocab_file
    tgt_vocab_size = src_vocab_size
  else:
    if hparams.task == "seq2label":
      # We don't want to have unk, sos, eos in the vocab
      tgt_vocab_size = vocab_utils.check_and_extract_vocab(
          tgt_vocab_file,
          hparams.train_prefix + "." + hparams.tgt,
          max_vocab_size=hparams.tgt_max_vocab_size,
      )
    else:
      tgt_vocab_size = vocab_utils.check_and_extract_vocab(
          tgt_vocab_file,
          hparams.train_prefix + "." + hparams.tgt,
          sos=hparams.sos,
          eos=hparams.eos,
          unk=vocab_utils.UNK,
          max_vocab_size=hparams.tgt_max_vocab_size,
      )
  hparams.add_hparam("src_vocab_size", src_vocab_size)
  hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)

  # Check out_dir
  if not tf.gfile.Exists(hparams.out_dir):
    utils.print_out("# Creating output directory %s ..." % hparams.out_dir)
    tf.gfile.MakeDirs(hparams.out_dir)

  # Evaluation
  hparams.add_hparam("best_ppl", 1e10)  # smaller is better
  hparams.add_hparam("best_ppl_dir", os.path.join(hparams.out_dir, "best_ppl"))
  tf.gfile.MakeDirs(hparams.best_ppl_dir)
  for metric in hparams.metrics:
    hparams.add_hparam("best_" + metric, 0)  # larger is better
    hparams.add_hparam("best_" + metric + "_dir", os.path.join(
        hparams.out_dir, "best_" + metric))
    tf.gfile.MakeDirs(getattr(hparams, "best_" + metric + "_dir"))

  return hparams


def load_train_hparams(out_dir):
  """Load training hparams."""
  hparams = utils.load_hparams(out_dir)
  new_hparams = create_hparams()
  new_hparams = extend_hparams(new_hparams)
  if not hparams:
    hparams = new_hparams
  else:
    # For compatible reason, if there are new fields in new_hparams,
    #   we add them to the current hparams
    new_config = new_hparams.values()
    config = hparams.values()
    for key in new_config:
      if key not in config:
        hparams.add_hparam(key, new_config[key])

    # Make sure that the loaded model has latest values for the below keys
    updated_keys = ["out_dir", "num_gpus", "test_prefix"]
    for key in updated_keys:
      if getattr(hparams, key) != new_config[key]:
        utils.print_out("# Updating hparams.%s: %s -> %s" %
                        (key, str(getattr(hparams, key)), str(new_config[key])))
        setattr(hparams, key, new_config[key])

  # Save HParams
  utils.save_hparams(out_dir, hparams)

  # Print HParams
  utils.print_hparams(hparams, skip_patterns=["vocab", "embed_matrix"])
  return hparams


def main(unused_argv):
  # Job
  jobid = FLAGS.jobid
  num_workers = FLAGS.num_workers
  utils.print_out("# Job id %d" % jobid)

  # Random
  random_seed = FLAGS.random_seed
  if random_seed is not None and random_seed > 0:
    utils.print_out("# Set random seed to %d" % random_seed)
    random.seed(random_seed + jobid)
    np.random.seed(random_seed + jobid)

  ## Train / Decode
  out_dir = FLAGS.out_dir
  if FLAGS.inference_file:
    # Model dir
    if FLAGS.model_dir:
      model_dir = FLAGS.model_dir
    else:
      model_dir = out_dir

    # Load hparams.
    hparams = inference.load_inference_hparams(
        model_dir,
        inference_list=FLAGS.inference_list)

    # Inference
    inference.inference(model_dir, FLAGS.inference_file, out_dir, hparams,
                        num_workers, jobid)
  else:
    # Load hparams.
    hparams = load_train_hparams(out_dir)

    # Train
    train.train(hparams)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
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

  # attention mechanisms
  parser.add_argument("--attention", type=str, default="", help="""\
      luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
      attention\
      """)
  parser.add_argument("--attention_type", type=str, default="softmax",
                      help="softmax | monotonic")
  parser.add_argument(
      "--attention_architecture",
      type=str,
      default="standard",
      help="""\
      standard | bottom | gnmt | gnmt_new.
      standard: use top layer to compute attention.
      bottom: use bottom layer to compute attention.
      gnmt: GNMT style of computing attention, use previous bottom layer to
          compute attention. (Not Implemented)
      gnmt_new: GNMT style of computing attention use current bottom
          layer to compute attention. (Not Implemented)\
      """)
  parser.add_argument("--alignment_history", type="bool", nargs="?", const=True,
                      default=False,
                      help="Whether to generate alignment visualization.")

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
  parser.add_argument("--src_max_vocab_size", type=int, default=None,
                      help="To limit vocabs if extract from train files.")
  parser.add_argument("--tgt_max_vocab_size", type=int, default=None,
                      help="To limit vocabs if extract from train files.")
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
                      help="lstm | gru")
  parser.add_argument("--forget_bias", type=float, default=0.0,
                      help="Forget bias for BasicLSTMCell.")
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout rate (not keep_prob)")
  parser.add_argument("--gradient_clip_value", type=float, default=None,
                      help=("Clip gradients to this value before "
                            "clipping by norm."))
  parser.add_argument("--gradient_clip_pattern", type=str, default=None,
                      help="To select specific parameters to clip, e.g. lstm.")
  parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                      help="Clip gradients to this norm.")
  parser.add_argument("--max_emb_gradient_norm", type=float, default=None,
                      help="""\
      Clip embedding variables' gradients to this norm. If None,clip all
      gradients to max_gradent_norm\
      """)
  parser.add_argument("--init_weight", type=float, default=0.1,
                      help="Initial weights from [-this, this].")
  parser.add_argument("--source_reverse", type="bool", nargs="?", const=True,
                      default=True, help="Reverse source sequence.")
  parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
  parser.add_argument(
      "--steps_per_stats",
      type=int,
      default=100,
      help=("How many training steps to do per stats logging."
            "Save checkpoint every 10x steps_per_stats"))
  parser.add_argument("--max_train", type=int, default=0,
                      help="Limit on the size of training data (0: no limit).")
  parser.add_argument("--num_buckets", type=int, default=5,
                      help="Put data into similar-length buckets.")

  # BPE
  parser.add_argument("--bpe_delimiter", type=str, default=None,
                      help="Set to @@ to activate BPE")

  # Seq2label for classification tasks
  parser.add_argument("--cl_num_classes", type=int, default=2,
                      help="Number of layers for classification.")
  parser.add_argument("--cl_num_layers", type=int, default=1,
                      help="Number of hidden layers for classification.")
  parser.add_argument("--cl_hidden_size", type=int, default=256,
                      help="Number of hidden units per layer.")
  parser.add_argument("--cl_dropout", type=float, default=0.2,
                      help="The amount of dropout (= 1 - keep_prob).")

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

  # Test
  parser.add_argument("--model_dir", type=str, default="",
                      help="To load model for inference.")
  parser.add_argument("--inference_file", type=str, default=None,
                      help="Set to the text to decode.")
  parser.add_argument("--inference_list", type=str, default=None,
                      help=("A comma-separated list of sentence indices "
                            "(0-based) to decode."))

  # Experimental features
  parser.add_argument("--ignore_list_file", type=str, default=None,
                      help="Ignored tokens during BLEU computation")
  parser.add_argument("--src_embed_file", type=str, default="",
                      help="Pretrained embeddings file for src.")
  parser.add_argument("--tgt_embed_file", type=str, default="",
                      help="Pretrained embeddings file for tgt.")
  parser.add_argument("--src_embed_trainable", type="bool", nargs="?",
                      const=True, default=False,
                      help="Whether pretrained embeddings for src can be"
                      "further trained")
  parser.add_argument("--tgt_embed_trainable", type="bool", nargs="?",
                      const=True, default=False,
                      help="Whether pretrained embeddings for tgt can be"
                      "further trained")
  parser.add_argument("--task", type=str, default="seq2seq",
                      help="which task to use seq2seq (translation)"
                           " or seq2label (classification).")

  # Not tested
  parser.add_argument("--random_seed", type=int, default=None,
                      help="Random seed (>0, set a specific seed).")

  # Job info
  parser.add_argument("--jobid", type=int, default=0,
                      help="Task id of the worker.")
  parser.add_argument("--num_workers", type=int, default=1,
                      help="Number of workers for the job.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
