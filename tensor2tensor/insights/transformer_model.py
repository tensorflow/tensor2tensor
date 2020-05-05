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

"""A QueryProcessor using the Transformer framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque

import glob
import os
import shutil
import time

import numpy as np

from six.moves import range
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.insights import graph
from tensor2tensor.insights import query_processor
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow.compat.v1 as tf
from tensorflow.python import debug as tfdbg

flags = tf.flags
FLAGS = flags.FLAGS


def topk_watch_fn(feeds, fetches):
  """TFDBG watch function for transformer beam search nodes.

  Args:
    feeds: Unused. Required by tfdbg.
    fetches: Unused. Required by tfdbg.

  Returns:
    a WatchOptions instance that will capture all beam search ops.
  """
  del fetches, feeds
  return tfdbg.WatchOptions(
      node_name_regex_whitelist=
      ".*grow_(finished|alive)_(topk_scores|topk_seq).*",
      debug_ops=["DebugIdentity"])


def seq_filter(datum, tensor):
  """TFDBG data directory filter for capturing topk_seq operation dumps.

  Args:
    datum: A datum to filter by node_name.
    tensor: Unused. Required by tfdbg

  Returns:
    a true when datum should be returned.
  """
  del tensor
  return "topk_seq" in datum.node_name


def scores_filter(datum, tensor):
  """TFDBG data directory filter for capturing topk_scores operation dumps.

  Args:
    datum: A datum to filter by node_name.
    tensor: Unused. Required by tfdbg

  Returns:
    a true when datum should be returned.
  """
  del tensor
  return "topk_scores" in datum.node_name


def sequence_key(sequence):
  """Returns a key for mapping sequence paths to graph vertices."""
  return ":".join([str(s) for s in sequence])


class TransformerModel(query_processor.QueryProcessor):
  """A QueryProcessor using a trained Transformer model.

  This processor supports the following visualizations:
    - processing: Basic source and target text processing
    - graph: A graph of the beam search process.
  """

  def __init__(self, processor_configuration):
    """Creates the Transformer estimator.

    Args:
      processor_configuration: A ProcessorConfiguration protobuffer with the
        transformer fields populated.
    """
    # Do the pre-setup tensor2tensor requires for flags and configurations.
    transformer_config = processor_configuration["transformer"]
    FLAGS.output_dir = transformer_config["model_dir"]
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
    data_dir = os.path.expanduser(transformer_config["data_dir"])

    # Create the basic hyper parameters.
    self.hparams = trainer_lib.create_hparams(
        transformer_config["hparams_set"],
        transformer_config["hparams"],
        data_dir=data_dir,
        problem_name=transformer_config["problem"])

    decode_hp = decoding.decode_hparams()
    decode_hp.add_hparam("shards", 1)
    decode_hp.add_hparam("shard_id", 0)

    # Create the estimator and final hyper parameters.
    self.estimator = trainer_lib.create_estimator(
        transformer_config["model"],
        self.hparams,
        t2t_trainer.create_run_config(self.hparams),
        decode_hparams=decode_hp, use_tpu=False)

    # Fetch the vocabulary and other helpful variables for decoding.
    self.source_vocab = self.hparams.problem_hparams.vocabulary["inputs"]
    self.targets_vocab = self.hparams.problem_hparams.vocabulary["targets"]
    self.const_array_size = 10000

    # Prepare the Transformer's debug data directory.
    run_dirs = sorted(glob.glob(os.path.join("/tmp/t2t_server_dump", "run_*")))
    for run_dir in run_dirs:
      shutil.rmtree(run_dir)

  def process(self, query):
    """Returns the visualizations for query.

    Args:
      query: The query to process.

    Returns:
      A dictionary of results with processing and graph visualizations.
    """
    tf.logging.info("Processing new query [%s]" %query)

    # Create the new TFDBG hook directory.
    hook_dir = "/tmp/t2t_server_dump/request_%d" %int(time.time())
    os.makedirs(hook_dir)
    hooks = [tfdbg.DumpingDebugHook(hook_dir, watch_fn=topk_watch_fn)]

    # TODO(kstevens): This is extremely hacky and slow for responding to
    # queries.  Figure out a reasonable way to pre-load the model weights before
    # forking and run queries through the estimator quickly.
    def server_input_fn():
      """Generator that returns just the current query."""
      for _ in range(1):
        input_ids = self.source_vocab.encode(query)
        input_ids.append(text_encoder.EOS_ID)
        x = [1, 100, len(input_ids)] + input_ids
        x += [0] * (self.const_array_size - len(x))
        d = {
            "inputs": np.array(x).astype(np.int32),
        }
        yield d

    def input_fn():
      """Generator that returns just the current query."""
      gen_fn = decoding.make_input_fn_from_generator(server_input_fn())
      example = gen_fn()
      # TODO(kstevens): Make this method public
      # pylint: disable=protected-access
      return decoding._interactive_input_tensor_to_features_dict(
          example, self.hparams)

    # Make the prediction for the current query.
    result_iter = self.estimator.predict(input_fn, hooks=hooks)
    result = None
    for result in result_iter:
      break

    # Extract the beam search information by reading the dumped TFDBG event
    # tensors.  We first read and record the per step beam sequences then record
    # the beam scores.  Afterwards we align the two sets of values to create the
    # full graph vertices and edges.
    decoding_graph = graph.Graph()
    run_dirs = sorted(glob.glob(os.path.join(hook_dir, "run_*")))
    for run_dir in run_dirs:
      # Record the different completed and active beam sequence ids.
      alive_sequences = deque()
      finished_sequences = deque()

      # Make the root vertex since it always needs to exist.
      decoding_graph.get_vertex(sequence_key([0]))

      # Create the initial vertices and edges for the active and finished
      # sequences.  We uniquely define each vertex using it's full sequence path
      # as a string to ensure there's no collisions when the same step has two
      # instances of an output id.
      dump_dir = tfdbg.DebugDumpDir(run_dir, validate=False)
      seq_datums = dump_dir.find(predicate=seq_filter)
      for seq_datum in seq_datums:
        sequences = np.array(seq_datum.get_tensor()).astype(int)[0]
        if "alive" in seq_datum.node_name:
          alive_sequences.append(sequences)
        if "finished" in seq_datum.node_name:
          finished_sequences.append(sequences)

        for sequence in sequences:
          pieces = self.targets_vocab.decode_list(sequence)
          index = sequence[-1]
          if index == 0:
            continue

          parent = decoding_graph.get_vertex(sequence_key(sequence[:-1]))
          current = decoding_graph.get_vertex(sequence_key(sequence))

          edge = decoding_graph.add_edge(parent, current)
          edge.data["label"] = pieces[-1]
          edge.data["label_id"] = index
          # Coerce the type to be a python bool.  Numpy bools can't be easily
          # converted to JSON.
          edge.data["completed"] = bool(index == 1)

      # Examine the score results and store the scores with the associated edges
      # in the graph.  We fetch the vertices (and relevant edges) by looking
      # into the saved beam sequences stored above.
      score_datums = dump_dir.find(predicate=scores_filter)
      for score_datum in score_datums:
        if "alive" in score_datum.node_name:
          sequences = alive_sequences.popleft()

        if "finished" in score_datum.node_name:
          sequences = finished_sequences.popleft()

        scores = np.array(score_datum.get_tensor()).astype(float)[0]
        for i, score in enumerate(scores):
          sequence = sequences[i]
          if sequence[-1] == 0:
            continue

          vertex = decoding_graph.get_vertex(sequence_key(sequence))
          edge = decoding_graph.edges[vertex.in_edges[0]]
          edge.data["score"] = score
          edge.data["log_probability"] = score
          edge.data["total_log_probability"] = score

    # Delete the hook dir to save disk space
    shutil.rmtree(hook_dir)

    # Create the graph visualization data structure.
    graph_vis = {
        "visualization_name": "graph",
        "title": "Graph",
        "name": "graph",
        "search_graph": decoding_graph.to_dict(),
    }

    # Create the processing visualization data structure.
    # TODO(kstevens): Make this method public
    # pylint: disable=protected-access
    output_ids = decoding._save_until_eos(result["outputs"].flatten(), False)
    output_pieces = self.targets_vocab.decode_list(output_ids)
    output_token = [{"text": piece} for piece in output_pieces]
    output = self.targets_vocab.decode(output_ids)

    source_steps = [{
        "step_name": "Initial",
        "segment": [{
            "text": query
        }],
    }]

    target_steps = [{
        "step_name": "Initial",
        "segment": output_token,
    }, {
        "step_name": "Final",
        "segment": [{
            "text": output
        }],
    }]

    processing_vis = {
        "visualization_name": "processing",
        "title": "Processing",
        "name": "processing",
        "query_processing": {
            "source_processing": source_steps,
            "target_processing": target_steps,
        },
    }

    return {
        "result": [processing_vis, graph_vis],
    }
