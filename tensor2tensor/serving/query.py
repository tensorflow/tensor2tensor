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

"""Query an exported model. Py2 only. Install tensorflow-serving-api."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from grpc.beta import implementations

from six.moves import input  # pylint: disable=redefined-builtin

from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("server", None, "Address to Tensorflow Serving server.")
flags.DEFINE_string("servable_name", None, "Name of served model.")
flags.DEFINE_string("problem", None, "Problem name.")
flags.DEFINE_string("data_dir", None, "Data directory, for vocab files.")
flags.DEFINE_string("t2t_usr_dir", None, "Usr dir for registrations.")
flags.DEFINE_string("inputs_once", None, "Query once with this input.")
flags.DEFINE_integer("timeout_secs", 10, "Timeout for query.")


def make_example(input_ids, feature_name="inputs"):
  features = {
      feature_name:
          tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids))
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


def create_stub():
  host, port = FLAGS.server.split(":")
  channel = implementations.insecure_channel(host, int(port))
  return prediction_service_pb2.beta_create_PredictionService_stub(channel)


def query(stub, input_ids, feature_name="inputs"):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.servable_name
  ex = make_example(input_ids, feature_name)
  request.inputs["input"].CopyFrom(
      tf.contrib.util.make_tensor_proto(ex.SerializeToString(), shape=[1]))
  response = stub.Predict(request, FLAGS.timeout_secs)
  output_ids = response.outputs["outputs"].int_val
  return output_ids


def encode(inputs, encoder):
  input_ids = encoder.encode(inputs)
  input_ids.append(text_encoder.EOS_ID)
  return input_ids


def decode(output_ids, output_decoder):
  return output_decoder.decode(output_ids)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  problem = registry.problem(FLAGS.problem)
  hparams = tf.contrib.training.HParams(
      data_dir=os.path.expanduser(FLAGS.data_dir))
  problem.get_hparams(hparams)

  fname = "inputs" if problem.has_inputs else "targets"
  input_encoder = problem.feature_info[fname].encoder
  output_decoder = problem.feature_info["targets"].encoder

  stub = create_stub()

  while True:
    prompt = ">> "
    if FLAGS.inputs_once:
      inputs = FLAGS.inputs_once
    else:
      inputs = input(prompt)

    input_ids = encode(inputs, input_encoder)
    output_ids = query(stub, input_ids, feature_name=fname)

    outputs = decode(output_ids, output_decoder)

    print_str = """
Input:
{inputs}

Output:
{outputs}
    """
    print(print_str.format(inputs=inputs, outputs=outputs))
    if FLAGS.inputs_once:
      break


if __name__ == "__main__":
  flags.mark_flags_as_required(
      ["server", "servable_name", "problem", "data_dir"])
  tf.app.run()
