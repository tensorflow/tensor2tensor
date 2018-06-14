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
"""Utilities for serving tensor2tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
from googleapiclient import discovery
from grpc.beta import implementations

from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import cloud_tpu as cloud
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2



def _make_example(input_ids, feature_name="inputs"):
  features = {
      feature_name:
          tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids))
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


def _create_stub(server):
  host, port = server.split(":")
  channel = implementations.insecure_channel(host, int(port))
  # TODO(bgb): Migrate to GA API.
  return prediction_service_pb2.beta_create_PredictionService_stub(channel)


def _encode(inputs, encoder, add_eos=True):
  input_ids = encoder.encode(inputs)
  if add_eos:
    input_ids.append(text_encoder.EOS_ID)
  return input_ids


def _decode(output_ids, output_decoder):
  return output_decoder.decode(output_ids, strip_extraneous=True)




def make_grpc_request_fn(servable_name, server, timeout_secs):
  """Wraps function to make grpc requests with runtime args."""
  stub = _create_stub(server)

  def _make_grpc_request(examples):
    """Builds and sends request to TensorFlow model server."""
    request = predict_pb2.PredictRequest()
    request.model_spec.name = servable_name
    request.inputs["input"].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            [ex.SerializeToString() for ex in examples], shape=[len(examples)]))
    response = stub.Predict(request, timeout_secs)
    outputs = tf.make_ndarray(response.outputs["outputs"])
    scores = tf.make_ndarray(response.outputs["scores"])
    assert len(outputs) == len(scores)
    return [{
        "outputs": outputs[i],
        "scores": scores[i]
    } for i in range(len(outputs))]

  return _make_grpc_request


def make_cloud_mlengine_request_fn(credentials, model_name, version):
  """Wraps function to make CloudML Engine requests with runtime args."""

  def _make_cloud_mlengine_request(examples):
    """Builds and sends requests to Cloud ML Engine."""
    api = discovery.build("ml", "v1", credentials=credentials)
    parent = "projects/%s/models/%s/versions/%s" % (cloud.default_project(),
                                                    model_name, version)
    input_data = {
        "instances": [{
            "input": {
                "b64": base64.b64encode(ex.SerializeToString())
            }
        } for ex in examples]
    }
    prediction = api.projects().predict(body=input_data, name=parent).execute()
    return prediction["predictions"]

  return _make_cloud_mlengine_request


def predict(inputs_list, problem, request_fn):
  """Encodes inputs, makes request to deployed TF model, and decodes outputs."""
  assert isinstance(inputs_list, list)
  fname = "inputs" if problem.has_inputs else "targets"
  input_encoder = problem.feature_info[fname].encoder
  input_ids_list = [
      _encode(inputs, input_encoder, add_eos=problem.has_inputs)
      for inputs in inputs_list
  ]
  examples = [_make_example(input_ids, fname) for input_ids in input_ids_list]
  predictions = request_fn(examples)
  output_decoder = problem.feature_info["targets"].encoder
  outputs = [
      (_decode(prediction["outputs"], output_decoder),
       prediction["scores"])
      for prediction in predictions
  ]
  return outputs
