# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Utilities for env_service_server.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import logging
from concurrent import futures
import grpc

from tensor2tensor.envs import env_service_pb2_grpc
from tensor2tensor.envs import env_service_servicer

# Since we're only dealing with 1 GPU machines here.
_MAX_CONCURRENCY = 1
_ADDRESS_FORMAT = "[::]:{}"


def add_port(server, port):
  return server.add_insecure_port(_ADDRESS_FORMAT.format(port))  # pylint: disable=unreachable


def serve(output_dir, env, port):
  del output_dir  # may use later.
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=_MAX_CONCURRENCY))
  servicer = env_service_servicer.EnvServiceServicer(env)
  env_service_pb2_grpc.add_EnvServiceServicer_to_server(servicer, server)
  serving_port = add_port(server, port)
  server.start()
  logging.info("Starting server on port %s", serving_port)
  while True:
    time.sleep(60 * 60 * 24)  # sleep for a day only to sleep again.
