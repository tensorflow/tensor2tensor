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

"""A base class for all query processing classes."""


class QueryProcessor(object):
  """Base class for any class that wants to process sequence queries.

  QueryProcessor classes are expected to convert a string query to a series of
  visualization structures.

  TODO(kstevens): Define how the visualization structures should look once the
  protos are in better shape.
  """

  def __init__(self):
    pass

  def process(self, query):
    """Returns the generated visualizations for query.

    Args:
      query: The string input

    Returns:
      A dictionary with one key: 'result' that maps to a list of visualization
      objects.
    """
    del query
    return {"result": []}
