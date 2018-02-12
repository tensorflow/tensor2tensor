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

"""Graph representation for building decoding graph visualizations."""


class Vertex(object):
  """Vertex stores in and out edge connections to other Vertex instances.

  The Vertex class supports serialization to a JSON data format expected by the
  client side representation.  When serializing, it generates the following
  fields:
    in_edge_index: The list of directed edge indices into the Vertex.
    out_edge_index: The list of directed edge indices from the Vertex.
  """

  def __init__(self, idx):
    """Initialize the Vertex.

    Args:
      idx: The index of the vertex.
    """
    self.idx = idx
    self.in_edges = []
    self.out_edges = []

  def to_dict(self):
    """Returns a simplified dictionary representing the Vertex.

    Returns:
      A dictionary that can easily be serialized to JSON.
    """
    return {
        "in_edge_index": self.in_edges,
        "out_edge_index": self.out_edges,
    }


class Edge(object):
  """Edge stores edge details connecting two Vertex instances.

  The Edge class supports serialization to a JSON data format expected by the
  client side representation.  When serializing, it generates the following
  fields:
    source_index: The source Vertex index for this Edge.
    target_index: The target Vertex index for this Edge.
    data: Arbitrary data for this Edge.
  """

  def __init__(self, idx):
    """Initialize the Edge.

    Args:
      idx: The index of the Edge.
    """
    self.idx = idx
    self.source = -1
    self.target = -1
    self.data = {}

  def to_dict(self):
    """Returns a simplified dictionary representing the Vertex.

    Returns:
      A dictionary that can easily be serialized to JSON.
    """
    return {
        "source_index": self.source,
        "target_index": self.target,
        "data": self.data,
    }

  def __str__(self):
    return str(self.to_dict())


class Graph(object):
  """A directed graph that can easily be JSON serialized for visualization.

  When serializing, it generates the following fields:
    edge: The list of all serialized Edge instances.
    node: The list of all serialized Vertex instances.
  """

  def __init__(self):
    self.vertices = []
    self.edges = []
    self.vertex_map = {}

  def new_vertex(self):
    """Creates and returns a new vertex.

    Returns:
      A new Vertex instance with a unique index.
    """
    vertex = Vertex(len(self.vertices))
    self.vertices.append(vertex)
    return vertex

  def get_vertex(self, key):
    """Returns or Creates a Vertex mapped by key.

    Args:
      key: A string reference for a vertex.  May refer to a new Vertex in which
      case it will be created.

    Returns:
      A the Vertex mapped to by key.
    """
    if key in self.vertex_map:
      return self.vertex_map[key]
    vertex = self.new_vertex()
    self.vertex_map[key] = vertex
    return vertex

  def add_edge(self, source, target):
    """Returns a new edge connecting source and target vertices.

    Args:
      source: The source Vertex.
      target: The target Vertex.

    Returns:
      A new Edge linking source to target.
    """
    edge = Edge(len(self.edges))
    self.edges.append(edge)
    source.out_edges.append(edge.idx)
    target.in_edges.append(edge.idx)
    edge.source = source.idx
    edge.target = target.idx
    return edge

  def to_dict(self):
    """Returns a simplified dictionary representing the Graph.

    Returns:
      A dictionary that can easily be serialized to JSON.
    """
    return {
        "node": [v.to_dict() for v in self.vertices],
        "edge": [e.to_dict() for e in self.edges]
    }
