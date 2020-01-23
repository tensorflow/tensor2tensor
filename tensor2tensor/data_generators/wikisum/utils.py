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

"""Wikisum data generation utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import datetime
import gzip
import os
import re
import urllib

import tensorflow.compat.v1 as tf

# pylint: disable=g-import-not-at-top
# To maintain compatibility with Python 2 and 3
try:
  import cStringIO as StringIO
except ImportError:
  import io as StringIO
# pylint: enable=g-import-not-at-top


# Each entry is a URL to the wet.paths.gz file for that CommonCrawl dump.
WET_PATHS_BY_DATE = {
    '0917': ('https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2017-39/'
             'wet.paths.gz'),
}

S3_HTTP_PREFIX = 'https://commoncrawl.s3.amazonaws.com/'
NUM_SHARDS = 1000
METADTA_SUFFIX = '.metadata.json'



def readahead(path):
  return path


class WETHeader(collections.namedtuple('WETHeader', ['url', 'length'])):
  URI_HEADER = 'WARC-Target-URI: '
  LENGTH_HEADER = 'Content-Length: '

  @classmethod
  def read(cls, f):
    """Read header from file. Headers end with length and then 1 blank line."""
    url = None

    line = f.readline()
    if not line:
      # EOF
      return None
    while not line.startswith(cls.LENGTH_HEADER):
      if line.startswith(cls.URI_HEADER):
        url = line[len(cls.URI_HEADER):].strip()
      line = f.readline()

    # Consume empty separator
    f.readline()

    # Read content
    length = int(line.split(':')[1])

    return cls(url, length)


class WETRecord(collections.namedtuple('WETRecord', ['url', 'content'])):

  @classmethod
  def read(cls, f):
    """Read WETRecord from file. Records end with 2 blank lines."""
    header = WETHeader.read(f)
    if header is None:
      # EOF
      return None
    content = f.read(header.length)

    # Consume empty separators
    f.readline()
    f.readline()

    return cls(header.url, content)


def wet_records_from_file_obj(f, take_ownership=False):
  """Iterate through records in WET file object."""
  while True:
    record = WETRecord.read(f)

    if record is None:
      break

    if not record.url:
      continue

    yield record

  if take_ownership:
    f.close()


def wet_records(wet_filepath):
  """Generate WETRecords from filepath."""
  if wet_filepath.endswith('.gz'):
    fopen = gzip.open
  else:
    fopen = tf.gfile.GFile

  with fopen(wet_filepath) as f:
    for record in wet_records_from_file_obj(f):
      yield record


def download(url, download_dir):
  outname = os.path.join(download_dir, os.path.basename(url))
  if tf.gfile.Exists(outname):
    print('Found %s, skipping download' % outname)
    return outname
  inprogress = outname + '.incomplete'
  print('Downloading %s' % url)
  inprogress, _ = urllib.urlretrieve(url, inprogress)
  tf.gfile.Rename(inprogress, outname)
  return outname


def wet_download_urls(wet_paths_url, tmp_dir, rm_after=True):
  paths_gz = download(wet_paths_url, tmp_dir)
  with gzip.open(paths_gz) as f:
    path = f.readline()
    while path:
      download_path = S3_HTTP_PREFIX + path[:-1]
      yield download_path
      path = f.readline()
  if rm_after:
    tf.gfile.Remove(paths_gz)


def wet_records_from_url(download_url, tmp_dir, rm_after=True):
  wet_gz = download(download_url, tmp_dir)
  try:
    for wet_record in wet_records(wet_gz):
      yield wet_record
  finally:
    if rm_after:
      tf.gfile.Remove(wet_gz)


class DummyPool(object):

  def __init__(self, processes=None):
    pass

  def apply_async(self, fn, args=None):
    args = args or tuple()
    return DummyResult(fn(*args))

  def map(self, fn, arg_list):
    return [fn(a) for a in arg_list]


class DummyResult(object):

  def __init__(self, result):
    self.result = result

  def get(self):
    return self.result


def shard(items, num_shards):
  """Split items into num_shards groups."""
  sharded = []
  num_per_shard = len(items) // num_shards
  start = 0
  for _ in range(num_shards):
    sharded.append(items[start:start + num_per_shard])
    start += num_per_shard

  remainder = len(items) % num_shards
  start = len(items) - remainder
  for i in range(remainder):
    sharded[i].append(items[start + i])

  assert sum([len(fs) for fs in sharded]) == len(items)
  return sharded


def gzip_memfile(fname):
  with tf.gfile.Open(readahead(fname)) as f:
    memfile = StringIO.StringIO(f.read())
  return gzip.GzipFile(fileobj=memfile)


_SOME_ALPHA_RE = re.compile(r'[A-Za-z]+')
_ONLY_ALPHA_RE = re.compile(r'^[A-Za-z]*$')


def filter_paragraph(p):
  """Simple filter to remove obviously bad paragraphs (bad text extraction).

  Note this needs to run very quickly as it is applied to every paragraph
  in the corpus, so nothing fancy! This whole method should be linear
  expected time in len(p).

  Args:
    p: string, paragraph

  Returns:
    True if we should remove the paragraph.
  """
  # Expect a minimum number of words.
  tokens = p.split()
  if len(tokens) < 6:
    return True

  # Require some letters.
  if not re.search(_SOME_ALPHA_RE, p):
    return True

  # Keep this one at the end, probably the most complicated logic.
  # We try to detect sentences, which should have a minimum of 3 tokens
  # with only alphabetic characters.
  last = 0
  found_sentence = False
  num_alpha = 0
  for i, x in enumerate(tokens):
    if x == '.':
      if i - last > 3 and num_alpha >= 3:
        found_sentence = True
        break
      last = i
      num_alpha = 0
    if re.match(_ONLY_ALPHA_RE, x):
      num_alpha += 1
  if not found_sentence:
    return True

  return False


@contextlib.contextmanager
def timing(name=''):
  """Log start, end, and duration."""
  start = datetime.datetime.now()
  timestamp = start.strftime('%H:%M')
  tf.logging.info('Starting job [%s] at %s', name, timestamp)
  yield
  end = datetime.datetime.now()
  timestamp = end.strftime('%H:%M')
  tf.logging.info('Finished job [%s] at %s', name, timestamp)
  duration = end - start
  duration_mins = duration.total_seconds() / 60
  tf.logging.info('Total time [%s] (m): %d', name, int(duration_mins))
