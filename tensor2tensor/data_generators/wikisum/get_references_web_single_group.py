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
"""Fetch reference URLs for a single group_id within a single shard_id.

See get_references_web.py to fetch URLs for all groups in within a single
shard_id.

Requires Python 3.5
pip3 install aiohttp cchardet aiodns bs4 tensorflow
"""

import datetime
import json
import math
import multiprocessing
import os
import random

import asyncio
import aiohttp
import bs4
import tensorflow as tf

from tensor2tensor.data_generators.wikisum import utils


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("urls_dir", "gs://tensor2tensor-data/wikisum/wiki_urls/",
                    "Directory with wiki_urls.json files.")
flags.DEFINE_string("out_dir", None, "Directory to write reference files.")
flags.DEFINE_integer("max_parallel_requests", 50,
                     "Number of web requests to make in parallel.")

# Identify which URLs to fetch
flags.DEFINE_integer("shard_id", 0, "ID of URL shard to process.")
flags.DEFINE_integer("group_id", 0, "ID of group within the shard to process.")

flags.DEFINE_bool("log_samples", False,
                  "Whether to write out samples of the text extraction.")
flags.DEFINE_integer("log_every", 1000,
                     "How often to log and write out samples.")
flags.DEFINE_integer("debug_num_urls", 0,
                     "If >0, limits number of URLs fetched per input shard. "
                     "For debugging purposes only.")


WIKI_URLS_FILE = "wiki_urls.json-%05d-of-01000"
REF_SHARD_FILE = "references.tfrecords.gz-%05d-of-01000"

# Note that this program leaks memory, likely due to a bug in Python's SSL
# implementation that leaks sockets. This constant is used here and in
# get_references_web.py to limit the number of requests made by a single
# Python process. The more requests made, the more memory required due to the
# leak.
# TODO(rsepassi): Document memory impact of changing this.
URLS_PER_CLIENT = 5000


def concat_tfrecord_files(fnames, out_fname, rm_after=True):
  with tf.gfile.Open(out_fname, "wb") as out_f:
    for fname in fnames:
      with tf.gfile.Open(fname, "rb") as in_f:
        while True:
          read = in_f.read(1000)
          if not read:
            break
          out_f.write(read)
      if rm_after:
        tf.gfile.Remove(fname)


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


def soup_strings(soup):
  paragraph_tags = set(["caption", "details", "h1", "h2", "h3", "h4", "h5",
                        "h6", "li", "p", "td", "div", "span"])

  skip_children = None
  for descendant in soup.descendants:
    # If we've treated a tag as a contiguous paragraph, don't re-emit the
    # children (see below).
    if skip_children is not None:
      try:
        in_skip = descendant in skip_children
      except RecursionError:
        # Possible for this check to hit a nasty infinite recursion because of
        # BeautifulSoup __eq__ checks.
        in_skip = True
      if in_skip:
        continue
      else:
        skip_children = None

    # Treat some tags as contigous paragraphs, regardless of other tags nested
    # inside (like <a> or <b>).
    if isinstance(descendant, bs4.Tag):
      if descendant.name in paragraph_tags:
        if descendant.find_all(paragraph_tags):
          # If there are nested paragraph tags, don't treat it as a single
          # contiguous tag.
          continue
        skip_children = list(descendant.descendants)
        text = " ".join(descendant.get_text(" ", strip=True).split())
        if text:
          yield text
        continue

    if (isinstance(descendant, bs4.Comment) or
        not isinstance(descendant, bs4.NavigableString)):
      continue

    text = " ".join(descendant.strip().split())
    if text:
      yield text


def mp_get_text(url, html):
  return url, get_text_from_html(html)


def get_text_from_html(html):
  try:
    soup = bs4.BeautifulSoup(html, 'html.parser')
  except:
    # Some docs don't parse
    return ""
  # Remove script and style tags
  for s in soup(["script", "style"]):
    s.decompose()
  return "\n".join([s for s in soup_strings(soup)])


def encode(s):
  return bytes(s, "utf-8")


def make_example_from_ref(url, ref):
  try:
    url = encode(url)
    ref = encode(ref)
  except UnicodeEncodeError:
    return None

  features = {
      "url":
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[url])),
      "content":
          tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[ref])),
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


def tfrecord_fname(out_dir, shard_id, idx=None):
  fname = os.path.join(out_dir, REF_SHARD_FILE % shard_id)
  if idx is not None:
    fname += ".%d" % idx
  return fname


def make_tfrecord_writer(fname):
  opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
  return tf.python_io.TFRecordWriter(fname, opts)


def write_ref_content(url, ref, f):
  if not ref:
    return False
  ex = make_example_from_ref(url, ref)
  if ex is None:
    return False
  f.write(ex.SerializeToString())
  return True


async def fetch_url(url, session, side_data):
  text = None
  try:
    async with session.get(url, timeout=10, verify_ssl=False) as response:
      if response.status == 200:
        text = await response.text()
      else:
        tf.logging.error("Status %d, url: %s", response.status, url)
  except:
    # Request can fail for many reasons.
    pass

  return text, side_data


async def throttled_fetch_url(url, sem, session, side_data):
  async with sem:
    return await fetch_url(url, session, side_data)


async def fetch_urls(urls,
                     out_fname,
                     logging_fnames=None):
  tasks = []
  connector = aiohttp.TCPConnector(limit_per_host=1)
  async with aiohttp.ClientSession(
      connector=connector, cookie_jar=aiohttp.DummyCookieJar()) as session:
    # Async fetch the urls
    sem = asyncio.Semaphore(FLAGS.max_parallel_requests)
    for url in urls:
      side_data = {"url": url}
      task = asyncio.ensure_future(
          throttled_fetch_url(url, sem, session, side_data))
      tasks.append(task)
    tf.logging.info("Async requested %d urls", len(urls))

    # Setup output files
    file_handles = []
    out_f = make_tfrecord_writer(out_fname)
    file_handles.append(out_f)

    logging_fnames = logging_fnames or {}

    samples_f = None
    if "samples" in logging_fnames:
      samples_f = tf.gfile.Open(logging_fnames["samples"], "w")
      file_handles.append(samples_f)

    refs_written = [0]  # Made a list so can be mutated

    def text_extraction_callback(callback_arg):
      url, text = callback_arg
      written = write_ref_content(url, text, out_f)
      if not written:
        return
      if not refs_written[0] % FLAGS.log_every:
        timestamp = datetime.datetime.now().strftime("%H:%M")
        tf.logging.info("%s: Wrote ref %d in group", timestamp, refs_written[0])
        if samples_f is not None:
          samples_f.write(url)
          samples_f.write("\n")
          samples_f.write(text)
          samples_f.write("\n\n---\n\n")
      refs_written[0] += 1

    try:
      # Process each URL as it comes in.
      # Using a multiprocessing Pool because the text extraction is expensive
      # and so we distribute across cores.
      pool = multiprocessing.Pool()
      results = []
      for task in asyncio.as_completed(tasks):
        html, side_data = await task
        url = side_data["url"]
        if not html:
          continue
        res = pool.apply_async(mp_get_text, (url, html), {},
                               text_extraction_callback)
        results.append(res)
      for res in results:
        try:
          res.get(timeout=10)
        except multiprocessing.TimeoutError:
          pass
    finally:
      for f in file_handles:
        f.close()

    return refs_written[0]


def get_urls_per_shard(urls_files):
  total_urls = 0
  per_shard = {}
  for urls_file in urls_files:
    ref_urls = set()
    shard_id = int(os.path.basename(urls_file)[15:20])
    with tf.gfile.Open(urls_file) as f:
      wiki_urls = json.loads(f.read())
    for _, wiki_info in wiki_urls.items():
      ref_urls |= set(wiki_info["refs"])

    per_shard[shard_id] = list(ref_urls)
    total_urls += len(ref_urls)
  return per_shard, total_urls


def get_urls_for_shard(urls_dir, shard_id):
  urls_file = os.path.join(urls_dir, WIKI_URLS_FILE % shard_id)
  urls_per_shard, _ = get_urls_per_shard([urls_file])
  assert len(urls_per_shard) == 1
  return urls_per_shard[shard_id]


def get_urls_for_shard_group(urls_dir, shard_id, group_id):
  shard_urls = get_urls_for_shard(urls_dir, shard_id)

  # Deterministic sort and shuffle to prepare for sharding
  shard_urls.sort()
  random.seed(123)
  random.shuffle(shard_urls)
  groups = shard(shard_urls, int(math.ceil(len(shard_urls) / URLS_PER_CLIENT)))
  group_urls = groups[group_id]
  if FLAGS.debug_num_urls:
    group_urls = group_urls[:FLAGS.debug_num_urls]
  return group_urls


def main(_):
  urls = get_urls_for_shard_group(
      FLAGS.urls_dir, FLAGS.shard_id, FLAGS.group_id)
  tf.logging.info("Fetching %d URLs for shard %d, group %d",
                  len(urls), FLAGS.shard_id, FLAGS.group_id)

  tf.gfile.MakeDirs(FLAGS.out_dir)
  out_fname = tfrecord_fname(FLAGS.out_dir, FLAGS.shard_id)

  with utils.timing("group_fetch"):
    logging_fnames = {}
    if FLAGS.log_samples:
      logging_fnames["samples"] = os.path.join(
          FLAGS.out_dir, "samples.%d.txt" % FLAGS.shard_id)
    loop = asyncio.get_event_loop()
    num_written = loop.run_until_complete(asyncio.ensure_future(
        fetch_urls(urls,
                   out_fname,
                   logging_fnames)))

  tf.logging.info("Total URLs: %d", len(urls))
  tf.logging.info("Num written: %d", num_written)
  tf.logging.info("Coverage: %.1f", (num_written / len(urls)) * 100)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
