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
"""Generate Wikipedia Summarization Dataset from CommonCrawl."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import multiprocessing as mp
from multiprocessing import pool as mp_pool
import os
import re
import string
import tempfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators.wikisum_commoncrawl import utils as cc_utils
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
import tensorflow as tf

PROCESS_FOLDER_PREFIX = "process"
REF_SHARD_FILE = "references.tfrecords.gz-%05d-of-01000"

# Support files
BASE_SUPPORT_DIR = "gs://tensor2tensor-data/wikisum"
WIKI_CONTENT_DIR = os.path.join(BASE_SUPPORT_DIR, "wiki_content")
WIKI_URLS_DIR = os.path.join(BASE_SUPPORT_DIR, "wiki_urls")
WET_METADATA_DIR = os.path.join(BASE_SUPPORT_DIR, "commoncrawl_metadata")
WIKI_CONTENT_FILE = "wiki_content.tfrecords-%05d-of-01000"
WIKI_URLS_FILE = "wiki_urls.json-%05d-of-01000"

EOT = "<EOT>"  # end-of-title string
_MIN_REFS = 1
_MIN_LEADSECTION_TOKENS = 1


@registry.register_problem
class WikisumCommoncrawl(problem.Problem):
  """Wikipedia references->article summarization task based on CommonCrawl."""

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
        "section_boundaries": tf.VarLenFeature(tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  @property
  def target_vocab_size(self):
    return 2**15

  @property
  def vocab_filename(self):
    return "vocab.%s.%d" % (self.dataset_filename(), self.target_vocab_size)

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_filename)
    encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    # Shared encoder for inputs and targets
    return {"inputs": encoder, "targets": encoder}

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.stop_at_eos = True

    source_vocab_size = self._encoders["inputs"].vocab_size
    target_vocab_size = self._encoders["targets"].vocab_size
    p.input_modality = {
        "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
    }
    p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)

  def eval_metrics(self):
    return super(WikisumCommoncrawl, self).eval_metrics() + [
        metrics.Metrics.ROUGE_2_F, metrics.Metrics.ROUGE_L_F
    ]

  def generate_lines_for_vocab(self, wikis_dir, refs_dir, max_chars=10**7):
    total_chars = 0
    for shard_id in range(cc_utils.NUM_SHARDS):
      # Wikipedia articles
      for wiki in _wiki_articles(shard_id, wikis_dir):
        yield _normalize_text(wiki.title) + EOT
        for section in wiki.sections:
          yield _format_title(_normalize_text(section.title))
          yield _normalize_text(section.text)
          total_chars += len(section.title)
          total_chars += len(section.text)

      # References
      for i, content in enumerate(
          _references_content_for_shard(shard_id, refs_dir).itervalues()):
        for line in content.split("\n"):
          if line:
            yield _normalize_text(line)
            total_chars += len(line)

        # Make sure we use at least 1k references
        if i >= 1000 and total_chars >= max_chars:
          break

      if total_chars >= max_chars:
        tf.logging.info("Seen enough chars: %d; finished.", max_chars)
        break
    tf.logging.info("Built vocabulary using %d chars", total_chars)

  def generate_vocab(self, data_dir, wikis_dir, refs_dir):
    # Produce a SubwordTextEncoder from a subset of the data
    return generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_filename, self.target_vocab_size,
        self.generate_lines_for_vocab(wikis_dir, refs_dir))

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    """Generate data for Wikipedia summarization based on CommonCrawl.

    To process such a large dataset, this data generator uses multiple processes
    using the `multiprocessing` library and generates data in shards.

    To efficiently produce the dataset from CommonCrawl, there are support files
    that have been pre-generated and are stored on Google Cloud Storage. These
    files will be read during data generation.

    `generate_data` follows this basic outline:
    * Stream through CommonCrawl WET files on AWS S3 (once a WET file has been
      processed it is deleted from local disk, so all files are not required to
      be local all at once).
    * For each file, iterate through its webpages.
    * If the webpage is in the WikisumCommoncrawl dataset (i.e. if it is a
      reference in one of the Wikipedia articles), write it out. Membership in
      the dataset is determined by the support files on GCS.
    * Produce a vocabulary (a `SubwordTextEncoder`, which is a word-piece
      encoder) from a subset of the references and Wikipedia articles.
    * At this point, all references are stored on local disk, all Wikipedia
      articles are on GCS (part of the pre-generated support files), and a vocab
      file is in `data_dir`.
    * Iterate through the Wikipedia articles and read in its references.
    * Rank the reference paragraphs by a tf-idf against the tokens in the
      Wikipedia article title.
    * Encode the reference paragraphs and the Wikipedia article with the vocab.
    * Write out tensorflow.Example protos to TFRecord files, ready for training
      and evaluation.

    Args:
      data_dir: directory to write to.
      tmp_dir: directory to download to, scratch dir.
      task_id: unused.

    Returns:
      None. Training and dev files will have been written to data_dir.
    """
    del task_id

    # Get the download urls for CommonCrawl WET files
    download_urls = list(
        cc_utils.wet_download_urls(cc_utils.WET_PATHS_BY_DATE["0917"], tmp_dir))

    # Extract references from CommonCrawl WET files.
    # Parallel by WET file.
    # Each process writes into its own process_X folder. A reference belongs to
    # one or more shards, determined by a hash of the Wikipedia article titles
    # for which it is a reference. These shard ids have been pre-generated and
    # are part of the support files on GCS.
    num_processes = mp.cpu_count() * 2
    sharded_download_urls = cc_utils.shard(download_urls, num_processes)
    pool = mp_pool.Pool(num_processes)
    pool.map(_extract_references_splat,
             [(sharded_download_urls[i], WET_METADATA_DIR,
               os.path.join(tmp_dir, "%s_%d" %
                            (PROCESS_FOLDER_PREFIX, i)), tmp_dir)
              for i in range(num_processes)])

    # The reference text is now in the process_X folders in the correct shard
    # i.e. a wiki that is part of shard 3 is guaranteed to have all of its
    # references in process_X/shard_00003.tfrecords.gz

    # Produce a SubwordTextEncoder from a subset of the data
    self.generate_vocab(data_dir, WIKI_CONTENT_DIR, tmp_dir)

    # Produce Examples
    # Parallel by output shard
    out_filepaths = self.out_filepaths(data_dir)
    pool.map(_produce_examples_splat, [
        ([shard_id], WIKI_CONTENT_DIR, tmp_dir, WIKI_URLS_DIR,
         os.path.join(data_dir, self.vocab_filename), [out_filepaths[shard_id]])
        for shard_id in range(cc_utils.NUM_SHARDS)
    ])

    # Cleanup intermediate results
    for folder in _process_folders(tmp_dir):
      tf.gfile.DeleteRecursively(folder)

  def out_filepaths(self, data_dir):
    train_shards = 800
    dev_shards = 100
    test_shards = 100
    train_filepaths = self.training_filepaths(
        data_dir, train_shards, shuffled=True)
    dev_filepaths = self.dev_filepaths(data_dir, dev_shards, shuffled=True)
    test_filepaths = self.test_filepaths(data_dir, test_shards, shuffled=True)
    out_filepaths = train_filepaths + dev_filepaths + test_filepaths
    out_filepaths.sort()
    assert len(out_filepaths) == cc_utils.NUM_SHARDS
    return out_filepaths


@registry.register_problem
class WikisumCommoncrawlLeadSection(WikisumCommoncrawl):
  """Wikipedia references->lead section summarization task."""

  def preprocess_example(self, example, mode, hparams):
    wiki = example["targets"]
    lead_boundary = example["section_boundaries"][0]
    # Concat a new EOS to the lead since the original one gets truncated.
    lead = tf.concat((wiki[:lead_boundary], [text_encoder.EOS_ID]), 0)
    example["targets"] = lead
    return super(WikisumCommoncrawlLeadSection, self).preprocess_example(
        example, mode, hparams)

  def dataset_filename(self):
    return WikisumCommoncrawl.name

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    tf.logging.warn("Problem %s reuses data from problem %s", self.name,
                    WikisumCommoncrawl.name)


def make_ref_shard_files(out_dir):
  tf.gfile.MakeDirs(out_dir)
  opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
  files = [
      tf.python_io.TFRecordWriter(
          os.path.join(out_dir, REF_SHARD_FILE % i), opts)
      for i in range(cc_utils.NUM_SHARDS)
  ]
  return files


def _make_example_from_record(record):
  features = {
      "url":
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[record.url])),
      "content":
          tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[record.content])),
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


def _references_content_for_shard(shard_id, refs_dir, rm_after=False):
  """dict<str ref_url, str ref_content>."""
  with tf.Graph().as_default():
    shard_file_paths = [
        cc_utils.readahead(os.path.join(folder, REF_SHARD_FILE % shard_id))
        for folder in _process_folders(refs_dir)
    ]
    dataset = tf.data.Dataset.from_tensor_slices(shard_file_paths)

    def _load_records(filename):
      return tf.data.TFRecordDataset(
          filename,
          compression_type=tf.constant("GZIP"),
          buffer_size=16 * 1000 * 1000)

    dataset = dataset.flat_map(_load_records)

    def _parse_example(ex_ser):
      features = {
          "url": tf.VarLenFeature(tf.string),
          "content": tf.VarLenFeature(tf.string),
      }
      ex = tf.parse_single_example(ex_ser, features)
      for k in ex.keys():
        ex[k] = ex[k].values[0]
      return ex

    dataset = dataset.map(_parse_example, num_parallel_calls=32)
    dataset = dataset.prefetch(100)
    record_it = dataset.make_one_shot_iterator().get_next()

    data = {}

    with tf.Session() as sess:
      i = 0
      while True:
        try:
          ex = sess.run(record_it)
        except tf.errors.OutOfRangeError:
          break

        data[ex["url"]] = ex["content"]
        i += 1

  if not shard_id % 100:
    tf.logging.info("Read %d refs in shard %d" % (i + 1, shard_id))

  if rm_after:
    for f in shard_file_paths:
      tf.gfile.Remove(f)

  return data


def _wiki_urls_for_shard(shard_id, urls_dir=None):
  """Urls for chunk: dict<str wiki_url, list<str> ref_urls>."""
  urls_dir = urls_dir or WIKI_URLS_DIR
  urls_filepath = os.path.join(urls_dir, WIKI_URLS_FILE % shard_id)
  with tf.gfile.GFile(urls_filepath) as f:
    return json.loads(f.read())


class WikipediaSection(
    collections.namedtuple("WikipediaSection", ["title", "text"])):
  pass


class WikipediaArticle(
    collections.namedtuple("WikipediaArticle", ["url", "title", "sections"])):
  pass


def _wiki_articles(shard_id, wikis_dir=None):
  """Generates WikipediaArticles from GCS that are part of shard shard_id."""
  if not wikis_dir:
    wikis_dir = WIKI_CONTENT_DIR
  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset(
        cc_utils.readahead(
            os.path.join(wikis_dir, WIKI_CONTENT_FILE % shard_id)),
        buffer_size=16 * 1000 * 1000)

    def _parse_example(ex_ser):
      """Parse serialized Example containing Wikipedia article content."""
      features = {
          "url": tf.VarLenFeature(tf.string),
          "title": tf.VarLenFeature(tf.string),
          "section_titles": tf.VarLenFeature(tf.string),
          "section_texts": tf.VarLenFeature(tf.string),
      }
      ex = tf.parse_single_example(ex_ser, features)
      for k in ex.keys():
        ex[k] = ex[k].values
      ex["url"] = ex["url"][0]
      ex["title"] = ex["title"][0]
      return ex

    dataset = dataset.map(_parse_example, num_parallel_calls=32)
    dataset = dataset.prefetch(100)
    record_it = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      while True:
        try:
          ex = sess.run(record_it)
        except tf.errors.OutOfRangeError:
          break

        sections = [
            WikipediaSection(title=title, text=text)
            for title, text in zip(ex["section_titles"], ex["section_texts"])
        ]
        yield WikipediaArticle(
            url=ex["url"], title=ex["title"], sections=sections)


def _token_counts(text, token_set=None):
  counts = collections.defaultdict(int)
  for token in tokenizer.encode(text_encoder.native_to_unicode(text)):
    if token_set and token not in token_set:
      continue
    counts[token] += 1
  return counts


def _normalize_text(text):
  text = text.lower()
  # Space around punctuation
  text = re.sub("[%s]" % re.escape(string.punctuation), r" \g<0> ", text)
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  return text


def _tokens_to_score(tokens):
  return {t for t in tokens if re.search("[a-z0-9]", t)}


def _rank_reference_paragraphs(wiki_title, references_content):
  """Rank and return reference paragraphs by tf-idf score on title tokens."""
  title_tokens = _tokens_to_score(set(
      tokenizer.encode(text_encoder.native_to_unicode(wiki_title))))
  ref_paragraph_info = []
  doc_counts = collections.defaultdict(int)
  for ref in references_content:
    for paragraph in ref.split("\n"):
      paragraph = _normalize_text(paragraph)
      if cc_utils.filter_paragraph(paragraph):
        # Skip paragraph
        continue
      counts = _token_counts(paragraph, title_tokens)
      for token in title_tokens:
        if counts[token]:
          doc_counts[token] += 1
      info = {"content": paragraph, "counts": counts}
      ref_paragraph_info.append(info)

  for info in ref_paragraph_info:
    score = 0.
    for token in title_tokens:
      term_frequency = info["counts"][token]
      inv_doc_frequency = (
          float(len(ref_paragraph_info)) / max(doc_counts[token], 1))
      score += term_frequency * math.log(inv_doc_frequency)
    info["score"] = score

  ref_paragraph_info.sort(key=lambda el: el["score"], reverse=True)
  return [info["content"] for info in ref_paragraph_info]


def _produce_examples_splat(args):
  return produce_examples(*args)


def produce_examples(shard_ids, wikis_dir, refs_dir, urls_dir, vocab_path,
                     out_filepaths):
  """Produce examples from shard_ids to out_filepaths."""
  # * Join the Wikipedia articles with their references
  # * Run Tf-idf to sort reference paragraphs
  # * Encode the Wikipedia and reference text with the vocabulary
  # * Write out TFRecords of tensorflow.Example
  tf.logging.info("Processing %d input shards into %d output files.",
                  len(shard_ids), len(out_filepaths))

  vocab = text_encoder.SubwordTextEncoder(vocab_path)
  eot_ids = vocab.encode(EOT)

  def example_generator():
    """Generate Example dicts."""
    stats = dict(total_original_wikis=0, total_original_refs=0,
                 total_found_refs=0, ref_lengths=[], wiki_original_refs=[],
                 wiki_found_refs=[], wikis_skipped_no_refs=0,
                 wikis_skipped_short_lead=0, num_wikis_written=0)
    for shard_id in shard_ids:
      tf.logging.info("Processing shard %d", shard_id)
      wiki_urls = _wiki_urls_for_shard(shard_id, urls_dir)
      tf.logging.info("Loaded wiki URLs for shard")
      refs_content = _references_content_for_shard(shard_id, refs_dir)
      tf.logging.info("Loaded reference content for shard")
      for i, wiki in enumerate(_wiki_articles(shard_id, wikis_dir)):
        if not i % 1000:
          tf.logging.info("Processing wiki index %d for shard %d", i, shard_id)
        stats["total_original_wikis"] += 1

        # Get reference content
        wiki_ref_content = []
        ref_urls = wiki_urls[wiki.url]["refs"]
        stats["total_original_refs"] += len(ref_urls)
        stats_wiki_original_refs = len(ref_urls)
        stats_wiki_found_refs = 0
        for ref_url in ref_urls:
          ref_content = refs_content.get(ref_url)
          if not ref_content:
            continue
          stats["total_found_refs"] += 1
          stats["ref_lengths"].append(len(ref_content))
          stats_wiki_found_refs += 1
          wiki_ref_content.append(ref_content)

        stats["wiki_original_refs"].append(stats_wiki_original_refs)
        stats["wiki_found_refs"].append(stats_wiki_found_refs)
        if not wiki_ref_content or len(wiki_ref_content) < _MIN_REFS:
          # No/few refs were found
          stats["wikis_skipped_no_refs"] += 1
          continue

        # Rank reference paragraphs with TFIDF
        wiki_title = _normalize_text(wiki.title)
        ranked_paragraphs = _rank_reference_paragraphs(wiki_title,
                                                       wiki_ref_content)

        # Construct inputs from Wiki title and references
        inputs = []
        inputs.extend(vocab.encode(wiki_title))
        inputs.extend(eot_ids)
        for paragraph in ranked_paragraphs:
          if len(inputs) >= 1e6:
            break
          paragraph += " "
          inputs.extend(vocab.encode(paragraph))

        # Construct targets from article sections
        targets, section_boundaries = _encode_wiki_sections(
            wiki.sections, vocab)

        # Skip if lead section is too short
        if (not section_boundaries or
            section_boundaries[0] < _MIN_LEADSECTION_TOKENS):
          stats["wikis_skipped_short_lead"] += 1
          continue

        inputs.append(text_encoder.EOS_ID)
        targets.append(text_encoder.EOS_ID)

        stats["num_wikis_written"] += 1
        yield {
            "inputs": inputs,
            "targets": targets,
            "section_boundaries": section_boundaries,
        }

    tf.logging.info("Total: %d, Skipped: %d",
                    stats["num_wikis_written"],
                    stats["total_original_wikis"] - stats["num_wikis_written"])
    tf.logging.info("Total refs: %d, Skipped refs: %d",
                    stats["total_found_refs"],
                    stats["total_original_refs"] - stats["total_found_refs"])
    stats_fname = os.path.join(os.path.split(out_filepaths[0])[0],
                               "stats.%d.json" % shard_ids[0])
    with tf.gfile.Open(stats_fname, "w") as f:
      f.write(json.dumps(stats))

  generator_utils.generate_files(example_generator(), out_filepaths)


def _format_title(title):
  return " == %s == " % title


def _encode_wiki_sections(sections, vocab):
  """Encodes sections with vocab. Returns ids and section boundaries."""
  ids = []
  section_boundaries = []
  for i, section in enumerate(sections):
    if i > 0:
      # Skip including article title
      ids.extend(vocab.encode(_format_title(_normalize_text(section.title))))
    ids.extend(vocab.encode(_normalize_text(section.text)))
    section_boundaries.append(len(ids))

  return ids, section_boundaries


def _process_folders(tmp_dir):
  return tf.gfile.Glob(os.path.join(tmp_dir, PROCESS_FOLDER_PREFIX) + "*")


def _extract_references_splat(args):
  return extract_references_from_wets(*args)


def extract_references_from_wets(wet_files, metadata_dir, out_dir,
                                 tmp_dir=None):
  """Extract references from WET files into sharded output files."""
  # Setup output files
  shard_files = make_ref_shard_files(out_dir)

  num_refs = 0
  for i, wet_file in enumerate(wet_files):
    num_refs_in_wet = 0
    tf.logging.info("Processing file %d", i)

    # Read metadata file
    metadata_fname = os.path.join(
        metadata_dir, os.path.basename(wet_file)) + cc_utils.METADTA_SUFFIX
    with tf.gfile.Open(cc_utils.readahead(metadata_fname)) as f:
      wet_metadata = json.loads(f.read())

    if not wet_metadata:
      # No references in this WET file
      continue

    if wet_file.startswith("http"):
      # download
      if not tmp_dir:
        tmp_dir = tempfile.gettempdir()
      record_gen = cc_utils.wet_records_from_url(wet_file, tmp_dir)
    else:
      # local
      record_gen = cc_utils.wet_records_from_file_obj(
          cc_utils.gzip_memfile(wet_file), take_ownership=True)

    for wet_record in record_gen:
      shard_ids = wet_metadata.get(wet_record.url)
      if not shard_ids:
        # URL not in dataset
        continue

      # Serialize and write out
      ex = _make_example_from_record(wet_record)
      ex_str = ex.SerializeToString()
      for shard_id in shard_ids:
        shard_files[shard_id].write(ex_str)
      num_refs += 1
      num_refs_in_wet += 1

    tf.logging.info("Wrote out %d references for this WET", num_refs_in_wet)

  tf.logging.info("Wrote out %d references total", num_refs)

  # Cleanup
  for shard_file in shard_files:
    shard_file.close()
