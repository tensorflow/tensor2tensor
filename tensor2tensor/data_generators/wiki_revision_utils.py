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

"""Utilties for data generation for Wikipedia Revision problem.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import re
import subprocess

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_encoder

import tensorflow.compat.v1 as tf


def include_revision(revision_num, skip_factor=1.1):
  """Decide whether to include a revision.

  If the number of revisions is large, we exclude some revisions to avoid
  a quadratic blowup in runtime, since the article is likely also large.

  We make the ratio between consecutive included revision numbers
  appproximately equal to "factor".

  Args:
    revision_num: an integer
    skip_factor: a floating point number >= 1.0

  Returns:
    a boolean
  """
  if skip_factor <= 1.0:
    return True
  return (int(math.log1p(revision_num) / math.log(skip_factor)) != int(
      math.log(revision_num + 2.0) / math.log(skip_factor)))


def file_page_generator(my_file, max_page_size=2**28):
  """Read wikipedia pages from a history dump.

  Since some pages can be terabytes in size (with all the revisions),
  we limit page size to max_page_size bytes.

  Args:
    my_file: an open file object.
    max_page_size: an integer

  Yields:
    strings
  """
  page_start = "  <page>\n"
  page_end = "  </page>\n"
  chunk_size = max_page_size
  page_start = "  <page>\n"
  page_end = "  </page>\n"
  leftovers = ""
  while True:
    chunk = my_file.read(chunk_size)
    if not chunk:
      break
    chunk = leftovers + chunk
    current_pos = 0
    while True:
      start_pos = chunk.find(page_start, current_pos)
      if start_pos == -1:
        break
      end_pos = chunk.find(page_end, start_pos)
      if end_pos == -1:
        if len(chunk) - start_pos > max_page_size:
          leftovers = ""
        else:
          leftovers = chunk[start_pos:]
        break
      raw_page = chunk[start_pos + len(page_start):end_pos]
      if len(raw_page) < max_page_size:
        ret = parse_page(raw_page)
        if ret:
          yield ret
      current_pos = end_pos + len(page_end)


def get_title(page):
  """Extract the title from a page.

  Args:
    page: a string
  Returns:
    a string
  """
  start_pos = page.find("<title>")
  end_pos = page.find("</title>")
  assert start_pos != -1
  assert end_pos != -1
  start_pos += len("<title>")
  return text_encoder.to_unicode_utf8(page[start_pos:end_pos])


def get_id(page):
  """Extract the id from a page.

  Args:
    page: a string
  Returns:
    an integer
  """
  start_pos = page.find("<id>")
  end_pos = page.find("</id>")
  assert start_pos != -1
  assert end_pos != -1
  start_pos += len("<id>")
  return int(page[start_pos:end_pos])


def get_revisions(page):
  """Extract the revisions of a page.

  Args:
    page: a string
  Returns:
    a list of strings
  """
  start_string = "    <revision>\n"
  end_string = "    </revision>\n"
  ret = []
  current_pos = 0
  while True:
    start_pos = page.find(start_string, current_pos)
    if start_pos == -1:
      break
    end_pos = page.find(end_string, start_pos)
    assert end_pos != -1
    ret.append(page[start_pos + len(start_string):end_pos])
    current_pos = end_pos + len(end_string)
  return ret


def parse_page(raw_page):
  """Create a dictionary with title, id, and list of revisions.

  The dictionary contains:
  "title": a string
  "id": an integer
  "revisions": a list of strings

  Args:
    raw_page: a string

  Returns:
    a dictionary, or None in the case of an error.
  """
  ret = {"title": get_title(raw_page), "id": get_id(raw_page)}
  if ":" in ret["title"]:
    return None
  ret["revisions"] = get_revisions(raw_page)
  return ret


def maybe_copy_file_to_directory(source_filepath, target_directory):
  """Copy a file to a directory if it is not already there.

  Returns the target filepath.

  Args:
    source_filepath: a string
    target_directory: a string

  Returns:
    a string
  """
  if not tf.gfile.Exists(target_directory):
    tf.logging.info("Creating directory %s" % target_directory)
    os.mkdir(target_directory)
  target_filepath = os.path.join(target_directory,
                                 os.path.basename(source_filepath))
  if not tf.gfile.Exists(target_filepath):
    tf.logging.info("Copying %s to %s" % (source_filepath, target_filepath))
    tf.gfile.Copy(source_filepath, target_filepath)
    statinfo = os.stat(target_filepath)
    tf.logging.info("Successfully copied %s, %s bytes." % (target_filepath,
                                                           statinfo.st_size))
  else:
    tf.logging.info("Not copying, file already found: %s" % target_filepath)
  return target_filepath


def corpus_page_generator(corpus_files, tmp_dir, max_page_size_exp):
  """Generate pages from a list of .7z encoded history dumps.

  Args:
    corpus_files: a list of strings
    tmp_dir: a string
    max_page_size_exp: an integer

  Yields:
    strings
  """
  for remote_filepath in corpus_files:

    filepath = maybe_copy_file_to_directory(remote_filepath, tmp_dir)
    tf.logging.info("Reading from " + filepath)

    command = ["7z", "x", "-so", filepath]
    tf.logging.info("Running command: %s", command)

    p = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=-1)

    for page in file_page_generator(p.stdout, 2**max_page_size_exp):
      yield page


def get_text(revision, strip=True):
  """Extract the text from a revision.

  Args:
    revision: a string
    strip: a boolean

  Returns:
    a string
  """
  # text start tag looks like "<text ..otherstuff>"
  start_pos = revision.find("<text")
  assert start_pos != -1
  end_tag_pos = revision.find(">", start_pos)
  assert end_tag_pos != -1
  end_tag_pos += len(">")
  end_pos = revision.find("</text>")
  if end_pos == -1:
    ret = ""
  else:
    ret = revision[end_tag_pos:end_pos]
  if strip:
    ret = strip_text(ret)
  ret = text_encoder.to_unicode_utf8(ret)
  return ret


def strip_text(text):
  """Strip wikipedia-stuff out of text, making it mostly prose.

  The reason for this is to learn a model that is good at editing prose.

  Args:
    text: a string

  Returns:
    a string
  """
  return _remove_boring_lines(
      _remove_triple_quotes(
          _remove_double_brackets(
              _remove_references(_remove_curly_braces(text)))))


def _find_and_replace(text, start_string, end_string, replace_fn):
  """Remove everything found between instances of start_string and end_string.

  Replace each such instance with replace_fn(removed_text)

  e.g. _find_and_replace("the [[fat]] cat [[sat]]", "[[", "]]", lambda x: x)
    = "the fat cat sat"

  Args:
    text: a string
    start_string: a string
    end_string: a string
    replace_fn: a unary function from string to string

  Returns:
    a string
  """
  ret = ""
  current_pos = 0
  while True:
    start_pos = text.find(start_string, current_pos)
    if start_pos == -1:
      ret += text[current_pos:]
      break
    ret += text[current_pos:start_pos]
    end_pos = text.find(end_string, start_pos + len(start_string))
    if end_pos == -1:
      break
    ret += replace_fn(text[start_pos + len(start_string):end_pos])
    current_pos = end_pos + len(end_string)
  return ret


def _remove_references(text):
  return _find_and_replace(text, "&lt;ref", "&lt;/ref&gt;", lambda s: "")


def _remove_triple_quotes(text):
  return _find_and_replace(text, "'''", "'''", lambda s: s)


def _remove_curly_braces(text):
  """Remove everything in curly braces.

  Curly braces may be nested, so we keep track of depth.

  Args:
    text: a string
  Returns:
    a string
  """
  current_pos = 0
  depth = 0
  ret = ""
  for match in re.finditer("[{}]", text):
    if depth == 0:
      ret += text[current_pos:match.start()]
    depth += 1 if text[match.start()] == "{" else -1
    current_pos = match.end()
  if depth != 0:
    # Many articles have mismatched braces, but it still seems better to remove
    # them than not.
    pass
  else:
    ret += text[current_pos:]
  return ret


def _remove_double_brackets(text):
  """Remove double brackets, but leave the viewable text.

  Args:
    text: a string
  Returns:
    a string
  """

  def replacement_fn(s):
    if ":" in s:
      # this is probably a category or something like that.
      return ""
    # keep the part after the bar.
    bar_pos = s.find("|")
    if bar_pos == -1:
      return s
    return s[bar_pos + 1:]

  return _find_and_replace(text, "[[", "]]", replacement_fn)


def _remove_boring_lines(text):
  """Remove lines that do not start with a letter or a quote.

  From inspecting the data, this seems to leave in most prose and remove
  most weird stuff.

  Args:
    text: a string
  Returns:
    a string
  """
  lines = text.split("\n")
  filtered = [line for line in lines if re.match("[a-zA-z\"\']", line)]
  return "\n".join(filtered)


def all_corpus_files(data_prefix):
  return sorted(tf.gfile.Glob(data_prefix + "*"))


def corpus_files_for_shard(shard_num, train_shards, dev_shards, data_prefix):
  corpus_files = [
      filename for i, filename in enumerate(all_corpus_files(data_prefix))
      if i % (train_shards + dev_shards) == shard_num
  ]
  tf.logging.info("Corpus files for shard %s: %s", shard_num, corpus_files)

  assert shard_num < (train_shards + dev_shards)
  return corpus_files


def vocab_filename(approx_vocab_size, strip):
  return "vocab.wiki_revision%s.%d" % (".strip" if strip else "",
                                       approx_vocab_size)


def get_or_generate_vocabulary(data_dir,
                               tmp_dir,
                               data_prefix,
                               max_page_size_exp,
                               approx_vocab_size=32768,
                               strip=True):
  """Get or generate the vocabulary.

  Args:
    data_dir: a string
    tmp_dir: a string
    data_prefix: a string
    max_page_size_exp: an integer
    approx_vocab_size: an integer
    strip: a boolean

  Returns:
    a TextEncoder
  """
  num_pages_for_vocab_generation = approx_vocab_size // 3
  vocab_file = vocab_filename(approx_vocab_size, strip)

  def my_generator(data_prefix):
    """Line generator for vocab."""
    count = 0
    for page in corpus_page_generator(
        all_corpus_files(data_prefix)[::-1], tmp_dir, max_page_size_exp):
      revisions = page["revisions"]
      if revisions:
        text = get_text(revisions[-1], strip=strip)
        yield text
        count += 1
        if count % 100 == 0:
          tf.logging.info("reading pages for vocab %d" % count)
        if count > num_pages_for_vocab_generation:
          break

  return generator_utils.get_or_generate_vocab_inner(data_dir, vocab_file,
                                                     approx_vocab_size,
                                                     my_generator(data_prefix))


def get_encoder_from_vocab(vocab_filepath):
  """Get encoder from vocab file.

  If vocab is not found in output dir, it will be copied there by
  copy_vocab_to_output_dir to clarify the vocab used to generate the data.

  Args:
    vocab_filepath: path to vocab, either local or cns

  Returns:
    A SubwordTextEncoder vocabulary object. None if the output_parallel_text
    is set.
  """
  if not tf.gfile.Exists(vocab_filepath):
    raise ValueError("Vocab file does not exist: {}.".format(vocab_filepath))

  tf.logging.info("Found vocab file: %s", vocab_filepath)
  encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
  return encoder


def throw_empty_pairs(src_tgt_pairs):
  """Filter [src,tgt] tuple from input list of pairs if either element is empty.

  Args:
    src_tgt_pairs: list of (src,tgt) pairs

  Returns:
    subset of input pair list for which all elements are non-empty
  """
  return [x for x in src_tgt_pairs if x[0] and x[1]]


def edit_distance_filter(source_target_input, max_equal_to_diff_ratio=0):
  """Filter out examples that exceed max_edit_ratio between source and target.

  Args:
    source_target_input:     a list of [source, target] pairs
    max_equal_to_diff_ratio: cutoff for ratio of equal chars / diff chars
      between source and target

  Returns:
    source_target_output:    filtered subset of [source, target] input pairs
    thrown_out_count:        number of examples filtered out
  """
  thrown_out_count = 0
  source_target_output = []

  if not max_equal_to_diff_ratio:
    return source_target_input, thrown_out_count

  for src_tgt in source_target_input:
    opcodes = fast_match_sequences(*src_tgt)
    diff_char_count = 0
    equal_char_count = 0
    for tag, i1, i2, j1, j2 in opcodes:
      if tag == "diff":
        # max() prevents double-counting substitutions.
        diff_char_count += max(i2 - i1, j2 - j1)
      else:
        equal_char_count += i2 - i1
    if diff_char_count <= max_equal_to_diff_ratio * equal_char_count:
      source_target_output.append(src_tgt)
    else:
      thrown_out_count += 1
  return source_target_output, thrown_out_count


def introduce_errors(s,
                     corruption_rate=3e-3,
                     infill_marker="|?|",
                     max_infill_len=8):
  """Artificially add spelling errors and infill markers.

  This function should be applied to the inputs of a correction model.

  The artificial errors are particularly useful to train a network to
  correct spelling when the training data does not contain many
  natural errors.

  Also replaces some substrings with an "infill" marker.  e.g.
  "the fat cat sat on the mat" -> "the fat ca??? the mat"

  This causes the trained model to learn infilling (predicting what text
  to insert at the current cursor position).

  Args:
    s: a string (the uncorrupted text)
    corruption_rate: a floating point value.  Probability of introducing an
      error/infill at each character.
    infill_marker: a string
    max_infill_len: an optional integer - maximum number of characters to remove
      and replace by an infill marker.  None means no infilling.

  Returns:
    a string
  """
  num_errors = 0
  ret = []
  operations = [
      "delete",  # delete a character
      "insert",  # insert a random character from the input string
      "replace",  # replace a character with a random character from
      #   the input string
      "transpose",  # transpose two adjacent characters
  ]
  if max_infill_len:
    operations.append("infill")
  pos = 0
  while pos < len(s):
    if random.random() >= corruption_rate:
      ret.append(s[pos])
      pos += 1
      continue
    num_errors += 1
    operation = operations[random.randint(0, len(operations) - 1)]
    if operation == "delete":
      pos += 1
    elif operation == "insert":
      ret.append(s[random.randint(0, len(s) - 1)])
    elif operation == "replace":
      ret.append(s[random.randint(0, len(s) - 1)])
      pos += 1
    elif operation == "transpose":
      ret.append(s[pos + 1] if pos + 1 < len(s) else "")
      ret.append(s[pos])
      pos += 2
    else:
      assert operation == "infill"
      ret.append(infill_marker)
      pos += random.randint(0, max_infill_len)
  return "".join(ret), num_errors


def fast_match_sequences(a,
                         b,
                         a_start=0,
                         a_end=None,
                         b_start=0,
                         b_end=None,
                         min_match_length=3,
                         max_recursion_depth=128):
  """Compute diffs between two sequences.

  This function is similar in functionality and spirit to
  difflib.SequenceMatcher.get_opcodes, but it seems to run faster.

  if a_start, a_end, b_start, b_end are specified, then we compute diffs of
  the segments a[a_start:a_end] and b[b_start:b_end].  Returned indices
  are relative to the full sequence.

  We try to match the longest matching segments first, but due to heuristics
  in finding the matches, this is not guaranteed.

  Matching segments shorter than min_match_length are counted as part of the
  surrounding differing segments, unless they are at the beginning or end of
  both sequences.  This helps eliminate junk matches.

  Args:
    a: a sequence
    b: a sequence
    a_start: an optional integer
    a_end: an optional integer
    b_start: an optional integer
    b_end: an optional integer
    min_match_length: an integer
    max_recursion_depth: an integer - avoids crashes in weird corner cases
      involving pairs of long repetitive sequences.

  Returns:
    a list of 5-tuples (tag, i1, i2, j1, j2).
    Each tuple represents the alignment of segment a[i1:i2] with b[j1:j2].
      tag is either "equal" or "diff".  Note that the tags differ from those
      returned by difflib.SequenceMatcher.get_opcodes.
  """
  if a_end is None:
    a_end = len(a)
  if b_end is None:
    b_end = len(b)
  if a_start == a_end and b_start == b_end:
    return []
  if a_start == a_end or b_start == b_end:
    return [("diff", a_start, a_end, b_start, b_end)]
  # Compute an index from value to first occurrence in the b segment.
  # Technically, we should index and explore all occurrences of a value,
  # but that might be much slower.
  b_index = {}
  for j in range(b_end - 1, b_start - 1, -1):
    b_index[b[j]] = j
  # we will look for the longest match we can find.
  max_match_length = 0
  a_pos = a_start
  while a_pos < a_end:
    val = a[a_pos]
    b_pos = b_index.get(val)
    if b_pos is None:
      a_pos += 1
      continue
    else:
      a_match_start = a_pos
      a_match_end = a_pos + 1
      b_match_start = b_pos
      b_match_end = b_pos + 1
      while (a_match_start > a_start and b_match_start > b_start and
             a[a_match_start - 1] == b[b_match_start - 1]):
        a_match_start -= 1
        b_match_start -= 1
      while (a_match_end < a_end and b_match_end < b_end and
             a[a_match_end] == b[b_match_end]):
        a_match_end += 1
        b_match_end += 1
      # Compute the length of the matching segment.  We prefer the longest.
      match_length = a_match_end - a_match_start
      # Extra credit for matching at the beginning or end of the sequence.
      if a_match_start == 0 and b_match_start == 0:
        match_length += min_match_length
      if a_match_end == len(a) and b_match_end == len(b):
        match_length += min_match_length
      if match_length > max_match_length:
        max_match_length = match_length
        best_match = (a_match_start, a_match_end, b_match_start, b_match_end)
      # advance a_pos to the end of this match to avoid wasting time
      # rediscovering this match.
      a_pos = a_match_end
  if max_match_length < min_match_length or max_recursion_depth == 0:
    return [("diff", a_start, a_end, b_start, b_end)]
  a_match_start, a_match_end, b_match_start, b_match_end = best_match
  return (fast_match_sequences(
      a, b, a_start, a_match_start, b_start, b_match_start, min_match_length,
      max_recursion_depth - 1) + [
          ("equal", a_match_start, a_match_end, b_match_start, b_match_end)
      ] + fast_match_sequences(a, b, a_match_end, a_end, b_match_end, b_end,
                               min_match_length, max_recursion_depth - 1))
