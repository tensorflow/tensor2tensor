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

"""Utils to parse HTML content into plaintext."""

import bs4


def get_text_from_html(html):
  """Returns a plaintext representation of HTML content."""

  try:
    soup = bs4.BeautifulSoup(html, "html.parser")
  except:  # pylint: disable=bare-except
    # Some docs don't parse
    return ""
  # Remove script and style tags
  for s in soup(["script", "style"]):
    s.decompose()
  return "\n".join([s for s in _soup_strings(soup)])


def _soup_strings(soup):
  """Return text strings in soup."""
  paragraph_tags = set([
      "caption", "details", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "td",
      "div", "span"
  ])

  skip_children = None
  for descendant in soup.descendants:
    # If we've treated a tag as a contiguous paragraph, don't re-emit the
    # children (see below).
    if skip_children is not None:
      try:
        in_skip = descendant in skip_children  # pylint: disable=unsupported-membership-test
      except RecursionError:  # pylint: disable=undefined-variable
        # Possible for this check to hit a nasty infinite recursion because of
        # BeautifulSoup __eq__ checks.
        in_skip = True
      if in_skip:
        continue
      else:
        skip_children = None

    # Treat some tags as contiguous paragraphs, regardless of other tags nested
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
