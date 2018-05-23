# coding=utf-8
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

"""Tests of data download and subimage generation utilities."""

import unittest
import logging
import tempfile
import shutil

from tensor2tensor.data_generators import allen_downloader


class TemporaryDirectory(object):
    """For py2 support of `with tempfile.TemporaryDirectory() as name:`"""
    def __enter__(self):
        self.name = tempfile.mkdtemp()
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.name)

        
class TestDownload(unittest.TestCase):
    
    def test_e2e_tiny(self):
        """Test that we can run the downloader and subimage steps e2e."""
        
        with TemporaryDirectory() as tmp_dir:

            allen_downloader.maybe_download_image_datasets(
                tmp_dir,
                section_offset=0,
                num_sections=1,
                images_per_section=1)

            allen_downloader.subimages_for_image_files(tmp_dir)
        

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
