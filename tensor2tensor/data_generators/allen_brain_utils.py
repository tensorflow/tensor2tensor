# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils. for Allen Brain Atlas dataset, download and subimages."""

import math
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf


# Gracefully inform users of the need to install problem-specific deps.
try:
  from allensdk.api.queries.rma_api import RmaApi
  from allensdk.api.queries.image_download_api import ImageDownloadApi
  from PIL import Image
  import pandas as pd
except ImportError as e:
  tf.logging.error("This problem requires the installation of additional "
                   "dependencies. To proceed, please run "
                   "`pip install tensor2tensor[allen]`.")
  # Don't throw the error, it just buries the informative log message.
  exit(1)


def maybe_get_section_list(data_root,
                           product_abbreviation="Mouse",
                           num_rows="all"):
  """Retrieve a list of section datasets.

  Notes:

    Uses file structure:
      data_root/
        meta/
          section_list.csv

    Equivalent to:
      http://api.brain-map.org/api/v2/data/query.xml?\
      criteria=model::SectionDataSet,\
      rma::criteria,[failed$eq%27false%27],\
      products[abbreviation$eq%27Mouse%27],\
      rma::options[num_rows$eq{num_rows}]

    See also: http://help.brain-map.org/display/api/Downloading+an+Image

  Args:
    data_root (str): Root path of where data and meta. are written.
    product_abbreviation (str): Allen api product abbreviation (e.g.
      Mouse).
    num_rows (int): The number of rows to return from Allen API
      SectionDataSet query.

  """

  rma = RmaApi()

  meta_root = os.path.join(data_root, "meta")

  tf.gfile.MakeDirs(meta_root)
  section_list_path = os.path.join(meta_root,
                                   "section_list_%s.csv" % num_rows)

  if tf.gfile.Exists(section_list_path):
    tf.logging.info("Section list found, skipping download.")
    data = pd.read_csv(section_list_path)
    return data

  criteria = ','.join([
      "[failed$in\'false\']",
      "products[abbreviation$eq\'%s\']" % product_abbreviation
  ])

  tf.logging.info("Getting section list with criteria, num rows: "
                  "%s, %s" % (criteria, num_rows))

  data = pd.DataFrame(
      rma.model_query('SectionDataSet',
                      criteria=criteria,
                      num_rows=num_rows))

  data.to_csv(section_list_path)

  return data


def maybe_get_image_list(section_dataset_id,
                         data_root,
                         num_rows="all",
                         image_api_client=None):
  """Obtain a list of images given a section dataset ID.

  Notes:
    Uses file structure:
      data_root/
        meta/
          image_list_for_section{section_dataset_id}.csv

    Equivalent to:
      http://api.brain-map.org/api/v2/data/query.json?\
      q=model::SectionImage,\
      rma::criteria,[data_set_id$eq{section_dataset_id}],\
      rma::options[num_rows$eq{num_rows}][count$eqfalse]

    See also: http://help.brain-map.org/display/api/Downloading+an+Image

  Args:
    section_dataset_id (int): The integer ID of an Allen Institute API
      SectionDataSet.
    data_root (str): Root path of where data and meta. are written.
    num_rows (int): The number of images to obtain from the
      SectionDataSet.
    image_api_client (object): An Allen Institute API client object.

  """

  meta_root = os.path.join(data_root, "meta")
  tf.gfile.MakeDirs(meta_root)
  section_list_path = os.path.join(
      meta_root, ("image_list_for_section"
                  "%s_%s.csv" % (section_dataset_id, num_rows)))

  if tf.gfile.Exists(section_list_path):
    tf.logging.info("Image list found for section id "
                    "%s, skipping download." % section_dataset_id)
    data = pd.read_csv(section_list_path)
    return data

  tf.logging.info("Getting image list for section: "
                  "%s" % section_dataset_id)

  if image_api_client is None:
    image_api_client = ImageDownloadApi()

  data = pd.DataFrame(
      image_api_client.section_image_query(
          section_dataset_id, num_rows=num_rows))

  data.to_csv(section_list_path)

  tf.logging.info("Finished getting image list for section: "
                  "%s" % section_dataset_id)

  return data


def maybe_download_image_dataset(image_list,
                                 image_dataset_id,
                                 data_root,
                                 image_api_client=None):
  """Given a list of image IDs, download the corresponding images.

  Notes:

    Uses file structure:
      data_root/
        raw/
          section_id/
            image_id/
              image_id_raw.jpg

    See also: http://help.brain-map.org/display/api/Downloading+an+Image

  Args:
    image_list (list): A list of section image ID's.
    image_dataset_id (int): The ID for the SectionDataSet for which images
      are being downloaded.
    data_root (str): Root path of where data and meta. are written.
    image_api_client (object): An Allen Institute API client object.

  """

  dataset_root = os.path.join(data_root, "raw", str(image_dataset_id))
  tf.gfile.MakeDirs(dataset_root)

  if image_api_client is None:
    image_api_client = ImageDownloadApi()

  tf.logging.info("Starting download of %s images..." % len(image_list))
  num_images = len(image_list)
  output_paths = []

  for i, image_id in enumerate(image_list[:]["id"]):
    url = ("http://api.brain-map.org/api/v2/section_image_download/"
           "%s" % image_id)
    filename = "raw_%s.jpg" % image_id
    output_base = os.path.join(dataset_root, str(image_id))
    tf.gfile.MakeDirs(output_base)
    output_path = os.path.join(output_base, filename)
    output_paths.append(output_path)
    if tf.gfile.Exists(output_path):
      tf.logging.info("Skipping download, image "
                      "%s of %s at path %s already exists." % (
                          i, num_images, output_path))
      continue

    tf.logging.info("Downloading image "
                    "%s of %s to path %s" % (
                        i, num_images, output_path))

    tmp_file = tempfile.NamedTemporaryFile(delete=False)

    image_api_client.retrieve_file_over_http(url, tmp_file.name)
    shutil.move(tmp_file.name, output_path)

  tf.logging.info("Finished downloading images.")

  return output_paths


def maybe_download_image_datasets(data_root,
                                  section_offset=0,
                                  num_sections="all",
                                  images_per_section="all"):
  """Maybe download all images from specified subset of studies.

  Args:
    data_root (str): Root path of where data and meta. are written.
    section_offset (int): The number of sections to skip at the beginning
      of the section list.
    num_sections (int): The number of sections to obtain from the section
      list.
    images_per_section (int): The number of images to obtain for each
      section.

  """

  section_list = maybe_get_section_list(data_root, num_rows=num_sections)
  total_num_sections = len(section_list)
  if num_sections == "all":
    num_sections = total_num_sections
  tf.logging.info("Obtained section list with "
                  "%s num_sections" % total_num_sections)

  if section_offset > total_num_sections:
    raise ValueError("Can't apply offset %s " % section_offset,
                     "for section list of length %s" % (
                         total_num_sections))
  end_index = section_offset + num_sections
  if end_index > total_num_sections:
    raise ValueError("Saw end_index, num_sections that index past"
                     "num_sections, respectively: "
                     "%s, %s, %s" % (
                         end_index, num_sections, total_num_sections))

  section_list_subset = section_list["id"][section_offset:end_index]

  image_api_client = ImageDownloadApi()

  for image_dataset_id in section_list_subset:

    image_list = maybe_get_image_list(image_dataset_id,
                                      data_root,
                                      images_per_section,
                                      image_api_client)

    image_data_paths = maybe_download_image_dataset(image_list,
                                                    image_dataset_id,
                                                    data_root,
                                                    image_api_client)

    return image_data_paths


def subimage_files_for_image_file(raw_image_path,
                                  metadata_base_path,
                                  xy_size=1024,
                                  max_output=None,
                                  subimage_format="jpeg",
                                  strip_input_prefix=True):
  """Tile an image file to produce sub images of specified size and format.

  Args:
    raw_image_path (str): The path to an input image from which subimages
      will be produced.
    metadata_base_path (str): The directory to which subimage path lists
      should be written.
    xy_size (int): The x and y dimension of sub-image to produce, i.e.
      xy_size by xy_size images.
    max_output (int): The maximum number of sub-images to produce.
    subimage_format (str): The format of subimage to produce (e.g. jpeg,
      png).
    strip_input_prefix (bool): Whether to strip input prefix in
      constructing output fnames (e.g. raw_some_tag ->
      1024x1024_some_tag vs. raw_some_tag -> 1024x1024_raw_some_tag)

  """

  image_dir, image_filename = os.path.split(raw_image_path)

  prefix = "%sx%s" % (xy_size, xy_size)

  if strip_input_prefix:
    image_filename = "_".join(image_filename.split("_")[1:])

  path_manifest_filename = ".".join([image_filename.split(".")[0], "csv"])
  path_manifest_filename = "_".join([prefix,
                                     "path_manifest",
                                     path_manifest_filename])
  path_manifest_path = os.path.join(metadata_base_path,
                                    path_manifest_filename)

  if tf.gfile.Exists(path_manifest_path):
    tf.logging.info("Skipping generation of subimages which already "
                    "exist with path manifest: %s" % path_manifest_path)
    return

  tf.logging.info("Generating subimages for image at path: "
                  "%s" % raw_image_path)
  img = Image.open(raw_image_path)
  img = np.float32(img)
  shape = np.shape(img)

  count = 0

  with tf.gfile.Open(path_manifest_path, "w") as path_manifest_file:

    for h_index in range(0, int(math.floor(shape[0]/xy_size))):
      h_offset = h_index * xy_size
      h_end = h_offset + xy_size - 1
      for v_index in range(0, int(math.floor(shape[1]/xy_size))):
        v_offset = v_index * xy_size
        v_end = v_offset + xy_size - 1

        # Extract a sub-image tile and convert to float in range
        # [0, 1-ish].
        # pylint: disable=invalid-sequence-index
        std_sub = img[h_offset:h_end, v_offset:v_end]/255.0

        # Clip the ish, convert from [0,1] to [0, 255], then to
        # uint8 type.
        subimage = np.uint8(np.clip(std_sub, 0, 1)*255)

        subimage_filename = "%s_%s_%s" % (
            prefix, count, image_filename)
        subimage_path = os.path.join(image_dir, subimage_filename)

        with tf.gfile.Open(subimage_path, "w") as f:
          Image.fromarray(subimage).save(f, subimage_format)

        # Write the name of the generated subimage to the path
        # manifest
        path_manifest_file.write(subimage_path + "\n")

        count += 1
        if max_output is not None and count >= max_output:
          tf.logging.info("Reached maximum number of subimages: "
                          "%s" % max_output)
          return


def mock_raw_image(x_dim=1024, y_dim=1024, num_channels=3,
                   output_path=None, write_image=True):
  """Generate random `x_dim` by `y_dim`, optionally to `output_path`.

  Args:
    output_path (str): Path to which to write image.
    x_dim (int): The x dimension of generated raw image.
    y_dim (int): The x dimension of generated raw image.
    return_raw_image (bool): Whether to return the generated image (as a
      numpy array).

  Returns:
    numpy.array: The random `x_dim` by `y_dim` image (i.e. array).

  """

  rand_shape = (x_dim, y_dim, num_channels)
  tf.logging.debug(rand_shape)

  if num_channels != 3:
    raise NotImplementedError("mock_raw_image for channels != 3 not yet "
                              "implemented.")

  img = np.random.random(rand_shape)
  img = np.uint8(img*255)

  if write_image:
    if not isinstance(output_path, str):
      raise ValueError("Output path must be of type str if write_image=True, "
                       "saw %s." % output_path)

    pil_img = Image.fromarray(img, mode="RGB")
    with tf.gfile.Open(output_path, "w") as f:
      pil_img.save(f, "jpeg")

  return img


def mock_raw_data(tmp_dir, raw_dim=1024, num_channels=3, num_images=1):
  """Mock a raw data download directory with meta and raw subdirs.

  Notes:

    * This utility is shared by tests in both allen_brain_utils and
      allen_brain so kept here instead of in one of *_test.

  E.g.
    {data_root}/
      meta/
      raw/
        dataset_id/
          image_id/
            raw_{image_id}.jpg [random image]

  Args:
    tmp_dir (str): Temporary dir in which to mock data.
    raw_dim (int): The x and y dimension of generated raw imgs.

  Returns:
    tmp_dir (str): Path to root of generated data dir.

  """

  meta = os.path.join(tmp_dir, "meta")
  raw = os.path.join(tmp_dir, "raw")
  os.mkdir(meta)
  os.mkdir(raw)
  mock_dataset_id = "70474875"
  mock_image_id = "70450649"

  # Write dummy section list
  header = (",blue_channel,delegate,expression,failed,failed_facet,"
            "green_channel,id,name,plane_of_section_id,qc_date,red_channel,"
            "reference_space_id,rnaseq_design_id,section_thickness,"
            "specimen_id,sphinx_id,storage_directory,weight\n")
  record = ("0,,False,True,False,734881840,,70474875,,2,"
            "2009-05-02T22:52:23Z,,10,,25.0,70430933,6981,/external/aibssan/"
            "production32/prod334/image_series_70474875/,5270\n")

  with tf.gfile.Open(os.path.join(meta, "section_list_all.csv"), "w") as f:
    f.write(header)
    f.write(record)

  dataset = os.path.join(raw, mock_dataset_id)
  os.mkdir(dataset)

  # Write dummy image list
  header = (",annotated,axes,bits_per_component,data_set_id,"
            "expression_path,failed,height,id,image_height,"
            "image_type,image_width,isi_experiment_id,lims1_id,"
            "number_of_components,ophys_experiment_id,path,"
            "projection_function,resolution,section_number,"
            "specimen_id,structure_id,tier_count,width,x,y\n")

  image_list_fname = "image_list_for_section70474875_all.csv"
  with tf.gfile.Open(os.path.join(meta, image_list_fname), "w") as f:
    f.write(header)
    for image_id in range(1, num_images):
      image_id = str(image_id)
      f.write(",,,,%s,,,,%s,,,,,%s,,,,,,,,,,,,\n" % (mock_dataset_id,
                                                     image_id,
                                                     image_id))

      image_dir = os.path.join(dataset, image_id)
      os.mkdir(image_dir)
      raw_image_path = os.path.join(image_dir, "raw_%s.jpg" % image_id)

      mock_raw_image(x_dim=raw_dim, y_dim=raw_dim, num_channels=num_channels,
                     output_path=raw_image_path)


class TemporaryDirectory(object):
  """For py2 support of `with tempfile.TemporaryDirectory() as name:`"""

  def __enter__(self):
    self.name = tempfile.mkdtemp()
    return self.name

  def __exit__(self, exc_type, exc_value, traceback):
    shutil.rmtree(self.name)
