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

import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf


def try_importing_allensdk():
  """Import necessary allensdk objects if the function is called."""
  try:
    from allensdk.api.queries.rma_api import RmaApi
    from allensdk.api.queries.image_download_api import ImageDownloadApi
  except ImportError:
    tf.logging.error("Can't import allensdk. Please install it, "
                     "such as by running `pip install allensdk`.")
    exit(1)

  return RmaApi, ImageDownloadApi


def try_importing_pandas():
  """Import pandas if the function is called."""
  try:
    import pandas
  except ImportError:
    tf.logging.error("Can't import pandas. Please install it, "
                     "such as by running `pip install pandas`.")
    exit(1)

  return pandas


def try_importing_pil_image():
  """Import a PIL Image object if the function is called."""
  try:
    from PIL import Image
  except ImportError:
    tf.logging.error("Can't import Image from PIL (Pillow). Please install it, "
                     "such as by running `pip install Pillow`.")
    exit(1)

  return Image


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
    data_root: str, root path of where data and meta. are written.
    product_abbreviation: str, Allen api product abbreviation (e.g.
      Mouse).
    num_rows: int, The number of rows to return from Allen API
      SectionDataSet query.

  """

  rma, _ = try_importing_allensdk()

  pd = try_importing_pandas()

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
                         client=None):
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
    section_dataset_id: int, the integer ID of an Allen Institute API
      SectionDataSet.
    data_root: str, root path of where data and meta. are written.
    num_rows: int, the number of images to obtain from the
      SectionDataSet.
    client: object, an Allen Institute API client object.

  """

  meta_root = os.path.join(data_root, "meta")
  tf.gfile.MakeDirs(meta_root)
  section_list_path = os.path.join(
      meta_root, ("image_list_for_section"
                  "%s_%s.csv" % (section_dataset_id, num_rows)))

  if tf.gfile.Exists(section_list_path):
    tf.logging.info("Image list found for section id "
                    "%s, skipping download." % section_dataset_id)
    pd = try_importing_pandas()
    data = pd.read_csv(section_list_path)
    return data

  tf.logging.info("Getting image list for section: "
                  "%s" % section_dataset_id)

  if client is None:
    allensdk = try_importing_allensdk()
    client = allensdk.api.queries.image_download_api.ImageDownloadApi()

  data = pd.DataFrame(
      client.section_image_query(
          section_dataset_id, num_rows=num_rows))

  data.to_csv(section_list_path)

  tf.logging.info("Finished getting image list for section: "
                  "%s" % section_dataset_id)

  return data


def maybe_download_image_dataset(image_list,
                                 image_dataset_id,
                                 data_root,
                                 client=None):
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
    image_list: list, a list of section image ID's.
    image_dataset_id: int, the ID for the SectionDataSet for which images
      are being downloaded.
    data_root: str, root path of where data and meta. are written.
    client: object, an Allen Institute API client object.

  """

  dataset_root = os.path.join(data_root, "raw", str(image_dataset_id))
  tf.gfile.MakeDirs(dataset_root)

  if client is None:
    _, client = try_importing_allensdk()

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

    client.retrieve_file_over_http(url, tmp_file.name)
    tf.gfile.Rename(tmp_file.name, output_path)

  tf.logging.info("Finished downloading images.")

  return output_paths


def maybe_download_image_datasets(data_root,
                                  section_offset=0,
                                  num_sections="all",
                                  images_per_section="all"):
  """Maybe download all images from specified subset of studies.

  Args:
    data_root: str, root path of where data and meta. are written.
    section_offset: int, the number of sections to skip at the beginning
      of the section list.
    num_sections: int, the number of sections to obtain from the section
      list.
    images_per_section: int, the number of images to obtain for each
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

  _, client = try_importing_allensdk()

  for image_dataset_id in section_list_subset:

    image_list = maybe_get_image_list(image_dataset_id,
                                      data_root,
                                      images_per_section,
                                      client)

    image_data_paths = maybe_download_image_dataset(image_list,
                                                    image_dataset_id,
                                                    data_root,
                                                    client)

    return image_data_paths


def mock_raw_image(x_dim=1024, y_dim=1024, num_channels=3,
                   output_path=None, write_image=True):
  """Generate random `x_dim` by `y_dim`, optionally to `output_path`.

  Args:
    output_path: str, path to which to write image.
    x_dim: int, the x dimension of generated raw image.
    y_dim: int, the x dimension of generated raw image.
    return_raw_image: bool, whether to return the generated image (as a
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

    image_obj = try_importing_pil_image()
    pil_img = image_obj.fromarray(img, mode="RGB")
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
    tmp_dir: str, temporary dir in which to mock data.
    raw_dim int, the x and y dimension of generated raw imgs.

  Returns:
    tmp_dir: str, path to root of generated data dir.

  """

  meta = os.path.join(tmp_dir, "meta")
  raw = os.path.join(tmp_dir, "raw")
  tf.gfile.MakeDirs(meta)
  tf.gfile.MakeDirs(raw)
  mock_dataset_id = "70474875"

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
  tf.gfile.MakeDirs(dataset)

  header = (",annotated,axes,bits_per_component,data_set_id,"
            "expression_path,failed,height,id,image_height,"
            "image_type,image_width,isi_experiment_id,lims1_id,"
            "number_of_components,ophys_experiment_id,path,"
            "projection_function,resolution,section_number,"
            "specimen_id,structure_id,tier_count,width,x,y").split(",")

  pd = try_importing_pandas()
  image_list = pd.DataFrame(np.zeros(shape=(num_images, len(header))),
                            columns=header)

  image_list_fname = os.path.join(meta, "image_list*_all.csv")

  for image_id in range(1, num_images):
    image_list["data_set_id"][image_id] = mock_dataset_id
    image_list["id"][image_id] = image_id

    image_dir = os.path.join(dataset, image_id)
    tf.gfile.MakeDirs(image_dir)
    raw_image_path = os.path.join(image_dir, "raw_%s.jpg" % image_id)

    mock_raw_image(x_dim=raw_dim, y_dim=raw_dim,
                   num_channels=num_channels,
                   output_path=raw_image_path)

  image_list.write_csv(image_list_fname)


class TemporaryDirectory(object):
  """For py2 support of `with tempfile.TemporaryDirectory() as name:`"""

  def __enter__(self):
    self.name = tempfile.mkdtemp()
    return self.name

  def __exit__(self, exc_type, exc_value, traceback):
    shutil.rmtree(self.name)
