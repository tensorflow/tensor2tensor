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

"""A GUnicorn + Flask Debug Frontend for Transformer models."""

from flask import Flask
from flask import jsonify
from flask import request
from flask import send_from_directory
from gunicorn.app.base import BaseApplication
from gunicorn.six import iteritems
from tensor2tensor.insights import transformer_model

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("t2t_model_dir", "", "")
flags.DEFINE_string("t2t_data_dir", "", "")
flags.DEFINE_string("static_path", "",
                    "Path to static javascript and html files to serve.")


class DebugFrontendApplication(BaseApplication):
  """A local custom application for GUnicorns.

  This custom application enables us to run with a custom main that parses
  tensorflow ops and does some internal setup prior to processing queries.  The
  underlying app registered instances of this class will be forked.
  """

  def __init__(self, app, options=None):
    """Creates the GUnicorn application.

    Args:
      app: A Flask application that will process requests.
      options: A dict of GUnicorn options.
    """
    self.options = options or {}
    self.application = app
    super(DebugFrontendApplication, self).__init__()

  def load_config(self):
    """Loads the configuration."""
    config = dict([(key, value) for key, value in iteritems(self.options)
                   if key in self.cfg.settings and value is not None])
    for key, value in iteritems(config):
      self.cfg.set(key.lower(), value)

  def load(self):
    """Loads the application.

    Returns:
      The Flask application.
    """
    return self.application


def main(_):
  # Create the models we support:
  processors = {}
  transformer_key = ("en", "de", "transformers_wmt32k")
  # TODO(kstevens): Turn this into a text proto configuration that's read in on
  # startup.
  processors[transformer_key] = transformer_model.TransformerModel(
      FLAGS.t2t_data_dir, FLAGS.t2t_model_dir)

  # Create flask to serve all paths starting with '/static' from the static
  # path.
  app = Flask(
      __name__.split(".")[0],
      static_url_path="/static",
      static_folder=FLAGS.static_path)

  # Disable static file caching.
  app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

  @app.route("/api/language_list/")
  def language_list():  # pylint: disable=unused-variable
    """Responds to /api/language_list with the supported languages.

    Returns:
      JSON for the languages.
    """
    # TODO(kstevens): Figure this out automatically by processing the
    # configuration.
    result = {
        "language": [
            {"code": "en", "name": "English"},
            {"code": "de", "name": "German"},
        ],
    }
    return jsonify(result)

  @app.route("/api/list_models/")
  def list_models():  # pylint: disable=unused-variable
    """Responds to /api/list_models with the supported modes.


    Returns:
      JSON for the supported models.
    """
    # TODO(kstevens): Turn this into a configuration text proto that's read in
    # on startup.
    result = {
        "configuration": [
            {
                "id": "transformers_wmt32k",
                "source_language": {
                    "code": "en",
                    "name": "English",
                },
                "target_language": {
                    "code": "de",
                    "name": "German",
                },
            },
        ],
    }
    return jsonify(result)

  @app.route("/debug", methods=["GET"])
  def query():  # pylint: disable=unused-variable
    """Responds to /debug with processing results.

    Returns:
      JSON for the query's result.
    """
    query = request.args.get("source")
    source_language = request.args.get("sl")
    target_language = request.args.get("tl")
    model_name = request.args.get("id")
    processor = processors[(source_language, target_language, model_name)]
    return jsonify(processor.process(query))

  # Catchall for all other paths.  Any other path should get the basic index
  # page, the polymer side will determine what view to show and what REST calls
  # to make for data.
  @app.route("/", defaults={"path": ""})
  @app.route("/<path:path>")
  def root(path):  # pylint: disable=unused-variable
    """Responds to all other non-static paths with index.html.

    Args:
      path: Unused path.

    Returns:
      The landing page html text.
    """
    del path
    return send_from_directory(FLAGS.static_path, "index.html")

  # Run the server.
  tf.logging.info("############# READY ##################")
  options = {
      "bind": ":8010",
      "timeout": 600,
      "workers": 4,
      "reload": True,
      "spew": True,
      "worker_class": "gevent",
  }
  DebugFrontendApplication(app, options).run()


if __name__ == "__main__":
  tf.app.run()
