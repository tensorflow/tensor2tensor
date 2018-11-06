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

"""A GUnicorn + Flask Debug Frontend for Transformer models."""

import json

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

flags.DEFINE_string("configuration", "",
                    "A JSON InsightConfiguration message that configures which "
                    "models to run in the insight frontend.")
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
  with open(FLAGS.configuration) as configuration_file:
    configuration = json.load(configuration_file)

  # Read in the set of query processors.
  processors = {}
  for processor_configuration in configuration["configuration"]:
    key = (processor_configuration["source_language"],
           processor_configuration["target_language"],
           processor_configuration["label"])

    processors[key] = transformer_model.TransformerModel(
        processor_configuration)

  # Read in the list of supported languages.
  languages = {}
  for language in configuration["language"]:
    languages[language["code"]] = {
        "code": language["code"],
        "name": language["name"],
    }

  # Create flask to serve all paths starting with '/polymer' from the static
  # path.  This is to served non-vulcanized components.
  app = Flask(
      __name__.split(".")[0],
      static_url_path="/polymer",
      static_folder=FLAGS.static_path)

  # Disable static file caching.
  app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

  @app.route("/api/language_list/")
  def language_list():  # pylint: disable=unused-variable
    """Responds to /api/language_list with the supported languages.

    Returns:
      JSON for the languages.
    """
    return jsonify({
        "language": languages.values()
    })

  @app.route("/api/list_models/")
  def list_models():  # pylint: disable=unused-variable
    """Responds to /api/list_models with the supported modes.


    Returns:
      JSON for the supported models.
    """
    configuration_list = []
    for source_code, target_code, label in processors:
      configuration_list.append({
          "id": label,
          "source_language": languages[source_code],
          "target_language": languages[target_code],
      })
    return jsonify({
        "configuration": configuration_list
    })

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
    if (path == "index.js" or
        path == "webcomponentsjs/custom-elements-es5-adapter.js" or
        path == "webcomponentsjs/webcomponents-lite.js"):
      # Some vulcanizing methods bundle the javascript into a index.js file
      # paired with index.html but leave two important webcomponents js files
      # outside of the bundle.  If requesting those special files, fetch them
      # directly rather than from a /static sub-directory.
      return send_from_directory(FLAGS.static_path, path)
    # Everything else should redirect to the main landing page.  Since we
    # use a single page app, any initial url requests may include random
    # paths (that don't start with /api or /static) which all should be
    # served by the main landing page.
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
