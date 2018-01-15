/**
 * @license
 * Copyright 2017 The Tensor2Tensor Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * `<translation-result>` Presents zero or more visualization of a translation.
 *
 * This inspects the set of visualization fields provided and triggers the
 * corresponding visualization component in the set of available views in tabbed
 * layout.
 *
 * ### Usage
 *
 *   <translation-result result="[[result]]">
 *   </translation-result>
 */
class TranslationResult extends Polymer.Element {
  /**
   * @return {string} The component name.
   */
  static get is() {
    return 'translation-result';
  }

  /**
   * @return {!Object} The component properties.
   */
  static get properties() {
    return {
      /**
       * @type {{
       *   response: {
       *     visualization_name: string,
       *     title: string,
       *     name: string,
       *     query_processing: ?Object,
       *     search_graph: ?Object,
       *     word_heat_map: ?Object,
       *   },
       *   model: !Model,
       *   query: string
       * }}
       */
      result: {
        type: Object,
        observer: 'resultUpdated_',
      },
      /**
       * @type {string}
       */
      view: {
        type: String,
        value: 'processing',
      },
    };
  }

  /**
   * Sets internal data structures given the updated result.
   * @private
   */
  resultUpdated_() {
    var response = this.result.response;
    if (!response || !response.result || response.result.length == 0) {
      return;
    }

    for (var i = 0; i < response.result.length; ++i) {
      let visualizationResult = response.result[i];

      // Dynamically create the visualization element based on the name field.
      // This will enable multiple versions of the same visualization to be
      // created later on when the data mapping is generalized.
      let analysisEle = document.createElement(
          visualizationResult.visualization_name + '-visualization');

      // Set the generic attributes.
      analysisEle.name = visualizationResult.name;
      analysisEle.model = this.result.model;
      analysisEle.query = this.result.query;

      // Set the visualization specific data attribute.
      // TODO(kstevens): Cleanup by setting visualization_name the same as the
      // protobuffer field names so we don't need this mapping.
      if (visualizationResult.visualization_name == 'processing') {
        analysisEle.data = visualizationResult.query_processing;
      } else if (visualizationResult.visualization_name == 'attention') {
        analysisEle.data = visualizationResult.word_heat_map;
      } else if (visualizationResult.visualization_name == 'graph') {
        analysisEle.data = visualizationResult.search_graph;
      }

      Polymer.dom(this.$.view).appendChild(analysisEle);
    }
    // Don't make assumptions about which visualizations are available.  Instead
    // preselect the initial view based on data.
    this.set('view', response.result[0].name);
  }
}

customElements.define(TranslationResult.is, TranslationResult);
