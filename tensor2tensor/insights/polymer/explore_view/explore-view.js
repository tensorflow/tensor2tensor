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
 * `<explore-view>` Presents a view for debuging translations.
 *
 * This provides an interactive interface for querying a backend service to
 * fetch detailed analysis of a translation process.  Each result will be
 * provided as a stack.
 *
 * ### Usage
 *
 *   <explore-view></explore-view>
 */
class ExploreView extends Polymer.Element {
  /**
   * @return {string} The component name.
   */
  static get is() {
    return 'explore-view';
  }

  /**
   * @return {!Object} The component properties.
   */
  static get properties() {
    return {
      route: {
        type: Object,
      },
      /**
       * @type {!Array<!{
       *   source: string,
       *   bad_translations: string,
       *   good_translations: string,
       *   attention_threshold: number
       * }>}
       */
      rules_: {
        type: Array,
      },
      /**
       * @type {?Model}
       */
      model_: {
        type: Object
      },
      /**
       * @type {string}
       */
      query_: {
        type: Object,
      }
    };
  }

  /**
   * @return {!Array<string>} The component observers.
   */
  static get observers() {
    return [
      'modelChanged_(queryData, model_)',
    ];
  }

  /**
   * @override
   */
  ready() {
    super.ready();
    this.set('rules_', []);
    this.set('fetchingResult', false);
  }

  /**
   * Noop
   * @public
   */
  refresh() {
    // Noop
  }

  /**
   * Resets the results when a model changes and triggers a query automatically
   * if one exists.
   * @param {?{query: string}} queryData The current route data.
   * @param {?Model} model Unused, but needed for triggering.
   * @private
   */
  modelChanged_(queryData, model) {
    if (queryData && queryData.query) {
      // Compose the query from the querydata field and the path in the rest of
      // the route.  If the link includes an escaped "/" app-route splits the
      // query and remaining path on that escaped "/".  So query appears to not
      // include the rest of the intended query.
      let query = unescape(queryData.query) + this.get('tailRoute').path;
      this.set('query_', query);
      this.translate_();
    }
    this.set('results', []);
    this.set('rules_', []);
  }

  /**
   * Sends a translation request to the server.
   * @private
   */
  translate_() {
    if (!this.model_ || !this.model_.id) {
      return;
    }

    var params = {
      'source': this.query_,
      'id': this.model_.id,
      'sl': this.model_.source_language.code,
      'tl': this.model_.target_language.code,
    };
    var paramList = this.createBodyValue_(params);
    this.set('url', '/debug?' + paramList);
    this.set('fetchingResult', true);
    this.$.translateAjax.generateRequest();
  }

  /**
   * Returns a string with all the query parameters composed together.  This
   * also serializes the rapid response rules provided.
   * @param {!Object} params The params to combine.
   * @returns {string} The params collapsed together.
   * @private
   */
  createBodyValue_(params) {
    // Add the key value body parts.
    var bodyParts = [];
    for (var param in params) {
      var value = window.encodeURIComponent(params[param]);
      bodyParts.push(param + "=" + value);
    }

    // Add the rapid response rules.
    for (var i = 0; i < this.rules_.length; ++i) {
      var rule = this.rules_[i];
      var value =
        'src_lang: "' + this.model_.source_language.code + '" ' +
        'trg_lang: "' + this.model_.target_language.code + '" ' +
        'source: "' + rule['source'] + '" ' +
        'bad_translations: "' + rule.bad_translations + '" ' +
        'good_translations: "' + rule.good_translations + '" ' +
        'attention_threshold: ' + rule.attention_threshold;
      bodyParts.push('rule=' + window.encodeURIComponent(value));
    }

    // Combine everything together.
    return bodyParts.join('&');
  }

  /**
   * Adds the translation response to the list of results.
   * @param {!Event} event The event object from the `response` event. This is
   *   required to access the current response, as there are timing issues when
   *   accessing the latest response with iron-ajax's `last-response` attribute.
   * @private
   */
  handleTranslationResponse_(event) {
    this.set('fetchingResult', false);
    this.push('results', {
      response: event.detail.response,
      query: this.query_,
      model: this.model_,
    });
  }

  /**
   * Adds a new rapid response rule to be filled out.
   * @private
   */
  addRule_() {
    this.push('rules_', {
      source: '',
      bad_translations: '',
      good_translations: '',
      attention_threshold: 0.9,
    });
  }

  /**
   * Deletes a rapid response rule.
   * @param {Event} e The event in the dom repeat template element.
   * @private
   */
  deleteRule_(e) {
    let model = e.model;
    this.splice('rules_', model.index, 1);
  }
}

customElements.define(ExploreView.is, ExploreView);
