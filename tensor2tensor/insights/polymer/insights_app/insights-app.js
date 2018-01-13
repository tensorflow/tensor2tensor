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
 * `<insights-app>` Manages the views of the NMT Insights App.
 *
 * ### Usage
 *
 *   <insights-app>
 *   </insights-app>
 */
class InsightsApp extends Polymer.Element {
  /**
   * @return {string} The component name.
   */
  static get is() {
    return 'insights-app';
  }

  /**
   * @return {!Object} The component properties.
   */
  static get properties() {
    return {
      /**
       * @type {string}
       */
      page: {
        type: String,
        reflectToAttribute: true,
      },
    };
  }

  /**
   * @return {!Array<string>} The component observers.
   */
  static get observers() {
    return [
      'routePageChanged_(routeData.page)',
    ];
  }

  /**
   * Updates the page field if page exists or uses a default value.
   * @param {?string} page The current page name being viewed.
   * @private
   */
  routePageChanged_(page) {
    if (page == this.page) {
      return;
    }
    this.page = page || 'explore';
    this.set('routeData.page', this.page);

    // Refresh the now selected page in case it needs new data on a new view.
    let currentPage = this.get('currentPage');
    if (currentPage) {
      currentPage.refresh();
    }
  }
}

customElements.define(InsightsApp.is, InsightsApp);
