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
 * `<language-selector>` provides a searchable dropdown of languages.
 *
 * The dropdown will present the selected language's Name.  When opened, the
 * search bar will filter available languages by any language name or code that
 * has the query text as a substring.
 *
 * By default, this will auto select a provided language with language code
 * 'en'.
 *
 * ### Usage
 *
 *   <language-selector languages="[[languages]]" value="{{language}}">
 *   </language-selector>
 */
class LanguageSelector extends Polymer.Element {
  /**
   * @return {string} The component name.
   */
  static get is() {
    return 'language-selector';
  }

  /**
   * @return {!Object} The component properties.
   */
  static get properties() {
    return {
      /**
       * @type {string}
       */
      label: {
        type: String,
      },
      /**
       * @type {?Array<Language>}
       */
      languages: {
        type: Array,
      },
      /**
       * @type {!Language}
       */
      value: {
        type: Object,
        notify: true,
      },
      /**
       * @type {string}
       */
      defaultCode: {
        type: String,
        value: 'en',
      },
    };
  }

  /**
   * Selects the language in the drop down.
   * @param {Language} language The language to pre-select.
   * @public
   */
  forceSelection(language) {
    this.$.selector.forceSelection(language);
  }
}

customElements.define(LanguageSelector.is, LanguageSelector);
