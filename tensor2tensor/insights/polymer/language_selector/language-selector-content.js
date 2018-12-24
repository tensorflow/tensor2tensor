/**
 * @license
 * Copyright 2018 The Tensor2Tensor Authors.
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
 * `<language-selector-content>` provides menu content for language selection.
 *
 * The content provides a search bar that will filter available languages by any
 * language name or code that has the query text as a substring.
 *
 * By default, this will auto select a provided language with language code
 * 'en'.
 *
 * ### Usage
 *
 *   <language-selector-content
 *       languages="[[languages]]"
 *       value="{{language}}">
 *   </language-selector-content>
 */
class LanguageSelectorContent extends Polymer.Element {
  /**
   * @return {string} The component name.
   */
  static get is() {
    return 'language-selector-content';
  }

  /**
   * @return {!Object} The component properties.
   */
  static get properties() {
    return {
      /**
       * @type {?Array<!Language>}
       */
      languages: {
        type: Array,
        observer: 'languagesUpdated_',
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
      }
    };
  }

  /**
   * @return {!Array<string>} The component observers.
   */
  static get observers() {
    return [
      'selectDefault_(languages, renderedItemCount)',
      'filterUpdated_(filter)',
    ];
  }

  /**
   * Selects the language in the drop down.
   * @param {Language} language The language to pre-select.
   * @public
   */
  forceSelection(language) {
    this.set('filter', '');
    for (var i = 0; i < this.languages.length; ++i) {
      if (this.languages[i].code == language.code) {
        this.set('value', this.languages[i]);
        this.updateSelected_(Polymer.dom(this.$.items).children[i]);
        return;
      }
    }
  }

  /**
   * Updates the internal languages and resets selection.
   * @param {?Array<!Language>} newLanguages The new language list.
   * @private
   */
  languagesUpdated_(newLanguages) {
    if (newLanguages) {
      for (var i = 0; i < newLanguages.length; ++i) {
        newLanguages[i].hidden = false;
      }
    }

    this.set('filter', '');
    this.set('selected', undefined);
  }

  /**
   * Selects the default language if one can be found after all languages have
   * been rendered in the menu.
   * @param {?Array<Language>} languages The languages
   * @param {number} renderedItemCount The number of languages rendered.
   * @private
   */
  selectDefault_(languages, renderedItemCount) {
    if (this.get('selected') || !languages ||
        languages.length != renderedItemCount) {
      return;
    }

    this.$.languageList.render();
    if (this.value) {
      for (var i = 0; i < languages.length; ++i) {
        if (languages[i].code == this.value.code) {
          this.updateSelected_(Polymer.dom(this.$.items).children[i]);
          return;
        }
      }
    }

    let defaultCode = this.get('defaultCode');
    for (var i = 0; i < languages.length; ++i) {
      if (languages[i].code == defaultCode || languages.length == 1) {
        this.set('value', languages[i]);
        this.updateSelected_(Polymer.dom(this.$.items).children[i]);
        return;
      }
    }
  }

  /**
   * Selects the rendered language if only one is visible given the current
   * search filter.
   * @private
   */
  enterPressed_() {
    let visibleLanguagesIndices = [];
    for (var i = 0; i < this.languages.length; ++i) {
      if (!this.languages[i].hidden) {
        visibleLanguagesIndices.push(i);
      }
    }
    if (visibleLanguagesIndices.length == 1) {
      this.set('value', this.languages[visibleLanguagesIndices[0]]);
      this.updateSelected_(Polymer.dom(this.$.items).children[0]);
    }
  }

  /**
   * Sets the hidden state of languages given the current filter.
   * @param {string} newFilter The new filter to match languages against.
   * @private
   */
  filterUpdated_(newFilter) {
    if (!this.get('languages')) {
      return;
    }

    let filter = newFilter.toLowerCase();
    for (var i = 0; i < this.languages.length; ++i) {
      let hidden = !this.languageMatchesQuery_(this.languages[i], filter);
      this.set('languages.' + i + '.hidden', hidden);
    }
  }

  /**
   * Returns true if the language is visible.
   * @param {!Language} language The language being evaluated.
   * @return {boolean} True if visible.
   * @private
   */
  isShown_(language) {
    return !language.hidden;
  }

  /**
   * Returns true if the language matches the filter.
   * @param {!Language} language The language being evaluated.
   * @param {string} filter The filter to compare against.
   * @return {boolean} True if language matches filter.
   * @private
   */
  languageMatchesQuery_(language, filter) {
    let languageName = language.name.toLowerCase();
    return filter == '' || languageName.indexOf(filter) >= 0 ||
        language.code.indexOf(filter) >= 0;
  }

  /**
   * Selects the tapped element and updates the value with the corresponding
   * language value.
   * @param {!EventTarget} e The tap event.
   * @private
   */
  select_(e) {
    let language = this.$.languageList.itemForElement(e.target);
    this.set('value', language);
    this.updateSelected_(e.target);
  }

  /**
   * Updates the selection with the given element.
   * @param {!Element} ele The selected dom element.
   * @private
   */
  updateSelected_(ele) {
    let oldSelection = this.get('selected');
    if (oldSelection) {
      this.dispatchEvent(new CustomEvent('iron-deselect', {
        bubbles: true,
        composed: true,
        detail: {
          item: oldSelection,
        },
      }));
    }
    this.set('selected', ele);
    this.dispatchEvent(new CustomEvent('iron-select', {
      bubbles: true,
      composed: true,
      detail: {
        item: ele,
      },
    }));
  }
}

customElements.define(LanguageSelectorContent.is, LanguageSelectorContent);
