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
 * `<query-card>` presents a material card for selecting a supported mdoel.
 *
 * This will fetch a set of supported models for debugging and provide three
 * selectors:
 *   - Source Language
 *   - Target Language
 *   - Model
 * Once all three have been populated, it will emit a `Model` object through
 * `model`.
 *
 * ### Usage
 *
 *   <query-card model="{{model}}">
 *     <input type-text>Custom InputField</input>
 *   </query-card>
 */
class QueryCard extends Polymer.Element {
  constructor() {
    super();

    /**
     * A general mapping from language code to the language objects.
     * @type {!Object<string, !Language>}
     * @private
     */
    this.languageToNameMap_ = {};

    /**
     * A nested mapping of languages to a list of models.
     * @type {!Object<string, !Object<string, !Object<string, !Array<!Model>>>>}
     * @private
     */
    this.languagePairToModelMap_ = {};
  }

  /**
   * @return {string} The component name.
   */
  static get is() {
    return 'query-card';
  }

  /**
   * @return {!Object} The component properties.
   */
  static get properties() {
    return {
      /**
       * @type {!Object}
       */
      route: {
        type: String,
      },
      /**
       * @type {!Object}
       */
      subRoute: {
        type: String,
        notify: true,
      },
      /**
       * @type {?Model}
       */
      model: {
        type: Object,
        notify: true,
      },
      /**
       * @type {string}
       */
      url: {
        type: String,
      },
      /**
       * @type {?Language}
       */
      sourceLanguage_: {
        type: Object,
      },
      /**
       * @type {?Language}
       */
      targetLanguage_: {
        type: Object,
      },
      /**
       * @type {string}
       */
      defaultModelId: {
        type: String,
        value: 'prod',
      }
    };
  }

  /**
   * @return {!Array<string>} The component observers.
   */
  static get observers() {
    return [
      'routeActiveUpdated_(routeActive)',

      'modelsUpdated_(modelConfigurations)',
      'sourceLanguagesUpdated_(sourceLanguages, routeData)',
      'targetLanguagesUpdated_(targetLanguages, routeData)',

      'sourceLanguageUpdated_(sourceLanguage_)',
      'targetLanguageUpdated_(targetLanguage_)',
      'modelListUpdated_(modelList, routeData)',
      'modelUpdated_(model)',
    ];
  }

  /**
   * Resets the route data if the route is inactive.
   * @param {boolean} routeActive The active state of the route.
   * @private
   */
  routeActiveUpdated_(routeActive) {
    if (!routeActive) {
      this.set('routeData', {});
    }
  }

  /**
   * Sets the sourceLanguage if a new source language matches the route
   * path or marks it as undefined.
   * @param {Array<Language>} sourceLanguages A list of source languages.
   * @param {{sourceLanguage: string}} routeData The current route paths.
   * @private
   */
  sourceLanguagesUpdated_(sourceLanguages, routeData) {
    if (this.routeActive && sourceLanguages) {
      for (var i = 0; i < sourceLanguages.length; ++i) {
        if (routeData.sourceLanguage == sourceLanguages[i].code) {
          this.$.sourceSelector.forceSelection(sourceLanguages[i]);
          return;
        }
      }
    }
  }

  /**
   * Selects the available target language list based on the new selected source
   * language.
   * @param {Language} sourceLanguage The selected source language index.
   * @private
   */
  sourceLanguageUpdated_(sourceLanguage) {
    if (sourceLanguage == undefined) {
      this.set('targetLanguages', []);
      return;
    }

    this.set('routeData.sourceLanguage', sourceLanguage.code);

    var targetLanguages = [];
    for (var key in this.languagePairToModelMap_[sourceLanguage.code]) {
      targetLanguages.push(this.languageToNameMap_[key]);
    }
    targetLanguages.sort(sort_);
    this.set('targetLanguage', undefined);
    this.set('targetLanguages', targetLanguages);
  }

  /**
   * Sets the targetLanguage if a new target language matches the route
   * path or marks it as undefined.
   * @param {Array<Language>} targetLanguages A list of target languages.
   * @param {{targetLanguage: string}} routeData The current route paths.
   * @private
   */
  targetLanguagesUpdated_(targetLanguages, routeData) {
    if (this.routeActive && targetLanguages) {
      for (var i = 0; i < targetLanguages.length; ++i) {
        if (routeData.targetLanguage == targetLanguages[i].code) {
          this.$.targetSelector.forceSelection(targetLanguages[i]);
          return;
        }
      }
    }
  }

  /**
   * Selects the available model list based on the new selected target
   * language.
   * @param {Language} targetLanguage The selected target language index.
   * @private
   */
  targetLanguageUpdated_(targetLanguage) {
    this.set('model', undefined);
    if (targetLanguage == undefined) {
      this.set('modelList', []);
      return;
    }

    let sourceLanguage = this.sourceLanguage_;
    this.set('routeData.targetLanguage', targetLanguage.code);
    var models = [];
    var targetLanguageMap = this.languagePairToModelMap_[sourceLanguage.code];
    for (var key in targetLanguageMap[targetLanguage.code]) {
      models.push(targetLanguageMap[targetLanguage.code][key]);
    }
    this.set('modelList', models);
  }

  /**
   * Sets the modelIndex  if a new model matches the route path or marks it as
   * undefined.
   * @param {?Array<!Model>} modelList A list of models.
   * @param {{modelId: string}} routeData The current route paths.
   * @private
   */
  modelListUpdated_(modelList, routeData) {
    if (this.routeActive && modelList) {
      for (var i = 0; i < modelList.length; ++i) {
        if (routeData.modelId == modelList[i].id) {
          this.set('model', modelList[i]);
          return;
        }
      }
    }

    if (modelList && modelList.length >= 1) {
      // Chose the default model if it exists, otherwise choose the first entry.
      // This ensures that the ordering of models does't impact the default
      // selection.
      for (var i = 0; i < modelList.length; ++i) {
        if (this.defaultModelId == modelList[i].id) {
          this.set('model', modelList[i]);
          return;
        }
      }
      this.set('model', modelList[0]);
    }
  }

  /**
   * Updates the selected model with the current model index.
   * @param {?Model} model The current selected model index.
   * @private
   */
  modelUpdated_(model) {
    if (!model) {
      return;
    }

    this.set('routeData.modelId', this.model.id);
  }

  /**
   * Updates the set of available language sets and models.
   * @param {{configuration: !Array<!Model>}} modelConfigurations A list of
   *     models.
   * @private
   */
  modelsUpdated_(modelConfigurations) {
    var models = modelConfigurations.configuration;

    this.languageToNameMap_ = {};
    this.languagePairToModelMap_ = {};

    for (var i = 0; i < models.length; ++i) {
      let model = models[i];
      // Extract the language codes and store the code to language mappings.
      var source_language = model.source_language.code;
      this.languageToNameMap_[source_language] = model.source_language;
      var target_language = model.target_language.code;
      this.languageToNameMap_[target_language] = model.target_language;

      // Create the first level nested map, from source languages to target
      // language maps.
      var targetLanguageMap;
      if (source_language in this.languagePairToModelMap_) {
        targetLanguageMap = this.languagePairToModelMap_[source_language];
      } else {
        targetLanguageMap = {};
        this.languagePairToModelMap_[source_language] = targetLanguageMap;
      }

      // Create the second level nested map, from target languages to model
      // maps.
      var model_map;
      if (target_language in targetLanguageMap) {
        model_map = targetLanguageMap[target_language];
      } else {
        model_map = {};
        targetLanguageMap[target_language] = model_map;
      }

      // Store the mapping from a model id to a model.
      model_map[model.id] = model;
    }

    // Prepare the initial set of available source languages.
    var sourceLanguageList = [];
    for (var key in this.languagePairToModelMap_) {
      sourceLanguageList.push(this.languageToNameMap_[key]);
    }
    sourceLanguageList.sort(sort_);
    this.set('sourceLanguages', sourceLanguageList);
  }
}

customElements.define(QueryCard.is, QueryCard);

/**
 * Returns the ordering of two language's based on their name.
 * @param {!Language} a The first language to compare.
 * @param {!Language} b The second language to compare.
 * @return {number} Negative if a comes before b.
 */
function sort_(a, b) {
  if (a.name != b.name) {
    return a.name < b.name ? -1 : 1;
  }
  return 0;
}
