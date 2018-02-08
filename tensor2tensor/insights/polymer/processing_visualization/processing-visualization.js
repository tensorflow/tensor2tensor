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
 * `<processing-visualization>` summarises pre/post processing steps.
 *
 * This element presents the pre-processing segmentation steps and
 * post-processing de-segmentation and rewrite steps that are applied to a
 * translation query.
 *
 * ### Usage
 *
 *   <processing-visualization data="[[data]]"></processing-visualization>
 */
class ProcessingVisualization extends Polymer.Element {
  /**
   * @return {string} The component name.
   */
  static get is() {
    return 'processing-visualization';
  }

  /**
   * @return {!Object} The component properties.
   */
  static get properties() {
    return {
      /**
       * @type {!QueryProcessingVisualization}
       */
      data: {
        type: Object,
      },
    };
  }
}

customElements.define(ProcessingVisualization.is, ProcessingVisualization);
