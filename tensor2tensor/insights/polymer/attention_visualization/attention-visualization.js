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
 * `<attention-visualization>` presents a heatmap of input-output associations.
 *
 * The heat map association shows source to target word association strengths
 * according to some method.
 *
 * ### Usage
 *
 *   <attention-visualization data="[[data]]"></attention-visualization>
 */
class AttentionVisualization extends Polymer.Element {
  constructor() {
    super();

    /**
     * D3.js DOM element.
     * @private
     */
    this.container_ = undefined;
    /**
     * @private
     */
    this.margin_ = {
      top: 150,
      bottom: 50,
      right: 10,
      left: 100
    };
    /**
     * D3.js DOM element.
     * @private
     */
    this.svg_ = undefined;
    /**
     * D3.js DOM element.
     * @private
     */
    this.vis_ = undefined;
    /**
     * D3.js DOM element.
     * @private
     */
    this.zoom_ = undefined;
  }

  /**
   * @return {string} The component name.
   */
  static get is() {
    return 'attention-visualization';
  }

  /**
   * @return {!Object} The component properties.
   */
  static get properties() {
    return {
      /**
       * @type {AttentionData}
       */
      data: {
        type: Object,
        observer: 'dataUpdated_',
      },
      /**
       * @type {number}
       */
      zoomDepth_: {
        type: Number,
      },
    };
  }

  /**
   * @return {!Array<string>} The component observers.
   */
  static get observers() {
    return [
      'zoomDepthChanged_(zoomDepth_)',
    ];
  }

  /**
   * Sets the default zoom depth.
   * @override
   */
  ready() {
    super.ready();
    this.set('zoomDepth_', 20);
  }

  /**
   * Sets the zoom state based on the updated depth.
   * @param {number} zoomDepth the zoom depth.
   * @private
   */
  zoomDepthChanged_(zoomDepth) {
    if (!this.container_) { return; }

    if (zoomDepth == 0) {
      zoomDepth = 0.000001;
    }
    let transform = d3.zoomTransform(this.vis_.node()).scale(zoomDepth / 20.0);
    this.container_.attr("transform", transform);
  }

  /**
   * Updates the heatmap.
   * @param {!AttentionData} newData the new alignment data.
   * @private
   */
  dataUpdated_(newData) {
    // Create the bounding areas and margins for the heatmap.
    let cellDimension = 40;
    let sourceTokens = newData.source_tokens;
    let targetTokens = newData.target_tokens;

    // Convert the attention weights to cell objects which also give access to
    // the row and column indices.
    let mapCells = newData.weights.map(function(d, i) {
      return {
        value: d,
        row: Math.floor(i / targetTokens.length),
        col: i % targetTokens.length
      };
    });

    // Create the color scale.
    let colorScale = d3.scaleQuantile().domain([0.0, 1.0]).range([
      '#cccccc', '#b2b2b2', '#999999', '#7f7f7f',
      '#666666', '#4c4c4c', '#333333', '#191919'
    ]);

    this.zoom_ = d3.zoom().scaleExtent([1, 10]).on('zoom', zoomed.bind(this));

    d3.select(this.$.chart).selectAll("*").remove();

    // Create the bounding div and svgs which will contain all details.
    this.svg_ = d3.select(this.$.chart)
        .append('div')
        .classed('svg-container', true)
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .classed('svg-content-responsive', true);

    this.vis_ = this.svg_.append('g')
        .attr('transform',
              'translate(' + this.margin_.left + ',' + this.margin_.top + ')')
        .call(this.zoom_)
        .on('dblclick.zoom', null)
        .on('wheel.zoom', null);

    // Create a bounding rectangle upon which zooming and panning will take
    // place.
    this.vis_.append('rect')
        .attr('width', '100%')
        .attr('height', '100%')
        .style('fill', 'none')
        .style('pointer-events', 'all');

    this.container_ = this.vis_.append('g');

    // Initiate the panning and/or zooming.
    function zoomed() {
      this.container_.attr("transform",
          d3.event.transform.scale(this.zoomDepth_ / 20.0));
    }

    // Place the source tokens along the vertical axis.  Each token has an id
    // based on it's index.
    var sourceLabels = this.container_.append('g');

    sourceLabels.selectAll('.source-label')
        .data(sourceTokens)
        .enter()
        .append('text')
        .text(function(d) {
          return d;
        })
        .style('text-anchor', 'end')
        .attr(
            'id',
            function(d, i) {
              return 'row-' + i;
            })
        .attr('class', 'source-label mono')
        .attr('transform', 'translate(-6,' + cellDimension / 1.5 + ')')
        .attr('x', 0)
        .attr('y', function(d, i) {
          return i * cellDimension;
        });

    var targetLabels = this.container_.append('g');

    // Place the target tokens along the horizontal axis.  Each token has an id
    // based on it's index.
    targetLabels.selectAll('.target-label')
        .data(targetTokens)
        .enter()
        .append('text')
        .text(function(d) {
          return d;
        })
        .style('text-anchor', 'left')
        .attr(
            'id',
            function(d, i) {
              return 'col-' + i;
            })
        .attr('class', 'target-label mono')
        .attr(
            'transform', 'translate(' + cellDimension / 2 + ',-6) rotate(-90)')
        .attr(
            'y',
            function(d, i) {
              return i * cellDimension;
            })
        .attr('x', 0);

    // Create the heat map and populate with cells.  Each cell will
    // highlight when hovered over.  Additionally, the column and row tokens
    // will highlight to make clear which tokens are being observed.  Lastly,
    // each cell will trigger a popup showing details of the alignment state.
    var heatMap = this.container_.append('g');

    // Group the rectangle and text elements and capture the mouse events from
    // both so that the rectangle can be highlighted when it's in focus.
    let cellGroup = heatMap.selectAll('.cell')
        .data(mapCells)
        .enter()
        .append('g')
        .attr('class', 'cell-group')
        .on('mouseover', function(d, i) {
          // Highlight the newly hovered over cell and it's row/column
          // tokens.
          d3.select(this).classed('cell-hover', true);
          sourceLabels.select('#row-' + d.row)
              .classed('text-highlight', true);
          targetLabels.select('#col-' + d.col)
              .classed('text-highlight', true);
        })
        .on('mouseout', function(d) {
          // Clear all highlighting.
          d3.select(this).classed('cell-hover', false);

          sourceLabels.select('#row-' + d.row)
              .classed('text-highlight', false);
          targetLabels.select('#col-' + d.col)
              .classed('text-highlight', false);
        });

    // Add the rectangles for each cell.
    cellGroup
        .append('rect')
        .attr(
            'id',
            function(d, i) {
              return 'cell-' + i;
            })
        .attr('class', 'cell cell-border')
        .attr(
            'x',
            function(d) {
              return d.col * cellDimension;
            })
        .attr(
            'y',
            function(d) {
              return d.row * cellDimension;
            })
        .attr('width', cellDimension)
        .attr('height', cellDimension)
        .style(
            'fill',
            function(d) {
              return colorScale(d.value);
            });

    // Add the text for each cell.
    cellGroup
        .append('text')
        .text(function(d) { return d.value.toFixed(2); })
        .attr('class', 'weight weight-label')
        .attr('x', function(d) { return 5 + (d.col * cellDimension); })
        .attr('y', function(d) { return 25 + (d.row * cellDimension); });
  }

  /**
   * Resets the pan and zoom state.
   * @private
   */
  reset_() {
    if (!this.svg_) { return; }
    this.vis_.call(this.zoom_.transform, d3.zoomIdentity);
    this.set('zoomDepth_', 20);
  }
}

customElements.define(AttentionVisualization.is, AttentionVisualization);
