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
 * `<graph-visualization>` Presents a beam search decoding graph.
 *
 * The Beam Search decoding graph visualizes the entire search space of a
 * sequence generation model.  Each layer in the graph displays a decoding step
 * with nodes in that layer representing generated candidates.  If supported by
 * the backend server, the graph can enter interactive mode where candidates can
 * be selected for each generation step.
 *
 *
 * ### Usage
 *
 *   <graph-visualization data="[[data]]"></graph-visualization>
 */
class GraphVisualization extends Polymer.Element {
  constructor() {
    super();

    /**
     * @private
     */
    this.svg_ = undefined;
    /**
     * @private
     */
    this.vis_ = undefined;

    /**
     * @type {!TreeNode}
     * @private
     */
    this.rootTree_ = {
      name: '',
      localProbability: 0,
      cumalitiveProbability: 0,
      score: 0,
      attention: [],
      children: [],
    };
    /**
     * @type {!InteractiveNode}
     * @private
     */
    this.interactiveRoot_ = {
      id: this.nodeId_,
      stepIndex: 0,
      candidate: {
        label: '<s>',
        label_id: 1,
        log_probability: 0,
        total_log_probability: 0,
        score: 0,
        parent_id: 0
      },
      children: [],
    };
    /**
     * @type {Array<!InteractiveNode>}
     * @private
     */
    this.selectedNodes_ = [];
    /**
     * @private
     */
    this.stepNodes_ = [];

    /**
     * Metadata for navigating nodes.
     * @private
     */
    this.nodeId_ = 0;

    /**
     * D3.js helper object.
     * @private
     */
    this.partition_ = undefined;
    /**
     * D3.js helper object.
     * @private
     */
    this.zoom_ = undefined;

    /**
     * D3.js DOM element.
     * @private
     */
    this.container_ = undefined;
  }

  /**
   * @return {string} The component name.
   */
  static get is() {
    return 'graph-visualization';
  }

  /**
   * @return {!Object} The component properties.
   */
  static get properties() {
    return {
      /**
       * @type {!SearchGraphVisualization}
       */
      data: {
        type: Object,
        observer: 'dataUpdated_',
      },
      /**
       * @type {!Model}
       */
      model: {
        type: Object,
      },
      /**
       * @type {string}
       */
      query: {
        type: String,
      },
      /**
       * @type {number}
       */
      zoomDepth_: {
        type: Number,
        value: 20,
      },
      /**
       * @type {!StartTranslationResponse}
       */
      startResponse_: {
        type: Object,
      },
      /**
       * @type {!GenerateCandidateResponse}
       */
      generateResponse_: {
        type: Object,
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
    this.set('stepMode', 'view');
  }

  /**
   * Sets the zoom state based on the updated depth.
   * @param {number} zoomDepth the zoom depth.
   * @private
   */
  zoomDepthChanged_(zoomDepth) {
    if (!this.svg_) {
      return;
    }

    if (zoomDepth == 0) {
      zoomDepth = 0.000001;
    }
    let transform = d3.zoomTransform(this.svg_.node()).scale(zoomDepth / 20.0);
    this.vis_.attr("transform", transform);
  }

  /**
   * Converts the NMT Graph JSON format to a nested tree heirachy and plots the
   * tree as a collapsible tree visualization.
   * @private
   */
  dataUpdated_() {
    // We need to determine two key nodes in the graph:
    //   Root: This is the node with no in links and some out links.
    //   Term: This is the terminal node with no out links and some in links.
    //
    // Our plot will associate token with actual nodes.  For all nodes except
    // the Term node, this will work fine since in the tree, each node is
    // referenced only once as the head of an edge.
    //
    // The Term node however needs to be duplicated for each edge ending at it
    // so that each instance can have a unique token associated with it.

    // Step 1) Find Root and Term node indices so they can be refered to later.

    var rootIndex = -1;
    var nodes = this.data.node;
    for (var i = 0; i < nodes.length && rootIndex == -1; ++i) {
      var node = nodes[i];
      if (node.in_edge_index.length == 0 && node.out_edge_index.length != 0) {
        rootIndex = i;
      }
    }

    // Step 2) Create the root node in the tree.  The tree structure will have
    // the following components:
    //   name: The display name of the node.  This will be some token.
    //   localProbability: The per time step probability of this node.
    //   cumulativeProbability: The total probability of this path in the beam
    //       search.
    //   score: A final score for this path in the beam search.  This is
    //       typically the cumulativeProbability with zero or more penalties.
    //   attention: The attention vector associated with this node transition.
    //   children: The list of children in the tree, which are themselves trees.
    this.rootTree_ = {
      name: '',
      localProbability: 0,
      cumalitiveProbability: 0,
      score: 0,
      attention: [],
      children: [],
    };

    // Step3) Add each child and it's children recursively starting from the
    // root node.
    var rootNode = nodes[rootIndex];
    var edges = this.data.edge;
    for (var i = 0; i < rootNode.out_edge_index.length; ++i) {
      // Get the edge.
      var outEdge = edges[rootNode.out_edge_index[i]];
      this.addChildToTree_(this.rootTree_, outEdge, nodes, edges);
    }
    this.propagateLabel_(this.rootTree_);

    this.createSVG_();
    this.plotTree_(this.rootTree_);
  }

  /**
   * Forwards path labels from a node's child to the current node.
   * @param {!TreeNode} node The node to annotate.
   * @private
   */
  propagateLabel_(node) {
    var hasNBest = false;
    var hasBeam = false;
    var hasAlternative = false;
    for (var i = 0; i < node.children.length; ++i) {
      hasNBest = hasNBest || node.children[i].pathType == 'nbest';
      hasBeam = hasBeam || node.children[i].pathType == 'beam';
      hasAlternative = hasAlternative ||
          node.children[i].pathType == 'alternative';
    }

    if (hasNBest) {
      node.pathType = 'nbest';
    } else if (hasBeam) {
      node.pathType = 'beam';
    } else if (hasAlternative) {
      node.pathType = 'beam';
    } else {
      node.pathType = 'unknown';
    }
  }

  /**
   * Iterates through all the children in tree and adds them as children to the
   * top level tree.
   * @param {!TreeNode} tree The current node in the tree to update with
   *     children.
   * @param {!BeamSearchEdge} currentEdge The edge going into tree.
   * @param {!Array<!BeamSearchNode>} nodes The list of all node objects.
   * @param {!Array<!BeamSearchEdge>} edges The list of all edges between nodes.
   * @private
   */
  addChildToTree_(tree, currentEdge, nodes, edges) {
    // The real edge information is nested in wonderfully named proto
    // extensions.  Extract the extension information appropriately.
    var candidate = currentEdge.data;

    // When the label for the new child is empty, we're at a terminal sink.  So
    // we ignore that node and instead label the parent.
    if (candidate.label == '') {
      tree.pathType = 'alternative';
      return;
    }

    var node = nodes[currentEdge.target_index];
    /**
     * @type {TreeNode}
     */
    var childTree = {
      name: candidate.label,
      attention: [],
      localProbability: Math.pow(Math.E, candidate.log_probability),
      cumalitiveProbability: Math.pow(Math.E, candidate.total_log_probability),
      score: Math.pow(Math.E, candidate.score),
      finished: currentEdge.completed || false,
      children: [],
      node: node,
      edge: currentEdge,
      pathType: 'unknown',
    };
    tree.children.push(childTree);

    if (node.out_edge_index.length == 0) {
      if (childTree.name == '</s>') {
        childTree.pathType = 'nbest';
      } else if (childTree.name == '' || candidate.finished) {
        childTree.pathType = 'alternative';
      } else {
        childTree.pathType = 'beam';
      }
    } else {
      for (var i = 0; i < node.out_edge_index.length; ++i) {
        // Get the edge.
        var outEdge = edges[node.out_edge_index[i]];
        this.addChildToTree_(childTree, outEdge, nodes, edges);
        this.propagateLabel_(childTree);
      }
    }
  }

  /**
   * Creates the initial SVG canvas and associated structures.  This will remove
   * all previous svg elements.
   * @private
   */
  createSVG_() {
    // Create the margins, width, and height.
    var maxWidth = 1600;
    var maxHeight = 1600;
    var margins = [20, 120, 20, 20];
    var width = maxWidth - margins[1] - margins[3];
    var height = maxHeight - margins[0] - margins[2];

    // Use a d3 partition which will place each node based it's number of
    // descendents with the highest ranked path along the top.
    this.partition_ = d3.partition().size([height, width]).padding(1);

    // Set the initial position of the root of the tree to be a half the height
    // and on the left..
    this.rootTree_.x0 = height / 2;
    this.rootTree_.y0 = 0;

    this.zoom_ = d3.zoom()
        .scaleExtent([1, 10])
        .on("zoom", zoomed.bind(this));

    d3.select(this.$.chart).selectAll('.svg-container').remove();

    // Embed the SVG to host the tree and rotate it so that horizontal matches
    // the height of the canvas.
    this.svg_ = d3.select(this.$.chart)
        .append("div")
        .classed("svg-container", true)
        .append("svg")
        .attr("height", "100%")
        .attr("width", "100%")
        .classed("svg-content-responsive", true)
        .call(this.zoom_)
        .on('dblclick.zoom', null)
        .on('wheel.zoom', null);

    /**
     * Note: For reasons not understood, the javascript compiler can't figure
     * out the type of _zoomDepth at this line, so we need to coerce it into
     * being a number.
     * @type {number}
     */
    let zoomDepth = parseInt(this.zoomDepth_, 10);
    let transform = d3.zoomTransform(this.svg_.node()).scale(zoomDepth / 20.0);
    this.vis_ = this.svg_.append('g')
        .attr("transform", transform);

    // Ensure that the entire svg element can be used for panning.
    this.vis_.append("rect")
        .attr("width", maxWidth)
        .attr("height", maxWidth)
        .style("fill", "none")
        .style("pointer-events", "all");

    this.container_ = this.vis_.append("g");

    // Apply the zoom transformation.
    function zoomed() {
      this.vis_.attr("transform",
          d3.event.transform.scale(this.zoomDepth_ / 20.0));
    }
  }

  /**
   * Examines and plots all reachable nodes in the rootTree with respect to the
   * given current root.
   * @param {!TreeNode} root The current root node to focus on.
   * @private
   */
  plotTree_(root) {
    // Create the hierarchy.  We accumulate node values by just counting the
    // number of elements, rather than placing a weight on each node..
    var treeHierachy = d3.hierarchy(this.rootTree_)
                           .sum(function(d) {
                             return 1;
                           })
                           .sort(function(a, b) {
                             return a.data.score - b.data.score;
                           });

    this.partition_(treeHierachy);

    // Create an enter object where we can add both nodes and links.
    var enter = this.container_.selectAll(".node")
        .data(treeHierachy.descendants())
        .enter();

    // Add the nodes in four steps:
    //   1) A general group element to hold all node portions.
    //   2) A rectangle with no visible elements.
    //   3) A circle for the node.
    //   4) a text label.
    var node = enter.append("g")
        .attr("class", function(d) {
          return "node" + (d.children ? " node--internal" : " node--leaf");
        })
        .attr("transform", function(d) {
          return "translate(" + d.y0 + "," + d.x0 + ")";
        })
        .attr('id', function(d, i) { return "g-" + i; });

    node.append("rect")
        .attr("width", function(d) { return d.y1 - d.y0; })
        .attr("height", 24);

    node.append("circle")
        .attr("r", 10)
        .attr("transform", "translate(10, 10)");

    node.append("text")
        .attr("x", 24)
        .attr("y", 13)
        .text(function(d) { return d.data.name; });

    // Add out links from each node to it's parent.  We link two nodes using the
    // bottom center of the circle so that the text label can be placed at
    // approximately the vertical center of the circle.  This gives a decent
    // layout while also not hiding any text.
    enter.append("path")
        .attr("class", "link")
        .attr("d", function(d) {
          if (!d.parent) { return ""; }
          // Pad the placement of the links just below the center.  We have to
          // use x0 and y0 for location due to partition, which doesn't create
          // standard x/y fields.
          var nodeX = d.x0 + 16;
          var nodeY = d.y0 + 10;
          var parentX = d.parent.x0 + 16;
          var parentY = d.parent.y0 + 10;
          return "M" + + nodeY + "," + nodeX +
                 "C" + (nodeY + parentY) / 2 + "," + nodeX + " " +
                 (nodeY + parentY) / 2 + "," + parentX + " " +
                 parentY + "," + parentX;
        })
        .style('stroke', function(d) {
          // Associate a different path color depend on the path type for the
          // node.
          if (d.data.pathType == 'unknown')
            return '#222';
          if (d.data.pathType == 'nbest')
            return '#66ff33';
          if (d.data.pathType == 'beam')
            return '#ccc';
          if (d.data.pathType == 'alternative')
            return '#ff3300';
        });

    // Setup hover events on each node to place focus and highligh on the node
    // being hovered over.  We do this by adding opacity to all other nodes.
    var nodes = this.container_.selectAll(".node");
    node.on('mouseover', function(d, i) {
        nodes.classed('fade', function(d, j) {
          return i != j;
        });
        d3.select(this).classed('hover', true);
        this.set('currentName', d.data.name);
        this.set(
            'currentProbability', this.displayNumber(d.data.localProbability));
        this.set(
            'currentTotalProbability',
            this.displayNumber(d.data.cumalitiveProbability));
        this.set('score', this.displayNumber(d.data.score));
      }.bind(this))
      .on('mouseout', function(d, i) {
        nodes.classed("fade", false);
        d3.select(this).classed("hover", false);
      });
  }

  /**
   * Resets the pan and zoom state.
   * @private
   */
  reset_() {
    if (!this.svg_) {
      return;
    }
    this.svg_.call(this.zoom_.transform, d3.zoomIdentity);
    this.set('zoomDepth_', 20);
  }

  /**
   * Returns the number value with only 2 significant digits.
   * @param {number} value The value to present.
   * @return {string} value with just two significant digits.
   */
  displayNumber(value) {
    return value.toFixed(2);
  }

  /**
   * Enters step by step decoding mode.
   * @private
   */
  startStepMode_() {
    this.set('stepMode', 'edit');
    this.startTranslation_();
  }

  /**
   * Exits step by step decoding mode.
   * @private
   */
  exitStepMode_() {
    this.set('stepMode', 'view');
    this.dataUpdated_();
  }

  /**
   * Begins step by step decoding with the current model and query.
   * @private
   */
  startTranslation_() {
    this.set('startBody', JSON.stringify({
      model_id: {
        language_pair: {
          source_language: this.model.source_language.code,
          target_language: this.model.target_language.code,
        },
        name: this.model.id,
      },
      input: this.query,
    }));
    this.$.startAjax.generateRequest();
  }

  /**
   * Handles a start error.
   * @private
   */
  handleStartError_() {
    console.log("failed");
  }

  /**
   * Initializes the step by step decoding graph with the root note and makes
   * the first generation step.
   * @private
   */
  handleStartResponse_() {
    // Reset the node state and create the root of the tree.  Later candidates
    // that are returned from the generation call will be added.
    this.nodeId_ = 0;
    this.interactiveRoot_ = {
      id: this.nodeId_,
      stepIndex: 0,
      candidate: {
        label: '<s>',
        label_id: 1,
        log_probability: 0,
        total_log_probability: 0,
        score: 0,
        parent_id: 0
      },
      children: [],
    };
    this.nodeId_++;

    // Track which nodes are active and available as inputs to the next
    // generation step.  These will be updated with the candidates they
    // generate.
    this.selectedNodes_ = [this.interactiveRoot_];

    // Redraw the entire plot with an interactive version.
    this.createSVG_();
    this.drawInteractiveTree_(this.interactiveRoot_);

    // Make the first generation request.
    this.step_(true);
  }

  /**
   * Handles a generate ajax error.
   * @private
   */
  handleGenerateError_() {
    console.log("generate failed");
  }

  /**
   * Processes the returned candidates and adds them to the graph.
   * @private
   */
  handleGenerateResponse_() {
    // Add the candidates returned and tag them with unique identifiers so we
    // can ensure later generation steps don't try to include candidates that
    // can't be proccesed any more (we can only use candidates from the most
    // recent generation step as input due to limitations in the remote
    // decoder).
    let stepIndex = 0;
    let newlySelectedNodes = [];
    this.stepNodes_ = [];
    for (var i = 0; i < this.generateResponse_.candidate_list.length; ++i) {
      let selectedNode = this.selectedNodes_[i];
      let candidateList = this.generateResponse_.candidate_list[i];
      for (var j = 0; j < candidateList.candidate.length && j < 5; ++j) {
        let candidate = candidateList.candidate[j];
        // Tag the parent id so that the next generate call knows what network
        // states to maintain.
        candidate.parent_id = i;
        let newNode = {
          id: this.nodeId_,
          stepIndex: stepIndex,
          candidate: candidate,
          children: [],
        };
        this.nodeId_++;
        stepIndex++;
        this.stepNodes_.push(newNode);
        selectedNode.children.push(newNode);

        // Select the first candidate.
        if (j === 0) {
          newNode.selected = true;
          newlySelectedNodes.push(newNode);
        }
      }
    }
    this.selectedNodes_ = newlySelectedNodes;

    // Reset the graph.
    this.createSVG_();
    this.drawInteractiveTree_(this.interactiveRoot_);
  }

  /**
   * Draws the interactive tree.
   * @param {InteractiveNode} rootNode The root node to draw out.
   * @private
   */
  drawInteractiveTree_(rootNode) {
    let treeHierachy = d3.hierarchy(rootNode)
                           .sum(function(d) {
                             return 1;
                           })
                           .sort(function(a, b) {
                             return b.data.candidate.total_log_probability -
                                    a.data.candidate.total_log_probability;
                           });

    this.partition_(treeHierachy);

    // Create an enter object where we can add both nodes and links.
    var enter = this.container_.selectAll(".node")
        .data(treeHierachy.descendants())
        .enter();

    // Add the nodes in four steps:
    //   1) A general group element to hold all node portions.
    //   2) A rectangle with no visible elements.
    //   3) A circle for the node.
    //   4) a text label.
    var node = enter.append("g")
        .attr("class", function(d) {
          return "node" +
              (d.children ? " node--internal" : " node--leaf") +
              (d.data.selected ? " selected" : "");
        })
        .attr("transform", function(d) {
          return "translate(" + d.y0 + "," + d.x0 + ")";
        })
        .attr('id', function(d, i) { return "g-" + i; });

    node.append("rect")
        .attr("width", function(d) { return d.y1 - d.y0; })
        .attr("height", 24);

    node.append("circle")
        .attr("r", 10)
        .attr("transform", "translate(10, 10)");

    node.append("text")
        .attr("x", 24)
        .attr("y", 13)
        .text(function(d) { return d.data.candidate.label; });

    // Add out links from each node to it's parent.  We link two nodes using the
    // bottom center of the circle so that the text label can be placed at
    // approximately the vertical center of the circle.  This gives a decent
    // layout while also not hiding any text.
    enter.append("path")
        .attr("class", "link")
        .attr("d", function(d) {
          if (!d.parent) { return ""; }
          // Pad the placement of the links just below the center.  We have to
          // use x0 and y0 for location due to partition, which doesn't create
          // standard x/y fields.
          var nodeX = d.x0 + 16;
          var nodeY = d.y0 + 10;
          var parentX = d.parent.x0 + 16;
          var parentY = d.parent.y0 + 10;
          return "M" + + nodeY + "," + nodeX +
                 "C" + (nodeY + parentY) / 2 + "," + nodeX + " " +
                 (nodeY + parentY) / 2 + "," + parentX + " " +
                 parentY + "," + parentX;
        })
        .style('stroke', '#ccc');

    node.on('mouseover', function(d, i) {
      this.set('currentName', d.data.candidate.label);
      this.set(
          'currentProbability',
          this.displayNumber(Math.exp(d.data.candidate.log_probability)));
      this.set(
          'currentTotalProbability',
          this.displayNumber(Math.exp(d.data.candidate.total_log_probability)));
      this.set('score', this.displayNumber(Math.exp(d.data.candidate.score)));
    }.bind(this));

    // Store a local pointer to stepNodes and selectedNodes so that the click
    // handler can access them without having to replace the 'this' pointer.
    // The click handler needs the default 'this' handler to update the state of
    // the clicked upon node.
    let stepNodes = this.stepNodes_;
    let selectedNodes = this.selectedNodes_;

    node.on('click', function(d, i) {
      // Ignore nodes that fall out of bounds.
      let stepIndex = d.data.stepIndex;
      if (stepIndex >= stepNodes.length) {
        return;
      }

      // Ignore nodes that are from different steps.
      let node = stepNodes[stepIndex];
      if (node.id != d.data.id) {
        return;
      }

      // Update the selected state of the node and either add it to the selected
      // list or remove it.
      if (!node.selected) {
        node.selected = true;
        selectedNodes.push(node);
      } else {
        node.selected = false;
        selectedNodes.splice(selectedNodes.indexOf(node), 1);
      }
      d3.select(this).classed('selected', node.selected);
    });
  }

  /**
   * Make one generation step with the candidates in the current selectedNodes
   * list.  If no nodes are selected, this silently does nothing.
   * @param {boolean=} opt_skipNext If true, skips the next step during
   *     generation.
   * @private
   */
  step_(opt_skipNext) {
    // Running generate without any nodes can put the decoder into a bad state
    // and make the session unusable, so for now, silently skip this case.
    if (this.selectedNodes_.length == 0) {
      console.log("Skipping empty step.");
      return;
    }

    this.set('generateParams', {
      skip_next: opt_skipNext || false,
    });
    this.set('generateBody', JSON.stringify({
      model_id: {
        language_pair: {
          source_language: this.model.source_language.code,
          target_language: this.model.target_language.code,
        },
        name: this.model.id,
      },
      session_id: this.startResponse_.session_id,
      candidate: this.selectedNodes_.map(function(node) {
        return node.candidate;
      }),
    }));
    this.$.generateAjax.generateRequest();
  }

}

customElements.define(GraphVisualization.is, GraphVisualization);
