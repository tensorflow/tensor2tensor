/**
 * @fileoverview Transformer Visualization D3 javascript code.
 */

requirejs(['jquery', 'd3'],
function($, d3) {

var attention = window.attention;

const TEXT_SIZE = 15;
const BOXWIDTH = TEXT_SIZE * 8;
const BOXHEIGHT = TEXT_SIZE * 1.5;
const WIDTH = 2000;
const HEIGHT = attention.all.bot_text.length * BOXHEIGHT * 2 + 100;
const MATRIX_WIDTH = 150;
const head_colours = d3.scale.category10();
const CHECKBOX_SIZE = 20;

function lighten(colour) {
  var c = d3.hsl(colour);
  var increment = (1 - c.l) * 0.6;
  c.l += increment;
  c.s -= increment;
  return c;
}

function transpose(mat) {
  return mat[0].map(function(col, i) {
    return mat.map(function(row) {
      return row[i];
    });
  });
}

function zip(a, b) {
  return a.map(function (e, i) {
    return [e, b[i]];
  });
}


function renderVis(id, top_text, bot_text, attention_heads, config) {
  $(id).empty();
  var svg = d3.select(id)
            .append('svg')
            .attr("width", WIDTH)
            .attr("height", HEIGHT);

  var att_data = [];
  for (var i=0; i < attention_heads.length; i++) {
    var att_trans = transpose(attention_heads[i]);
    att_data.push(zip(attention_heads[i], att_trans));
  }

  renderText(svg, top_text, true, att_data, 0);
  renderText(svg, bot_text, false, att_data, MATRIX_WIDTH + BOXWIDTH);

  renderAttentionHighlights(svg, att_data);

  svg.append("g").classed("attention_heads", true);

  renderAttention(svg, attention_heads);

  draw_checkboxes(config, 0, svg, attention_heads);
}


function renderText(svg, text, is_top, att_data, left_pos) {
  var id = is_top ? "top" : "bottom";
  var textContainer = svg.append("svg:g")
                         .attr("id", id);

  textContainer.append("g").classed("attention_boxes", true)
               .selectAll("g")
               .data(att_data)
               .enter()
               .append("g")
               .selectAll("rect")
               .data(function(d) {return d;})
               .enter()
               .append("rect")
               .attr("x", function(d, i, j) {
                 return left_pos + box_offset(j);
               })
               .attr("y", function(d, i) {
                 return (+1) * BOXHEIGHT;
               })
               .attr("width", BOXWIDTH/active_heads())
               .attr("height", function() { return BOXHEIGHT; })
               .attr("fill", function(d, i, j) {
                  return head_colours(j);
                })
               .style("opacity", 0.0);


  var tokenContainer = textContainer.append("g").selectAll("g")
                                    .data(text)
                                    .enter()
                                    .append("g");

  tokenContainer.append("rect")
                .classed("background", true)
                .style("opacity", 0.0)
                .attr("fill", "lightgray")
                .attr("x", left_pos)
                .attr("y", function(d, i) {
                  return (i+1) * BOXHEIGHT;
                })
                .attr("width", BOXWIDTH)
                .attr("height", BOXHEIGHT);

  var theText = tokenContainer.append("text")
                              .text(function(d) { return d; })
                              .attr("font-size", TEXT_SIZE + "px")
                              .style("cursor", "default")
                              .style("-webkit-user-select", "none")
                              .attr("x", left_pos)
                              .attr("y", function(d, i) {
                                return (i+1) * BOXHEIGHT;
                              });

  if (is_top) {
    theText.style("text-anchor", "end")
           .attr("dx", BOXWIDTH - TEXT_SIZE)
           .attr("dy", TEXT_SIZE);
  } else {
    theText.style("text-anchor", "start")
           .attr("dx", + TEXT_SIZE)
           .attr("dy", TEXT_SIZE);
  }

  tokenContainer.on("mouseover", function(d, index) {
    textContainer.selectAll(".background")
                 .style("opacity", function(d, i) {
                   return i == index ? 1.0 : 0.0;
                 });

    svg.selectAll(".attention_heads").style("display", "none");

    svg.selectAll(".line_heads")  // To get the nesting to work.
       .selectAll(".att_lines")
       .attr("stroke-opacity", function(d) {
          return 1.0;
        })
       .attr("y1", function(d, i) {
        if (is_top) {
          return (index+1) * BOXHEIGHT + (BOXHEIGHT/2);
        } else {
          return (i+1) * BOXHEIGHT + (BOXHEIGHT/2);
        }
     })
     .attr("x1", BOXWIDTH)
     .attr("y2", function(d, i) {
       if (is_top) {
          return (i+1) * BOXHEIGHT + (BOXHEIGHT/2);
        } else {
          return (index+1) * BOXHEIGHT + (BOXHEIGHT/2);
        }
     })
     .attr("x2", BOXWIDTH + MATRIX_WIDTH)
     .attr("stroke-width", 2)
     .attr("stroke", function(d, i, j) {
        return head_colours(j);
      })
     .attr("stroke-opacity", function(d, i, j) {
      if (is_top) {d = d[0];} else {d = d[1];}
      if (config.head_vis[j]) {
        if (d) {
          return d[index];
        } else {
          return 0.0;
        }
      } else {
        return 0.0;
      }
     });


    function updateAttentionBoxes() {
      var id = is_top ? "bottom" : "top";
      var the_left_pos = is_top ? MATRIX_WIDTH + BOXWIDTH : 0;
      svg.select("#" + id)
         .selectAll(".attention_boxes")
         .selectAll("g")
         .selectAll("rect")
         .attr("x", function(d, i, j) { return the_left_pos + box_offset(j); })
         .attr("y", function(d, i) { return (i+1) * BOXHEIGHT; })
         .attr("width", BOXWIDTH/active_heads())
         .attr("height", function() { return BOXHEIGHT; })
         .style("opacity", function(d, i, j) {
            if (is_top) {d = d[0];} else {d = d[1];}
            if (config.head_vis[j])
              if (d) {
                return d[index];
              } else {
                return 0.0;
              }
            else
              return 0.0;

         });
    }

    updateAttentionBoxes();
  });

  textContainer.on("mouseleave", function() {
    d3.select(this).selectAll(".background")
                   .style("opacity", 0.0);

    svg.selectAll(".att_lines").attr("stroke-opacity", 0.0);
    svg.selectAll(".attention_heads").style("display", "inline");
    svg.selectAll(".attention_boxes")
       .selectAll("g")
       .selectAll("rect")
       .style("opacity", 0.0);
  });
}

function renderAttentionHighlights(svg, attention) {
  var line_container = svg.append("g");
  line_container.selectAll("g")
                .data(attention)
                .enter()
                .append("g")
                .classed("line_heads", true)
                .selectAll("line")
                .data(function(d){return d;})
                .enter()
                .append("line").classed("att_lines", true);
}

function renderAttention(svg, attention_heads) {
  var line_container = svg.selectAll(".attention_heads");
  line_container.html(null);
  for(var h=0; h<attention_heads.length; h++) {
    for(var a=0; a<attention_heads[h].length; a++) {
      for(var s=0; s<attention_heads[h][a].length; s++) {
        line_container.append("line")
        .attr("y1", (s+1) * BOXHEIGHT + (BOXHEIGHT/2))
        .attr("x1", BOXWIDTH)
        .attr("y2", (a+1) * BOXHEIGHT + (BOXHEIGHT/2))
        .attr("x2", BOXWIDTH + MATRIX_WIDTH)
        .attr("stroke-width", 2)
        .attr("stroke", head_colours(h))
        .attr("stroke-opacity", function() {
          if (config.head_vis[h]) {
            return attention_heads[h][a][s]/active_heads();
          } else {
            return 0.0;
          }
        }());
      }
    }
  }
}

// Checkboxes
function box_offset(i) {
  var num_head_above = config.head_vis.reduce(
      function(acc, val, cur) {return val && cur < i ? acc + 1: acc;}, 0);
  return num_head_above*(BOXWIDTH / active_heads());
}

function active_heads() {
  return config.head_vis.reduce(function(acc, val) {
    return val ? acc + 1: acc;
  }, 0);
}

function draw_checkboxes(config, top, svg, attention_heads) {
  var checkboxContainer = svg.append("g");
  var checkbox = checkboxContainer.selectAll("rect")
                                  .data(config.head_vis)
                                  .enter()
                                  .append("rect")
                                  .attr("fill", function(d, i) {
                                    return head_colours(i);
                                  })
                                  .attr("x", function(d, i) {
                                    return (i+1) * CHECKBOX_SIZE;
                                  })
                                  .attr("y", top)
                                  .attr("width", CHECKBOX_SIZE)
                                  .attr("height", CHECKBOX_SIZE);

  function update_checkboxes() {
    checkboxContainer.selectAll("rect")
                              .data(config.head_vis)
                              .attr("fill", function(d, i) {
      var head_colour = head_colours(i);
      var colour = d ? head_colour : lighten(head_colour);
      return colour;
    });
  }

  update_checkboxes();

  checkbox.on("click", function(d, i) {
    if (config.head_vis[i] && active_heads() == 1) return;
    config.head_vis[i] = !config.head_vis[i];
    update_checkboxes();
    renderAttention(svg, attention_heads);
  });

  checkbox.on("dblclick", function(d, i) {
    // If we double click on the only active head then reset
    if (config.head_vis[i] && active_heads() == 1) {
      config.head_vis = new Array(config.num_heads).fill(true);
    } else {
      config.head_vis = new Array(config.num_heads).fill(false);
      config.head_vis[i] = true;
    }
    update_checkboxes();
    renderAttention(svg, attention_heads);
  });
}

var config = {
  layer: 0,
  att_type: 'all',
};

function visualize() {
  var num_heads = attention['all']['att'][0].length;
  config.head_vis  = new Array(num_heads).fill(true);
  config.num_heads = num_heads;
  config.attention = attention;

  render();
}

function render() {
  var conf = config.attention[config.att_type];

  var top_text = conf.top_text;
  var bot_text = conf.bot_text;
  var attention = conf.att[config.layer];

  $("#vis svg").empty();
  renderVis("#vis", top_text, bot_text, attention, config);
}

$("#layer").empty();
for(var i=0; i<6; i++) {
  $("#layer").append($("<option />").val(i).text(i));
}

$("#layer").on('change', function(e) {
  config.layer = +e.currentTarget.value;
  render();
});

$("#att_type").on('change', function(e) {
  config.att_type = e.currentTarget.value;
  render();
});

$("button").on('click', visualize);

visualize();

});
