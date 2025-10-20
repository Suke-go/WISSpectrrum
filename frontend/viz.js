import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

// Global state
let svg, g, root, focus;
let containerEl;
let onSelectConceptCallback = () => {};
let treeData = null;

const width = 928;
const radius = width / 6;

// Color scale for top-level categories
function color(d) {
  const colors = d3.quantize(d3.interpolateRainbow, d.children ? d.children.length + 1 : 1);
  if (d.depth === 0) return "#fff";
  while (d.depth > 1) d = d.parent;
  const index = d.parent.children.indexOf(d);
  return colors[index];
}

/**
 * Initialize the zoomable sunburst visualization
 * Based on Observable's canonical implementation
 */
export function initBubble(data, options = {}) {
  containerEl = document.getElementById("viz-container");
  if (!containerEl) return;

  containerEl.innerHTML = "";
  treeData = data;
  onSelectConceptCallback = options.onSelectConcept ?? (() => {});

  if (!data || !Array.isArray(data.children) || data.children.length === 0) {
    containerEl.innerHTML = `<p class="empty-note">No concepts available.</p>`;
    return;
  }

  // Create hierarchy
  root = d3.hierarchy(data)
    .sum(d => d.count || 1)
    .sort((a, b) => b.value - a.value);

  focus = root;

  // Compute partition layout
  const partition = data => {
    const root = d3.hierarchy(data)
      .sum(d => d.count || 1)
      .sort((a, b) => b.value - a.value);
    return d3.partition()
      .size([2 * Math.PI, root.height + 1])
    (root);
  };

  root = partition(data);
  root.each(d => d.current = d);

  // Arc generator
  const arc = d3.arc()
    .startAngle(d => d.x0)
    .endAngle(d => d.x1)
    .padAngle(d => Math.min((d.x1 - d.x0) / 2, 0.005))
    .padRadius(radius * 1.5)
    .innerRadius(d => d.y0 * radius)
    .outerRadius(d => Math.max(d.y0 * radius, d.y1 * radius - 1));

  // Create SVG
  svg = d3.select(containerEl)
    .append("svg")
    .attr("viewBox", [-width / 2, -width / 2, width, width])
    .attr("width", "100%")
    .attr("height", "100%")
    .style("font", "10px sans-serif");

  g = svg.append("g");

  // Create paths
  const path = g.append("g")
    .selectAll("path")
    .data(root.descendants().slice(1))
    .join("path")
      .attr("fill", d => { while (d.depth > 1) d = d.parent; return color(d); })
      .attr("fill-opacity", d => arcVisible(d.current) ? (d.children ? 0.6 : 0.4) : 0)
      .attr("pointer-events", d => arcVisible(d.current) ? "auto" : "none")
      .attr("d", d => arc(d.current))
      .style("cursor", "pointer");

  path.filter(d => d.children)
    .style("cursor", "pointer")
    .on("click", clicked);

  path.append("title")
    .text(d => `${d.ancestors().map(d => d.data.name).reverse().join("/")}\n${d.value} papers`);

  // Create labels
  const label = g.append("g")
    .attr("pointer-events", "none")
    .attr("text-anchor", "middle")
    .style("user-select", "none")
    .selectAll("text")
    .data(root.descendants().slice(1))
    .join("text")
      .attr("dy", "0.35em")
      .attr("fill-opacity", d => +labelVisible(d.current))
      .attr("transform", d => labelTransform(d.current))
      .text(d => d.data.name);

  // Create center circle
  g.append("circle")
    .datum(root)
    .attr("r", radius)
    .attr("fill", "none")
    .attr("pointer-events", "all")
    .style("cursor", "pointer")
    .on("click", clicked);

  // Click handler
  function clicked(event, p) {
    if (!p) p = focus.parent || root;

    focus = focus === p ? p : p;

    // Notify about selection if it's a leaf
    if (!p.children && p.data.id) {
      onSelectConceptCallback(p.data);
    }

    root.each(d => d.target = {
      x0: Math.max(0, Math.min(1, (d.x0 - p.x0) / (p.x1 - p.x0))) * 2 * Math.PI,
      x1: Math.max(0, Math.min(1, (d.x1 - p.x0) / (p.x1 - p.x0))) * 2 * Math.PI,
      y0: Math.max(0, d.y0 - p.depth),
      y1: Math.max(0, d.y1 - p.depth)
    });

    const t = g.transition().duration(750);

    path.transition(t)
      .tween("data", d => {
        const i = d3.interpolate(d.current, d.target);
        return t => d.current = i(t);
      })
      .filter(function(d) {
        return +this.getAttribute("fill-opacity") || arcVisible(d.target);
      })
      .attr("fill-opacity", d => arcVisible(d.target) ? (d.children ? 0.6 : 0.4) : 0)
      .attr("pointer-events", d => arcVisible(d.target) ? "auto" : "none")
      .attrTween("d", d => () => arc(d.current));

    label.filter(function(d) {
        return +this.getAttribute("fill-opacity") || labelVisible(d.target);
      }).transition(t)
      .attr("fill-opacity", d => +labelVisible(d.target))
      .attrTween("transform", d => () => labelTransform(d.current));
  }

  function arcVisible(d) {
    return d.y1 <= 3 && d.y0 >= 1 && d.x1 > d.x0;
  }

  function labelVisible(d) {
    return d.y1 <= 3 && d.y0 >= 1 && (d.y1 - d.y0) * (d.x1 - d.x0) > 0.03;
  }

  function labelTransform(d) {
    const x = (d.x0 + d.x1) / 2 * 180 / Math.PI;
    const y = (d.y0 + d.y1) / 2 * radius;
    return `rotate(${x - 90}) translate(${y},0) rotate(${x < 180 ? 0 : 180})`;
  }
}

/**
 * Update highlight based on active concept
 */
export function updateBubbleHighlight({ activeId, searchTerm }) {
  if (!svg) return;

  const normalizedSearch = (searchTerm || "").toLowerCase();

  svg.selectAll("path")
    .attr("stroke", d => {
      if (activeId && d.data.id === activeId) return "#fbbf24";
      if (normalizedSearch && matchesSearch(d, normalizedSearch)) return "#0ea5e9";
      return "none";
    })
    .attr("stroke-width", d => {
      if (activeId && d.data.id === activeId) return 3;
      if (normalizedSearch && matchesSearch(d, normalizedSearch)) return 2;
      return 0;
    });
}

function matchesSearch(node, searchTerm) {
  const text = [node.data.name, node.data.path, node.data.id]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return text.includes(searchTerm);
}

/**
 * Resize visualization
 */
export function resizeBubble() {
  if (!treeData) return;
  initBubble(treeData, { onSelectConcept: onSelectConceptCallback });
}

/**
 * Focus on a specific concept
 */
export function focusOnConcept(conceptId) {
  if (!svg || !root) return;

  const targetNode = root.descendants().find(d => d.data.id === conceptId);
  if (!targetNode) return;

  svg.selectAll("path")
    .filter(d => d === targetNode)
    .dispatch("click");
}

/**
 * Focus on root
 */
export function focusOnRoot() {
  if (!svg || !root) return;

  svg.select("circle").dispatch("click");
}
