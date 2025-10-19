import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

let containerEl;
let svg;
let rootHierarchy;
let focusNode;
let view;
let packLayout;
let circleSelection;
let labelSelection;
let onSelectConceptRef = () => {};
let nodeById = new Map();
let latestTreeData = null;
let lastHighlightState = { activeId: null, searchTerm: "" };

const MIN_DIMENSION = 480;
const TRANSITION_DURATION = 700;

export function initBubble(treeData, options = {}) {
  containerEl = document.getElementById("viz-container");
  if (!containerEl) {
    return;
  }

  containerEl.innerHTML = "";
  latestTreeData = treeData;
  onSelectConceptRef = options.onSelectConcept ?? (() => {});

  if (!treeData || !Array.isArray(treeData.children) || treeData.children.length === 0) {
    containerEl.innerHTML = `<p class="empty-note">表示できる概念がありません。</p>`;
    circleSelection = null;
    labelSelection = null;
    nodeById = new Map();
    return;
  }

  const width = containerEl.clientWidth || MIN_DIMENSION;
  const height = Math.max(MIN_DIMENSION, containerEl.clientHeight || MIN_DIMENSION);
  const diameter = Math.min(width, height);

  svg = d3
    .select(containerEl)
    .append("svg")
    .attr("class", "viz-bubble")
    .attr("viewBox", `0 0 ${diameter} ${diameter}`)
    .attr("width", "100%")
    .attr("height", diameter);

  packLayout = d3.pack().size([diameter, diameter]).padding(4);

  rootHierarchy = d3
    .hierarchy(treeData)
    .sum((d) => Math.max(1, d.count || 0))
    .sort((a, b) => (b.value || 0) - (a.value || 0));

  packLayout(rootHierarchy);
  focusNode = rootHierarchy;
  view = [focusNode.x, focusNode.y, focusNode.r * 2];

  const nodes = rootHierarchy.descendants();
  nodeById = new Map();
  nodes.forEach((node) => {
    if (node.data && node.data.id) {
      nodeById.set(node.data.id, node);
    }
  });

  const colorScale = d3.scaleLinear().domain([0, rootHierarchy.height || 1]).range(["#dbeafe", "#1d4ed8"]);

  const g = svg.append("g");

  circleSelection = g
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("class", (node) => {
      if (!node.parent) return "bubble-node bubble-root";
      if (node.children) return "bubble-node bubble-branch";
      return "bubble-node bubble-leaf";
    })
    .style("fill", (node) => (node.children ? colorScale(node.depth) : "#ffffff"))
    .on("click", (event, node) => {
      event.stopPropagation();
      if (node.children) {
        if (focusNode !== node) {
          zoom(node);
        } else if (node.parent) {
          zoom(node.parent);
        }
      } else if (node.data && node.data.id) {
        onSelectConceptRef(node.data);
      }
    });

  labelSelection = g
    .selectAll("text")
    .data(nodes)
    .join("text")
    .attr("class", "bubble-label")
    .style("text-anchor", "middle")
    .style("pointer-events", "none")
    .text((node) => node.data.name);

  svg.on("click", () => zoom(rootHierarchy));

  zoomTo([focusNode.x, focusNode.y, focusNode.r * 2]);
  updateLabelVisibility();
  applyHighlightState();
}

export function updateBubbleHighlight({ activeId, searchTerm }) {
  lastHighlightState = {
    activeId: activeId || null,
    searchTerm: searchTerm || "",
  };
  applyHighlightState();
}

export function resizeBubble() {
  if (!latestTreeData) {
    return;
  }
  initBubble(latestTreeData, { onSelectConcept: onSelectConceptRef });
  applyHighlightState();
}

export function focusOnConcept(conceptId) {
  if (!conceptId || !nodeById.has(conceptId)) {
    return;
  }
  let target = nodeById.get(conceptId);
  if (!target) return;

  // Zoom to the nearest ancestor that provides context (at least the parent branch)
  if (!target.children && target.parent) {
    target = target.parent;
  }
  zoom(target);
}

export function focusOnRoot() {
  if (rootHierarchy) {
    zoom(rootHierarchy);
  }
}

function zoom(target) {
  focusNode = target;
  const transition = svg
    .transition()
    .duration(TRANSITION_DURATION)
    .tween("zoom", () => {
      const interpolator = d3.interpolateZoom(view, [focusNode.x, focusNode.y, focusNode.r * 2]);
      return (t) => zoomTo(interpolator(t));
    });

  transition
    .selectAll(".bubble-label")
    .filter(function (node) {
      return node.parent === focusNode || node === focusNode || this.style.display === "inline";
    })
    .style("opacity", (node) => (node.parent === focusNode || node === focusNode ? 1 : 0))
    .on("start", function (node) {
      if (node.parent === focusNode || node === focusNode) {
        this.style.display = "inline";
      }
    })
    .on("end", function (node) {
      if (node.parent !== focusNode && node !== focusNode) {
        this.style.display = "none";
      }
    });

  applyHighlightState();
}

function zoomTo(viewFrame) {
  const diameter = Math.min(containerEl.clientWidth || MIN_DIMENSION, Math.max(containerEl.clientHeight || MIN_DIMENSION, MIN_DIMENSION));
  const scale = diameter / viewFrame[2];
  view = viewFrame;

  circleSelection
    .attr("transform", (node) => `translate(${(node.x - viewFrame[0]) * scale},${(node.y - viewFrame[1]) * scale})`)
    .attr("r", (node) => node.r * scale);

  labelSelection
    .attr("transform", (node) => `translate(${(node.x - viewFrame[0]) * scale},${(node.y - viewFrame[1]) * scale})`)
    .style("font-size", (node) => `${Math.max(10, (node.r * scale) / 3)}px`);
}

function updateLabelVisibility() {
  if (!labelSelection) return;
  labelSelection.each(function (node) {
    if (node.parent === focusNode || node === focusNode) {
      this.style.display = "inline";
      this.style.opacity = "1";
    } else {
      this.style.display = "none";
      this.style.opacity = "0";
    }
  });
}

function applyHighlightState() {
  if (!circleSelection) {
    return;
  }

  const { activeId, searchTerm } = lastHighlightState;
  const normalized = searchTerm ? searchTerm.toLowerCase() : "";

  circleSelection.classed("active", (node) => Boolean(activeId) && node.data.id === activeId);

  circleSelection.classed("ancestor", (node) => {
    if (!activeId || node.data.id === activeId) return false;
    return node.descendants().some((descendant) => descendant.data.id === activeId);
  });

  if (!normalized) {
    circleSelection.classed("dimmed", false);
    labelSelection.classed("dimmed", false);
    updateLabelVisibility();
    return;
  }

  circleSelection.classed("dimmed", (node) => {
    if (node.data.id === activeId) return false;
    if (activeId) {
      const isAncestor = node.descendants().some((descendant) => descendant.data.id === activeId);
      if (isAncestor) return false;
    }
    return !nodeMatchesSearch(node, normalized);
  });

  const dimmedNodes = new WeakSet();
  circleSelection.each(function (node) {
    if (d3.select(this).classed("dimmed")) {
      dimmedNodes.add(node);
    }
  });

  labelSelection.classed("dimmed", (node) => dimmedNodes.has(node));

  updateLabelVisibility();
}

function nodeMatchesSearch(node, normalized) {
  const selfTokens = [node.data.name, node.data.path, node.data.id]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  if (selfTokens.includes(normalized)) {
    return true;
  }

  return node
    .descendants()
    .some((descendant) => {
      const tokens = [descendant.data.name, descendant.data.path, descendant.data.id]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return tokens.includes(normalized);
    });
}
