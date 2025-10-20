import {
  initBubble,
  updateBubbleHighlight,
  resizeBubble,
  focusOnConcept,
  focusOnRoot,
} from "./viz.js";
import config from "./config.js";

// Utility function to sanitize HTML and prevent XSS
function escapeHtml(unsafe) {
  if (typeof unsafe !== 'string') return '';
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Utility function to safely set text content
function safeSetText(element, text) {
  if (!element) return;
  element.textContent = text || '';
}

// Enhanced fetch with timeout and better error handling
async function fetchWithTimeout(url, options = {}) {
  const timeout = options.timeout || 10000;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  try {
    const cacheStrategy = config.cache.enabled && config.cache.strategy === 'cache-first'
      ? 'force-cache'
      : 'no-cache';

    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      cache: cacheStrategy,
    });
    clearTimeout(id);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response;
  } catch (error) {
    clearTimeout(id);
    if (error.name === 'AbortError') {
      throw new Error('Request timeout - please check your connection');
    }
    throw error;
  }
}

const state = {
  index: null,
  language: "ja",
  selectedConceptId: null,
  selectedPaper: null,
  conceptSearch: "",
  paperSearch: "",
  paperSearchRaw: "",
  logs: [],
  lastLoggedPaperSearch: "",
  sidebarCollapsed: false,
  conceptMap: new Map(),
  paperDirectory: [],
  paperByPath: new Map(),
  paperCache: new Map(),
};

const conceptListEl = document.getElementById("concept-list");
const conceptSearchEl = document.getElementById("concept-search");
const clearConceptBtn = document.getElementById("clear-concept");
const paperSearchEl = document.getElementById("paper-search");
const bannerEl = document.getElementById("status-banner");
const generatedAtEl = document.getElementById("generated-at");
const logListEl = document.getElementById("log-list");
const clearLogBtn = document.getElementById("clear-log");
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle");
const detailPanel = document.getElementById("detail-panel");
const detailContent = document.getElementById("detail-content");
const detailClose = document.getElementById("detail-close");
const appShell = document.querySelector(".app-shell");

const MAX_LOG_ENTRIES = config.ui.maxLogEntries;

function setSidebarCollapsed(collapsed) {
  state.sidebarCollapsed = collapsed;
  if (sidebar) {
    sidebar.classList.toggle("collapsed", collapsed);
  }
  if (appShell) {
    appShell.classList.toggle("sidebar-hidden", collapsed);
  }
  if (sidebarToggle) {
    sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
  }
}

setSidebarCollapsed(state.sidebarCollapsed);

// -----------------------------------------------------------------------------
// Event wiring
// -----------------------------------------------------------------------------

document.querySelectorAll("input[name='language']").forEach((input) => {
  input.addEventListener("change", () => {
    state.language = input.value;
    if (state.selectedPaper) {
      renderPaperDetail(state.selectedPaper.data, state.selectedPaper.meta);
    } else if (state.selectedConceptId) {
      showConceptDetail(state.selectedConceptId);
    }
  });
});

if (sidebarToggle) {
  sidebarToggle.addEventListener("click", () => {
    setSidebarCollapsed(!state.sidebarCollapsed);
  });
}

if (detailClose) {
  detailClose.addEventListener("click", () => {
    closeDetailPanel();
  });
}

if (conceptSearchEl) {
  conceptSearchEl.addEventListener("input", (event) => {
    state.conceptSearch = event.target.value.trim().toLowerCase();
    renderConcepts();
  });

  conceptSearchEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      const concepts = getFilteredConcepts();
      if (concepts.length > 0) {
        const chosen = concepts[0];
        applyConceptFilter(chosen.id, {
          message: `Concept search selected: ${chosen.path || chosen.id}`,
        });
      }
    } else if (event.key === "Escape") {
      const hadQuery = Boolean(state.conceptSearch);
      state.conceptSearch = "";
      conceptSearchEl.value = "";
      renderConcepts();
      refreshBubbleHighlight();
      if (hadQuery) {
        logEvent("Concept search cleared");
      }
    }
  });
}

if (clearConceptBtn) {
  clearConceptBtn.addEventListener("click", () => {
    applyConceptFilter(null, { message: "Concept filter cleared" });
  });
}

if (paperSearchEl) {
  paperSearchEl.addEventListener("input", (event) => {
    state.paperSearchRaw = event.target.value;
    state.paperSearch = state.paperSearchRaw.trim().toLowerCase();
    if (state.paperSearch && state.paperSearch !== state.lastLoggedPaperSearch) {
      logEvent(`Keyword search: "${state.paperSearchRaw.trim()}"`);
      state.lastLoggedPaperSearch = state.paperSearch;
    } else if (!state.paperSearch && state.lastLoggedPaperSearch) {
      logEvent("Keyword search cleared");
      state.lastLoggedPaperSearch = "";
    }
    handleSearchChange();
  });

  paperSearchEl.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      const hadValue = Boolean(state.paperSearch);
      state.paperSearch = "";
      state.paperSearchRaw = "";
      paperSearchEl.value = "";
      handleSearchChange();
      if (hadValue) {
        logEvent("Keyword search cleared");
        state.lastLoggedPaperSearch = "";
      }
    }
  });
}

if (clearLogBtn) {
  clearLogBtn.addEventListener("click", () => {
    state.logs = [];
    renderLog();
  });
}

// -----------------------------------------------------------------------------
// Data loading
// -----------------------------------------------------------------------------

async function loadIndex() {
  try {
    showLoadingState('Loading concepts...');
    const response = await fetchWithTimeout(config.indexUrl, { timeout: 15000 });
    const payload = await response.json();
    state.index = payload;
    safeSetText(generatedAtEl, payload.generated_at || '');
    hideLoadingState();

    state.conceptMap = new Map();
    (payload.concepts || []).forEach((concept) => {
      state.conceptMap.set(concept.id, concept);
    });

    state.paperDirectory = [];
    state.paperByPath = new Map();
    (payload.years || []).forEach((block) => {
      (block.papers || []).forEach((paper) => {
        const meta = { ...paper, year: block.year };
        state.paperDirectory.push(meta);
        state.paperByPath.set(paper.path, meta);
      });
    });

    if (window.innerWidth < config.ui.sidebarCollapseBreakpoint) {
      setSidebarCollapsed(true);
    } else {
      setSidebarCollapsed(false);
    }

    renderConcepts();
    updateBanner();
    initBubble(payload.concept_tree, {
      onSelectConcept: handleConceptSelectedFromViz,
    });
    focusOnRoot();
    refreshBubbleHighlight();
    renderLog();
  } catch (error) {
    console.error('Failed to load index:', error);
    showError(`Failed to load data: ${error.message}`, true);
    throw error;
  }
}

// UI helper functions
function showLoadingState(message) {
  if (bannerEl) {
    bannerEl.textContent = message;
    bannerEl.style.color = 'var(--accent)';
  }
}

function hideLoadingState() {
  if (bannerEl) {
    bannerEl.textContent = '';
  }
}

function showError(message, isPersistent = false) {
  if (bannerEl) {
    bannerEl.textContent = message;
    bannerEl.style.color = '#ef4444';
    if (!isPersistent) {
      setTimeout(() => {
        bannerEl.textContent = '';
        bannerEl.style.color = '';
      }, 5000);
    }
  }
  logEvent(`Error: ${message}`);
}

// -----------------------------------------------------------------------------
// Rendering helpers
// -----------------------------------------------------------------------------

function getFilteredConcepts() {
  if (!state.index) return [];
  const concepts = state.index.concepts || [];
  if (!state.conceptSearch) {
    return concepts;
  }
  return concepts.filter((concept) => {
    const target = `${concept.path || ""} ${concept.id || ""}`.toLowerCase();
    return target.includes(state.conceptSearch);
  });
}

function renderConcepts() {
  if (!state.index) {
    conceptListEl.innerHTML = `<li class="placeholder">Loading concepts...</li>`;
    return;
  }

  const concepts = getFilteredConcepts();
  if (concepts.length === 0) {
    conceptListEl.innerHTML = `<li class="empty-note">No matching concepts.</li>`;
    return;
  }

  conceptListEl.innerHTML = "";
  concepts.forEach((concept) => {
    const li = document.createElement("li");
    li.className = "concept-item";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = concept.path || concept.id;
    if (concept.id === state.selectedConceptId) {
      button.classList.add("active");
    }
    button.addEventListener("click", () => {
      const isActive = state.selectedConceptId === concept.id;
      applyConceptFilter(isActive ? null : concept.id);
    });
    li.appendChild(button);
    conceptListEl.appendChild(li);
  });
  refreshBubbleHighlight();
}

function applyConceptFilter(conceptId, options = {}) {
  const normalizedId = conceptId || null;
  const previous = state.selectedConceptId || null;
  if (normalizedId === previous) {
    refreshBubbleHighlight();
    if (options.message && !options.suppressLog) {
      logEvent(options.message);
    }
    return;
  }

  if (normalizedId && state.paperSearch) {
    state.paperSearch = "";
    state.paperSearchRaw = "";
    if (paperSearchEl) {
      paperSearchEl.value = "";
    }
  }

  state.selectedConceptId = normalizedId;
  renderConcepts();
  updateBanner();

  if (normalizedId) {
    focusOnConcept(normalizedId);
    showConceptDetail(normalizedId);
  } else {
    focusOnRoot();
    handleSearchChange();
  }

  refreshBubbleHighlight();

  if (options.suppressLog) {
    return;
  }

  let message = options.message;
  if (!message) {
    if (normalizedId) {
      const concept = state.conceptMap.get(normalizedId);
      message = `Concept filter applied: ${concept?.path || normalizedId}`;
    } else if (previous) {
      message = "Concept filter cleared";
    }
  }
  if (message) {
    logEvent(message);
  }
}

function handleConceptSelectedFromViz(concept) {
  if (!concept) return;
  const isActive = state.selectedConceptId === concept.id;
  applyConceptFilter(isActive ? null : concept.id, {
    message: isActive
      ? `Concept bubble cleared: ${concept.path || concept.id}`
      : `Concept bubble selected: ${concept.path || concept.id}`,
  });
}

function handleSearchChange() {
  if (state.paperSearch) {
    renderSearchResults(state.paperSearch);
  } else if (state.selectedConceptId) {
    showConceptDetail(state.selectedConceptId);
  } else {
    closeDetailPanel();
  }
  updateBanner();
  refreshBubbleHighlight();
}

function showConceptDetail(conceptId) {
  const concept = state.conceptMap.get(conceptId);
  state.selectedPaper = null;
  if (!concept) {
    closeDetailPanel();
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "detail-section";

  const header = document.createElement("header");
  header.className = "detail-header";
  const title = document.createElement("h2");
  title.textContent = concept.path || concept.id;
  const meta = document.createElement("p");
  const count = concept.count ?? concept.papers?.length ?? 0;
  meta.textContent = `${count} papers`;
  header.appendChild(title);
  header.appendChild(meta);
  wrapper.appendChild(header);

  const list = document.createElement("div");
  list.className = "concept-paper-list";
  const papers = concept.papers || [];
  if (papers.length === 0) {
    const empty = document.createElement("p");
    empty.className = "placeholder";
    empty.textContent = "No papers attached to this concept.";
    list.appendChild(empty);
  } else {
    papers
      .slice()
      .sort((a, b) => {
        const metaA = state.paperByPath.get(a.path) || a;
        const metaB = state.paperByPath.get(b.path) || b;
        return (metaB.year || 0) - (metaA.year || 0);
      })
      .slice(0, 50)
      .forEach((paper) => {
        const metaInfo = state.paperByPath.get(paper.path) || paper;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "concept-paper";

        const titleSpan = document.createElement("span");
        titleSpan.className = "paper-title";
        titleSpan.textContent = paper.title || paper.title_en || paper.slug;

        const metaSpan = document.createElement("span");
        metaSpan.className = "paper-meta";
        metaSpan.textContent = metaInfo.year ?? "";

        button.appendChild(titleSpan);
        button.appendChild(metaSpan);

        button.addEventListener("click", () => {
          logEvent(`Paper selected: ${paper.title || paper.title_en || paper.slug}`);
          loadPaper(metaInfo);
        });
        list.appendChild(button);
      });
  }
  wrapper.appendChild(list);

  detailContent.innerHTML = "";
  detailContent.appendChild(wrapper);
  openDetailPanel();
}

function renderSearchResults(keyword) {
  state.selectedPaper = null;
  const term = keyword.toLowerCase();
  const matches = state.paperDirectory
    .filter((paper) => {
      const fields = [
        paper.title,
        paper.title_en,
        paper.slug,
        paper.year,
        ...(paper.authors || []),
        ...(paper.authors_en || []),
      ]
        .filter(Boolean)
        .map((value) => value.toString().toLowerCase());
      return fields.some((field) => field.includes(term));
    })
    .slice(0, 40);

  const wrapper = document.createElement("div");
  wrapper.className = "detail-section";

  const header = document.createElement("header");
  header.className = "detail-header";
  const title = document.createElement("h2");
  title.textContent = `Search results (${matches.length})`;
  header.appendChild(title);
  wrapper.appendChild(header);

  const list = document.createElement("div");
  list.className = "concept-paper-list";
  if (matches.length === 0) {
    const empty = document.createElement("p");
    empty.className = "placeholder";
    empty.textContent = "No papers matched your query.";
    list.appendChild(empty);
  } else {
    matches.forEach((paper) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "concept-paper";

      const titleSpan = document.createElement("span");
      titleSpan.className = "paper-title";
      titleSpan.textContent = paper.title || paper.title_en || paper.slug;

      const metaSpan = document.createElement("span");
      metaSpan.className = "paper-meta";
      metaSpan.textContent = paper.year ?? "";

      button.appendChild(titleSpan);
      button.appendChild(metaSpan);

      button.addEventListener("click", () => {
        logEvent(`Paper selected: ${paper.title || paper.title_en || paper.slug}`);
        loadPaper(paper);
      });
      list.appendChild(button);
    });
  }
  wrapper.appendChild(list);

  detailContent.innerHTML = "";
  detailContent.appendChild(wrapper);
  openDetailPanel();
}

async function loadPaper(meta) {
  if (!meta || !meta.path) {
    return;
  }
  if (state.paperCache.has(meta.path)) {
    const cached = state.paperCache.get(meta.path);
    state.selectedPaper = { data: cached, meta };
    renderPaperDetail(cached, meta);
    return;
  }

  try {
    showLoadingState('Loading paper details...');
    const response = await fetchWithTimeout(`${config.dataBasePath}${meta.path}`, { timeout: 10000 });
    const data = await response.json();
    state.paperCache.set(meta.path, data);
    state.selectedPaper = { data, meta };
    renderPaperDetail(data, meta);
    hideLoadingState();
  } catch (error) {
    console.error('Failed to load paper:', error);
    showError(`Failed to load paper: ${error.message}`);
    hideLoadingState();
  }
}

function renderPaperDetail(data, meta) {
  if (!data) return;

  const wrapper = document.createElement("div");
  wrapper.className = "detail-section";

  const header = document.createElement("header");
  header.className = "detail-header";
  const title = document.createElement("h2");
  title.textContent = pickLanguage(data.title, data.title_en);
  const subtitle = document.createElement("p");
  subtitle.className = "paper-meta";
  subtitle.textContent = [
    meta.year ?? data.year ?? "",
    formatAuthors(data.authors, data.authors_en),
  ]
    .filter(Boolean)
    .join(" · ");
  header.appendChild(title);
  if (subtitle.textContent) {
    header.appendChild(subtitle);
  }
  wrapper.appendChild(header);

  appendSection(wrapper, "Abstract", data.abstract, data.abstract_en);
  appendSection(wrapper, "Positioning", data.positioning_summary, data.positioning_summary_en);
  appendSection(wrapper, "Purpose", data.purpose_summary, data.purpose_summary_en);
  appendSection(wrapper, "Method", data.method_summary, data.method_summary_en);
  appendSection(wrapper, "Evaluation", data.evaluation_summary, data.evaluation_summary_en);

  if (data.links && (data.links.pdf || data.links.code)) {
    const links = document.createElement("div");
    links.className = "links";
    if (data.links.pdf) {
      const pdfLink = document.createElement("a");
      pdfLink.href = data.links.pdf;
      pdfLink.textContent = "PDF";
      pdfLink.target = "_blank";
      pdfLink.rel = "noopener";
      links.appendChild(pdfLink);
    }
    if (data.links.code) {
      const codeLink = document.createElement("a");
      codeLink.href = data.links.code;
      codeLink.textContent = "Code";
      codeLink.target = "_blank";
      codeLink.rel = "noopener";
      links.appendChild(codeLink);
    }
    wrapper.appendChild(links);
  }

  detailContent.innerHTML = "";
  detailContent.appendChild(wrapper);
  openDetailPanel();
}

function appendSection(parent, label, valueJa, valueEn) {
  const text = pickLanguage(valueJa, valueEn);
  if (!text) return;

  const section = document.createElement("section");
  section.className = "detail-subsection";
  const heading = document.createElement("h3");
  heading.textContent = label;
  const body = document.createElement("p");
  body.textContent = text;
  section.appendChild(heading);
  section.appendChild(body);
  parent.appendChild(section);
}

function openDetailPanel() {
  detailPanel.classList.add("open");
  if (appShell) {
    appShell.classList.add("detail-visible");
  }
  // Move focus to detail panel for accessibility
  if (detailContent) {
    detailContent.focus();
  }
}

function closeDetailPanel() {
  detailPanel.classList.remove("open");
  state.selectedPaper = null;
  if (appShell) {
    appShell.classList.remove("detail-visible");
  }
  if (!state.paperSearch && !state.selectedConceptId) {
    detailContent.innerHTML = '<p class="placeholder">Select a concept or paper to inspect details.</p>';
  }
  // Return focus to the main area
  const vizContainer = document.getElementById('viz-container');
  if (vizContainer) {
    vizContainer.focus();
  }
}

function updateBanner() {
  const parts = [];
  if (state.selectedConceptId) {
    const concept = state.conceptMap.get(state.selectedConceptId);
    if (concept) {
      const count = concept.count ?? concept.papers?.length ?? 0;
      parts.push(`Concept: ${concept.path || concept.id} (${count} items)`);
    }
  }
  if (state.paperSearch) {
    parts.push(`Search: "${state.paperSearchRaw.trim()}"`);
  }
  bannerEl.textContent = parts.join(" / ");
}

function refreshBubbleHighlight() {
  updateBubbleHighlight({
    activeId: state.selectedConceptId,
    searchTerm: state.conceptSearch,
  });
}

function logEvent(message) {
  const timestamp = new Date();
  state.logs.unshift({ message, time: timestamp });
  if (state.logs.length > MAX_LOG_ENTRIES) {
    state.logs.length = MAX_LOG_ENTRIES;
  }
  renderLog();
}

function renderLog() {
  if (!state.logs.length) {
    logListEl.innerHTML = `<li class="empty-note">No interactions yet.</li>`;
    return;
  }

  logListEl.innerHTML = "";
  state.logs.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "log-entry";
    const timeEl = document.createElement("time");
    timeEl.textContent = entry.time.toLocaleTimeString("ja-JP", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    const messageEl = document.createElement("span");
    messageEl.textContent = entry.message;
    li.appendChild(timeEl);
    li.appendChild(messageEl);
    logListEl.appendChild(li);
  });
}

function formatAuthors(authorsJa, authorsEn) {
  const ja = Array.isArray(authorsJa) ? authorsJa.filter(Boolean) : [];
  const en = Array.isArray(authorsEn) ? authorsEn.filter(Boolean) : [];
  switch (state.language) {
    case "ja":
      return ja.length ? ja.join(", ") : en.join(", ");
    case "en":
      return en.length ? en.join(", ") : ja.join(", ");
    case "both":
      if (ja.length && en.length) {
        return `${ja.join(", ")} / ${en.join(", ")}`;
      }
      return ja.join(", ") || en.join(", ");
    default:
      return ja.join(", ") || en.join(", ");
  }
}

function pickLanguage(valueJa, valueEn) {
  switch (state.language) {
    case "ja":
      return valueJa || valueEn || "";
    case "en":
      return valueEn || valueJa || "";
    case "both":
      if (valueJa && valueEn && valueJa !== valueEn) {
        return `${valueJa}\n${valueEn}`;
      }
      return valueJa || valueEn || "";
    default:
      return valueJa || valueEn || "";
  }
}

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (window.innerWidth < config.ui.sidebarCollapseBreakpoint && !state.sidebarCollapsed) {
      setSidebarCollapsed(true);
    }
    resizeBubble();
    if (state.selectedConceptId) {
      focusOnConcept(state.selectedConceptId);
    } else {
      focusOnRoot();
    }
    refreshBubbleHighlight();
  }, config.performance.debounceDelay);
});

loadIndex().catch((error) => {
  console.error(error);
  conceptListEl.innerHTML = `<li class="placeholder">Failed to load index: ${error.message}</li>`;
  detailContent.innerHTML = `<p class="placeholder">Failed to load data.</p>`;
});
