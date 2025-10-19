import {
  initBubble,
  updateBubbleHighlight,
  resizeBubble,
  focusOnConcept,
  focusOnRoot,
} from "./viz.js";

const INDEX_URL = "../Pre-Processing/output/summaries/index.json";
const SUMMARIES_BASE = "../Pre-Processing/output/summaries/";

const state = {
  index: null,
  language: "ja",
  selectedPaper: null,
  selectedConceptId: null,
  conceptSearch: "",
  paperSearch: "",
  paperSearchRaw: "",
  logs: [],
  lastLoggedPaperSearch: "",
};

const conceptListEl = document.getElementById("concept-list");
const conceptSearchEl = document.getElementById("concept-search");
const clearConceptBtn = document.getElementById("clear-concept");
const paperSearchEl = document.getElementById("paper-search");
const yearListEl = document.getElementById("year-list");
const detailsEl = document.getElementById("paper-details");
const bannerEl = document.getElementById("status-banner");
const generatedAtEl = document.getElementById("generated-at");
const logListEl = document.getElementById("log-list");
const clearLogBtn = document.getElementById("clear-log");

const MAX_LOG_ENTRIES = 60;

document.querySelectorAll("input[name='language']").forEach((input) => {
  input.addEventListener("change", () => {
    state.language = input.value;
    if (state.selectedPaper) {
      renderDetails(state.selectedPaper.data);
    }
  });
});

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
    if (hadQuery) {
      logEvent("Concept search cleared");
    }
  }
});

clearConceptBtn.addEventListener("click", () => {
  applyConceptFilter(null);
});

paperSearchEl.addEventListener("input", (event) => {
  state.paperSearchRaw = event.target.value;
  state.paperSearch = state.paperSearchRaw.trim().toLowerCase();
  renderYears();
  updateBanner();
  if (state.paperSearch && state.paperSearch !== state.lastLoggedPaperSearch) {
    logEvent(`Keyword search: "${state.paperSearchRaw.trim()}"`);
    state.lastLoggedPaperSearch = state.paperSearch;
  } else if (!state.paperSearch && state.lastLoggedPaperSearch) {
    logEvent("Keyword search cleared");
    state.lastLoggedPaperSearch = "";
  }
});

paperSearchEl.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    state.paperSearch = "";
    state.paperSearchRaw = "";
    paperSearchEl.value = "";
    renderYears();
    updateBanner();
    if (state.lastLoggedPaperSearch) {
      logEvent("Keyword search cleared");
      state.lastLoggedPaperSearch = "";
    }
  }
});

clearLogBtn.addEventListener("click", () => {
  state.logs = [];
  renderLog();
});

async function loadIndex() {
  const response = await fetch(INDEX_URL);
  if (!response.ok) {
    throw new Error(`failed to fetch index: ${response.status}`);
  }
  const payload = await response.json();
  state.index = payload;
  generatedAtEl.textContent = payload.generated_at || "";
  renderConcepts();
  renderYears();
  updateBanner();
  initBubble(state.index.concept_tree, {
    onSelectConcept: handleConceptSelectedFromViz,
  });
  resizeBubble();
  refreshBubbleHighlight();
  renderLog();
}

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
    conceptListEl.innerHTML = `<li class="placeholder">データを読み込み中…</li>`;
    refreshBubbleHighlight();
    return;
  }

  const concepts = getFilteredConcepts();

  if (concepts.length === 0) {
    conceptListEl.innerHTML = `<li class="empty-note">該当する概念がありません。</li>`;
    refreshBubbleHighlight();
    return;
  }

  conceptListEl.innerHTML = "";
  concepts.forEach((concept) => {
    const item = document.createElement("li");
    item.className = "concept-item";
    if (concept.id === state.selectedConceptId) {
      item.classList.add("active");
    }
    item.tabIndex = 0;
    item.innerHTML = `
      <span class="concept-label">${concept.path || concept.id}</span>
      <span class="count">${concept.count ?? concept.papers.length}</span>
    `;
    const activate = () => {
      const isActive = state.selectedConceptId === concept.id;
      applyConceptFilter(isActive ? null : concept.id);
    };
    item.addEventListener("click", activate);
    item.addEventListener("keypress", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
    conceptListEl.appendChild(item);
  });
  refreshBubbleHighlight();
}

function renderYears() {
  if (!state.index) {
    yearListEl.innerHTML = `<p class="placeholder">データを読み込み中…</p>`;
    return;
  }

  const conceptId = state.selectedConceptId;
  const years = state.index.years || [];
  const searchTerm = state.paperSearch;

  yearListEl.innerHTML = "";
  let totalVisible = 0;

  years.forEach((yearBlock) => {
    const filteredPapers = (yearBlock.papers || []).filter((paper) => {
      if (conceptId) {
        const conceptIds = (paper.concepts || []).map((concept) => concept.id);
        if (!conceptIds.includes(conceptId)) {
          return false;
        }
      }
      if (searchTerm) {
        const normalizedFields = [
          paper.title,
          paper.title_en,
          paper.slug,
          ...(paper.authors || []),
          ...(paper.authors_en || []),
          ...(paper.concepts || []).map((concept) => concept.path || concept.id),
        ]
          .filter(Boolean)
          .map((value) => value.toString().toLowerCase());
        const matched = normalizedFields.some((field) => field.includes(searchTerm));
        if (!matched) {
          return false;
        }
      }
      return true;
    });

    if (filteredPapers.length === 0) {
      return;
    }

    totalVisible += filteredPapers.length;

    const container = document.createElement("div");
    container.className = "year-block";

    const caption = document.createElement("p");
    caption.className = "year-title";
    caption.textContent = `${yearBlock.year} (${filteredPapers.length})`;
    container.appendChild(caption);

    const list = document.createElement("ul");
    list.className = "paper-list";

    filteredPapers.forEach((paper) => {
      const entry = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = paper.title || paper.title_en || paper.slug;
      if (
        state.selectedPaper &&
        state.selectedPaper.entry.path === paper.path
      ) {
        button.classList.add("active");
      }
      button.addEventListener("click", () => {
        setActivePaper(button);
        loadPaper(yearBlock.year, paper).catch((error) => {
          console.error(error);
          detailsEl.innerHTML = `<p class="placeholder">論文の読み込みに失敗しました: ${error.message}</p>`;
        });
        logEvent(`Paper selected: ${paper.title || paper.title_en || paper.slug}`);
      });
      entry.appendChild(button);
      list.appendChild(entry);
    });

    container.appendChild(list);
    yearListEl.appendChild(container);
  });

  if (totalVisible === 0) {
    yearListEl.innerHTML = `<p class="empty-note">フィルタに一致する論文がありません。</p>`;
  }
}

function setActivePaper(activeButton) {
  yearListEl.querySelectorAll("button").forEach((btn) => {
    if (btn === activeButton) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
}

async function loadPaper(year, paperEntry) {
  const url = `${SUMMARIES_BASE}${paperEntry.path}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch paper json: ${response.status}`);
  }
  const data = await response.json();
  state.selectedPaper = { year, entry: paperEntry, data };
  renderDetails(data);
}

function renderDetails(data) {
  detailsEl.innerHTML = "";

  const title = document.createElement("h2");
  title.textContent = pickLanguage(data.title, data.title_en);

  const meta = document.createElement("div");
  meta.className = "paper-meta";
  meta.innerHTML = [
    data.year ? `Year: ${data.year}` : null,
    formatAuthors(data.authors, data.authors_en),
    formatConceptBadges(data),
  ]
    .filter(Boolean)
    .join(" | ");

  detailsEl.appendChild(title);
  if (meta.textContent) {
    detailsEl.appendChild(meta);
  }

  appendSection(detailsEl, "Abstract", data.abstract, data.abstract_en);
  appendSection(detailsEl, "Positioning", data.positioning_summary, data.positioning_summary_en);
  appendSection(detailsEl, "Purpose", data.purpose_summary, data.purpose_summary_en);
  appendSection(detailsEl, "Method", data.method_summary, data.method_summary_en);
  appendSection(detailsEl, "Evaluation", data.evaluation_summary, data.evaluation_summary_en);

  if (data.links && (data.links.pdf || data.links.code)) {
    const links = document.createElement("div");
    links.className = "links";
    if (data.links.pdf) {
      const pdf = document.createElement("a");
      pdf.href = data.links.pdf;
      pdf.textContent = "PDF";
      pdf.target = "_blank";
      pdf.rel = "noopener";
      links.appendChild(pdf);
    }
    if (data.links.code) {
      const code = document.createElement("a");
      code.href = data.links.code;
      code.textContent = "Code";
      code.target = "_blank";
      code.rel = "noopener";
      links.appendChild(code);
    }
    detailsEl.appendChild(links);
  }
}

function appendSection(root, label, valueJa, valueEn) {
  const text = pickLanguage(valueJa, valueEn);
  if (!text) return;

  const card = document.createElement("section");
  card.className = "section-card";
  const heading = document.createElement("h3");
  heading.textContent = label;
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  card.appendChild(heading);
  card.appendChild(paragraph);
  root.appendChild(card);
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

function formatAuthors(authorsJa, authorsEn) {
  const ja = Array.isArray(authorsJa) ? authorsJa.filter(Boolean) : [];
  const en = Array.isArray(authorsEn) ? authorsEn.filter(Boolean) : [];

  switch (state.language) {
    case "ja":
      return ja.length ? `Authors: ${ja.join(", ")}` : en.length ? `Authors: ${en.join(", ")}` : "";
    case "en":
      return en.length ? `Authors: ${en.join(", ")}` : ja.length ? `Authors: ${ja.join(", ")}` : "";
    case "both":
      if (ja.length && en.length) {
        return `Authors: ${ja.join(", ")} / ${en.join(", ")}`;
      }
      return ja.length
        ? `Authors: ${ja.join(", ")}`
        : en.length
        ? `Authors: ${en.join(", ")}`
        : "";
    default:
      return "";
  }
}

function formatConceptBadges(data) {
  if (!data.ccs || !Array.isArray(data.ccs.paths)) {
    return "";
  }
  const items = data.ccs.paths.map((path, index) => {
    const explanation =
      data.ccs.llm_explanations && data.ccs.llm_explanations[index]
        ? data.ccs.llm_explanations[index]
        : {};
    const confidence = explanation.confidence ? `<span class="badge">${explanation.confidence}</span>` : "";
    return `${confidence}${path}`;
  });
  return items.length ? `CCS: ${items.join(" / ")}` : "";
}

function updateBanner() {
  const parts = [];

  if (state.selectedConceptId && state.index) {
    const concept = (state.index.concepts || []).find(
      (item) => item.id === state.selectedConceptId,
    );
    if (concept) {
      parts.push(`Concept: ${concept.path || concept.id} (${concept.count ?? concept.papers.length} items)`);
    }
  }

  if (state.paperSearch) {
    parts.push(`Search: "${state.paperSearchRaw.trim()}"`);
  }

  bannerEl.textContent = parts.join(" / ");
}

function getConceptById(conceptId) {
  if (!state.index || !conceptId) return null;
  return (state.index.concepts || []).find((concept) => concept.id === conceptId) || null;
}

function applyConceptFilter(conceptId, options = {}) {
  const normalizedId = conceptId || null;
  const { message, suppressLog } = options;
  const previous = state.selectedConceptId || null;
  if (previous === normalizedId) {
    refreshBubbleHighlight();
    if (!suppressLog && message) {
      logEvent(message);
    }
    return;
  }

  state.selectedConceptId = normalizedId;
  renderConcepts();
  renderYears();
  updateBanner();
  if (normalizedId) {
    focusOnConcept(normalizedId);
  } else {
    focusOnRoot();
  }
  refreshBubbleHighlight();

  if (suppressLog) {
    return;
  }

  let logMessage = message;
  if (!logMessage) {
    if (normalizedId) {
      const concept = getConceptById(normalizedId);
      logMessage = `概念フィルタを適用: ${concept?.path || normalizedId}`;
    } else if (previous) {
      logMessage = "概念フィルタを解除";
    }
  }

  if (logMessage) {
    logEvent(logMessage);
  }
}

function refreshBubbleHighlight() {
  updateBubbleHighlight({
    activeId: state.selectedConceptId,
    searchTerm: state.conceptSearch,
  });
}

function handleConceptSelectedFromViz(concept) {
  if (!concept) return;
  const isActive = state.selectedConceptId === concept.id;
  applyConceptFilter(isActive ? null : concept.id, {
    message: isActive
      ? `概念マップで解除: ${concept.path || concept.id}`
      : `概念マップで選択: ${concept.path || concept.id}`,
    suppressLog: false,
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

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    resizeBubble();
    if (state.selectedConceptId) {
      focusOnConcept(state.selectedConceptId);
    } else {
      focusOnRoot();
    }
    refreshBubbleHighlight();
  }, 150);
});

loadIndex().catch((error) => {
  console.error(error);
  conceptListEl.innerHTML = `<li class="placeholder">Failed to load index: ${error.message}</li>`;
  yearListEl.innerHTML = `<p class="placeholder">Failed to load index.</p>`;
});
