// Static asset base for processed summaries
const SUMMARIES_BASE_URL = new URL('../Pre-Processing/output/summaries/', import.meta.url);

// State Management
const state = {
    data: null,
    currentView: 'network',
    selectedConcept: null,
    selectedPaper: null,
    searchTerm: '',
    yearFilter: '',
    conceptSearchTerm: '',
    embeddingSection: 'overview',
    filteredPapers: [],
    conceptMap: new Map(),
    paperMap: new Map(),
    paperEmbeddingsCache: new Map(),
    similarityThreshold: 0.7
};

// Initialize
async function init() {
    try {
        showLoading();
        await loadData();
        setupEventListeners();
        renderConceptTree();
        updateHeader();
        renderVisualization();
        hideLoading();
    } catch (error) {
        console.error('Initialization failed:', error);
        showError('データの読み込みに失敗しました');
    }
}

// Data Loading
async function loadData() {
    // Try enhanced index first, fall back to regular index
    let response = await fetch(new URL('index_enhanced.json', SUMMARIES_BASE_URL));
    if (!response.ok) {
        console.warn('Enhanced index not found, using regular index');
        response = await fetch(new URL('index.json', SUMMARIES_BASE_URL));
        if (!response.ok) throw new Error('Failed to load data');
    } else {
        console.log('Using enhanced index with embeddings');
    }

    state.data = await response.json();

    // Build concept map
    if (state.data.concepts) {
        state.data.concepts.forEach(concept => {
            state.conceptMap.set(concept.id, concept);
        });
    }

    // Build paper map and collect all papers
    state.filteredPapers = [];
    if (state.data.years) {
        state.data.years.forEach(yearBlock => {
            if (yearBlock.papers) {
                yearBlock.papers.forEach(paper => {
                    const paperWithYear = { ...paper, year: yearBlock.year };
                    state.paperMap.set(paper.slug, paperWithYear);
                    state.filteredPapers.push(paperWithYear);
                });
            }
        });
    }
}

// Event Listeners
function setupEventListeners() {
    // View mode buttons
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            state.currentView = e.target.dataset.view;
            renderVisualization();
        });
    });

    // Concept search
    const conceptSearch = document.getElementById('concept-search');
    conceptSearch.addEventListener('input', (e) => {
        state.conceptSearchTerm = e.target.value.toLowerCase();
        renderConceptTree();
    });

    // Paper search
    const paperSearch = document.getElementById('paper-search');
    paperSearch.addEventListener('input', (e) => {
        state.searchTerm = e.target.value.toLowerCase();
        filterPapers();
        renderConceptTree(); // Update concept tree based on filtered papers
        renderVisualization();
    });

    // Year filter
    const yearFilter = document.getElementById('year-filter');
    if (state.data.years) {
        state.data.years.forEach(yearBlock => {
            const option = document.createElement('option');
            option.value = yearBlock.year;
            option.textContent = yearBlock.year;
            yearFilter.appendChild(option);
        });
    }

    yearFilter.addEventListener('change', (e) => {
        state.yearFilter = e.target.value;
        filterPapers();
        renderConceptTree(); // Update concept tree based on filtered papers
        renderVisualization();
    });

    // Embedding section selector
    const embeddingSection = document.getElementById('embedding-section');
    embeddingSection.addEventListener('change', (e) => {
        state.embeddingSection = e.target.value;
        renderVisualization();
        updateSelectionInfo();
    });

    // Similarity threshold slider
    const similarityThreshold = document.getElementById('similarity-threshold');
    const similarityValue = document.getElementById('similarity-value');
    if (similarityThreshold && similarityValue) {
        similarityThreshold.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            state.similarityThreshold = value / 100;
            similarityValue.textContent = `${value}%`;
        });
    }

    // Close detail button
    document.getElementById('close-detail').addEventListener('click', closeDetail);
}

// Filter papers based on search and year
function filterPapers() {
    let papers = [];

    if (state.data.years) {
        state.data.years.forEach(yearBlock => {
            if (state.yearFilter && yearBlock.year !== parseInt(state.yearFilter)) {
                return;
            }

            if (yearBlock.papers) {
                yearBlock.papers.forEach(paper => {
                    papers.push({ ...paper, year: yearBlock.year });
                });
            }
        });
    }

    // Apply search filter
    if (state.searchTerm) {
        papers = papers.filter(paper => {
            const searchText = [
                paper.title,
                paper.title_en,
                paper.slug,
                ...(paper.authors || []),
                ...(paper.authors_en || [])
            ].join(' ').toLowerCase();

            return searchText.includes(state.searchTerm);
        });
    }

    // Apply concept filter
    if (state.selectedConcept) {
        papers = papers.filter(paper => {
            return paper.concepts && paper.concepts.some(c =>
                c.id === state.selectedConcept || c.id.startsWith(state.selectedConcept + '.')
            );
        });
    }

    state.filteredPapers = papers;
    updateSelectionInfo();
}

// Render Concept Tree
function renderConceptTree() {
    const container = document.getElementById('concept-tree');
    container.innerHTML = '';

    if (!state.data.concept_tree) return;

    // Build filtered tree based on current papers
    const filteredTree = buildFilteredConceptTree();
    renderConceptNode(filteredTree, container, 0);
}

function buildFilteredConceptTree() {
    if (!state.data.concept_tree) return null;

    // Get relevant concept IDs from filtered papers
    const relevantConceptIds = new Set();
    state.filteredPapers.forEach(paper => {
        if (paper.concepts) {
            paper.concepts.forEach(c => {
                relevantConceptIds.add(c.id);
                // Also add parent IDs
                const parts = c.id.split('.');
                for (let i = 1; i < parts.length; i++) {
                    relevantConceptIds.add(parts.slice(0, i).join('.'));
                }
            });
        }
    });

    // Filter tree recursively
    function filterNode(node) {
        if (!node) return null;

        // Check if this node or any descendant is relevant
        const isRelevant = node.id && relevantConceptIds.has(node.id);
        const hasRelevantChildren = node.children && node.children.length > 0;

        if (!isRelevant && !hasRelevantChildren && node.depth > 0) {
            return null;
        }

        const filteredChildren = [];
        if (node.children) {
            node.children.forEach(child => {
                const filtered = filterNode(child);
                if (filtered) {
                    filteredChildren.push(filtered);
                }
            });
        }

        // Skip nodes with no papers and only one child (collapse single chains)
        if (filteredChildren.length === 1 && (!node.count || node.count === 0) && node.depth > 0) {
            return filteredChildren[0];
        }

        return {
            ...node,
            children: filteredChildren.length > 0 ? filteredChildren : undefined
        };
    }

    return filterNode(state.data.concept_tree);
}

function renderConceptNode(node, container, depth) {
    if (!node) return;

    const nodeDiv = document.createElement('div');
    nodeDiv.className = 'concept-node';
    nodeDiv.style.marginLeft = `${depth * 0.5}rem`;

    // Check if node matches search
    const matchesSearch = !state.conceptSearchTerm ||
        (node.name && node.name.toLowerCase().includes(state.conceptSearchTerm)) ||
        (node.path && node.path.toLowerCase().includes(state.conceptSearchTerm));

    if (!matchesSearch && depth > 0) {
        // Check if any children match
        const hasMatchingChild = node.children &&
            hasMatchingDescendant(node, state.conceptSearchTerm);
        if (!hasMatchingChild) return;
    }

    const hasChildren = node.children && node.children.length > 0;

    // Only show nodes that have papers or are top-level
    const hasPapers = (node.count && node.count > 0) || (node.papers && node.papers.length > 0);
    if (!hasPapers && !hasChildren && depth > 1) {
        return;
    }

    const itemDiv = document.createElement('div');
    itemDiv.className = 'concept-item';
    if (state.selectedConcept === node.id) {
        itemDiv.classList.add('selected');
    }

    if (hasChildren) {
        const toggle = document.createElement('button');
        toggle.className = 'concept-toggle';
        toggle.textContent = '▶';
        toggle.dataset.expanded = 'false';
        itemDiv.appendChild(toggle);
    } else {
        const spacer = document.createElement('span');
        spacer.style.width = '1.5rem';
        spacer.style.display = 'inline-block';
        itemDiv.appendChild(spacer);
    }

    const label = document.createElement('span');
    label.className = 'concept-label';
    // Simplify label by showing only the last part of the path
    const simpleName = node.name ? node.name.split(' → ').pop() : 'Unknown';
    label.textContent = simpleName;
    label.title = node.name || node.path; // Full name on hover
    itemDiv.appendChild(label);

    if (hasPapers) {
        const count = document.createElement('span');
        count.className = 'concept-count';
        count.textContent = node.count || node.papers.length;
        itemDiv.appendChild(count);
    }

    itemDiv.addEventListener('click', (e) => {
        if (e.target.classList.contains('concept-toggle')) {
            e.stopPropagation();
            toggleConceptChildren(e.target);
            return;
        }

        if (node.id) {
            selectConcept(node.id);
        }
    });

    nodeDiv.appendChild(itemDiv);

    if (hasChildren) {
        const childrenDiv = document.createElement('div');
        childrenDiv.className = 'concept-children collapsed';
        node.children.forEach(child => {
            renderConceptNode(child, childrenDiv, depth + 1);
        });
        nodeDiv.appendChild(childrenDiv);
    }

    container.appendChild(nodeDiv);
}

function hasMatchingDescendant(node, searchTerm) {
    if (!searchTerm) return true;

    if ((node.name && node.name.toLowerCase().includes(searchTerm)) ||
        (node.path && node.path.toLowerCase().includes(searchTerm))) {
        return true;
    }

    if (node.children) {
        return node.children.some(child => hasMatchingDescendant(child, searchTerm));
    }

    return false;
}

function toggleConceptChildren(toggleBtn) {
    const children = toggleBtn.closest('.concept-node').querySelector('.concept-children');
    if (!children) return;

    const isExpanded = toggleBtn.dataset.expanded === 'true';
    toggleBtn.dataset.expanded = !isExpanded;
    toggleBtn.textContent = isExpanded ? '▶' : '▼';
    children.classList.toggle('collapsed', isExpanded);
}

function selectConcept(conceptId) {
    if (state.selectedConcept === conceptId) {
        state.selectedConcept = null;
    } else {
        state.selectedConcept = conceptId;
    }

    state.selectedPaper = null;
    filterPapers();
    renderConceptTree();
    renderVisualization();

    if (state.selectedConcept) {
        showConceptDetail(conceptId);
    } else {
        closeDetail();
    }
}

// Visualization
function renderVisualization() {
    const container = document.getElementById('visualization');

    try {
        container.innerHTML = '';

        switch (state.currentView) {
            case 'network':
                renderNetworkView(container);
                break;
            case 'timeline':
                renderTimelineView(container);
                break;
            case 'sunburst':
                renderSunburstView(container);
                break;
        }

        renderLegend();
    } catch (error) {
        console.error('Visualization error:', error);
        container.innerHTML = `<div class="empty-state"><p>エラー: ${error.message}</p></div>`;
    }
}

function renderLegend() {
    const legendContainer = document.getElementById('viz-legend');
    if (!legendContainer) return;

    const conceptColors = getConceptColorMap();
    const labels = new Map([
        ['10010147', 'Computing methodologies'],
        ['10003120', 'HCI'],
        ['10002951', 'Information systems'],
        ['10003033', 'Networks'],
        ['10002978', 'Security'],
        ['10011007', 'Software engineering'],
        ['10003752', 'Theory'],
        ['10010198', 'Hardware'],
        ['10010520', 'Computer systems'],
        ['10003456', 'Applied computing'],
    ]);

    let html = '';
    labels.forEach((label, id) => {
        const color = conceptColors.get(id);
        if (color) {
            html += `
                <div class="legend-item">
                    <div class="legend-color" style="background: ${color}"></div>
                    <span>${label}</span>
                </div>
            `;
        }
    });

    legendContainer.innerHTML = html;
}

// Network Visualization
function renderNetworkView(container) {
    if (!state.filteredPapers.length) {
        container.innerHTML = '<div class="empty-state"><p>表示する論文がありません</p></div>';
        return;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;

    console.log('Network view:', { width, height, papers: state.filteredPapers.length });

    // Clear container first
    container.innerHTML = '';

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('id', 'network-viz')
        .style('background', 'transparent');

    // Add zoom behavior
    const g = svg.append('g');

    const zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);

    // Setup zoom control buttons
    setupZoomControls(svg, zoom);


    // Check if we have embedding data
    const hasEmbeddings = state.filteredPapers.some(p => p.embedding_2d);
    const useEmbeddings = hasEmbeddings && state.filteredPapers.length <= 500;

    // Calculate embedding bounds for better scaling
    let embeddingBounds = { minX: 0, maxX: 0, minY: 0, maxY: 0 };
    if (useEmbeddings) {
        const coords = [];
        state.filteredPapers.forEach(paper => {
            if (paper.embedding_2d) {
                const sectionData = paper.embedding_2d[state.embeddingSection];
                if (sectionData && sectionData.tsne) {
                    coords.push({ x: sectionData.tsne[0], y: sectionData.tsne[1] });
                }
            }
        });

        if (coords.length > 0) {
            embeddingBounds.minX = Math.min(...coords.map(c => c.x));
            embeddingBounds.maxX = Math.max(...coords.map(c => c.x));
            embeddingBounds.minY = Math.min(...coords.map(c => c.y));
            embeddingBounds.maxY = Math.max(...coords.map(c => c.y));
        }
    }

    // Create nodes from papers
    const nodes = state.filteredPapers.map((paper, i) => {
        const node = {
            id: paper.slug,
            paper: paper
        };

        // Use embedding coordinates if available
        if (useEmbeddings && paper.embedding_2d) {
            const sectionData = paper.embedding_2d[state.embeddingSection];
            if (sectionData && sectionData.tsne) {
                // Normalize coordinates to fit in view with padding
                const padding = 100;
                const rangeX = embeddingBounds.maxX - embeddingBounds.minX;
                const rangeY = embeddingBounds.maxY - embeddingBounds.minY;

                node.x = ((sectionData.tsne[0] - embeddingBounds.minX) / rangeX) * (width - padding * 2) + padding;
                node.y = ((sectionData.tsne[1] - embeddingBounds.minY) / rangeY) * (height - padding * 2) + padding;
            } else {
                node.x = width / 2;
                node.y = height / 2;
            }
        } else {
            node.x = width / 2;
            node.y = height / 2;
        }

        return node;
    });

    console.log('Nodes sample:', nodes.slice(0, 3));
    console.log('Using embeddings:', useEmbeddings);

    // Create links based on shared concepts
    const links = [];
    const maxLinks = 2000; // Limit for performance
    let linkCount = 0;

    for (let i = 0; i < nodes.length && linkCount < maxLinks; i++) {
        for (let j = i + 1; j < nodes.length && linkCount < maxLinks; j++) {
            const paper1 = nodes[i].paper;
            const paper2 = nodes[j].paper;

            const sharedConcepts = getSharedConcepts(paper1, paper2);
            if (sharedConcepts > 0) {
                links.push({
                    source: nodes[i].id,
                    target: nodes[j].id,
                    strength: sharedConcepts
                });
                linkCount++;
            }
        }
    }

    // Get primary concepts for color coding
    const conceptColors = getConceptColorMap();

    function getPaperColor(paper) {
        if (paper.concepts && paper.concepts.length > 0) {
            const primaryConcept = paper.concepts[0];
            const topLevel = primaryConcept.id.split('.')[0];
            return conceptColors.get(topLevel) || '#6366f1';
        }
        return '#6366f1';
    }

    // Create force simulation (only if not using embeddings)
    let simulation = null;
    if (!useEmbeddings) {
        // Dynamic force layout
        simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(100).strength(d => d.strength * 0.1))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(20));
    }

    // Draw links
    const link = g.append('g')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('class', 'link')
        .attr('x1', d => {
            const source = typeof d.source === 'object' ? d.source : nodes.find(n => n.id === d.source);
            return source ? source.x : width / 2;
        })
        .attr('y1', d => {
            const source = typeof d.source === 'object' ? d.source : nodes.find(n => n.id === d.source);
            return source ? source.y : height / 2;
        })
        .attr('x2', d => {
            const target = typeof d.target === 'object' ? d.target : nodes.find(n => n.id === d.target);
            return target ? target.x : width / 2;
        })
        .attr('y2', d => {
            const target = typeof d.target === 'object' ? d.target : nodes.find(n => n.id === d.target);
            return target ? target.y : height / 2;
        })
        .style('stroke-width', d => Math.sqrt(d.strength));

    console.log('Links drawn:', links.length);

    // Draw nodes
    const node = g.append('g')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('class', 'node')
        .attr('r', 6)
        .attr('cx', d => d.x || width / 2)
        .attr('cy', d => d.y || height / 2)
        .attr('fill', d => getPaperColor(d.paper))
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 1.5)
        .style('opacity', 0);

    // Add drag behavior
    if (useEmbeddings) {
        node.call(d3.drag()
            .on('start', dragStarted)
            .on('drag', dragged)
            .on('end', dragEnded));
    } else if (simulation) {
        node.call(drag(simulation));
    }

    // Animate nodes appearing
    node.transition()
        .duration(600)
        .delay((d, i) => Math.min(i * 1, 500))
        .style('opacity', 0.9);

    console.log('Nodes drawn:', nodes.length);

    // Drag functions for embedding mode
    function dragStarted(event, d) {
        d3.select(this).raise().attr('stroke-width', 3);
    }

    function dragged(event, d) {
        d3.select(this).attr('cx', event.x).attr('cy', event.y);
    }

    function dragEnded(event, d) {
        d3.select(this).attr('stroke-width', 1.5);
    }

    node.on('click', async (event, d) => {
        // Animate selected node
        d3.select(event.target)
            .transition()
            .duration(200)
            .attr('r', 10)
            .transition()
            .duration(200)
            .attr('r', 6);

        // Find similar papers based on embeddings
        const similarPapers = await findSimilarPapers(d.paper, 10);

        // Highlight similar nodes
        const similarNodeIds = new Set(similarPapers.map(sp => sp.paper.slug));

        node.style('opacity', n => {
            if (n.id === d.id) return 1;
            if (similarNodeIds.has(n.id)) return 0.9;
            return 0.2;
        })
        .attr('stroke-width', n => {
            if (n.id === d.id) return 3;
            if (similarNodeIds.has(n.id)) return 2.5;
            return 1.5;
        });

        // Highlight links to similar papers
        link.style('opacity', l => {
            const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
            const targetId = typeof l.target === 'object' ? l.target.id : l.target;

            if ((sourceId === d.id && similarNodeIds.has(targetId)) ||
                (targetId === d.id && similarNodeIds.has(sourceId))) {
                return 0.8;
            }
            return 0.1;
        });

        // Show paper detail with similar papers
        await selectPaper(d.paper, similarPapers);
    });

    // Tooltip
    const tooltip = createTooltip(container);

    node.on('mouseover', (event, d) => {
        // Highlight node
        d3.select(event.target)
            .transition()
            .duration(200)
            .attr('r', 10)
            .attr('stroke-width', 2.5);

        // Highlight connected links
        const connectedNodeIds = new Set();
        link.attr('class', l => {
            if (l.source.id === d.id || l.target.id === d.id) {
                if (l.source.id === d.id) connectedNodeIds.add(l.target.id);
                if (l.target.id === d.id) connectedNodeIds.add(l.source.id);
                return 'link active';
            }
            return 'link';
        });

        // Highlight connected nodes
        node.style('opacity', n => {
            if (n.id === d.id) return 1;
            if (connectedNodeIds.has(n.id)) return 0.9;
            return 0.3;
        });

        const paper = d.paper;
        tooltip.html(`
            <strong>${paper.title || paper.title_en}</strong><br>
            <span style="color: var(--text-tertiary)">${paper.year} · ${(paper.authors || []).slice(0, 2).join(', ')}</span>
        `);
        tooltip.style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 28) + 'px')
            .classed('visible', true);
    })
    .on('mouseout', (event) => {
        // Reset node
        d3.select(event.target)
            .transition()
            .duration(200)
            .attr('r', 6)
            .attr('stroke-width', 1.5);

        // Reset links
        link.attr('class', 'link');

        // Reset all nodes
        node.style('opacity', 0.85);

        tooltip.classed('visible', false);
    });

    // Update positions for force simulation
    if (simulation) {
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
        });
    }

    console.log('Network visualization complete');
}

// Timeline Visualization
function renderTimelineView(container) {
    if (!state.filteredPapers.length) {
        container.innerHTML = '<div class="empty-state"><p>表示する論文がありません</p></div>';
        return;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = { top: 40, right: 40, bottom: 60, left: 60 };

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('id', 'timeline-viz');

    // Group papers by year
    const papersByYear = d3.group(state.filteredPapers, d => d.year);
    const years = Array.from(papersByYear.keys()).sort();

    // Scales
    const xScale = d3.scaleBand()
        .domain(years)
        .range([margin.left, width - margin.right])
        .padding(0.1);

    const maxPapersInYear = Math.max(...Array.from(papersByYear.values()).map(arr => arr.length));
    const yScale = d3.scaleLinear()
        .domain([0, maxPapersInYear])
        .range([height - margin.bottom, margin.top]);

    // Color by concept
    const conceptColors = getConceptColorMap();

    function getPaperColorForTimeline(paper) {
        if (paper.concepts && paper.concepts.length > 0) {
            const primaryConcept = paper.concepts[0];
            const topLevel = primaryConcept.id.split('.')[0];
            return conceptColors.get(topLevel) || '#6366f1';
        }
        return '#6366f1';
    }

    // Draw axes
    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(xScale))
        .selectAll('text')
        .style('fill', 'var(--text-secondary)');

    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale))
        .selectAll('text')
        .style('fill', 'var(--text-secondary)');

    // Draw bars
    years.forEach(year => {
        const papers = papersByYear.get(year);
        const barWidth = xScale.bandwidth();
        const barX = xScale(year);

        papers.forEach((paper, i) => {
            const barHeight = (height - margin.bottom - margin.top) / maxPapersInYear;
            const barY = height - margin.bottom - (i + 1) * barHeight;

            const color = getPaperColorForTimeline(paper);

            svg.append('rect')
                .attr('class', 'timeline-paper')
                .attr('x', barX)
                .attr('y', barY)
                .attr('width', barWidth)
                .attr('height', barHeight - 1)
                .attr('fill', color)
                .attr('stroke', '#1e2139')
                .attr('stroke-width', 0.5)
                .style('opacity', 0.8)
                .on('click', () => selectPaper(paper))
                .append('title')
                .text(paper.title || paper.title_en);
        });
    });
}

// Sunburst Visualization
function renderSunburstView(container) {
    if (!state.data.concept_tree) {
        container.innerHTML = '<div class="empty-state"><p>概念ツリーがありません</p></div>';
        return;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;
    const radius = Math.min(width, height) / 2;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('id', 'sunburst-viz')
        .append('g')
        .attr('transform', `translate(${width / 2},${height / 2})`);

    // Create hierarchy
    const root = d3.hierarchy(state.data.concept_tree)
        .sum(d => d.count || 1)
        .sort((a, b) => b.value - a.value);

    // Create partition layout
    const partition = d3.partition()
        .size([2 * Math.PI, radius]);

    partition(root);

    // Color scale using concept color map
    const conceptColors = getConceptColorMap();
    const color = (node) => {
        // Get the top-level ancestor
        let topNode = node;
        while (topNode.parent && topNode.depth > 1) {
            topNode = topNode.parent;
        }

        // Try to get color from concept ID
        if (topNode.data.id) {
            const topLevel = topNode.data.id.split('.')[0];
            const mappedColor = conceptColors.get(topLevel);
            if (mappedColor) return mappedColor;
        }

        // Fallback to ordinal scale
        return d3.schemeCategory10[topNode.parent.children.indexOf(topNode) % 10];
    };

    // Arc generator
    const arc = d3.arc()
        .startAngle(d => d.x0)
        .endAngle(d => d.x1)
        .padAngle(d => Math.min((d.x1 - d.x0) / 2, 0.005))
        .padRadius(radius / 2)
        .innerRadius(d => d.y0)
        .outerRadius(d => d.y1 - 1);

    // Draw arcs
    svg.selectAll('path')
        .data(root.descendants().filter(d => d.depth))
        .join('path')
        .attr('fill', d => color(d))
        .attr('fill-opacity', d => 0.7 - (d.depth * 0.1))
        .attr('stroke', '#1e2139')
        .attr('stroke-width', 0.5)
        .attr('d', arc)
        .style('cursor', 'pointer')
        .on('click', (event, d) => {
            if (d.data.id) {
                selectConcept(d.data.id);
            }
        })
        .append('title')
        .text(d => `${d.ancestors().map(d => d.data.name).reverse().join(' / ')}\n${d.value} papers`);
}

// Color Mapping for Concepts
function getConceptColorMap() {
    // High-quality color palette for ACM CCS categories
    const palette = new Map([
        // Computing methodologies
        ['10010147', '#8b5cf6'], // Purple
        // Human-centered computing
        ['10003120', '#ec4899'], // Pink
        // Information systems
        ['10002951', '#06b6d4'], // Cyan
        // Networks
        ['10003033', '#10b981'], // Green
        // Security and privacy
        ['10002978', '#ef4444'], // Red
        // Software and its engineering
        ['10011007', '#f59e0b'], // Orange
        // Theory of computation
        ['10003752', '#6366f1'], // Indigo
        // Hardware
        ['10010198', '#14b8a6'], // Teal
        // Computer systems organization
        ['10010520', '#a855f7'], // Violet
        // Applied computing
        ['10003456', '#84cc16'], // Lime
        // Mathematics of computing
        ['10002950', '#0ea5e9'], // Sky
        // Social and professional topics
        ['10003461', '#f97316'], // Deep Orange
    ]);

    return palette;
}

// Zoom Controls
function setupZoomControls(svg, zoom) {
    const zoomInBtn = document.getElementById('zoom-in');
    const zoomOutBtn = document.getElementById('zoom-out');
    const zoomResetBtn = document.getElementById('zoom-reset');

    if (zoomInBtn) {
        zoomInBtn.onclick = () => {
            svg.transition().duration(300).call(zoom.scaleBy, 1.3);
        };
    }

    if (zoomOutBtn) {
        zoomOutBtn.onclick = () => {
            svg.transition().duration(300).call(zoom.scaleBy, 0.7);
        };
    }

    if (zoomResetBtn) {
        zoomResetBtn.onclick = () => {
            svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
        };
    }
}

// Embedding Similarity Functions
async function loadPaperEmbedding(paper) {
    if (state.paperEmbeddingsCache.has(paper.slug)) {
        return state.paperEmbeddingsCache.get(paper.slug);
    }

    try {
        const response = await fetch(new URL(paper.path, SUMMARIES_BASE_URL));
        if (response.ok) {
            const data = await response.json();
            const embeddings = data.embeddings || {};
            state.paperEmbeddingsCache.set(paper.slug, embeddings);
            return embeddings;
        }
    } catch (error) {
        console.warn('Failed to load embeddings for', paper.slug, error);
    }
    return null;
}

function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function findSimilarPapers(targetPaper, topN = 10) {
    const targetEmbeddings = await loadPaperEmbedding(targetPaper);
    if (!targetEmbeddings) return [];

    const section = state.embeddingSection;
    const targetVector = targetEmbeddings[section];
    if (!targetVector) return [];

    // Calculate similarities with all filtered papers
    const similarities = [];

    for (const paper of state.filteredPapers) {
        if (paper.slug === targetPaper.slug) continue;

        const paperEmbeddings = await loadPaperEmbedding(paper);
        if (!paperEmbeddings || !paperEmbeddings[section]) continue;

        const similarity = cosineSimilarity(targetVector, paperEmbeddings[section]);
        if (similarity >= state.similarityThreshold) {
            similarities.push({ paper, similarity });
        }
    }

    // Sort by similarity and return top N
    similarities.sort((a, b) => b.similarity - a.similarity);
    return similarities.slice(0, topN);
}

// Helper Functions
function getSharedConcepts(paper1, paper2) {
    if (!paper1.concepts || !paper2.concepts) return 0;

    const concepts1 = new Set(paper1.concepts.map(c => c.id));
    const concepts2 = new Set(paper2.concepts.map(c => c.id));

    let shared = 0;
    concepts1.forEach(c1 => {
        concepts2.forEach(c2 => {
            if (c1 === c2 || c1.startsWith(c2) || c2.startsWith(c1)) {
                shared++;
            }
        });
    });

    return shared;
}

function drag(simulation) {
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    return d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
}

function createTooltip(container) {
    return d3.select('body')
        .append('div')
        .attr('class', 'tooltip');
}

// Detail Panel
function showConceptDetail(conceptId) {
    const concept = state.conceptMap.get(conceptId);
    if (!concept) return;

    const detailContent = document.getElementById('detail-content');

    const papers = state.filteredPapers.filter(paper =>
        paper.concepts && paper.concepts.some(c => c.id === conceptId)
    );

    detailContent.innerHTML = `
        <div class="detail-section">
            <h3>概念</h3>
            <div class="content">
                <div class="paper-title">${concept.path || concept.id}</div>
                <div class="paper-meta">
                    <span class="meta-item">${papers.length} 件の論文</span>
                </div>
            </div>
        </div>

        <div class="detail-section">
            <h3>関連論文</h3>
            <div class="content">
                <div class="paper-list">
                    ${papers.slice(0, 50).map(paper => `
                        <div class="paper-item" onclick="selectPaperBySlug('${paper.slug}')">
                            <div class="paper-item-title">${paper.title || paper.title_en}</div>
                            <div class="paper-item-meta">${paper.year} · ${(paper.authors || []).slice(0, 2).join(', ')}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
}

async function selectPaper(paper, similarPapers = null) {
    state.selectedPaper = paper;
    await showPaperDetail(paper, similarPapers);
}

window.selectPaperBySlug = async function(slug) {
    const paper = state.paperMap.get(slug);
    if (paper) {
        await selectPaper(paper);
    }
};

async function showPaperDetail(paper, similarPapers = null) {
    const detailContent = document.getElementById('detail-content');

    const concepts = paper.concepts || [];

    // Show loading state
    detailContent.innerHTML = `
        <div class="loading">
            <div class="loading-spinner"></div>
        </div>
    `;

    // Load full paper data
    let fullData = null;
    try {
        const response = await fetch(new URL(paper.path, SUMMARIES_BASE_URL));
        if (response.ok) {
            fullData = await response.json();
        }
    } catch (error) {
        console.warn('Failed to load full paper data:', error);
    }

    // Render detail
    detailContent.innerHTML = `
        <div class="detail-section">
            <h3>論文情報</h3>
            <div class="content">
                <div class="paper-title">${paper.title || paper.title_en}</div>
                <div class="paper-meta">
                    <span class="meta-item">${paper.year}</span>
                    ${paper.authors && paper.authors.length ?
                        `<span class="meta-item">${paper.authors.join(', ')}</span>` : ''}
                </div>
            </div>
        </div>

        ${fullData && fullData.abstract ? `
        <div class="detail-section">
            <h3>概要</h3>
            <div class="content">
                <p>${fullData.abstract || fullData.abstract_en || '（概要なし）'}</p>
            </div>
        </div>
        ` : ''}

        ${fullData && (fullData.positioning_summary || fullData.positioning_summary_en) ? `
        <div class="detail-section">
            <h3>位置付け</h3>
            <div class="content">
                <p>${fullData.positioning_summary || fullData.positioning_summary_en}</p>
            </div>
        </div>
        ` : ''}

        ${fullData && (fullData.purpose_summary || fullData.purpose_summary_en) ? `
        <div class="detail-section">
            <h3>目的</h3>
            <div class="content">
                <p>${fullData.purpose_summary || fullData.purpose_summary_en}</p>
            </div>
        </div>
        ` : ''}

        ${fullData && (fullData.method_summary || fullData.method_summary_en) ? `
        <div class="detail-section">
            <h3>手法</h3>
            <div class="content">
                <p>${fullData.method_summary || fullData.method_summary_en}</p>
            </div>
        </div>
        ` : ''}

        ${fullData && (fullData.evaluation_summary || fullData.evaluation_summary_en) ? `
        <div class="detail-section">
            <h3>評価</h3>
            <div class="content">
                <p>${fullData.evaluation_summary || fullData.evaluation_summary_en}</p>
            </div>
        </div>
        ` : ''}

        ${concepts.length > 0 ? `
        <div class="detail-section">
            <h3>概念</h3>
            <div class="content">
                <div class="paper-concepts">
                    ${concepts.map(c => `
                        <span class="concept-tag ${c.confidence}-confidence"
                              onclick="selectConcept('${c.id}')">
                            ${c.path ? c.path.split(' → ').pop() : c.id}
                        </span>
                    `).join('')}
                </div>
            </div>
        </div>
        ` : ''}

        ${similarPapers && similarPapers.length > 0 ? `
        <div class="detail-section">
            <h3>類似論文（${state.embeddingSection} embeddings）</h3>
            <div class="content">
                <div class="paper-list">
                    ${similarPapers.map(sp => `
                        <div class="paper-item" onclick="selectPaperBySlug('${sp.paper.slug}')">
                            <div class="paper-item-title">${sp.paper.title || sp.paper.title_en}</div>
                            <div class="paper-item-meta">
                                ${sp.paper.year} ·
                                <span class="similarity-score">類似度: ${(sp.similarity * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
        ` : ''}
    `;
}

function closeDetail() {
    const detailContent = document.getElementById('detail-content');
    detailContent.innerHTML = '<div class="empty-state"><p>論文または概念を選択すると詳細が表示されます</p></div>';
    state.selectedPaper = null;
}

// Update UI
function updateHeader() {
    const paperCount = state.filteredPapers.length;
    const conceptCount = state.conceptMap.size;

    document.getElementById('paper-count').textContent = `${paperCount} papers`;
    document.getElementById('concept-count').textContent = `${conceptCount} concepts`;

    if (state.data.years && state.data.years.length > 0) {
        const years = state.data.years.map(y => y.year);
        const minYear = Math.min(...years);
        const maxYear = Math.max(...years);
        document.getElementById('year-range').textContent = `${minYear} - ${maxYear}`;
    }
}

function updateSelectionInfo() {
    const info = document.getElementById('selection-info');
    const parts = [];

    // Show embedding section in network view
    if (state.currentView === 'network') {
        const sectionLabels = {
            'abstract': 'Abstract',
            'overview': 'Overview',
            'positioning': 'Positioning',
            'purpose': 'Purpose',
            'method': 'Method',
            'evaluation': 'Evaluation'
        };
        parts.push(`📊 ${sectionLabels[state.embeddingSection]}`);
    }

    if (state.selectedConcept) {
        const concept = state.conceptMap.get(state.selectedConcept);
        if (concept) {
            const simpleName = concept.path ? concept.path.split(' → ').pop() : concept.id;
            parts.push(`🎯 ${simpleName}`);
        }
    }

    if (state.searchTerm) {
        parts.push(`🔍 "${state.searchTerm}"`);
    }

    if (state.yearFilter) {
        parts.push(`📅 ${state.yearFilter}`);
    }

    parts.push(`📄 ${state.filteredPapers.length} 件`);

    info.textContent = parts.join(' · ');
}

// Loading and Error States
function showLoading() {
    const viz = document.getElementById('visualization');
    viz.innerHTML = '<div class="loading"><div class="loading-spinner"></div></div>';
}

function hideLoading() {
    // Loading will be cleared by visualization render
}

function showError(message) {
    const viz = document.getElementById('visualization');
    viz.innerHTML = `<div class="empty-state"><p>エラー: ${message}</p></div>`;
}

// Start the application
init();
