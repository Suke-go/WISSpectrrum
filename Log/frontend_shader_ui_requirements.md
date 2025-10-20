# Front-End Visualization Direction (Three.js + Shader Bubbles)

## 1. Experience Goals
- Deliver a full-screen, shader-driven explorer that lets users dive from ACM CCS top-level concepts to individual papers while retaining analytic clarity.
- Emphasise hierarchy depth with living bubbles, smooth transitions, and embedding-aware layouts so users can judge conceptual distance at a glance.
- Support fast focus on a single paper, with actionable shortcuts into related work and back out to the broader concept context.

## 2. Scene Layout & Camera
- Camera: single `THREE.OrthographicCamera`. On resize, compute half-width `hw = virtualWidth * 0.5` (default `virtualWidth = 10`). `virtualWidth` is an arbitrary logical unit expressed in Three.js world units; all bubble radii, layouts, and interaction distances are relative to this width so changing it globally scales the entire scene without re-tuning child values. Half-height `hh = hw * (canvasHeight / canvasWidth)`. Assign `camera.left = -hw`, `camera.right = hw`, `camera.top = hh`, `camera.bottom = -hh`, `camera.near = -10`, `camera.far = 10`, then call `camera.updateProjectionMatrix()`. Camera position is `(0, 0, 5)` looking at `(0, 0, 0)` with `up = (0, 1, 0)`. Leave an inline comment in code clarifying that near/far span negative to positive so flat content on the z-plane remains visible even if minor positive or negative offsets are applied.
- Root scene graph: `const rootGroup = new THREE.Group();` All bubble geometries are children of this group. Zoom animations operate on `rootGroup.scale.setScalar(zoomFactor)` and panning adjusts `rootGroup.position`.
- Canvas fills `100vw x 100vh`. Overlay UI (breadcrumbs, HUD, command palette, etc.) renders via React portals positioned with CSS; default fade duration 250 ms using ease-in-out cubic Bezier `(0.4,0,0.2,1)`.
- Safe area & responsive logic: maintain a design reference at 16:9 for desktop. Compute `renderWidth = Math.min(window.innerWidth, window.innerHeight * 16 / 9)` and treat this value as `canvasWidth` for camera math (distinct from `window.innerWidth`). When `window.innerWidth < 768` (iPad portrait included) switch HUD/detail panels to stacked layout, increase label density, and reduce the default zoom so key content stays visible. Taller aspect ratios still render the same instanced layout; only overlay components reorganise around the centred canvas. Limit `renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))` for performance.

## 3. Concept Representation
- Geometry: instantiate one `CircleGeometry(1, 128)` and reuse it through `THREE.InstancedMesh`. Keep UVs centred (0.5, 0.5) for radial shader math.
- Radius encoding: `radius = baseRadius * Math.sqrt(paperCount)` with `baseRadius = 0.25`. `baseRadius` is expressed relative to the virtual width (2.5 percent of the default camera width). Store per-instance scale in `instanceMatrix`. Apply GLSL `smoothstep(0.8, 1.0, length(uvCentered))` to soften bubble edges.
- Layout pipeline (executed in `layoutConceptChildren(parentId)`):
  1. Gather child concepts, calculate their radii.
  2. Sort children by embedding angle (`angleSeed`) and concept similarity weight (Section 7).
  3. Place children on polar rings: `radiusRing = parentRadius * (ringIndex + 1) * ringSpacing`, with `ringSpacing = 0.6`. When a child would overlap, increment `ringIndex` and recompute angle spacing.
  4. For fallback, call `packSiblings` from `d3-hierarchy` (`import { packSiblings } from 'd3-hierarchy'`) on the 2D radii to resolve collisions, then normalise output to the parent radius space.
  5. Write target positions to `conceptLayout[childId] = { x, y, scale }`.
- Adaptive spacing: `ringSpacing` starts at `0.6` but is increased by `+0.1` until collisions disappear or until it reaches `1.2`; beyond that, fallback packing is enforced. Add unit tests that stress 50+ child concepts to ensure overlaps are resolved.
- Dataset reality: current WISS data tops out at 12 child concepts for a single parent, but the 50+ stress test keeps buffer for future datasets or manual overrides.
- Concept layout uses polar rings to preserve a clear hierarchical "orbit" structure; paper sparks (below) use a Fermat spiral because papers are numerous per concept and benefit from even, gap-free packing inside the bubble interior.
- Z-order: keep all instances on `z = 0`. When necessary to guarantee draw order (e.g., focused bubble), apply epsilon offsets `z = +/-0.01`.
- Paper sparks: render via `THREE.Points` with `BufferGeometry` storing `position`, `size`, and `sectionKey`. Use a single `THREE.ShaderMaterial` with circular point sprite. Hover pulse is controlled by `uPulseStrength` uniform per paper (Section 4).

## 4. Shader & Material Strategy
- Core concept material: `new THREE.ShaderMaterial({ transparent: true, depthWrite: false, uniforms: {...}, vertexShader, fragmentShader })`.
- Uniforms:
  - `uniform float uConceptDensity;` value in `[0,1]` mapped from paper count. Normalise by dividing the paper count by the maximum count within the current depth level and clamp the result. Applied as `vec3 baseColor = mix(vec3(0.18,0.24,0.34), vec3(0.9,0.98,1.0), pow(uConceptDensity, 0.4));`.
  - `uniform float uFocus;` in `[0,1]` for selection intensity.
  - `uniform float uTime;` global clock seconds (`THREE.Clock.getElapsedTime()`).
  - `uniform float uPulseAmp;` (default `0.06`) and `uniform float uPulseRate;` (default `0.6` Hz) for heartbeat animation.
  - `uniform float uWaveMix;` blending factor for distance lens ripple (Section 7).
- Heartbeat radius modulation: In vertex shader, declare `const float TWO_PI = 6.28318530718;` and scale radius by `float pulse = pow(max(0.0, sin(TWO_PI * uPulseRate * uTime)), 1.6); float heartbeat = 1.0 + uPulseAmp * mix(0.0, pulse, uFocus);`. Multiply `position.xy` by `heartbeat`.
- Rim highlight: fragment shader computes `float edge = smoothstep(0.85, 0.98, length(uvCentered)); vec3 rimColor = mix(baseColor, vec3(1.0,0.84,0.3), uFocus * edge);`.
- Distance lens ripple (activated via Alt key): `float distWave = sin(distanceFactor * 10.0 + uTime * 2.0); float ripple = mix(0.0, distWave * 0.1, uWaveMix);` where `distanceFactor` is provided per instance (Section 7). Blend ripple into alpha and rim color to visualise proximity. For performance, move the sine sample into the vertex shader where possible so fragment shader receives an interpolated value, and fall back to disabling the ripple (set `uWaveMix = 0`) in low-fidelity mode to avoid heavy trigonometry on low-end GPUs.
- Vertex implementation detail: add an instanced attribute `aDistanceFactor` that stores the precomputed distance, pass `varying float vDistWave = sin(aDistanceFactor * 10.0 + uTime * 2.0);` to the fragment shader, and multiply by `uWaveMix` there.
- Post-processing: `EffectComposer` with `RenderPass`, `UnrealBloomPass({ threshold: 0.6, strength: 0.7, radius: 0.85 })`, and optional `FXAAShader`. Provide a medium-fidelity preset (strength 0.3, radius 0.65) before disabling bloom entirely in low-fidelity mode so mid-tier devices keep some glow without the full cost.

## 5. Interaction Model

### 5.1 Tutorial Onboarding
- Component: `TutorialOverlay.tsx`.
- Trigger: `useEffect` checks `localStorage.getItem('wissViz.hasSeenTutorial')`. If absent, show overlay sequence.
- Steps (each step uses translucent dark backdrop `rgba(10,12,28,0.82)` and white text):
  1. "Top Concepts" overlay highlights outer ring; instruct wheel/pinch to dive. Duration 6 s or dismissed via click.
  2. "Mid Concepts" overlay explains embedding-based arrangement. Duration 6 s.
  3. "Paper Sparks" overlay explains spark pulses and selection.
- On completion, set flag in localStorage and allow replay via command palette action `tutorial:replay`.

### 5.2 Dive & Navigation
- Event binding: `useGesture` or raw listeners on canvas capturing `onWheel`, `onPinch`, `onPointerMove`.
- Wheel/pinch increments `navigation.depth` by +/- 1 with guard `clamp(depth, 0, maxDepth)`. Each change enqueues animation: `gsap.to(rootGroup.scale, { x: target, y: target, duration: 0.5, ease: 'expo.out' })`.
- Breadcrumb overlay: component `BreadcrumbHUD.tsx` renders current path using `SmallCaps` style. Provide keyboard shortcuts: `Esc` triggers `navigateUp()`, arrow keys call `focusSibling(delta)`.

### 5.3 HUD (Hierarchy Status Display)
- Component: `HierarchyHUD.tsx`.
- Placement: top-left corner, container width 260 px.
- Typography: `font-family: "Noto Sans JP", "Inter", sans-serif; font-weight: 600; font-size: clamp(16px, 1.6vw, 20px); letter-spacing: 0.04em; color: #f7fafc`. Noto Sans JP sits first to guarantee Japanese glyph coverage; Inter follows so Latin text retains a clean appearance on English-heavy concepts.
- Background: blurred glass `background: rgba(14,21,42,0.38); border-radius: 12px; box-shadow: 0 12px 32px rgba(6,10,24,0.35); backdrop-filter: blur(12px)`.
- Content fields:
  - `Level`: textual label (Top Concepts, Mid Concepts, Paper Sparks).
  - `Visible nodes`: count of rendered concepts/papers on current level.
  - `Next level`: count of descendants that will appear after diving.
  - `Distance lens`: icon toggles glow when active.
- Update frequency: `useFrame` hook pushes state to HUD once per animation frame; expensive stats (counts) cached per depth in Zustand store.

### 5.4 Focus & Detail Interaction
- Bubble click handler `handleConceptSelect(conceptId)` sets `selection.highlightedConceptId` and raises `uFocus` to 1.0 for that instance.
- Heartbeat: while focused, `uFocus` stays at 1.0 and `uPulseRate` is set to 0.35 Hz, `uPulseAmp = 0.09`. On blur, tween `uFocus` back to 0 over 0.4 s.
- Tooltip: `Tooltip.tsx` appears on hover; content: title (bold), paper count, top 3 keywords (comma separated). Auto-dismiss after 1.5 s of pointer inactivity.
- Side panel `DetailPanel.tsx`: opens on concept click. Sections:
  - Concept summary (title, description).
  - Metric chips (paper count, embedding variance, latest year).
  - Action buttons: `Jump to parent`, `Toggle distance lens`, `Open related papers`.
  - Paper list grouped by section (purpose, method, evaluation, abstract).
- Paper select `handlePaperSelect(paperId)` zooms the spark via tween `gsap.to(paperInstance.scale, { value: 1.6, duration: 0.4 })` and shows `PaperCard.tsx` anchored near selection.

### 5.5 Paper Card & Similarity Shortcuts
- `PaperCard.tsx` layout: 360 px width, dark glass background, 16 px padding, 12 px rounded corners.
- Sections:
  - Paper metadata (title, authors, year, DOI link).
  - Section selector: tabs for `abstract`, `purpose`, `method`, `evaluation`, `positioning`. Each tab loads section-specific embedding vector.
  - Similar papers list: call `computeSectionSimilarity(paperId, sectionKey, limit, scope)` (see Section 6) to show top 3 with score badges. Each entry has `Open` (panel) and `Set focus` (zoom to spark).
  - Scope toggle: segmented control with options `Same concept`, `Sibling concepts`, `Entire taxonomy`.
  - Buttons: `Back to concept`, `Open PDF`.

## 6. Data Binding & State
- Normalised stores (`Zustand` slice names):
  - `useNavigationStore`: `activeConceptId`, `breadcrumbIds`, `depth`, `hasSeenTutorial`.
  - `useSelectionStore`: `highlightedConceptId`, `selectedPaperId`, `distanceLensActive`, `activeSectionKey`.
  - `useResourceStore`: caches for geometries, materials, instanced attributes; actions `getConceptMaterial(conceptId)` and `disposeConceptResources(conceptId)`.
- `index.json` augmentation: extend each concept node with `{ clusterId, angleSeed, sectionKeywords }`. Extend papers with `{ embeddings: { abstract: [], purpose: [], ... }, sectionKeywords: { purpose: [], ... } }`.
- Ensure all stored embedding vectors are L2-normalised during preprocessing so cosine similarity remains stable; record `embedding_norm` metadata for validation.
- `paperEmbeddingStore` is a read-through cache backed by a `Map<string, Record<SectionKey, Float32Array>>`; entries may be absent temporarily while section embeddings stream in lazily (`get` returns `undefined` until the fetch completes), which is why the similarity helper guards for `!target`. TypeScript shape:
  ```ts
  type PaperEmbeddingStore = {
    get(paperId: string, section: SectionKey): Float32Array | undefined;
    preload(paperIds: string[], sections?: SectionKey[]): Promise<void>;
    clear(): void;
  };
  ```
- Similarity computation module `similarity.ts`:
  ```ts
  export function cosineSim(vecA: Float32Array, vecB: Float32Array): number {
    if (vecA.length !== vecB.length) {
      throw new Error(`Embedding length mismatch: ${vecA.length} vs ${vecB.length}`);
    }
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      const a = vecA[i];
      const b = vecB[i];
      dot += a * b;
      normA += a * a;
      normB += b * b;
    }
    const denom = Math.max(Math.sqrt(normA) * Math.sqrt(normB), 1e-6);
    return denom === 0 ? 0 : dot / denom;
  }

  export function computeSectionSimilarity(paperId: string, sectionKey: SectionKey, limit: number, scope: Scope): SimilarPaper[] {
    const target = paperEmbeddingStore.get(paperId, sectionKey);
    if (!target) return [];
    const pool = filterByScope(paperId, scope); // same concept, sibling concepts, entire taxonomy
    const scored = pool
      .map(otherId => {
        const vector = paperEmbeddingStore.get(otherId, sectionKey);
        return vector ? { paperId: otherId, score: cosineSim(target, vector) } : null;
      })
      .filter((item): item is { paperId: string; score: number } => !!item && item.score < 0.999) // exclude self
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
    return scored;
  }
  ```
- Scope filters:
  - `Same concept`: paper shares `conceptId`.
  - `Sibling concepts`: concept parent identical.
  - `Entire taxonomy`: all papers under active top-level category.
- Cache results per `(paperId, sectionKey, scope)` for 5 minutes via `Map` with TTL. Cap the cache at 5 000 entries and evict least-recently-used items to avoid unbounded memory growth.
- Worker protocol: the main thread posts `{ type: 'SIMILARITY_REQUEST', requestId, paperId, sectionKey, scope, limit }` to `similarity.worker.ts`; the worker responds with `{ type: 'SIMILARITY_RESPONSE', requestId, results }`. Abort the request and fall back to synchronous computation if no response arrives within 10 seconds.
- For `Entire taxonomy` scope, dispatch the similarity computation to a Web Worker (`similarity.worker.ts`) so the main thread stays responsive. ANN graphs are built offline (HNSW preferred) and shipped as JSON/Parquet artifacts; the worker first checks the ANN result, then falls back to cosine over the full pool only when neighbours are missing.

## 7. Embedding-Driven Enhancements
- Child placement angle: `angle = angleSeed + orderIndex * (2 * Math.PI / childCount)` where `angleSeed` is deterministic from `clusterId`. Generate the seed using a stable MurmurHash3 on the clusterId string and divide by `UINT32_MAX` to obtain `hashValue` in `[0,1)`, then multiply by `2 * Math.PI`.
- Distance metric: `distance = 0.5 * (1.0 - cosineSim(targetEmbedding, otherEmbedding))` giving values in `[0,1]`.
- Distance lens colouring (Alt key toggled):
  - Base hue `baseHue = 200 / 360` (blue), accent hue `accentHue = 32 / 360` (amber).
  - Compute `mixFactor = smoothstep(0.0, 0.6, distance)`.
  - Convert HSL to RGB: `vec3 color = mix(hsl2rgb(baseHue, 0.65, 0.55), hsl2rgb(accentHue, 0.72, 0.62), mixFactor);`. Include GLSL helper:
    ```glsl
    vec3 hsl2rgb(float h, float s, float l) {
      float c = (1.0 - abs(2.0 * l - 1.0)) * s;
      float hp = mod(h * 6.0, 6.0);
      float x = c * (1.0 - abs(mod(hp, 2.0) - 1.0));
      vec3 rgb = vec3(0.0);
      if (0.0 <= hp && hp < 1.0) rgb = vec3(c, x, 0.0);
      else if (1.0 <= hp && hp < 2.0) rgb = vec3(x, c, 0.0);
      else if (2.0 <= hp && hp < 3.0) rgb = vec3(0.0, c, x);
      else if (3.0 <= hp && hp < 4.0) rgb = vec3(0.0, x, c);
      else if (4.0 <= hp && hp < 5.0) rgb = vec3(x, 0.0, c);
      else rgb = vec3(c, 0.0, x);
      float m = l - 0.5 * c;
      return rgb + vec3(m);
    }
    ```
    *Note: `h` is always normalised to `[0,1)` so `mod` never receives negative input; document this invariant in code comments.*
  - Optional branchless alternative for ultra-low-end GPUs:
    ```glsl
    vec3 hsl2rgb_branchless(float h, float s, float l) {
      vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
      return l + s * (rgb - 0.5) * (1.0 - abs(2.0 * l - 1.0));
    }
    ```
  - Set `uWaveMix = distanceLensActive ? 1.0 : 0.0`.
- Wave pattern on bubble boundary:
  - `float boundary = smoothstep(0.88, 1.0, length(uvCentered));`.
  - `float wave = sin(distance * 12.0 - uTime * 3.0);`.
  - Rim modulation: `edgeColor *= 1.0 + 0.08 * wave * uWaveMix; alpha *= 1.0 - 0.05 * boundary * uWaveMix;`.
- Paper spark ordering: within the same concept, sort paper sparks by `distance` so that closest papers cluster towards the bubble center. Compute positions on a Fermat spiral: `r = sqrt(idx / total) * conceptRadius * 0.85`, `theta = idx * 2.399963`. Multiply `r` by `(1.0 - distance * 0.4)` to pull closer papers inward.

## 8. Implementation Notes
- React integration:
  - Scene wrapper `ShaderBubbleScene.tsx` uses `@react-three/fiber`. Provide hooks `useConceptInstances()` and `usePaperPoints()` to push instanced attributes each frame.
  - Keep `useFrame` for updating uniforms (`uTime`, `uFocus`, `uWaveMix`).
  - Scene graph mutations (adding/removing instances) occur through React state reducers so ownership stays with React; avoid direct imperative adds outside controlled hooks.
- Input handling:
  - `useGesture` for wheel/pinch/pan, `KeyboardEvent` listeners for shortcuts (Esc, Arrow keys, Enter, ?, Alt).
  - For Alt key distance lens, guard to avoid interfering with OS menu shortcuts: enable only when canvas focused.
- Accessibility:
  - Canvas receives `tabIndex={0}`. On focus, read out `Current level: ${levelLabel}. ${visibleCount} items visible.` via `aria-live="polite"`.
  - Provide `ListView.tsx` that mirrors currently visible concepts/papers in textual form for screen-reader browsing.
- Performance:
  - Instanced concepts limit: 4 096 per level. Papers limit: 10 000 sparks; use dynamic buffer updates with `usage = DynamicDrawUsage`. When limits are exceeded, fall back to progressive rendering: display the top-N nodes sorted by paper count (largest visible first), queue the remainder for incremental rendering in idle callbacks, and surface both a HUD notice and an on-canvas badge (e.g., `+123 more`) anchored to the parent bubble to show hidden children.
  - Throttle similarity recomputation by using `requestIdleCallback` where available; fallback `setTimeout(..., 16)`. For long-running tasks, move to Web Worker as described above.
- Low-fidelity mode automatically sets `uPulseAmp = 0`, disables bloom, reduces sample count to 64 segments for circle geometry, and disables wave modulation.
- Progressive rendering: break concept loading into batches of <=512 nodes per frame, animate in newly mounted nodes so the scene stays responsive on large taxonomies.
- FPS monitoring: start collecting frame times after an initial 5-second warm-up (to ignore asset loading stalls). Drop to low fidelity when rolling average fps < 45 and only return to full quality once fps recovers above 55 to introduce hysteresis and prevent rapid toggling.

## 9. Command Palette Actions
- Implement with `CommandPalette.tsx` (e.g., using `cmdk` React library).
- Action list:
  - `jump:concept` - fuzzy search (Fuse.js) over concept names and paths, execute `navigateToConcept(conceptId)`.
  - `jump:paper` - fuzzy search over paper titles.
  - `lens:toggle` - toggles distance lens (Alt fallback).
  - `quality:low` / `quality:high` - force low/high fidelity.
  - `tutorial:replay` - reset tutorial flag and restart overlay sequence.
  - `section:set:<key>` - directly switch paper card section tab.
  - `search:scope:same` / `search:scope:sibling` / `search:scope:all` - set similarity scope.
  - `help:shortcuts` - open overlay listing all keyboard and palette shortcuts.
  - `export:screenshot` / `export:current-view` - trigger high-resolution capture of the current canvas and HUD state.
  - `debug:show-stats` - toggle FPS/react-three-fiber stats panel and memory usage overlay.
  - `navigate:home` - focus root concept and reset zoom/pan.
  - `clipboard:copy-paper-id` - copy the currently selected paper ID (and section key) to the clipboard for debugging.

## 10. Milestones
1a. [2-3 days] Static shader bubble (no animation) with mock hierarchy rendered via `InstancedMesh`. Deliverable: CodeSandbox link demonstrating circle instancing and base colour map.
1b. [1-2 days] Heartbeat and ripple uniforms wired (`uTime`, `uPulseAmp`, distance factor). Deliverable: short capture showing pulsation and Alt-key ripple toggle.
2a. [3-4 days] Core interaction scaffold: wheel/pinch depth changes, breadcrumb HUD placeholders, keyboard navigation. Deliverable: interaction checklist with GIF.
2b. [2 days] Tutorial overlay and hierarchy HUD visual polish, including responsive behaviour. Deliverable: merged PR with Storybook entries for HUD and tutorial.
3a. [4-5 days] Data integration: load real `index.json`, hydrate concept tree, map embeddings into state stores. Deliverable: unit tests validating layout output, plus developer docs update.
3b. [3 days] Similarity engine: Web Worker path, section selector, PaperCard shortcuts. Deliverable: profiling log showing worker offload and latency metrics.
4. [3-4 days] Distance lens final polish: colour waves, Fermat spiral layout, command palette toggles, export actions. Deliverable: before/after visuals and usability notes.
5. [4 days] Launch readiness: accessibility verification, progressive rendering for large datasets, low-fidelity detection and manual override. Deliverable: QA matrix, performance report (>=60 fps on Apple M1 Air, >=30 fps on Intel UHD 620).

## 11. Proposed Resolutions for Previously Open Questions
- Command palette as central control surface reduces UI clutter and keeps advanced actions discoverable.
- Paper-level view uses in-place zoom plus detailed card; no separate modal needed except optional "open in modal" for accessibility.
- Low-fidelity toggles automatically based on `navigator.hardwareConcurrency < 4` or `deviceMemory < 8`, with manual override through command palette. When those APIs are unavailable, sample frame time over the first 5 seconds and drop to low fidelity if rolling average fps < 45.

## 12. Backend & Embedding Pipeline Requirements

### 12.1 Data Production Pipeline
- Primary data generation continues in `Pre-Processing/orchestrator.py` with summarisation jobs producing JSON records under `Pre-Processing/output/summaries/<year>/*.json`.
- Extend `embedding/compute_embeddings.py` to emit section-level embeddings (abstract, purpose, method, evaluation, positioning) and ensure they are L2-normalised. Persist vectors using float32 arrays inside the JSON or a sidecar binary (`.npy`) referenced by path.
- After embeddings are generated, run `output/build_index.py`; extend it with a helper `build_embedding_metadata()` (new function inside the same module) that computes:
  - Concept centroid vectors (average of member paper embeddings).
  - Max paper count per concept depth (used for `uConceptDensity` normalisation).
  - Top keywords per concept via TF-IDF on summaries.
- Store precomputed k-NN graphs per section using HNSW (preferred for accuracy/speed trade-off; Annoy acceptable for low-memory deployments) via `python -m embeddings.build_ann --section abstract --M 16 --ef 200`. For corpora >100k papers increase to `M=32`, `ef=400`. Serialize adjacency lists to `output/ann/abstract_neighbors.parquet`. Parquet keeps file size small and columnar, even though the frontend will read it via a backend service rather than directly in the browser.

### 12.2 Similarity Service
- Launch a lightweight Node/Express or FastAPI service exposing:
  - `GET /concepts/index` -> returns `index.json` plus metadata.
  - `GET /papers/{paperId}/similarity?section=abstract&scope=sibling&limit=50` -> returns precomputed k-NN candidates (reads from ANN graph; falls back to on-the-fly cosine via Web Worker if missing).
  - `POST /embeddings/recompute` -> enqueues recomputation job with parameters (section subset, scope, priority).
- Optional endpoints: `GET /health` for monitoring, `GET /metadata` to list sections and last refresh timestamps, `POST /feedback` to capture manual similarity annotations.
- Service pulls similarity data from disk into memory on start (bounded by LRU). Provide configuration for maximum loaded indices (`MAX_SECTION_INDICES=3`). Enforce simple rate limiting (e.g., `100 req/min/ip`) and set CORS rules matching the deployed frontend domain. The system can operate in "static mode" (frontend reads precomputed ANN JSON directly) or "hybrid mode" (top-k baked into bundle, long-tail via API); document which mode is active in `/metadata`.

### 12.3 Job Scheduling & Workers
- Reuse the existing SQLite job queue (`utils/state.py`) to track embedding refresh tasks. Add job type `embedding_refresh` with payload `{ sectionKey, scope, strategy }`.
- Implement a background worker `workers/embedding_worker.py` that:
  1. Monitors queue for new jobs via polling.
  2. Runs `compute_embeddings.py --section <key> --scope <scope> --ann-output ...`.
  3. Updates ANN artifacts and triggers `build_index.py`.
- Add `validate_embedding_output(ann_path: Path) -> bool` to verify ANN integrity (node count, connectivity, checksum) before promoting new artifacts.
- Schedule periodic refresh (e.g., nightly) to rebuild centroids and keyword caches, ensuring frontend metrics stay fresh while preserving backwards compatibility with existing queue consumers.

### 12.4 Deployment & Ops
- Provide a new CLI `python manage.py export-frontend-data` that bundles `index.json`, ANN graphs, and metadata into `dist/frontend-data-v{version}.tar.gz`. Publish artifact hashes in `dist/frontend-data-v{version}.manifest.json` for CI verification.
- Use checksum manifests to verify client assets (`data-manifest.json` listing file paths and SHA256). Frontend checks manifest before loading to ensure synced versions.
- Document environment variables:
  - `VITE_WISS_VIRTUAL_WIDTH` (default 10) for frontend scaling (injected at build time).
  - `WISS_SIMILARITY_ENDPOINT` for Web Worker RPC.
  - `WISS_EMBEDDING_WORKER_CONCURRENCY` controlling parallel ANN rebuilds.
  - `WISS_DATA_VERSION` matching the exported tarball.
  - `WISS_FEATURE_FLAGS` (comma-separated, e.g., `distance_lens_v2`).

## 13. Data Scale & Embedding Characteristics
- Current WISS dataset snapshot: 12 top-level concepts, 277 total concept nodes, maximum hierarchy depth 5, and 315 papers.
- Section embeddings use 3 072 dimensions (OpenAI `text-embedding-3-large` scale); six sections per paper -> ~5.8 million floats (~23 MB as float32). Monitor bundle size and consider float16 compression (halve footprint) or product quantisation for offline archives if the corpus grows past ~5 000 papers.
- Progressive fetch strategy: ship float32 for precision in the first release, add opt-in quantised variants (`.f16`, `.u8`) behind `WISS_FEATURE_FLAGS`.
- Similarity ANN graphs are generated offline; the frontend never invokes native Node libraries such as `@vladmandic/hnswlib-node`.
