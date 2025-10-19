# Local LLM Mode – Implementation Direction

## Goal
- Allow users to swap the PDF summarisation pipeline between OpenAI APIs and locally hosted OSS models (e.g., `gpt-oss-120b`) with minimal friction.
- Keep embeddings flexible so summaries can rely on local generation while embeddings may still use external providers when needed.

## Provider Abstraction
- Extend `Pre-Processing/orchestrator.py` `SummariserConfig` with `llm_provider`, `llm_model`, `llm_endpoint`, `llm_device_map`, `llm_precision`.
- Pass the new fields through the job state (`utils/state.py`) so queued jobs retain their provider choice; add matching CLI flags and `.env` support (`LLM_PROVIDER`, etc.).
- Introduce `summary/providers/` with a factory that returns a provider object implementing `generate_summary(prompt: str, params: LLMCallParams) -> str`.
  - `openai.py`: wrap existing `load_openai_client` logic.
  - `hf_local.py`: load `AutoModelForCausalLM` + `AutoTokenizer` once, expose a simple `pipeline("text-generation")`-based call, respect temperature/max_tokens.

## Summarisation Flow
- In `summary/summarize_pdf.py`, replace direct uses of `load_openai_client()` with provider lookup:
  - Build the model prompt as today.
  - Call `provider.generate_summary(...)`.
  - Keep retry/backoff logic provider-agnostic (only adjust when a provider signals rate limits).
- Share common call parameters via a dataclass (temperature, max_tokens, top_p, etc.) so OpenAI/OSS behave similarly.

## Embedding Strategy
- Default to `embedding_provider=sentence-transformers` for OSS mode via `.env`/CLI.
- Reuse `maybe_compute_embeddings_local()` for local SentenceTransformer runs; continue supporting Gemini/Vertex for teams relying on external embeddings.
- Document trade-off: OSS summary + external embeddings is acceptable, but embeddings can also be local if `sentence-transformers` is installed.

## Runtime & Dependencies
- Add `transformers`, `accelerate`, and (optionally) `bitsandbytes` to `requirements.txt`; note that 120B-class models demand multi-GPU setups or 4-bit quantisation.
- Recommend users preload the model (e.g., `device_map="auto"`, `load_in_4bit=True`) and, if desired, run a long-lived FastAPI/TGI server; allow `.env` to provide `LLM_ENDPOINT` for HTTP-based inference instead of direct model loading.
- Surface clear errors when required environment variables or model paths are missing.

## Testing & Validation
- Mirror existing OpenAI-mode tests: ensure JSON schema, dual-language output, and retry behaviour remain intact whichever provider is selected.
- For embeddings, add assertions that output vectors are normalised and dimension-consistent regardless of provider.
- Provide a minimal smoke test pipeline (`tests/test_local_provider.py`) loading a tiny HF model (e.g., `sshleifer/tiny-gpt2`) to keep CI lightweight.

## Documentation & UX
- Update README / developer guide with setup steps:
  1. Install extra Python deps.
  2. Place/serve the local HF model.
  3. Set `.env` with `LLM_PROVIDER=oss` and optional endpoint/device parameters.
  4. Run `python orchestrator.py run --llm-provider oss ...`.
- Clarify resource expectations and suggest caching behaviour (model loads once per process; consider daemon or HTTP server for shared use).
