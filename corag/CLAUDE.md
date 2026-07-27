# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CoRAG (Chain-of-Retrieval Augmented Generation) is the release code for the paper
[Chain-of-Retrieval Augmented Generation](https://arxiv.org/abs/2501.14342). This repo contains
**inference/evaluation code only** — there is no training loop here. Training data and the
fine-tuned model (`corag/CoRAG-Llama3.1-8B-MultihopQA`) are distributed via HuggingFace; this repo
consumes them.

At a high level: an LLM agent iteratively generates subqueries, retrieves supporting documents for
each subquery from a dense retrieval server, generates subanswers, and finally produces an answer
conditioned on the full chain of (subquery, subanswer, retrieved docs) — evaluated on multi-hop QA
benchmarks (HotpotQA, 2WikiMultihopQA, MuSiQue, Bamboogle).

## Setup

```bash
pip install -r requirements.txt
```

Requires vLLM + an OpenAI-compatible serving stack; tested on 8×A100 40GB.

## Running evaluation

Three services/steps, run in order from the repo root:

1. Download precomputed E5 corpus embeddings (40 shards) into `data/e5-large-index/`:
   ```bash
   bash scripts/download_embeddings.sh
   ```
2. Start the dense retrieval (E5) server on port 8090 (background, logs to `e5_server.log`):
   ```bash
   bash scripts/start_e5_server.sh
   ```
3. Start the vLLM server on port 8000 (background, logs to `vllm_server.log`), tensor-parallel size
   auto-detected from `nvidia-smi`:
   ```bash
   bash scripts/start_vllm_server.sh corag/CoRAG-Llama3.1-8B-MultihopQA
   ```
4. Run evaluation across all four benchmarks (2wikimultihopqa, bamboogle, hotpotqa, musique):
   ```bash
   bash scripts/eval_multihopqa.sh
   ```
   This wraps `src/inference/run_inference.py` (see below) via `torchrun --nproc_per_node 1`, with
   `--max_path_length 6`, greedy decoding, `--do_eval`. Predictions and metrics land in
   `${OUTPUT_DIR}/6/` (`OUTPUT_DIR` defaults to `tmp/` — override by exporting it or passing extra
   args, which are forwarded via `"$@"`).

To run a single task/split directly instead of the full sweep, call `run_inference.py` yourself,
e.g.:
```bash
PYTHONPATH=src/ python src/inference/run_inference.py \
  --eval_task hotpotqa --eval_split validation \
  --max_path_length 6 --output_dir tmp/6 --do_eval \
  --num_threads 32 --overwrite_output_dir --report_to none
```

All CLI flags are defined in the `Arguments` dataclass in `src/config.py` (a `TrainingArguments`
subclass parsed with `HfArgumentParser`) — notably `decode_strategy` (`greedy`/`tree_search`/
`best_of_n`), `context_placement` (`forward`/`backward`/`random`), `num_contexts`,
`max_path_length`, `sample_temperature`, `best_n`.

There is no automated test suite in this repo.

## Architecture

The chain-of-retrieval loop has three moving parts that talk to each other over HTTP, not in-process:

- **`src/inference/run_inference.py`** — top-level driver. Loads the `corag/multihopqa` HF dataset
  for the requested task/split, builds a `CoRagAgent` + `VllmClient`, fans out over examples with a
  `ThreadPoolExecutor`, and writes `preds_{decode_strategy}_{task}_{split}.jsonl` +
  `metrics_{task}_{split}_{decode_strategy}.json` to `output_dir`.

- **`src/agent/corag_agent.py`** (`CoRagAgent`) — the core algorithm:
  - `sample_path()`: greedy loop — generate subquery → retrieve via `search_by_http` → generate
    subanswer → append to a `RagPath` (`src/agent/agent_utils.py`).
  - `tree_search()` / `_search()`: beam-search-style expansion, scoring partial states by the
    prompt-logprob of "No relevant information found" as a value heuristic
    (`_eval_state_without_answer`, `_eval_single_path`).
  - `best_of_n()`: samples N candidate paths, picks the best by that same scoring heuristic.
  - `generate_final_answer()`: produces the final answer from the accumulated path (+ retrieved
    docs), using prompt builders from `src/prompts.py`.

- **`src/vllm_client.py`** (`VllmClient`) — thin OpenAI-client wrapper around the locally-served
  vLLM model (`localhost:8000`, api key `token-123`); tracks total tokens consumed via
  `AtomicCounter`.

- **`src/search/`** — the retrieval side, served as its own process:
  - `start_e5_server_main.py` — Starlette app exposing `POST /`; batches concurrent queries (up to
    64, or 1ms timeout) before calling into `E5Searcher`. Config via env vars `INDEX_DIR`,
    `E5_MODEL_NAME_OR_PATH`, `TOP_K`.
  - `e5_searcher.py` (`E5Searcher`) — loads sharded E5 embedding tensors across all visible GPUs,
    does brute-force top-k similarity search sharded by GPU.
  - `simple_encoder.py` / `model_config.py` — wraps a HF `AutoModel` (default
    `intfloat/e5-large-v2`) for query encoding, with pooling/prefix strategy looked up per model
    name.
  - `search_utils.py` (`search_by_http`) — client used by the agent to call this server.

- **`src/data_utils.py`** — `load_corpus()` (pulls `corag/kilt-corpus` from HF, used both by the
  search server's document lookup and by the agent for formatting), plus context
  truncation/ordering logic keyed on `context_placement`.

- **`src/inference/metrics.py`** + **`src/inference/qa_utils.py`** — EM/F1 scoring
  (`compute_metrics_dict` dispatches on `eval_metrics`: `em_and_f1` is implemented here; `kilt` is
  explicitly *not* implemented and requires a separate script not present in this repo).

Everything is configured through CLI flags (`src/config.py`) or shell-script env vars — there are
no YAML/JSON experiment config files anywhere in this repo.

## Data

No local `data/` directory is checked in — it's created at runtime:
- `data/e5-large-index/` — embedding shards from `scripts/download_embeddings.sh`
  (HF dataset `corag/kilt-corpus-embeddings`).
- `data/log.txt` — created by `src/logger_config.py`.

Datasets/corpus are pulled live from HuggingFace Hub (`corag/multihopqa`, `corag/kilt-corpus`), not
stored locally.
