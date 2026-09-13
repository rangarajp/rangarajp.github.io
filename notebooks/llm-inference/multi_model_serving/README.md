# Multi-Model LLM Serving Demo

Serves multiple local causal LMs behind one FastAPI process with an LRU model cache
(default capacity: **2**). Configured for:

| Model ID | Checkpoint |
|----------|------------|
| `qwen2.5-0.5b` | `models/Qwen2.5-0.5B-Instruct` |
| `tinyllama-1.1b` | `models/TinyLlama-1.1B-Chat-v1.0` |

Both paths are resolved separately via gitignored `../local_paths.json`
(copy from `../local_paths.example.json`):

- `MULTI_MODEL_NOTEBOOK_DIR` — serving package (`app/`, `config/`)
- `MODELS_DIR` — local checkpoints

`config/models.json` stores folder names under `MODELS_DIR` (e.g. `Qwen2.5-0.5B-Instruct`), not absolute machine paths.

## Setup

1. Create and activate a virtual environment, then install deps:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy path config and edit for your machine:

```bash
cp ../local_paths.example.json ../local_paths.json
# edit MODELS_DIR and MULTI_MODEL_NOTEBOOK_DIR
```

Place checkpoints under `MODELS_DIR` (`Qwen2.5-0.5B-Instruct`, `TinyLlama-1.1B-Chat-v1.0`).

3. Run the service — either from the CLI or notebooks:

```bash
# default port 8001
python -m app.server

# custom port
# Windows PowerShell:
$env:PORT=8002; python -m app.server
```

Or open `main.ipynb` (starts uvicorn on port 8001 in a background thread), then open `test.ipynb` to exercise the APIs.

## API

### List models

```bash
curl http://localhost:8001/models
```

### Generate (recommended)

```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d "{\"model_id\": \"qwen2.5-0.5b\", \"prompt\": \"Explain KV cache in one sentence.\", \"max_new_tokens\": 64}"
```

```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d "{\"model_id\": \"tinyllama-1.1b\", \"prompt\": \"What is 2+2?\", \"max_new_tokens\": 32}"
```

### Predict (generic)

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d "{\"model_id\": \"qwen2.5-0.5b\", \"input_data\": {\"prompt\": \"Hello\", \"max_new_tokens\": 32}}"
```

`input_data` may also be a plain string prompt.

## Architecture

1. **Server** — HTTP API
2. **Store** — reads `config/models.json`, resolves `path` against the repo root
3. **Manager** — LRU cache of up to 2 loaded workers
4. **Engine** — creates / deletes workers
5. **LLMCausalWorker** — loads `AutoModelForCausalLM` + tokenizer from disk, runs `generate()`

With exactly two registered models and `max_models=2`, both stay resident once warm. Adding a third model to `models.json` would trigger LRU eviction and unload.

## Tests

```bash
python -m unittest tests.test_models
```

Generation tests skip automatically if the local checkpoints are missing.
