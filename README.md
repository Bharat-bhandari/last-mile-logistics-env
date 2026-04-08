# Last-Mile Logistics OpenEnv

Repository for the Last-Mile Logistics Controller environment used in OpenEnv evaluation.

## Structure

- `last_mile_env/`: environment package (server, models, tasks, grader, manifest)
- `Dockerfile`: container build
- `inference.py`: agent evaluator script

## Quick Start

```bash
cd last_mile_env
uv sync
uv run --project . server
```

Server runs on `http://localhost:8000`.

## Task Modes

```bash
export LMLC_TASK=easy   # or medium / hard
```

## Validate Manifest

```bash
cd last_mile_env
openenv validate
```

## Run Inference

```bash
export HF_TOKEN="<token>"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

Logs emitted: `[START]`, `[STEP]`, `[END]`.
