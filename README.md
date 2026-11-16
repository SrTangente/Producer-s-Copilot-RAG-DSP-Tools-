---
title: Producer's Copilot
emoji: 🎛️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Analyze a mix, compare it with references, get tips.
---

# Producer's Copilot

Upload a mix or master and get:

- **Measurements**: integrated loudness (ITU-R BS.1770), loudness range (EBU Tech 3342), 4× oversampled true peak, peak-to-loudness ratio, crest factor, clipping, tonal balance in five bands, spectral centroid, stereo width, L/R correlation, bass-mono check, tempo and key.
- **Comparison** with a reference track you upload, or with a genre loudness profile (streaming, pop, EDM, hip-hop, rock, acoustic, lo-fi).
- **Production tips**, prioritised. Rules flag what the numbers show, the relevant passages are retrieved from a production knowledge base (RAG), and Claude writes concrete advice that cites them. Without an API key the app still works and writes the tips from the rules.

## How it works

```
audio ──► DSP analysis (app/tools.py)
            │
            ├─► compare with reference / genre profile (app/references.py)
            │
            ├─► diagnose: rules turn metrics into findings (app/tips.py)
            │
            ├─► retrieve: TF-IDF search over app/knowledge/*.md (app/knowledge.py)
            │
            └─► write tips: Claude (structured JSON, cites retrieved passages)
                            └─ falls back to rule-based tips if no key / API error
```

Reference tracks are stored as measurements only (never audio), scoped to a random workspace id kept in the browser.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
make install
export ANTHROPIC_API_KEY=...   # optional: enables AI-written tips
make run                       # http://localhost:8000
```

Tests and lint: `make test`, `make lint`. Docker: `make docker` (serves on port 7860).

### Configuration

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Enables Claude-written tips |
| `COPILOT_MODEL` | `claude-opus-5-5` | Model used for tips |
| `LLM_RATE_LIMIT_PER_HOUR` | `20` | AI-tip requests per client IP per hour |
| `MAX_UPLOAD_MB` | `60` | Upload size limit |
| `DATA_DIR` | `/data` if writable, else `./data` | Where reference metrics are stored |

## API

Interactive docs at `/docs`.

| Method | Path | |
|---|---|---|
| `POST` | `/analyze` | Multipart `file` → metrics |
| `POST` | `/tips` | Multipart `file`, optional `reference_id` or `target_id`, `use_llm` → metrics, comparison, findings, tips, sources |
| `GET` | `/targets` | Built-in genre profiles |
| `POST` / `GET` | `/references` | Add / list reference tracks (header `X-Workspace`) |
| `DELETE` | `/references/{id}` | Remove a reference |
| `GET` | `/health` | Status and active tip engine |

## Deploy to Hugging Face Spaces

The repository is a ready-made Docker Space (see the YAML header above).

1. Create a Space with the **Docker** SDK, then push this repo to it:
   `git push https://huggingface.co/spaces/<user>/<space> main`
2. In the Space settings, add the secret `ANTHROPIC_API_KEY` to enable AI tips.
3. Optional: enable persistent storage so reference libraries survive restarts (the app uses `/data` automatically).

Load testing: `locust -f tests/load/locustfile.py --host https://<user>-<space>.hf.space` (it requests rule-based tips only, so it doesn't spend API credits).

## Limits

Measurements are objective; taste isn't. Genre profiles are approximate, key detection can confuse relative major/minor, and tempo detection can land on half or double time. Use the numbers to confirm what you hear.
