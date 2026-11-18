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

The audio is read block by block, and only the middle two minutes are held in memory for tempo and key, so the app fits on a 512 MB instance. In the web app, reference tracks are kept as measurements in the browser; the audio is never stored.

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
| `DATA_DIR` | `/data` if writable, else `./data` | Where the `/references` API stores metrics |

## API

Interactive docs at `/docs`.

| Method | Path | |
|---|---|---|
| `POST` | `/analyze` | Multipart `file` → metrics |
| `POST` | `/tips` | Multipart `file`, optional `target_id`, `reference_id` or `reference_json` + `reference_name`, and `use_llm` → metrics, comparison, findings, tips, sources |
| `GET` | `/targets` | Built-in genre profiles |
| `POST` / `GET` | `/references` | Add / list reference tracks (header `X-Workspace`) |
| `DELETE` | `/references/{id}` | Remove a reference |
| `GET` | `/health` | Status and active tip engine |

## Deploy for free on Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/SrTangente/Producer-s-Copilot-RAG-DSP-Tools-)

The button reads `render.yaml` and builds the Dockerfile on Render's free plan (512 MB, 0.1 CPU). On that plan the service sleeps after 15 minutes without traffic, the first visit afterwards takes about a minute, and each analysis is slow because of the small CPU share.

The app works without any API key, using rule-based tips. To have Claude write them, add `ANTHROPIC_API_KEY` under *Environment* in Render; API usage is billed by Anthropic.

Load testing: `locust -f tests/load/locustfile.py --host https://<your-app>.onrender.com` (it requests rule-based tips only, so it doesn't spend API credits).

## Limits

Measurements are objective; taste isn't. Genre profiles are approximate, key detection can confuse relative major/minor, and tempo detection can land on half or double time. Use the numbers to confirm what you hear.
