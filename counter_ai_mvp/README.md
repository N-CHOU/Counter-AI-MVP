# Counter AI — Working MVP

A live anti-money-laundering detection prototype built around your trained XGBoost (`xgb_new_v2.pkl`). Designed for the Phase 2 panel demo and grounded in the system architecture decisions from the team's GRIP and assignment work.

## What this delivers

A working end-to-end pilot loop, with the **real model in the loop** — not a mockup. Every prediction comes from `xgb_new_v2.pkl` via a Flask API; the frontend is a single HTML file talking to `/api/score` and `/api/explain`.

### Modules (mapped to Q2 system design)

| UI page | Module from your design | What it does |
| --- | --- | --- |
| Dashboard | overview | MVP loop status, KPIs, FP rate vs. industry baseline |
| Ingestion | 6.2.1 ingestion & validation | CSV upload, schema check, KYC enrichment, dedup, PII screen |
| Alert queue | 6.b transaction monitoring | Ranked alerts above threshold, click any row for explainability |
| Alert detail | 6.3.10–6.3.12 explainability | Plain-English reason, counterfactual, ablation-based feature contributions |
| Feedback toggle | Q2(i) human-in-the-loop | TP/FP/Need-context decision + reason code library + free-text |
| Threshold control | AIRM §5.3 reliance | Configurable cut-off, recomputes confusion matrix and per-segment fairness |
| Model registry | AIRM §5.2.4 | Every required inventory attribute, populated from the live model |
| Governance pack | AIRM + FEAT + NIST | Six artifacts (model card, fairness report, audit log, drift, materiality, compensatory test) with regulator-ready export |
| Audit log | AIRM §6.5 | Every event timestamped — append-only, exportable CSV |

## Running it

### macOS / Linux

```bash
chmod +x run.sh
./run.sh
```

### Windows

Double-click `run.bat`.

The launcher creates a Python venv, installs Flask + sklearn + joblib + numpy, and starts the server on port 5000. First launch takes ~30 seconds (mostly model load — 500 trees take time). Subsequent launches are instant.

Open **http://localhost:5000** in your browser.

### Manual install (if you prefer)

```bash
python3 -m venv venv
source venv/bin/activate          # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python3 server.py
```

## Demo flow for the panel

1. **Dashboard** — point out the "model live · 500 trees, 543 features" status pill (top-left). This is the actual `xgb_new_v2.pkl` you trained.
2. Click **Load synthetic demo data** — 300 transactions generated client-side.
3. Go to **Ingestion** — show schema validation passing, preview the data, click **Run detection engine**.
4. Watch the spinner — the frontend POSTs all 300 transactions to `/api/score`, the server runs `rf.predict_proba()` on each, returns scores. Should take 1–3 seconds.
5. **Alert queue** opens automatically. Click any high-risk alert (red pill).
6. Right pane shows: the actual amount, locations, payment type. Below, **plain-English reason** ("flagged because amount X; payment type Y..."), the **counterfactual** ("would not have been flagged if..."), and the **SHAP-style contribution panel** showing each feature's impact. These are computed by the server doing per-feature ablation against the model — real attribution, not made up.
7. **Feedback panel** on the right — choose TP/FP/Need context, pick a reason code, optionally add a note, click Submit. Watch the alert status pill change and a new entry appear in the audit log.
8. **Threshold control** — drag the slider. Watch the confusion matrix and per-segment fairness recalculate live against the labelled `Is_laundering` column.
9. **Governance pack** — six tiles, each tagged to a specific FEAT/AIRM/NIST clause. Click **Export regulator-ready compliance pack** to download the bundle.
10. **Audit log** — every action you took during the demo is logged with timestamp, actor, and details.

## Schema notes

The model expects 543 features after one-hot encoding. The mapping from human-readable categories (e.g. `Payment_type = "Cash Deposit"`) to feature indices is in `schema.json`. Block sizes were derived from the model's tree structure:

- `Payment_type` (7 features, indices 0–6)
- `Sender_account` (330 features, 7–336)
- `Sender_bank_location` (17 features, 337–353)
- `Receiver_account` (149 features, 354–502)
- `Payment_currency` (13 features, 503–515)
- `Received_currency` (13 features, 516–528)
- `Receiver_bank_location_extra` (13 features, 529–541)
- `Amount` standardized (feature 542)

The vocab inside each block uses pandas `get_dummies` default ordering (alphabetical) based on the documented SAML-D categories. If your training pipeline used a different ordering, predictions on new data will be miscalibrated — in that case, drop the **original training CSV** into the **Schema fitting** card on the Model registry page, and the server will rederive vocab and Amount standardization exactly. Existing `xgb_new_v2.pkl` is not retrained — only the input encoding is corrected.

## File layout

```
counter_ai_mvp/
  xgb_new_v2.pkl              your trained model (unchanged)
  schema.json         feature→category mapping, fittable
  server.py           Flask backend with /score, /explain, /fit-schema, /health
  index.html          frontend (single file, no build step)
  requirements.txt    Python deps
  run.sh              Mac/Linux launcher
  run.bat             Windows launcher
  README.md           this file
```

## Troubleshooting

**"Model offline" pill in the UI** — the backend isn't running or hasn't finished loading. Wait 30 seconds after `run.sh`, then refresh the browser.

**Predictions all look similar (~0.5)** — the schema vocab order doesn't match your training data. Use Schema fitting in the Model registry page with your original training CSV. The model itself is fine; only the input encoding is stale.

**Port 5000 already in use** — set a different port: `PORT=8080 python3 server.py`.

**Slow scoring** — 500 trees × N features × M trees is non-trivial. Score in batches (the UI already does this). For production, swap in a faster runtime (e.g. ONNX export of the RF, or `lightgbm` retrain).

## Architecture in one paragraph

The frontend is one HTML file with vanilla JS — no build, no framework. The backend is one Python file (`server.py`) using Flask. The model is loaded once at server startup. Every UI action that needs the model (scoring a batch, explaining a single transaction, fitting the schema) makes a POST request to a JSON API. Audit events are recorded client-side and exportable as CSV. There's no database — for the pilot, this is by design (no PII at rest, no infra to deploy). For production, add Postgres for the audit log, a queue for async scoring, and put it behind a reverse proxy with auth.
