"""
Counter AI MVP backend.

Loads the trained random forest (rf.pkl) and exposes:
  POST /api/score   - Score a batch of transactions
  POST /api/explain - SHAP-style feature contributions for one transaction
  POST /api/fit-schema - Re-derive vocab from a training CSV (optional)
  GET  /api/health  - Sanity check and model metadata

The schema mapping (which feature index corresponds to which category)
is loaded from schema.json. If predictions look off, run fit_schema.py
on the original training CSV to regenerate it exactly.
"""
import os
import json
import io
import csv
from datetime import datetime
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "rf.pkl")
SCHEMA_PATH = os.path.join(ROOT, "schema.json")

print("[Counter AI] Loading model from", MODEL_PATH)
MODEL = joblib.load(MODEL_PATH)
print(f"[Counter AI] Loaded {type(MODEL).__name__} with {MODEL.n_estimators} trees, "
      f"{MODEL.n_features_in_} features")

with open(SCHEMA_PATH) as f:
    SCHEMA = json.load(f)
print(f"[Counter AI] Schema loaded: {list(SCHEMA['blocks'].keys())}")

N_FEATS = SCHEMA["n_features"]
AMOUNT_IDX = SCHEMA["amount_index"]
AMT_MEAN = SCHEMA["amount_standardization"]["mean"]
AMT_STD = SCHEMA["amount_standardization"]["std"]


def encode_row(tx):
    """Encode a single transaction dict into the 543-dim feature vector."""
    x = np.zeros(N_FEATS, dtype=np.float32)

    # Numeric: standardized amount
    amt = float(tx.get("Amount", 0) or 0)
    x[AMOUNT_IDX] = (amt - AMT_MEAN) / AMT_STD

    # Categoricals: one-hot using vocab order
    for col, block in SCHEMA["blocks"].items():
        if col.startswith("_") or "extra" in col:
            continue
        vocab = SCHEMA["vocab"].get(col)
        if vocab is None:
            continue
        val = tx.get(col)
        if val is None:
            continue
        try:
            i = vocab.index(val)
            if i < block["size"]:
                x[block["start"] + i] = 1.0
        except ValueError:
            # Unknown category: leave the entire block at zero
            pass
    return x


def score_batch(txs):
    """Score a list of transaction dicts. Returns probabilities for class=1."""
    if not txs:
        return np.array([])
    X = np.vstack([encode_row(tx) for tx in txs])
    probs = MODEL.predict_proba(X)[:, 1]
    return probs


def explain_row(tx):
    """
    Approximate per-prediction feature contributions without the SHAP package.
    Strategy: for each set feature in the input vector, compute the model's
    marginal effect by ablating it (setting to 0) and measuring the change
    in predicted probability. Cheap, deterministic, and correctly attributes
    to whichever ones the tree ensemble actually used.
    """
    x = encode_row(tx).reshape(1, -1)
    base = MODEL.predict_proba(x)[0, 1]

    contribs = []
    for col, block in SCHEMA["blocks"].items():
        if col.startswith("_") or "extra" in col:
            continue
        s, sz = block["start"], block["size"]
        active = [(i, x[0, i]) for i in range(s, s + sz) if x[0, i] != 0]
        for i, v in active:
            x_alt = x.copy()
            x_alt[0, i] = 0.0
            alt = MODEL.predict_proba(x_alt)[0, 1]
            delta = float(base - alt)
            vocab = SCHEMA["vocab"].get(col, [])
            label = vocab[i - s] if i - s < len(vocab) else f"{col}[{i-s}]"
            contribs.append({
                "feature": col,
                "value": label,
                "weight": delta,
                "type": "pos" if delta > 0 else "neg",
            })

    # Amount contribution: shift to mean (i.e. as if it were a typical amount)
    x_alt = x.copy()
    x_alt[0, AMOUNT_IDX] = 0.0  # standardized 0 = mean
    alt = MODEL.predict_proba(x_alt)[0, 1]
    delta_amt = float(base - alt)
    contribs.append({
        "feature": "Amount",
        "value": f"{tx.get('Amount', 0):,.2f}",
        "weight": delta_amt,
        "type": "pos" if delta_amt > 0 else "neg",
    })

    contribs.sort(key=lambda c: abs(c["weight"]), reverse=True)
    return {
        "score": float(base),
        "contributions": contribs[:8],
        "plain_reason": build_plain_reason(tx, contribs[:8], base),
        "counterfactual": build_counterfactual(tx, contribs[:8]),
    }


def build_plain_reason(tx, contribs, score):
    pos = [c for c in contribs if c["type"] == "pos"][:3]
    if not pos:
        return f"Score {score:.2f}. No strong positive indicators — the model gave moderate weight across multiple weak signals."
    parts = []
    for c in pos:
        if c["feature"] == "Amount":
            parts.append(f"amount {c['value']}")
        else:
            parts.append(f"{c['feature'].replace('_', ' ').lower()} = {c['value']}")
    return (f"Flagged because: {'; '.join(parts)}. "
            f"Sender: {tx.get('Sender_account','?')} ({tx.get('Sender_bank_location','?')}), "
            f"receiver: {tx.get('Receiver_account','?')} ({tx.get('Receiver_bank_location','?')}), "
            f"payment type {tx.get('Payment_type','?')}.")


def build_counterfactual(tx, contribs):
    if not contribs:
        return None
    top = contribs[0]
    if top["feature"] == "Amount":
        amt = float(tx.get("Amount", 0))
        target = AMT_MEAN
        return f"Would not have been flagged if the amount were closer to the typical value (around {target:,.0f})."
    if top["feature"] in ("Sender_bank_location", "Receiver_bank_location"):
        return f"Would not have been flagged if {top['feature'].replace('_',' ').lower()} were not '{top['value']}'."
    if top["feature"] == "Payment_type":
        return f"Would not have been flagged for the same parties via a less risky payment type (e.g. ACH)."
    return f"Would not have been flagged if {top['feature'].replace('_',' ').lower()} were different."


# Flask app
app = Flask(__name__, static_folder=ROOT, static_url_path="")


@app.route("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "rf_aml_v1",
        "n_estimators": MODEL.n_estimators,
        "n_features": MODEL.n_features_in_,
        "schema_blocks": {k: v.get("size", 0) for k, v in SCHEMA["blocks"].items() if not k.startswith("_")},
        "loaded_at": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/score", methods=["POST"])
def score_endpoint():
    data = request.get_json(force=True)
    txs = data.get("transactions", [])
    if not isinstance(txs, list):
        return jsonify({"error": "transactions must be a list"}), 400
    probs = score_batch(txs)
    out = []
    for tx, p in zip(txs, probs):
        out.append({
            "id": tx.get("id"),
            "score": float(p),
            "prediction": int(p >= 0.5),
        })
    return jsonify({"results": out, "n_scored": len(out)})


@app.route("/api/explain", methods=["POST"])
def explain_endpoint():
    data = request.get_json(force=True)
    tx = data.get("transaction")
    if not tx:
        return jsonify({"error": "transaction required"}), 400
    return jsonify(explain_row(tx))


@app.route("/api/fit-schema", methods=["POST"])
def fit_schema_endpoint():
    """
    Optional: derive vocab and standardization from a training CSV.
    POST a CSV file under field 'file'. Updates schema.json on disk
    and reloads in memory. Use to remove guesswork if you have the
    original training data.
    """
    global SCHEMA, AMT_MEAN, AMT_STD
    if "file" not in request.files:
        return jsonify({"error": "file field required"}), 400
    f = request.files["file"]
    text = f.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return jsonify({"error": "empty CSV"}), 400

    new_schema = json.loads(json.dumps(SCHEMA))  # deep copy

    cols = ["Payment_type", "Sender_bank_location", "Receiver_bank_location",
            "Payment_currency", "Received_currency"]
    for col in cols:
        if col in rows[0]:
            uniq = sorted(set(r.get(col, "").strip() for r in rows if r.get(col)))
            new_schema["vocab"][col] = uniq

    # Fit Amount standardization
    if "Amount" in rows[0]:
        amts = []
        for r in rows:
            try:
                amts.append(float(r["Amount"]))
            except Exception:
                pass
        if amts:
            arr = np.array(amts)
            new_schema["amount_standardization"] = {
                "mean": float(arr.mean()),
                "std": float(arr.std() or 1.0),
                "_note": "fitted from uploaded training CSV"
            }

    with open(SCHEMA_PATH, "w") as out:
        json.dump(new_schema, out, indent=2)

    SCHEMA = new_schema
    AMT_MEAN = SCHEMA["amount_standardization"]["mean"]
    AMT_STD = SCHEMA["amount_standardization"]["std"]
    return jsonify({"status": "schema fitted", "n_rows_seen": len(rows),
                    "amount_mean": AMT_MEAN, "amount_std": AMT_STD})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[Counter AI] Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
