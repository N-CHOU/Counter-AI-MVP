"""
Counter AI MVP backend (v2 with KYC).

Endpoints:
  GET  /api/health           - Sanity check + model metadata
  POST /api/score            - Score a batch of transactions
  POST /api/explain          - Per-prediction feature contributions
  POST /api/fit-schema       - Re-derive vocab from a training CSV

  KYC module:
  POST /api/kyc/onboard      - Onboard a new customer (returns risk profile)
  POST /api/kyc/screen       - Re-screen an existing customer
  GET  /api/kyc/customers    - List all KYC entities
  GET  /api/kyc/customer/<id> - Get full profile + transaction history
  POST /api/kyc/cdd          - Trigger CDD review

The KYC entity store is an in-memory dict that persists for the server's lifetime.
Production would back this with Postgres; in-memory is fine for the pilot demo.
"""
import os
import json
import io
import csv
import uuid
import hashlib
import random
from datetime import datetime, timedelta
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
print(f"[Counter AI] Schema loaded with {len(SCHEMA['blocks'])} categorical blocks")

N_FEATS = SCHEMA["n_features"]
AMOUNT_IDX = SCHEMA["amount_index"]
AMT_MEAN = SCHEMA["amount_standardization"]["mean"]
AMT_STD = SCHEMA["amount_standardization"]["std"]

# In-memory KYC entity store. Production: replace with Postgres.
KYC_STORE = {}
CDD_QUEUE = []


# Detection engine
def encode_row(tx):
    x = np.zeros(N_FEATS, dtype=np.float32)
    amt = float(tx.get("Amount", 0) or 0)
    x[AMOUNT_IDX] = (amt - AMT_MEAN) / AMT_STD
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
            pass
    return x


def score_batch(txs):
    if not txs:
        return np.array([])
    X = np.vstack([encode_row(tx) for tx in txs])
    return MODEL.predict_proba(X)[:, 1]


def explain_row(tx):
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

    x_alt = x.copy()
    x_alt[0, AMOUNT_IDX] = 0.0
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
        return (f"Risk score {score:.2f}. No single strong indicator — the model assigned moderate weight "
                f"across multiple weak signals.")
    parts = []
    for c in pos:
        if c["feature"] == "Amount":
            parts.append(f"transaction amount {c['value']}")
        else:
            parts.append(f"{c['feature'].replace('_', ' ').lower()} {c['value']}")
    return (f"Flagged because: {'; '.join(parts)}. "
            f"Sender {tx.get('Sender_account','unknown')} ({tx.get('Sender_bank_location','?')}) → "
            f"receiver {tx.get('Receiver_account','unknown')} ({tx.get('Receiver_bank_location','?')}), "
            f"payment type {tx.get('Payment_type','?')}.")


def build_counterfactual(tx, contribs):
    if not contribs:
        return None
    top = contribs[0]
    if top["feature"] == "Amount":
        return f"Would not have been flagged if the amount were closer to the customer's typical value (~{AMT_MEAN:,.0f})."
    if top["feature"] in ("Sender_bank_location", "Receiver_bank_location"):
        return f"Would not have been flagged if {top['feature'].replace('_',' ').lower()} were not '{top['value']}'."
    if top["feature"] == "Payment_type":
        return f"Would not have been flagged for the same parties via a lower-risk payment type (e.g. ACH)."
    return f"Would not have been flagged if {top['feature'].replace('_',' ').lower()} were different."


# KYC module
def screen_customer(full_name, dob, nationality, occupation, industry, country):
    """Simulate sanctions, PEP, and adverse-media screening."""
    hits = []
    name_lower = (full_name or "").lower()
    occ_lower = (occupation or "").lower()
    ind_lower = (industry or "").lower()

    for kw in SCHEMA.get("sanctions_keywords", []):
        if kw in name_lower:
            hits.append({"list": "OFAC SDN", "match": kw, "severity": "critical"})

    if any(kw in occ_lower for kw in SCHEMA.get("pep_keywords", [])):
        hits.append({"list": "PEP", "match": occupation, "severity": "high"})

    if country in SCHEMA.get("high_risk_jurisdictions", []):
        hits.append({"list": "FATF high-risk jurisdiction", "match": country, "severity": "medium"})

    if industry in SCHEMA.get("high_risk_industries", []):
        hits.append({"list": "Internal high-risk industry list", "match": industry, "severity": "medium"})

    # Demo: deterministic adverse-media simulation based on name hash
    h = int(hashlib.md5(name_lower.encode()).hexdigest(), 16) if name_lower else 0
    if name_lower and h % 17 == 0:
        hits.append({"list": "Adverse media", "match": "Negative news article (simulated)", "severity": "low"})

    return hits


def compute_risk_score(hits, country, industry):
    """Map screening hits + risk factors → risk score and tier."""
    score = 0.1
    for h in hits:
        score += {"critical": 0.6, "high": 0.35, "medium": 0.15, "low": 0.05}.get(h["severity"], 0.0)
    if country in SCHEMA.get("high_risk_jurisdictions", []):
        score += 0.05
    if industry in SCHEMA.get("high_risk_industries", []):
        score += 0.05
    score = min(score, 0.99)
    if score < 0.30:
        tier = "Low"
    elif score < 0.60:
        tier = "Medium"
    elif score < 0.85:
        tier = "High"
    else:
        tier = "Critical"
    return score, tier


def derive_cdd_track(tier):
    return {
        "Low": "Simplified Due Diligence (SDD)",
        "Medium": "Standard Due Diligence",
        "High": "Enhanced Due Diligence (EDD)",
        "Critical": "Enhanced Due Diligence (EDD) + senior management approval",
    }.get(tier, "Standard Due Diligence")


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
        "kyc_customers": len(KYC_STORE),
        "cdd_queue_size": len(CDD_QUEUE),
    })


@app.route("/api/score", methods=["POST"])
def score_endpoint():
    data = request.get_json(force=True)
    txs = data.get("transactions", [])
    if not isinstance(txs, list):
        return jsonify({"error": "transactions must be a list"}), 400
    probs = score_batch(txs)
    out = [{"id": tx.get("id"), "score": float(p), "prediction": int(p >= 0.5)}
           for tx, p in zip(txs, probs)]
    return jsonify({"results": out, "n_scored": len(out)})


@app.route("/api/explain", methods=["POST"])
def explain_endpoint():
    data = request.get_json(force=True)
    tx = data.get("transaction")
    if not tx:
        return jsonify({"error": "transaction required"}), 400
    return jsonify(explain_row(tx))


# KYC endpoints
@app.route("/api/kyc/onboard", methods=["POST"])
def kyc_onboard():
    data = request.get_json(force=True)
    required = ["full_name", "dob", "nationality", "country", "id_type", "id_number"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        return jsonify({"error": f"missing fields: {', '.join(missing)}"}), 400

    cust_id = "CUS" + str(uuid.uuid4())[:8].upper()
    hits = screen_customer(
        data.get("full_name"), data.get("dob"), data.get("nationality"),
        data.get("occupation"), data.get("industry"), data.get("country")
    )
    risk_score, tier = compute_risk_score(hits, data.get("country"), data.get("industry"))
    cdd_track = derive_cdd_track(tier)

    customer = {
        "customer_id": cust_id,
        "full_name": data["full_name"],
        "dob": data["dob"],
        "nationality": data["nationality"],
        "country": data["country"],
        "id_type": data["id_type"],
        "id_number": data["id_number"],
        "occupation": data.get("occupation", ""),
        "industry": data.get("industry", ""),
        "source_of_funds": data.get("source_of_funds", ""),
        "expected_monthly_volume": data.get("expected_monthly_volume", 0),
        "onboarded_at": datetime.utcnow().isoformat() + "Z",
        "screening_hits": hits,
        "risk_score": risk_score,
        "risk_tier": tier,
        "cdd_track": cdd_track,
        "status": "active" if not any(h["severity"] == "critical" for h in hits) else "blocked",
        "review_due": (datetime.utcnow() + timedelta(days=365 if tier == "Low" else 180 if tier == "Medium" else 90)).isoformat() + "Z",
        "transactions": [],
        "history": [
            {"ts": datetime.utcnow().isoformat() + "Z",
             "action": "onboarded",
             "actor": "system",
             "detail": f"Risk tier: {tier} · CDD track: {cdd_track}"}
        ]
    }

    if customer["status"] == "blocked":
        customer["history"].append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "blocked",
            "actor": "system",
            "detail": "Critical screening hit — onboarding blocked, escalated to compliance."
        })

    KYC_STORE[cust_id] = customer
    return jsonify(customer)


@app.route("/api/kyc/customers")
def kyc_list():
    out = []
    for c in KYC_STORE.values():
        out.append({
            "customer_id": c["customer_id"],
            "full_name": c["full_name"],
            "country": c["country"],
            "industry": c.get("industry", ""),
            "risk_tier": c["risk_tier"],
            "risk_score": c["risk_score"],
            "status": c["status"],
            "onboarded_at": c["onboarded_at"],
            "screening_hits": len(c["screening_hits"]),
            "n_transactions": len(c.get("transactions", [])),
        })
    out.sort(key=lambda x: x["onboarded_at"], reverse=True)
    return jsonify({"customers": out})


@app.route("/api/kyc/customer/<cust_id>")
def kyc_get(cust_id):
    c = KYC_STORE.get(cust_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    return jsonify(c)


@app.route("/api/kyc/screen", methods=["POST"])
def kyc_rescreen():
    data = request.get_json(force=True)
    cust_id = data.get("customer_id")
    c = KYC_STORE.get(cust_id)
    if not c:
        return jsonify({"error": "customer not found"}), 404

    hits = screen_customer(
        c["full_name"], c["dob"], c["nationality"],
        c.get("occupation"), c.get("industry"), c["country"]
    )
    risk_score, tier = compute_risk_score(hits, c["country"], c.get("industry"))
    old_tier = c["risk_tier"]
    c["screening_hits"] = hits
    c["risk_score"] = risk_score
    c["risk_tier"] = tier
    c["cdd_track"] = derive_cdd_track(tier)
    c["history"].append({
        "ts": datetime.utcnow().isoformat() + "Z",
        "action": "re-screened",
        "actor": data.get("actor", "analyst"),
        "detail": f"Risk tier: {old_tier} → {tier}"
    })
    return jsonify(c)


@app.route("/api/kyc/cdd", methods=["POST"])
def kyc_cdd():
    data = request.get_json(force=True)
    cust_id = data.get("customer_id")
    reason = data.get("reason", "manual review")
    c = KYC_STORE.get(cust_id)
    if not c:
        return jsonify({"error": "customer not found"}), 404
    c["history"].append({
        "ts": datetime.utcnow().isoformat() + "Z",
        "action": "CDD triggered",
        "actor": data.get("actor", "analyst"),
        "detail": reason,
    })
    CDD_QUEUE.append({"customer_id": cust_id, "ts": datetime.utcnow().isoformat() + "Z", "reason": reason})
    return jsonify({"status": "queued", "customer": c})


@app.route("/api/kyc/link-transactions", methods=["POST"])
def kyc_link():
    """Link a batch of scored transactions to KYC customers by Sender_account match."""
    data = request.get_json(force=True)
    txs = data.get("transactions", [])
    linked = 0
    for tx in txs:
        sender = tx.get("Sender_account", "")
        for c in KYC_STORE.values():
            if c.get("linked_account") == sender or sender in (c.get("linked_accounts") or []):
                c.setdefault("transactions", []).append({
                    "id": tx.get("id"), "Date": tx.get("Date"), "Amount": tx.get("Amount"),
                    "Payment_currency": tx.get("Payment_currency"),
                    "Sender_bank_location": tx.get("Sender_bank_location"),
                    "Receiver_bank_location": tx.get("Receiver_bank_location"),
                    "score": tx.get("score"), "status": tx.get("status"),
                })
                linked += 1
                break
    return jsonify({"linked": linked})


@app.route("/api/kyc/seed-demo", methods=["POST"])
def kyc_seed():
    """Populate KYC store with realistic demo customers."""
    demos = [
        {"full_name": "Sarah Chen", "dob": "1985-03-12", "nationality": "Singaporean", "country": "Singapore",
         "id_type": "NRIC", "id_number": "S8512XXX",
         "occupation": "Software Engineer", "industry": "Technology",
         "source_of_funds": "Salary", "expected_monthly_volume": 15000},
        {"full_name": "Marcus Weber", "dob": "1972-08-23", "nationality": "German", "country": "Germany",
         "id_type": "Passport", "id_number": "C01XXXXX",
         "occupation": "Restaurant owner", "industry": "Cash-intensive",
         "source_of_funds": "Business revenue", "expected_monthly_volume": 80000},
        {"full_name": "Priya Sharma", "dob": "1990-11-04", "nationality": "Indian", "country": "India",
         "id_type": "Aadhaar", "id_number": "XXXX-XXXX-1234",
         "occupation": "Marketing Director", "industry": "Retail",
         "source_of_funds": "Salary + investments", "expected_monthly_volume": 12000},
        {"full_name": "Adekunle Okafor", "dob": "1968-02-19", "nationality": "Nigerian", "country": "Nigeria",
         "id_type": "Passport", "id_number": "A09XXXXX",
         "occupation": "Senator", "industry": "Government",
         "source_of_funds": "Government salary", "expected_monthly_volume": 45000},
        {"full_name": "Yuki Tanaka", "dob": "1982-07-30", "nationality": "Japanese", "country": "Japan",
         "id_type": "MyNumber", "id_number": "XXXX-XXXX-5678",
         "occupation": "Financial Analyst", "industry": "Banking",
         "source_of_funds": "Salary + bonus", "expected_monthly_volume": 25000},
        {"full_name": "Ahmed Al-Rashid", "dob": "1975-05-14", "nationality": "Emirati", "country": "UAE",
         "id_type": "Emirates ID", "id_number": "784-XXXX-XXXX",
         "occupation": "Currency Exchange Owner", "industry": "Money services business",
         "source_of_funds": "Business revenue", "expected_monthly_volume": 200000},
        {"full_name": "Elena Volkov", "dob": "1988-09-21", "nationality": "Russian", "country": "Switzerland",
         "id_type": "Passport", "id_number": "K20XXXXX",
         "occupation": "Investor", "industry": "Cryptocurrency",
         "source_of_funds": "Crypto trading", "expected_monthly_volume": 150000},
        {"full_name": "James Robertson", "dob": "1965-12-08", "nationality": "British", "country": "UK",
         "id_type": "Passport", "id_number": "5XXXXXXX",
         "occupation": "Retired", "industry": "None",
         "source_of_funds": "Pension + savings", "expected_monthly_volume": 8000},
    ]
    seeded = 0
    for d in demos:
        # Avoid re-seeding
        if any(c["full_name"] == d["full_name"] for c in KYC_STORE.values()):
            continue
        with app.test_request_context(json=d):
            kyc_onboard()
            seeded += 1
    return jsonify({"seeded": seeded, "total_customers": len(KYC_STORE)})


@app.route("/api/fit-schema", methods=["POST"])
def fit_schema_endpoint():
    global SCHEMA, AMT_MEAN, AMT_STD
    if "file" not in request.files:
        return jsonify({"error": "file field required"}), 400
    f = request.files["file"]
    text = f.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return jsonify({"error": "empty CSV"}), 400
    new_schema = json.loads(json.dumps(SCHEMA))
    cols = ["Payment_type", "Sender_bank_location", "Receiver_bank_location",
            "Payment_currency", "Received_currency"]
    for col in cols:
        if col in rows[0]:
            uniq = sorted(set(r.get(col, "").strip() for r in rows if r.get(col)))
            new_schema["vocab"][col] = uniq
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
                "std": float(arr.std() or 1.0)
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
    print(f"[Counter AI] Starting server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
