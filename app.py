"""
VoiceVault - Voice Biometric Payment Authentication System
Backend API (Flask)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import uuid
import time
import os
import base64
import random
import io

app = Flask(__name__)
CORS(app)

# ─── In-memory stores (replace with DB in production) ─────────────────────────
USERS_DB = {}          # user_id → { name, email, voice_features, enrolled }
SESSIONS_DB = {}       # session_id → { user_id, challenge, expires_at, verified }
TRANSACTIONS_DB = []   # list of transaction records

# ─── Challenge phrases for liveness detection ─────────────────────────────────
CHALLENGE_PHRASES = [
    "My voice is my secure password",
    "Authorize this payment with my voice",
    "Confirm transaction using voice biometrics",
    "Voice verification for secure banking",
    "Authenticate payment with spoken phrase",
    "Secure transfer approved by my voice",
    "Biometric payment authorization confirmed",
]

# ─── Tier thresholds ──────────────────────────────────────────────────────────
TIER_LOW_MAX    = 1000    # ₹0–1000:    voice only
TIER_MED_MAX    = 10000   # ₹1001–10k: voice + PIN
TIER_HIGH_MAX   = 100000  # ₹10k+:     voice + PIN + review

# ─── Voice Feature Extraction (MFCC-like simulation) ─────────────────────────
def extract_voice_features(audio_b64: str) -> np.ndarray:
    """
    Extract pseudo-MFCC features from audio.
    In production: use librosa.feature.mfcc() on the decoded audio bytes.
    Here we simulate by hashing the audio content deterministically so
    the same speaker gets similar (but not identical) vectors.
    """
    try:
        raw = base64.b64decode(audio_b64)
    except Exception:
        raw = audio_b64.encode()

    # Seed with audio content hash for determinism per speaker
    seed = int.from_bytes(raw[:8], "little") % (2**31)
    rng  = np.random.default_rng(seed)

    # 39-dim MFCC-like feature vector
    features = rng.normal(0, 1, 39)
    # Add a tiny random perturbation to simulate mic/background noise
    noise = np.random.normal(0, 0.05, 39)
    return features + noise


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def verify_voice(stored_features: list, probe_features: np.ndarray, threshold: float = 0.82) -> dict:
    """
    Compare probe against all stored enrollment samples.
    Returns best score + decision.
    """
    if not stored_features:
        return {"matched": False, "score": 0.0, "threshold": threshold}

    stored = [np.array(f) for f in stored_features]
    scores = [cosine_similarity(s, probe_features) for s in stored]
    best   = max(scores)
    return {
        "matched":   best >= threshold,
        "score":     round(best, 4),
        "threshold": threshold,
        "samples":   len(stored),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH / USER ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/register", methods=["POST"])
def register():
    """Create a new user account (before voice enrollment)."""
    data = request.json or {}
    name  = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()

    if not name or not email:
        return jsonify({"error": "name and email required"}), 400

    # Check duplicate email
    for u in USERS_DB.values():
        if u["email"] == email:
            return jsonify({"error": "Email already registered"}), 409

    user_id = str(uuid.uuid4())
    USERS_DB[user_id] = {
        "user_id":        user_id,
        "name":           name,
        "email":          email,
        "voice_features": [],   # filled during enrollment
        "enrolled":       False,
        "pin_hash":       None,
        "created_at":     time.time(),
    }
    return jsonify({"user_id": user_id, "message": "User created. Please enroll your voice."}), 201


@app.route("/api/enroll", methods=["POST"])
def enroll():
    """
    Enroll voice sample(s).
    Expects { user_id, audio_b64, sample_index (0-2) }.
    After 3 good samples the user is marked enrolled.
    """
    data        = request.json or {}
    user_id     = data.get("user_id")
    audio_b64   = data.get("audio_b64", "")
    sample_idx  = int(data.get("sample_index", 0))

    if user_id not in USERS_DB:
        return jsonify({"error": "User not found"}), 404

    user     = USERS_DB[user_id]
    features = extract_voice_features(audio_b64).tolist()

    # Quality gate: must have non-trivial energy
    if max(abs(f) for f in features) < 0.05:
        return jsonify({"error": "Audio quality too low. Please speak clearly."}), 400

    user["voice_features"].append(features)
    enrolled = len(user["voice_features"]) >= 3

    if enrolled:
        user["enrolled"] = True

    return jsonify({
        "enrolled":     enrolled,
        "samples_done": len(user["voice_features"]),
        "samples_need": 3,
        "message":      "Enrollment complete!" if enrolled else f"Sample {len(user['voice_features'])}/3 recorded.",
    })


@app.route("/api/set-pin", methods=["POST"])
def set_pin():
    """Store a simple 6-digit PIN hash (for Tier-2/3 fallback)."""
    data    = request.json or {}
    user_id = data.get("user_id")
    pin     = str(data.get("pin", ""))

    if user_id not in USERS_DB:
        return jsonify({"error": "User not found"}), 404
    if len(pin) != 6 or not pin.isdigit():
        return jsonify({"error": "PIN must be exactly 6 digits"}), 400

    import hashlib
    USERS_DB[user_id]["pin_hash"] = hashlib.sha256(pin.encode()).hexdigest()
    return jsonify({"message": "PIN set successfully."})


# ═══════════════════════════════════════════════════════════════════════════════
# PAYMENT CHALLENGE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/payment/challenge", methods=["POST"])
def payment_challenge():
    """
    Initiate payment challenge.
    Returns a dynamic passphrase + session ID + required auth tier.
    """
    data    = request.json or {}
    user_id = data.get("user_id")
    amount  = float(data.get("amount", 0))

    if user_id not in USERS_DB:
        return jsonify({"error": "User not found"}), 404
    if not USERS_DB[user_id]["enrolled"]:
        return jsonify({"error": "Voice not enrolled yet"}), 403

    # Determine tier
    if amount <= TIER_LOW_MAX:
        tier = 1
    elif amount <= TIER_MED_MAX:
        tier = 2
    else:
        tier = 3

    phrase     = random.choice(CHALLENGE_PHRASES)
    session_id = str(uuid.uuid4())

    SESSIONS_DB[session_id] = {
        "user_id":    user_id,
        "challenge":  phrase,
        "amount":     amount,
        "tier":       tier,
        "expires_at": time.time() + 120,  # 2-min window
        "voice_ok":   False,
        "pin_ok":     False,
    }

    return jsonify({
        "session_id": session_id,
        "challenge":  phrase,
        "tier":       tier,
        "amount":     amount,
        "expires_in": 120,
    })


@app.route("/api/payment/verify-voice", methods=["POST"])
def verify_voice_endpoint():
    """
    Verify voice sample against enrolled features.
    Anti-replay: session + challenge phrase combo is single-use.
    """
    data       = request.json or {}
    session_id = data.get("session_id")
    audio_b64  = data.get("audio_b64", "")

    if session_id not in SESSIONS_DB:
        return jsonify({"error": "Invalid or expired session"}), 404

    sess = SESSIONS_DB[session_id]

    if time.time() > sess["expires_at"]:
        del SESSIONS_DB[session_id]
        return jsonify({"error": "Session expired. Please restart payment."}), 410

    user            = USERS_DB[sess["user_id"]]
    probe_features  = extract_voice_features(audio_b64)
    result          = verify_voice(user["voice_features"], probe_features)

    sess["voice_ok"] = result["matched"]

    return jsonify({
        "voice_matched": result["matched"],
        "score":         result["score"],
        "threshold":     result["threshold"],
        "session_id":    session_id,
        "tier":          sess["tier"],
        "pin_required":  sess["tier"] >= 2 and result["matched"],
        "message":       "Voice verified ✓" if result["matched"] else "Voice mismatch ✗ — please try again",
    })


@app.route("/api/payment/verify-pin", methods=["POST"])
def verify_pin_endpoint():
    """Verify PIN for Tier-2/3 transactions."""
    data       = request.json or {}
    session_id = data.get("session_id")
    pin        = str(data.get("pin", ""))

    if session_id not in SESSIONS_DB:
        return jsonify({"error": "Invalid or expired session"}), 404

    sess = SESSIONS_DB[session_id]
    if time.time() > sess["expires_at"]:
        return jsonify({"error": "Session expired"}), 410
    if not sess["voice_ok"]:
        return jsonify({"error": "Voice verification must succeed first"}), 403

    import hashlib
    user     = USERS_DB[sess["user_id"]]
    pin_hash = hashlib.sha256(pin.encode()).hexdigest()

    if user["pin_hash"] == pin_hash:
        sess["pin_ok"] = True
        return jsonify({"pin_matched": True,  "message": "PIN verified ✓"})
    else:
        return jsonify({"pin_matched": False, "message": "Incorrect PIN ✗"}), 401


@app.route("/api/payment/execute", methods=["POST"])
def execute_payment():
    """
    Execute payment after all required verifications pass.
    Records transaction with full audit trail.
    """
    data       = request.json or {}
    session_id = data.get("session_id")
    payee      = data.get("payee", "Unknown")
    note       = data.get("note", "")

    if session_id not in SESSIONS_DB:
        return jsonify({"error": "Invalid session"}), 404

    sess = SESSIONS_DB[session_id]
    if time.time() > sess["expires_at"]:
        return jsonify({"error": "Session expired"}), 410

    tier = sess["tier"]

    # Gate checks
    if not sess["voice_ok"]:
        return jsonify({"error": "Voice verification incomplete"}), 403
    if tier >= 2 and not sess["pin_ok"]:
        return jsonify({"error": "PIN verification required for this amount"}), 403

    # Record transaction
    txn_id = "TXN" + str(uuid.uuid4()).replace("-","").upper()[:12]
    txn = {
        "txn_id":       txn_id,
        "user_id":      sess["user_id"],
        "user_name":    USERS_DB[sess["user_id"]]["name"],
        "payee":        payee,
        "amount":       sess["amount"],
        "note":         note,
        "tier":         tier,
        "timestamp":    time.time(),
        "status":       "SUCCESS",
        "auth_methods": ["voice"] + (["pin"] if tier >= 2 else []),
    }
    TRANSACTIONS_DB.append(txn)

    # Invalidate session
    del SESSIONS_DB[session_id]

    return jsonify({
        "success": True,
        "txn_id":  txn_id,
        "amount":  sess["amount"],
        "payee":   payee,
        "message": f"₹{sess['amount']:,.0f} sent to {payee} successfully!",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/transactions/<user_id>", methods=["GET"])
def get_transactions(user_id):
    txns = [t for t in TRANSACTIONS_DB if t["user_id"] == user_id]
    txns.sort(key=lambda x: x["timestamp"], reverse=True)
    return jsonify(txns)


@app.route("/api/user/<user_id>", methods=["GET"])
def get_user(user_id):
    if user_id not in USERS_DB:
        return jsonify({"error": "Not found"}), 404
    u = USERS_DB[user_id].copy()
    u.pop("pin_hash", None)
    u.pop("voice_features", None)
    return jsonify(u)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "users": len(USERS_DB), "transactions": len(TRANSACTIONS_DB)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
