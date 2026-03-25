# 🔐 VoiceVault — Voice Biometric Payment Authentication System

---

## 📌 Problem Solved

Traditional payment systems rely on passwords, OTPs, and PINs — all of which can be:
- Stolen via phishing
- Intercepted via SIM swapping (OTP)
- Guessed or brute-forced

**VoiceVault replaces these with voice biometrics**, making payments authenticatable only by the actual account holder's voice.

---

## ✅ Key Fixes from Original Idea

| Loophole | Fix Applied |
|----------|-------------|
| Voice recording replay attack | Dynamic challenge phrase changes every session |
| Single-factor weakness | Tiered auth: Tier 2 & 3 require PIN too |
| Bad voice enrollment = bad matching | 3-sample enrollment with quality check |
| Deepfake / spoofed voice | Liveness: user must repeat a new random phrase |
| No fallback if user is sick | Confidence threshold + retry allowed |

---

## 🏗️ Architecture

```
voicevault/
├── backend/
│   ├── app.py              ← Flask REST API (all logic here)
│   └── requirements.txt
└── frontend/
    └── index.html          ← Single-file React-free SPA
```

---

## 🔄 Flow (Step-by-Step)

```
Register → Enroll Voice (3 samples) → Set PIN → Make Payment
                                                      ↓
                                          GET /payment/challenge
                                          (server issues random phrase)
                                                      ↓
                                          User speaks phrase
                                          POST /payment/verify-voice
                                          (MFCC extraction + cosine similarity)
                                                      ↓
                                    ┌─────────────────────────────────┐
                                    │         Amount Tier              │
                                    │  ≤ ₹1k  → Voice only           │
                                    │  ≤ ₹10k → Voice + PIN          │
                                    │  > ₹10k → Voice + PIN + Review │
                                    └─────────────────────────────────┘
                                                      ↓
                                          POST /payment/execute
                                          (transaction recorded)
```

---

## 🚀 Running the App

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
# API runs at http://localhost:5000
```

### Frontend

```bash
cd frontend
# Just open index.html in a browser — no build step needed
# Or serve with: python -m http.server 3000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/register` | Create user account |
| POST | `/api/enroll` | Add voice sample (call 3 times) |
| POST | `/api/set-pin` | Set 6-digit PIN |
| POST | `/api/payment/challenge` | Get challenge phrase + session |
| POST | `/api/payment/verify-voice` | Submit voice for verification |
| POST | `/api/payment/verify-pin` | Submit PIN (Tier 2/3) |
| POST | `/api/payment/execute` | Execute verified payment |
| GET  | `/api/transactions/:id` | Get user's transaction history |
| GET  | `/api/user/:id` | Get user profile |

---

## 🧠 ML / Signal Processing

### Current (Demo Mode)
- Audio → Base64
- Base64 → deterministic seed → numpy pseudo-MFCC (39 features)
- Cosine similarity against enrolled samples
- Threshold: 0.82

### Production Upgrade Path
```python
# Replace extract_voice_features() in app.py with:
import librosa

def extract_voice_features(audio_b64):
    audio_bytes = base64.b64decode(audio_b64)
    audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc.mean(1), delta.mean(1), delta2.mean(1)])
    return features
```

---

## 🛡️ Security Features

1. **Anti-replay**: New random phrase per session (120-second TTL)
2. **Liveness detection**: Must repeat displayed phrase (not a recording)
3. **Tiered authentication**: Higher amounts = more factors
4. **Multi-sample enrollment**: 3 samples for robustness
5. **Single-use sessions**: Session deleted after payment executes
6. **PIN hashing**: SHA-256 stored, never plaintext
7. **Quality gate**: Low-energy audio rejected at enrollment

---

## 📊 Viva Talking Points

1. **Why voice over OTP?** — OTP requires phone access; voice is always with you
2. **How does MFCC work?** — Divides audio into time frames, applies mel-scale filterbank, takes cosine transform → speaker-unique fingerprint
3. **What is cosine similarity?** — Measures angle between feature vectors; same speaker → vectors point in similar direction
4. **Why 3 enrollment samples?** — Averages out background noise, mic variation, speaking variation
5. **Can voice be deepfaked?** — Yes, but we mitigate with liveness (random phrase) + confidence threshold + tier escalation
6. **Why tiered auth?** — Risk-proportional security; low-risk transactions stay frictionless

---

## 🔮 Future Scope

- [ ] Replace simulation with real librosa MFCC
- [ ] Add anti-spoofing CNN model (trained on ASVspoof dataset)
- [ ] Store voice templates in encrypted DB (PostgreSQL + pgcrypto)
- [ ] JWT-based authentication
- [ ] React Native mobile app
- [ ] Integration with actual payment gateway (Razorpay / Stripe)
- [ ] Continuous authentication during session

---

## 👨‍💻 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | HTML5, CSS3, Vanilla JS (no framework) |
| Backend | Python, Flask, Flask-CORS |
| ML/DSP | NumPy (+ librosa in production) |
| Auth | Voice biometrics + SHA-256 PIN |
| Protocol | REST JSON API |
