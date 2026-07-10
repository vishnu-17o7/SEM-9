# 18 · Multi-Factor Biometric Authentication (Face + OTP)

Develop a secure authentication mechanism by combining facial recognition with one-time password (OTP) verification. Validates user identity through multiple authentication factors.

## Quick Start

```bash
# Run the full pipeline (setup MFA)
python run.py all

# Or run individual steps
python run.py data       # Set up MFA (enroll users, generate TOTP + face embeddings)
python run.py classify   # Run authentication (CLI)
python run.py web        # Launch Flask web UI (port 5003)
```

## Files

| File | Purpose |
|---|---|
| `setup_mfa.py` | Initialize database, user profiles, TOTP secrets, face embeddings |
| `mfa_auth.py` | CLI authentication (face + OTP verification) |
| `app.py` | Flask web UI with authentication flow + audit log |
| `data/mfa_users.db` | SQLite database with user profiles |
| `data/qr_codes/` | TOTP setup QR codes per user |
| `data/impostors/` | Impostor embeddings for testing |

## Authentication Flow

```
1. Face Verification → Cosine similarity against stored 128-dim embedding
2. OTP Verification → TOTP (RFC 6238) 6-digit code
3. Both must pass → Access Granted
4. All attempts logged to audit trail
```

## Setup & Usage

```bash
cd 18-multi-factor-biometric-auth
pip install -r requirements.txt

# 1. Setup
python setup_mfa.py

# 2. CLI test
python mfa_auth.py --test

# 3. Authenticate via CLI
python mfa_auth.py --user "Alice Johnson" --otp 123456

# 4. Web UI
python app.py
# Open http://127.0.0.1:5003
```

## Test Sequence

- Registered users authenticate successfully with their own face + valid OTP
- Impostor faces are rejected even with valid OTP
- Auth log tracks all attempts for audit
