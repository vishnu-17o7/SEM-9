"""
Setup Multi-Factor Authentication (MFA) system.

Creates user profiles with:
- Face embeddings (synthetic)
- TOTP secrets for OTP generation
- SQLite database for user profiles
"""

import base64
import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np
import pyotp
import qrcode

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "mfa_users.db"
EMBEDDING_DIM = 128

REGISTERED_USERS = [
    {"name": "Alice Johnson", "role": "Admin", "email": "alice@company.com"},
    {"name": "Bob Smith", "role": "Engineer", "email": "bob@company.com"},
    {"name": "Carol Davis", "role": "HR", "email": "carol@company.com"},
]


def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            email TEXT UNIQUE,
            totp_secret TEXT NOT NULL,
            embedding BLOB,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS auth_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            face_verified INTEGER DEFAULT 0,
            otp_verified INTEGER DEFAULT 0,
            authenticated INTEGER DEFAULT 0,
            ip_address TEXT DEFAULT '',
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    conn.commit()
    return conn


def generate_face_embedding(seed_name):
    """Generate a synthetic face embedding for a user."""
    rng = np.random.RandomState(hashlib.md5(seed_name.encode()).digest()[0])
    emb = rng.randn(EMBEDDING_DIM).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb.tobytes()


def setup_users(conn):
    """Register users with face embeddings and TOTP secrets."""
    print("  Setting up user profiles...")
    for user in REGISTERED_USERS:
        user_id = hashlib.md5(user["name"].encode()).hexdigest()[:12]

        # Check if already exists
        existing = conn.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if existing:
            print(f"    {user['name']:20s} — already registered")
            continue

        # Generate TOTP secret
        mfa_key = os.environ.get("MFA_SECRET", "setup-value")
        totp = pyotp.TOTP(mfa_key)

        # Generate face embedding
        embedding_bytes = generate_face_embedding(user["name"])

        conn.execute("""
            INSERT INTO users (user_id, name, role, email, totp_secret, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, user["name"], user["role"], user["email"], _secret, embedding_bytes))

        # Generate QR code for OTP setup
        provisioning_uri = totp.provisioning_uri(name=user["email"], issuer_name="CTI MFA System")
        qr_dir = DATA_DIR / "qr_codes"
        qr_dir.mkdir(parents=True, exist_ok=True)
        qr = qrcode.make(provisioning_uri)
        qr.save(str(qr_dir / f"{user_id}_otp.png"))

        print(f"    {user['name']:20s} — registered (TOTP secret: {mfa_key[:8]}...)")
        print(f"    QR: {qr_dir / f'{user_id}_otp.png'}")

    conn.commit()


def create_impostor_embeddings():
    """Create impostor face embeddings for testing."""
    imp_dir = DATA_DIR / "impostors"
    imp_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        np.save(imp_dir / f"impostor_{i}.npy", emb)
    print(f"  Created 5 impostor embeddings in {imp_dir}")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 50)
    print("  Multi-Factor Auth — Setup")
    print("=" * 50)

    conn = init_db()
    setup_users(conn)
    create_impostor_embeddings()

    # Summary
    cursor = conn.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    print(f"\n  Total registered users: {user_count}")
    print("  Setup complete.")

    conn.close()


if __name__ == "__main__":
    main()
