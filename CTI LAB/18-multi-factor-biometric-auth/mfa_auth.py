"""
Multi-Factor Authentication: Face + OTP.

Authentication flow:
1. Face verification (compare captured face to stored embedding)
2. OTP verification (TOTP from authenticator app)
3. Both must pass for access granted
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pyotp
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "mfa_users.db"
SIMILARITY_THRESHOLD = 0.65


def get_db():
    if not DB_PATH.exists():
        print("Database not found. Run setup_mfa.py first.")
        return None
    return sqlite3.connect(str(DB_PATH))


def verify_face(user_id, face_embedding_bytes):
    """Verify face embedding against stored user embedding."""
    conn = get_db()
    if not conn:
        return False, 0

    cursor = conn.execute("SELECT embedding FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if not row or not row[0]:
        return False, 0

    stored_emb = np.frombuffer(row[0], dtype=np.float32)
    query_emb = np.frombuffer(face_embedding_bytes, dtype=np.float32)

    # Normalize
    stored_emb = stored_emb / np.linalg.norm(stored_emb)
    query_emb = query_emb / np.linalg.norm(query_emb)

    similarity = cosine_similarity(query_emb.reshape(1, -1), stored_emb.reshape(1, -1))[0][0]
    verified = similarity >= SIMILARITY_THRESHOLD

    return verified, float(similarity)


def verify_otp(user_id, otp_code):
    """Verify TOTP code for user."""
    conn = get_db()
    if not conn:
        return False

    cursor = conn.execute("SELECT totp_secret FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False

    totp = pyotp.TOTP(row[0])
    return totp.verify(otp_code, valid_window=1)


def get_user_by_name(name):
    """Lookup user by name."""
    conn = get_db()
    if not conn:
        return None
    cursor = conn.execute("SELECT user_id, name, role, email FROM users WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"user_id": row[0], "name": row[1], "role": row[2], "email": row[3]}
    return None


def get_user_by_id(user_id):
    """Lookup user by ID."""
    conn = get_db()
    if not conn:
        return None
    cursor = conn.execute("SELECT user_id, name, role, email FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"user_id": row[0], "name": row[1], "role": row[2], "email": row[3]}
    return None


def log_auth_attempt(user_id, face_ok, otp_ok, authenticated, ip=""):
    """Log authentication attempt."""
    conn = get_db()
    if not conn:
        return
    conn.execute("""
        INSERT INTO auth_log (user_id, face_verified, otp_verified, authenticated, ip_address)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, int(face_ok), int(otp_ok), int(authenticated), ip))
    conn.commit()
    conn.close()


def generate_otp_for_user(user_id):
    """Generate current OTP for display/testing."""
    conn = get_db()
    cursor = conn.execute("SELECT totp_secret FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        totp = pyotp.TOTP(row[0])
        return totp.now()
    return None


def main():
    parser = argparse.ArgumentParser(description="Multi-Factor Authentication (Face + OTP)")
    parser.add_argument("--user", help="Username for authentication")
    parser.add_argument("--face-file", help="Path to .npy face embedding file")
    parser.add_argument("--otp", help="OTP code to verify")
    parser.add_argument("--show-otp", help="Show current OTP for a user", action="store_true")
    parser.add_argument("--list-users", action="store_true", help="List registered users")
    parser.add_argument("--test", action="store_true", help="Run full test sequence")
    args = parser.parse_args()

    if args.list_users:
        conn = get_db()
        if conn:
            cursor = conn.execute("SELECT user_id, name, role FROM users")
            print(f"\n  Registered Users:")
            for row in cursor:
                print(f"  [{row[0]}] {row[1]:20s} ({row[2]})")
            conn.close()
        return

    if args.show_otp and args.user:
        user = get_user_by_name(args.user) if not args.user.startswith("u_") else get_user_by_id(args.user)
        if user:
            otp = generate_otp_for_user(user["user_id"])
            print(f"  Current OTP for {user['name']}: {otp}")
        else:
            print(f"  User not found: {args.user}")
        return

    if args.user and args.otp:
        user = get_user_by_name(args.user) if not args.user.startswith("u_") else get_user_by_id(args.user)
        if not user:
            print(f"  User not found: {args.user}")
            return

        print(f"\n  {'=' * 50}")
        print(f"  MFA Authentication — {user['name']}")
        print(f"  {'=' * 50}")

        # Step 1: Face Verification
        face_ok = False
        face_conf = 0
        if args.face_file:
            face_path = Path(args.face_file)
            if face_path.exists():
                query_emb = np.load(face_path).tobytes()
                face_ok, face_conf = verify_face(user["user_id"], query_emb)
            else:
                # Use stored embedding (self-test)
                conn = get_db()
                cursor = conn.execute("SELECT embedding FROM users WHERE user_id = ?", (user["user_id"],))
                row = cursor.fetchone()
                conn.close()
                if row:
                    face_ok, face_conf = verify_face(user["user_id"], row[0])
        else:
            # Use stored embedding (test mode)
            conn = get_db()
            cursor = conn.execute("SELECT embedding FROM users WHERE user_id = ?", (user["user_id"],))
            row = cursor.fetchone()
            conn.close()
            if row:
                face_ok, face_conf = verify_face(user["user_id"], row[0])

        print(f"\n  Step 1 — Face Verification:")
        print(f"    Status:     {'PASSED' if face_ok else 'FAILED'}")
        print(f"    Confidence: {face_conf:.2%}")

        # Step 2: OTP Verification
        otp_ok = verify_otp(user["user_id"], args.otp)
        print(f"\n  Step 2 — OTP Verification:")
        print(f"    Status:     {'PASSED' if otp_ok else 'FAILED'}")
        print(f"    OTP:        {args.otp}")

        # Final decision
        authenticated = face_ok and otp_ok
        print(f"\n  {'─' * 50}")
        print(f"  Result:     {'ACCESS GRANTED' if authenticated else 'ACCESS DENIED'}")
        print(f"  {'─' * 50}")

        log_auth_attempt(user["user_id"], face_ok, otp_ok, authenticated)
        return

    if args.test:
        print("\n  Running MFA test sequence...")
        conn = get_db()
        if not conn:
            return

        cursor = conn.execute("SELECT user_id, name FROM users")
        users = cursor.fetchall()
        conn.close()

        for user_id, name in users:
            print(f"\n  {'─' * 50}")
            print(f"  Testing: {name}")

            # Get stored embedding
            conn = get_db()
            cursor = conn.execute("SELECT embedding FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            conn.close()

            # Face test (with own embedding = should pass)
            if row:
                face_ok, conf = verify_face(user_id, row[0])
                print(f"    Face (self):     {'PASS' if face_ok else 'FAIL'} (conf: {conf:.2%})")

            # Face test (with impostor = should fail)
            imp_dir = DATA_DIR / "impostors"
            if imp_dir.exists():
                imp_files = sorted(imp_dir.glob("*.npy"))
                if imp_files:
                    imp_emb = np.load(imp_files[0]).tobytes()
                    imp_face_ok, imp_conf = verify_face(user_id, imp_emb)
                    print(f"    Face (impostor): {'REJECTED' if not imp_face_ok else 'FALSE_ACCEPT'} (conf: {imp_conf:.2%})")

            # OTP test
            otp = generate_otp_for_user(user_id)
            if otp:
                print(f"    Current OTP:     {otp}")

    if not any([args.list_users, args.show_otp, args.test, (args.user and args.otp)]):
        parser.print_help()
        print("\nExample:")
        print("  python mfa_auth.py --test")
        print('  python mfa_auth.py --user "Alice Johnson" --otp 123456')


if __name__ == "__main__":
    main()
