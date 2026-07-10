"""
Flask web UI for Multi-Factor Authentication (Face + OTP).
"""

import base64
import json
import sqlite3
from io import BytesIO
from pathlib import Path

import numpy as np
import pyotp
import qrcode
from flask import Flask, jsonify, render_template_string, request

from mfa_auth import (
    get_db, get_user_by_name, get_user_by_id,
    verify_face, verify_otp, generate_otp_for_user, log_auth_attempt,
)

app = Flask(__name__)
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "mfa_users.db"


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/users")
def api_users():
    conn = get_db()
    if not conn:
        return jsonify({"users": []})
    cursor = conn.execute("SELECT user_id, name, role FROM users")
    users = [{"user_id": row[0], "name": row[1], "role": row[2]} for row in cursor]
    conn.close()
    return jsonify({"users": users})


@app.route("/api/auth", methods=["POST"])
def api_auth():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    user_id = data.get("user_id", "")
    otp = data.get("otp", "")
    use_stored_face = data.get("use_stored_face", True)

    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Face verification (using stored embedding)
    face_ok = False
    face_conf = 0
    if use_stored_face:
        conn = get_db()
        cursor = conn.execute("SELECT embedding FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            face_ok, face_conf = verify_face(user_id, row[0])

    # OTP verification
    otp_ok = verify_otp(user_id, otp)
    authenticated = face_ok and otp_ok

    log_auth_attempt(user_id, face_ok, otp_ok, authenticated)

    return jsonify({
        "user": user,
        "face_verified": face_ok,
        "face_confidence": round(face_conf, 4),
        "otp_verified": otp_ok,
        "authenticated": authenticated,
        "access": "GRANTED" if authenticated else "DENIED",
    })


@app.route("/api/otp/<user_id>")
def api_otp(user_id):
    otp = generate_otp_for_user(user_id)
    if otp:
        return jsonify({"otp": otp})
    return jsonify({"error": "User not found"}), 404


@app.route("/api/qr/<user_id>")
def api_qr(user_id):
    qr_path = DATA_DIR / "qr_codes" / f"{user_id}_otp.png"
    if qr_path.exists():
        with open(qr_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        return jsonify({"qr": f"data:image/png;base64,{img_b64}"})
    return jsonify({"error": "QR not found"}), 404


@app.route("/api/logs")
def api_logs():
    conn = get_db()
    if not conn:
        return jsonify({"logs": []})
    cursor = conn.execute("""
        SELECT a.user_id, u.name, a.timestamp, a.face_verified, a.otp_verified, a.authenticated
        FROM auth_log a LEFT JOIN users u ON a.user_id = u.user_id
        ORDER BY a.timestamp DESC LIMIT 50
    """)
    logs = [{
        "user_id": row[0], "name": row[1], "timestamp": row[2],
        "face": bool(row[3]), "otp": bool(row[4]), "auth": bool(row[5]),
    } for row in cursor]
    conn.close()
    return jsonify({"logs": logs})


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MFA — Face + OTP Authentication</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:2rem}
.container{max-width:800px;margin:0 auto}
h1{font-size:1.5rem;font-weight:700;color:#f8fafc;margin-bottom:.25rem}
.subtitle{color:#94a3b8;font-size:.85rem;margin-bottom:2rem}
.card{background:#1e293b;border-radius:12px;padding:1.5rem;border:1px solid #334155;margin-bottom:1rem}
.card h2{font-size:1rem;color:#f8fafc;margin-bottom:1rem}
label{display:block;font-size:.8rem;color:#94a3b8;margin-bottom:.3rem}
select,input{width:100%;padding:.6rem;background:#0f172a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:.9rem;margin-bottom:1rem}
select option{background:#1e293b}
button{background:#2563eb;color:white;border:none;padding:.6rem 1.5rem;border-radius:6px;cursor:pointer;font-size:.9rem;margin-right:.5rem}
button:hover{background:#1d4ed8}
button.secondary{background:#334155;color:#e2e8f0}
button.secondary:hover{background:#475569}
.result{margin-top:1rem;padding:1rem;border-radius:8px}
.granted{background:#064e3b;border:1px solid #047857;color:#a7f3d0}
.denied{background:#7f1d1d;border:1px solid #b91c1c;color:#fecaca}
.progress{display:flex;gap:1rem;margin:1rem 0}
.step{flex:1;padding:.75rem;border-radius:8px;text-align:center;font-size:.8rem;background:#1e293b;border:1px solid #334155}
.step.active{background:#1e3a5f;border-color:#3b82f6;color:#93c5fd}
.step.pass{background:#064e3b;border-color:#047857;color:#a7f3d0}
.step.fail{background:#7f1d1d;border-color:#b91c1c;color:#fecaca}
table{width:100%;border-collapse:collapse;font-size:.8rem}
th{text-align:left;padding:.5rem .75rem;color:#94a3b8;font-size:.7rem;text-transform:uppercase;border-bottom:1px solid #334155}
td{padding:.5rem .75rem;border-bottom:1px solid #1e293b;color:#cbd5e1}
.status{display:inline-block;padding:2px 6px;border-radius:4px;font-size:.65rem;font-weight:600}
.status-pass{background:#064e3b;color:#a7f3d0}.status-fail{background:#7f1d1d;color:#fecaca}
.qr-img{max-width:200px;margin:.5rem 0}
.otp-display{font-family:monospace;font-size:1.5rem;color:#fbbf24;text-align:center;padding:.5rem;background:#0f172a;border-radius:6px;margin:.5rem 0}
</style>
</head>
<body>
<div class="container">
<h1>Multi-Factor Authentication</h1>
<p class="subtitle">Face Recognition + TOTP One-Time Password</p>

<div class="card">
<h2>Authenticate</h2>
<label>Select User</label>
<select id="userSelect"><option value="">-- Select user --</option></select>

<label>OTP Code (6 digits)</label>
<input type="text" id="otpInput" placeholder="000000" maxlength="6" style="font-family:monospace;font-size:1.2rem;letter-spacing:4px">

<button onclick="authenticate()">Authenticate</button>
<button class="secondary" onclick="showOTP()">Show Current OTP</button>

<div id="otpDisplay" class="otp-display" style="display:none"></div>
<div id="qrDisplay" style="text-align:center;display:none"><img class="qr-img" id="qrImage"><p style="color:#94a3b8;font-size:.8rem">Scan with authenticator app</p></div>
</div>

<div id="result"></div>

<div class="card" style="margin-top:1rem">
<h2>Auth Log</h2>
<div style="max-height:300px;overflow-y:auto">
<table><thead><tr><th>User</th><th>Time</th><th>Face</th><th>OTP</th><th>Auth</th></tr></thead>
<tbody id="logTable"></tbody></table>
</div>
</div>
</div>

<script>
async function loadUsers() {
    const r = await fetch('/api/users');
    const data = await r.json();
    const sel = document.getElementById('userSelect');
    data.users.forEach(u => {
        const opt = document.createElement('option');
        opt.value = u.user_id; opt.textContent = `${u.name} (${u.role})`;
        sel.appendChild(opt);
    });
}

async function authenticate() {
    const userId = document.getElementById('userSelect').value;
    const otp = document.getElementById('otpInput').value;
    if (!userId || otp.length !== 6) { alert('Select user and enter 6-digit OTP'); return; }

    const r = await fetch('/api/auth', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({user_id: userId, otp: otp, use_stored_face: true})
    });
    const data = await r.json();

    // Update result
    const div = document.getElementById('result');
    const cls = data.authenticated ? 'granted' : 'denied';
    div.innerHTML = `
        <div class="result ${cls}">
            <h3>Access ${data.access}</h3>
            <p>User: ${data.user.name} (${data.user.role})</p>
            <div class="progress">
                <div class="step ${data.face_verified ? 'pass' : 'fail'}">Face<br>${data.face_verified ? 'PASS' : 'FAIL'}<br><small>${(data.face_confidence * 100).toFixed(1)}%</small></div>
                <div class="step ${data.otp_verified ? 'pass' : 'fail'}">OTP<br>${data.otp_verified ? 'PASS' : 'FAIL'}</div>
            </div>
        </div>`;

    loadLogs();
}

async function showOTP() {
    const userId = document.getElementById('userSelect').value;
    if (!userId) return;
    const r = await fetch(`/api/otp/${userId}`);
    const data = await r.json();
    document.getElementById('otpDisplay').textContent = data.otp || 'Error';
    document.getElementById('otpDisplay').style.display = 'block';

    // Show QR
    const qrR = await fetch(`/api/qr/${userId}`);
    const qrData = await qrR.json();
    if (qrData.qr) {
        document.getElementById('qrImage').src = qrData.qr;
        document.getElementById('qrDisplay').style.display = 'block';
    }
}

async function loadLogs() {
    const r = await fetch('/api/logs');
    const data = await r.json();
    const tbody = document.getElementById('logTable');
    tbody.innerHTML = data.logs.map(l => `
        <tr>
            <td>${l.name || l.user_id}</td>
            <td>${l.timestamp}</td>
            <td><span class="status ${l.face ? 'status-pass' : 'status-fail'}">${l.face ? 'PASS' : 'FAIL'}</span></td>
            <td><span class="status ${l.otp ? 'status-pass' : 'status-fail'}">${l.otp ? 'PASS' : 'FAIL'}</span></td>
            <td><span class="status ${l.auth ? 'status-pass' : 'status-fail'}">${l.auth ? 'GRANTED' : 'DENIED'}</span></td>
        </tr>
    `).join('');
}

loadUsers();
loadLogs();
setInterval(loadLogs, 5000);
</script>
</body>
</html>
"""


def main():
    print("  MFA Web UI")
    print("  Open http://127.0.0.1:5003")
    app.run(debug=False, host="127.0.0.1", port=5003)


if __name__ == "__main__":
    main()
