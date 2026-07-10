"""
Generate synthetic user activity data for anomaly detection.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

NUM_USERS = 100
NUM_DAYS = 45
TOTAL_SESSIONS = 30000
ANOMALY_RATIO = 0.08

CMDS = ["ls", "cd", "cat", "grep", "ps", "top", "vim", "nano", "python", "git",
        "ssh", "scp", "curl", "wget", "chmod", "chown", "sudo", "su", "kill",
        "rm", "cp", "mv", "mkdir", "find", "tar", "zip", "unzip", "make",
        "gcc", "npm", "docker", "kubectl", "aws", "gcloud"]
RISKY_CMDS = ["curl", "wget", "sudo", "su", "chmod", "chown", "rm", "kill", "docker"]
SENSITIVE_RESOURCES = ["/etc/passwd", "/etc/shadow", "/etc/ssh", "/root/",
                       "/var/log/auth.log", "/proc/[0-9]/mem",
                       "/home/*/.ssh/id_rsa", "/home/*/.bash_history",
                       "/tmp/malware", "/usr/bin/python3"]


def generate_normal_session(user_id, day, base_date):
    """Generate a normal user session."""
    n_cmds = random.randint(5, 30)
    start_hour = random.uniform(8, 18)
    start = base_date + timedelta(days=day, hours=start_hour, minutes=random.uniform(0, 59))
    session = []
    for _ in range(n_cmds):
        cmd = random.choice(CMDS)
        resource = ""
        duration = random.uniform(0.5, 30)
        session.append({
            "timestamp": (start + timedelta(seconds=random.uniform(0, 3600))).strftime("%Y-%m-%dT%H:%M:%S"),
            "user_id": user_id, "command": cmd, "resource": resource,
            "duration_s": round(duration, 2), "exit_code": 0,
            "hour_of_day": int(start.hour), "day_of_week": day % 7,
            "session_id": f"{user_id}_{day}_{int(start_hour)}",
            "is_anomaly": 0,
        })
    return session


def generate_anomaly_session(user_id, day, base_date):
    """Generate an anomalous user session."""
    anomaly_type = random.choice(["off_hours", "exfiltration", "privilege_abuse",
                                  "mass_delete", "sensitive_access"])
    n_cmds = random.randint(10, 80)
    start_hour = random.choice([random.uniform(0, 5), random.uniform(20, 23)])
    start = base_date + timedelta(days=day, hours=start_hour, minutes=random.uniform(0, 59))
    session = []

    for _ in range(n_cmds):
        if anomaly_type == "off_hours":
            cmd = random.choice(CMDS)
        elif anomaly_type == "exfiltration":
            cmd = random.choice(["scp", "curl", "wget", "python"])
        elif anomaly_type == "privilege_abuse":
            cmd = random.choice(["sudo", "su", "chmod", "chown"])
        elif anomaly_type == "mass_delete":
            cmd = random.choice(["rm", "find", "mv"])
        elif anomaly_type == "sensitive_access":
            cmd = random.choice(["cat", "vim", "nano", "grep", "ssh"])
        else:
            cmd = random.choice(CMDS)

        resource = random.choice(SENSITIVE_RESOURCES) if anomaly_type == "sensitive_access" else ""
        duration = random.uniform(0.1, 60)
        exit_code = random.choice([0, 0, 0, 1, 2, 127])
        session.append({
            "timestamp": (start + timedelta(seconds=random.uniform(0, 7200))).strftime("%Y-%m-%dT%H:%M:%S"),
            "user_id": user_id, "command": cmd, "resource": resource,
            "duration_s": round(duration, 2), "exit_code": exit_code,
            "hour_of_day": int(start.hour), "day_of_week": day % 7,
            "session_id": f"{user_id}_{day}_{int(start_hour)}",
            "is_anomaly": 1,
        })
    return session


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 50); print("  User Activity — Anomaly Data Generator"); print("=" * 50)
    base_date = datetime(2026, 3, 1)
    n_anom = int(TOTAL_SESSIONS * ANOMALY_RATIO); n_norm = TOTAL_SESSIONS - n_anom

    sessions = []
    for _ in tqdm(range(n_norm), desc="Normal sessions"):
        uid = random.randint(0, NUM_USERS - 1); day = random.randint(0, NUM_DAYS - 1)
        sessions.extend(generate_normal_session(f"u_{uid:04d}", day, base_date))
    for _ in tqdm(range(n_anom), desc="Anomaly sessions"):
        uid = random.randint(0, NUM_USERS - 1); day = random.randint(0, NUM_DAYS - 1)
        sessions.extend(generate_anomaly_session(f"u_{uid:04d}", day, base_date))

    df = pd.DataFrame(sessions)
    df = df.sort_values("timestamp").reset_index(drop=True)
    out_path = DATA_DIR / "user_activity_logs.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Total: {len(df)} commands")
    print(f"  Anomaly ratio: {df['is_anomaly'].mean():.1%}")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
