"""
Generate synthetic login logs with normal patterns and credential stuffing attacks.

Produces login_logs.csv in data/ — 500,000+ entries, 1,000 users, 60 days.
Attack patterns modelled on real credential stuffing behaviour from:
  - Wiefling et al., ACM TOPS 2022 (large-scale SSO RBA study)
  - OWASP Credential Stuffing Cheat Sheet patterns
  - CIC-IDS2017 Brute Force Web attack characteristics
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NUM_USERS = 1000
NUM_DAYS = 60
TARGET_ENTRIES = 500_000

ROLES = {
    "engineer":   {"weight": 0.28, "login_hours": (7, 20),  "avg_daily": 14, "fail_rate": 0.06},
    "hr":         {"weight": 0.08, "login_hours": (9, 17),  "avg_daily": 7,  "fail_rate": 0.04},
    "finance":    {"weight": 0.08, "login_hours": (8, 18),  "avg_daily": 9,  "fail_rate": 0.03},
    "admin":      {"weight": 0.04, "login_hours": (6, 21),  "avg_daily": 22, "fail_rate": 0.02},
    "exec":       {"weight": 0.04, "login_hours": (7, 21),  "avg_daily": 6,  "fail_rate": 0.05},
    "support":    {"weight": 0.15, "login_hours": (5, 23),  "avg_daily": 18, "fail_rate": 0.07},
    "intern":     {"weight": 0.12, "login_hours": (9, 18),  "avg_daily": 5,  "fail_rate": 0.14},
    "contractor": {"weight": 0.14, "login_hours": (8, 19),  "avg_daily": 8,  "fail_rate": 0.09},
    "sales":      {"weight": 0.07, "login_hours": (8, 19),  "avg_daily": 10, "fail_rate": 0.08},
}

GEO_COUNTRIES = ["US", "UK", "DE", "FR", "JP", "AU", "BR", "IN", "CA", "NL", "SG", "KR", "ES", "IT"]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/120.0",
    "Mozilla/5.0 (X11; Linux x86_64) Firefox/121.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Mobile/15E148",
    "Mozilla/5.0 (iPad; CPU OS 17_0) Mobile/15E148",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) Chrome/119.0",
    "Mozilla/5.0 (Android 14; Mobile) Chrome/120.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) Safari/17.4",
]


def _make_users():
    role_pool = []
    for role, cfg in ROLES.items():
        role_pool.extend([role] * int(cfg["weight"] * 100))
    users = []
    for uid in range(NUM_USERS):
        role = random.choice(role_pool)
        cfg = ROLES[role]
        users.append({
            "username": f"user_{uid:04d}",
            "role": role,
            "hours": cfg["login_hours"],
            "avg_daily": cfg["avg_daily"],
            "fail_rate": cfg["fail_rate"],
            "country": random.choice(GEO_COUNTRIES),
            "ua": random.choice(USER_AGENTS),
        })
    return users


def _gen_normal(users, base_date):
    logs = []
    for u in users:
        h_start, h_end = u["hours"]
        for day in range(NUM_DAYS):
            dt = base_date + timedelta(days=day)
            if dt.weekday() >= 5 and random.random() < 0.60:
                continue
            n = max(1, int(np.random.poisson(u["avg_daily"] if dt.weekday() < 5 else u["avg_daily"] * 0.25)))
            # Burst clustering — logins often come in 1–3 minute bursts
            burst_start = dt + timedelta(hours=random.uniform(h_start, h_end))
            for i in range(n):
                ts = burst_start + timedelta(minutes=random.uniform(0, 3) * (i // 3))
                logs.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "username": u["username"],
                    "source_ip": f"10.{random.randint(0,5)}.{random.randint(1,200)}.{random.randint(2,254)}",
                    "user_agent": u["ua"],
                    "success": 0 if random.random() < u["fail_rate"] else 1,
                    "geo_country": u["country"],
                    "is_attack": 0,
                })
    return logs


def _gen_attacks(users, base_date):
    """Credential stuffing: rapid bursts from many IPs, many countries, high failure rate."""
    logs = []
    targets = random.sample(users, int(len(users) * 0.25))
    for u in targets:
        n_attacks = random.randint(1, 5)
        for _ in range(n_attacks):
            day = random.randint(0, NUM_DAYS - 1)
            dt = base_date + timedelta(days=day)
            burst = random.randint(60, 400)
            n_ips = max(3, burst // random.randint(15, 40))
            ips = [f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(2, 254)}"
                    for _ in range(n_ips)]
            countries = random.sample(GEO_COUNTRIES, min(n_ips, len(GEO_COUNTRIES))) or ["US"]
            start = dt + timedelta(hours=random.uniform(0, 24))
            for i in range(burst):
                ts = start + timedelta(seconds=random.expovariate(1 / 0.3))
                logs.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "username": u["username"],
                    "source_ip": random.choice(ips),
                    "user_agent": random.choice(USER_AGENTS),
                    "success": 0 if random.random() < random.uniform(0.88, 0.995) else 1,
                    "geo_country": random.choice(countries),
                    "is_attack": 1,
                })
    return logs


def _gen_noise():
    logs = []
    for _ in range(5000):
        day = random.randint(0, NUM_DAYS - 1)
        dt = base_date + timedelta(days=day,
                                   hours=random.uniform(0, 24),
                                   minutes=random.uniform(0, 60))
        logs.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "username": f"user_{random.randint(0, NUM_USERS - 1):04d}",
            "source_ip": f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(2, 254)}",
            "user_agent": random.choice(USER_AGENTS),
            "success": random.choices([0, 1], weights=[0.3, 0.7])[0],
            "geo_country": random.choice(GEO_COUNTRIES),
            "is_attack": 1 if random.random() < 0.25 else 0,
        })
    return logs


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("  Credential Stuffing — Login Log Generator (Enhanced)")
    print("=" * 60)

    global base_date
    base_date = datetime(2026, 3, 1)

    print(f"\n  Generating {NUM_USERS} user profiles...")
    users = _make_users()

    print("  Generating normal login patterns...")
    normal = _gen_normal(users, base_date)
    print(f"    {len(normal):>8} normal entries")

    print("  Generating credential stuffing attacks...")
    attacks = _gen_attacks(users, base_date)
    print(f"    {len(attacks):>8} attack entries")

    print("  Adding background noise...")
    noise = _gen_noise()
    print(f"    {len(noise):>8} noise entries")

    all_logs = normal + attacks + noise
    random.shuffle(all_logs)
    all_logs.sort(key=lambda r: r["timestamp"])

    out_path = DATA_DIR / "login_logs.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp", "username", "source_ip", "user_agent",
            "success", "geo_country", "is_attack",
        ])
        w.writeheader()
        w.writerows(all_logs)

    total = len(all_logs)
    n_attacks = sum(1 for r in all_logs if r["is_attack"])
    print(f"\n  Total entries: {total}")
    print(f"  Attack entries: {n_attacks} ({n_attacks / total * 100:.1f}%)")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
