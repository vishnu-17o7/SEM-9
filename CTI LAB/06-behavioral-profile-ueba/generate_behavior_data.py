"""
Generate synthetic user behavior data mimicking an enterprise environment.

Creates 50 users with roles, normal behavior patterns, and injected anomalies.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NUM_USERS = 50
NUM_DAYS = 60
TOTAL_LOGS = 50_000

ROLES = {
    "Engineer": 0.25,
    "HR": 0.10,
    "Finance": 0.10,
    "Admin": 0.15,
    "Exec": 0.05,
    "Support": 0.15,
    "Sales": 0.10,
    "Marketing": 0.10,
}

RESOURCE_POOLS = {
    "Engineer": ["repo/internal", "repo/shared", "ci/cd", "docs/tech", "db/dev", "monitoring", "jira", "confluence"],
    "HR": ["hr/dashboard", "hr/employee_db", "payroll/viewer", "docs/hr", "calendar", "timesheets"],
    "Finance": ["finance/erp", "finance/reports", "finance/invoices", "finance/budget", "db/finance"],
    "Admin": ["admin/users", "admin/groups", "admin/audit", "admin/config", "admin/logs", "admin/security"],
    "Exec": ["exec/dashboard", "exec/reports", "exec/strategy", "exec/analytics", "any"],
    "Support": ["support/tickets", "support/kb", "support/customers", "docs/support", "monitoring"],
    "Sales": ["sales/crm", "sales/leads", "sales/reports", "sales/contracts", "docs/sales"],
    "Marketing": ["marketing/campaigns", "marketing/analytics", "marketing/assets", "marketing/social", "docs/marketing"],
}

ACTIONS = ["login", "file_access", "file_download", "file_upload", "db_query", "db_write", "privilege_change", "config_change"]


def generate_users():
    users = []
    role_pool = []
    for role, weight in ROLES.items():
        role_pool.extend([role] * int(weight * 100))
    for uid in range(NUM_USERS):
        role = random.choice(role_pool)
        resources = RESOURCE_POOLS[role]
        users.append({
            "user_id": f"u_{uid:04d}",
            "role": role,
            "resources": resources,
            "login_window": (8, 18) if role not in ("Support", "Admin") else (0, 23),
            "avg_actions_per_day": random.randint(10, 30) if role in ("Admin", "Engineer") else random.randint(4, 15),
            "anomaly_rate": 0.0,  # will be set per injected anomaly
        })
    return users


def generate_normal_logs(users, base_date):
    logs = []
    for user in users:
        uid = user["user_id"]
        role = user["role"]
        resources = user["resources"]
        start_h, end_h = user["login_window"]
        avg = user["avg_actions_per_day"]

        for day in range(NUM_DAYS):
            date = base_date + timedelta(days=day)
            weekday = date.weekday()
            is_weekend = weekday >= 5

            if is_weekend and random.random() < 0.8:
                continue  # 80% skip weekends

            daily_count = max(1, int(np.random.poisson(avg * (0.3 if is_weekend else 1.0))))

            for _ in range(daily_count):
                hour = random.uniform(start_h + 1, end_h - 1)
                minute = random.uniform(0, 60)
                ts = date + timedelta(hours=hour, minutes=minute)
                action = random.choices(ACTIONS, weights=[15, 30, 10, 5, 15, 5, 1, 1], k=1)[0]
                resource = random.choice(resources) if resources else "unknown"
                success = 1 if random.random() > 0.05 else 0

                logs.append({
                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                    "user_id": uid,
                    "role": role,
                    "action": action,
                    "resource": resource,
                    "success": success,
                    "hour_of_day": int(hour),
                    "day_of_week": weekday,
                    "is_anomaly": 0,
                })
    return logs


def inject_anomalies(users, base_date, logs):
    """Inject anomalous behaviors into the log stream."""
    anomaly_logs = []
    anomaly_types = [
        "off_hours", "mass_download", "role_violation",
        "privilege_escalation", "geo_velocity",
    ]

    for user in random.sample(users, int(len(users) * 0.5)):
        uid = user["user_id"]
        role = user["role"]
        num_anomalies = random.randint(1, 4)

        for _ in range(num_anomalies):
            atype = random.choice(anomaly_types)
            day = random.randint(10, NUM_DAYS - 5)
            date = base_date + timedelta(days=day)

            if atype == "off_hours":
                # Login at 3 AM
                for _ in range(random.randint(3, 10)):
                    ts = date + timedelta(hours=random.uniform(2, 4), minutes=random.uniform(0, 60))
                    anomaly_logs.append({
                        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "user_id": uid, "role": role,
                        "action": "login", "resource": "unknown",
                        "success": 1, "hour_of_day": int(ts.hour),
                        "day_of_week": ts.weekday(), "is_anomaly": 1,
                    })

            elif atype == "mass_download":
                # Download 50-200 files in 10 minutes
                for _ in range(random.randint(50, 200)):
                    ts = date + timedelta(hours=random.uniform(9, 17), seconds=random.uniform(1, 10))
                    resource = random.choice(user["resources"]) if user["resources"] else "documents"
                    anomaly_logs.append({
                        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "user_id": uid, "role": role,
                        "action": "file_download", "resource": resource,
                        "success": 1, "hour_of_day": int(ts.hour),
                        "day_of_week": ts.weekday(), "is_anomaly": 1,
                    })

            elif atype == "role_violation":
                # Access HR/finance resources from non-HR user
                forbidden = ["hr/employee_db", "payroll/viewer", "finance/erp", "finance/budget", "exec/strategy"]
                target = random.choice(forbidden)
                for _ in range(random.randint(3, 8)):
                    ts = date + timedelta(hours=random.uniform(9, 17), minutes=random.uniform(0, 60))
                    anomaly_logs.append({
                        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "user_id": uid, "role": role,
                        "action": random.choice(["db_query", "file_access"]),
                        "resource": target,
                        "success": random.choice([0, 0, 1]),
                        "hour_of_day": int(ts.hour),
                        "day_of_week": ts.weekday(), "is_anomaly": 1,
                    })

            elif atype == "privilege_escalation":
                for _ in range(random.randint(2, 6)):
                    ts = date + timedelta(hours=random.uniform(0, 23), minutes=random.uniform(0, 60))
                    anomaly_logs.append({
                        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "user_id": uid, "role": role,
                        "action": "privilege_change", "resource": "admin/security",
                        "success": random.choice([0, 0, 1]),
                        "hour_of_day": int(ts.hour),
                        "day_of_week": ts.weekday(), "is_anomaly": 1,
                    })

            elif atype == "geo_velocity":
                # Logins from different countries within 1 hour
                for h in range(3):
                    ts = date + timedelta(hours=12 + h * 0.5, minutes=random.uniform(0, 30))
                    anomaly_logs.append({
                        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "user_id": uid, "role": role,
                        "action": "login", "resource": f"geo_{random.choice(['US', 'CN', 'RU', 'NG', 'KR'])}",
                        "success": 1, "hour_of_day": int(ts.hour),
                        "day_of_week": ts.weekday(), "is_anomaly": 1,
                    })

    return logs + anomaly_logs


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 50)
    print("  Behavioral Profile Data Generator")
    print("=" * 50)

    base_date = datetime(2026, 4, 1)

    print("\n  Generating user profiles...")
    users = generate_users()
    print(f"  {len(users)} users created")

    print("  Generating normal behavior...")
    logs = generate_normal_logs(users, base_date)
    print(f"  {len(logs)} normal log entries")

    print("  Injecting anomalies...")
    logs = inject_anomalies(users, base_date, logs)
    print(f"  After injection: {len(logs)} total entries")

    random.shuffle(logs)
    df = pd.DataFrame(logs)
    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = DATA_DIR / "user_activity.csv"
    df.to_csv(out_path, index=False)

    anomaly_count = df["is_anomaly"].sum()
    print(f"\n  Total entries: {len(df)}")
    print(f"  Anomaly entries: {anomaly_count} ({anomaly_count / len(df) * 100:.1f}%)")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
