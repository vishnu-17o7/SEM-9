"""Streamlit dashboard for the spam/ham IMAP watcher.

Run:
    streamlit run dashboard.py
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

import db

st.set_page_config(
    page_title="Spam/Ham Watcher",
    page_icon="📬",
    layout="wide",
)


def _since_for_filter(key: str) -> datetime | None:
    now = datetime.now(timezone.utc)
    return {
        "Last 1 hour": now - timedelta(hours=1),
        "Last 24 hours": now - timedelta(hours=24),
        "Last 7 days": now - timedelta(days=7),
        "Last 30 days": now - timedelta(days=30),
        "All time": None,
    }[key]


@st.cache_data(ttl=3, show_spinner=False)
def _load_summary(since_iso: str | None) -> dict:
    return db.fetch_summary(since_iso)


@st.cache_data(ttl=3, show_spinner=False)
def _load_recent(since_iso: str | None, limit: int) -> list[dict]:
    rows = db.fetch_recent(limit=limit, since=since_iso)
    return [dict(r) for r in rows]


@st.cache_data(ttl=3, show_spinner=False)
def _load_buckets(since_iso: str | None) -> list[dict]:
    return [
        {"bucket": b, "spam": s, "ham": h}
        for (b, s, h) in db.fetch_probability_buckets(since_iso)
    ]


def _label_pill(label: str) -> str:
    color = "#d62728" if label == "spam" else "#2ca02c"
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:10px;font-size:0.85em;font-weight:600'>{label.upper()}</span>"
    )


def main() -> None:
    st.title("📬 Spam / Ham Watcher Dashboard")
    st.caption("Live view of the IMAP watcher's classifications.")

    with st.sidebar:
        st.header("Filters")
        window = st.selectbox(
            "Time window",
            ["Last 1 hour", "Last 24 hours", "Last 7 days", "Last 30 days", "All time"],
            index=2,
        )
        recent_limit = st.slider("Recent rows to show", 10, 500, 100, step=10)
        st.divider()
        st.caption("Run the watcher in another terminal:")
        st.code("python gmail_watcher.py", language="bash")

    since_dt = _since_for_filter(window)
    since_iso = since_dt.isoformat(timespec="seconds") if since_dt else None

    try:
        summary = _load_summary(since_iso)
    except Exception as e:
        st.error(f"Could not read database: {e}")
        st.stop()

    total = summary["total"]
    spam = summary["spam"]
    ham = summary["ham"]
    ratio = summary["spam_ratio"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total emails", f"{total:,}")
    m2.metric("Spam", f"{spam:,}")
    m3.metric("Ham", f"{ham:,}")
    m4.metric("Spam ratio", f"{ratio:.1%}")

    st.divider()

    if total == 0:
        st.info(
            "No classifications yet for this window. Start the watcher with "
            "`python gmail_watcher.py` and send a test email to populate the dashboard."
        )
        st.stop()

    left, right = st.columns(2)

    with left:
        st.subheader("Spam vs ham over time")
        daily = summary["daily"]
        if daily:
            df_daily = pd.DataFrame(
                [
                    {"day": d, "spam": v["spam"], "ham": v["ham"]}
                    for d, v in sorted(daily.items())
                ]
            )
            df_daily["day"] = pd.to_datetime(df_daily["day"])
            df_daily = df_daily.set_index("day")
            st.line_chart(df_daily, height=280)
        else:
            st.info("No daily data.")

    with right:
        st.subheader("Probability distribution")
        buckets = _load_buckets(since_iso)
        if buckets:
            df_b = pd.DataFrame(buckets).set_index("bucket")
            st.bar_chart(df_b, height=280, stack=False)
            st.caption(
                "Probability is the model's confidence the email is spam. "
                "Ham emails at the right tail and spam emails at the left tail "
                "indicate misclassifications worth inspecting."
            )

    st.divider()
    st.subheader(f"Recent classifications (last {recent_limit})")
    rows = _load_recent(since_iso, recent_limit)
    if not rows:
        st.info("No rows in this window.")
        return

    df = pd.DataFrame(rows)
    df["received_at"] = pd.to_datetime(df["received_at"])
    df["label_pill"] = df["label"].apply(_label_pill)
    df["probability"] = df["probability"].map(lambda p: f"{p:.1%}")

    view = df[["received_at", "sender", "subject", "label_pill", "probability"]].copy()
    view.columns = ["Received (UTC)", "From", "Subject", "Label", "Spam prob."]

    st.write(
        view.to_html(escape=False, index=False, classes="dash-table"),
        unsafe_allow_html=True,
    )
    st.caption(f"DB: `{db.DB_PATH}`")


if __name__ == "__main__":
    main()
