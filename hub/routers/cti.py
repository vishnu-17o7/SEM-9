"""CTI Lab router — catalog and project detail pages."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from hub.templates import render
from hub.supermemory import memorize, recall

router = APIRouter(prefix="/cti", tags=["cti"])

# ── Project registry ──────────────────────────────────────────────────────────
# Each project has a single "Run All" button that executes `run.py all`.
# Web projects also show an "Open Web UI" button linking to their port.
PROJECTS = [
    {
        "id": "01-spam-ham-watcher",
        "num": "01",
        "title": "Spam/Ham Watcher",
        "type": "web",
        "port": 8000,
        "description": "Real-time spam classification via FastAPI dashboard + Gmail IMAP watcher. Uses TF-IDF + Naive Bayes. Streamlit dashboard available too.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "02-sms-spam-ml-compare",
        "num": "02",
        "title": "SMS Spam ML Compare",
        "type": "web",
        "port": 5004,
        "description": "Compares 6 ML models (NB, LR, SVM, RF, GBDT, Complement NB) + 2 ensembles on SMS spam classification. Generates HTML report with confusion matrices and ROC curves. Web UI for live classification.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "03-phishing-url-detector",
        "num": "03",
        "title": "Phishing URL Detector",
        "type": "web",
        "port": 5000,
        "description": "Detects phishing URLs using an ensemble of ML models. Extracts URL features (length, special chars, domain patterns) and predicts phishing/legitimate with confidence scores. Two-tier: dataset lookup then ML model.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "04-email-phishing-nlp",
        "num": "04",
        "title": "Email Phishing NLP",
        "type": "web",
        "port": 5005,
        "description": "NLP-based phishing email classifier using TF-IDF vectorization and machine learning. Supports single email classification and interactive mode. Generates training reports. Web UI for live email classification.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "05-credential-stuffing-detector",
        "num": "05",
        "title": "Credential Stuffing Detector",
        "type": "web",
        "port": 5006,
        "description": "Detects credential stuffing attacks in login logs using behavioral analysis. Generates 690k+ synthetic login entries (1,000 users, 60 days) with 25% attack rate, then compares Rule-based, IsolationForest, RandomForest, and LOF detectors. RandomForest achieves 0.97 F1 at 0.17% false positive rate. Web UI for real-time detection.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "06-behavioral-profile-ueba",
        "num": "06",
        "title": "Behavioral Profile / UEBA",
        "type": "web",
        "port": 5001,
        "description": "User and Entity Behavior Analytics (UEBA) dashboard. Builds behavioral profiles from activity data, detects anomalies via statistical methods, and visualizes results with charts.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "07-malicious-executable-detector",
        "num": "07",
        "title": "Malicious Executable Detector",
        "type": "cli",
        "port": None,
        "description": "Detects malicious PE (Portable Executable) files using static feature analysis. Extracts features from PE headers, sections, imports, and metadata, then trains a classifier.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "08-malicious-pdf-detector",
        "num": "08",
        "title": "Malicious PDF Detector",
        "type": "cli",
        "port": None,
        "description": "Detects malicious PDF documents by analyzing structural features. Generates synthetic benign/malicious PDFs and trains detection models.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "09-ransomware-behavior-detector",
        "num": "09",
        "title": "Ransomware Behavior Detector",
        "type": "cli",
        "port": None,
        "description": "Detects ransomware-like file system behavior by analyzing file operation patterns. Simulates ransomware and benign file operations, then trains a behavioral classifier.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "10-crypto-ransomware-detector",
        "num": "10",
        "title": "Crypto-Ransomware Detector",
        "type": "cli",
        "port": None,
        "description": "Detects crypto-ransomware by analyzing encryption behavior patterns. Simulates cryptographic file operations and trains models to distinguish ransomware from benign activity.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "11-network-intrusion-detector",
        "num": "11",
        "title": "Network Intrusion Detector",
        "type": "cli",
        "port": None,
        "description": "Network intrusion detection using the NSL-KDD dataset. Downloads benchmark data, trains classifiers, and evaluates detection performance.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "12-botnet-traffic-classifier",
        "num": "12",
        "title": "Botnet Traffic Classifier",
        "type": "web",
        "port": 5007,
        "description": "Classifies network flows as botnet or normal traffic. Generates synthetic flow data and trains classifiers to identify C&C communication patterns. Web UI for live flow classification.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "13-user-activity-anomaly-detector",
        "num": "13",
        "title": "User Activity Anomaly Detector",
        "type": "cli",
        "port": None,
        "description": "Detects anomalous user activity patterns using statistical and ML methods. Generates synthetic user activity logs and identifies deviations from normal behavior.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "14-threat-intel-repository",
        "num": "14",
        "title": "Threat Intel Repository",
        "type": "web",
        "port": 5008,
        "description": "SQLite-based threat intelligence repository. Stores and queries IoCs (IPs, domains, URLs, hashes, CVEs). Supports importing from feeds and exporting to CSV/JSON formats. Web UI for searching and browsing.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "15-threat-intel-social-mining",
        "num": "15",
        "title": "Social Media Mining",
        "type": "cli",
        "port": None,
        "description": "Mines threat intelligence IoCs from simulated social media feeds (Twitter, Reddit, blogs). Extracts IPs, domains, URLs, hashes, and CVEs using regex patterns.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "16-threat-intel-dashboard",
        "num": "16",
        "title": "Threat Intel Dashboard",
        "type": "web",
        "port": 5002,
        "description": "Real-time threat intelligence dashboard with Chart.js visualizations. Displays indicators, alerts, and trends. Integrates data from the Threat Intel Repository and Social Mining projects.",
        "scripts": [],
    },
    {
        "id": "17-facial-recognition-auth",
        "num": "17",
        "title": "Facial Recognition Auth",
        "type": "cli",
        "port": None,
        "description": "Face recognition authentication system. Enrolls users via webcam or images, then authenticates by comparing live face embeddings against enrolled profiles using cosine similarity.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
    {
        "id": "18-multi-factor-biometric-auth",
        "num": "18",
        "title": "Multi-Factor Auth (Face + OTP)",
        "type": "web",
        "port": 5003,
        "description": "Combines facial recognition with time-based OTP for multi-factor authentication. Generates QR codes for OTP setup and logs all authentication attempts.",
        "scripts": [
            {"path": "run.py", "desc": "Run all steps", "args": "all"},
        ],
    },
]

CTI_PROJECT_MAP = {p["id"]: p for p in PROJECTS}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def cti_index(request: Request):
    """Show all CTI projects as a grid."""
    html = render("cti/index.html", request=request, projects=PROJECTS, active="cti")
    return HTMLResponse(html)


@router.get("/{project_id}", response_class=HTMLResponse)
async def cti_project(request: Request, project_id: str):
    """Show a single CTI project detail page."""
    project = CTI_PROJECT_MAP.get(project_id)
    if not project:
        return HTMLResponse("Project not found", status_code=404)

    # Memorise page view (fire-and-forget)
    memorize(
        content=f"Viewed CTI project '{project['title']}' (id: {project_id})",
        container_tag="sem9-hub",
        metadata={
            "project_id": project_id,
            "project_title": project["title"],
            "type": "page_view",
            "lab": "cti",
        },
    )

    # Recall relevant memories from previous script runs
    past_runs = recall(query=f"CTI {project['title']} script_run", limit=5)

    html = render(
        "cti/project.html",
        request=request,
        project=project,
        active="cti",
        memories=past_runs,
    )
    return HTMLResponse(html)
