const classifyTabButton = document.getElementById("classifyTabButton");
const manualButton = document.getElementById("manualButton");
const manualUrl = document.getElementById("manualUrl");
const resultPanel = document.getElementById("resultPanel");
const verdict = document.getElementById("verdict");
const urlText = document.getElementById("urlText");
const confidenceFill = document.getElementById("confidenceFill");
const metaText = document.getElementById("metaText");
const signalsList = document.getElementById("signalsList");
const serverState = document.getElementById("serverState");

function setBusy(isBusy) {
  classifyTabButton.disabled = isBusy;
  manualButton.disabled = isBusy;
  classifyTabButton.textContent = isBusy ? "Checking..." : "Classify Current Tab";
}

function normalizeManualUrl(url) {
  const trimmed = url.trim();
  if (!trimmed) {
    return "";
  }
  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed;
  }
  return `https://${trimmed}`;
}

function setError(message, detail) {
  resultPanel.className = "result-panel error";
  verdict.textContent = "Detector unavailable";
  urlText.textContent = message;
  confidenceFill.style.width = "0%";
  metaText.textContent = detail || "Start the local CTI 03 Flask app and try again.";
  signalsList.innerHTML = '<span class="empty">No signals available.</span>';
  serverState.textContent = "Offline";
}

function setResult(data) {
  const isPhishing = data.label === "phishing";
  const confidence = Math.max(0, Math.min(1, Number(data.confidence || 0)));
  const pct = `${(confidence * 100).toFixed(1)}%`;

  resultPanel.className = `result-panel ${data.label}`;
  verdict.textContent = isPhishing ? "Phishing detected" : "Legitimate";
  urlText.textContent = data.url;
  confidenceFill.style.width = pct;
  metaText.textContent = `${pct} confidence via ${data.source || "model"} prediction`;
  serverState.textContent = "Online";

  const signals = [
    ...(data.sus_signals || []).map((name) => ({ name, kind: "suspicious" })),
    ...(data.leg_signals || []).map((name) => ({ name, kind: "legitimate" })),
  ];

  if (!signals.length) {
    signalsList.innerHTML = '<span class="empty">No highlighted signals returned.</span>';
    return;
  }

  signalsList.replaceChildren(
    ...signals.map((signal) => {
      const node = document.createElement("span");
      node.className = `signal ${signal.kind}`;
      node.textContent = signal.name;
      return node;
    }),
  );
}

async function sendMessage(message) {
  return chrome.runtime.sendMessage(message);
}

async function classifyActiveTab() {
  setBusy(true);
  try {
    const response = await sendMessage({ type: "classifyActiveTab" });
    if (response && response.ok) {
      setResult(response.result);
    } else {
      setError(response?.error || "Classification failed.", response?.detail);
    }
  } finally {
    setBusy(false);
  }
}

async function classifyManualUrl() {
  const url = normalizeManualUrl(manualUrl.value);
  if (!url) {
    setError("Enter a URL to classify.", "Manual checks accept http and https URLs.");
    return;
  }

  setBusy(true);
  try {
    const response = await sendMessage({ type: "classifyUrl", url });
    if (response && response.ok) {
      setResult(response.result);
    } else {
      setError(response?.error || "Classification failed.", response?.detail);
    }
  } finally {
    setBusy(false);
  }
}

classifyTabButton.addEventListener("click", classifyActiveTab);
manualButton.addEventListener("click", classifyManualUrl);
manualUrl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    classifyManualUrl();
  }
});

classifyActiveTab();
