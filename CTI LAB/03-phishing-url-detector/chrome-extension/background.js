const API_URL = "http://127.0.0.1:5000/predict";
const REQUEST_TIMEOUT_MS = 5000;

const BADGES = {
  legitimate: { text: "OK", color: "#15803d" },
  phishing: { text: "PHISH", color: "#b91c1c" },
  offline: { text: "OFF", color: "#52525b" },
  idle: { text: "", color: "#52525b" },
};

function isClassifiableUrl(url) {
  return typeof url === "string" && /^https?:\/\//i.test(url);
}

async function postPrediction(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
      signal: controller.signal,
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || `Detector returned ${response.status}`);
    }

    return data;
  } finally {
    clearTimeout(timeout);
  }
}

async function setBadge(tabId, state) {
  const badge = BADGES[state] || BADGES.idle;
  await chrome.action.setBadgeText({ tabId, text: badge.text });
  await chrome.action.setBadgeBackgroundColor({ tabId, color: badge.color });
}

async function classifyUrl(url, tabId) {
  if (!isClassifiableUrl(url)) {
    if (Number.isInteger(tabId)) {
      await setBadge(tabId, "idle");
    }
    return {
      ok: false,
      error: "Only http and https URLs can be classified.",
      url,
    };
  }

  try {
    const result = await postPrediction(url);
    if (Number.isInteger(tabId)) {
      await setBadge(tabId, result.label === "phishing" ? "phishing" : "legitimate");
    }
    return { ok: true, result };
  } catch (error) {
    if (Number.isInteger(tabId)) {
      await setBadge(tabId, "offline");
    }
    return {
      ok: false,
      error: "Detector is unavailable. Start CTI LAB/03-phishing-url-detector with python run.py web.",
      detail: error instanceof Error ? error.message : String(error),
      url,
    };
  }
}

async function classifyActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id) {
    return { ok: false, error: "No active tab found." };
  }
  return classifyUrl(tab.url || "", tab.id);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || typeof message.type !== "string") {
    return false;
  }

  if (message.type === "classifyActiveTab") {
    classifyActiveTab().then(sendResponse);
    return true;
  }

  if (message.type === "classifyUrl") {
    const tabId = Number.isInteger(message.tabId) ? message.tabId : undefined;
    classifyUrl(String(message.url || ""), tabId).then(sendResponse);
    return true;
  }

  return false;
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete" || !isClassifiableUrl(tab.url)) {
    return;
  }
  classifyUrl(tab.url, tabId);
});

chrome.tabs.onActivated.addListener(async (activeInfo) => {
  try {
    const tab = await chrome.tabs.get(activeInfo.tabId);
    if (isClassifiableUrl(tab.url)) {
      classifyUrl(tab.url, activeInfo.tabId);
    }
  } catch (_error) {
    await setBadge(activeInfo.tabId, "idle");
  }
});
