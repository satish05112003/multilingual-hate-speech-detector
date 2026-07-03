/**
 * CivicGuard AI — Popup Script
 * Handles text analysis via the backend API.
 */

const API_BASE = "http://127.0.0.1:8000";

// ─── DOM Elements ──────────────────────────────────────────────
const textInput = document.getElementById("textInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorMsg = document.getElementById("errorMsg");
const resultsSection = document.getElementById("resultsSection");
const labelBadge = document.getElementById("labelBadge");
const badgeIcon = document.getElementById("badgeIcon");
const labelText = document.getElementById("labelText");
const confidenceValue = document.getElementById("confidenceValue");
const autoScan = document.getElementById("autoScan");
const showBadges = document.getElementById("showBadges");
const blurToggle = document.getElementById("blurToggle");

// ─── Badge Config ──────────────────────────────────────────────
const BADGE_CONFIG = {
  hate: { icon: "🔴", label: "Hate Speech", class: "hate" },
  offensive: { icon: "🟡", label: "Offensive", class: "offensive" },
  neutral: { icon: "🟢", label: "Neutral", class: "neutral" },
};

// Backend connection checks removed.
// ─── Analyze ───────────────────────────────────────────────────
async function analyzeText() {
  const text = textInput.value.trim();
  if (!text) return;

  // Show loading
  analyzeBtn.classList.add("loading");
  analyzeBtn.textContent = "Analyzing";
  errorMsg.classList.remove("visible");
  resultsSection.classList.remove("visible");

  try {
    const resp = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

    const data = await resp.json();
    displayResults(data);
  } catch (e) {
    showError(e.message || "Failed to connect to backend");
  } finally {
    analyzeBtn.classList.remove("loading");
    analyzeBtn.textContent = "Analyze";
  }
}

// ─── Display Results ───────────────────────────────────────────
function displayResults(data) {
  const config = BADGE_CONFIG[data.label] || BADGE_CONFIG.neutral;

  // Label badge
  labelBadge.className = `label-badge ${config.class}`;
  badgeIcon.textContent = config.icon;
  labelText.textContent = config.label;

  // Confidence
  const confPct = (data.confidence * 100).toFixed(1);
  confidenceValue.textContent = `${confPct}%`;

  // Show results
  resultsSection.classList.add("visible");

  // Send badge update to content script
  chrome.tabs?.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs?.[0]?.id) {
      chrome.tabs.sendMessage(tabs[0].id, {
        type: "CIVICGUARD_RESULT",
        label: data.label,
        confidence: data.confidence,
      });
    }
  });
}

// ─── Error ─────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = `⚠️ ${msg}`;
  errorMsg.classList.add("visible");
}

// ─── Settings ──────────────────────────────────────────────────
function loadSettings() {
  chrome.storage?.local?.get(["autoScan", "showBadges", "blurEnabled"], (data) => {
    if (data.autoScan !== undefined) autoScan.checked = data.autoScan;
    if (data.showBadges !== undefined) showBadges.checked = data.showBadges;
    if (data.blurEnabled !== undefined) blurToggle.checked = data.blurEnabled;
  });
}

function saveSettings() {
  chrome.storage?.local?.set({
    autoScan: autoScan.checked,
    showBadges: showBadges.checked,
    blurEnabled: blurToggle.checked,
  });

  // Notify content script
  chrome.tabs?.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs?.[0]?.id) {
      chrome.tabs.sendMessage(tabs[0].id, {
        type: "CIVICGUARD_SETTINGS",
        autoScan: autoScan.checked,
        showBadges: showBadges.checked,
        blurEnabled: blurToggle.checked,
      });
    }
  });
}

// ─── Event Listeners ───────────────────────────────────────────
analyzeBtn.addEventListener("click", analyzeText);

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    analyzeText();
  }
});

autoScan.addEventListener("change", saveSettings);
showBadges.addEventListener("change", saveSettings);
blurToggle.addEventListener("change", saveSettings);

// ─── Init ──────────────────────────────────────────────────────
loadSettings();
