/**
 * CivicGuard AI — Background Service Worker
 * Handles extension lifecycle and badge updates.
 */

const API_BASE = "http://127.0.0.1:8000";

// ─── Extension Install ─────────────────────────────────────────
chrome.runtime.onInstalled.addListener((details) => {
  console.log("[CivicGuard] Extension installed:", details.reason);

  // Set default settings
  chrome.storage.local.set({
    autoScan: false,
    showBadges: true,
  });

  // Set default badge
  chrome.action.setBadgeText({ text: "" });
  chrome.action.setBadgeBackgroundColor({ color: "#6366f1" });
});

// ─── Health Check ───────────────────────────────────────────────
async function checkBackendHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`);
    if (resp.ok) {
      chrome.action.setBadgeText({ text: "ON" });
      chrome.action.setBadgeBackgroundColor({ color: "#22c55e" });
      return true;
    }
  } catch (e) {
    // Backend not available
  }

  chrome.action.setBadgeText({ text: "OFF" });
  chrome.action.setBadgeBackgroundColor({ color: "#ef4444" });
  return false;
}

// ─── Message Handling ───────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "CIVICGUARD_RESULT") {
    // Update extension badge based on analysis result
    const badgeConfig = {
      hate: { text: "!", color: "#ef4444" },
      offensive: { text: "⚠", color: "#f59e0b" },
      neutral: { text: "✓", color: "#22c55e" },
    };

    const config = badgeConfig[msg.label] || badgeConfig.neutral;
    chrome.action.setBadgeText({ text: config.text });
    chrome.action.setBadgeBackgroundColor({ color: config.color });

    // Reset badge after 5 seconds
    setTimeout(() => {
      checkBackendHealth();
    }, 5000);
  }

  sendResponse({ ok: true });
  return true;
});

// ─── Periodic Health Check ──────────────────────────────────────
// Check every 30 seconds
setInterval(checkBackendHealth, 30000);

// Initial check
checkBackendHealth();
