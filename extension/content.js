// ─── CivicGuard AI — Content Script ─────────────────────────────────────────
console.log("CivicGuard script loaded");

// ─── MAIN CONTENT SCRIPT ─────────────────────────────────────────────────────

// Global text-based dedup lock (survives React re-renders)
const processedTexts = new Set();

// Throttle state for Instagram observer
let lastRun = 0;

const intersectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const el = entry.target;

            if (!el.dataset.civicguardObserved) {
                el.dataset.civicguardObserved = "true";

                if (location.href.includes("instagram.com")) {
                    processSingleInstagram(el);
                }

                if (location.href.includes("reddit.com")) {
                    processSingleReddit(el);
                }
            }
        }
    });
}, {
    threshold: 0.5
});

// TASK 1: CREATE REUSABLE BLUR FUNCTION
function applyBlurIfNeeded(textNode, label) {
    if (!textNode) return;

    chrome.storage.local.get(["blurEnabled"], (res) => {
        if (!res.blurEnabled) return;

        if (label !== "hate" && label !== "offensive") return;

        // prevent duplicate blur
        if (textNode.classList.contains("civicguard-blurred")) return;

        textNode.classList.add("civicguard-blurred");

        textNode.style.filter = "blur(6px)";
        textNode.style.transition = "0.3s";

        textNode.onmouseenter = () => {
            textNode.style.filter = "none";
        };

        textNode.onmouseleave = () => {
            textNode.style.filter = "blur(6px)";
        };
    });
}

function processElement(element, text) {
    if (!text || text.length < 3) return;

    if (element.dataset.processed) return;
    element.dataset.processed = "true";

    fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    })
    .then(res => res.json())
    .then(data => {
        console.log("Platform:", window.location.hostname);
        console.log("Text:", text);
        console.log("Response:", data);
        showLabel(element, data);

        // TASK 2: APPLY TO YOUTUBE COMMENTS
        applyBlurIfNeeded(element, data.label);
    })
    .catch(err => console.error("CivicGuard API Error:", err));
}

function showLabel(container, data, platform) {
    if (!container) return;

    // TASK 6: Duplicate guard
    if (container.querySelector(".civicguard-label")) return;

    const label = document.createElement("div");
    label.className = "civicguard-label";
    label.innerText = data.label.toUpperCase();

    // Base styles
    label.style.fontWeight = "bold";
    label.style.borderRadius = "999px";
    label.style.color = "white";
    label.style.fontSize = "11px";
    label.style.padding = "2px 6px";
    label.style.flexShrink = "0";

    if (platform === "twitter") {
        label.style.display = "inline-flex";
        label.style.alignItems = "center";

        label.style.marginLeft = "8px";   // spacing from time
        label.style.marginRight = "6px";

        label.style.fontSize = "11px";
        label.style.padding = "2px 6px";
        label.style.borderRadius = "999px";
        label.style.whiteSpace = "nowrap";

        // Insert AFTER time text
        container.appendChild(label);
    } else {
        // YouTube / generic — inline after content
        label.style.display = "inline-block";
        label.style.marginLeft = "10px";
        label.style.padding = "3px 8px";
        container.appendChild(label);
    }

    if (data.label === "hate") {
        label.style.background = "#ff4d4f";
    } else if (data.label === "offensive") {
        label.style.background = "#faad14";
    } else {
        label.style.background = "#52c41a";
    }
}

function processTwitter() {
    const isDetailView = window.location.pathname.includes("/status/");
    const tweets = isDetailView ? document.querySelectorAll('article') : document.querySelectorAll('[data-testid="cellInnerDiv"]');

    tweets.forEach(tweet => {
        const textNode = tweet.querySelector('[data-testid="tweetText"]');
        if (!textNode) return;

        const text = textNode.innerText;
        if (!text || text.length < 5) return;

        if (tweet.dataset.processed) return;
        tweet.dataset.processed = "true";

        if (isDetailView) {
            // TASK 1: TARGET TWEET CONTAINER ROW
            const usernameRow = tweet.querySelector('div[dir="ltr"]');
            if (!usernameRow) return;

            fetch("http://127.0.0.1:8000/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            })
            .then(res => res.json())
            .then(data => {
                if (usernameRow.querySelector(".civicguard-label")) return;

                const label = document.createElement("span");
                label.className = "civicguard-label";
                label.innerText = data.label.toUpperCase();
                
                // TASK 1: LABEL LOCATION & STYLE FOR DETAIL PAGE
                label.style.display = "inline-flex";
                label.style.alignItems = "center";
                label.style.marginLeft = "8px";
                label.style.fontSize = "12px";
                label.style.padding = "3px 8px";
                label.style.borderRadius = "999px";
                label.style.color = "white";
                label.style.fontWeight = "bold";

                if (data.label === "hate") {
                    label.style.background = "#ff4d4f";
                } else if (data.label === "offensive") {
                    label.style.background = "#faad14";
                } else {
                    label.style.background = "#52c41a";
                }

                usernameRow.appendChild(label);

                // TASK 2: BLUR TOGGLE APPLY
                if (data.label === "hate" || data.label === "offensive") {
                    chrome.storage.local.get(["blurEnabled"], (res) => {
                        if (res.blurEnabled) {
                            textNode.style.filter = "blur(6px)";
                            textNode.style.transition = "0.3s";
                            textNode.onmouseenter = () => { textNode.style.filter = "none"; };
                            textNode.onmouseleave = () => { textNode.style.filter = "blur(6px)"; };
                        }
                    });
                }
            })
            .catch(err => console.error("CivicGuard detail fetch error:", err));

            return; // Skip timeline logic entirely
        }

        // ORIGINAL TIMELINE LOGIC
        // TASK 1: TARGET METADATA TEXT ROW
        const meta = tweet.querySelector('[data-testid="User-Name"]');
        if (!meta) return;

        // TASK 2: FIND TIME ELEMENT
        const timeElement = meta.querySelector('time');
        if (!timeElement) return;

        fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        })
        .then(res => res.json())
        .then(data => {
            // TASK 3: INSERT LABEL AFTER TIME
            showLabel(timeElement.parentElement, data, "twitter");

            // TASK 2: BLUR TOGGLE APPLY (Timeline)
            if (data.label === "hate" || data.label === "offensive") {
                chrome.storage.local.get(["blurEnabled"], (res) => {
                    if (res.blurEnabled) {
                        textNode.style.filter = "blur(6px)";
                        textNode.style.transition = "0.3s";
                        textNode.onmouseenter = () => { textNode.style.filter = "none"; };
                        textNode.onmouseleave = () => { textNode.style.filter = "blur(6px)"; };
                    }
                });
            }
        })
        .catch(err => console.error("CivicGuard fetch error:", err));
    });
}

function processYouTube() {
    const comments = document.querySelectorAll("#content-text");

    comments.forEach(comment => {
        processElement(comment, comment.innerText);
    });
}

function processInstagram() {

    let comments = [];

    // Reels (popup comments)
    if (location.href.includes("/reel")) {
        comments = document.querySelectorAll('div[role="dialog"] ul li');
    } 
    // Normal posts
    else {
        comments = document.querySelectorAll('ul ul li');
    }

    comments.forEach(root => {

        setTimeout(() => {

            const spans = root.querySelectorAll("span");
            let text = "";

            spans.forEach(s => {
                const t = s.innerText?.trim();

                // skip unwanted UI text
                if (
                    !t ||
                    t.length < 2 ||
                    t === "Reply" ||
                    t.includes("Reply") ||
                    t.includes("See translation") ||
                    t.includes("View replies") ||
                    t.includes("likes") ||
                    t.includes("Like") ||
                    t.includes("Follow")
                ) return;

                text += t + " ";
            });

            text = text.trim();

            if (!text || text.length < 3) return;

            // prevent duplicates
            if (processedTexts.has(text)) return;
            processedTexts.add(text);

            // prevent double label
            if (root.querySelector(".civicguard-label")) return;

            fetch("http://127.0.0.1:8000/analyze", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ text })
            })
            .then(res => res.json())
            .then(result => {

                if (root.querySelector(".civicguard-label")) return;

                const label = document.createElement("span");
                label.className = "civicguard-label";
                label.innerText = result.label.toUpperCase();

                label.style.position = "absolute";
                label.style.right = "10px";
                label.style.top = "8px";
                label.style.padding = "3px 8px";
                label.style.borderRadius = "999px";
                label.style.fontSize = "11px";
                label.style.color = "white";
                label.style.zIndex = "9999";

                if (result.label === "hate") label.style.background = "#e53935";
                else if (result.label === "offensive") label.style.background = "#fbc02d";
                else label.style.background = "#43a047";

                root.style.position = "relative";
                root.appendChild(label);

                applyBlurIfNeeded(root, result.label);
            });

        }, 500); // KEY FIX → wait for DOM render

    });
}

function processReddit() {
    const posts = document.querySelectorAll('div[data-testid="post-container"]');
    posts.forEach(post => {
        setTimeout(() => {
            const textNode = post.querySelector("h3");
            if (!textNode) return;
            const text = textNode.innerText.trim();
            if (!text || text.length < 3) return;
            if (processedTexts.has(text)) return;
            processedTexts.add(text);
            if (post.querySelector(".civicguard-label")) return;
            fetch("http://127.0.0.1:8000/analyze", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ text })
            })
            .then(res => res.json())
            .then(result => {
                if (post.querySelector(".civicguard-label")) return;
                const label = document.createElement("span");
                label.className = "civicguard-label";
                label.innerText = result.label.toUpperCase();
                label.style.marginLeft = "8px";
                label.style.padding = "3px 8px";
                label.style.borderRadius = "999px";
                label.style.fontSize = "11px";
                label.style.color = "white";
                if (result.label === "hate") label.style.background = "#e53935";
                else if (result.label === "offensive") label.style.background = "#fbc02d";
                else label.style.background = "#43a047";
                textNode.appendChild(label);
                applyBlurIfNeeded(post, result.label);
            });
        }, 500);
    });

    const comments = document.querySelectorAll('div[data-testid="comment"]');
    comments.forEach(comment => {
        setTimeout(() => {
            const textNode = comment.querySelector("p");
            if (!textNode) return;
            const text = textNode.innerText.trim();
            if (!text || text.length < 3) return;
            if (processedTexts.has(text)) return;
            processedTexts.add(text);
            if (comment.querySelector(".civicguard-label")) return;
            fetch("http://127.0.0.1:8000/analyze", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ text })
            })
            .then(res => res.json())
            .then(result => {
                if (comment.querySelector(".civicguard-label")) return;
                const label = document.createElement("span");
                label.className = "civicguard-label";
                label.innerText = result.label.toUpperCase();
                label.style.marginLeft = "8px";
                label.style.padding = "2px 6px";
                label.style.borderRadius = "999px";
                label.style.fontSize = "10px";
                label.style.color = "white";
                if (result.label === "hate") label.style.background = "#e53935";
                else if (result.label === "offensive") label.style.background = "#fbc02d";
                else label.style.background = "#43a047";
                textNode.appendChild(label);
                applyBlurIfNeeded(textNode, result.label);
            });
        }, 500);
    });
}

function processSingleInstagram(root) {

    const spans = root.querySelectorAll("span");
    let text = "";

    spans.forEach(s => {
        const t = s.innerText?.trim();

        // skip unwanted UI text
        if (
            !t ||
            t.length < 2 ||
            t === "Reply" ||
            t.includes("Reply") ||
            t.includes("See translation") ||
            t.includes("View replies") ||
            t.includes("likes") ||
            t.includes("Like") ||
            t.includes("Follow")
        ) return;

        text += t + " ";
    });

    text = text.trim();

    if (!text || text.length < 3) return;

    if (root.querySelector(".civicguard-label")) return;

    fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text })
    })
    .then(res => res.json())
    .then(result => {

        if (root.querySelector(".civicguard-label")) return;

        const label = document.createElement("span");
        label.className = "civicguard-label";
        label.innerText = result.label.toUpperCase();

        label.style.position = "absolute";
        label.style.right = "10px";
        label.style.top = "8px";
        label.style.padding = "3px 8px";
        label.style.borderRadius = "999px";
        label.style.fontSize = "11px";
        label.style.color = "white";

        if (result.label === "hate") label.style.background = "#e53935";
        else if (result.label === "offensive") label.style.background = "#fbc02d";
        else label.style.background = "#43a047";

        root.style.position = "relative";
        root.appendChild(label);

        applyBlurIfNeeded(root, result.label);
    })
    .catch(err => console.error("CivicGuard Single Instagram error:", err));
}

function processSingleReddit(comment) {

    const textNode = comment.querySelector("p");

    if (!textNode) return;

    const text = textNode.innerText.trim();

    if (!text || text.length < 3) return;

    if (comment.querySelector(".civicguard-label")) return;

    fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text })
    })
    .then(res => res.json())
    .then(result => {

        if (comment.querySelector(".civicguard-label")) return;

        const label = document.createElement("span");
        label.className = "civicguard-label";
        label.innerText = result.label.toUpperCase();

        label.style.marginLeft = "8px";
        label.style.padding = "2px 6px";
        label.style.borderRadius = "999px";
        label.style.fontSize = "10px";
        label.style.color = "white";

        if (result.label === "hate") label.style.background = "#e53935";
        else if (result.label === "offensive") label.style.background = "#fbc02d";
        else label.style.background = "#43a047";

        textNode.appendChild(label);

        applyBlurIfNeeded(textNode, result.label);
    })
    .catch(err => console.error("CivicGuard Single Reddit error:", err));
}

function detectPlatform() {
    const url = location.href;

    if (url.includes("twitter.com") || url.includes("x.com")) {
        processTwitter();
    } else if (url.includes("youtube.com")) {
        processYouTube();
    } else if (url.includes("instagram.com")) {
        processInstagram();
    } else if (url.includes("reddit.com")) {
        processReddit();
    }
}

// ─── Handle Dynamic Loading (content mutation) ───────────────────────────────
const observer = new MutationObserver(() => {
    detectPlatform();
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

// ─── SPA URL Change Detection (TASK 4) ───────────────────────────────────────
// Twitter is a single-page app — URL changes don't reload the script.
// We watch for URL changes and reprocess tweets automatically.
let lastUrl = location.href;

new MutationObserver(() => {
    const currentUrl = location.href;
    if (currentUrl !== lastUrl) {
        lastUrl = currentUrl;
        console.log("CivicGuard: URL changed →", currentUrl);
        setTimeout(processTwitter, 1000);
    }
}).observe(document, { subtree: true, childList: true });

// ─── Force Execution Loop ───────────────────────────
setInterval(() => {
    detectPlatform();
}, 1500);

// ─── Initial Run ─────────────────────────────────────────────────────────────
detectPlatform();
