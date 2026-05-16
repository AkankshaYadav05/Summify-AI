// ── SVG icons ───────────────────────────────────────────────────
const ICONS = {
  zap:  `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`,
  align:`<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="17" y1="10" x2="3" y2="10"/><line x1="21" y1="6" x2="3" y2="6"/><line x1="21" y1="14" x2="3" y2="14"/><line x1="17" y1="18" x2="3" y2="18"/></svg>`,
  list: `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>`,
  clip: `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/></svg>`,
  check:`<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`,
  sun:  `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`,
  moon: `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`,
};

// ── Summary types ────────────────────────────────────────────────
const TYPES = [
  { value:"short",   label:"Short Summary", icon:"zap",  desc:"2–3 sentence overview" },
  { value:"bullets", label:"Bullet Points", icon:"list", desc:"Key points as bullets" },
  { value:"notes",   label:"Notes",         icon:"clip", desc:"Decisions & action items" },
];

const MAX = 4000;
let selectedType = "short";
let history = [];
let activeHistId = null;
let isDark = true;
let currentSource = "text";   // ← declared BEFORE any function that uses it

// ── Init dropdown ─────────────────────────────────────────────────
function buildDropdown() {
  const menu = document.getElementById("dropdown-menu");
  menu.innerHTML = TYPES.map(t => `
    <button class="dropdown-option ${t.value===selectedType?"selected":""}" onclick="selectType('${t.value}')">
      <span class="dropdown-option-icon">${ICONS[t.icon]}</span>
      <span>
        <span class="dropdown-option-label">${t.label}</span>
        <span class="dropdown-option-desc">${t.desc}</span>
      </span>
    </button>
  `).join("");
  const cur = TYPES.find(t => t.value === selectedType);
  document.getElementById("dd-label").textContent = cur.label;
  document.getElementById("dd-icon").innerHTML = ICONS[cur.icon];
}

function toggleDropdown() {
  const menu = document.getElementById("dropdown-menu");
  const chev = document.getElementById("dd-chevron");
  const open = menu.classList.toggle("open");
  chev.classList.toggle("open", open);
}

function selectType(val) {
  selectedType = val;
  buildDropdown();
  document.getElementById("dropdown-menu").classList.remove("open");
  document.getElementById("dd-chevron").classList.remove("open");
}

document.addEventListener("click", e => {
  const dd = document.getElementById("dropdown");
  if (!dd.contains(e.target)) {
    document.getElementById("dropdown-menu").classList.remove("open");
    document.getElementById("dd-chevron").classList.remove("open");
  }
});

// ── Tab switching ─────────────────────────────────────────────────
// FIX: each tab now shows ONLY its own container and hides all others.
function switchSource(event, source) {
  currentSource = source;

  // Update active tab button
  document.querySelectorAll(".source-tab").forEach(btn => btn.classList.remove("active"));
  event.currentTarget.classList.add("active");   // use currentTarget, not target

  // Hide every container first
  ["text-input-container",
   "dialogue-input-container",
   "pdf-input-container",
   "link-input-container"          // YouTube/Links tab
  ].forEach(id => hide(id));

  // Show only the matching one
  const containerMap = {
    text:     "text-input-container",
    dialogue: "dialogue-input-container",
    pdf:      "pdf-input-container",
    youtube:  "link-input-container",   // same ID as in your HTML
  };
  show(containerMap[source]);

  // Re-validate so Summarize button state is correct for the new tab
  onInputChange();
}

// ── Input validation ──────────────────────────────────────────────
// FIX: renamed to onInputChange and covers all four sources correctly.
function onInputChange() {
  let hasInput = false;

  if (currentSource === "text") {
    const val = document.getElementById("text").value.trim();
    hasInput = val.length > 0;
    // Update char counter only for text tabs
    document.getElementById("char-count").textContent =
      `${val.length.toLocaleString()} / ${MAX.toLocaleString()}`;
  } else if (currentSource === "dialogue") {
    const val = document.getElementById("dialogue").value.trim();
    hasInput = val.length > 0;
    document.getElementById("char-count").textContent =
      `${val.length.toLocaleString()} / ${MAX.toLocaleString()}`;
  } else if (currentSource === "pdf") {
    hasInput = document.getElementById("pdf-file").files.length > 0;
    document.getElementById("char-count").textContent = "";
  } else if (currentSource === "youtube") {
    hasInput = document.getElementById("link-input").value.trim().length > 0;
    document.getElementById("char-count").textContent = "";
  }

  document.getElementById("summarize-btn").disabled = !hasInput;
  updateClearBtn();
}

// Keep old name as alias so the HTML oninput="onDialogueInput()" still works
function onDialogueInput() { onInputChange(); }

function updateClearBtn() {
  let hasContent = false;
  if (currentSource === "text")     hasContent = !!document.getElementById("text")?.value;
  if (currentSource === "dialogue") hasContent = !!document.getElementById("dialogue")?.value;
  if (currentSource === "pdf")      hasContent = !!document.getElementById("pdf-file")?.files.length;
  if (currentSource === "youtube")  hasContent = !!document.getElementById("link-input")?.value;

  const hasOutput = document.getElementById("output-text").textContent.trim().length > 0;
  document.getElementById("clear-btn").disabled = !hasContent && !hasOutput;
}

// ── Theme ─────────────────────────────────────────────────────────
function toggleTheme() {
  isDark = !isDark;
  document.documentElement.setAttribute("data-theme", isDark ? "dark" : "light");
  document.getElementById("theme-icon").innerHTML = isDark ? ICONS.sun : ICONS.moon;
  document.getElementById("theme-label").textContent = isDark ? "Light mode" : "Dark mode";
}

function getLinkSource(url) {
  if (!url || !url.trim()) return "youtube";
  try {
    const parsed = new URL(url.trim());
    const hostname = parsed.hostname.toLowerCase();
    if (hostname.includes("youtube.com") || hostname.includes("youtu.be")) {
      return "youtube";
    }
    return "article";
  } catch {
    return "article";
  }
}

// ── API call ──────────────────────────────────────────────────────
async function callSummarizer(text, type, source = currentSource) {
  let res;

  if (source === "pdf") {
    const file = document.getElementById("pdf-file").files[0];
    if (!file) throw new Error("No PDF file selected");

    const formData = new FormData();
    formData.append("source", "pdf");
    formData.append("summary_type", type);
    formData.append("pdf", file);

    res = await fetch("/summarize", { method: "POST", body: formData });
  } else {
    const payload = {
      source,
      summary_type: type,
      text: source === "youtube" || source === "article" ? "" : text,
      link: source === "youtube" || source === "article" ? text : "",
    };

    res = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data.summary || "";
}

// ── Summarize ─────────────────────────────────────────────────────
async function handleSummarize() {
  let inputText = "";
  let source = currentSource;

  if (currentSource === "text") {
    inputText = document.getElementById("text").value.trim();
  } else if (currentSource === "dialogue") {
    inputText = document.getElementById("dialogue").value.trim();
  } else if (currentSource === "youtube") {
    inputText = document.getElementById("link-input").value.trim();
    source = getLinkSource(inputText);
  }

  if (currentSource !== "pdf" && !inputText) return;

  setLoading(true);
  hide("error-banner");
  hide("output-card");
  show("skeleton-card");

  try {
    const result = await callSummarizer(inputText, selectedType, source);
    showOutput(result);

    const entry = { id: Date.now(), summary: result, type: selectedType, ts: Date.now() };
    history.unshift(entry);
    if (history.length > 30) history.pop();
    activeHistId = entry.id;
    renderHistory();
  } catch (e) {
    console.error(e);
    document.getElementById("error-text").textContent = e.message || "Something went wrong. Please try again.";
    show("error-banner");
  } finally {
    setLoading(false);
    hide("skeleton-card");
  }
}

function setLoading(on) {
  const btn = document.getElementById("summarize-btn");
  if (on) {
    btn.innerHTML = `<span class="spinner"></span> Summarizing…`;
    btn.disabled = true;
  } else {
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg> Summarize`;
    onInputChange();
  }
}

function showOutput(text) {
  const typeName = TYPES.find(t => t.value === selectedType)?.label ?? selectedType;
  document.getElementById("output-text").textContent = text;
  document.getElementById("output-type-pill").textContent = typeName;
  const words = text.trim().split(/\s+/).filter(w => w.length > 0).length;
  document.getElementById("output-footer").textContent =
    `${words} words · ${text.length.toLocaleString()} characters`;
  show("output-card");
  updateClearBtn();
}

// ── Copy ──────────────────────────────────────────────────────────
function handleCopy() {
  const text = document.getElementById("output-text").textContent;
  navigator.clipboard.writeText(text);
  const btn = document.getElementById("copy-btn");
  btn.innerHTML = `${ICONS.check} Copied!`;
  btn.classList.add("copied");
  setTimeout(() => {
    btn.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
    btn.classList.remove("copied");
  }, 2000);
}

// ── Clear ─────────────────────────────────────────────────────────
function handleClear() {
  if (document.getElementById("text"))       document.getElementById("text").value = "";
  if (document.getElementById("dialogue"))   document.getElementById("dialogue").value = "";
  if (document.getElementById("link-input")) document.getElementById("link-input").value = "";
  if (document.getElementById("pdf-file"))   document.getElementById("pdf-file").value = "";

  document.getElementById("output-text").textContent = "";
  hide("output-card");
  hide("error-banner");

  onInputChange();
  activeHistId = null;
  renderHistory();
}

// ── History ───────────────────────────────────────────────────────
function timeAgo(ts) {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

function renderHistory() {
  const list  = document.getElementById("history-list");
  const empty = document.getElementById("history-empty");
  const count = document.getElementById("hist-count");

  if (history.length === 0) {
    list.innerHTML = "";
    list.appendChild(empty);
    count.classList.add("hidden");
    return;
  }

  count.textContent = history.length;
  count.classList.remove("hidden");

  list.innerHTML = history.map(item => {
    const typeLabel = TYPES.find(t => t.value === item.type)?.label ?? item.type;
    const active  = item.id === activeHistId ? " active" : "";
    const preview = item.summary.replace(/</g,"&lt;").replace(/>/g,"&gt;");
    return `
      <div class="history-item${active}" onclick="loadHistory(${item.id})">
        <p class="history-item-text">${preview}</p>
        <div class="history-item-meta">
          <span class="history-item-badge">${typeLabel}</span>
          <span>·</span>
          <span>${timeAgo(item.ts)}</span>
        </div>
        <button class="history-delete" onclick="deleteHistory(event, ${item.id})">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>`;
  }).join("");
}

function loadHistory(id) {
  const item = history.find(x => x.id === id);
  if (!item) return;
  activeHistId = id;
  selectType(item.type);
  showOutput(item.summary);
  renderHistory();
}

function deleteHistory(e, id) {
  e.stopPropagation();
  history = history.filter(x => x.id !== id);
  if (activeHistId === id) {
    activeHistId = null;
    hide("output-card");
    updateClearBtn();
  }
  renderHistory();
}

// ── Helpers ───────────────────────────────────────────────────────
function show(id) { document.getElementById(id)?.classList.remove("hidden"); }
function hide(id) { document.getElementById(id)?.classList.add("hidden"); }

// ── PDF file-input listener ───────────────────────────────────────
// FIX: removed reference to non-existent #file-name element
document.getElementById("pdf-file").addEventListener("change", function () {
  // Update the upload-box label text to show selected filename
  const label = document.querySelector(".upload-content p");
  if (label) label.textContent = this.files.length > 0 ? this.files[0].name : "Upload PDF/Docs";
  onInputChange();   // re-validate → enables Summarize button
});

// Also update link-input on typing (it has no oninput in HTML)
document.getElementById("link-input").addEventListener("input", onInputChange);

// ── Boot ──────────────────────────────────────────────────────────
buildDropdown();
document.getElementById("theme-icon").innerHTML = ICONS.sun;