// Edge Chat — Bonsai 1.7B (1-bit) on WebGPU, with self-improving memory.
//
// Memory model:
// - Short-term: sliding window of recent turns (in memory only). Pruned FIFO,
//   wiped entirely when the conversation is cleared.
// - Long-term: durable user facts in localStorage, extracted by the model
//   itself every few exchanges, deduplicated, and injected into future
//   prompts. This is what compounds: the model never changes, but the
//   facts it reasons with grow — output quality rises with use.

const STM_MAX_MESSAGES = 16; // short-term window: last 8 exchanges, FIFO
const EXTRACT_EVERY = 4; // run fact extraction every N exchanges
const MODEL_PREF_KEY = "edgechat.model.v1";
const MODEL_PROFILES = {
  light: {
    label: "SmolLM2 Instruct 135M · q4f16",
    id: "HuggingFaceTB/SmolLM2-135M-Instruct",
    dtype: "q4f16",
    maxTokens: 144,
  },
  fast: {
    label: "SmolLM2 Instruct 360M · q4f16",
    id: "HuggingFaceTB/SmolLM2-360M-Instruct",
    dtype: "q4f16",
    maxTokens: 192,
  },
  quality: {
    label: "Bonsai 1.7B · q1",
    id: "onnx-community/Bonsai-1.7B-ONNX",
    dtype: "q1",
    maxTokens: 256,
  },
};
const LTM_KEY = "edgechat.ltm.v1";
const LTM_MAX_FACTS = 200;
const LTM_TOP_K = 5; // facts injected per prompt

const $ = (id) => document.getElementById(id);
const clearButton = $("clearButton");
const memoryButton = $("memoryButton");
const retryButton = $("retryButton");
const modelChoice = $("modelChoice");
const modelGuidance = $("modelGuidance");
const loadModelButton = $("loadModelButton");
const promptInput = $("promptInput");
const sendButton = $("sendButton");
const composer = $("composer");
const chatLog = $("chatLog");
const statusText = $("statusText");
const statusDot = $("statusDot");
const progressBar = $("progressBar");
const loadMessage = $("loadMessage");
const memoryStatus = $("memoryStatus");
const downloadStats = $("downloadStats");

let worker = null;
let ready = false;
let generating = false;
let extracting = false;
let booted = false;
let activeModelKey = null;
let selectedModelKey = getPreferredModel();
let stm = []; // [{role, content}] — short-term, memory only
let exchanges = 0;
let tagSeq = 0;
const pending = new Map(); // tag -> { resolve, onChunk, acc }
let factCache = loadLtm();

// ---- UI helpers -------------------------------------------------------------
function getPreferredModel() {
  try {
    const saved = localStorage.getItem(MODEL_PREF_KEY);
    if (saved && MODEL_PROFILES[saved]) return saved;
  } catch {
    /* use the conservative default if storage is unavailable */
  }
  return Number.isFinite(navigator.deviceMemory) && navigator.deviceMemory > 8 ? "quality" : "light";
}
function updateModelGuidance() {
  const key = modelChoice.value;
  const memoryEstimate = Number.isFinite(navigator.deviceMemory)
    ? `The browser estimates about ${navigator.deviceMemory} GB total RAM.`
    : "This browser does not expose a RAM estimate.";
  const modelNote = key === "light"
    ? "Lightest uses a 135M instruction model for the lowest compute and a smaller working set; answers are much less capable."
    : key === "fast"
      ? "Fast uses a compact 360M instruction model for lower compute, with less capability than Bonsai 1.7B."
      : "Higher capacity uses Bonsai 1.7B q1. It may be significantly slower and needs more available memory.";
  modelGuidance.textContent = `${modelNote} ${memoryEstimate} Browsers cannot report current free RAM.`;
}
function updateModelControls() {
  const busy = generating || extracting || (!ready && booted);
  const needsSwitch = ready && selectedModelKey !== activeModelKey;
  modelChoice.disabled = busy;
  loadModelButton.disabled = busy || (ready && !needsSwitch);
  loadModelButton.textContent = busy
    ? "Loading model…"
    : ready
      ? needsSwitch ? "Switch model" : "Model loaded"
      : "Load selected model";
}
function setStatus(state, text, detail = "") {
  statusDot.dataset.state = state;
  statusDot.setAttribute("aria-label", text);
  statusText.textContent = text;
  if (detail) loadMessage.textContent = detail;
}
function setProgress(value) {
  progressBar.style.width = `${Math.max(0, Math.min(100, value))}%`;
}
function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}
function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "calculating time left";
  const rounded = Math.ceil(seconds);
  if (rounded < 60) return `about ${rounded}s left`;
  const minutes = Math.floor(rounded / 60);
  const secs = rounded % 60;
  return `about ${minutes}m${secs ? ` ${secs}s` : ""} left`;
}
function updateDownloadStats(m) {
  if (!m || !m.total) {
    downloadStats.textContent = "Preparing model download…";
    return;
  }
  const speed = m.bytesPerSecond > 0 ? `${formatBytes(m.bytesPerSecond)}/s` : "download complete";
  const eta = m.bytesPerSecond > 0 ? formatDuration(m.remaining) : "preparing GPU";
  downloadStats.textContent = `${formatBytes(m.loaded)} / ${formatBytes(m.total)} · ${speed} · ${eta}`;
}
function showRetry(show) {
  retryButton.classList.toggle("hidden", !show);
}
function memoryLine() {
  return factCache.length
    ? `Everything runs on this device. ${factCache.length} facts remembered.`
    : "Everything runs on this device. No message leaves it.";
}
function updateMemoryStatus() {
  memoryStatus.textContent = factCache.length
    ? `${factCache.length} remembered fact${factCache.length === 1 ? "" : "s"}`
    : "No remembered facts";
}
function addMessage(role, text = "") {
  document.getElementById("emptyState")?.remove();
  const node = document.createElement("article");
  node.className = `message ${role}`;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "Edge";
  const body = document.createElement("span");
  body.className = "message-body";
  body.textContent = text;
  node.append(label, body);
  chatLog.append(node);
  chatLog.scrollTop = chatLog.scrollHeight;
  return body;
}
function setComposerEnabled(enabled) {
  promptInput.disabled = !enabled;
  sendButton.disabled = !enabled || !promptInput.value.trim() || generating;
  if (enabled) promptInput.placeholder = "Ask something…";
}
function resizeInput() {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`;
}

// ---- Long-term memory -------------------------------------------------------
function loadLtm() {
  try {
    return JSON.parse(localStorage.getItem(LTM_KEY)) ?? [];
  } catch {
    return [];
  }
}
function saveLtm() {
  try {
    localStorage.setItem(LTM_KEY, JSON.stringify(factCache));
  } catch {
    /* storage full or unavailable: memory simply doesn't persist */
  }
}
function tokens(s) {
  return s.toLowerCase().split(/[^a-z0-9]+/).filter((w) => w.length > 3);
}
// Retrieve the facts most relevant to the current query (keyword overlap).
function relevantFacts(query, k) {
  const q = new Set(tokens(query));
  if (!q.size || !factCache.length) return [];
  return factCache
    .map((f) => ({ f, score: tokens(f.fact).filter((w) => q.has(w)).length }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || b.f.uses - a.f.uses)
    .slice(0, k)
    .map((x) => x.f);
}
function isDuplicateFact(fact) {
  const norm = fact.toLowerCase();
  const bt = new Set(tokens(norm));
  return factCache.some((f) => {
    const a = f.fact.toLowerCase();
    if (a === norm) return true;
    const at = new Set(tokens(a));
    const inter = [...at].filter((w) => bt.has(w)).length;
    return inter / Math.max(at.size, bt.size, 1) > 0.75;
  });
}
function buildSystemPrompt(query) {
  let p = "You are a concise, helpful assistant running privately on the user's device.";
  const facts = relevantFacts(query, LTM_TOP_K);
  if (facts.length) {
    facts.forEach((f) => f.uses++);
    saveLtm();
    p += "\nFacts about the user:\n" + facts.map((f) => `- ${f.fact}`).join("\n");
  }
  return p;
}
// Ask the model to distill durable facts from recent turns. Runs in the
// background after every EXTRACT_EVERY exchanges; never blocks chat.
async function maybeExtractFacts() {
  if (extracting || exchanges % EXTRACT_EVERY !== 0 || stm.length < 4) return;
  extracting = true;
  try {
    const convo = stm
      .slice(-8)
      .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
      .join("\n");
    const text = await runGenerate(
      [
        {
          role: "user",
          content:
            "Extract durable facts about the user from this conversation (preferences, goals, personal details, recurring topics). " +
            "One fact per line, each starting with '- '. If there is nothing durable, reply exactly: none\n\n" +
            convo,
        },
      ],
      { maxTokens: 160, silent: true }
    );
    if (!text || /^none\b/i.test(text.trim())) return;
    let added = 0;
    for (const line of text.split("\n")) {
      const fact = line.replace(/^[-\s*•]+/, "").trim();
      if (!fact || fact.length < 8 || fact.length > 220) continue;
      if (/^(none|no durable|no facts)/i.test(fact)) continue;
      if (isDuplicateFact(fact)) continue;
      factCache.push({ fact, addedAt: Date.now(), uses: 0 });
      added++;
    }
    if (added > 0) {
      factCache = factCache.slice(-LTM_MAX_FACTS);
      saveLtm();
      if (ready && !generating) setStatus("ready", "Ready", memoryLine());
    }
  } catch {
    /* extraction is best-effort; chat is unaffected */
  } finally {
    extracting = false;
    updateModelControls();
  }
}

// ---- Worker plumbing --------------------------------------------------------
function runGenerate(messages, { maxTokens = 256, silent = false, onChunk = null } = {}) {
  const tag = `t${++tagSeq}`;
  return new Promise((resolve) => {
    pending.set(tag, { resolve, onChunk, acc: "", silent });
    worker.postMessage({ type: "generate", messages, maxTokens, tag, silent });
  });
}
function onWorkerMessage(e) {
  const m = e.data ?? {};
  switch (m.status) {
    case "gpu-ok": {
      const profile = MODEL_PROFILES[activeModelKey] || MODEL_PROFILES.fast;
      setStatus("loading", "Downloading model", `${profile.label}. ${m.data ? `WebGPU adapter: ${m.data}. ` : ""}First use downloads model files to this device.`);
      worker.postMessage({ type: "load", modelKey: activeModelKey });
      break;
    }
    case "progress":
      setProgress(m.progress);
      updateDownloadStats(m);
      break;
    case "loading":
      setStatus("loading", "Loading model", m.data || "");
      if ((m.data || "").toLowerCase().includes("download complete")) downloadStats.textContent = "Download complete · preparing GPU…";
      else if ((m.data || "").toLowerCase().includes("gpu warm-up")) downloadStats.textContent = "GPU warm-up complete · starting chat…";
      break;
    case "ready":
      ready = true;
      setProgress(100);
      setStatus("ready", "Ready", memoryLine());
      setComposerEnabled(true);
      downloadStats.textContent = "Model ready · weights are cached locally for this browser.";
      modelChoice.disabled = false;
      updateModelControls();
      promptInput.focus();
      break;
    case "start":
      break;
    case "update": {
      const p = pending.get(m.tag);
      if (p && !p.silent) {
        p.acc += m.chunk ?? "";
        p.onChunk?.(p.acc);
      }
      break;
    }
    case "complete": {
      const p = pending.get(m.tag);
      pending.delete(m.tag);
      p?.resolve(m.text ?? "");
      break;
    }
    case "error": {
      const p = pending.get(m.tag);
      pending.delete(m.tag);
      if (p) p.resolve(""); // background/extraction failure: resolve empty, never throw
      else {
        setStatus("error", "Couldn't start", m.data || "Unknown error.");
        showRetry(true);
      }
      break;
    }
    case "fatal":
      ready = false;
      booted = false;
      activeModelKey = null;
      setProgress(0);
      downloadStats.textContent = "Model did not finish loading. Tap Retry to try again.";
      setStatus("error", "Couldn't start", m.data || "Unknown error.");
      updateModelControls();
      showRetry(true);
      break;
  }
}

// ---- Chat -------------------------------------------------------------------
async function sendMessage(event) {
  event?.preventDefault();
  const text = promptInput.value.trim();
  if (!text || !ready || generating) return;
  promptInput.value = "";
  resizeInput();
  addMessage("user", text);
  const body = addMessage("assistant", "");
  generating = true;
  setComposerEnabled(false);
  updateModelControls();
  setStatus("loading", "Thinking", "");
  const messages = [
    { role: "system", content: buildSystemPrompt(text) },
    ...stm,
    { role: "user", content: text },
  ];
  try {
    const full = await runGenerate(messages, {
      maxTokens: (MODEL_PROFILES[activeModelKey] || MODEL_PROFILES.fast).maxTokens,
      onChunk: (acc) => {
        body.textContent = acc;
        chatLog.scrollTop = chatLog.scrollHeight;
      },
    });
    if (full) {
      body.textContent = full; // authoritative final text
      stm.push({ role: "user", content: text }, { role: "assistant", content: full });
      stm = stm.slice(-STM_MAX_MESSAGES); // short-term window: FIFO prune
      exchanges++;
    } else {
      body.textContent = "Generation failed — try again.";
    }
    setStatus("ready", "Ready", memoryLine());
    maybeExtractFacts(); // background; never blocks
  } finally {
    generating = false;
    setComposerEnabled(true);
    updateModelControls();
    promptInput.focus();
  }
}
function clearConversation() {
  if (generating) return;
  stm = []; // short-term memory wiped; long-term facts persist
  worker?.postMessage({ type: "reset" });
  chatLog.replaceChildren();
  const state = document.createElement("div");
  state.className = "empty-state";
  state.id = "emptyState";
  state.innerHTML =
    '<span class="empty-icon" aria-hidden="true">◌</span><h2>Private chat, local model</h2><p>Ask for an explanation, rewrite, plan, or idea.</p>';
  chatLog.append(state);
  if (ready) setStatus("ready", "Ready", memoryLine());
}
function forgetMemory() {
  if (!factCache.length) {
    loadMessage.textContent = "There are no remembered facts to forget.";
    return;
  }
  if (!window.confirm(`Forget ${factCache.length} durable fact${factCache.length === 1 ? "" : "s"}? Your current chat will remain.`)) return;
  factCache = [];
  saveLtm();
  updateMemoryStatus();
  if (ready) setStatus("ready", "Ready", "Durable memory cleared. Current chat remains available.");
}

// ---- Boot -------------------------------------------------------------------
function boot() {
  if (booted) return;
  booted = true;
  activeModelKey = selectedModelKey;
  modelChoice.disabled = true;
  loadModelButton.disabled = true;
  loadModelButton.textContent = "Loading model…";
  showRetry(false);
  setStatus("loading", "Starting", `Checking WebGPU for ${MODEL_PROFILES[activeModelKey].label}…`);
  setProgress(4);
  try {
    worker?.terminate();
  } catch {
    /* ignore */
  }
  try {
    worker = new Worker("./bonsai-worker.js", { type: "module" });
  } catch {
    setStatus("error", "Couldn't start", "Workers unavailable in this browser.");
    showRetry(true);
    return;
  }
  worker.onmessage = onWorkerMessage;
  worker.onerror = () => {
    setProgress(0);
    setStatus("error", "Couldn't start", "Inference worker failed to load. Check your connection and retry.");
    booted = false;
    activeModelKey = null;
    updateModelControls();
    showRetry(true);
  };
  worker.postMessage({ type: "check", modelKey: activeModelKey });
}

function startSelectedModel() {
  if (generating || extracting) return;
  if (booted) {
    if (!ready || selectedModelKey === activeModelKey) return;
    worker?.terminate();
    worker = null;
    ready = false;
    booted = false;
    activeModelKey = null;
    setComposerEnabled(false);
    setProgress(0);
    setStatus("loading", "Switching model", "Stopping the current local model…");
  }
  boot();
}

clearButton.addEventListener("click", clearConversation);
memoryButton.addEventListener("click", forgetMemory);
composer.addEventListener("submit", sendMessage);
promptInput.addEventListener("input", () => {
  resizeInput();
  setComposerEnabled(ready);
});
promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});
retryButton.addEventListener("click", () => {
  booted = false;
  activeModelKey = null;
  boot();
});
loadModelButton.addEventListener("click", startSelectedModel);
modelChoice.addEventListener("change", () => {
  selectedModelKey = modelChoice.value;
  try {
    localStorage.setItem(MODEL_PREF_KEY, selectedModelKey);
  } catch {
    /* model selection still works for this page */
  }
  updateModelGuidance();
  updateModelControls();
});
window.addEventListener("beforeunload", () => {
  try {
    worker?.terminate();
  } catch {
    /* ignore */
  }
});
window.addEventListener("online", () => {
  if (!ready && booted) setStatus("loading", "Waiting for connection", "Reconnect, then tap Retry to continue model loading.");
});

updateMemoryStatus();
modelChoice.value = selectedModelKey;
updateModelGuidance();
updateModelControls();
setStatus("idle", "Not loaded", "The local model will only use memory after you press Load.");
