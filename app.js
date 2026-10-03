import { Engine } from "https://cdn.jsdelivr.net/npm/@litert-lm/core/+esm";

// EXPERIMENT (per user order, 2026-10-02): SmolVLM2-500M swapped in for an
// on-device test. Plain .litertlm bundles are expected to fail in-browser
// with "Streaming LlmExecutorMetadata section is not supported yet" (LFM2.5-VL
// did, verified on-device); Gemma 4 E2B remains the only web-packaged
// multimodal bundle known to load. Revert to it if this fails the same way.
const MODEL_URL = "https://huggingface.co/litert-community/SmolVLM2-500M/resolve/main/SmolVLM2-500M.litertlm";
const MODEL_FILE = "edge-model-smolvlm.litertlm"; // new name: don't pick up the stored Gemma bundle
const MIN_MODEL_BYTES = 300_000_000; // sanity floor: partial downloads are discarded
const MAX_RETRIES = 3;
const RETRY_DELAYS = [2000, 5000, 12000];
const DOWNLOAD_PARTS = 8; // parallel Range-request chunks: multiplies throughput on the same connection

const $ = (id) => document.getElementById(id);
const retryButton = $("retryButton");
const promptInput = $("promptInput");
const sendButton = $("sendButton");
const composer = $("composer");
const chatLog = $("chatLog");
const statusText = $("statusText");
const statusDot = $("statusDot");
const progressBar = $("progressBar");
const loadMessage = $("loadMessage");

let engine = null;
let conversation = null;
let generating = false;
let booted = false;

function setStatus(state, text, detail = "") {
  statusDot.dataset.state = state;
  statusDot.setAttribute("aria-label", text);
  statusText.textContent = text;
  if (detail) loadMessage.textContent = detail;
}
function setProgress(value) { progressBar.style.width = `${Math.max(0, Math.min(100, value))}%`; }
function showRetry(show) { retryButton.classList.toggle("hidden", !show); }

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
function resizeInput() { promptInput.style.height = "auto"; promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`; }

// ---- Local model store (Origin Private File System) -------------------------
async function opfsDir() {
  if (!navigator.storage?.getDirectory) return null;
  try { return await navigator.storage.getDirectory(); }
  catch { return null; }
}
async function storedModelFile() {
  const dir = await opfsDir();
  if (!dir) return null;
  try {
    const handle = await dir.getFileHandle(MODEL_FILE);
    const file = await handle.getFile();
    if (file.size >= MIN_MODEL_BYTES) return file;
    await dir.removeEntry(MODEL_FILE); // partial download: discard
    return null;
  } catch { return null; }
}
async function downloadModel(onProgress) {
  const dir = await opfsDir();
  const probe = await probeParallelDownload();
  if (dir && probe) return downloadModelParallel(dir, probe.total, onProgress);
  return downloadModelSingle(dir, onProgress);
}

// HEAD gives the total size (Content-Length is CORS-visible); a 1-byte Range
// request proves the server honors ranges (status is always visible, no need
// to read the non-exposed Content-Range header).
async function probeParallelDownload() {
  try {
    const head = await fetch(MODEL_URL, { method: "HEAD" });
    const total = Number(head.headers.get("content-length")) || 0;
    try { await head.arrayBuffer(); } catch { /* HEAD has no body */ }
    if (!head.ok || total <= 0) return null;
    const probe = await fetch(MODEL_URL, { headers: { Range: "bytes=0-0" } });
    const ranges = probe.status === 206;
    try { if (ranges) await probe.arrayBuffer(); else await probe.body?.cancel(); } catch { /* ignore */ }
    return ranges ? { total } : null;
  } catch { return null; }
}

async function downloadPart(dir, name, start, end, onBytes) {
  let lastError = null;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      try { await dir.removeEntry(name); } catch { /* fresh part */ }
      const res = await fetch(MODEL_URL, { headers: { Range: `bytes=${start}-${end}` } });
      if (res.status !== 206 || !res.body) throw new Error(`Part failed (HTTP ${res.status})`);
      const handle = await dir.getFileHandle(name, { create: true });
      const writable = await handle.createWritable();
      try {
        const reader = res.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          await writable.write(value);
          onBytes(value.byteLength);
        }
        await writable.close();
      } catch (error) {
        try { await writable.abort(); } catch { /* ignore */ }
        throw error;
      }
      return;
    } catch (error) {
      lastError = error;
      if (attempt < MAX_RETRIES - 1) await new Promise((r) => setTimeout(r, RETRY_DELAYS[attempt]));
    }
  }
  throw lastError;
}

async function downloadModelParallel(dir, total, onProgress) {
  const received = new Array(DOWNLOAD_PARTS).fill(0);
  let lastPct = -1;
  const report = () => {
    const pct = Math.min(100, Math.floor((received.reduce((a, b) => a + b, 0) / total) * 100));
    if (pct !== lastPct) { lastPct = pct; onProgress(pct); }
  };
  const partSize = Math.ceil(total / DOWNLOAD_PARTS);
  const partName = (i) => `${MODEL_FILE}.part${i}`;
  const cleanParts = async () => {
    for (let i = 0; i < DOWNLOAD_PARTS; i++) { try { await dir.removeEntry(partName(i)); } catch { /* ignore */ } }
  };
  try {
    await Promise.all(Array.from({ length: DOWNLOAD_PARTS }, (_, i) => {
      const start = i * partSize;
      const end = Math.min(start + partSize, total) - 1;
      return downloadPart(dir, partName(i), start, end, (n) => { received[i] += n; report(); });
    }));
    report();
    const finalHandle = await dir.getFileHandle(MODEL_FILE, { create: true });
    const out = await finalHandle.createWritable();
    try {
      for (let i = 0; i < DOWNLOAD_PARTS; i++) {
        const partFile = await (await dir.getFileHandle(partName(i))).getFile();
        await partFile.stream().pipeTo(out, { preventClose: true });
      }
      await out.close();
    } catch (error) {
      try { await out.abort(); } catch { /* ignore */ }
      throw error;
    }
    const file = await finalHandle.getFile();
    if (file.size < MIN_MODEL_BYTES) throw new Error("Download incomplete");
    await cleanParts();
    return file;
  } catch (error) {
    await cleanParts();
    try { await dir.removeEntry(MODEL_FILE); } catch { /* ignore */ }
    throw error;
  }
}

// Fallback: original single-stream download for servers without Range support.
async function downloadModelSingle(dir, onProgress) {
  const response = await fetch(MODEL_URL);
  if (!response.ok || !response.body) throw new Error(`Download failed (HTTP ${response.status})`);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  let writable = null;
  if (dir) {
    const handle = await dir.getFileHandle(MODEL_FILE, { create: true });
    writable = await handle.createWritable();
  }
  const chunks = [];
  let received = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      received += value.byteLength;
      if (writable) await writable.write(value);
      else chunks.push(value);
      if (total) onProgress(Math.round((received / total) * 100));
    }
    if (writable) {
      await writable.close();
      const file = await (await dir.getFileHandle(MODEL_FILE)).getFile();
      if (file.size < MIN_MODEL_BYTES) throw new Error("Download incomplete");
      return file;
    }
    return new Blob(chunks, { type: "application/octet-stream" });
  } catch (error) {
    try { await writable?.abort(); } catch { /* ignore */ }
    try { await dir?.removeEntry(MODEL_FILE); } catch { /* ignore */ }
    throw error;
  }
}
async function withRetry(task, label) {
  let lastError = null;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try { return await task(); }
    catch (error) {
      lastError = error;
      if (attempt < MAX_RETRIES - 1) {
        setStatus("loading", label, `Attempt ${attempt + 1} failed (${error?.message || "network error"}). Retrying…`);
        await new Promise((r) => setTimeout(r, RETRY_DELAYS[attempt]));
      }
    }
  }
  throw lastError;
}

// ---- Background boot: everything happens without user interaction -----------
async function boot() {
  if (booted) return;
  booted = true;
  showRetry(false);
  if (!navigator.gpu) {
    setStatus("error", "WebGPU unavailable", "Open this app in a current Chrome or Edge browser with WebGPU enabled.");
    return;
  }
  try { await navigator.storage?.persist?.(); } catch { /* optional */ }

  try {
    setStatus("loading", "Preparing", "Checking for the local model…");
    setProgress(4);
    let modelSource = await storedModelFile();
    if (modelSource) {
      setStatus("loading", "Loading model", "Found the stored model. Initializing…");
      setProgress(55);
    } else {
      setStatus("loading", "Downloading model", "First run downloads ~0.4 GB once, then it lives on this device.");
      modelSource = await withRetry(
        () => downloadModel((pct) => { setProgress(Math.round(pct * 0.9)); setStatus("loading", "Downloading model", `Downloaded ${pct}% — keep this tab open.`); }),
        "Downloading model"
      );
      setProgress(92);
    }
    setStatus("loading", "Starting engine", "Initializing WebGPU inference…");
    engine = await withRetry(
      () => Engine.create({ model: modelSource, mainExecutorSettings: { maxNumTokens: 8192 } }),
      "Starting engine"
    );
    conversation = await engine.createConversation({
      preface: {
        messages: [{ role: "system", content: "You are a concise, helpful assistant running privately on the user's device." }],
      },
      sessionConfig: { maxOutputTokens: 2048 },
    });
    setProgress(100);
    setStatus("ready", "Ready", "Everything runs on this device. No message leaves it.");
    setComposerEnabled(true);
    promptInput.focus();
  } catch (error) {
    engine = null;
    conversation = null;
    setProgress(0);
    setStatus("error", "Couldn't start", `${error?.message || "Unknown error"}. Check your connection and retry.`);
    showRetry(true);
  }
}

async function sendMessage(event) {
  event?.preventDefault();
  const text = promptInput.value.trim();
  if (!text || !conversation || generating) return;
  promptInput.value = "";
  resizeInput();
  addMessage("user", text);
  const responseBody = addMessage("assistant", "");
  generating = true;
  setComposerEnabled(true);
  try {
    for await (const chunk of conversation.sendMessageStreaming(text)) {
      for (const item of chunk.content ?? []) {
        if (item.type === "text") responseBody.textContent += item.text;
      }
      chatLog.scrollTop = chatLog.scrollHeight;
    }
  } catch (error) {
    responseBody.textContent = `Generation error: ${error?.message || "unknown error"}`;
  } finally {
    generating = false;
    setComposerEnabled(true);
    promptInput.focus();
  }
}
function clearConversation() {
  if (generating) return;
  conversation?.cancel?.();
  engine?.createConversation({
    preface: {
      messages: [{ role: "system", content: "You are a concise, helpful assistant running privately on the user's device." }],
    },
    sessionConfig: { maxOutputTokens: 2048 },
  }).then((c) => { conversation = c; }).catch(() => {});
  chatLog.replaceChildren();
  const state = document.createElement("div");
  state.className = "empty-state";
  state.id = "emptyState";
  state.innerHTML = '<span class="empty-icon" aria-hidden="true">◌</span><h2>Private chat, local model</h2><p>Ask for an explanation, rewrite, plan, or idea.</p>';
  chatLog.append(state);
}

document.getElementById("clearButton").addEventListener("click", clearConversation);
composer.addEventListener("submit", sendMessage);
promptInput.addEventListener("input", () => { resizeInput(); setComposerEnabled(Boolean(conversation)); });
promptInput.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); composer.requestSubmit(); } });
retryButton.addEventListener("click", () => { booted = false; boot(); });
window.addEventListener("beforeunload", () => { engine?.delete?.(); });
window.addEventListener("online", () => { if (!engine && !booted) boot(); });

// Start in the background after first paint; the UI never blocks on the model.
if ("requestIdleCallback" in window) requestIdleCallback(() => boot(), { timeout: 1500 });
else setTimeout(boot, 300);
setStatus("idle", "Starting", "The model loads in the background.");
