import { Engine } from "https://cdn.jsdelivr.net/npm/@litert-lm/core/+esm";

const MODEL_URL = "https://huggingface.co/litert-community/LFM2.5-VL-450M/resolve/main/LFM2.5-VL-450M_int4_fixB.litertlm";
const DEFAULT_SYSTEM_PROMPT = "You are a concise, helpful assistant running privately on the user's device.";
const HISTORY_KEY = "edgechat.history.v1";
const SETTINGS_KEY = "edgechat.settings.v1";
const MAX_HISTORY = 30;
const MAX_ATTACH = 1; // LFM2.5-VL supports a single image per prompt

const $ = (id) => document.getElementById(id);
const loadButton = $("loadButton");
const clearButton = $("clearButton");
const installButton = $("installButton");
const historyButton = $("historyButton");
const settingsButton = $("settingsButton");
const promptInput = $("promptInput");
const sendButton = $("sendButton");
const attachButton = $("attachButton");
const imageInput = $("imageInput");
const attachStrip = $("attachStrip");
const composer = $("composer");
const chatLog = $("chatLog");
const emptyState = $("emptyState");
const statusText = $("statusText");
const statusDot = $("statusDot");
const progressBar = $("progressBar");
const loadMessage = $("loadMessage");
const historyBackdrop = $("historyBackdrop");
const historyPanel = $("historyPanel");
const historyList = $("historyList");
const newChatButton = $("newChatButton");
const settingsBackdrop = $("settingsBackdrop");
const settingsSheet = $("settingsSheet");
const tabModel = $("tabModel");
const tabPrompt = $("tabPrompt");
const paneModel = $("paneModel");
const panePrompt = $("panePrompt");
const toastEl = $("toast");

let engine = null;
let conversation = null;
let conversationHasVision = false;
let generating = false;
let pendingImages = []; // {blob, dataUrl}
let deferredInstallPrompt = null;

function isStandalone() {
  return window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
}
function showInstallButton(show) { installButton.classList.toggle("hidden", !show || isStandalone()); }

/* ---------------- settings ---------------- */
const defaultSettings = () => ({
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
  maxOutputTokens: 2048,
  temperature: 1.0,
  topK: 64,
  topP: 0.95,
  enableThinking: false,
});
let settings = loadSettings();
function loadSettings() {
  try { return { ...defaultSettings(), ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") }; }
  catch { return defaultSettings(); }
}
function saveSettings() { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); }

/* ---------------- chat history ---------------- */
let chats = loadChats();
let currentChatId = null;
function loadChats() {
  try { const list = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); return Array.isArray(list) ? list : []; }
  catch { return []; }
}
function persistChats() {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(chats.slice(0, MAX_HISTORY))); } catch { /* storage full: drop oldest */ }
}
function currentChat() { return chats.find((c) => c.id === currentChatId) || null; }
function ensureChat() {
  let chat = currentChat();
  if (!chat) {
    chat = { id: `c${Date.now().toString(36)}`, title: "New chat", updatedAt: Date.now(), messages: [] };
    chats.unshift(chat);
    currentChatId = chat.id;
  }
  return chat;
}
function touchChat(chat, title) {
  chat.updatedAt = Date.now();
  if (title && chat.messages.length <= 2) chat.title = title.slice(0, 42) || "New chat";
  chats.sort((a, b) => b.updatedAt - a.updatedAt);
  persistChats();
  renderHistoryList();
}
const fmtDate = (ts) => new Date(ts).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

function renderHistoryList() {
  historyList.replaceChildren();
  if (!chats.length) {
    const p = document.createElement("p");
    p.className = "history-empty";
    p.textContent = "No saved chats yet.";
    historyList.append(p);
    return;
  }
  for (const chat of chats) {
    const item = document.createElement("div");
    item.className = "history-item" + (chat.id === currentChatId ? " current" : "");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "history-text";
    btn.style.cssText = "all:unset;flex:1;min-width:0;cursor:pointer;display:block";
    const title = document.createElement("span");
    title.className = "history-title";
    title.textContent = chat.title;
    const date = document.createElement("span");
    date.className = "history-date";
    date.textContent = `${fmtDate(chat.updatedAt)} · ${chat.messages.length} msgs`;
    btn.append(title, date);
    btn.addEventListener("click", () => openChat(chat.id));
    const del = document.createElement("button");
    del.type = "button";
    del.className = "history-delete";
    del.title = "Delete chat";
    del.setAttribute("aria-label", `Delete ${chat.title}`);
    del.textContent = "×";
    del.addEventListener("click", (e) => { e.stopPropagation(); deleteChat(chat.id); });
    item.append(btn, del);
    historyList.append(item);
  }
}

function renderChat(chat) {
  chatLog.replaceChildren();
  if (!chat || !chat.messages.length) { renderEmptyState(); return; }
  for (const m of chat.messages) addMessage(m.role, m, { record: false });
}

function renderEmptyState() {
  const state = document.createElement("div");
  state.className = "empty-state";
  state.id = "emptyState";
  state.innerHTML = '<span class="empty-icon" aria-hidden="true">◌</span><h2>Private chat, local model</h2><p>Ask LFM for an explanation, rewrite, plan, or idea.</p>';
  chatLog.append(state);
}

async function openChat(id) {
  const chat = chats.find((c) => c.id === id);
  if (!chat) return;
  currentChatId = id;
  conversation?.cancel?.();
  renderChat(chat);
  renderHistoryList();
  closeHistory();
  if (engine) {
    try { conversation = await newConversation(); conversationHasVision = true; }
    catch { /* keep old conversation */ }
    toast("Earlier messages shown for reference — fresh session started.");
  }
}

function deleteChat(id) {
  chats = chats.filter((c) => c.id !== id);
  if (currentChatId === id) {
    currentChatId = null;
    conversation?.cancel?.();
    renderEmptyState();
    if (engine) newConversation().then((c) => { conversation = c; conversationHasVision = true; }).catch(() => {});
  }
  persistChats();
  renderHistoryList();
}

function startNewChat() {
  conversation?.cancel?.();
  currentChatId = null;
  pendingImages = [];
  renderAttachStrip();
  renderEmptyState();
  closeHistory();
  if (engine) newConversation().then((c) => { conversation = c; conversationHasVision = true; }).catch(() => {});
}

/* ---------------- ui helpers ---------------- */
function toast(msg, ms = 2600) {
  toastEl.textContent = msg;
  toastEl.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => toastEl.classList.add("hidden"), ms);
}
function setStatus(state, text, detail = "") {
  statusDot.dataset.state = state;
  statusDot.setAttribute("aria-label", text);
  statusText.textContent = text;
  if (detail) loadMessage.textContent = detail;
}
function setProgress(value) { progressBar.style.width = `${Math.max(0, Math.min(100, value))}%`; }

function addMessage(role, opts = {}, { record = true } = {}) {
  const { text = "", images = [], tokPerSec = null, secs = null, thinking = "" } = opts;
  document.getElementById("emptyState")?.remove();
  const node = document.createElement("article");
  node.className = `message ${role}`;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "LFM";
  node.append(label);
  if (images.length) {
    const wrap = document.createElement("div");
    wrap.className = "message-images";
    for (const src of images) {
      const img = document.createElement("img");
      img.src = src;
      img.alt = "Attached image";
      wrap.append(img);
    }
    node.append(wrap);
  }
  let thinkBody = null;
  if (role === "assistant") {
    const details = document.createElement("details");
    details.className = "thinking";
    details.hidden = !thinking;
    const summary = document.createElement("summary");
    summary.textContent = "Thought process";
    thinkBody = document.createElement("div");
    thinkBody.className = "thinking-body";
    thinkBody.textContent = thinking;
    details.append(summary, thinkBody);
    node.append(details);
  }
  const body = document.createElement("span");
  body.className = "message-body";
  body.textContent = text;
  node.append(body);
  const meta = document.createElement("div");
  meta.className = "message-meta";
  if (tokPerSec != null && secs != null) meta.textContent = `⚡ ${tokPerSec.toFixed(1)} tok/s · ${secs.toFixed(1)}s`;
  node.append(meta);
  chatLog.append(node);
  chatLog.scrollTop = chatLog.scrollHeight;
  const api = {
    node, body, meta,
    setThinking(t) { if (thinkBody) { thinkBody.textContent = t; thinkBody.parentElement.hidden = !t; } },
    setMeta(tps, s) { meta.textContent = tps != null ? `⚡ ${tps.toFixed(1)} tok/s · ${s.toFixed(1)}s` : `${s.toFixed(1)}s`; },
  };
  if (record) {
    const chat = ensureChat();
    chat.messages.push({ role, text, images: [...images], thinking, tokPerSec, secs });
    touchChat(chat, role === "user" ? text : undefined);
  }
  return api;
}

function setComposerEnabled(enabled) {
  const canSend = enabled && !generating && (promptInput.value.trim() || pendingImages.length);
  promptInput.disabled = !enabled;
  sendButton.disabled = !canSend;
  attachButton.disabled = !enabled;
}
function resizeInput() { promptInput.style.height = "auto"; promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`; }

/* ---------------- image attach ---------------- */
async function processImage(file) {
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, 1024 / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * scale));
  const h = Math.max(1, Math.round(bmp.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  canvas.getContext("2d").drawImage(bmp, 0, 0, w, h);
  bmp.close?.();
  const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.85));
  const dataUrl = await new Promise((r) => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); });
  return { blob, dataUrl };
}
function renderAttachStrip() {
  attachStrip.replaceChildren();
  attachStrip.classList.toggle("hidden", !pendingImages.length);
  pendingImages.forEach((img, i) => {
    const wrap = document.createElement("div");
    wrap.className = "attach-thumb";
    const el = document.createElement("img");
    el.src = img.dataUrl;
    el.alt = `Attached image ${i + 1}`;
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "attach-remove";
    rm.textContent = "×";
    rm.setAttribute("aria-label", "Remove image");
    rm.addEventListener("click", () => { pendingImages.splice(i, 1); renderAttachStrip(); setComposerEnabled(Boolean(conversation)); });
    wrap.append(el, rm);
    attachStrip.append(wrap);
  });
}

/* ---------------- engine ---------------- */
function sessionConfig() {
  return {
    maxOutputTokens: settings.maxOutputTokens,
    visionModalityEnabled: true,
    samplerParams: { k: settings.topK, p: settings.topP, temperature: settings.temperature },
  };
}
async function newConversation() {
  const base = {
    preface: {
      messages: [{ role: "system", content: settings.systemPrompt }],
      extra_context: { enable_thinking: settings.enableThinking },
    },
  };
  try {
    return await engine.createConversation({ ...base, sessionConfig: sessionConfig() });
  } catch (error) {
    // Fall back for text-only model builds that reject the vision modality flag.
    if (/vision|modality|image/i.test(error?.message || "")) {
      const cfg = sessionConfig();
      delete cfg.visionModalityEnabled;
      return engine.createConversation({ ...base, sessionConfig: cfg });
    }
    throw error;
  }
}

async function loadModel() {
  if (engine || loadButton.disabled) return;
  if (!navigator.gpu) {
    setStatus("error", "WebGPU unavailable", "Open this app in a current Chrome or Edge browser with WebGPU enabled.");
    return;
  }
  loadButton.disabled = true;
  loadButton.textContent = "Loading…";
  setStatus("loading", "Loading model", "Downloading the model from Hugging Face. This happens once per browser cache.");
  setProgress(12);
  try {
    engine = await Engine.create({ model: MODEL_URL, mainExecutorSettings: { maxNumTokens: 8192 } });
    setProgress(82);
    conversation = await newConversation();
    conversationHasVision = true;
    setProgress(100);
    setStatus("ready", "Ready", "Inference runs locally in this tab. No message is sent to a server.");
    loadButton.textContent = "Model loaded";
    setComposerEnabled(true);
    promptInput.focus();
  } catch (error) {
    engine = null;
    conversation = null;
    loadButton.disabled = false;
    loadButton.textContent = "Retry model load";
    setProgress(0);
    setStatus("error", "Load failed", error?.message || "The model could not be initialized.");
  }
}

async function sendMessage(event) {
  event?.preventDefault();
  const text = promptInput.value.trim();
  if ((!text && !pendingImages.length) || !conversation || generating) return;
  const images = pendingImages;
  pendingImages = [];
  renderAttachStrip();
  promptInput.value = "";
  resizeInput();
  addMessage("user", { text, images: images.map((i) => i.dataUrl) });
  const reply = addMessage("assistant", { text: "" });
  generating = true;
  setComposerEnabled(true);
  const t0 = performance.now();
  let lastUi = 0;
  try {
    let message = text;
    if (images.length) {
      const parts = [{ type: "text", text: text || "Describe the attached image(s)." }];
      for (const img of images) parts.push({ type: "image", data: img.blob });
      message = { role: "user", content: parts };
    }
    let thinking = "";
    for await (const chunk of conversation.sendMessageStreaming(message)) {
      for (const item of chunk.content ?? []) {
        if (item.type === "text") reply.body.textContent += item.text;
      }
      const channels = chunk.channels ?? {};
      for (const [key, value] of Object.entries(channels)) {
        if (typeof value === "string" && value && /think/i.test(key)) {
          thinking += value;
          reply.setThinking(thinking);
        }
      }
      const now = performance.now();
      if (now - lastUi > 500) {
        lastUi = now;
        const secs = (now - t0) / 1000;
        const approx = reply.body.textContent.length / 4 / Math.max(secs, 0.1);
        reply.meta.textContent = `~${approx.toFixed(1)} tok/s · ${secs.toFixed(1)}s`;
      }
      chatLog.scrollTop = chatLog.scrollHeight;
    }
    const secs = (performance.now() - t0) / 1000;
    let tps = null;
    try {
      const bench = await conversation.getBenchmarkInfo();
      if (bench && bench.lastDecodeTokensPerSecond > 0) tps = bench.lastDecodeTokensPerSecond;
    } catch { /* benchmark info unavailable */ }
    reply.setMeta(tps, secs);
    const chat = ensureChat();
    const last = chat.messages[chat.messages.length - 1];
    if (last && last.role === "assistant") {
      last.text = reply.body.textContent;
      last.thinking = thinking;
      last.tokPerSec = tps;
      last.secs = secs;
    }
    touchChat(chat);
  } catch (error) {
    const msg = /image|vision|modality/i.test(error?.message || "")
      ? "This model build does not support image input."
      : `Generation error: ${error?.message || "unknown error"}`;
    reply.body.textContent = msg;
    const chat = ensureChat();
    const last = chat.messages[chat.messages.length - 1];
    if (last && last.role === "assistant") last.text = msg;
    touchChat(chat);
  } finally {
    generating = false;
    setComposerEnabled(true);
    promptInput.focus();
  }
}

/* ---------------- history panel ---------------- */
function openHistory() { renderHistoryList(); historyBackdrop.classList.remove("hidden"); historyPanel.classList.remove("hidden"); }
function closeHistory() { historyBackdrop.classList.add("hidden"); historyPanel.classList.add("hidden"); }

/* ---------------- settings sheet ---------------- */
const cfg = {
  maxTokens: [$("cfgMaxTokens"), $("cfgMaxTokensNum")],
  topK: [$("cfgTopK"), $("cfgTopKNum")],
  topP: [$("cfgTopP"), $("cfgTopPNum")],
  temp: [$("cfgTemp"), $("cfgTempNum")],
};
function syncPair(range, num) {
  range.addEventListener("input", () => { num.value = range.value; });
  num.addEventListener("input", () => {
    const v = parseFloat(num.value);
    if (Number.isFinite(v)) range.value = Math.min(parseFloat(range.max), Math.max(parseFloat(range.min), v));
  });
}
for (const [r, n] of Object.values(cfg)) syncPair(r, n);

function openSettings() {
  cfg.maxTokens[0].value = cfg.maxTokens[1].value = settings.maxOutputTokens;
  cfg.topK[0].value = cfg.topK[1].value = settings.topK;
  cfg.topP[0].value = cfg.topP[1].value = settings.topP;
  cfg.temp[0].value = cfg.temp[1].value = settings.temperature;
  $("cfgThinking").setAttribute("aria-checked", String(settings.enableThinking));
  $("cfgSystemPrompt").value = settings.systemPrompt;
  settingsBackdrop.classList.remove("hidden");
  settingsSheet.classList.remove("hidden");
}
function closeSettings() { settingsBackdrop.classList.add("hidden"); settingsSheet.classList.add("hidden"); }
function switchTab(model) {
  tabModel.classList.toggle("active", model);
  tabPrompt.classList.toggle("active", !model);
  tabModel.setAttribute("aria-selected", String(model));
  tabPrompt.setAttribute("aria-selected", String(!model));
  paneModel.classList.toggle("hidden", !model);
  panePrompt.classList.toggle("hidden", model);
}
async function applySettings() {
  const num = (v, d) => { const n = parseFloat(v); return Number.isFinite(n) ? n : d; };
  const int = (v, d) => { const n = parseInt(v, 10); return Number.isFinite(n) ? n : d; };
  settings = {
    systemPrompt: $("cfgSystemPrompt").value.trim() || DEFAULT_SYSTEM_PROMPT,
    maxOutputTokens: Math.max(256, Math.min(8192, int(cfg.maxTokens[1].value, 2048))),
    topK: Math.max(1, Math.min(100, int(cfg.topK[1].value, 64))),
    topP: Math.max(0, Math.min(1, num(cfg.topP[1].value, 0.95))),
    temperature: Math.max(0, Math.min(2, num(cfg.temp[1].value, 1))),
    enableThinking: false, // LFM2.5-VL is a non-thinking model; toggle disabled in UI
  };
  saveSettings();
  closeSettings();
  if (engine) {
    conversation?.cancel?.();
    try {
      conversation = await newConversation();
      conversationHasVision = true;
      toast("Settings applied — fresh session started.");
    } catch (error) {
      toast(`Could not apply settings: ${error?.message || "unknown error"}`);
    }
  } else {
    toast("Settings saved — they apply when the model loads.");
  }
}

/* ---------------- events ---------------- */
loadButton.addEventListener("click", loadModel);
clearButton.addEventListener("click", startNewChat);
installButton.addEventListener("click", async () => {
  if (!deferredInstallPrompt) {
    loadMessage.textContent = "Use your browser menu and choose Install Edge Chat or Add to Home Screen.";
    return;
  }
  deferredInstallPrompt.prompt();
  await deferredInstallPrompt.userChoice;
  deferredInstallPrompt = null;
  showInstallButton(false);
});
window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  deferredInstallPrompt = event;
  showInstallButton(true);
});
window.addEventListener("appinstalled", () => { deferredInstallPrompt = null; showInstallButton(false); });
historyButton.addEventListener("click", openHistory);
newChatButton.addEventListener("click", startNewChat);
historyBackdrop.addEventListener("click", closeHistory);
settingsButton.addEventListener("click", openSettings);
settingsBackdrop.addEventListener("click", closeSettings);
$("settingsCancel").addEventListener("click", closeSettings);
$("settingsOk").addEventListener("click", applySettings);
tabModel.addEventListener("click", () => switchTab(true));
tabPrompt.addEventListener("click", () => switchTab(false));
$("cfgThinking").addEventListener("click", (e) => {
  const btn = e.currentTarget;
  btn.setAttribute("aria-checked", String(btn.getAttribute("aria-checked") !== "true"));
});
$("restorePrompt").addEventListener("click", () => { $("cfgSystemPrompt").value = DEFAULT_SYSTEM_PROMPT; });
attachButton.addEventListener("click", () => imageInput.click());
imageInput.addEventListener("change", async () => {
  const files = [...imageInput.files].slice(0, MAX_ATTACH - pendingImages.length);
  imageInput.value = "";
  for (const file of files) {
    if (!file.type.startsWith("image/")) continue;
    try { pendingImages.push(await processImage(file)); }
    catch { toast("Could not read that image."); }
  }
  renderAttachStrip();
  setComposerEnabled(Boolean(conversation));
});
composer.addEventListener("submit", sendMessage);
promptInput.addEventListener("input", () => { resizeInput(); setComposerEnabled(Boolean(conversation)); });
promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); composer.requestSubmit(); }
});
window.addEventListener("beforeunload", () => { engine?.delete?.(); });

renderHistoryList();
setStatus("idle", "Model not loaded", "Load LFM 2.5 to begin. The model runs locally with WebGPU.");
