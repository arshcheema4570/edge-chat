import { Engine } from "https://cdn.jsdelivr.net/npm/@litert-lm/core/+esm";

const MODEL_URL = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm";
const $ = (id) => document.getElementById(id);
const loadButton = $("loadButton");
const clearButton = $("clearButton");
const installButton = $("installButton");
const promptInput = $("promptInput");
const sendButton = $("sendButton");
const composer = $("composer");
const chatLog = $("chatLog");
const emptyState = $("emptyState");
const statusText = $("statusText");
const statusDot = $("statusDot");
const progressBar = $("progressBar");
const loadMessage = $("loadMessage");

let engine = null;
let conversation = null;
let generating = false;
let deferredInstallPrompt = null;

function isStandalone() {
  return window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
}
function showInstallButton(show) { installButton.classList.toggle("hidden", !show || isStandalone()); }

function setStatus(state, text, detail = "") {
  statusDot.dataset.state = state;
  statusDot.setAttribute("aria-label", text);
  statusText.textContent = text;
  if (detail) loadMessage.textContent = detail;
}
function setProgress(value) { progressBar.style.width = `${Math.max(0, Math.min(100, value))}%`; }
function addMessage(role, text = "") {
  emptyState?.remove();
  const node = document.createElement("article");
  node.className = `message ${role}`;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "Gemma";
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
}
function resizeInput() { promptInput.style.height = "auto"; promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`; }

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
    engine = await Engine.create({
      model: MODEL_URL,
      mainExecutorSettings: { maxNumTokens: 8192 },
    });
    setProgress(82);
    conversation = await engine.createConversation({
      preface: {
        messages: [{ role: "system", content: "You are a concise, helpful assistant running privately on the user's device." }],
        extra_context: { enable_thinking: false },
      },
      sessionConfig: { maxOutputTokens: 2048 },
    });
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
  conversation?.cancel?.();
  chatLog.replaceChildren();
  const state = document.createElement("div");
  state.className = "empty-state";
  state.id = "emptyState";
  state.innerHTML = '<span class="empty-icon" aria-hidden="true">◌</span><h2>Private chat, local model</h2><p>Ask Gemma for an explanation, rewrite, plan, or idea.</p>';
  chatLog.append(state);
}

loadButton.addEventListener("click", loadModel);
clearButton.addEventListener("click", clearConversation);
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
composer.addEventListener("submit", sendMessage);
promptInput.addEventListener("input", () => { resizeInput(); setComposerEnabled(Boolean(conversation)); });
promptInput.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); composer.requestSubmit(); } });
window.addEventListener("beforeunload", () => { engine?.delete?.(); });
setStatus("idle", "Model not loaded", "Load Gemma to begin. The model runs locally with WebGPU.");
