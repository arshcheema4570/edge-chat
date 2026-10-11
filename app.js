const DEFAULT_ENDPOINT = "http://127.0.0.1:9379";
const MODEL = "smolvlm2-500m";
const $ = (id) => document.getElementById(id);
const endpointInput = $("endpointInput");
const connectButton = $("connectButton");
const statusText = $("statusText");
const clearButton = $("clearButton");
const chatLog = $("chatLog");
const composer = $("composer");
const promptInput = $("promptInput");
const imageInput = $("imageInput");
const sendButton = $("sendButton");

let endpoint = localStorage.getItem("edgechat.litertlm.endpoint") || DEFAULT_ENDPOINT;
let connected = false;
let busy = false;
endpointInput.value = endpoint;

function setStatus(text, state = "") {
  statusText.textContent = text;
  statusText.dataset.state = state;
}
function baseUrl() { return endpoint.replace(/\/+$/, ""); }
function addMessage(role, text, imageUrl = "") {
  $("emptyState")?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  if (imageUrl) {
    const image = document.createElement("img");
    image.src = imageUrl;
    image.alt = "Attached image";
    article.append(image);
  }
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  article.append(body);
  chatLog.append(article);
  chatLog.scrollTop = chatLog.scrollHeight;
  return body;
}
function setEnabled(enabled) {
  promptInput.disabled = !enabled;
  imageInput.disabled = !enabled;
  sendButton.disabled = !enabled || busy || (!promptInput.value.trim() && !imageInput.files[0]);
}
function imageData(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
async function connect() {
  endpoint = endpointInput.value.trim().replace(/\/+$/, "");
  if (!endpoint) return setStatus("Enter a LiteRT-LM endpoint", "error");
  localStorage.setItem("edgechat.litertlm.endpoint", endpoint);
  setStatus("Connecting…", "loading");
  connectButton.disabled = true;
  try {
    const response = await fetch(`${endpoint}/v1/models`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!data.data?.some((model) => model.id === MODEL)) throw new Error(`${MODEL} is not available`);
    connected = true;
    setStatus("Connected · SmolVLM2-500M", "ready");
    setEnabled(true);
  } catch (error) {
    connected = false;
    setStatus(`Connection failed: ${error.message}`, "error");
    setEnabled(false);
  } finally {
    connectButton.disabled = false;
  }
}
async function sendMessage(event) {
  event.preventDefault();
  if (!connected || busy) return;
  const prompt = promptInput.value.trim() || "Describe this image.";
  const file = imageInput.files[0];
  if (!file && !prompt) return;
  busy = true;
  setEnabled(false);
  const preview = file ? URL.createObjectURL(file) : "";
  addMessage("user", prompt, preview);
  const responseBody = addMessage("assistant", "Thinking…");
  try {
    const content = [{ type: "text", text: prompt }];
    if (file) content.push({ type: "image_url", image_url: { url: await imageData(file) } });
    const response = await fetch(`${baseUrl()}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content }], max_tokens: 256, stream: false })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    responseBody.textContent = result.choices?.[0]?.message?.content?.trim() || "No response.";
  } catch (error) {
    responseBody.textContent = `Error: ${error.message}`;
  } finally {
    if (preview) URL.revokeObjectURL(preview);
    promptInput.value = "";
    imageInput.value = "";
    busy = false;
    setEnabled(connected);
    promptInput.focus();
  }
}
function clearChat() {
  chatLog.replaceChildren();
  const empty = document.createElement("div");
  empty.id = "emptyState";
  empty.className = "empty-state";
  empty.innerHTML = "<span>◌</span><h2>Ready for a photo</h2><p>Attach an image, ask a question, and send.</p>";
  chatLog.append(empty);
}
connectButton.addEventListener("click", connect);
composer.addEventListener("submit", sendMessage);
clearButton.addEventListener("click", clearChat);
promptInput.addEventListener("input", () => setEnabled(connected));
imageInput.addEventListener("change", () => setEnabled(connected));
setEnabled(false);
connect();
