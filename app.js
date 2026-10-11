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
const footerNote = $("footerNote");

// ---------- Memory + feedback-learning ----------
// Short-term: last HISTORY_WINDOW messages, sent as context, wiped on clear.
// Long-term: durable user facts in localStorage, extracted by the model itself
// every EXTRACT_EVERY exchanges, keyword-retrieved into the system prompt.
// Feedback: thumbs up/down on each answer is distilled by the model into a
// one-line preference ("do more of this" / "avoid this"), stored durably and
// injected into the system prompt. This is the in-browser stand-in for RLHF:
// no weight updates happen, but future answers steer toward rated preferences.
const LTM_KEY = "edgechat.ltm.v2";
const PREF_KEY = "edgechat.prefs.v1";
const LTM_CAP = 200;
const PREF_CAP = 100;
const HISTORY_WINDOW = 12;
const EXTRACT_EVERY = 4;

const loadJSON = (key, fallback) => {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; }
  catch { return fallback; }
};
const saveJSON = (key, value) => {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
};

let facts = loadJSON(LTM_KEY, []);   // [{fact, ts}]
let prefs = loadJSON(PREF_KEY, []);  // [{text, polarity: 1|-1, ts}]
let history = [];                   // [{role, text}] — short-term, reset on clear
let exchangesSinceExtract = 0;

function keywords(text) {
  return text.toLowerCase().split(/[^a-z0-9]+/).filter((w) => w.length > 2);
}
function relevantFacts(prompt, count = 5) {
  const wanted = new Set(keywords(prompt));
  if (!wanted.size) return [];
  return facts
    .map((entry) => ({ entry, score: keywords(entry.fact).filter((w) => wanted.has(w)).length }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, count)
    .map((x) => x.entry.fact);
}
function systemPrompt(prompt) {
  const lines = ["You are Edge Chat, a private assistant serving one user. Be direct, useful, and concrete."];
  const remembered = relevantFacts(prompt);
  if (remembered.length) {
    lines.push("Things you remember about the user:\n" + remembered.map((f) => "- " + f).join("\n"));
  }
  const liked = prefs.filter((p) => p.polarity > 0).slice(-10).map((p) => p.text);
  const disliked = prefs.filter((p) => p.polarity < 0).slice(-10).map((p) => p.text);
  if (liked.length) {
    lines.push("The user rated these kinds of answers highly — do more of this:\n" + liked.map((t) => "- " + t).join("\n"));
  }
  if (disliked.length) {
    lines.push("The user disliked these kinds of answers — avoid this:\n" + disliked.map((t) => "- " + t).join("\n"));
  }
  return lines.join("\n\n");
}
async function chatOnce(system, user, maxTokens = 128) {
  const response = await fetch(`${baseUrl()}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, messages: [{ role: "system", content: system }, { role: "user", content: user }], max_tokens: maxTokens, stream: false })
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const result = await response.json();
  return (result.choices?.[0]?.message?.content || "").trim();
}
// Runs in the background after every EXTRACT_EVERY exchanges. Never blocks chat.
function maybeExtractFacts() {
  if (!connected) return;
  if (++exchangesSinceExtract < EXTRACT_EVERY) return;
  exchangesSinceExtract = 0;
  (async () => {
    try {
      const convo = history.slice(-8).map((m) => `${m.role}: ${m.text}`).join("\n");
      if (!convo.trim()) return;
      const raw = await chatOnce(
        "Extract durable facts about the user from the conversation below (name, preferences, routines, devices, goals, likes, dislikes). Reply with ONLY a JSON array of short strings, e.g. [\"prefers concise answers\"]. If there are none, reply [].",
        convo
      );
      const found = JSON.parse(raw.match(/\[[\s\S]*\]/)?.[0] || "[]");
      if (!Array.isArray(found)) return;
      const seen = new Set(facts.map((f) => f.fact.toLowerCase()));
      for (const item of found) {
        const fact = String(item).trim().slice(0, 200);
        if (fact && !seen.has(fact.toLowerCase())) {
          seen.add(fact.toLowerCase());
          facts.push({ fact, ts: Date.now() });
        }
      }
      facts = facts.slice(-LTM_CAP);
      saveJSON(LTM_KEY, facts);
      updateFooter();
    } catch { /* extraction is best-effort; never surfaces */ }
  })();
}
// Distills one vote into a durable preference line. Background, best-effort.
function learnFromFeedback(polarity, prompt, response) {
  if (!connected) return;
  (async () => {
    try {
      const line = await chatOnce(
        polarity > 0
          ? "The user gave a thumbs-up to the assistant response below. In ONE short line, state what preference this reveals about the user (e.g. \"prefers concise bullet-point answers\"). Reply with ONLY that line, no quotes, no preamble."
          : "The user gave a thumbs-down to the assistant response below. In ONE short line, state what the assistant should avoid in future (e.g. \"avoid long introductions\"). Reply with ONLY that line, no quotes, no preamble.",
        `User asked: ${prompt}\nAssistant answered: ${response}`
      );
      const text = line.split("\n")[0].trim().slice(0, 200);
      if (!text) return;
      const seen = new Set(prefs.map((p) => `${p.polarity}:${p.text.toLowerCase()}`));
      const key = `${polarity}:${text.toLowerCase()}`;
      if (!seen.has(key)) {
        prefs.push({ text, polarity, ts: Date.now() });
        prefs = prefs.slice(-PREF_CAP);
        saveJSON(PREF_KEY, prefs);
        updateFooter();
      }
    } catch { /* feedback learning is best-effort; never surfaces */ }
  })();
}
function updateFooter() {
  if (!footerNote) return;
  const parts = [`Model: ${MODEL} · LiteRT-LM · 4-bit · 344 MiB`];
  if (facts.length || prefs.length) parts.push(`${facts.length} memories · ${prefs.length} learned prefs`);
  footerNote.textContent = parts.join(" · ");
}

// ---------- Chat UI ----------
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
  return article;
}
function addFeedbackButtons(article, prompt, getResponse) {
  const row = document.createElement("div");
  row.className = "feedback";
  const up = document.createElement("button");
  up.type = "button"; up.textContent = "👍"; up.title = "Good answer — learn from this"; up.setAttribute("aria-label", "Thumbs up");
  const down = document.createElement("button");
  down.type = "button"; down.textContent = "👎"; down.title = "Bad answer — avoid this"; down.setAttribute("aria-label", "Thumbs down");
  const vote = (polarity, chosen, other) => {
    chosen.classList.add("active");
    other.classList.remove("active");
    up.disabled = true; down.disabled = true;
    learnFromFeedback(polarity, prompt, getResponse());
  };
  up.addEventListener("click", () => vote(1, up, down));
  down.addEventListener("click", () => vote(-1, down, up));
  row.append(up, down);
  article.append(row);
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
  const article = addMessage("assistant", "Thinking…");
  const responseBody = article.querySelector(".message-body");
  try {
    const messages = [{ role: "system", content: systemPrompt(prompt) }];
    for (const m of history.slice(-HISTORY_WINDOW)) messages.push({ role: m.role, content: m.text });
    const content = [{ type: "text", text: prompt }];
    if (file) content.push({ type: "image_url", image_url: { url: await imageData(file) } });
    messages.push({ role: "user", content });
    const response = await fetch(`${baseUrl()}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: MODEL, messages, max_tokens: 256, stream: false })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    const text = result.choices?.[0]?.message?.content?.trim() || "No response.";
    responseBody.textContent = text;
    addFeedbackButtons(article, prompt, () => responseBody.textContent);
    history.push({ role: "user", text: file ? `${prompt} [image attached]` : prompt });
    history.push({ role: "assistant", text });
    maybeExtractFacts();
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
  history = [];
  exchangesSinceExtract = 0;
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
updateFooter();
setEnabled(false);
connect();
