// Edge Chat inference worker — Bonsai 1.7B (1-bit) via Transformers.js + WebGPU.
// Model: onnx-community/Bonsai-1.7B-ONNX, dtype q1. Transformers.js downloads
// and caches the weights itself; progress is reported per file and aggregated.
import {
  pipeline,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.1.0/+esm";

const MODEL_ID = "onnx-community/Bonsai-1.7B-ONNX";

let generator = null;
const stoppingCriteria = new InterruptableStoppingCriteria();
const fileProgress = new Map(); // file -> { loaded, total }

function overallProgress() {
  let loaded = 0;
  let total = 0;
  for (const { l, t } of fileProgress.values()) {
    loaded += l;
    total += t;
  }
  return total > 0 ? Math.min(99, Math.round((loaded / total) * 100)) : 0;
}

async function check() {
  try {
    if (!navigator.gpu) throw new Error("WebGPU API is unavailable in this browser");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU is present, but no compatible adapter was found");
    const info = adapter.info || {};
    self.postMessage({ status: "gpu-ok", data: [info.vendor, info.architecture, info.device].filter(Boolean).join(" · ") });
  } catch {
    self.postMessage({
      status: "fatal",
      data: "WebGPU is unavailable or no compatible adapter was found. Try a current Chrome/Edge browser on a device with WebGPU enabled.",
    });
  }
}

async function load() {
  try {
    self.postMessage({ status: "loading", data: "Downloading 1-bit model…" });
    generator = await pipeline("text-generation", MODEL_ID, {
      device: "webgpu",
      dtype: "q1",
      progress_callback: (info) => {
        if (info.file) {
          fileProgress.set(info.file, {
            l: Number(info.loaded ?? 0),
            t: Number(info.total ?? 0),
          });
          self.postMessage({ status: "progress", progress: overallProgress() });
        }
      },
    });
    self.postMessage({ status: "loading", data: "Warming up…" });
    const inputs = generator.tokenizer("a");
    await generator.model.generate({ ...inputs, max_new_tokens: 1 });
    self.postMessage({ status: "progress", progress: 100 });
    self.postMessage({ status: "ready" });
  } catch (e) {
    self.postMessage({
      status: "fatal",
      data: `Couldn't start (${e?.message || "unknown error"}). Check your connection and retry.`,
    });
  }
}

async function generate(messages, { maxTokens = 1024, tag = "" } = {}) {
  stoppingCriteria.reset();
  let startTime = 0;
  let numTokens = 0;
  let tps = 0;
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => self.postMessage({ status: "update", tag, chunk, tps }),
    token_callback_function: () => {
      const now = performance.now();
      if (!startTime) startTime = now;
      numTokens++;
      if (numTokens > 1) tps = Math.round((numTokens / (now - startTime)) * 1000);
    },
  });
  self.postMessage({ status: "start", tag });
  try {
    const output = await generator(messages, {
      max_new_tokens: maxTokens,
      do_sample: true,
      temperature: 0.6,
      top_k: 40,
      top_p: 0.9,
      repetition_penalty: 1.0,
      streamer,
      stopping_criteria: stoppingCriteria,
    });
    self.postMessage({
      status: "complete",
      tag,
      text: output[0].generated_text.at(-1).content,
      tps,
    });
  } catch (e) {
    self.postMessage({ status: "error", tag, data: `Generation error: ${e?.message || "unknown error"}` });
  }
}

self.addEventListener("message", (e) => {
  const { type, messages, maxTokens, tag } = e.data ?? {};
  if (type === "check") check();
  else if (type === "load") {
    if (!generator) load();
    else self.postMessage({ status: "ready" });
  } else if (type === "generate") generate(messages, { maxTokens, tag });
  else if (type === "interrupt") stoppingCriteria.interrupt();
  else if (type === "reset") stoppingCriteria.reset();
});
