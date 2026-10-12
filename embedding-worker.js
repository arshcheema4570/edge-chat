// Optional semantic memory worker for Edge Chat.
// Classic worker is required because LiteRT-LM's WASM loader uses importScripts.
// Dynamically import the SDK so this file itself remains a classic worker script.
const sdkPromise = import("https://cdn.jsdelivr.net/npm/@litert-lm/core@0.18.0/+esm");

const MODEL_URL = "https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/resolve/main/embeddinggemma-2-text-270m.litertlm";
const WASM_BASE_URL = "https://cdn.jsdelivr.net/npm/@litert-lm/core@0.18.0/wasm/";
const OUTPUT_SIZE = 256;
let engine = null;
let initializing = null;
let runtimeInitializing = null;
let truncateInputStrategy = null;

async function ensureRuntime() {
  if (!runtimeInitializing) {
    runtimeInitializing = sdkPromise.then(async ({ loadLiteRtLm }) => {
      // Emscripten otherwise derives a relative .wasm URL from this worker's
      // local script URL, which returns the app's HTML fallback instead.
      self.Module = {
        locateFile: (path) => new URL(path, WASM_BASE_URL).href,
      };
      await loadLiteRtLm(WASM_BASE_URL);
    });
  }
  return runtimeInitializing;
}

async function ensureEngine() {
  if (engine) return engine;
  if (!initializing) {
    initializing = Promise.all([sdkPromise, ensureRuntime()])
      .then(([{ Backend, EmbeddingEngine, InputOverflowStrategy }]) => {
        truncateInputStrategy = InputOverflowStrategy.TRUNCATE;
        return EmbeddingEngine.create({
          model: MODEL_URL,
          backend: Backend.GPU,
          maxInputLength: 512,
          minInputLength: 128,
        });
      });
  }
  try {
    engine = await initializing;
    return engine;
  } finally {
    initializing = null;
  }
}

self.addEventListener("message", async (event) => {
  const { type, id, text = "", purpose = "query" } = event.data ?? {};
  try {
    if (type === "init") {
      await ensureEngine();
      self.postMessage({ type: "ready", id });
      return;
    }
    if (type === "embed") {
      const model = await ensureEngine();
      const prefix = purpose === "document"
        ? "task: search result | text: "
        : "task: search query | text: ";
      const result = await model.computeEmbedding(`${prefix}${String(text).trim()}`, {
        normalize: true,
        outputSize: OUTPUT_SIZE,
        inputOverflowStrategy: truncateInputStrategy,
      });
      self.postMessage({ type: "vector", id, embedding: result.embedding });
      return;
    }
    if (type === "shutdown") {
      if (initializing) await initializing;
      const current = engine;
      engine = null;
      if (current) await current.delete();
      self.postMessage({ type: "stopped", id });
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      id,
      message: error?.message || "Embedding model operation failed.",
    });
  }
});
