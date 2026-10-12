# Edge Chat — private, on-device AI

Edge Chat runs text generation locally in the browser with Transformers.js and WebGPU. Chat messages and remembered facts stay in the browser; they are not sent to a cloud AI service.

## Choose a chat model

- **Lightest — SmolLM2 Instruct 135M (`q4f16`)** is the default when the browser estimates 8 GB RAM or less, or cannot provide an estimate. It has the smallest working set and lowest compute, but its answers are the least capable.
- **Fast — SmolLM2 Instruct 360M (`q4f16`)** is a middle option with more capability and compute than Lightest.
- **Higher capacity — Bonsai 1.7B (`q1`)** is available for devices with more headroom. It may take longer to initialize and generate, especially when free RAM is low or the GPU is integrated.

The model choice is saved locally. The app does not load a model until **Load selected model** is pressed. First use downloads model files to the browser cache. The Lightest ONNX weight file is about 118 MB; the 360M and 1.7B variants are each about 273–277 MB on disk. Changing models stops the current inference worker and loads the selected one. The app cannot read Windows' live free-RAM figure; `navigator.deviceMemory`, when available, is only a coarse total-memory estimate.

Normal responses are capped at 144 tokens in Lightest mode, 192 in Fast mode, and 256 in Higher capacity mode. Background memory extraction is capped at 160 tokens. Chat inference is serialized in one worker.

## Optional semantic memory (RAG)

Press **Enable semantic memory** to load LiteRT-LM's EmbeddingGemma 2 **Text 270M** model and index Edge Chat's remembered facts. It is an opt-in download of about 165 MB. It does not import journal files or workspace documents; the retrieval corpus is the existing user-fact memory. The web embedder uses LiteRT-LM's WebGPU backend and requires a compatible GPU adapter; it also uses additional GPU and system memory while enabled. The embedder prepares a bounded 128–512-token text context. The text model uses the `task: search result | text:` prefix for facts and `task: search query | text:` for queries. It produces normalized 256-dimensional vectors, ranks facts by cosine similarity, and supplies the top five to the selected chat model as context.

The embedding worker starts only when explicitly enabled, remains resident for semantic queries during that tab session, and can be disabled to call the SDK's `delete()` and release its model resources. The vectors are cached locally in IndexedDB so the app can avoid re-embedding unchanged facts. Disabling semantic mode retains that local vector cache; **Forget remembered facts** clears both facts and their vectors. If IndexedDB is unavailable, vectors are kept only for the current page session. The existing keyword-overlap retrieval remains the fallback when semantic mode is off or fails.

Semantic embeddings use additional memory while enabled alongside the selected chat model. With limited available RAM, start with **Lightest**, enable semantic memory only if the browser remains responsive, and disable it before trying **Higher capacity**. The initial model download and GPU allocation can still take time. Browsers without a compatible WebGPU adapter cannot enable semantic memory; Edge Chat reports the failure and retains keyword retrieval. Semantic search improves matching by meaning; it does not guarantee that the generated answer is correct.

## Requirements and performance

Use a current Chrome or Edge browser with WebGPU enabled and a working graphics adapter. Quantization reduces model download and storage needs, but model loading and inference still consume additional working memory. On an 8 GB computer with less than 1 GB available, close memory-heavy apps and browser tabs before loading models. If the browser becomes unresponsive, stop the tab rather than repeatedly retrying larger models.

WebGPU behavior and speed vary by browser, graphics driver, and adapter. No benchmark is claimed across hardware.

## Memory controls

- **Clear current chat** removes visible messages and short-term context.
- **Memory button (♧)** removes durable facts saved in `localStorage` under `edgechat.ltm.v1`; it also clears cached semantic vectors.
- Durable fact extraction runs occasionally in the background and is best-effort. Semantic embeddings for new facts are indexed after that extraction when semantic mode is enabled.

## Images and audio

This build does not expose image or audio controls. The chat models are wired for text generation only. The EmbeddingGemma Text model is text-only as well.

## PWA and deployment

The service worker caches the app shell and worker scripts. The model files are fetched separately and are not bundled into the service worker.

GitHub Pages deployment: <https://arshcheema4570.github.io/edge-chat/>

For local development:

```bash
python3 -m http.server 4179
```

Open `http://localhost:4179/` in a WebGPU-capable browser.

## Model sources

- Lightest chat model: <https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct>
- Fast chat model: <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>
- Higher-capacity chat model: <https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX>
- Semantic embedding model: <https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm>
- LiteRT-LM JavaScript API: <https://developers.google.com/edge/litert-lm/embedding_models>
- Transformers.js: <https://huggingface.co/docs/transformers.js>
- WebGPU: <https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API>
