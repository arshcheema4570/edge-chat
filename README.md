# Edge Chat — private, on-device AI

Edge Chat runs text generation locally in the browser with Transformers.js and WebGPU. Chat messages and remembered facts stay in the browser; they are not sent to a cloud AI service.

## Choose a model

- **Lightest — SmolLM2 Instruct 135M (`q4f16`)** is the default when the browser estimates 8 GB RAM or less, or cannot provide an estimate. It has the smallest working set and lowest compute, but its answers are the least capable.
- **Fast — SmolLM2 Instruct 360M (`q4f16`)** is a middle option with more capability and compute than Lightest.
- **Higher capacity — Bonsai 1.7B (`q1`)** is available for devices with more headroom. It may take longer to initialize and generate, especially when free RAM is low or the GPU is integrated.

The model choice is saved locally. The app does not load a model until **Load selected model** is pressed, avoiding an automatic download and GPU allocation merely from opening the page. The first load downloads model files to the browser’s local model cache. The Lightest ONNX weight file is about 118 MB; the 360M and 1.7B variants are each about 273–277 MB on disk. Download time depends on the connection. Changing models stops the current worker and loads the selected one. The app cannot read Windows’ live free-RAM figure; `navigator.deviceMemory`, when available, is only a coarse total-memory estimate.

Normal responses are capped at 144 tokens in Lightest mode, 192 in Fast mode, and 256 in Higher capacity mode. This keeps answers concise and bounds generation time. Background memory extraction is capped at 160 tokens. All inference jobs are serialized through one worker queue so memory extraction cannot run concurrently with a user response.

## Requirements and performance

Use a current Chrome or Edge browser with WebGPU enabled and a working graphics adapter. Quantization reduces model download and storage needs, but model loading and inference still consume additional working memory. On an 8 GB computer with less than 1 GB available, close memory-heavy apps and browser tabs before loading a model; start with Lightest. If it is too limited, try Fast after freeing memory. If the browser becomes unresponsive, stop the tab rather than repeatedly retrying a larger model.

The status panel shows download progress and the WebGPU adapter when available. WebGPU behavior and speed vary by browser, graphics driver, and adapter. No benchmark is claimed across hardware.

## Memory controls

- **Clear current chat** removes visible messages and short-term context.
- **Memory button (♧)** removes durable facts saved in `localStorage` under `edgechat.ltm.v1`.
- Durable fact extraction runs occasionally in the background, is best-effort, and is queued behind other inference work.

## Images and audio

This build does not expose image or audio controls. The current models are wired for text generation only.

## PWA and deployment

The service worker caches the app shell and icons. Model weights are downloaded separately and cached by Transformers.js, not bundled into the service worker.

GitHub Pages deployment: <https://arshcheema4570.github.io/edge-chat/>

For local development:

```bash
python3 -m http.server 4179
```

Open `http://localhost:4179/` in a WebGPU-capable browser.

## Model sources

- Lightest model: <https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct>
- Fast model: <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>
- Higher-capacity model: <https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX>
- Runtime: <https://huggingface.co/docs/transformers.js>
- WebGPU: <https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API>
