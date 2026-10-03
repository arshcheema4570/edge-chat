# Edge Chat — Bonsai 1.7B

Private, on-device chat running the **Bonsai 1.7B ONNX model** with **1-bit `q1` weights** through Transformers.js and WebGPU. No account, no cloud API key, and chat messages are generated locally after the model is downloaded.

## Requirements

- A current browser with WebGPU support and a working WebGPU adapter. Chrome or Edge are the best-tested targets.
- Approximately **0.3 GB** of browser storage for the quantized model; the exact download size can vary by model revision and cache state.
- HTTPS hosting, such as GitHub Pages, or localhost during development.

## How it works

1. The app starts a module Web Worker (`bonsai-worker.js`).
2. The worker checks for both the WebGPU API and a usable GPU adapter.
3. Transformers.js downloads and caches `onnx-community/Bonsai-1.7B-ONNX` with `dtype: "q1"`.
4. A one-token warm-up verifies model initialization before chat is enabled.
5. Responses stream back to the UI while generation is running.

During first load, the app shows downloaded bytes, total bytes, transfer speed, and an estimated time remaining. After the download finishes, it switches to separate **Preparing the GPU** and **GPU warm-up** stages. The progress bar may remain at 99% during this final preparation; that means the file is downloaded and the runtime is initializing it.

The app now displays a clearer adapter failure message. If WebGPU exists but no adapter is found, the device/browser cannot run this build yet. The sandbox browser cannot provide an adapter, so final hardware validation must be performed on the target phone.

## Memory controls

- **Clear current chat** removes the visible conversation and short-term context.
- **Memory button (♧)** removes durable facts stored in `localStorage` under `edgechat.ltm.v1`.
- Durable memory is extracted occasionally from conversations and is best-effort. It is never uploaded.

## Images and audio

This build intentionally does **not** expose image or audio controls. Bonsai is currently wired as a text-generation pipeline; the app has not established a compatible multimodal processor or audio-to-text path. Adding those buttons before that verification would create misleading controls.

## PWA

The manifest registers 192px and 512px icons, supports standalone portrait display, and the service worker caches the app shell and icons. The model remains a separate browser cache download and is not bundled into the service worker.

## Deploy to GitHub Pages

Push the files to a repository, then enable **Settings → Pages → Deploy from branch → main → /(root)**. The current deployment is:

https://arshcheema4570.github.io/edge-chat/

## Development

```bash
python3 -m http.server 4179
```

Open `http://localhost:4179/` in a WebGPU-capable browser.

## Technical sources

- Model: https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX
- Transformers.js: https://huggingface.co/docs/transformers.js
- WebGPU: https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
