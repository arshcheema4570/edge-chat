# Edge Chat — Gemma 4 E2B

A lightweight GitHub Pages-ready chat UI for running Gemma 4 E2B locally in a browser with LiteRT-LM Web API and WebGPU.

## Requirements

- Current Chrome or Edge with WebGPU enabled
- Approximately 2.6 GB available for the web model and browser cache
- HTTPS hosting, such as GitHub Pages

## Run locally

```bash
python3 -m http.server 4179
```

Open `http://localhost:4179/` in a WebGPU-capable browser. The model is downloaded from the official LiteRT Community Hugging Face repository when **Load model** is clicked.

## Deploy to GitHub Pages

This is a static site. Push the files to a repository, then enable **Settings → Pages → Deploy from branch → main → /(root)**.

## Technical notes

- Uses `Engine.create()` and `engine.createConversation()` from `@litert-lm/core` via jsDelivr ESM.
- Uses the official web-compatible model: `gemma-4-E2B-it-web.litertlm`.
- Responses stream through `sendMessageStreaming()`.
- Inference is local after the model is downloaded; no application server or API key is used.
- LiteRT-LM Web API is an early preview and currently supports a limited set of web-compatible models.

Sources:

- https://developers.google.com/edge/litert-lm/js
- https://github.com/google-ai-edge/LiteRT-LM
- https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm
