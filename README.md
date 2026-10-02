# Edge Chat

Private, on-device chat in the browser powered by LiteRT-LM and WebGPU. No account, no cloud API key, no message ever leaves the device.

## Requirements

- Current Chrome or Edge with WebGPU enabled
- ~2.6 GB of local storage for the model (downloaded once, then reused)

## How it works

Open the app and everything happens in the background: the model file is
downloaded once with real progress, stored in the browser's private file
system, and loaded automatically on every later visit. There are no settings
to configure — just chat.

## Deploy to GitHub Pages

Static site. Push to a repository, then enable **Settings → Pages → Deploy
from branch → branch → /(root)**.

## Technical notes

- Uses `Engine.create()` and `engine.createConversation()` from `@litert-lm/core` via jsDelivr ESM.
- The model is passed to the engine as a locally stored `Blob`, so startup never re-downloads.
- Responses stream through `sendMessageStreaming()`.
- Only web-packaged (`-web`) `.litertlm` bundles load in the LiteRT-LM web runtime.
