# Edge Chat — SmolVLM2-500M via LiteRT-LM

Minimal private image chat frontend for a local **LiteRT-LM** server running the 4-bit `smolvlm2-500m` model. The browser sends images and prompts to the endpoint you configure; the model runs on that server.

## Start the LiteRT-LM server

```bash
pip install -U litert-lm
litert-lm import --from-huggingface-repo litert-community/SmolVLM2-500M SmolVLM2-500M.litertlm smolvlm2-500m
litert-lm serve --host 127.0.0.1 --port 9379 --cors-origin http://localhost:4179
```

The model bundle is approximately **344 MiB** and supports text plus one image per prompt. The server exposes an OpenAI-compatible `/v1/models` and `/v1/chat/completions` API.

## Run the frontend locally

```bash
python3 -m http.server 4179
```

Open `http://localhost:4179/`, keep the endpoint as `http://127.0.0.1:9379`, attach an image, and ask a question. The frontend remembers the endpoint in local storage.

The GitHub Pages deployment is:

https://arshcheema4570.github.io/edge-chat/

A hosted HTTPS page cannot call an arbitrary HTTP localhost endpoint in every browser because of mixed-content rules. For local use, serve the frontend over HTTP as shown above, or expose the LiteRT-LM server through an HTTPS local tunnel that allows the GitHub Pages origin.

## Model

- `smolvlm2-500m`
- LiteRT-LM `.litertlm` bundle
- 4-bit decoder, int8 vision path
- 2048-token context
- 512×512 image preprocessing
- Apache 2.0 model license
