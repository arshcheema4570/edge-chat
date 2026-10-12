# Experimental IWA bundle packaging

This repository includes a packaging-only path for evaluating an Isolated Web App (IWA). It does **not** replace or change the public GitHub Pages PWA.

## Build an unsigned bundle

Run from the repository root with Node.js and npm installed:

```sh
node scripts/package-iwa.mjs
```

The script stages tracked site files, then invokes the pinned `wbn@0.0.9` tool to create an unsigned `.wbn` file in the system temporary directory. To choose a different output path:

```sh
node scripts/package-iwa.mjs --output /path/to/app-unsigned.wbn
```

The GitHub Actions workflow builds the same artifact on pull requests and when manually dispatched. Download the Actions artifact named for this repository to inspect the unsigned bundle; artifacts are retained for seven days.

## Important limits

- An unsigned `.wbn` is **not an installable IWA**. This workflow deliberately creates no key, signature, `.swbn`, application identity, update manifest, or production release.
- Do not put a private signing key or passphrase in this repository or in a public Actions artifact. A future signing/release workflow needs a unique, protected key per app and stable key custody.
- Current Chrome IWA distribution is limited to eligible/managed environments. Production deployment requires an eligible Chrome management path and a Web App Update Manifest; GitHub Pages alone does not install an IWA for users.
- IWAs require executable page resources to be included in the signed bundle. This archive is a packaging baseline, not a compatibility certification. Audit cross-origin scripts and each app's runtime behavior before signing. Model/data downloads and WebAssembly have different IWA rules and still need their own end-to-end tests.
- Existing `manifest.json`, service-worker behavior, GitHub Pages deployment, and public app URLs remain unchanged by this packaging path.

## Edge Chat-specific compatibility note

The chat worker currently imports Transformers.js from jsDelivr. The optional semantic-memory worker in the open Edge Chat change also imports LiteRT-LM from jsDelivr. IWA restrictions require these runtime modules to be packaged into the signed bundle before those features can be considered installable. This unsigned packaging step does not vendor or rewrite those dependencies.
