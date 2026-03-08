# ComfyUI-Spectrum

A custom **ComfyUI** node that brings a **Spectrum-style acceleration path for FLUX** into a real ComfyUI workflow.

This node is designed to sit directly on the **MODEL** path in a FLUX workflow:

`UNETLoader -> LoRA(s) -> Spectrum Apply Flux -> CFGGuider`

## What it does

This custom node patches the internal **FLUX** model used by ComfyUI and applies a **pre-head feature forecasting** strategy inspired by [Spectrum](https://github.com/hanjq17/Spectrum).

Instead of forecasting the final denoised output directly, it forecasts an internal image feature **right before the final FLUX head** (`final_layer`), then still applies the original head with the current conditioning. In practice, this gave much better image stability than forecasting the model output directly.

## Current scope

- **Backend:** FLUX only
- **Integration target:** ComfyUI FLUX workflows
- **Tested setup:** ComfyUI + FLUX + LoRA stack + `CFGGuider` + `RES4LYF`
- **Node type:** `MODEL -> MODEL`

This is not a universal Spectrum port for every diffusion backend. It is a targeted ComfyUI implementation for FLUX.

## Why this exists

The official Spectrum repository reports large speedups in its own pipelines. In ComfyUI, the internal FLUX execution path is different enough that a direct drop-in port is not practical.

This node adapts the idea to ComfyUI by:

- patching `Flux.forward_orig`
- forecasting the **pre-head image feature**
- keeping the original `final_layer`
- synchronizing decisions with the sigma schedule from `transformer_options`

That makes it much more stable than a naive block-skipping or output-forecasting approach.

## Installation

1. Download this repository or unzip the release.
2. Copy the folder into:

   `ComfyUI/custom_nodes/ComfyUI-Spectrum/`

3. Restart ComfyUI.

## Node included

### Spectrum Apply Flux

This is the only production node in the package.

It takes a `MODEL` input and returns a patched `MODEL`.

Place it:

- **after** your FLUX model loaders / LoRA stack
- **before** `CFGGuider`

Example:

`UNETLoader -> LoRA Loader(s) -> Spectrum Apply Flux -> CFGGuider`

## Recommended settings

These defaults were the best working settings in the tested workflow:

- `m = 4`
- `lam = 0.10`
- `mix_w = 0.50`
- `window_size = 2.0`
- `flex_window = 0.75`
- `warmup_steps = 0`
- `k_history = 16`
- `debug = false`

## Observed result in the tested workflow

In the tested setup, the best result observed was approximately:

- **baseline:** ~160 s
- **with Spectrum Apply Flux:** ~90 s

That is about a **44% speedup** while keeping the image visually coherent in the tested cases.

Your results may vary depending on:

- FLUX variant
- sampler
- scheduler
- step count
- LoRA stack
- image-to-image vs text-to-image
- other custom nodes that wrap the model

## Limitations

- FLUX only
- tuned for a specific ComfyUI execution path
- not guaranteed to match the exact speedup figures from the official Spectrum repository
- not guaranteed to behave the same on every sampler or custom extension
- this is still an adaptation, not an official upstream implementation

## Troubleshooting

### The node appears but does nothing
Make sure it is inserted on the **MODEL** line, not on latent, conditioning, or image paths.

Correct placement:

`UNETLoader -> LoRA(s) -> Spectrum Apply Flux -> CFGGuider`

### The image quality collapses
Use the recommended defaults first. More aggressive settings are not always faster or better in real ComfyUI workflows.

### It is slower than expected
That can happen if:

- your step count is very low
- your sampler interacts with the model in a way that reduces the benefit
- startup tasks or background fetches are happening during the benchmark
- your workflow differs significantly from the tested one

## Credits

- Inspired by the official **Spectrum** project: <https://github.com/hanjq17/Spectrum>
- Adapted for ComfyUI FLUX workflows

## First custom node

If this is your first custom node: nice one. This project went through several iterations before reaching a stable and useful result. The current version exists because the earlier, simpler approaches were tested against real ComfyUI behavior and then replaced with a more faithful pre-head strategy.

## License

No license file is included by default in this package. Add one before publishing publicly if you want others to reuse or modify the code under clear terms.
