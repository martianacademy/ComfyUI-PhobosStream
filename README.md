# ComfyUI-PhobosStream 🌌

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Backends](https://img.shields.io/badge/Backends-CUDA%20•%20MPS%20•%20MLX%20•%20GGUF-purple.svg)]()

**ComfyUI-PhobosStream** is a self-contained zero-VRAM stage-offloading and hook-based per-block streaming extension for ComfyUI. It enables consumer hardware — including **8GB/16GB/24GB Apple Silicon Macs** and **8GB/12GB NVIDIA/AMD GPUs** — to run massive 12B–14B+ image and video Diffusion Transformers without Out-Of-Memory (OOM) crashes.

---

## 📊 Verified Empirical Benchmarks (Real 19.5 GB Model Suite @ 1024×1024, 8 Steps)

Measured end-to-end on Apple Silicon Metal GPU in isolated subprocesses using `qwen_3_4b.safetensors` (7.5 GB) + `z_image_turbo_bf16.safetensors` (11.7 GB, 34 blocks) + `ae.safetensors` (320 MB):

| Execution Mode | Peak Metal GPU Memory | VRAM Saved | Active Resident Weights | Wall Time (`Step`) | Fidelity vs Stage-Offload BF16 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Monolithic Baseline** *(All 3 models loaded)* | **26,445.0 MB** | `0.0%` | 19,637.5 MB | `41.97 s` (`4.60 s/step`) | Reference |
| **Phobos 3-Stage Offload (BF16)** | **12,906.4 MB** | **`51.2%`** | 11,742.5 MB | `46.99 s` (`5.52 s/step`) | Full BF16 (`32.68 dB` PSNR vs non-strip VAE) |
| **Phobos 3-Stage + Q4 Quantized** | **4,814.7 MB** | **`81.8%`** | 3,310.4 MB | `69.72 s` (`8.36 s/step`) | 4-bit Quantized (`19.25 dB` PSNR) |
| **Phobos Zero-VRAM Block-Stream (Full BF16)** | **1,601.0 MB** *(DiT: **1,248.2 MB**)* | **`93.9%`** *(DiT: **95.3%**)* | **23.2 MB** (`91.6 MB` active) | `75.04 s` (`8.93 s/step`) | **Bit-Exact BF16 (`0.0000 px` diff)** |

---

## 🎯 Supported Architectures

* **Alibaba Wan2.1 (14B / 1.3B)**: 5D spatio-temporal video latents (`[B, 16, (T-1)//4+1, H//8, W//8]`).
* **Black Forest Labs FLUX.1 (12B)**: Single or DualCLIP (`clip_l` + `t5xxl`) + 16-ch latents.
* **Tencent HunyuanVideo (13B)** & **LTX-Video / CogVideoX**: 5D video latents + Tiled VAE decode.
* **Lumina2 / Z-Image-Turbo** & **SD3 / SDXL**: Dynamic 4-channel & 16-channel latent detection.

---

## 🚀 Installation

### Option 1: Via ComfyUI Manager (Recommended)
1. Open ComfyUI → **ComfyUI Manager** → **Install Custom Nodes**.
2. Search for `ComfyUI-PhobosStream` and click **Install** (fully self-contained; only standard `numpy`, `safetensors`, `pydantic` required).
3. Restart ComfyUI.

### Option 2: Git Clone
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/martianacademy/ComfyUI-PhobosStream.git
cd ComfyUI-PhobosStream
pip install -r requirements.txt
```

*(Optional)* Set `PHOBOS_EXTRA_MODELS_DIR=/path/to/shared/models` if you store `diffusion_models`, `text_encoders`, and `vae` outside ComfyUI's default `models/` directory.

---

## 🧩 Available Nodes (Category: `PhobosStream`)

1. **`🚀 Phobos Zero-VRAM 2K/4K & Video Pipeline` (`PhobosStageOffloadPipeline`)**
   * All-in-one 3-stage sequential offloader (`Text Encoder -> Unload -> DiT/UNet -> Unload -> Tiled VAE -> Unload`).
   * Supports `clip_name2` (`DualCLIPLoader`), `num_frames` (image & 5D video), `weight_dtype` (`default`, `fp8_e4m3fn`), and `block_stream_mode` (`stage_only`, `cpu_offload`, `mmap_stream`).
2. **`🔌 PhobosStream Block-Streamer (Any MODEL)` (`PhobosBlockStreamPatcher`)**
   * Drop-in `MODEL -> MODEL` patcher node for any standard ComfyUI workflow (LoRAs, ControlNet, WanVideo, FLUX).
   * Attaches PyTorch pre/post forward hooks onto repeating transformer blocks (`cpu_offload` pinned-RAM streaming or `mmap_stream` zero-RAM `.safetensors` streaming) with dynamic `cpu`/`mps`/`cuda` device matching.
3. **`🌌 PhobosStream Model Loader (Zero-VRAM)` (`PhobosStreamLoader`)** → outputs `phobos_model`.
4. **`⚡ PhobosStream Diffusion Sampler (Stage-Offload)` (`PhobosStreamSampler`)** → takes `phobos_model`, outputs `images` (`IMAGE`) & `latents` (`PHOBOS_LATENT`).
5. **`🎞️ PhobosStream Tiled VAE Decoder` (`PhobosStreamVAEDecode`)** → decodes `PHOBOS_LATENT` to `IMAGE` using tiled VAE fallback.

---

## 📜 License

MIT License — 100% Free & Open-Source.
