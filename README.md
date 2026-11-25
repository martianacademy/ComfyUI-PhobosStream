# <img src="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/icon.png" width="42" align="top" alt="PhobosStream Icon" /> ComfyUI-PhobosStream

[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-v0.2.3-EE6A3E)](https://registry.comfy.org/publishers/martianacademy/nodes/comfyui-phobosstream)
[![PyPI Core](https://img.shields.io/pypi/v/phobos-stream?color=EE6A3E&label=phobos-stream)](https://pypi.org/project/phobos-stream/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/LICENSE)

**ComfyUI-PhobosStream** is a self-contained zero-VRAM stage-offloading and hook-based per-block streaming extension for ComfyUI. It enables consumer hardware — including **8GB/16GB/24GB Apple Silicon Macs** and **8GB/12GB NVIDIA/AMD GPUs** — to run massive 12B–14B+ image and video Diffusion Transformers without Out-Of-Memory (OOM) crashes.

---

## 📊 Verified Empirical Benchmarks

### A. NVIDIA CUDA GPU — 8.0 GiB Hardware VRAM Cap & 3-Tier `HYBRID_BUDGET` (Tesla T4)

* **`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (~14.5 GB BF16)** under strict **`8.00 GiB` CUDA VRAM cap**: standard `WanPipeline.to("cuda")` crashes with `CUDA out of memory`, while **PhobosStream (`hybrid_budget`)** completes 17-frame `480×320` video generation in **`4,373.1 MB` (`4.27 GiB`) Peak CUDA VRAM**.
* **24-Block FP16 DiT (4 Steps on Tesla T4)**:
  * **`cpu_offload` (Pinned + Async CUDA `record_stream` Prefetch)**: **`322.2 MB` Peak VRAM (`93.0%` saved)** in **`1.074 s`** vs `1.024 s` monolithic (`+4.9%` latency overhead, `0.0` max abs diff).
  * **`hybrid_budget` (`7 GPU + 9 CPU + 8 NVMe`)**: **`994.3 MB` Peak VRAM (`78.4%` saved)** in **`2.418 s`** (`2.34×` faster than pure `mmap_stream`, `0.0` max abs diff).
  * **`mmap_stream` (`24 NVMe`, Zero-RAM `meta`)**: **`226.1 MB` Peak VRAM (`95.1%` saved)** in `5.653 s` (`0.0` max abs diff).

### B. Apple Silicon Metal GPU — Real 19.5 GB Model Suite (`1024×1024`, 8 Steps)

Measured end-to-end in isolated subprocesses using `qwen_3_4b.safetensors` (7.5 GB) + `z_image_turbo_bf16.safetensors` (11.7 GB, 34 blocks) + `ae.safetensors` (320 MB):

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

### Option 1: Via Comfy Registry CLI
```bash
comfy node install comfyui-phobosstream
```

### Option 2: Via ComfyUI Manager
1. Open ComfyUI → **ComfyUI Manager** → **Install Custom Nodes**.
2. Search for `ComfyUI-PhobosStream` and click **Install**.
3. Restart ComfyUI.

### Option 3: Git Clone
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
   * Supports `clip_name2` (`DualCLIPLoader`), `num_frames` (image & 5D video), `weight_dtype` (`default`, `fp8_e4m3fn`), `block_stream_mode` (`stage_only`, `hybrid_budget`, `cpu_offload`, `mmap_stream`), and `vram_budget_mb` (`0` = auto hardware query).
2. **`🔌 PhobosStream Block-Streamer (Any MODEL)` (`PhobosBlockStreamPatcher`)**
   * Drop-in `MODEL -> MODEL` patcher node for any standard ComfyUI workflow (LoRAs, ControlNet, WanVideo, FLUX).
   * Attaches PyTorch pre/post forward hooks onto repeating transformer blocks across 3 tiers (`hybrid_budget` auto GPU+CPU+NVMe split, `cpu_offload` double-buffered pinned-RAM streaming, or `mmap_stream` zero-RAM `.safetensors` streaming) with dynamic `cpu`/`mps`/`cuda` device matching.
3. **`🌌 PhobosStream Model Loader (Zero-VRAM)` (`PhobosStreamLoader`)** → outputs `phobos_model`.
4. **`⚡ PhobosStream Diffusion Sampler (Stage-Offload)` (`PhobosStreamSampler`)** → takes `phobos_model`, outputs `images` (`IMAGE`) & `latents` (`PHOBOS_LATENT`).
5. **`🎞️ PhobosStream Tiled VAE Decoder` (`PhobosStreamVAEDecode`)** → decodes `PHOBOS_LATENT` to `IMAGE` using tiled VAE fallback.

---

## 📜 License

[MIT License](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/LICENSE) — 100% Free & Open-Source.
