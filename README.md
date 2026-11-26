# <img src="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/icon.png" width="42" align="top" alt="PhobosStream Icon" /> ComfyUI-PhobosStream

[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-v0.5.0-EE6A3E)](https://registry.comfy.org/publishers/martianacademy/nodes/comfyui-phobosstream)
[![PyPI Core](https://img.shields.io/pypi/v/phobos-stream?color=EE6A3E&label=phobos-stream)](https://pypi.org/project/phobos-stream/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martianacademy/phobos-stream/blob/main/notebooks/MiniMax_H3_Free_T4_PhobosStream.ipynb)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/LICENSE)

**ComfyUI-PhobosStream** is a self-contained zero-VRAM stage-offloading and hook-based per-block streaming extension for ComfyUI. It enables consumer hardware — including **8GB/16GB/24GB Apple Silicon Macs** and **8GB/12GB/16GB NVIDIA/AMD GPUs** — to run massive **12B–33B+ image and video Diffusion Transformers (MiniMax-H3, Wan2.1, FLUX.1, HunyuanVideo)** without Out-Of-Memory (OOM) crashes.

<p align="center">
  <img src="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/minimax_h3_demo.gif" width="560" alt="MiniMax-H3 5.9s Video + 32kHz Stereo Audio generated via ComfyUI + PhobosStream on Tesla T4" />
  <br/>
  <sub><b>Live Output — MiniMax-H3 (5.88s, 141 frames @ 24fps + 32 kHz Stereo Audio) on a Free 12 GiB RAM + Tesla T4</b><br/>PhobosStream streamed the 36-block 4.85 GB <code>ViT3D</code> Video VAE in just <b>466.8 MB Peak VRAM (96.8% saved vs &gt;14.56 GiB OOM)</b>.</sub>
</p>

---

## 📊 Verified Empirical Benchmarks

### A. NVIDIA CUDA GPU — MiniMax-H3 (33B-Class), Wan2.1 (8 GiB Cap) & 3-Tier `HYBRID_BUDGET` (Tesla T4)

* **`MiniMaxAI/MiniMax-H3` (`32B Qwen3-VL` + `20B` 4-Step Turbo + `4.85 GB` 36-block `ViT3D` Video VAE + `32 kHz` Stereo Audio)** on a **Tesla T4 (`14.58 GiB` VRAM)**:
  * **Isolated `PhobosVAEStreamPatcher` Ablation (`141` frames @ `736×416`, `torch 2.11.0+cu128`, `16 GiB` RAM):** Stock `MiniMaxH3VideoVAE` crashes with **`CUDA OOM` (`> 14,300.8 MB`)**, while **`PhobosVAEStreamPatcher`** decodes all `141` frames with **`0.0` bit-exact diff**:
    * **`cpu_offload` (`0 GPU / 36 CPU`):** **`469.7 MB` Peak VRAM (`96.8%` saved)** in **`51.10 s`**
    * **`hybrid_budget 3 GB` (`21 GPU / 15 CPU`):** **`3,159.1 MB` Peak VRAM (`78.3%` saved)** in **`38.37 s`** (**`1.33×` faster**)
    * **`hybrid_budget 6 GB` (`36 GPU / 0 CPU`):** **`4,822.5 MB` Peak VRAM (`66.9%` saved)** in **`32.75 s`** (**`1.56×` faster**)
    * **4D Stereo `32 kHz` BigVGAN Audio VAE (`188,000` stereo samples):** **`259.6 MB` Peak VRAM** in **`1.95 s`**!
  * **End-to-End 3-Stage Free Colab (`12 GiB` Host RAM, `0 B` swap):** Total `25m 32s` (`74.1s` Stage 1 + `19m 01s` Stage 2 disk-offloaded Turing compute + `5m 08s` Stage 3A + `8.2s` Stage 3B).

* **Head-to-Head `Wan2.1-T2V-14B FP8` (`14.29 GB`, 40 blocks, `832×480` × `17` frames × `12` steps) vs ComfyUI Native `--lowvram` on Tesla T4 (`14.58 GiB` VRAM, `32 GiB` RAM)**:

| Engine / Mode | Peak CUDA VRAM | VRAM Saved | 12-Step Time (`sec/step`) | Speedup vs ComfyUI | Fidelity (`PSNR / Cosine`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ComfyUI Native `--lowvram`** *(Baseline)* | `14,135.9 MB` (`13.80 GiB`) | `0.0%` | `180.73 s` (`15.06 s`) | `1.00×` | Reference |
| **`PhobosBlockStreamPatcher` (`cpu_offload`, Lossless)** | **`1,808.2 MB` (`1.77 GiB`)** | **`87.2%`** | **`178.83 s` (`14.90 s`)** | **`1.01×` faster** | **`0.0` Bit-Exact (`1.0000`)** |
| **`PhobosBlockStreamPatcher` + TeaCache/FBCache (`0.12`)** | **`1,960.5 MB` (`1.91 GiB`)** | **`86.1%`** | **`120.39 s` (`10.03 s`)** | **`1.50×` faster** | **`42.67 dB` (`0.9995`)** |
| **`PhobosBlockStreamPatcher` + TeaCache/FBCache (`0.22`)** | **`1,960.5 MB` (`1.91 GiB`)** | **`86.1%`** | **`93.11 s` (`7.76 s`)** | **`1.94×` faster** | **`35.91 dB` (`0.9973`)** |

* **Dual-Stream MMDiT `HunyuanVideo-13B` Full BF16 (`25.64 GB`, `20 double_blocks + 40 single_blocks`) under enforced `24.0 GiB` RTX 3090 / 4090 VRAM Cap**:
  * Stock `--highvram` & ComfyUI `--lowvram`: ❌ **`CUDA OutOfMemoryError` (`24,534.5 MB` > `24.0 GiB`)**.
  * **`PhobosBlockStreamPatcher` (`cpu_offload`, Lossless):** ✅ **`1,807.1 MB` (`1.76 GiB`) Peak VRAM (`92.8%` less than model size)** in **`12.12 s` (`1.01 s/step`)**, **`0.0` bit-exact (`99.99 dB` PSNR)**.
  * **`PhobosBlockStreamPatcher` + Multi-Group MMDiT FBCache (`0.12` / `0.18`):** ✅ **`1,861.5 MB` Peak VRAM**, **`10.45 s` (`1.16×`, `36.85 dB` PSNR / `0.9950` cos)** and **`7.92 s` (`1.53×`, `28.05 dB` PSNR / `0.9616` cos)**!

* **`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (~14.5 GB BF16)** under strict **`8.00 GiB` CUDA VRAM cap**: standard `WanPipeline.to("cuda")` crashes with `CUDA out of memory`, while **PhobosStream (`hybrid_budget`)** completes full widescreen **`832×480` × 25-frame (40-step)** photorealistic video generation in **`3,263.9 MB` (`3.19 GiB`) Peak CUDA VRAM** (`22m 12s` on Tesla T4).
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
| **Phobos `hybrid_stream` (`6 GB` Budget — 16 GB Mac)** | **6,068.5 MB** | **`77.1%`** | 4,882.3 MB (`14/32`) | **`54.05 s`–`68.20 s`** (`50.69 s` denoise) | **BF16 (`38.00 dB` vs Block-Stream)** |
| **Phobos `hybrid_stream` (`8 GB` Budget — 16/24 GB Mac)** | **8,138.8 MB** | **`69.2%`** | 6,924.5 MB (`20/32`) | **`54.60 s` (`1.37×` faster)** | **BF16 (`38.18 dB` vs Block-Stream)** |

---

## 🎯 Supported Architectures

* **MiniMax-H3 (33B / 20B Pruned Omni Video+Audio)**: `refiner_blocks` + 36-block `4.85 GB` `ViT3D` Video VAE + `32 kHz` Stereo BigVGAN Audio VAE (see [`workflows/minimax_h3_lowvram.json`](workflows/minimax_h3_lowvram.json)).
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
   * Drop-in `MODEL -> MODEL` patcher node for any standard ComfyUI workflow (LoRAs, ControlNet, MiniMax-H3, WanVideo, FLUX).
   * Attaches PyTorch pre/post forward hooks onto repeating transformer blocks across 3 tiers (`hybrid_budget` auto GPU+CPU+NVMe split, `cpu_offload` double-buffered pinned-RAM streaming, or `mmap_stream` zero-RAM `.safetensors` streaming) + built-in `cache_rel_l1_thresh` for Zero-DMA TeaCache / First-Block Residual Caching.
3. **`⚡ PhobosStream TeaCache / First-Block Cache` (`PhobosFirstBlockCache`)**
   * Standalone or chainable `MODEL -> MODEL` step-caching accelerator (`1.50×–1.94×` speedup on `Wan2.1-14B` at `42.7 dB` / `35.9 dB` PSNR). Automatically resets across new generations and isolates `cond` / `uncond` CFG branches; skips PCIe DMA when chained with `PhobosBlockStreamPatcher`.
4. **`🎬 PhobosStream VAE Streamer (Video & Audio VAE)` (`PhobosVAEStreamPatcher`)**
   * Drop-in `VAE -> VAE` patcher node right between `VAELoader` and `VAEDecode` / `VAEDecodeAudio`.
   * Drops unused `encoder`/`quant_conv` weights on decode, streams heavy 3D/ViT3D Video VAEs (`MiniMaxH3VideoVAE`, `WanVAE`) via `HYBRID_BUDGET` or `CPU_OFFLOAD` (`469.7 MB` Peak VRAM on 36-block `4.85 GB` `ViT3D`), and automatically applies smooth linear-crossfade overlapped chunk decoding to 3D/4D Audio VAEs (`BigVGAN` `32 kHz` Stereo in `259.6 MB` / `1.95 s`).
   * Ready-to-import workflow: [`workflows/minimax_h3_lowvram.json`](workflows/minimax_h3_lowvram.json).
5. **`🌌 PhobosStream Model Loader (Zero-VRAM)` (`PhobosStreamLoader`)** → outputs `phobos_model`.
6. **`⚡ PhobosStream Diffusion Sampler (Stage-Offload)` (`PhobosStreamSampler`)** → takes `phobos_model`, outputs `images` (`IMAGE`) & `latents` (`PHOBOS_LATENT`).
7. **`🎞️ PhobosStream Tiled VAE Decoder` (`PhobosStreamVAEDecode`)** → decodes `PHOBOS_LATENT` to `IMAGE` using tiled VAE fallback.

---

## 📜 License

[MIT License](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/LICENSE) — 100% Free & Open-Source.
