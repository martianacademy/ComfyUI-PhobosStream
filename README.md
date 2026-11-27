# <img src="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/icon.png" width="42" align="top" alt="ComfyUI-PhobosStream Icon" /> ComfyUI-PhobosStream

[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-comfyui--phobosstream-EE6A3E)](https://registry.comfy.org/publishers/martianacademy/nodes/comfyui-phobosstream)
[![PyPI Core](https://img.shields.io/pypi/v/phobos-stream?color=EE6A3E&label=phobos-stream)](https://pypi.org/project/phobos-stream/)
[![Martian Suite: DeimosStream](https://img.shields.io/badge/Twin_Engine-DeimosStream-EE6A3E)](https://github.com/martianacademy/deimos-stream)
[![Colab Demo](https://img.shields.io/badge/Colab-MiniMax--H3_Free_T4-1E293B?logo=googlecolab&logoColor=38BDF8)](https://colab.research.google.com/github/martianacademy/phobos-stream/blob/main/notebooks/MiniMax_H3_Free_T4_PhobosStream.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**ComfyUI-PhobosStream (`v1.1.1`)** is a standalone **Zero-RAM & Zero-VRAM block-streaming extension for ComfyUI** powered by [**PhobosStream**](https://github.com/martianacademy/phobos-stream) — running **`22B–33B+` (`14 GB–68 GB+`)** `.safetensors` and `.gguf` pipelines (**`LTX-2.5 22B` + `Gemma-4 12B`**, **`FLUX.2 [dev] 32B`**, **`Wan2.2-A14B` `57.2 GB` Dual-Expert MoE**, **`MiniMax-H3 33B` Video+Audio**, **`HunyuanVideo-13B`**, **`Wan2.1-14B`**) inside **`1.8–6.6 GiB` Peak VRAM** and **`~1.0–1.6 GiB` Host Heap RAM** with zero `ComfyUI-GGUF` dependency.

---

## 🏆 Verified Head-to-Head Benchmarks (Consumer GPU Caps & Zero Quality Loss)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/benchmark_comparison_dark.png" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/benchmark_comparison.png" />
    <img src="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/benchmark_comparison.png" width="100%" alt="ComfyUI-PhobosStream Consumer Hardware Performance Overview" />
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/technical_benchmarks_dark.png" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/technical_benchmarks.png" />
    <img src="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/technical_benchmarks.png" width="100%" alt="Technical Performance Across Diffusion & Video Inference Engines" />
  </picture>
</p>

> **Engineering Trade-Off Summary:** Use stock `ComfyUI --highvram` on `48–80 GB` workstation GPUs when the full pipeline fits in VRAM. Use **`ComfyUI-PhobosStream` (`MIT`)** on **`8–24 GB` consumer GPUs** and **`12–32 GB` RAM hosts/containers** to eliminate `31–41 GB` CPU heap spikes (`meta` + `MADV_DONTNEED`), cut GGUF PCIe DMA by up to `71%` (`PackedGGUFShadow` + `545.6 GB/s` single-pass `nvcc` unpack), and stream Stage-1 CLIP + Stage-2 DiT + Stage-3 3D/Audio VAEs end-to-end.

<details>
<summary><b>📊 Expand Full Reproducible Telemetry Tables (24 GB Frontier Heavyweights & 15 GB T4 Head-to-Head)</b></summary>

<br />

| Model (`Checkpoint Footprint`) | Stock ComfyUI (`24.0 GiB` Cap) | PhobosStream Peak VRAM | Host Heap (`VmData`) | Latency (`sec/step`) | Fidelity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`FLUX.2 [dev]` `32B`** (`35.5 GB` FP8 / `35.0 GB` `Q8_0.gguf`, `1024×1024`) | `CUDA OOM` (`24,488 MB`) + `34,872 MB` Heap | **`4,063.8 MB` (`3.97 GiB`)** | **`1,023.7 MB` (`1.00 GiB`, `-97.1%`)** | **`17.11 s` (`4.28 s`)** | `Q8_0` Ref (`64.5 GB` BF16 class) |
| **`FLUX.2 [dev]` `32B`** (`19.3 GB` `Q4_K_S.gguf`, `PackedGGUFShadow`) | `CUDA OOM` | **`5,935.8 MB`** *(CPU)* / **`4,063.8 MB`** *(mmap)* | **`1,069.9 MB` (`-96.9%`)** | **`6.06 s` (`1.51 s`, `2.82×`)** | **`44.04 dB` PSNR (`0.9974` cos)** |
| **`LTX-2.5` `Gemma-4 12B` Encoder** (`26.26 GB` BF16, 48 blocks) | `CUDA OOM` (`23,620 MB`) | **`4,359.9–6,578.0 MB` (`4.26–6.42 GiB`)** | Streamed (`0 MB` GPU vocab) | **`8.81 s`** | Lossless BF16 |
| **`LTX-2.5` `22B` Audio+Video** (`42.02 GB` BF16 → `1280×704 × 97f` + `48kHz`) | `CUDA OOM` (`23,867 MB`) + `41,107 MB` Heap | **`6,749.1 MB` (`6.59 GiB`)** | **`1,048.4 MB` (`1.02 GiB`, `-97.4%`)** | **`29.03 s` (`7.26 s`)** | Joint Video + Audio BF16 Ref |
| **`LTX-2.5` `22B` Audio+Video** (`22.8 GB` `Q8_0` / `13.6 GB` `Q4_K_M`) | `CUDA OOM` | **`6,796.7 MB`** / **`7,537.2 MB`** | **`1,075–1,088 MB`** | **`25.85 s`** / **`5.75 s` (`1.44 s`, `5.05×`)** | `41.61 dB` / `34.72 dB` Audio PSNR |
| **`Wan2.2-T2V-A14B` `27B` MoE** (`57.16 GB` Dual `14B` High+Low FP16) | `CUDA OOM` (`24,508 MB`) + `31,111 MB` Heap | **`2,314.7–5,409.3 MB`** | **`1,032.8 MB` (`1.01 GiB`, `-96.7%`)** | **`34.15 s` (`5.69 s`, `1.51 s` load)** | Lossless Dual-Expert Handoff |
| **`Wan2.1-T2V-14B` `FP8`** (`14.29 GB`, `12` steps on `15 GB T4`) | `14,135.9 MB` (`180.73 s`) | **`1,808.2–1,960.5 MB` (`-87%`)** | Pinned | **`178.8s` (`τ=0`) / `120.4s` (`τ=0.12`) / `93.1s` (`τ=0.22`)** | **`99.99 dB` / `42.7 dB` / `35.9 dB`** |
| **`MiniMax-H3` `33B` Video+Audio VAE** (`4.85 GB` `ViT3D` + `32 kHz` Stereo, `141f`) | `CUDA OOM` on `14.5 GiB` T4 | **`466.8 MB`** *(Video)* / **`259.6 MB`** *(Audio)* | Fits `12 GiB` VM | `38.37 s` (`hybrid_budget`) | **`0.0` Bit-Exact Diff (`96.8%` saved)** |

</details>

---

## 🏛️ System Architecture (`v1.1.1`)

<p align="center">
  <img src="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/architecture.png" width="100%" alt="PhobosStream Execution Architecture and Memory Benchmarks" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/martianacademy/ComfyUI-PhobosStream/main/assets/minimax_h3_demo.gif" width="560" alt="MiniMax-H3 141-frame video + 32kHz stereo audio streamed via ComfyUI-PhobosStream" />
  <br/>
  <sub><b>MiniMax-H3 (141 frames @ 24fps + 32 kHz Stereo Audio)</b> — 36-block <code>4.85 GB</code> ViT3D Video VAE streamed in <b>466.8 MB Peak VRAM</b>.</sub>
</p>

---

## ⚡ Installation

### Option 1: Comfy Registry CLI (Recommended)
```bash
comfy node install comfyui-phobosstream
```

### Option 2: ComfyUI Manager
Open **ComfyUI Manager → Install Custom Nodes**, search **`ComfyUI-PhobosStream`**, install, and restart ComfyUI.

### Option 3: Git Clone
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/martianacademy/ComfyUI-PhobosStream.git
pip install -r ComfyUI-PhobosStream/requirements.txt
```

---

## 🧩 Node Reference (Category: `PhobosStream` — 9 Nodes)

| # | Node | Signature | Purpose |
| :---: | :--- | :---: | :--- |
| **1** | **`PhobosZeroRAMLoader`** | `→ MODEL` | Loads `14 GB–68 GB+` `.safetensors` & `.gguf` (`Q8_0`, `Q6_K`, `Q5_K`, `Q4_K`, `Q4_1`, `Q4_0`, `BF16`, `F16`) directly onto `meta` in `~1.0–1.6 GiB` Host Heap (`mmap_stream`, `hybrid_budget`, or compressed `cpu_offload`). |
| **2** | **`PhobosCLIPStreamPatcher`** | `CLIP → CLIP` | Streams `Gemma-4 12B`, `Mistral-3`, `UMT5-XXL`, `T5-XXL`, `Qwen-VL`, and `DualCLIP` encoders block-by-block with `0 MB`-VRAM CPU giant-vocabulary lookup, FP8 `QuantizedTensor` support, and post-encode eviction. |
| **3** | **`PhobosBlockStreamPatcher`** | `MODEL → MODEL` | Drop-in 3-tier block streamer (`hybrid_budget`, `cpu_offload`, `mmap_stream`) compatible with standard ComfyUI LoRA, ControlNet, FLUX, LTX, and Wan workflows. |
| **4** | **`PhobosFirstBlockCache`** | `MODEL → MODEL` | Multi-group Zero-DMA Residual Cache (`1.50×–1.94×` faster) with automatic `cond`/`uncond` CFG isolation across `(x)`, `(img, txt)`, and `(vx, ax)` blocks. |
| **5** | **`PhobosVAEStreamPatcher`** | `VAE → VAE` | Streams heavy 3D/`ViT3D` Video VAEs (`466.8 MB` Peak VRAM) and applies crossfaded chunk decoding to 3D/4D `32–48 kHz` Stereo Audio VAEs (`259.6 MB`). |
| **6** | **`PhobosStageOffloadPipeline`** | `→ IMAGE` | All-in-one 3-stage sequential offloader (`Text Encoder → Unload → DiT → Unload → Tiled VAE`). |
| **7–9** | **`PhobosStreamLoader` / `Sampler` / `VAEDecode`** | Modular | Stage-isolated loader, sampler, and tiled VAE decode nodes. |

**Ready-to-Import Workflows (`workflows/`):** [`flux2_32b_zeroram.json`](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/workflows/flux2_32b_zeroram.json) · [`ltx25_22b_audiovideo_zeroram.json`](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/workflows/ltx25_22b_audiovideo_zeroram.json) · [`wan22_dual14b_zeroram.json`](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/workflows/wan22_dual14b_zeroram.json) · [`minimax_h3_lowvram.json`](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/workflows/minimax_h3_lowvram.json)

---

## License

Released under the [MIT License](LICENSE).
