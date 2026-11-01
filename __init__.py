"""ComfyUI-PhobosStream: Zero-VRAM Video & Image Diffusion Streaming Extension.

Symlink or copy this directory directly into ComfyUI/custom_nodes/
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from phobos_stream.integrations.comfyui import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError:
    from nodes import (
        PhobosStageOffloadPipelineNode,
        PhobosStreamLoaderNode,
        PhobosStreamSamplerNode,
        PhobosStreamVAEDecodeNode,
    )

    NODE_CLASS_MAPPINGS = {
        "PhobosStreamLoader": PhobosStreamLoaderNode,
        "PhobosStreamSampler": PhobosStreamSamplerNode,
        "PhobosStreamVAEDecode": PhobosStreamVAEDecodeNode,
        "PhobosStageOffloadPipeline": PhobosStageOffloadPipelineNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "PhobosStreamLoader": "🌌 PhobosStream Model Loader (Zero-VRAM)",
        "PhobosStreamSampler": "⚡ PhobosStream Diffusion Sampler (Stage-Offload)",
        "PhobosStreamVAEDecode": "🎞️ PhobosStream VAE Decoder",
        "PhobosStageOffloadPipeline": "🚀 Phobos Zero-VRAM 2K/4K Pipeline (Stage-Offload)",
    }

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
