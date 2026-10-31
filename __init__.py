"""ComfyUI-PhobosStream: Zero-VRAM Video & Image Diffusion Streaming Extension.

Symlink or copy this directory directly into ComfyUI/custom_nodes/
"""

import sys
from pathlib import Path

# Add phobos_stream source path to sys.path if not installed globally
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
    # Fallback to local import if inside standalone node folder
    from nodes import (
        PhobosStreamLoaderNode,
        PhobosStreamSamplerNode,
        PhobosStreamVAEDecodeNode,
    )

    NODE_CLASS_MAPPINGS = {
        "PhobosStreamLoader": PhobosStreamLoaderNode,
        "PhobosStreamSampler": PhobosStreamSamplerNode,
        "PhobosStreamVAEDecode": PhobosStreamVAEDecodeNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "PhobosStreamLoader": "🌌 PhobosStream Model Loader (Zero-VRAM)",
        "PhobosStreamSampler": "⚡ PhobosStream Diffusion Sampler (Disk-Stream)",
        "PhobosStreamVAEDecode": "🎞️ PhobosStream VAE Decoder",
    }

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
