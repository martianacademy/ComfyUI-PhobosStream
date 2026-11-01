"""ComfyUI custom nodes for PhobosStream zero-VRAM diffusion streaming."""

from pathlib import Path
import sys

try:
    from phobos_stream.integrations.comfyui.nodes import (
        PhobosStageOffloadPipelineNode,
        PhobosStreamLoaderNode,
        PhobosStreamSamplerNode,
        PhobosStreamVAEDecodeNode,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from phobos_stream.integrations.comfyui.nodes import (
        PhobosStageOffloadPipelineNode,
        PhobosStreamLoaderNode,
        PhobosStreamSamplerNode,
        PhobosStreamVAEDecodeNode,
    )

__all__ = [
    "PhobosStageOffloadPipelineNode",
    "PhobosStreamLoaderNode",
    "PhobosStreamSamplerNode",
    "PhobosStreamVAEDecodeNode",
]
