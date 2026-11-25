"""ComfyUI custom node re-exports for standalone ComfyUI-PhobosStream installations."""

from pathlib import Path
import sys

_ext_dir = Path(__file__).resolve().parent
_dev_src_dir = _ext_dir.parent / "src"

if (_dev_src_dir / "phobos_stream").is_dir():
    if str(_dev_src_dir) not in sys.path:
        sys.path.insert(0, str(_dev_src_dir))
elif (_ext_dir / "phobos_stream").is_dir():
    if str(_ext_dir) not in sys.path:
        sys.path.insert(0, str(_ext_dir))

from phobos_stream.integrations.comfyui.nodes import (  # noqa: E402
    PhobosBlockStreamPatcherNode,
    PhobosStageOffloadPipelineNode,
    PhobosStreamLoaderNode,
    PhobosStreamSamplerNode,
    PhobosStreamVAEDecodeNode,
)

__all__ = [
    "PhobosBlockStreamPatcherNode",
    "PhobosStageOffloadPipelineNode",
    "PhobosStreamLoaderNode",
    "PhobosStreamSamplerNode",
    "PhobosStreamVAEDecodeNode",
]
