"""ComfyUI-PhobosStream: Standalone Zero-VRAM Video & Image Diffusion Streaming Extension.

Works out-of-the-box whether installed standalone via ComfyUI-Manager (`git clone` into
`ComfyUI/custom_nodes/ComfyUI-PhobosStream`) or linked inside a `phobos-stream` checkout.
"""

from pathlib import Path
import sys

_ext_dir = Path(__file__).resolve().parent
_dev_src_dir = _ext_dir.parent / "src"

# Prefer editable development `src/` when inside monorepo; otherwise load bundled `phobos_stream/`
if (_dev_src_dir / "phobos_stream").is_dir():
    if str(_dev_src_dir) not in sys.path:
        sys.path.insert(0, str(_dev_src_dir))
elif (_ext_dir / "phobos_stream").is_dir():
    if str(_ext_dir) not in sys.path:
        sys.path.insert(0, str(_ext_dir))

from phobos_stream.integrations.comfyui import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
