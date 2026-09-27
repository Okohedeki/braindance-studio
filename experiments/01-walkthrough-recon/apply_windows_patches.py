"""Small Windows fixes for the pinned gsplat v1.5.3 example stack.

1. rmbrualla/pycolmap (the example's COLMAP reader) unpacks 64-bit counts with
   struct format 'L', which is 8 bytes on Linux but 4 on Windows, so every .bin
   model fails to load. 'Q' is 8 bytes everywhere. (Its .txt reader is broken
   separately by Python 2 map() calls, so the binary path is the one to fix.)
2. gsplat's examples/datasets/colmap.py matches image files by relative path,
   but builds them with os.sep while COLMAP stores '/'. Images in subfolders
   (one folder per clip) then fail to resolve on Windows.
3. SAM 3.1's tracker forces PyTorch's FlashAttention kernel when FA3 is off,
   but PyTorch's Windows wheels are built without FlashAttention, so video
   tracking stops with "No available kernel". Memory-efficient attention
   computes the same result and is allowed as a fallback.

Safe to run more than once. Run it with the reconstruction environment for
1 (it patches that environment's pycolmap); 2 and 3 patch files under tools/.
"""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def patch(path: Path, replacements):
    src = path.read_text()
    out = src
    for old, new in replacements:
        out = out.replace(old, new)
    if out != src:
        path.write_text(out)
        print(f"patched {path}")
    else:
        print(f"unchanged {path}")


if importlib.util.find_spec("pycolmap") is None:
    print("skipped pycolmap (not in this environment)")
else:
    patch(Path(importlib.util.find_spec("pycolmap.scene_manager").origin), [
        ("struct.unpack('L', f.read(8))", "struct.unpack('Q', f.read(8))"),
        ("struct.unpack('IiLL', f.read(24))", "struct.unpack('IiQQ', f.read(24))"),
        ("struct.pack('L',", "struct.pack('Q',"),
    ])

patch(REPO / "tools" / "gsplat-src" / "examples" / "datasets" / "colmap.py", [
    ("paths.append(os.path.relpath(os.path.join(dp, f), path_dir))",
     "paths.append(os.path.relpath(os.path.join(dp, f), path_dir).replace(os.sep, \"/\"))"),
])

patch(REPO / "tools" / "sam3" / "sam3" / "model" / "decoder.py", [
    ("with sdpa_kernel(SDPBackend.FLASH_ATTENTION):",
     "with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):"),
])
