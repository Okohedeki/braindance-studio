"""Download the walkthrough clips used by experiment 01.

Clips are by Kindel Media on Pexels (https://www.pexels.com/@kindelmedia),
used under the Pexels license (https://www.pexels.com/license/). They are
fetched rather than committed, so the repository never redistributes them.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

CLIPS = {
    "7578540": "kitchen island, sideways arc",
    "7578552": "dining table with chandelier",
    "7578546": "hallway into living room",
    "7578547": "living room through to stairs",
}
QUALITY = "uhd_3840_2160_30fps"
DEST = Path(__file__).parent / "data" / "clips"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for clip_id, label in CLIPS.items():
        out = DEST / f"{clip_id}.mp4"
        if not out.exists():
            url = f"https://videos.pexels.com/video-files/{clip_id}/{clip_id}-{QUALITY}.mp4"
            print(f"downloading {clip_id} ({label})")
            # curl uses the OS certificate store, which is more reliable than
            # Python's bundled one on some Windows installs.
            subprocess.run(["curl", "-sSfL", "-o", str(out), url], check=True)
        print(f"{clip_id}  {out.stat().st_size / 1e6:6.1f} MB  sha256 {sha256(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
