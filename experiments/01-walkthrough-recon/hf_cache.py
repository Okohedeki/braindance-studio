"""Point Hugging Face at the repo's model cache (tools/hf), where install.sh puts the models.

A gated model (Stable Virtual Camera, SAM 3) is checked online on every load, even when it is cached, so the
login has to be found too: `hf auth login` saves it in the default cache, not in tools/hf. This points the
library at that file (it never reads it). An HF_HOME or HF_TOKEN_PATH that is already set wins.
"""

import os
from pathlib import Path


def use_repo_cache(repo: Path):
    os.environ.setdefault("HF_HOME", str(repo / "tools" / "hf"))
    default_login = Path.home() / ".cache" / "huggingface" / "token"
    if not (Path(os.environ["HF_HOME"]) / "token").exists() and default_login.exists():
        os.environ.setdefault("HF_TOKEN_PATH", str(default_login))
