"""Start imajev's playground server (tools/imajev/scripts/playground/server.py) with
the system Python's too-old mistral_common hidden from transformers (see
objects_list.py). Arguments pass straight through; run from tools/imajev.
"""

import runpy
import sys

sys.modules.setdefault("mistral_common", None)
sys.argv = ["scripts/playground/server.py", *sys.argv[1:]]
runpy.run_path("scripts/playground/server.py", run_name="__main__")
