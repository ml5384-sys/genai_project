"""Download the sentence-embedding model into ./models (run at Docker build time)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.embed import Embedder  # noqa: E402

if __name__ == "__main__":
    e = Embedder()
    print(e.model_name, e.encode(["warm-up sentence"]).shape)
