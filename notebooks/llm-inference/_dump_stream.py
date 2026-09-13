import json
import pathlib
import re

here = pathlib.Path(__file__).resolve().parent
base = here / "single_model_llm_serving"
out = here / "_stream_dump"
out.mkdir(exist_ok=True)

for name in ["main.ipynb", "test.ipynb"]:
    nb = json.loads((base / name).read_text(encoding="utf-8"))
    print(name, "cells", len(nb["cells"]))
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        texts = []
        for o in c.get("outputs", []):
            t = o.get("text", "")
            if isinstance(t, list):
                t = "".join(t)
            if t:
                texts.append(t)
        blob = "\n".join(texts)
        if "FLOW" in blob or ">>>" in blob or "sequence_id" in blob or "generate_stream" in src:
            safe = re.sub(r"/home/[^\s'\"]+", "<HOME>", blob)
            safe = re.sub(r"[A-Za-z]:\\\\Users\\\\[^\s\"']+", "<REPO>", safe)
            (out / f"{name.replace('.', '_')}_cell{i:02d}.txt").write_text(safe, encoding="utf-8")
            print(" wrote", name, "cell", i, "chars", len(safe))
