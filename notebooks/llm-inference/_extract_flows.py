import json
import pathlib
import re

here = pathlib.Path(__file__).resolve().parent
nb_path = here / "single_model_llm_serving" / "main.ipynb"
out_dir = here / "_flows"
out_dir.mkdir(exist_ok=True)

nb = json.loads(nb_path.read_text(encoding="utf-8"))
blob_parts = []
for c in nb["cells"]:
    for o in c.get("outputs", []):
        t = o.get("text", "")
        if isinstance(t, list):
            t = "".join(t)
        if t:
            blob_parts.append(t)

blob = "\n".join(blob_parts)
blob = re.sub(r"/home/[^\s'\"]+", "<HOME>", blob)
blob = blob.replace("mbrdiuser", "user")

# Split on REQUEST banners if present
chunks = re.split(r"(?=REQUEST\s+POST)", blob)
for i, chunk in enumerate(chunks):
    if not chunk.strip():
        continue
    name = f"flow_{i:02d}.txt"
    m = re.search(r"REQUEST\s+POST\s+(\S+)", chunk)
    if m:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", m.group(1)).strip("_")
        name = f"{safe}_{i}.txt"
    (out_dir / name).write_text(chunk, encoding="utf-8")
    print("wrote", name, "chars", len(chunk))
