import os
import json
from openai import OpenAI

# setup

my_api_key = ""
path_to_key = os.path.join(os.path.expanduser("~"), ".tudai-api-key")
if os.path.exists(path_to_key):
    with open(path_to_key) as keyfile:
        my_api_key = keyfile.readline().strip()
if len(my_api_key) < 1:
    print("Error: The key file '.tudai-api-key' did not contain any key. Please make sure the file exists and contains only your API key.")
    exit(1)

# load model
client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=my_api_key)
MODEL = "google/embeddinggemma-2"

# vars
IN_DIR = "poem"
OUT_DIR = "poem_emb"
MODEL = "google/embeddinggemma-2"
DIMS = 128
BATCH = 256

os.makedirs(OUT_DIR, exist_ok=True)

def embed(words):
    out = []
    for i in range(0, len(words), BATCH):
        chunk = words[i:i + BATCH]
        resp = client.embeddings.create(model=MODEL, input=chunk, dimensions=DIMS)
        out.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
    return out



def main():
    for fname in sorted(os.listdir(IN_DIR)):
        path = os.path.join(IN_DIR, fname)
        if not os.path.isfile(path) or not fname.endswith(".md"):
            continue

        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()

        records = []
        all_words = []
        for idx, line in enumerate(lines):
            words = line.split()
            if not words:
                continue
            records.append({"line_index": idx, "line": line, "words": words})
            all_words.extend(words)

        vectors = embed(all_words)

        out_path = os.path.join(OUT_DIR, "eg2_d128_" + fname[:-3] + ".jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            pos = 0
            for rec in records:
                n = len(rec["words"])
                rec["words"] = [
                    {"word": w, "embedding": v}
                    for w, v in zip(rec["words"], vectors[pos:pos + n])
                ]
                pos += n
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"{fname} -> {out_path} ({len(records)} lines, {len(all_words)} words)")


if __name__ == "__main__":
    main()
