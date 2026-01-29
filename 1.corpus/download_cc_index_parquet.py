import os
import sys
import urllib.request

BASE_URL = "https://data.commoncrawl.org/"
PATH_FILE = "cc-index-table.paths"
OUT_DIR = "cc_index_parquet"

os.makedirs(OUT_DIR, exist_ok=True)

with open(PATH_FILE, "r", encoding="utf-8") as f:
    paths = [line.strip() for line in f if line.strip()]

print(f"[INFO] total files: {len(paths)}")

for i, p in enumerate(paths, 1):
    url = BASE_URL + p
    fname = os.path.basename(p)
    out = os.path.join(OUT_DIR, fname)

    if os.path.exists(out):
        print(f"[{i}/{len(paths)}] skip exists {fname}")
        continue

    print(f"[{i}/{len(paths)}] downloading {fname}")
    try:
        urllib.request.urlretrieve(url, out)
    except Exception as e:
        print(f"  ERROR: {e}")
