import sys
import re
import pandas as pd
from collections import Counter
from pathlib import Path

# ========= 設定 =========
PATH_COLUMN_CANDIDATES = ["cleaned_path", "path", "url"]
MIN_TOKEN_LEN = 2
IGNORE_NUMERIC = True
IGNORE_EXTENSIONS = False
RECURSIVE = True
OUTPUT_TOP_N = None   # None = 全部輸出

TOKEN_SPLIT_RE = re.compile(r"[\/\-_\.]+")

COMMON_EXT = {
    "html", "htm", "php", "asp", "aspx", "jsp", "jspx",
    "json", "xml", "rss"
}

# ========= 工具 =========
def find_path_column(df: pd.DataFrame) -> str | None:
    for c in PATH_COLUMN_CANDIDATES:
        if c in df.columns:
            return c
    return None

def extract_tokens(path: str):
    if not isinstance(path, str) or not path:
        return []

    path = path.strip().lower()
    path = path.split("?")[0]

    tokens = TOKEN_SPLIT_RE.split(path)
    out = []

    for t in tokens:
        if not t:
            continue
        if len(t) < MIN_TOKEN_LEN:
            continue
        if IGNORE_NUMERIC and t.isdigit():
            continue
        if IGNORE_EXTENSIONS and t in COMMON_EXT:
            continue
        out.append(t)

    return out

# ========= 主程式 =========
def main(input_dir: str, output_csv: str):
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] 不是有效資料夾：{input_dir}")
        sys.exit(1)

    csv_files = (
        input_dir.rglob("*.csv")
        if RECURSIVE
        else input_dir.glob("*.csv")
    )

    counter = Counter()
    total_rows = 0
    used_files = 0

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] 讀取失敗：{csv_path} ({e})")
            continue

        path_col = find_path_column(df)
        if not path_col:
            print(f"[SKIP] 找不到 path 欄位：{csv_path}")
            continue

        used_files += 1

        for p in df[path_col]:
            counter.update(extract_tokens(p))
            total_rows += 1

    if not counter:
        print("[WARN] 沒有任何 token 被統計")
        sys.exit(0)

    # 輸出
    rows = counter.most_common(OUTPUT_TOP_N)
    out_df = pd.DataFrame(rows, columns=["token", "count"])
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    print("\n=== 完成 ===")
    print(f"使用 CSV 檔案數量 : {used_files}")
    print(f"處理 path 筆數     : {total_rows}")
    print(f"唯一 token 數量    : {len(counter)}")
    print(f"輸出檔案           : {output_csv}")

# ========= CLI =========
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python extract_path_keywords_from_folder.py <input_folder> <output.csv>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
