import os
import random
import pandas as pd
from glob import glob
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# --- 配置區 ---
WORK_FOLDER = "datacleanworkfolder"
OUTPUT_FILE = "grouped_train_data.txt"

TOKENIZER_DIR = "url_tokenizer"
MAX_LEN = 256            # 要跟你訓練時一致
MIN_PATHS_PER_LINE = 2   # 太短的 line 對「聯想」幫助有限

# 你原本是用固定筆數 CHUNK_SIZE / OVERLAP
# 這版改成「token 長度控制」：永遠盡量塞到不超過 MAX_LEN，並保留 EOS
# 但仍提供 overlap（用 path 數來做滑動），避免完全無重疊
OVERLAP_PATHS = 2        # 建議 1~3，不要再用 10 了（會強化重複）
SHUFFLE_PER_DOMAIN = True
RANDOM_SEED = 42

def die(msg: str):
    print(msg)
    raise SystemExit(1)

def load_tokenizer():
    if not os.path.exists(TOKENIZER_DIR):
        die(f"❌ 找不到 tokenizer 資料夾：{TOKENIZER_DIR}")

    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    # 綁定 special token 角色（確保 id 存在）
    tok.pad_token = "[PAD]"
    tok.bos_token = "[BOS]"
    tok.eos_token = "[EOS]"
    tok.sep_token = "[SEP]"

    if tok.bos_token_id is None or tok.eos_token_id is None or tok.sep_token_id is None:
        die("❌ tokenizer 的 BOS/EOS/SEP token id 缺失，請確認 tokenizer 訓練時有加入 special_tokens。")

    # 確認它們會被 encode 成單一 id（很重要）
    def check_single(tok_str, expected_id):
        ids = tok.encode(tok_str, add_special_tokens=False)
        if len(ids) != 1 or ids[0] != expected_id:
            die(f"❌ {tok_str} 沒有對齊成單一 id。encode -> {ids}, expected -> [{expected_id}]")

    check_single("[BOS]", tok.bos_token_id)
    check_single("[EOS]", tok.eos_token_id)
    check_single("[SEP]", tok.sep_token_id)

    print(f"--- Tokenizer OK | vocab={len(tok)} | BOS={tok.bos_token_id} SEP={tok.sep_token_id} EOS={tok.eos_token_id} ---")
    return tok

def pack_paths_by_token_len(paths, tokenizer, max_len, overlap_paths=0, min_paths_per_line=2):
    """
    將同 domain 的 paths 依照 token 長度「動態打包」，確保：
      - 每行最後一定包含 EOS
      - 不會因為 truncation 把 EOS 切掉（因為這裡就控制長度了）
    overlap_paths：用 path 數量做重疊（建議 1~3）
    """
    bos_id = tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id
    eos_id = tokenizer.eos_token_id

    # 預先把每個 path 編成 token ids（不加特殊 token）
    # 注意：這裡只用來估長度，最後輸出仍是文字（[BOS] ... [EOS]）
    path_ids = [tokenizer.encode(p, add_special_tokens=False) for p in paths]

    lines = []
    i = 0
    n = len(paths)

    # 每行至少會有：BOS + path + EOS（中間可能有 SEP + path...）
    # token budget：max_len
    # 我們用「id 長度」估算：1(BOS) + len(path1) + ... + 1(EOS) + SEP(每多一個 path 增 1)
    while i < n:
        cur_paths = []
        # 起始長度：BOS + EOS
        cur_len = 1 + 1

        j = i
        while j < n:
            cand_ids = path_ids[j]
            # 如果是第一個 path：只加 path 本身
            # 如果不是第一個：要多一個 SEP
            add_len = len(cand_ids) if len(cur_paths) == 0 else (1 + len(cand_ids))

            # 加進來後是否超過 max_len？
            if cur_len + add_len > max_len:
                break

            cur_paths.append(paths[j])
            cur_len += add_len
            j += 1

        # 如果連 1 個 path 都塞不進去（代表單一路徑本身就太長）
        # 仍然硬塞一個（讓下游用 truncation 時再處理），但這種資料通常要另外清洗
        if len(cur_paths) == 0:
            cur_paths = [paths[i]]
            j = i + 1

        # 太短的行可選擇丟掉（除非 domain 本身就很少資料）
        if len(cur_paths) >= min_paths_per_line or n < min_paths_per_line:
            line = "[BOS] " + " [SEP] ".join(cur_paths) + " [EOS]"
            lines.append(line)

        if j >= n:
            break

        # overlap：下一個 window 從尾端往回 overlap_paths
        if overlap_paths > 0:
            i = max(i + 1, j - overlap_paths)
        else:
            i = j

    return lines

def main():
    random.seed(RANDOM_SEED)

    print(f"--- 正在從 {WORK_FOLDER} 讀取 CSV 並合併數據 ---")
    all_files = glob(os.path.join(WORK_FOLDER, "*.csv"))
    if not all_files:
        die("❌ 找不到任何 CSV 檔案，請確認 WORK_FOLDER 是否正確。")

    df_list = []
    for f in all_files:
        df_list.append(pd.read_csv(f))

    full_df = pd.concat(df_list, ignore_index=True).drop_duplicates()

    # 基本欄位檢查
    required_cols = {"domain", "cleaned_path"}
    if not required_cols.issubset(set(full_df.columns)):
        die(f"❌ CSV 缺少必要欄位：{required_cols}，目前欄位：{list(full_df.columns)}")

    tokenizer = load_tokenizer()

    print("--- 正在依 domain 分組並動態打包（以 token 長度控制，不會截掉 EOS） ---")
    training_lines = []
    domain_count = 0
    kept_domains = 0

    grouped = full_df.groupby("domain")
    for domain, group in tqdm(grouped, total=full_df["domain"].nunique(), desc="Packing by domain"):
        domain_count += 1

        # 取得 paths
        paths = group["cleaned_path"].dropna().astype(str).tolist()
        # 去重：保留原順序（不要 sort，避免學到字典序規律）
        seen = set()
        uniq_paths = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                uniq_paths.append(p)

        if len(uniq_paths) < 2:
            continue

        if SHUFFLE_PER_DOMAIN:
            random.shuffle(uniq_paths)

        lines = pack_paths_by_token_len(
            uniq_paths,
            tokenizer=tokenizer,
            max_len=MAX_LEN,
            overlap_paths=OVERLAP_PATHS,
            min_paths_per_line=MIN_PATHS_PER_LINE
        )

        if lines:
            kept_domains += 1
            training_lines.extend(lines)

    print(f"--- 正在寫入最終訓練檔: {OUTPUT_FILE} ---")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in training_lines:
            f.write(line + "\n")

    print("\n" + "=" * 60)
    print("完成！")
    print(f"- 總站點數：{domain_count}")
    print(f"- 有產生訓練行的站點數：{kept_domains}")
    print(f"- 生成訓練行數：{len(training_lines)}")
    print(f"- 輸出檔案：{OUTPUT_FILE}")
    print("=" * 60)

    # 追加：簡單抽樣檢查幾行的 token 長度（確保 <= MAX_LEN）
    if training_lines:
        sample = random.sample(training_lines, k=min(5, len(training_lines)))
        print("\n[Sample length check]")
        for s in sample:
            ids = tokenizer.encode(s, add_special_tokens=False)
            print(f"len={len(ids):3d} | {s[:80]}...")

if __name__ == "__main__":
    main()
