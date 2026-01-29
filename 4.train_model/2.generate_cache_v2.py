import os
import torch
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# --- 設定 ---
INPUT_FILE = "grouped_train_data.txt"
TOKENIZER_DIR = "url_tokenizer"
OUTPUT_CACHE = "dataset_cache.pt"
MAX_LEN = 256

def die(msg):
    print(msg)
    raise SystemExit(1)

def generate_dataset_cache():
    if not os.path.exists(TOKENIZER_DIR):
        die(f"❌ 找不到 tokenizer 資料夾 {TOKENIZER_DIR}")

    if not os.path.exists(INPUT_FILE):
        die(f"❌ 找不到輸入檔 {INPUT_FILE}")

    # 載入 tokenizer 並綁定 special tokens
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    tokenizer.pad_token = "[PAD]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.sep_token = "[SEP]"

    # 取得 token ids
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    if pad_id is None or bos_id is None or eos_id is None:
        die("❌ tokenizer 缺少 PAD/BOS/EOS token id，請檢查 tokenizer 訓練流程")

    print(f"--- Tokenizer OK | BOS={bos_id}, EOS={eos_id}, PAD={pad_id} ---")
    print(f"--- 讀取 {INPUT_FILE} 並產生 cache（保 EOS，不用 truncation） ---")

    encoded_examples = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    for line in tqdm(lines, desc="Encoding"):
        text = line.strip()
        if not text:
            continue

        # 1️⃣ 不讓 tokenizer 自動加 special tokens
        ids = tokenizer.encode(text, add_special_tokens=False)

        # 2️⃣ 保證 BOS 在開頭
        if len(ids) == 0 or ids[0] != bos_id:
            ids = [bos_id] + ids

        # 3️⃣ 保證一定有 EOS
        if eos_id not in ids:
            ids.append(eos_id)

        # 4️⃣ 若太長，截到 MAX_LEN-1，最後一格強制 EOS
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN - 1] + [eos_id]
        else:
            # 確保 EOS 在 PAD 前的最後有效位置
            if ids[-1] != eos_id:
                if len(ids) == MAX_LEN:
                    ids[-1] = eos_id
                else:
                    ids.append(eos_id)

        # 5️⃣ padding
        if len(ids) < MAX_LEN:
            ids = ids + [pad_id] * (MAX_LEN - len(ids))

        encoded_examples.append(torch.tensor(ids, dtype=torch.long))

    print(f"--- 儲存 cache 至 {OUTPUT_CACHE} ---")
    torch.save(encoded_examples, OUTPUT_CACHE)

    # --- Sanity check ---
    sample = encoded_examples[0].tolist()
    decoded = tokenizer.decode(sample, skip_special_tokens=False)

    print("\n" + "=" * 50)
    print("[Sanity Check]")
    print(f"- 第一個 token id: {sample[0]} (BOS)")
    print(f"- 最後 10 個 token id: {sample[-10:]}")
    print(f"- decoded (前 200 字元): {decoded[:200]}")
    print("=" * 50)
    print("✅ dataset_cache.pt 產生完成")

if __name__ == "__main__":
    generate_dataset_cache()
