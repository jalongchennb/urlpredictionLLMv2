import os
import pandas as pd
from urllib.parse import urlparse
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

# =========================
# 1) 配置設定
# =========================
INPUT_FILE = "unique_urls.csv"  # 你的原始資料檔名 (csv 或 txt)
CLEANED_FILE = "cleaned_paths.txt"
OUTPUT_DIR = "url_tokenizer"    # 儲存分詞器的資料夾
VOCAB_SIZE = 10000              # 字典大小 (根據資料量調整)

# =========================
# 2) 資料清洗：移除 Domain 與參數，只留 Path
# =========================
def clean_url_to_path(url):
    try:
        if not isinstance(url, str):
            return None
        # 使用 urlparse 提取路徑部分 (例如: /admin/login.php)
        path = urlparse(url).path
        # 過濾掉空路徑或純斜線，並確保以 / 開頭
        if not path or path == "/":
            return None
        return path
    except:
        return None

print(f"正在讀取 {INPUT_FILE} 並進行清理...")
# 讀取 CSV (假設欄位名稱是 'url')
df = pd.read_csv(INPUT_FILE)
df['clean_path'] = df['url'].apply(clean_url_to_path)

# 移除重複與空值
unique_paths = df['clean_path'].dropna().unique()

# 存成純文字檔，每行一個路徑，供訓練器讀取
with open(CLEANED_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(unique_paths))

print(f"清理完成！共提取出 {len(unique_paths)} 筆不重複路徑。")

# =========================
# 3) 訓練 ByteLevelBPE 分詞器
# =========================
print("開始訓練分詞器 (ByteLevelBPE)...")
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[CLEANED_FILE],
    vocab_size=VOCAB_SIZE,
    min_frequency=2,  # 一個詞至少出現兩次才加入字典
    show_progress=True,
    # 定義特殊標籤 (必須與主程式一致)
    special_tokens=[
        "[PAD]", # 0: 填充
        "[BOS]", # 1: 開始
        "[EOS]", # 2: 結束
        "[SEP]", # 3: 分隔
        "\n"     # 換行
    ]
)

# =========================
# 4) 轉換並儲存
# =========================
# 轉換為 Transformers 庫通用的 Fast 格式
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="[PAD]",
    bos_token="[BOS]",
    eos_token="[EOS]",
    sep_token="[SEP]"
)

# 建立資料夾並存檔
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

fast_tokenizer.save_pretrained(OUTPUT_DIR)

print(f"=" * 30)
print(f"訓練成功！")
print(f"分詞器已儲存至: {OUTPUT_DIR}")
print(f"字典大小: {len(fast_tokenizer)}")
print(f"=" * 30)

# 測試一下
test_path = "/admin/api/v1/user_login.php"
tokens = fast_tokenizer.tokenize(test_path)
ids = fast_tokenizer.encode(test_path)
print(f"測試文字: {test_path}")
print(f"切分結果: {tokens}")
print(f"編碼 ID: {ids}")