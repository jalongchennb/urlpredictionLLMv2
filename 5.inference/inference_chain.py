import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# =========================
# 1) Config (must match training)
# =========================
EMBED_DIM = 512
N_HEAD = 8
NUM_LAYERS = 8
FF_DIM = 1024
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOKENIZER_DIR = "url_tokenizer"
MODEL_PATH = "url_gpt_v2_latest.pth"

# Generation params
DEFAULT_MAX_SEGMENTS = 50     # 接龍最多吐幾條 segment（類似你原本的 DEFAULT_N_CANDIDATES）
SEG_MAX_TOKENS = 80          # 每條 segment 最多 token
TEMPERATURE = 0.9
TOP_P = 0.96
REPETITION_PENALTY = 1.05

# 可選：整條接龍新增 token 總上限（None = 不限制）
MAX_TOTAL_NEW_TOKENS = None  # 例如設成 2000 避免無限長


# =========================
# 2) Model
# =========================
class URLTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_embedding = nn.Parameter(torch.zeros(1, MAX_LEN, EMBED_DIM))
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=N_HEAD,
            dim_feedforward=FF_DIM,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=NUM_LAYERS)
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        sz = x.size(1)
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=x.device), diagonal=1)
        x = self.embedding(x) + self.pos_embedding[:, :sz, :]
        x = self.transformer(x, mask=mask)
        return self.fc_out(x)


# =========================
# 3) Tokenizer loader
# =========================
def load_tokenizer(tokenizer_dir):
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir, use_fast=True)
    except Exception:
        from tokenizers import Tokenizer
        tok = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json")),
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            sep_token="[SEP]",
        )

    tok.pad_token = "[PAD]"
    tok.bos_token = "[BOS]"
    tok.eos_token = "[EOS]"
    tok.sep_token = "[SEP]"
    return tok


# =========================
# 4) Decoding helpers
# =========================
def apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty <= 1.0:
        return logits
    logits = logits.clone()
    for tid in set(generated_ids):
        v = logits[tid]
        logits[tid] = v / penalty if v > 0 else v * penalty
    return logits


def top_p_sample(logits, top_p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum > top_p
    mask[..., 0] = False  # ensure at least one token
    sorted_probs[mask] = 0
    s = sorted_probs.sum()
    if s.item() == 0:
        # fallback: if everything is masked somehow, pick argmax
        return torch.argmax(probs).item()
    sorted_probs = sorted_probs / s
    pick = torch.multinomial(sorted_probs, 1).item()
    return sorted_idx[pick].item()


# =========================
# 5) Chain generation (接龍)
# =========================
def sample_chain_segments(
    model,
    tokenizer,
    prefix_ids,
    max_segments,
    seg_max_tokens,
    max_total_new_tokens=None,
):
    """
    一次推論接龍吐出多條 segment：
      - 每條 segment 生成直到遇到 [SEP] 或 [EOS] 或 seg_max_tokens
      - 遇到 [SEP]：結束當前 segment，繼續下一條
      - 遇到 [EOS]：整條接龍結束
    回傳：segments_text(list[str])
    """
    sep_id = tokenizer.sep_token_id
    eos_id = tokenizer.eos_token_id

    generated = list(prefix_ids)
    segments = []
    total_new = 0

    with torch.no_grad():
        for _ in range(max_segments):
            cur_ids = []

            for _t in range(seg_max_tokens):
                if max_total_new_tokens is not None and total_new >= max_total_new_tokens:
                    break

                ctx = generated[-MAX_LEN:]
                logits = model(torch.tensor([ctx], device=DEVICE))[0, -1, :]

                logits = logits / TEMPERATURE
                logits = apply_repetition_penalty(logits, generated, REPETITION_PENALTY)
                next_id = top_p_sample(logits, TOP_P)

                generated.append(next_id)
                total_new += 1

                # 分段/停止
                if next_id == sep_id:
                    break
                if next_id == eos_id:
                    break

                cur_ids.append(next_id)

            # decode 本段（空段就跳過）
            text = tokenizer.decode(cur_ids, skip_special_tokens=True).strip()
            if text:
                segments.append(text)

            # 若觸發整體 token 上限
            if max_total_new_tokens is not None and total_new >= max_total_new_tokens:
                break

            # 若最後一個 token 是 EOS：整條停止
            if generated and generated[-1] == eos_id:
                break

    return segments


def generate_chain_paths(model, tokenizer, user_lines, max_segments):
    bos = tokenizer.bos_token
    prompt = " [SEP] ".join(user_lines) + " [SEP]"
    prefix_ids = tokenizer.encode(f"{bos} {prompt}", add_special_tokens=False)

    segments = sample_chain_segments(
        model=model,
        tokenizer=tokenizer,
        prefix_ids=prefix_ids,
        max_segments=max_segments,
        seg_max_tokens=SEG_MAX_TOKENS,
        max_total_new_tokens=MAX_TOTAL_NEW_TOKENS,
    )

    # 去重但保留順序
    seen = set()
    results = []
    for s in segments:
        if s not in seen:
            seen.add(s)
            results.append(s)

    # 最多回傳 max_segments 條（空段被濾掉後可能不足）
    return results[:max_segments]


# =========================
# 6) CLI
# =========================
def main():
    tokenizer = load_tokenizer(TOKENIZER_DIR)
    model = URLTransformer(len(tokenizer)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    print("=" * 60)
    user_n = input(f"請輸入接龍最多產出幾條 path/segment (Enter 使用預設 {DEFAULT_MAX_SEGMENTS}): ").strip()
    try:
        max_segments = int(user_n) if user_n else DEFAULT_MAX_SEGMENTS
    except ValueError:
        max_segments = DEFAULT_MAX_SEGMENTS

    print(f"→ 本次接龍最多產出 {max_segments} 條")
    print("=" * 60)

    while True:
        print("\n請輸入或貼上路徑（多行，空行送出；q/quit/exit 離開）：")
        lines = []
        while True:
            line = input().strip()
            if line.lower() in ("q", "quit", "exit"):
                return
            if line == "" and lines:
                break
            if line:
                lines.append(line)

        results = generate_chain_paths(model, tokenizer, lines, max_segments=max_segments)

        print("\n[接龍候選 paths/segments]")
        for p in results:
            print(p)


if __name__ == "__main__":
    main()
