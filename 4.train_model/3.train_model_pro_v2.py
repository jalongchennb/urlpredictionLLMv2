import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

MAX_LEN = 256
BATCH_SIZE = 32
ACCUMULATION_STEPS = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 512
N_HEAD = 8
NUM_LAYERS = 8
FF_DIM = 1024

CACHE_FILE = "dataset_cache.pt"
CHECKPOINT_NAME = "url_gpt_v2_latest.pth"

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
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=NUM_LAYERS)
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        sz = x.size(1)
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=x.device), diagonal=1)
        x = self.embedding(x) + self.pos_embedding[:, :sz, :]
        x = self.transformer(x, mask=mask)
        return self.fc_out(x)

class URLGroupDataset(Dataset):
    def __init__(self, cache_path):
        self.examples = torch.load(cache_path, weights_only=True)
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

def main():
    torch.cuda.empty_cache()

    tokenizer = PreTrainedTokenizerFast.from_pretrained("url_tokenizer")
    tokenizer.pad_token = "[PAD]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.sep_token = "[SEP]"

    vocab_size = len(tokenizer)
    print(f"--- vocab_size: {vocab_size} | device: {DEVICE} ---")
    print(f"PAD={tokenizer.pad_token_id}, BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, SEP={tokenizer.sep_token_id}")

    if not os.path.exists(CACHE_FILE):
        print("錯誤：找不到 dataset_cache.pt，請先執行 generate_cache.py")
        return

    dataset = URLGroupDataset(CACHE_FILE)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = URLTransformer(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    if os.path.exists(CHECKPOINT_NAME):
        print(f"--- 載入權重續傳: {CHECKPOINT_NAME} ---")
        model.load_state_dict(torch.load(CHECKPOINT_NAME, map_location=DEVICE, weights_only=True))

    history = {"train_loss": [], "val_loss": []}
    print(f"--- 訓練啟動 (LR={LEARNING_RATE}) ---")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        total_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, batch in enumerate(pbar):
            batch = batch.to(DEVICE)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * ACCUMULATION_STEPS
            if i % 10 == 0:
                pbar.set_postfix(loss=f"{(total_train_loss/(i+1)):.4f}")

        # 殘留梯度也要 clip
        if len(train_loader) % ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # 驗證
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                inputs, targets = batch[:, :-1], batch[:, 1:]
                outputs = model(inputs)
                loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                total_val_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        print(f"Epoch {epoch+1} 完成 | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        torch.save(model.state_dict(), CHECKPOINT_NAME)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"url_gpt_v2_epoch_{epoch+1}.pth")

        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Val")
        plt.title(f"Stable Training (Epoch {epoch+1})")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curve_stable.png")
        plt.close()

if __name__ == "__main__":
    main()
