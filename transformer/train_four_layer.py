# train.py

import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from mini_transformer import (
    WordEmbedding,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
)


# -------------------------
# 1. Config
# -------------------------

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MAX_LEN = 64
MIN_FREQ = 2
VOCAB_SIZE_LIMIT = 20000

EMB_D = 64
HEAD_SIZE = 4
FFN_D = 256
NUM_LAYERS = 4
NUM_CLASSES = 4

BATCH_SIZE = 64
EPOCHS = 5
LR = 3e-4

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


# -------------------------
# 2. Tokenizer
# -------------------------

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def build_vocab(texts, min_freq=2, max_size=20000):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len):
    tokens = tokenize(text)

    ids = []
    for token in tokens[:max_len]:
        ids.append(vocab.get(token, vocab[UNK_TOKEN]))

    # padding
    while len(ids) < max_len:
        ids.append(vocab[PAD_TOKEN])

    return ids


# -------------------------
# 3. Dataset
# -------------------------

class TextClassificationDataset(Dataset):
    def __init__(self, hf_dataset, vocab, max_len):
        self.dataset = hf_dataset
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        text = item["text"]
        label = item["label"]

        input_ids = encode_text(text, self.vocab, self.max_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


# -------------------------
# 4. Model
# -------------------------

class MiniTransformerTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        emb_d,
        head_size,
        ffn_d,
        num_layers,
        num_classes,
        pad_id=0,
        dropout=0.1,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.emb_d = emb_d
        self.max_len = max_len

        self.word_emb = WordEmbedding(vocab_size, emb_d)
        self.pos_enc = SinusoidalPositionalEncoding(max_len, emb_d)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                emb_d=emb_d,
                head_size=head_size,
                ffn_d=ffn_d,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(emb_d, num_classes)

    def make_padding_mask(self, input_ids):
        # input_ids: [B, S]
        # mask: [B, 1, 1, S]
        mask = (input_ids != self.pad_id).unsqueeze(1).unsqueeze(2)
        return mask

    def forward(self, input_ids):
        # input_ids: [B, S]
        B, S = input_ids.shape

        padding_mask = self.make_padding_mask(input_ids)

        x = self.word_emb(input_ids)  # [B, S, D]

        pe = self.pos_enc.get_PE(S).to(input_ids.device)  # [S, D]
        pe = pe.unsqueeze(0)                              # [1, S, D]

        x = x + pe

        for block in self.blocks:
            x = block(x, mask=padding_mask)

        # mean pooling, 단 padding token은 제외
        valid_mask = (input_ids != self.pad_id).unsqueeze(-1)  # [B, S, 1]
        x = x * valid_mask

        summed = x.sum(dim=1)  # [B, D]
        counts = valid_mask.sum(dim=1).clamp(min=1)  # [B, 1]

        pooled = summed / counts  # [B, D]

        logits = self.classifier(self.dropout(pooled))  # [B, num_classes]

        return logits


# -------------------------
# 5. Train / Eval
# -------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(input_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += input_ids.size(0)

    avg_loss = total_loss / total_count
    acc = total_correct / total_count

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(input_ids)
        loss = criterion(logits, labels)

        total_loss += loss.item() * input_ids.size(0)

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += input_ids.size(0)

    avg_loss = total_loss / total_count
    acc = total_correct / total_count

    return avg_loss, acc


# -------------------------
# 6. Main
# -------------------------

def main():
    print("device:", DEVICE)

    dataset = load_dataset("ag_news")

    train_raw = dataset["train"]
    test_raw = dataset["test"]

    # 빠른 실험용으로 일부만 사용
    train_raw = train_raw.select(range(20000))
    test_raw = test_raw.select(range(2000))

    print("building vocab...")
    vocab = build_vocab(
        texts=[item["text"] for item in train_raw],
        min_freq=MIN_FREQ,
        max_size=VOCAB_SIZE_LIMIT,
    )

    print("vocab size:", len(vocab))

    train_dataset = TextClassificationDataset(train_raw, vocab, MAX_LEN)
    test_dataset = TextClassificationDataset(test_raw, vocab, MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MiniTransformerTextClassifier(
        vocab_size=len(vocab),
        max_len=MAX_LEN,
        emb_d=EMB_D,
        head_size=HEAD_SIZE,
        ffn_d=FFN_D,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        pad_id=vocab[PAD_TOKEN],
        dropout=0.1,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
        )

        print(
            f"epoch {epoch} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} | "
            f"test acc {test_acc:.4f}"
        )

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "max_len": MAX_LEN,
            "emb_d": EMB_D,
            "head_size": HEAD_SIZE,
            "ffn_d": FFN_D,
            "num_layers": NUM_LAYERS,
            "num_classes": NUM_CLASSES,
        }
    }, "mini_transformer_classifier.pt")

    print("saved: mini_transformer_classifier.pt")


if __name__ == "__main__":
    main()