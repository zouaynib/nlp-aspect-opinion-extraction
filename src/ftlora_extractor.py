import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from accelerate import Accelerator


ASPECTS = ["Price", "Food", "Service"]
LABELS = ["Positive", "Negative", "Mixed", "No Opinion"]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

NUM_ASPECTS = len(ASPECTS)
NUM_LABELS = len(LABELS)

MAX_LENGTH = 256

MODEL_NAME = "almanach/moderncamembert-base"


def _normalize(label: str) -> str:
    """Map noisy annotations (e.g. 'Positive#NE') to one of the 4 canonical labels."""
    head = str(label).split("#")[0].strip()
    return head if head in LABEL2ID else "No Opinion"


class AspectDataset(Dataset):
    """
    Wraps tokenized reviews + (optional) per-aspect labels.
    labels shape per example: LongTensor of size [NUM_ASPECTS].
    """

    def __init__(self, texts: list[str], labels: list[list[str]] | None, tokenizer, max_length: int = MAX_LENGTH) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        if labels is not None:
            label_ids = [[LABEL2ID[_normalize(l)] for l in row] for row in labels]
            self.labels = torch.tensor(label_ids, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self) -> int:
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class MultiHeadClassifier(nn.Module):
    """
    Shared encoder (CamemBERT/etc.) + 3 independent linear heads, one per aspect.
    Output: logits of shape [batch, NUM_ASPECTS, NUM_LABELS].
    """

    def __init__(self, model_name: str, num_aspects: int = NUM_ASPECTS, num_labels: int = NUM_LABELS, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_labels) for _ in range(num_aspects)]
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]   # [batch, hidden]
        cls = self.dropout(cls)
        # stack per-head logits along a new "aspect" dim -> [batch, num_aspects, num_labels]
        logits = torch.stack([head(cls) for head in self.heads], dim=1)
        return logits


class OpinionExtractor:

    method: Literal["NOFT", "FT"] = "FT"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model_name = MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = MultiHeadClassifier(self.model_name)

    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        """
        Trains the model, if OpinionExtractor.method=="FT"
        """
        # --- hyperparameters ---
        epochs = 4
        target_effective_batch = 32
        per_device_batch = 32
        base_lr = 2e-5
        warmup_ratio = 0.1
        weight_decay = 0.01

        # --- device-aware batching ---
        num_devices = max(1, torch.cuda.device_count())
        grad_accum_steps = max(1, target_effective_batch // (per_device_batch * num_devices))
        effective_batch = per_device_batch * num_devices * grad_accum_steps
        lr = base_lr * (effective_batch / target_effective_batch)

        # --- accelerator (handles DDP + grad accum) ---
        accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)

        # --- data ---
        texts = [d["Review"] for d in train_data]
        labels = [[d[a] for a in ASPECTS] for d in train_data]
        train_ds = AspectDataset(texts, labels, self.tokenizer)
        train_dl = DataLoader(train_ds, batch_size=per_device_batch, shuffle=True)

        # --- optimizer: exclude bias & LayerNorm from weight decay ---
        no_decay = ["bias", "LayerNorm.weight"]
        named = list(self.model.named_parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in named if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
                {"params": [p for n, p in named if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ],
            lr=lr,
        )

        # --- linear warmup + linear decay scheduler ---
        steps_per_epoch = math.ceil(len(train_dl) / grad_accum_steps)
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # --- move everything to device / wrap for DDP ---
        self.model, optimizer, train_dl, scheduler = accelerator.prepare(
            self.model, optimizer, train_dl, scheduler
        )

        # --- training loop ---
        self.model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            n_batches = 0
            for batch in train_dl:
                with accelerator.accumulate(self.model):
                    logits = self.model(batch["input_ids"], batch["attention_mask"])
                    # logits: [B, 3, 4]   labels: [B, 3]
                    loss = F.cross_entropy(
                        logits.reshape(-1, NUM_LABELS),
                        batch["labels"].reshape(-1),
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                running_loss += loss.item()
                n_batches += 1
            avg = running_loss / max(1, n_batches)
            accelerator.print(f"Epoch {epoch}/{epochs} - avg loss: {avg:.4f}")

        # unwrap model for inference (remove DDP wrapper) and keep on its current device
        self.model = accelerator.unwrap_model(self.model)
        self.model.eval()

    def predict(self, texts: list[str]) -> list[dict]:
        """
        :param texts: list of reviews from which to extract the opinion values
        :return: a list of dicts, one per input review, containing the opinion values for the 3 aspects.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        enc = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"])  # [B, 3, 4]
        pred_ids = logits.argmax(dim=-1).cpu().tolist()  # [B, 3]

        return [
            {aspect: ID2LABEL[row[a]] for a, aspect in enumerate(ASPECTS)}
            for row in pred_ids
        ]
