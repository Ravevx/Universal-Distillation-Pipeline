# student_generic.py
import os, json, gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoConfig,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()


class DistillConfig:
    def __init__(self,
                 model_name="google/flan-t5-large",
                 silver_jsonl="silver_dataset.jsonl",
                 output_dir="./student_model",
                 max_input_len=512,
                 max_target_len=512,
                 batch_size=4,
                 grad_accum=8,
                 epochs=5,
                 lr=3e-4,
                 warmup_ratio=0.1,
                 val_split=0.1,
                 early_stop_pat=3,
                 seed=42):
        self.model_name = model_name
        self.silver_jsonl = silver_jsonl
        self.output_dir = output_dir
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.epochs = epochs
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.val_split = val_split
        self.early_stop_pat = early_stop_pat
        self.seed = seed


def detect_architecture(model_name: str) -> str:
    """
    Returns "seq2seq" for T5-family encoder-decoder models,
    or "causal" for decoder-only models (Qwen, Llama, Mistral, DeepSeek-R1-Distill, etc.)
    """
    try:
        config = AutoConfig.from_pretrained(model_name)
        if getattr(config, "is_encoder_decoder", False):
            return "seq2seq"
        return "causal"
    except Exception:
        name = model_name.lower()
        if "t5" in name:
            return "seq2seq"
        return "causal"


def load_examples(jsonl_path):
    """
    Generic loader — works with ANY silver_generator.py output.
    Uses the reasoning chain if present, falls back to JSON-only target.
    """
    examples = []
    if not os.path.exists(jsonl_path):
        print(f"Silver file not found: {jsonl_path}")
        return examples

    with_cot, without_cot = 0, 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                reasoning = str(rec.get("reasoning", "")).strip()
                output = rec.get("output", "{}")

                if reasoning:
                    target = f"<think>\n{reasoning}\n</think>\n{output}"
                    with_cot += 1
                else:
                    target = output
                    without_cot += 1

                instruction = rec.get("instruction", "Distill the input as instructed.")
                examples.append({
                    "prompt": f"{instruction}\n\nTEXT:\n{rec['input']}",
                    "target": target,
                    "source": rec.get("source", "silver")
                })
            except Exception:
                continue

    print(f"Examples loaded: {len(examples)} (with CoT: {with_cot}, without: {without_cot})")
    return examples


# ── Seq2Seq (T5-family) dataset ──────────────────────────────────────────
class Seq2SeqDistillDataset(Dataset):
    def __init__(self, examples, tokenizer, max_input_len, max_target_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoded = self.tokenizer(
            text=ex["prompt"],
            text_target=ex["target"],
            max_length=self.max_input_len,
            max_target_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = encoded["labels"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": labels
        }


# ── Causal LM (Qwen/Llama/Mistral/DeepSeek-family) dataset ──────────────
class CausalDistillDataset(Dataset):
    """
    Builds a single sequence: prompt + target, masking prompt tokens with
    -100 so loss is only computed on the target/completion portion.
    """
    def __init__(self, examples, tokenizer, max_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["target"]}
        ]

        try:
            full_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                max_length=self.max_len, truncation=True
            )
            prompt_ids = self.tokenizer.apply_chat_template(
                [messages[0]], tokenize=True, add_generation_prompt=True,
                max_length=self.max_len, truncation=True
            )
        except Exception:
            full_text = ex["prompt"] + "\n" + ex["target"] + self.tokenizer.eos_token
            full_ids = self.tokenizer(full_text, truncation=True, max_length=self.max_len)["input_ids"]
            prompt_ids = self.tokenizer(ex["prompt"], truncation=True, max_length=self.max_len)["input_ids"]

        full_ids = full_ids[:self.max_len]
        prompt_len = min(len(prompt_ids), len(full_ids))

        pad_len = self.max_len - len(full_ids)
        input_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(full_ids) + [0] * pad_len

        labels = list(full_ids)
        labels[:prompt_len] = [-100] * prompt_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def build_model_and_tokenizer(cfg, device, arch):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if arch == "seq2seq":
        model = T5ForConditionalGeneration.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            tie_word_embeddings=False
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

    if device.type == "cuda":
        model.gradient_checkpointing_enable()
    return model, tokenizer


def train_student(cfg: DistillConfig, progress_callback=None):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    arch = detect_architecture(cfg.model_name)
    print(f"Detected architecture: {arch}")

    examples = load_examples(cfg.silver_jsonl)
    if not examples:
        raise ValueError("No training examples found — check silver_jsonl path.")

    np.random.seed(cfg.seed)
    np.random.shuffle(examples)
    val_size = max(1, int(len(examples) * cfg.val_split)) if len(examples) > 1 else 0
    val_examples = examples[:val_size]
    train_examples = examples[val_size:] if val_size < len(examples) else examples

    model, tokenizer = build_model_and_tokenizer(cfg, device, arch)

    if arch == "seq2seq":
        train_ds = Seq2SeqDistillDataset(train_examples, tokenizer, cfg.max_input_len, cfg.max_target_len)
        val_ds = Seq2SeqDistillDataset(val_examples, tokenizer, cfg.max_input_len, cfg.max_target_len) if val_examples else None
    else:
        train_ds = CausalDistillDataset(train_examples, tokenizer, cfg.max_input_len)
        val_ds = CausalDistillDataset(val_examples, tokenizer, cfg.max_input_len) if val_examples else None

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_ds else None

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    total_steps = max(1, (len(train_loader) // cfg.grad_accum) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / cfg.grad_accum
            loss.backward()
            total_loss += loss.item() * cfg.grad_accum

            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss / max(len(train_loader), 1)

        avg_val_loss = avg_train_loss
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
            avg_val_loss = val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")
        if progress_callback:
            progress_callback(epoch + 1, cfg.epochs, avg_train_loss, avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            model.save_pretrained(cfg.output_dir)
            tokenizer.save_pretrained(cfg.output_dir)
        else:
            patience += 1
            if patience >= cfg.early_stop_pat:
                print("Early stopping triggered.")
                break

        gc.collect()
        torch.cuda.empty_cache()

    return cfg.output_dir