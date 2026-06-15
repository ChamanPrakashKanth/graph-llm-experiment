# train_answer_decoder.py
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from pathlib import Path

class AnswerDecoderDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_input_len=128, max_target_len=128):
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        path_str = " -> ".join(item["reasoning_path"])
        answer = item["answer"]

        input_text = f"question: {question} path: {path_str}"
        target_text = answer

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = targets["input_ids"].squeeze(0)
        # Replace padding token id (0) with -100 so it is ignored by loss fn
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Load tokenizer and model
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    dataset_path = "data/mechanical_engineering_dataset.json"
    train_dataset = AnswerDecoderDataset(dataset_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    epochs = 15
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} loss: {total_loss / len(train_loader):.4f}")

    # Save fine-tuned model
    save_path = Path("checkpoints/answer_decoder_t5")
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer successfully saved to {save_path}")

if __name__ == "__main__":
    train()
