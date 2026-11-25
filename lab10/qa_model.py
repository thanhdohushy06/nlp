"""
qa_model.py — Extractive QA Model Implementation

Core implementation cho extractive question answering:
- QA Dataset class
- QA Model (pretrained Transformer + start/end classifiers)
- Training loop
- Evaluation (EM / F1)
- Baseline (keyword overlap)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
import string
from collections import Counter


# ==============================================================================
# 1. Dataset
# ==============================================================================

class SQuADDataset(Dataset):
    """Dataset cho SQuAD-format extractive QA."""

    def __init__(self, data, tokenizer, max_length=384, stride=128):
        """
        Args:
            data: list of dict, mỗi dict có keys: context, question, answers
            tokenizer: HuggingFace tokenizer
            max_length: max sequence length
            stride: stride cho sliding window
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.features = []

        for item in data:
            context = item["context"]
            question = item["question"]
            answer = item["answers"]
            answer_text = answer["text"][0] if isinstance(answer["text"], list) else answer["text"]
            answer_start = answer["answer_start"][0] if isinstance(answer["answer_start"], list) else answer["answer_start"]

            # Tokenize
            encoding = tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt"
            )

            # Xử lý từng window
            for i in range(len(encoding["input_ids"])):
                input_ids = encoding["input_ids"][i]
                attention_mask = encoding["attention_mask"][i]
                offsets = encoding["offset_mapping"][i].tolist()
                token_type_ids = encoding.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids[i]

                # Tìm token span cho answer
                answer_end = answer_start + len(answer_text)
                start_token = 0
                end_token = 0

                # Xác định context token range
                seq_ids = encoding.sequence_ids(i)
                context_start_idx = None
                context_end_idx = None
                for idx, sid in enumerate(seq_ids):
                    if sid == 1:
                        if context_start_idx is None:
                            context_start_idx = idx
                        context_end_idx = idx

                if context_start_idx is not None and context_end_idx is not None:
                    # Kiểm tra answer có nằm trong window không
                    if (offsets[context_start_idx][0] <= answer_start and
                            answer_end <= offsets[context_end_idx][1]):
                        # Tìm start token
                        for idx in range(context_start_idx, context_end_idx + 1):
                            if offsets[idx][0] <= answer_start < offsets[idx][1]:
                                start_token = idx
                                break
                        # Tìm end token
                        for idx in range(context_start_idx, context_end_idx + 1):
                            if offsets[idx][0] < answer_end <= offsets[idx][1]:
                                end_token = idx
                                break

                feature = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "start_positions": torch.tensor(start_token, dtype=torch.long),
                    "end_positions": torch.tensor(end_token, dtype=torch.long),
                    "offset_mapping": encoding["offset_mapping"][i],
                    "context": context,
                    "question": question,
                    "answer_text": answer_text,
                }
                if token_type_ids is not None:
                    feature["token_type_ids"] = token_type_ids

                self.features.append(feature)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        item = {
            "input_ids": feature["input_ids"],
            "attention_mask": feature["attention_mask"],
            "start_positions": feature["start_positions"],
            "end_positions": feature["end_positions"],
        }
        if "token_type_ids" in feature:
            item["token_type_ids"] = feature["token_type_ids"]
        return item


# ==============================================================================
# 2. QA Model
# ==============================================================================

class ExtractiveQAModel(nn.Module):
    """Extractive QA model: Pretrained Transformer + Start/End classifiers."""

    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        # Hai classifier riêng biệt cho start và end
        self.qa_outputs = nn.Linear(hidden_size, 2)  # 2 outputs: start, end

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                start_positions=None, end_positions=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        logits = self.qa_outputs(sequence_output)  # (batch, seq_len, 2)

        start_logits = logits[:, :, 0]  # (batch, seq_len)
        end_logits = logits[:, :, 1]  # (batch, seq_len)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fn = nn.CrossEntropyLoss()
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }


# ==============================================================================
# 3. Metrics: Exact Match & F1
# ==============================================================================

def normalize_answer(s):
    """Normalize answer string: lowercase, remove articles/punctuation, strip."""
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s


def compute_exact_match(prediction, ground_truth):
    """EM = 1 nếu normalized prediction == normalized ground truth."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction, ground_truth):
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def evaluate_predictions(predictions, references):
    """
    Tính EM và F1 trung bình.

    Args:
        predictions: list of predicted answer strings
        references: list of gold answer strings

    Returns:
        dict với 'exact_match' và 'f1'
    """
    em_scores = []
    f1_scores = []

    for pred, ref in zip(predictions, references):
        em_scores.append(compute_exact_match(pred, ref))
        f1_scores.append(compute_f1(pred, ref))

    return {
        "exact_match": np.mean(em_scores) * 100,
        "f1": np.mean(f1_scores) * 100,
    }


# ==============================================================================
# 4. Baseline: Keyword Overlap
# ==============================================================================

class KeywordOverlapBaseline:
    """
    Baseline đơn giản cho QA dựa trên keyword overlap.

    Strategy:
    1. Tách context thành câu
    2. Tính overlap giữa question words và mỗi câu
    3. Chọn câu có overlap cao nhất
    4. Trả về noun phrase đầu tiên hoặc toàn bộ câu
    """

    def predict(self, question, context):
        question_words = set(normalize_answer(question).split())

        # Tách context thành câu
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip()]

        best_score = -1
        best_sentence = sentences[0] if sentences else ""

        for sent in sentences:
            sent_words = set(normalize_answer(sent).split())
            overlap = len(question_words & sent_words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sent

        # Trả về 1-3 từ đầu tiên không nằm trong question
        sent_words = best_sentence.split()
        answer_words = []
        for w in sent_words:
            if normalize_answer(w) not in question_words:
                answer_words.append(w)
            if len(answer_words) >= 3:
                break

        return " ".join(answer_words) if answer_words else best_sentence.split()[0] if sent_words else ""


# ==============================================================================
# 5. Training Utilities
# ==============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train một epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )

        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_model(model, dataloader, device):
    """Evaluate model, trả về loss trung bình."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions
            )

            total_loss += outputs["loss"].item()
            n_batches += 1

    return total_loss / n_batches


def predict_answers(model, dataset, tokenizer, device):
    """
    Dự đoán answers cho toàn bộ dataset.

    Returns:
        list of predicted answer strings
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(len(dataset)):
            feature = dataset.features[i]
            input_ids = feature["input_ids"].unsqueeze(0).to(device)
            attention_mask = feature["attention_mask"].unsqueeze(0).to(device)
            token_type_ids = feature.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.unsqueeze(0).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            start_logits = outputs["start_logits"][0]
            end_logits = outputs["end_logits"][0]

            # Tìm best start và end (start <= end)
            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits).item()

            if end_idx < start_idx:
                end_idx = start_idx

            # Decode answer
            answer_ids = feature["input_ids"][start_idx:end_idx + 1]
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
            predictions.append(answer)

    return predictions


if __name__ == "__main__":
    # Quick test
    print("Testing QA components...")

    # Test metrics
    pred = "the city of Paris"
    gold = "Paris"
    print(f"Pred: '{pred}', Gold: '{gold}'")
    print(f"EM: {compute_exact_match(pred, gold)}")
    print(f"F1: {compute_f1(pred, gold):.4f}")

    # Test baseline
    baseline = KeywordOverlapBaseline()
    q = "What is the capital of France?"
    c = "The capital of France is Paris. It is a beautiful city."
    print(f"\nBaseline prediction: '{baseline.predict(q, c)}'")

    print("\nAll components OK!")
